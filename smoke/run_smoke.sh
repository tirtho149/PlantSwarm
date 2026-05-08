#!/bin/bash
# ============================================================================
# smoke/run_smoke.sh
# ============================================================================
# End-to-end smoke test of the Pathome pipeline on a 2-crop subset
# (Tomato + Soybean). Runs each phase as plain `python ...` calls — no
# SLURM batching — so it works on:
#   * a Nova interactive allocation (`salloc --gres=gpu:a100:1 --time=4:00:00`)
#   * a workstation / laptop with a CUDA GPU
#   * a CPU-only machine (Phases 2/4/5 are skipped automatically)
#
# Stages (mirrors the production chain):
#   Setup  : filter smoke CSV → 2-crop usable subset
#   0      : pathome_kb internet seed (--quick, --only-crops Tomato,Soybean)
#   1      : build PathomeDB v1_seed
#   2      : PlantSwarm trace generation (small budget)
#   3      : enhance from traces → v1_enhanced
#   4      : train OBSERVE × 2 (DT + GRPO, 3 + 1 epochs)
#   5      : eval × 4 + comparison
#
# Each phase below can be skipped via env vars:
#   SMOKE_SKIP_SETUP=1 SMOKE_SKIP_0=1 SMOKE_SKIP_1=1 ...
#
# Or run from a specific phase:
#   SMOKE_FROM=2 bash smoke/run_smoke.sh
#
# Auth requirements:
#   - claude CLI on PATH + claude auth login           (Phase 0)
#   - ANTHROPIC_API_KEY in env or repo-root .env      (Phase 0)
#   - CUDA + Qwen weights cached                      (Phases 2, 4, 5)
#
# Total runtime targets:
#   GPU (A100):   ~60-90 min
#   CPU-only:     ~10-15 min (Setup + Phase 0 + 1 + 3 only)
# ============================================================================

set -e

CONFIG="smoke/bugwood_pathome_smoke.yaml"
PV_EVAL="smoke/plantvillage_smoke_eval.yaml"
PW_EVAL="smoke/plantwild_smoke_eval.yaml"
RAW_CSV="smoke/BugWood_Diseases_smoke.csv"
USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
SEED_FILE="smoke/artifacts/pathome_seed/symptoms_seed.json"
SEED_DB="smoke/artifacts/pathome_v1_seed"
ENH_DB="smoke/artifacts/pathome_v1_enhanced"
TRACES="smoke/results/traces/plantswarm_traces.jsonl"
SEED_CKPT="smoke/observe/checkpoints/seed/observe_grpo_epoch_01.pt"
ENH_CKPT="smoke/observe/checkpoints/enhanced/observe_grpo_epoch_01.pt"

mkdir -p smoke/artifacts smoke/results smoke/observe/checkpoints/seed smoke/observe/checkpoints/enhanced

phase_active() {
  # skip if SMOKE_SKIP_<id> is set
  local id="$1"
  local key="SMOKE_SKIP_${id^^}"
  [ -n "${!key:-}" ] && return 1
  # skip if SMOKE_FROM is greater than this id (numeric only; setup gets index -1)
  if [ -n "${SMOKE_FROM:-}" ]; then
    local idx="$id"
    [ "$id" = "setup" ] && idx="-1"
    if [ "$idx" -lt "$SMOKE_FROM" ]; then return 1; fi
  fi
  return 0
}

step() {
  echo
  echo "==============================================================="
  echo "  $1"
  echo "==============================================================="
}

has_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"
}

# ----------------------------------------------------------------------------
# Setup — filter the 2-crop CSV
# ----------------------------------------------------------------------------
if phase_active setup; then
  step "Setup — filter smoke CSV"
  python scripts/filter_bugwood_csv.py \
    --input "$RAW_CSV" \
    --output "$USABLE_CSV" \
    --threshold "${SMOKE_THRESHOLD:-15}" \
    --report smoke/bugwood_classes_smoke.tsv
fi

# ----------------------------------------------------------------------------
# Phase 0 — Claude/SAGE seed (quick, 2 crops only)
# ----------------------------------------------------------------------------
if phase_active 0; then
  step "Phase 0 — pathome_kb seed (quick, Tomato+Soybean)"
  if ! command -v claude >/dev/null 2>&1; then
    echo "[skip] claude CLI not on PATH; Phase 0 needs it."
    echo "       install: curl -fsSL https://claude.ai/install.sh | bash"
  elif [ -z "${ANTHROPIC_API_KEY:-}" ] && [ ! -f .env ]; then
    echo "[skip] no ANTHROPIC_API_KEY in env and no .env at repo root."
  else
    python -m pathome_kb \
      --csv "$USABLE_CSV" \
      --out "$SEED_FILE" \
      --quick \
      --only-crops "Tomato,Soybean"
  fi
fi

# ----------------------------------------------------------------------------
# Phase 1 — Build PathomeDB v1_seed
# ----------------------------------------------------------------------------
if phase_active 1; then
  step "Phase 1 — Build PathomeDB v1_seed"
  python scripts/build_pathome.py --config "$CONFIG"
fi

# ----------------------------------------------------------------------------
# Phase 2 — PlantSwarm traces (GPU-only)
# ----------------------------------------------------------------------------
if phase_active 2; then
  step "Phase 2 — PlantSwarm trace generation"
  if ! has_gpu; then
    echo "[skip] no NVIDIA GPU detected; Phase 2 needs CUDA + Qwen2.5-VL-7B."
  else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    python scripts/run_pathome_traces.py \
      --config "$CONFIG" \
      --orchestrator "${SMOKE_ORCH:-hf_direct}" \
      --pathome-dir "$SEED_DB"
  fi
fi

# ----------------------------------------------------------------------------
# Phase 3 — Enhance DB from traces
# ----------------------------------------------------------------------------
if phase_active 3; then
  step "Phase 3 — Enhance DB from traces"
  if [ ! -f "$TRACES" ]; then
    echo "[skip] traces not found at $TRACES (Phase 2 was skipped)."
  else
    python scripts/enhance_pathome_from_traces.py \
      --seed-db "$SEED_DB" \
      --traces  "$TRACES" \
      --out     "$ENH_DB"
  fi
fi

# ----------------------------------------------------------------------------
# Phase 4 — Train OBSERVE × 2 (GPU-only)
# ----------------------------------------------------------------------------
if phase_active 4; then
  step "Phase 4 — Train OBSERVE × 2 (seed DB then enhanced DB)"
  if ! has_gpu; then
    echo "[skip] no GPU; Phase 4 needs CUDA."
  elif [ ! -f "$TRACES" ]; then
    echo "[skip] traces missing; Phase 2 must run first."
  else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    train_one() {
      local tag="$1" db="$2" ckpt_dir="$3"
      mkdir -p "$ckpt_dir"
      echo "── training [$tag] against $db ──"
      python -c "
import sys, yaml, tempfile, subprocess, os
cfg = yaml.safe_load(open(sys.argv[1]))
cfg.setdefault('pathome', {})['load_dir'] = sys.argv[2]
cfg.setdefault('observe', {})['checkpoint_dir'] = sys.argv[3]
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    yaml.safe_dump(cfg, tf); patched = tf.name
ret = subprocess.call(['python', 'scripts/train_observe_pathome.py',
                       '--config', patched, '--phase', 'both'])
os.unlink(patched); sys.exit(ret)
" "$CONFIG" "$db" "$ckpt_dir"
    }
    train_one "seed"     "$SEED_DB" "smoke/observe/checkpoints/seed"
    if [ -d "$ENH_DB" ]; then
      train_one "enhanced" "$ENH_DB" "smoke/observe/checkpoints/enhanced"
    else
      echo "[skip enhanced] $ENH_DB missing; Phase 3 was skipped."
    fi
  fi
fi

# ----------------------------------------------------------------------------
# Phase 5 — Eval × 4 + comparison (GPU-only)
# ----------------------------------------------------------------------------
if phase_active 5; then
  step "Phase 5 — Eval + comparison"
  if ! has_gpu; then
    echo "[skip] no GPU; Phase 5 needs CUDA."
  elif [ ! -f "$SEED_CKPT" ]; then
    echo "[skip] no seed-DB OBSERVE checkpoint at $SEED_CKPT; Phase 4 must run."
  else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false

    eval_one() {
      local tag="$1" cfg="$2" ckpt="$3" out="$4"
      mkdir -p "$out"
      echo "── eval [$tag] cfg=$cfg ckpt=$ckpt → $out ──"
      python -c "
import sys, yaml, tempfile, subprocess, os
cfg = yaml.safe_load(open(sys.argv[1]))
cfg.setdefault('output', {})['results_dir'] = sys.argv[2]
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    yaml.safe_dump(cfg, tf); patched = tf.name
ret = subprocess.call(['python', 'scripts/evaluate_pathome.py',
                       '--config', patched, '--observe-ckpt', sys.argv[3]])
os.unlink(patched); sys.exit(ret)
" "$cfg" "$out" "$ckpt"
    }

    # PV
    eval_one "seed_PV"     "$PV_EVAL" "$SEED_CKPT" "smoke/results/compare/seed/pv"
    if [ -f "$ENH_CKPT" ]; then
      eval_one "enhanced_PV" "$PV_EVAL" "$ENH_CKPT" "smoke/results/compare/enhanced/pv"
    fi
    # PW
    eval_one "seed_PW"     "$PW_EVAL" "$SEED_CKPT" "smoke/results/compare/seed/pw"
    if [ -f "$ENH_CKPT" ]; then
      eval_one "enhanced_PW" "$PW_EVAL" "$ENH_CKPT" "smoke/results/compare/enhanced/pw"
    fi

    # Comparison artefact
    if [ -f "$ENH_CKPT" ]; then
      python scripts/compare_pathome_versions.py \
        --seed-eval     "smoke/results/compare/seed/pv/pathome_eval.json" \
        --enhanced-eval "smoke/results/compare/enhanced/pv/pathome_eval.json" \
        --seed-traces   "$TRACES" \
        --enhanced-traces "$TRACES" \
        --out-dir       "smoke/results/compare"
    else
      echo "[skip compare] enhanced checkpoint missing; only seed eval ran."
    fi
  fi
fi

echo
echo "==============================================================="
echo "  Smoke run finished: $(date)"
echo "  Results: smoke/results/"
echo "  Compare: smoke/results/compare/comparison.md"
echo "==============================================================="
