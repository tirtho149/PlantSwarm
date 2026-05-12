#!/bin/bash
# ============================================================================
# smoke/run_phase0_full.sh
# ============================================================================
# Two-crop end-to-end runner. Build the canonical KB (Claude) and the
# regional deltas (Qwen swarm) for the smoke CSV, then merge into
# smoke/artifacts/pathome_seed/symptoms_seed.json.
#
# Default crops: Soybean + Tomato. Override with SMOKE_CROPS="Crop1,Crop2".
#
# Stages:
#   1. Filter the smoke CSV         → BugWood_Diseases_smoke_usable.csv
#   2. State-aware image cache      → one Bugwood photo per (crop, disease, state)
#   3. Phase 0  canonical KB        → discovery + extraction + reconciliation
#                                     via `claude -p` → final_registry.json
#   4. Phase 0R regional deltas     → Qwen swarm reads canonical + cached
#                                     image and emits state-specific deltas
#                                     {field, canonical_says, image_shows,
#                                      image_quote}; embedded back into
#                                     final_registry.json
#   5. Adapter merge                → symptoms_seed.json
#
# Auth requirements:
#   Phase 0    `claude` CLI on PATH, `claude auth login` (or ANTHROPIC_API_KEY)
#   Phase 0R   OpenAI-compatible vLLM endpoint serving Qwen2.5-VL-7B-Instruct
#              Reachable at $VLLM_BASE_URL (default http://localhost:8000/v1).
#              On a Mac, point this at a remote vLLM via SSH tunnel:
#                  ssh -L 8000:localhost:8000 nova-login
#              Or run on Nova directly where vLLM works natively.
#
# Knobs (env vars):
#   FULL_QUICK=1            cap sources / states for fast iteration
#   FULL_KEEP_CACHE=1       reuse cached final_registry.json (skip canonical re-run)
#   FULL_SKIP_SETUP=1       CSV already filtered
#   FULL_SKIP_CACHE=1       image cache already topped up
#   FULL_SKIP_KB=1          skip the python -m pathome_kb call entirely
#   FULL_SKIP_REGIONAL=1    skip Phase 0R (canonical-only; no Qwen needed)
#   VLLM_BASE_URL           vLLM endpoint for Phase 0R
#   VLLM_MODEL              served model id (default Qwen/Qwen2.5-VL-7B-Instruct)
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW_CSV="smoke/BugWood_Diseases_smoke.csv"
USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
OUT="smoke/artifacts/pathome_seed/symptoms_seed.json"
CACHE_DIR="smoke/.bugwood_cache"
CROPS="${SMOKE_CROPS:-Soybean,Tomato}"

step() {
  echo
  echo "================================================================="
  echo "  $1"
  echo "================================================================="
}

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  echo "  install: curl -fsSL https://claude.ai/install.sh | bash"
  echo "  auth:    claude auth login"
  exit 1
fi

if [ "${FULL_SKIP_REGIONAL:-0}" != "1" ]; then
  VLLM_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
  echo "[preflight] checking vLLM at $VLLM_URL ..."
  if ! curl -sf --max-time 5 "$VLLM_URL/models" >/dev/null 2>&1; then
    echo "  WARNING: vLLM endpoint not reachable at $VLLM_URL."
    echo "  Phase 0R (regional deltas via Qwen swarm) will fail."
    echo "  Set VLLM_BASE_URL or run with FULL_SKIP_REGIONAL=1."
    echo ""
    read -r -p "  Continue anyway? [y/N] " ans
    case "$ans" in
      [yY]*) ;;
      *) exit 1 ;;
    esac
  else
    echo "  ok."
  fi
fi

QUICK_ARG=()
if [ "${FULL_QUICK:-0}" = "1" ]; then
  QUICK_ARG=(--quick)
  echo "[mode] FULL_QUICK=1 — capping sources / states for fast iteration"
else
  echo "[mode] FULL coverage — every source URL, every state"
fi

# ----------------------------------------------------------------------------
# 1. Filter the smoke CSV
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_SETUP:-0}" != "1" ]; then
  step "1. Setup — filter smoke CSV"
  python3 scripts/filter_bugwood_csv.py \
    --input "$RAW_CSV" \
    --output "$USABLE_CSV" \
    --threshold "${SMOKE_THRESHOLD:-15}" \
    --report smoke/bugwood_classes_smoke.tsv
fi

# ----------------------------------------------------------------------------
# 2. State-aware image cache top-up
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_CACHE:-0}" != "1" ]; then
  step "2. State-aware image cache top-up"
  python3 scripts/ensure_state_image_cache.py \
    --csv "$USABLE_CSV" \
    --cache-dir "$CACHE_DIR"
fi

# ----------------------------------------------------------------------------
# 3. Optionally drop stale per-crop registries so the canonical pass re-runs
# ----------------------------------------------------------------------------
if [ "${FULL_KEEP_CACHE:-0}" != "1" ]; then
  step "3a. Clearing stale registries"
  for crop in $(echo "$CROPS" | tr ',' ' '); do
    rm -f "artifacts/pathome_kb/$crop/raw_extractions.json" \
          "artifacts/pathome_kb/$crop/final_registry.json" \
          "artifacts/pathome_kb/$crop/registry.md" \
          "artifacts/pathome_kb/$crop/internet.xlsx"
  done
  echo "  (kept discovery_results.json — re-using cached URLs)"
fi

# ----------------------------------------------------------------------------
# 4 + 5. Phase 0 canonical (claude) + Phase 0R regional (qwen swarm) + merge
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_KB:-0}" != "1" ]; then
  step "3b. Phase 0 (canonical, claude) + Phase 0R (regional, qwen swarm)"

  RESUME_ARG=()
  ALL_HAVE_DISCOVERY=1
  for crop in $(echo "$CROPS" | tr ',' ' '); do
    if [ ! -f "artifacts/pathome_kb/$crop/discovery_results.json" ]; then
      ALL_HAVE_DISCOVERY=0
      break
    fi
  done
  if [ "$ALL_HAVE_DISCOVERY" = "1" ]; then
    echo "  [resume] discovery_results.json present for every crop — resuming from extraction"
    RESUME_ARG=(--resume-from extraction)
  else
    echo "  [fresh]  no cached discovery for at least one crop — running full discovery"
  fi

  REGIONAL_ARG=(--regional)
  if [ "${FULL_SKIP_REGIONAL:-0}" = "1" ]; then
    echo "  [skip] FULL_SKIP_REGIONAL=1 — Phase 0R will not run"
    REGIONAL_ARG=()
  fi

  python3 -m pathome_kb \
    --csv "$USABLE_CSV" \
    --out "$OUT" \
    "${REGIONAL_ARG[@]}" \
    --only-crops "$CROPS" \
    "${RESUME_ARG[@]}" \
    "${QUICK_ARG[@]}"
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
step "Done"
python3 -c "
import json
s = json.load(open('$OUT'))
profiles = s['profiles']
n_canon = sum(1 for p in profiles if (p.get('canonical') or {}).get('summary'))
n_reg   = sum(1 for p in profiles if p.get('regional_observations'))
n_blocks = sum(len(p.get('regional_observations') or {}) for p in profiles)
n_treat = sum(1 for p in profiles if (p.get('canonical') or {}).get('treatments'))
n_deltas = 0
fields_hit = {}
for p in profiles:
    for state, obs in (p.get('regional_observations') or {}).items():
        for d in (obs.get('deltas') or []):
            n_deltas += 1
            fields_hit[d.get('field','other')] = fields_hit.get(d.get('field','other'), 0) + 1
print(f'profiles total                   : {len(profiles)}')
print(f'profiles w/ canonical summary    : {n_canon}')
print(f'profiles w/ canonical treatments : {n_treat}')
print(f'profiles w/ regional observations: {n_reg}')
print(f'total per-state blocks           : {n_blocks}')
print(f'total state-specific deltas      : {n_deltas}')
if fields_hit:
    print('deltas by canonical field:')
    for k, v in sorted(fields_hit.items(), key=lambda x: -x[1]):
        print(f'  {k:24s} {v}')
"
echo
echo "Seed JSON written: $OUT"
