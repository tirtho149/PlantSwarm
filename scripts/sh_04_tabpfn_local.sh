#!/bin/bash
# ============================================================================
# scripts/sh_04_tabpfn_local.sh         STEP 4 — LOCAL (or any small-GPU host)
# ============================================================================
# Replaces the old sh_04_finetune_nova.sh CLIP training step with the
# TabPFN-based pipeline. Frozen encoder + KB-derived caption embeddings +
# crop one-hot, classified by TabPFN. Designed for the small-data regime
# (~11K Bugwood images).
#
# Pipeline
#   1. git pull verified KB (step 3 output)
#   2. (re)build captions per strategy (reuses scripts/build_pathomeood_captions.py)
#   3. encoder forward pass over Bugwood + PV / PD / PW
#      -> data/bugwood_features/<encoder>_<strategy>.npz
#      -> data/eval_features/<encoder>_<strategy>_<dataset>.npz
#   4. TabPFN classifier over the 11-variant feature-ablation matrix
#      -> results/pathomeood_eval/<variant>/<dataset>.json
#   5. aggregate paper-style tables
#   6. git push results
#
# Cost
#   Step 3 (encoder pass): ~5-15 min on one A100, ~30 min on a single GPU.
#     With multiple encoders (T08/T09 use clip / siglip too) this scales
#     linearly per encoder.
#   Step 4 (TabPFN matrix): ~5-15 min on CPU.
#   Total: usually ≤ 1 hour, even for the full set.
#
# Knobs
#   CROPS                  "smoke" = Tomato (only KB-covered crop) or
#                          "all" = full Bugwood with fallback captions
#   PV_ROOT / PW_ROOT / PLANTDOC_ROOT   eval-set folder-per-class roots
#   ENCODERS               comma-list of encoders to build features for
#                          (default: bioclip,clip_vitb16,siglip_vitb16)
#   STRATEGIES             comma-list of caption strategies to build
#                          (default: all 7)
#   PATHOME_SKIP_{CAPTIONS,FEATURES,TABPFN,AGG,PUSH}   0/1
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

CROPS="${CROPS:-smoke}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

case "$CROPS" in
  smoke) CROP_TAG="Tomato";;
  all)   CROP_TAG="all";;
  *)     CROP_TAG="$CROPS";;
esac

RESULTS_DIR="${RESULTS_DIR:-results/pathomeood_eval}"
PV_ROOT="${PV_ROOT:-data/eval/PlantVillage}"
PW_ROOT="${PW_ROOT:-data/eval/PlantWild}"
PLANTDOC_ROOT="${PLANTDOC_ROOT:-data/eval/PlantDoc/test}"

ENCODERS="${ENCODERS:-bioclip,bioclip2,clip_vitb16,siglip_vitb16,fgclip,biotrove}"
STRATEGIES="${STRATEGIES:-label_only,summary_only,canonical_full,canonical_deltas_1,canonical_deltas_3,canonical_deltas_5,canonical_deltas_7}"
GRADCAM_PER_CLASS="${GRADCAM_PER_CLASS:-3}"

echo "================================================================="
echo " STEP 4 — TabPFN classification over PathomeOOD features"
echo "================================================================="
echo "  CROP_TAG     : $CROP_TAG"
echo "  ENCODERS     : $ENCODERS"
echo "  STRATEGIES   : $STRATEGIES"
echo "  RESULTS_DIR  : $RESULTS_DIR"

# Step 1 — pull verified KB.
echo
echo "[1/6] git pull verified KB"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only
mkdir -p data/bugwood_captions data/bugwood_features data/eval_features \
         "$RESULTS_DIR"

# Step 2 — captions per strategy.
if [ "${PATHOME_SKIP_CAPTIONS:-0}" != "1" ]; then
  echo
  echo "[2/6] Build captions for each strategy"
  IFS=',' read -ra strat_arr <<<"$STRATEGIES"
  for s in "${strat_arr[@]}"; do
    capt="data/bugwood_captions/${CROP_TAG}_${s}.parquet"
    if [ ! -f "$capt" ] && [ ! -f "${capt%.parquet}.tsv" ]; then
      echo "  [captions] strategy=$s"
      if [ "$CROP_TAG" = "all" ]; then
        python scripts/build_pathomeood_captions.py --strategy "$s" --out "$capt"
      else
        python scripts/build_pathomeood_captions.py --strategy "$s" --crop "$CROP_TAG" --out "$capt"
      fi
    else
      echo "  [captions] strategy=$s already built"
    fi
  done
else
  echo "  [skip] PATHOME_SKIP_CAPTIONS=1"
fi

# Step 3 — encoder forward pass over (encoder × strategy).
if [ "${PATHOME_SKIP_FEATURES:-0}" != "1" ]; then
  echo
  echo "[3/6] Encode Bugwood + PV/PD/PW features per (encoder × strategy)"
  IFS=',' read -ra enc_arr   <<<"$ENCODERS"
  IFS=',' read -ra strat_arr <<<"$STRATEGIES"
  for enc in "${enc_arr[@]}"; do
    for s in "${strat_arr[@]}"; do
      out_train="data/bugwood_features/${enc}_${s}.npz"
      if [ -f "$out_train" ]; then
        echo "  [features] $enc / $s already built; skipping"
        continue
      fi
      capt="data/bugwood_captions/${CROP_TAG}_${s}.parquet"
      [ -f "$capt" ] || capt="data/bugwood_captions/${CROP_TAG}_${s}.tsv"
      if [ ! -f "$capt" ]; then
        echo "  [features] missing captions for $enc / $s; skipping"
        continue
      fi
      echo "  [features] encoder=$enc strategy=$s"
      python scripts/build_features.py \
          --captions "$capt" \
          --encoder  "$enc" \
          --eval-pv  "$PV_ROOT" \
          --eval-pd  "$PLANTDOC_ROOT" \
          --eval-pw  "$PW_ROOT"
    done
  done
else
  echo "  [skip] PATHOME_SKIP_FEATURES=1"
fi

# Step 4 — TabPFN matrix (14 variants + 6 zero-shot baselines + few-shot).
if [ "${PATHOME_SKIP_TABPFN:-0}" != "1" ]; then
  echo
  echo "[4/7] TabPFN classifier — 14-variant matrix + zero-shot baselines + few-shot"
  python scripts/tabpfn_eval.py \
      --features-root data/bugwood_features \
      --eval-root     data/eval_features \
      --results-dir   "$RESULTS_DIR" \
      --include-baselines \
      --include-fewshot
else
  echo "  [skip] PATHOME_SKIP_TABPFN=1"
fi

# Step 5 — Grad-CAM (qualitative figures + quantitative energy-pointing
# if bboxes are present).
if [ "${PATHOME_SKIP_GRADCAM:-0}" != "1" ]; then
  echo
  echo "[5/7] Grad-CAM per encoder × eval set"
  IFS=',' read -ra enc_arr <<<"$ENCODERS"
  for enc in "${enc_arr[@]}"; do
    for kind_root_pair in "plantvillage:$PV_ROOT" \
                          "plantdoc:$PLANTDOC_ROOT" \
                          "plantwild:$PW_ROOT"; do
      IFS=':' read -r kind root <<<"$kind_root_pair"
      if [ ! -d "$root" ]; then
        echo "  [$enc/$kind] root $root not found; skipping"
        continue
      fi
      echo "  [gradcam] $enc / $kind"
      python scripts/gradcam_eval.py \
          --encoder        "$enc" \
          --eval-root      "$root" \
          --eval-kind      "$kind" \
          --max-per-class  "$GRADCAM_PER_CLASS" \
          ${BBOX_CSV:+--bbox-csv "$BBOX_CSV"} \
          || echo "    Grad-CAM failed for $enc/$kind; continuing"
    done
  done
else
  echo "  [skip] PATHOME_SKIP_GRADCAM=1"
fi

# Step 6 — aggregate paper-style tables.
if [ "${PATHOME_SKIP_AGG:-0}" != "1" ]; then
  echo
  echo "[6/7] Aggregate paper-style tables"
  python scripts/aggregate_pathomeood_tables.py --results-dir "$RESULTS_DIR" \
      --out-dir results/tables --report results/pathomeood_report.md
else
  echo "  [skip] PATHOME_SKIP_AGG=1"
fi

# Step 7 — push.
echo
echo "[7/7] git push results"
git add -f results/pathomeood_report.md \
           results/tables/*.md \
           "$RESULTS_DIR"/*/*.json \
           results/figures/gradcam/*/*/*/*.png 2>/dev/null || true
if git diff --cached --quiet; then
  echo "  no results changed; skipping commit"
else
  git commit -m "Step 4 (TabPFN): $CROP_TAG ($(date -u +%Y-%m-%dT%H:%MZ))"
  if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
    echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
  else
    git push "$GIT_REMOTE" "$GIT_BRANCH"
  fi
fi

echo
echo "STEP 4 done."
echo "  Master report: results/pathomeood_report.md"
