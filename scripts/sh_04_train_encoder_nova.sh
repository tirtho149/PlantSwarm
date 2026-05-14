#!/bin/bash
# ============================================================================
# scripts/sh_04_train_encoder_nova.sh           STEP 4 — NOVA  (NEW)
# ============================================================================
# Fine-tune your OWN encoder, BioCAP-style, on Bugwood + KB-grounded
# captions. This produces a domain-specialized image+text encoder that
# the next step (sh_05_tabpfn_local.sh) consumes alongside the off-
# shelf encoders in the importance ablation.
#
# What this trains
#   ViT-B/16 dual-projector CLIP, warm-started from BioCLIP, with
#   KB-grounded captions as the second supervisory signal (canonical
#   visual_symptoms + per-state regional deltas).
#
#   Default: just the MAIN variant (T04 = canonical_deltas_3 caption,
#            dual projector, 50 epochs, projectors-only training). One
#            checkpoint. ~30-60 min on one A100.
#
#   Optional: TRAIN_FULL_MATRIX=1 to sbatch all 11 caption-strategy /
#             projector-mode / epoch-count variants. ~5 GPU-h total.
#             Useful for the full BioCAP-paper-style ablation BEFORE
#             the TabPFN matrix takes over.
#
# Output
#   train_and_eval/checkpoints/<VARIANT>/<run-id>/checkpoints/epoch_50.pt
#   git push to origin/main so step 5 on LOCAL can git pull and use it.
#
# Knobs
#   CROPS              "smoke" = Tomato; "all" = full Bugwood
#   TRAIN_VARIANT      single variant tag to train (default T04)
#   TRAIN_FULL_MATRIX  set =1 to sbatch all 11 variants instead of just one
#   PATHOME_SKIP_{CAPTIONS,SHARDS,TRAIN,PUSH}   0/1 toggles
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

CROPS="${CROPS:-smoke}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
TRAIN_VARIANT="${TRAIN_VARIANT:-T04}"

PY="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
if [ -z "$PY" ]; then
  echo "ERROR: no python / python3 on PATH. Install Python 3 or set PYTHON_BIN."
  exit 2
fi

case "$CROPS" in
  smoke) CROP_TAG="Tomato";;
  all)   CROP_TAG="all";;
  *)     CROP_TAG="$CROPS";;
esac

echo "================================================================="
echo " STEP 4 — Encoder fine-tune (NOVA, BioCAP-style)"
echo "================================================================="
echo "  CROP_TAG          : $CROP_TAG"
echo "  TRAIN_VARIANT     : $TRAIN_VARIANT"
echo "  TRAIN_FULL_MATRIX : ${TRAIN_FULL_MATRIX:-0}"
echo

# Pull verified KB from step 3.
echo "[1/4] git pull verified KB"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only
mkdir -p logs data/bugwood_captions data/wds_shards train_and_eval/checkpoints

# Build captions for the strategy this training variant needs.
# Resolve the strategy from the variant tag using pathomeood_variants.sh.
# shellcheck disable=SC1091
source scripts/pathomeood_variants.sh
strategy=""
for v in "${PATHOMEOOD_VARIANTS[@]}"; do
  pathomeood_parse_variant "$v"
  if [ "$VARIANT_TAG" = "$TRAIN_VARIANT" ]; then
    strategy="$STRATEGY"
    break
  fi
done
if [ -z "$strategy" ]; then
  echo "ERROR: unknown TRAIN_VARIANT=$TRAIN_VARIANT"
  exit 2
fi

if [ "${PATHOME_SKIP_CAPTIONS:-0}" != "1" ]; then
  capt="data/bugwood_captions/${CROP_TAG}_${strategy}.parquet"
  if [ ! -f "$capt" ] && [ ! -f "${capt%.parquet}.tsv" ]; then
    echo
    echo "[2/4] Build captions for strategy=$strategy"
    if [ "$CROP_TAG" = "all" ]; then
      "$PY" scripts/build_pathomeood_captions.py --strategy "$strategy" --out "$capt"
    else
      "$PY" scripts/build_pathomeood_captions.py --strategy "$strategy" --crop "$CROP_TAG" --out "$capt"
    fi
  else
    echo "[2/4] captions for $strategy already exist; skipping"
  fi
else
  echo "[2/4] [skip] PATHOME_SKIP_CAPTIONS=1"
fi

# Build WebDataset shards for the strategy(ies) we'll train.
if [ "${PATHOME_SKIP_SHARDS:-0}" != "1" ]; then
  echo
  echo "[3/4] Build WebDataset shards"
  shards_root="data/wds_shards/${CROP_TAG}_${strategy}"
  if [ ! -d "$shards_root/train" ]; then
    capt="data/bugwood_captions/${CROP_TAG}_${strategy}.parquet"
    [ -f "$capt" ] || capt="data/bugwood_captions/${CROP_TAG}_${strategy}.tsv"
    "$PY" scripts/build_pathomeood_shards.py --captions "$capt" --out-dir "$shards_root"
  else
    echo "  shards already built at $shards_root; skipping"
  fi
else
  echo "[3/4] [skip] PATHOME_SKIP_SHARDS=1"
fi

# Train.
if [ "${PATHOME_SKIP_TRAIN:-0}" != "1" ]; then
  echo
  if [ "${TRAIN_FULL_MATRIX:-0}" = "1" ]; then
    echo "[4/4] sbatch all 11 BioCAP-style training variants"
    PATHOME_WAIT=1 PATHOME_SKIP_CAPTIONS=1 CROP="$CROP_TAG" \
      bash scripts/submit_pathomeood_matrix.sh
  else
    echo "[4/4] sbatch ONE training variant: $TRAIN_VARIANT"
    VARIANT="$TRAIN_VARIANT" CROP="$CROP_TAG" \
      sbatch --wait scripts/submit_pathomeood_train.sh
  fi
else
  echo "[4/4] [skip] PATHOME_SKIP_TRAIN=1"
fi

# Push the trained checkpoint(s) — note these are large (~600MB each).
# Only push if PATHOME_PUSH_CHECKPOINT=1 is set; otherwise just announce.
echo
echo "[push] handling trained checkpoints"
ckpt_glob="train_and_eval/checkpoints/${TRAIN_VARIANT}/*/checkpoints/epoch_*.pt"
last_ckpt=$(ls $ckpt_glob 2>/dev/null | sort -V | tail -n 1 || true)
if [ -z "$last_ckpt" ]; then
  echo "  no checkpoint found at $ckpt_glob — did training succeed?"
elif [ "${PATHOME_PUSH_CHECKPOINT:-0}" = "1" ]; then
  echo "  PATHOME_PUSH_CHECKPOINT=1 — committing checkpoint $last_ckpt"
  git add -f "$last_ckpt"
  git commit -m "Phase PathomeOOD: trained encoder ckpt ($TRAIN_VARIANT)"
  if [ "${PATHOME_SKIP_PUSH:-0}" != "1" ]; then
    git push "$GIT_REMOTE" "$GIT_BRANCH"
  fi
else
  echo "  checkpoint at $last_ckpt"
  echo "  (NOT pushing — checkpoints are large. Use PATHOME_PUSH_CHECKPOINT=1"
  echo "   to push, or copy manually via scp / rsync to your local machine"
  echo "   before running step 5.)"
fi

echo
echo "STEP 4 done."
echo "  Next: scp the checkpoint to LOCAL, then run scripts/sh_05_tabpfn_local.sh"
echo "        with PATHOMEOOD_CKPT=path/to/epoch_50.pt set."
