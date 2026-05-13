#!/bin/bash
#SBATCH --job-name=observe_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/observe_train-%j.out
#SBATCH --error=logs/observe_train-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Portable paths — override at submit time:
#   PATHOME_REPO=/path/to/PlantSwarm sbatch [--mail-user=...] scripts/submit_observe_train.sh
PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

# ============================================================================
# Train OBSERVE as a KB-augmented OOD classifier on Bugwood (Tomato by default).
# ============================================================================
# Architecture:
#   image -> SigLIP-2 vision tower (frozen base + LoRA q/k/v) -> embedding
#   class prototype texts (canonical + regional KB blocks; +healthy template)
#     -> SigLIP-2 text tower (frozen) -> [C, D]
#   prediction = argmax( cosine(image, class_proto) * temperature )
#
# Inputs:
#   $PATHOME_SEED_JSON     symptoms_seed.json  (Phase 0 + 0R output)
#   $PATHOME_BUGWOOD_CSV   filtered Bugwood CSV (NormCrop, image_number, etc.)
#   $PATHOME_BUGWOOD_CACHE comma-separated image cache dirs
#   $OBSERVE_CROP          crop filter, default "Tomato"
#
# Outputs:
#   observe/checkpoints/observe_best.pt   (best by val top-1)
#   observe/checkpoints/observe_last.pt
#   observe/checkpoints/history.json
#
# Common overrides:
#   PATHOME_SEED_JSON=artifacts/pathome_seed/symptoms_seed.json \
#   PATHOME_BUGWOOD_CSV=BugWood_Diseases_usable.csv \
#   PATHOME_BUGWOOD_CACHE=.bugwood_cache \
#   OBSERVE_CROP=Tomato OBSERVE_EPOCHS=10 OBSERVE_LR=1e-4 \
#     sbatch scripts/submit_observe_train.sh
# ============================================================================

set -e
echo "================================"
echo "OBSERVE training (KB-augmented OOD classifier)"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source "$PATHOME_REPO/.venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

SEED_JSON="${PATHOME_SEED_JSON:-artifacts/pathome_seed/symptoms_seed.json}"
BUGWOOD_CSV="${PATHOME_BUGWOOD_CSV:-BugWood_Diseases_usable.csv}"
CACHE_DIRS="${PATHOME_BUGWOOD_CACHE:-.bugwood_cache}"
CROP="${OBSERVE_CROP:-Tomato}"
SAVE_DIR="${OBSERVE_SAVE_DIR:-observe/checkpoints/}"
BACKBONE="${OBSERVE_BACKBONE:-google/siglip-base-patch16-224}"
EPOCHS="${OBSERVE_EPOCHS:-10}"
BATCH="${OBSERVE_BATCH:-32}"
LR="${OBSERVE_LR:-1e-4}"
LORA_R="${OBSERVE_LORA_R:-8}"
LORA_ALPHA="${OBSERVE_LORA_ALPHA:-16}"
VAL_FRAC="${OBSERVE_VAL_FRAC:-0.15}"
INCLUDE_HEALTHY_FLAG="${OBSERVE_INCLUDE_HEALTHY:-1}"

INCLUDE_HEALTHY_ARG=""
if [ "$INCLUDE_HEALTHY_FLAG" = "1" ] || [ "$INCLUDE_HEALTHY_FLAG" = "true" ]; then
  INCLUDE_HEALTHY_ARG="--include-healthy"
fi

if [ ! -f "$SEED_JSON" ]; then
  echo "ERROR: KB seed JSON not found at $SEED_JSON"
  echo "Run Phase 0 (and optionally Phase 0R) to populate it."
  exit 1
fi
if [ ! -f "$BUGWOOD_CSV" ]; then
  echo "ERROR: filtered Bugwood CSV not found at $BUGWOOD_CSV"
  exit 1
fi

mkdir -p logs "$SAVE_DIR"

python scripts/train_observe.py \
  --seed "$SEED_JSON" \
  --bugwood-csv "$BUGWOOD_CSV" \
  --cache-dir "$CACHE_DIRS" \
  --crop "$CROP" \
  $INCLUDE_HEALTHY_ARG \
  --save-dir "$SAVE_DIR" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --val-frac "$VAL_FRAC"

echo
echo "OBSERVE training complete: $(date)"
