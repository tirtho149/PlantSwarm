#!/bin/bash
#SBATCH --job-name=observe_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/observe_eval-%j.out
#SBATCH --error=logs/observe_eval-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Portable paths — override at submit time:
#   PATHOME_REPO=/path/to/PlantSwarm sbatch [--mail-user=...] \
#     scripts/submit_evaluate_observe.sh
PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

# ============================================================================
# Evaluate a trained OBSERVE checkpoint on PlantVillage and/or PlantWild
# (Tomato by default). Reports per-dataset top-1 / top-5 / macro-F1 and
# per-class accuracy (split by KB-known vs zero-shot synthesised classes).
#
# Common overrides:
#   OBSERVE_CKPT=observe/checkpoints/observe_best.pt \
#   PV_ROOT=/path/to/PlantVillage \
#   PW_ROOT=/path/to/PlantWild \
#   OBSERVE_CROP=Tomato \
#     sbatch scripts/submit_evaluate_observe.sh
# ============================================================================

set -e
echo "================================"
echo "OBSERVE evaluation"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source "$PATHOME_REPO/.venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CKPT="${OBSERVE_CKPT:-observe/checkpoints/observe_best.pt}"
OUT="${OBSERVE_EVAL_OUT:-results/observe_eval.json}"
CROP="${OBSERVE_CROP:-Tomato}"
PV_ROOT="${PV_ROOT:-}"
PW_ROOT="${PW_ROOT:-}"
PV_CLASSES_JSON="${PV_CLASSES_JSON:-data/pv_classes.json}"
BACKBONE="${OBSERVE_BACKBONE:-google/siglip-base-patch16-224}"
LORA_R="${OBSERVE_LORA_R:-8}"
LORA_ALPHA="${OBSERVE_LORA_ALPHA:-16}"
BATCH="${OBSERVE_EVAL_BATCH:-32}"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT"
  exit 1
fi
if [ -z "$PV_ROOT" ] && [ -z "$PW_ROOT" ]; then
  echo "ERROR: set at least one of PV_ROOT / PW_ROOT to a folder-per-class dataset."
  exit 1
fi

mkdir -p logs "$(dirname "$OUT")"

CMD=(python scripts/evaluate_observe.py
     --ckpt "$CKPT"
     --crop "$CROP"
     --pv-classes-json "$PV_CLASSES_JSON"
     --backbone "$BACKBONE"
     --lora-r "$LORA_R"
     --lora-alpha "$LORA_ALPHA"
     --batch-size "$BATCH"
     --out "$OUT")

if [ -n "$PV_ROOT" ]; then
  CMD+=(--pv-root "$PV_ROOT")
fi
if [ -n "$PW_ROOT" ]; then
  CMD+=(--pw-root "$PW_ROOT")
fi

"${CMD[@]}"

echo
echo "OBSERVE eval complete: $(date)"
echo "  results: $OUT"
