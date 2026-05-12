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
# Train OBSERVE on Phase 0R traces — delta-mode supervision.
# ============================================================================
# Inputs:
#   $PATHOME_TRACE_FILE   default: artifacts/observe_traces/phase0r_traces.jsonl
#                         produced by Phase 0R when PATHOME_TRACE_DIR is set
#
# Outputs:
#   observe/checkpoints/observe_best.pt
#   observe/checkpoints/observe_last.pt
#   observe/checkpoints/history.json
#
# Override at submit time:
#   PATHOME_TRACE_FILE=...  OBSERVE_EPOCHS=10  OBSERVE_LR=1e-4 \
#     sbatch scripts/submit_observe_train.sh
# ============================================================================

set -e
echo "================================"
echo "OBSERVE training"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source "$PATHOME_REPO/.venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

TRACES="${PATHOME_TRACE_FILE:-artifacts/observe_traces/phase0r_traces.jsonl}"
SAVE_DIR="${OBSERVE_SAVE_DIR:-observe/checkpoints/}"
EPOCHS="${OBSERVE_EPOCHS:-5}"
BATCH="${OBSERVE_BATCH:-4}"
LR="${OBSERVE_LR:-1e-4}"
LORA_R="${OBSERVE_LORA_R:-16}"
LORA_ALPHA="${OBSERVE_LORA_ALPHA:-32}"

if [ ! -f "$TRACES" ]; then
  echo "ERROR: trace JSONL not found at $TRACES"
  echo "Run Phase 0R first with PATHOME_TRACE_DIR set to populate it."
  exit 1
fi

mkdir -p logs "$SAVE_DIR"

python scripts/train_observe.py \
  --traces "$TRACES" \
  --save-dir "$SAVE_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA"

echo
echo "OBSERVE training complete: $(date)"
