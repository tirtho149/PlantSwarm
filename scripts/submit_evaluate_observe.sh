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
# Evaluate a trained OBSERVE checkpoint on the held-out slice of the trace
# JSONL. Reports routing_accuracy, backtrack_acc, kappa_ece + MAE, OC accuracy.
#
# Override at submit time:
#   OBSERVE_CKPT=...  PATHOME_TRACE_FILE=...  OBSERVE_HELD_FRAC=0.1 \
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
TRACES="${PATHOME_TRACE_FILE:-artifacts/observe_traces/phase0r_traces.jsonl}"
OUT="${OBSERVE_EVAL_OUT:-results/observe_eval.json}"
HELD_FRAC="${OBSERVE_HELD_FRAC:-0.1}"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT"
  exit 1
fi
if [ ! -f "$TRACES" ]; then
  echo "ERROR: trace JSONL not found at $TRACES"
  exit 1
fi

mkdir -p logs "$(dirname "$OUT")"

python scripts/evaluate_observe.py \
  --ckpt "$CKPT" \
  --traces "$TRACES" \
  --out "$OUT" \
  --held-frac "$HELD_FRAC"

echo
echo "OBSERVE eval complete: $(date)"
echo "  results: $OUT"
