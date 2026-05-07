#!/bin/bash
#SBATCH --job-name=plantswarm_traces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/phase1_plantswarm-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/phase1_plantswarm-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 1: Generate PlantSwarm Routing Traces on PlantVillage
# ============================================================================
# Output: plantswarm_metrics.json, traces/plantswarm_traces.jsonl
#
# Throughput notes:
#   * A100 + vLLM (autogen_swarm)  : ~12-18 h for 10K images
#   * V100 + hf_direct             : ~10 min/image — NOT VIABLE for 10K
#   * Use --subset 100 for a smoke test before launching the full run.
#
# Resume: run_plantswarm.py appends each trace to traces/plantswarm_traces.jsonl
# with fsync; if SLURM walltime kills the job, just resubmit — already-done
# image_ids are skipped automatically.
#
# Switch GPU type by editing the line:
#   #SBATCH --gres=gpu:a100:1   # A100 (preferred)
#   #SBATCH --gres=gpu:v100:1   # fallback — VERY SLOW with hf_direct
# ============================================================================

set -e  # Exit on error

echo "================================"
echo "Phase 1: PlantSwarm Routing Traces"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
nvidia-smi || true
echo ""

# Load modules
module load python cuda/11.8

# Activate Python environment (adjust path as needed)
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate

# Create logs directory
mkdir -p logs

# ----------------------------------------------------------------------------
# Mode A (default): hf_direct — single GPU, no server, simplest. Slow on V100.
# Mode B: autogen_swarm — boots a vLLM server in this job, then runs the swarm.
#         Much faster (batching). Uncomment the block below to use.
# ----------------------------------------------------------------------------

MODE="${PLANTSWARM_MODE:-hf_direct}"

if [ "$MODE" = "autogen_swarm" ]; then
  echo "Booting vLLM server (Qwen2.5-VL-7B-Instruct)..."
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --trust-remote-code \
    > logs/vllm-${SLURM_JOB_ID}.log 2>&1 &
  VLLM_PID=$!
  trap "kill $VLLM_PID 2>/dev/null || true" EXIT

  # Wait for server to be ready (up to 5 min)
  for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      echo "vLLM server ready after ${i}*5s"
      break
    fi
    sleep 5
  done

  echo "Starting PlantSwarm (autogen_swarm) generation..."
  python scripts/run_plantswarm.py \
    --config configs/plant_village_tfds.yaml \
    --orchestrator autogen_swarm
else
  echo "Starting PlantSwarm (hf_direct) generation..."
  python scripts/run_plantswarm.py \
    --config configs/plant_village_tfds.yaml \
    --orchestrator hf_direct
fi

echo ""
echo "Phase 1 Complete"
echo "Output:"
echo "  - results/plant_village_tfds/plantswarm_metrics.json"
echo "  - results/plant_village_tfds/traces/plantswarm_traces.jsonl"
echo "End time: $(date)"
