#!/bin/bash
#SBATCH --job-name=pathome_phase0r_regional
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/pathome_phase0r-%j.out
#SBATCH --error=logs/pathome_phase0r-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Portable paths — override at submit time via env vars, e.g.:
#   PATHOME_REPO=/path/to/PlantSwarm \
#   PATHOME_SLURM_EMAIL=you@example.com \
#     sbatch --mail-user="$PATHOME_SLURM_EMAIL" scripts/submit_phase0r_regional.sh
PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

# ============================================================================
# Phase 0R — Qwen-swarm regional delta extraction (Nova A100 + vLLM)
# ============================================================================
# Reads:   artifacts/pathome_kb/<Crop>/final_registry.json   (canonical KB,
#                                                             pushed from LOCAL)
#          .bugwood_cache/                                    (cached images)
#          BugWood_Diseases_usable.csv                        (filtered CSV)
# Writes:  artifacts/pathome_kb/<Crop>/final_registry.json   (deltas embedded
#                                                             under each disease)
#          artifacts/pathome_seed/symptoms_seed.json          (merged seed)
#
# Workflow on this node:
#   1. boot vLLM serving Qwen/Qwen2.5-VL-7B-Instruct on :8000
#   2. wait for /v1/models to respond
#   3. run `python -m pathome_kb --regional-only ...`
#   4. tear down vLLM on exit
#
# Override at submit time:
#   PATHOME_USABLE_CSV=...  PATHOME_SEED_FILE=...  sbatch this script.sh
#   PATHOME_ONLY_CROPS="Soybean,Tomato"  for a smoke-sized run.
#
# Swarm knobs (env vars consumed by plantswarm.delta_pipeline):
#   VLLM_N_RUNS=10        stochastic traces per (crop, disease, state) tuple
#   VLLM_AGREEMENT_MIN=3  K-of-N agreement to keep a delta
#   VLLM_TEMPERATURE=0.8  per-call sampling temperature
#   VLLM_TMAX=15          max path length per trace
#   VLLM_MAX_BACKTRACKS=1 paper §5.3
#   VLLM_SIM_THRESHOLD=0.4 Jaccard threshold for cross-run delta clustering
# ============================================================================

set -e
echo "================================"
echo "Phase 0R: regional deltas (qwen swarm)"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python cuda/11.8
source "$PATHOME_REPO/.venv/bin/activate"
mkdir -p logs

CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
OUT="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"

# Swarm knobs — propagate to plantswarm.delta_pipeline via env.
export VLLM_N_RUNS="${VLLM_N_RUNS:-10}"
export VLLM_AGREEMENT_MIN="${VLLM_AGREEMENT_MIN:-3}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.8}"
export VLLM_TMAX="${VLLM_TMAX:-15}"
export VLLM_MAX_BACKTRACKS="${VLLM_MAX_BACKTRACKS:-1}"
export VLLM_SIM_THRESHOLD="${VLLM_SIM_THRESHOLD:-0.4}"
export VLLM_TIMEOUT="${VLLM_TIMEOUT:-180}"
echo "[swarm] N=$VLLM_N_RUNS K=$VLLM_AGREEMENT_MIN T=$VLLM_TEMPERATURE Tmax=$VLLM_TMAX bt=$VLLM_MAX_BACKTRACKS sim>=$VLLM_SIM_THRESHOLD"

VLLM_LOG="logs/vllm-${SLURM_JOB_ID}.log"

# ---- boot vLLM in the background ------------------------------------------
echo "[vllm] booting $MODEL on :$PORT ..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port  "$PORT" \
  --max-model-len 8192 \
  --trust-remote-code \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "[trap] killing vllm pid=$VLLM_PID"; kill $VLLM_PID 2>/dev/null || true' EXIT

# ---- wait until /v1/models responds ----------------------------------------
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
echo "[vllm] waiting for $VLLM_BASE_URL/models ..."
for i in $(seq 1 60); do
  if curl -sf --max-time 5 "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
    echo "[vllm] up after ${i} * 10s = $((i*10))s"
    break
  fi
  sleep 10
done
if ! curl -sf --max-time 5 "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
  echo "[vllm] FAILED to come up — tail of $VLLM_LOG:"
  tail -n 40 "$VLLM_LOG"
  exit 1
fi

# ---- run Phase 0R ----------------------------------------------------------
ARGS=("--regional-only" "--csv" "$CSV" "--out" "$OUT")
if [ -n "${PATHOME_ONLY_CROPS:-}" ]; then ARGS+=("--only-crops" "$PATHOME_ONLY_CROPS"); fi
if [ "${PATHOME_SEED_QUICK:-0}" = "1" ]; then ARGS+=("--quick"); fi

echo "================================"
echo "running: python -m pathome_kb ${ARGS[*]}"
echo "================================"
python -m pathome_kb "${ARGS[@]}"

echo
echo "Phase 0R complete: $(date)"
