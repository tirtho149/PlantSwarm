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

# ---- guaranteed persistent log --------------------------------------------
# The #SBATCH --output=logs/... path is RELATIVE to the submit dir and is
# opened by SLURM at job start — before this script can `mkdir -p logs`.
# If that dir doesn't exist at submit time the SLURM log is lost. So we
# ALSO tee everything to an absolute path under $PATHOME_REPO/logs that
# we create right now. Override dir with PATHOME_LOG_DIR.
PATHOME_LOG_DIR="${PATHOME_LOG_DIR:-$PATHOME_REPO/logs}"
mkdir -p "$PATHOME_LOG_DIR"
RUNLOG="$PATHOME_LOG_DIR/phase0r_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUNLOG") 2>&1
echo "[log] full job output is being stored at: $RUNLOG"

# ---- load secrets/env from .env (gitignored) ------------------------------
# e.g. HF_TOKEN to remove the HuggingFace unauthenticated rate limit
# (the real cause of the slow Qwen2.5-VL download). Auto-export every
# var defined there. Contents are NEVER printed.
if [ -f "$PATHOME_REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$PATHOME_REPO/.env"
  set +a
  echo "[env-file] loaded $PATHOME_REPO/.env (HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset))"
fi

# ============================================================================
# Phase 0R — Qwen-swarm regional delta extraction (Nova A100, in-process vLLM)
# ============================================================================
# Reads:   artifacts/pathome_kb/<Crop>/final_registry.json   (canonical KB)
#          .bugwood_cache/                                    (cached images)
#          BugWood_Diseases_usable.csv                        (filtered CSV)
# Writes:  artifacts/pathome_kb/<Crop>/final_registry.json   (deltas embedded
#                                                             under each disease)
#          artifacts/pathome_seed/symptoms_seed.json          (merged seed)
#
# Workflow on this node:
#   1. top up .bugwood_cache with one image per (crop, disease, state) tuple
#   2. run `python -m pathome_kb --regional-only ...`
#      — the swarm loads vllm.LLM IN-PROCESS via utils/vllm_inproc.py
#      — NO `vllm serve` boot, NO HTTP, NO port to wait on
#
# Why no HTTP server: the original architecture booted `vllm serve` and
# talked to it from the same job over OpenAI-compatible JSON. On Nova this
# produced silent `HTTPError: 400 Client Error` on every specialist call
# under certain (image, prompt) combinations and zeroed out every tuple.
# An HTTP boundary inside a single-node job has no benefit; the engine is
# now embedded.
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
#
# In-process model knobs (utils/vllm_inproc.get_inproc_client):
#   VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
#   VLLM_MAX_MODEL_LEN=32768
#   VLLM_MIN_PIXELS=50176   ~224 px on the short side
#   VLLM_MAX_PIXELS=1003520 ~1024 px on the long side
#   VLLM_MAX_NEW_TOKENS=512
#   VLLM_GPU_MEMORY_UTIL=0.90
#   VLLM_DTYPE=auto
#   VLLM_INPROCESS=1        default; set 0 to use the legacy HTTP client
# ============================================================================

set -e
echo "================================"
echo "Phase 0R: regional deltas (qwen swarm, in-process vLLM)"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

# pip-installed torch BUNDLES its own CUDA runtime; it needs only the
# NVIDIA kernel driver, NOT a system CUDA toolkit. Loading a system
# `cuda/12.8` module shadows torch's bundled libcudart with a
# mismatched one -> "CUDA unknown error ... Setting the available
# devices to be zero" on the very first torch CUDA init. So by default
# we load ONLY python. Set PATHOME_LOAD_CUDA_MODULE=1 to restore the
# old behaviour if your torch was built against the system toolkit.
if [ "${PATHOME_LOAD_CUDA_MODULE:-0}" = "1" ]; then
  module load python cuda/12.8
else
  module load python
fi

# Resolve the venv. Default: $PATHOME_REPO/.venv (in-repo). Override with
# PATHOME_VENV=/path/to/venv (e.g. one level above the repo, shared
# across projects). Falls back to ../.venv if neither exists.
VENV="${PATHOME_VENV:-$PATHOME_REPO/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  if [ -f "$(dirname "$PATHOME_REPO")/.venv/bin/activate" ]; then
    VENV="$(dirname "$PATHOME_REPO")/.venv"
  else
    echo "ERROR: no venv found. Tried:"
    echo "  $PATHOME_VENV (PATHOME_VENV)"
    echo "  $PATHOME_REPO/.venv"
    echo "  $(dirname "$PATHOME_REPO")/.venv"
    echo "Set PATHOME_VENV=/path/to/venv and re-sbatch."
    exit 2
  fi
fi
echo "venv: $VENV"
source "$VENV/bin/activate"
mkdir -p logs

# ---- environment setter: self-heal torch/CUDA -----------------------------
# Idempotent: no-op (no pip) when torch.cuda + transformers are already
# good; otherwise detects the driver and installs a matching torch wheel.
# Set PATHOME_SKIP_ENV_SETUP=1 to skip. Compute nodes often have no
# internet — if a heal is actually needed it will say to run
# `bash scripts/setup_env.sh` on a login node, and we abort early.
# ---- HuggingFace cache on /work (NOT $HOME) + fast downloads ---------------
# $HOME on HPC is small/quota'd; a 16GB Qwen2.5-VL download there stalls
# or fills the quota. Put the HF cache on the big shared /work fs so it
# (a) never hits a home quota and (b) is reused across nodes/runs (one
# download ever). hf_transfer makes the pull much faster; the
# unauthenticated rate-limit is the usual "stuck at 20%" cause — set
# HF_TOKEN to remove it.
export HF_HOME="${HF_HOME:-$PATHOME_REPO/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
python -c "import hf_transfer" 2>/dev/null || \
  pip install --quiet hf_transfer 2>/dev/null || \
  export HF_HUB_ENABLE_HF_TRANSFER=0   # fall back if the pkg can't install
echo "[hf] HF_HOME=$HF_HOME  hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER  token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"

if [ "${PATHOME_SKIP_ENV_SETUP:-0}" != "1" ]; then
  if ! PATHOME_VENV="$VENV" bash "$PATHOME_REPO/scripts/setup_env.sh"; then
    echo "[env] environment not ready on node ${SLURMD_NODENAME:-?}."
    echo "      If setup_env reported a NODE CUDA FAULT (packages fine,"
    echo "      GPU node can't init CUDA), resubmit excluding this node:"
    echo "        sbatch --exclude=${SLURMD_NODENAME:-<thisnode>} scripts/submit_phase0r_regional.sh"
    echo "      If it was a missing-package issue, run on a login node:"
    echo "        PATHOME_VENV=$VENV bash $PATHOME_REPO/scripts/setup_env.sh"
    exit 3
  fi
fi

# ---- CUDA preflight: fail fast with a CLEAR diagnostic ---------------------
# torch._C._cuda_init() was raising "CUDA unknown error ... Setting the
# available devices to be zero" deep inside model load. Surface the real
# state here before anything heavy runs.
echo "---- CUDA preflight ----"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  SLURM_GPUS=${SLURM_GPUS:-<unset>}"
nvidia-smi -L 2>&1 || echo "  nvidia-smi -L failed (no GPU bound to this job?)"
# Driver version vs torch's bundled CUDA is the usual culprit: a
# cu128 torch needs driver >= R570. Log the driver's max CUDA so the
# right torch wheel is obvious from the log alone.
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
  | sed 's/^/  NVIDIA driver: /' || true
nvidia-smi 2>/dev/null | grep -m1 "CUDA Version" | sed 's/^/  /' || true
python - <<'PY' || { echo "[preflight] torch cannot init CUDA — aborting before model load. See hints below."; \
  echo "  1) Is a GPU actually allocated? (sbatch has --gres=gpu:a100:1; check 'squeue --me' / nvidia-smi above)"; \
  echo "  2) System CUDA module vs pip-torch bundled CUDA mismatch — this script now loads ONLY 'python' by default."; \
  echo "  3) torch built without CUDA: python -c 'import torch;print(torch.version.cuda)'"; \
  echo "  4) Try an interactive node: srun --gres=gpu:a100:1 --partition=nova --pty bash, then the same python check."; \
  exit 3; }
import sys, torch
print("torch", torch.__version__, "| torch.version.cuda", torch.version.cuda)
ok = torch.cuda.is_available()
print("torch.cuda.is_available():", ok, "| device_count:", torch.cuda.device_count() if ok else 0)
if not ok:
    sys.exit(1)
torch.zeros(1).cuda()          # force a real CUDA context now
print("CUDA preflight OK:", torch.cuda.get_device_name(0))
PY
echo "------------------------"

CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
OUT="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"

# Swarm knobs — propagate to plantswarm.delta_pipeline via env.
export VLLM_N_RUNS="${VLLM_N_RUNS:-10}"
export VLLM_AGREEMENT_MIN="${VLLM_AGREEMENT_MIN:-3}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.8}"
export VLLM_TMAX="${VLLM_TMAX:-15}"
export VLLM_MAX_BACKTRACKS="${VLLM_MAX_BACKTRACKS:-1}"
export VLLM_SIM_THRESHOLD="${VLLM_SIM_THRESHOLD:-0.4}"
echo "[swarm] N=$VLLM_N_RUNS K=$VLLM_AGREEMENT_MIN T=$VLLM_TEMPERATURE Tmax=$VLLM_TMAX bt=$VLLM_MAX_BACKTRACKS sim>=$VLLM_SIM_THRESHOLD"

# In-process vLLM knobs.
export VLLM_INPROCESS="${VLLM_INPROCESS:-1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MIN_PIXELS="${VLLM_MIN_PIXELS:-50176}"
export VLLM_MAX_PIXELS="${VLLM_MAX_PIXELS:-1003520}"
export VLLM_MAX_NEW_TOKENS="${VLLM_MAX_NEW_TOKENS:-512}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.90}"
export VLLM_DTYPE="${VLLM_DTYPE:-auto}"
# Run vLLM's engine core IN-PROCESS (no EngineCore subprocess). The
# subprocess path failed on Nova with "CUDA unknown error ... Setting
# the available devices to be zero" and then retry-stormed. Keep the
# engine in the same process; spawn for any residual worker.
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
echo "[vllm-inproc] model=$VLLM_MODEL max_model_len=$VLLM_MAX_MODEL_LEN pixels=[$VLLM_MIN_PIXELS..$VLLM_MAX_PIXELS] gpu_mem=$VLLM_GPU_MEMORY_UTIL v1mp=$VLLM_ENABLE_V1_MULTIPROCESSING"

# ---- ensure Bugwood image cache is populated -------------------------------
# Phase 0R's regional-observation runner tries every image_id per (crop,
# disease, state) and takes the first cache hit (regional_observation.py:174).
# Pre-download EVERY CSV row (deduped by Image Number) so no image_id can
# dead-end. ensure_state_image_cache.py is idempotent (file-exists short-circuit
# at both candidate-selection and HTTP-fetch levels), so re-runs are cheap.
export PATHOME_IMAGE_CACHE_DIR="${PATHOME_IMAGE_CACHE_DIR:-$PATHOME_REPO/.bugwood_cache}"
mkdir -p "$PATHOME_IMAGE_CACHE_DIR"
CACHE_WORKERS="${PATHOME_CACHE_WORKERS:-8}"
echo "[cache] populating $PATHOME_IMAGE_CACHE_DIR from $CSV (all-rows, workers=$CACHE_WORKERS)"
python scripts/ensure_state_image_cache.py \
    --csv "$CSV" \
    --cache-dir "$PATHOME_IMAGE_CACHE_DIR" \
    --all-rows \
    --workers "$CACHE_WORKERS" \
  || { echo "[cache] FAILED — aborting before swarm"; exit 2; }
echo "[cache] populated: $(find "$PATHOME_IMAGE_CACHE_DIR" -maxdepth 1 -type f | wc -l) files"

# ---- enable trace writer so we can diagnose if deltas end up empty ---------
# When PATHOME_TRACE_DIR is set, plantswarm/delta_pipeline appends one JSONL
# record per stochastic trace under <dir>/phase0r_traces.jsonl. Cheap; keeps
# the post-mortem possible without a re-run.
export PATHOME_TRACE_DIR="${PATHOME_TRACE_DIR:-$PATHOME_REPO/artifacts/phase0r_traces}"
mkdir -p "$PATHOME_TRACE_DIR"
echo "[trace] PATHOME_TRACE_DIR=$PATHOME_TRACE_DIR"

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
