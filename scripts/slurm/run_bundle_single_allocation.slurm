#!/bin/bash
# One GPU allocation: start vLLM on-node, then full experiment bundle (same node).
# Edit paths, mail, and VLLM_MODEL to match model.backbone in CONFIG_PATH.
#
# Submit:
#   sbatch scripts/slurm/run_bundle_single_allocation.slurm
#
# Override at submit time:
#   sbatch --export=ALL,SUBSET=100,VLLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct scripts/slurm/run_bundle_single_allocation.slurm

#SBATCH --job-name=ps-1alloc
#SBATCH --output=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm/logs/%x-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=nova
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm

set -euo pipefail

WORKDIR="${WORKDIR:-/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm}"
ENV_PATH="${ENV_PATH:-/work/mech-ai-scratch/tirtho/gpu_env}"
export CONFIG_PATH="${CONFIG_PATH:-configs/cyag_directory.yaml}"
export SUBSET="${SUBSET:-}"
export ORCHESTRATOR="${ORCHESTRATOR:-autogen_swarm}"
export ROUTING_SUBSET="${ROUTING_SUBSET:-${SUBSET:-500}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/single_alloc_${RUN_ID}}"
# Must match model.backbone in CONFIG_PATH (HF id passed to vllm serve)
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

cd "${WORKDIR}"

if [[ -f "${ENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
  export PYTHON_BIN="${ENV_PATH}/bin/python"
elif [[ -x "${ENV_PATH}/bin/python" ]]; then
  export PYTHON_BIN="${ENV_PATH}/bin/python"
else
  export PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export CYAG_IMAGE_ROOT="${CYAG_IMAGE_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"

echo "SINGLE ALLOCATION  VLLM_MODEL=${VLLM_MODEL}  CONFIG_PATH=${CONFIG_PATH}  RESULTS_DIR=${RESULTS_DIR}"
bash scripts/run_single_allocation.sh
