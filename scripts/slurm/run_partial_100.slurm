#!/bin/bash
# 100-image partial run (same stack as run_all_tests.slurm) + LaTeX sync.
#SBATCH --job-name=ps-100
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
export SUBSET="${SUBSET:-100}"
export ORCHESTRATOR="${ORCHESTRATOR:-autogen_swarm}"
export CONFIG_PATH="${CONFIG_PATH:-configs/cyag_directory.yaml}"
WORKDIR="${WORKDIR:-/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm}"
ENV_PATH="${ENV_PATH:-/work/mech-ai-scratch/tirtho/gpu_env}"
export CYAG_IMAGE_ROOT="${CYAG_IMAGE_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/partial100_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${WORKDIR}/logs" "${RESULTS_DIR}"
cd "${WORKDIR}"
export CYAG_IMAGE_ROOT

if [[ -f "${ENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
  PYTHON_BIN="${ENV_PATH}/bin/python"
elif [[ -x "${ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${ENV_PATH}/bin/python"
fi
export PYTHON_BIN

echo "PARTIAL-100  SUBSET=${SUBSET}  ORCHESTRATOR=${ORCHESTRATOR}  RESULTS_DIR=${RESULTS_DIR}"
export ROUTING_SUBSET="${ROUTING_SUBSET:-${SUBSET}}"
bash "${WORKDIR}/scripts/run_experiment_bundle.sh"
