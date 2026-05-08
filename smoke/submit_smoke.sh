#!/bin/bash
#SBATCH --job-name=pathome_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_smoke-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_smoke-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Smoke-test the Pathome pipeline end-to-end on Nova in a single A100 job.
# Wraps smoke/run_smoke.sh — same skip/from-phase env vars apply.
#
# Walltime budget: 4 h. Typical actual runtime on a single A100 is 60-90 min
# (Phase 2 ~10-20 min for 70-100 traces; Phase 4 ~30-40 min × 2 checkpoints).
# ============================================================================

set -e
echo "================================"
echo "Pathome smoke run"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
chmod +x smoke/run_smoke.sh
bash smoke/run_smoke.sh

echo
echo "Smoke run complete: $(date)"
echo "Compare table: smoke/results/compare/comparison.md"
