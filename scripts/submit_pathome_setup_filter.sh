#!/bin/bash
#SBATCH --job-name=pathome_setup_filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=nova
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_setup_filter-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_setup_filter-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Setup (one-time): filter the Bugwood IPMNet CSV into the usable subset
# ============================================================================
# Reads:   BugWood_Diseases.csv         (~19,749 rows, raw IPMNet export)
# Writes:  BugWood_Diseases_usable.csv  (~11,513 rows, 484 classes)
#          bugwood_classes_report.tsv   (per-class candidate counts)
#
# CPU-only, ~30 s. Downstream phases all read from the filtered CSV.
# Re-run after pulling a new IPMNet export, or when changing the threshold.
#
# Override at submit time:
#   PATHOME_THRESHOLD=15 sbatch scripts/submit_pathome_setup_filter.sh
#   (15 → 263 classes; 10 → 484; 5 → 982. See bugwood_classes_report.tsv)
# ============================================================================

set -e
echo "================================"
echo "Setup: filter Bugwood CSV"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate
mkdir -p logs

THRESHOLD="${PATHOME_THRESHOLD:-10}"
INPUT="${PATHOME_RAW_CSV:-BugWood_Diseases.csv}"
OUTPUT="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
REPORT="${PATHOME_CLASS_REPORT:-bugwood_classes_report.tsv}"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input CSV not found at $INPUT"
  echo "Pull from IPMNet (https://www.bugwood.org/ipmnet) and place at the repo root,"
  echo "or set PATHOME_RAW_CSV to its path."
  exit 1
fi

echo "input:     $INPUT"
echo "threshold: $THRESHOLD rows/class"
echo "output:    $OUTPUT"
echo "report:    $REPORT"
echo
python scripts/filter_bugwood_csv.py \
  --input     "$INPUT" \
  --output    "$OUTPUT" \
  --threshold "$THRESHOLD" \
  --report    "$REPORT"

echo
echo "Setup complete: $(date)"
echo "next: sbatch scripts/submit_pathome_phase0_seed.sh"
