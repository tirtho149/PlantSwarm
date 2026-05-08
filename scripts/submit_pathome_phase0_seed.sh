#!/bin/bash
#SBATCH --job-name=pathome_phase0_seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=nova
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase0_seed-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase0_seed-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 0: Build the seed PathomeDB knowledge base (provenance-tracked)
# ============================================================================
# Runs the SAGE-ported pathome_kb internet track:
#   1. discovery       — claude -p WebSearch per disease, parallel
#   2. extraction      — fetch each source URL, claude -p extracts disease
#                        records with verbatim quotes from page text
#   3. reconciliation  — merge per-source records into a canonical registry,
#                        every field carrying {value, url, quote}
# Then merges all per-crop registries into a single SymptomLibrary seed
# JSON that Phase 1 (build_pathome.py) consumes.
#
# Compute: CPU-only. No GPU. Outbound HTTPS required for both the Anthropic
# API (extraction + reconciliation) and the per-source page fetches.
#
# Authentication:
#   - claude -p needs the Claude Code CLI on PATH and an auth'd session:
#       curl -fsSL https://claude.ai/install.sh | bash
#       claude auth login
#   - The direct Anthropic SDK calls need ANTHROPIC_API_KEY in environment
#     OR a .env file at the repo root containing ANTHROPIC_API_KEY=...
#
# Output:
#   artifacts/pathome_kb/<Crop>/{discovery_results.json,
#       raw_extractions.json, final_registry.json, registry.md, internet.xlsx}
#   artifacts/pathome_seed/symptoms_seed.json    (merged seed for Phase 1)
#
# Cost / time:
#   Full run over 197 crops × (~5-15 sources/crop) is ~$50-150 in API spend
#   and 12-20 h wall time on the default 8-worker config. Use PATHOME_SEED_QUICK=1
#   for a smoke test (3 sources/crop, shorter timeouts, ~30 min).
#
# Knobs (env vars):
#   PATHOME_SEED_QUICK     "1" → --quick (smoke test mode)
#   PATHOME_SEED_LIMIT     N   → run only first N crops alphabetically
#   PATHOME_SEED_ONLY_CROPS    comma-separated allowlist
#   PATHOME_SEED_RESUME    discovery|extraction|reconciliation
#   PATHOME_SEED_NO_CACHE  "1" → re-run crops even if final_registry.json exists
# ============================================================================

set -e
echo "================================"
echo "Phase 0: PathomeDB KB build"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate
mkdir -p logs artifacts/pathome_seed artifacts/pathome_kb

# Required: claude CLI on PATH + auth'd
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found on PATH"
  echo "Install with: curl -fsSL https://claude.ai/install.sh | bash"
  exit 1
fi
claude --version || { echo "ERROR: claude not callable"; exit 1; }

# Required: ANTHROPIC_API_KEY for direct SDK calls (extraction, reconciliation)
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ ! -f .env ]; then
  echo "ERROR: ANTHROPIC_API_KEY not in environment and no .env file at repo root."
  echo "Set it: export ANTHROPIC_API_KEY=sk-ant-... (in ~/.bashrc on Nova)"
  exit 1
fi

CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
OUT="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"

if [ ! -f "$CSV" ]; then
  echo "ERROR: filtered CSV not found at $CSV"
  echo "Run setup first: sbatch scripts/submit_pathome_setup_filter.sh"
  exit 1
fi

ARGS=("--csv" "$CSV" "--out" "$OUT")
if [ "${PATHOME_SEED_QUICK:-0}" = "1" ]; then ARGS+=("--quick"); fi
if [ -n "${PATHOME_SEED_LIMIT:-}" ];      then ARGS+=("--limit-crops" "$PATHOME_SEED_LIMIT"); fi
if [ -n "${PATHOME_SEED_ONLY_CROPS:-}" ]; then ARGS+=("--only-crops"  "$PATHOME_SEED_ONLY_CROPS"); fi
if [ -n "${PATHOME_SEED_RESUME:-}" ];     then ARGS+=("--resume-from" "$PATHOME_SEED_RESUME"); fi
if [ "${PATHOME_SEED_NO_CACHE:-0}" = "1" ]; then ARGS+=("--no-cache"); fi

echo "csv:    $CSV"
echo "out:    $OUT"
echo "args:   ${ARGS[*]}"
echo
python -m pathome_kb "${ARGS[@]}"

echo
echo "Phase 0 complete: $(date)"
echo "Output:  $OUT"
echo "Per-crop artefacts: artifacts/pathome_kb/<Crop>/"
echo "Next:    sbatch scripts/submit_pathome_phase1_build.sh"
