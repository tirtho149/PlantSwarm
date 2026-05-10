#!/bin/bash
# ============================================================================
# smoke/run_phase0_local.sh
# ============================================================================
# Smoke-sized Phase 0 — run THIS on your local machine, NOT on Nova.
# Same Local→GitHub→Nova handoff as scripts/run_phase0_local.sh, just
# scoped to the 2-crop smoke subset.
#
# Prerequisites: same as production Phase 0 (claude CLI authed,
# ANTHROPIC_API_KEY set or .env at repo root).
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW_CSV="smoke/BugWood_Diseases_smoke.csv"
USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
OUT="smoke/artifacts/pathome_seed/symptoms_seed.json"

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  echo "Install: curl -fsSL https://claude.ai/install.sh | bash"
  exit 1
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ ! -f .env ]; then
  echo "ERROR: ANTHROPIC_API_KEY not set and no .env at repo root."
  exit 1
fi

# Ensure the smoke usable CSV exists (Setup phase if not).
if [ ! -f "$USABLE_CSV" ]; then
  echo "Filtering smoke CSV..."
  python scripts/filter_bugwood_csv.py \
    --input  "$RAW_CSV" \
    --output "$USABLE_CSV" \
    --threshold "${SMOKE_THRESHOLD:-15}" \
    --report smoke/bugwood_classes_smoke.tsv
fi

echo "================================================================="
echo "  Smoke Phase 0 — pathome_kb (LOCAL machine, 2 crops)"
echo "  Stages: cross-region (SAGE) + per-state image-grounded deltas"
echo "================================================================="
python -m pathome_kb \
  --csv     "$USABLE_CSV" \
  --out     "$OUT" \
  --quick \
  --regional \
  --only-crops "Tomato,Soybean"

echo
echo "================================================================="
echo "  Smoke Phase 0 complete. Push the seeded KB to GitHub:"
echo "================================================================="
echo
echo "  git add -f $OUT \\"
echo "             smoke/BugWood_Diseases_smoke_usable.csv \\"
echo "             artifacts/pathome_kb/Tomato/{discovery_results,final_registry,regional_registries}.json \\"
echo "             artifacts/pathome_kb/Soybean/{discovery_results,final_registry,regional_registries}.json"
echo "  git commit -m 'smoke: phase 0 seed'"
echo "  git push origin main"
echo
echo "Then on Nova:"
echo "  ssh tirtho@hpc-login.iastate.edu"
echo "  cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main"
echo "  sbatch smoke/submit_smoke.sh"
