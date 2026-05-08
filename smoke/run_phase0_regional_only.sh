#!/bin/bash
# ============================================================================
# smoke/run_phase0_regional_only.sh
# ============================================================================
# Re-run ONLY the per-state regional extraction stage on the cached
# cross-region artefacts (skip discovery + extraction + reconciliation).
#
# Use this when:
#   - You've already run smoke/run_phase0_local.sh (or the equivalent
#     cross-region pipeline) and have raw_extractions.json + final_registry.json
#     on disk under artifacts/pathome_kb/{Tomato,Soybean}/.
#   - You want to iterate on regional_extraction.py / its prompts /
#     the symptoms_adapter without redoing the expensive discovery and
#     per-source web extraction stages.
#
# Output (overwritten):
#   artifacts/pathome_kb/{Tomato,Soybean}/regional_registries.json
#   smoke/artifacts/pathome_seed/symptoms_seed.json   (re-merged)
#
# Walltime: ~3-5 min for the 2-crop smoke (~38 (profile, state) calls).
#
# Auth requirements: same as Phase 0 — `claude` CLI on PATH and
# authenticated. ANTHROPIC_API_KEY is honoured if set; otherwise the
# pipeline falls back to claude -p subprocess (slower but no key required).
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
OUT="smoke/artifacts/pathome_seed/symptoms_seed.json"

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  exit 1
fi

if [ ! -f "$USABLE_CSV" ]; then
  echo "ERROR: filtered smoke CSV not found at $USABLE_CSV"
  echo "Run smoke/run_phase0_local.sh first (it produces both the CSV and the cross-region artefacts)."
  exit 1
fi

# Verify the cross-region artefacts the regional pass needs
for crop in Tomato Soybean; do
  if [ ! -f "artifacts/pathome_kb/$crop/raw_extractions.json" ]; then
    echo "ERROR: artifacts/pathome_kb/$crop/raw_extractions.json missing"
    echo "The regional pass mines that file. Run smoke/run_phase0_local.sh first to produce it."
    exit 1
  fi
done

echo "================================================================="
echo "  Smoke Phase 0 — REGIONAL-ONLY pass (Tomato + Soybean)"
echo "================================================================="
python3 -m pathome_kb \
  --csv     "$USABLE_CSV" \
  --out     "$OUT" \
  --regional-only \
  --quick \
  --only-crops "Tomato,Soybean"

echo
echo "================================================================="
echo "  Done. Push the updated seed:"
echo "================================================================="
echo
echo "  git add -f $OUT \\"
echo "             artifacts/pathome_kb/Tomato/regional_registries.json \\"
echo "             artifacts/pathome_kb/Soybean/regional_registries.json"
echo "  git commit -m 'smoke: refresh regional seed'"
echo "  git push origin main"
