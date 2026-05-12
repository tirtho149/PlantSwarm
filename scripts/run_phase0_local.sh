#!/bin/bash
# ============================================================================
# scripts/run_phase0_local.sh
# ============================================================================
# Production Phase 0 (canonical KB only) — run on your local machine.
#
# This builds canonical entries for every (crop, disease) in the filtered
# Bugwood CSV via the SAGE-ported pathome_kb pipeline (claude -p WebSearch
# + extraction + reconciliation). It does NOT run Phase 0R (regional
# deltas via the Qwen swarm) — that needs a GPU host with vLLM.
#
# Prerequisites
#   - `claude` CLI authed (`claude auth login`)
#   - ANTHROPIC_API_KEY in env or .env at repo root (faster, optional)
#   - Filtered CSV at BugWood_Diseases_usable.csv
#       (generate with `python scripts/filter_bugwood_csv.py --threshold 10`)
#
# Output
#   artifacts/pathome_seed/symptoms_seed.json    (canonical-only seed)
#   artifacts/pathome_kb/<Crop>/...              (per-crop provenance)
#
# After this finishes, run Phase 0R against the same final_registry.json
# files on a GPU host serving Qwen2.5-VL-7B with vLLM:
#
#   VLLM_BASE_URL=http://localhost:8000/v1 \
#     python -m pathome_kb --regional-only \
#       --csv BugWood_Diseases_usable.csv \
#       --out artifacts/pathome_seed/symptoms_seed.json
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
OUT="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"

if [ ! -f "$CSV" ]; then
  echo "ERROR: filtered CSV not found at $CSV"
  echo "Generate it first:"
  echo "  python scripts/filter_bugwood_csv.py --threshold 10"
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  echo "Install: curl -fsSL https://claude.ai/install.sh | bash"
  echo "Then:    claude auth login"
  exit 1
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ ! -f .env ]; then
  echo "WARNING: ANTHROPIC_API_KEY not set and no .env at repo root."
  echo "         Phase 0 will use the claude -p CLI fallback (~5x slower)."
fi

ARGS=("--csv" "$CSV" "--out" "$OUT")
if [ "${PATHOME_SEED_QUICK:-0}" = "1" ]; then ARGS+=("--quick"); fi
if [ -n "${PATHOME_SEED_LIMIT:-}" ];      then ARGS+=("--limit-crops" "$PATHOME_SEED_LIMIT"); fi
if [ -n "${PATHOME_SEED_ONLY_CROPS:-}" ]; then ARGS+=("--only-crops"  "$PATHOME_SEED_ONLY_CROPS"); fi
if [ -n "${PATHOME_SEED_RESUME:-}" ];     then ARGS+=("--resume-from" "$PATHOME_SEED_RESUME"); fi
if [ "${PATHOME_SEED_NO_CACHE:-0}" = "1" ]; then ARGS+=("--no-cache"); fi

echo "================================================================="
echo "  Phase 0 (canonical KB only) — pathome_kb (LOCAL machine)"
echo "================================================================="
echo "  csv:  $CSV"
echo "  out:  $OUT"
echo "  args: ${ARGS[*]}"
echo "================================================================="
python -m pathome_kb "${ARGS[@]}"

echo
echo "================================================================="
echo "  Phase 0 complete. Next: Phase 0R on a GPU host with vLLM."
echo "================================================================="
echo
echo "  # push canonical artefacts"
echo "  git add -f $OUT artifacts/pathome_kb/"
echo "  git commit -m 'Phase 0: canonical KB ($(date -u +%Y-%m-%dT%H:%MZ))'"
echo "  git push origin main"
echo
echo "  # then on the GPU host, after starting vLLM:"
echo "  VLLM_BASE_URL=http://localhost:8000/v1 \\"
echo "    python -m pathome_kb --regional-only \\"
echo "      --csv $CSV --out $OUT"
