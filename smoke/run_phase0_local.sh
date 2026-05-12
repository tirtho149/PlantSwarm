#!/bin/bash
# ============================================================================
# smoke/run_phase0_local.sh
# ============================================================================
# Two-crop CANONICAL-only Phase 0 (no regional deltas).
#
# Use this when you only have access to Claude (`claude auth login`) and not
# a vLLM endpoint. The canonical KB is built locally; Phase 0R (Qwen-swarm
# regional deltas) is skipped and can be run later on a GPU host.
#
# Prerequisites:
#   - `claude` CLI authed (`claude auth login`)
#   - ANTHROPIC_API_KEY in env or .env at repo root (faster, optional)
#
# For the full smoke (canonical + regional), use smoke/run_phase0_full.sh.
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
  echo "WARNING: ANTHROPIC_API_KEY not set and no .env at repo root."
  echo "         Phase 0 will use the claude -p CLI fallback (~5x slower)."
fi

if [ ! -f "$USABLE_CSV" ]; then
  echo "Filtering smoke CSV..."
  python scripts/filter_bugwood_csv.py \
    --input  "$RAW_CSV" \
    --output "$USABLE_CSV" \
    --threshold "${SMOKE_THRESHOLD:-15}" \
    --report smoke/bugwood_classes_smoke.tsv
fi

echo "================================================================="
echo "  Smoke Phase 0 — canonical KB only (2 crops, LOCAL)"
echo "  Phase 0R regional deltas are NOT run; see smoke/run_phase0_full.sh"
echo "================================================================="
python -m pathome_kb \
  --csv     "$USABLE_CSV" \
  --out     "$OUT" \
  --quick \
  --only-crops "${SMOKE_CROPS:-Soybean,Tomato}"

echo
echo "================================================================="
echo "  Phase 0 (canonical) complete."
echo "  To add Phase 0R (regional deltas), run on a machine with a vLLM"
echo "  endpoint serving Qwen2.5-VL-7B:"
echo
echo "      VLLM_BASE_URL=... python -m pathome_kb --regional-only \\"
echo "                            --csv $USABLE_CSV --out $OUT \\"
echo "                            --only-crops ${SMOKE_CROPS:-Soybean,Tomato}"
echo "================================================================="
