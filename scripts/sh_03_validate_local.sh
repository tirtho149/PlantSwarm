#!/bin/bash
# ============================================================================
# scripts/sh_03_validate_local.sh           STEP 3 — LOCAL
# ============================================================================
# Pull the unverified-deltas KB from GitHub (step 2 output), run the
# Claude+WebSearch validator over every delta that still has
# verification_status="unverified", then push the verified KB back to
# GitHub so step 4 on Nova can fine-tune the CLIP using a clean KB.
#
# This step only needs:
#   - `claude` CLI authenticated (LOCAL has it; Nova does not)
#   - internet access for WebSearch
#
# Wall-clock: roughly 60-90 s per (crop, disease, state) tuple with
# Claude.  Smoke (~10-30 tuples): ~10-30 min.  Production
# (~2000+ tuples): ~1-3 days. Set MAX_TUPLES for a cost cap.
#
# Knobs
#   CROPS          comma-separated crop allowlist
#                   "smoke" = "Soybean,Tomato"; "all" = no filter
#   MAX_TUPLES     cap (0 = no cap)
#   DRY_RUN        set 1 to print plan without calling Claude
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

CROPS="${CROPS:-smoke}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

# Resolve Python interpreter (macOS: usually only python3 on PATH).
PY="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
if [ -z "$PY" ]; then
  echo "ERROR: no python / python3 on PATH. Install Python 3 or set PYTHON_BIN."
  exit 2
fi

echo "================================================================="
echo " STEP 3 — Claude+WebSearch validation (LOCAL)"
echo "================================================================="
echo "  CROPS        : $CROPS"
echo "  MAX_TUPLES   : ${MAX_TUPLES:-0}"
echo "  DRY_RUN      : ${DRY_RUN:-0}"

# Sanity check Claude CLI.
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found on PATH."
  exit 2
fi

# Pull Nova's unverified deltas.
echo
echo "[1/3] git pull unverified-deltas KB from $GIT_REMOTE $GIT_BRANCH"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only

# Validate.
echo
echo "[2/3] $PY scripts/validate_kb.py"
CROPS="$CROPS" MAX_TUPLES="${MAX_TUPLES:-0}" \
${DRY_RUN:+DRY_RUN=$DRY_RUN} \
  "$PY" scripts/validate_kb.py --kb-root artifacts/pathome_kb

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo
  echo "DRY_RUN=1 — no changes to commit."
  exit 0
fi

# Push verified KB.
echo
echo "[3/3] git push verified KB to $GIT_REMOTE $GIT_BRANCH"
git add -f artifacts/pathome_kb/*/final_registry.json
if git diff --cached --quiet; then
  echo "  no changes to commit (no deltas needed verification)"
else
  git commit -m "Phase 0R verification (LOCAL Claude+WebSearch): final KB ready for fine-tuning"
  if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
    echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
  else
    git push "$GIT_REMOTE" "$GIT_BRANCH"
  fi
fi

echo
echo "STEP 3 done."
echo "  Final KB: artifacts/pathome_kb/<Crop>/final_registry.json"
echo "  Next: ssh back to Nova, then run scripts/sh_04_train_encoder_nova.sh"
