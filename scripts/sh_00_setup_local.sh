#!/bin/bash
# ============================================================================
# scripts/sh_00_setup_local.sh             STEP 0 — LOCAL
# ============================================================================
# Produce BugWood_Diseases_usable.csv from the raw IPMNet export, with
# (optionally) the Claude two-layer label judge applied to drop INVALID
# / NON_CROP crops, INCORRECT diseases, and canonicalise MISSPELLED crop
# names in place.
#
# What this produces
#   BugWood_Diseases_usable.csv               filtered (+ judged) CSV
#                                             5 derived cols: NormCrop,
#                                             NormDisease, StateLat,
#                                             StateLon, AezCode
#   artifacts/bugwood_judgement.json          sidecar judgement report
#                                             (resume-key; re-running is
#                                             cheap because already-judged
#                                             crops are skipped)
#   artifacts/bugwood_judgement_progress.txt  streaming progress log
#
# Pre-reqs
#   - `claude` CLI authenticated when running with the judge.
#   - BugWood_Diseases.csv present at the repo root.
#
# Knobs (env vars)
#   THRESHOLD             min rows per (crop, disease) class (default 10)
#   JUDGE_LABELS          1 = run the Claude judge after threshold filter
#                         0 = threshold-only build (default 1)
#   DROP_QUESTIONABLE     1 = also drop QUESTIONABLE diseases (default 0)
#   PER_CLASS             optional cap on rows per class (default 0 = none)
#   PATHOME_RAW_CSV       default BugWood_Diseases.csv
#   PATHOME_USABLE_CSV    default BugWood_Diseases_usable.csv
#   PATHOME_SKIP_PUSH     set 1 to commit but not push
#   GIT_REMOTE            default origin
#   GIT_BRANCH            default main
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

THRESHOLD="${THRESHOLD:-10}"
JUDGE_LABELS="${JUDGE_LABELS:-1}"
DROP_QUESTIONABLE="${DROP_QUESTIONABLE:-0}"
PER_CLASS="${PER_CLASS:-0}"
PATHOME_RAW_CSV="${PATHOME_RAW_CSV:-BugWood_Diseases.csv}"
PATHOME_USABLE_CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

# Resolve Python interpreter (macOS: usually only python3 on PATH).
PY="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
if [ -z "$PY" ]; then
  echo "ERROR: no python / python3 on PATH. Install Python 3 or set PYTHON_BIN."
  exit 2
fi

echo "================================================================="
echo " STEP 0 — Filter + (optional) Claude label judge (LOCAL)"
echo "================================================================="
echo "  PATHOME_RAW_CSV     : $PATHOME_RAW_CSV"
echo "  PATHOME_USABLE_CSV  : $PATHOME_USABLE_CSV"
echo "  THRESHOLD           : $THRESHOLD"
echo "  JUDGE_LABELS        : $JUDGE_LABELS"
echo "  DROP_QUESTIONABLE   : $DROP_QUESTIONABLE"
echo "  PER_CLASS           : $PER_CLASS"
echo

if [ ! -f "$PATHOME_RAW_CSV" ]; then
  echo "ERROR: raw CSV not found at $PATHOME_RAW_CSV"
  exit 2
fi

# Judge needs the Claude CLI authenticated.
if [ "$JUDGE_LABELS" = "1" ]; then
  if ! command -v claude >/dev/null 2>&1; then
    echo "ERROR: 'claude' CLI not found on PATH (needed for JUDGE_LABELS=1)."
    echo "       Install + run 'claude' once interactively to authenticate,"
    echo "       or re-run with JUDGE_LABELS=0 for threshold-only filtering."
    exit 2
  fi
fi

mkdir -p artifacts

# Build the filter command.
cmd=("$PY" scripts/filter_bugwood_csv.py
     --input  "$PATHOME_RAW_CSV"
     --output "$PATHOME_USABLE_CSV"
     --threshold "$THRESHOLD")

if [ "$PER_CLASS" != "0" ]; then
  cmd+=(--per-class "$PER_CLASS")
fi

if [ "$JUDGE_LABELS" = "1" ]; then
  cmd+=(--judge
        --judge-report   artifacts/bugwood_judgement.json
        --judge-progress artifacts/bugwood_judgement_progress.txt)
  if [ "$DROP_QUESTIONABLE" = "1" ]; then
    cmd+=(--judge-drop-questionable)
  fi
fi

echo "[1/3] Running: ${cmd[*]}"
"${cmd[@]}"

echo
echo "[2/3] git add usable CSV + judgement report"
git add -f "$PATHOME_USABLE_CSV" \
           artifacts/bugwood_judgement.json 2>/dev/null || true

if git diff --cached --quiet; then
  echo "  no changes to commit"
else
  msg="Step 0 setup: filter"
  if [ "$JUDGE_LABELS" = "1" ]; then
    msg="$msg + Claude label judge"
  fi
  git commit -m "$msg ($(date -u +%Y-%m-%dT%H:%MZ))"
fi

echo
echo "[3/3] git push to $GIT_REMOTE $GIT_BRANCH"
if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
  echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
else
  git push "$GIT_REMOTE" "$GIT_BRANCH" || true
fi

echo
echo "STEP 0 done."
echo "  Usable CSV : $PATHOME_USABLE_CSV"
if [ "$JUDGE_LABELS" = "1" ]; then
  echo "  Judgement  : artifacts/bugwood_judgement.json"
fi
echo "  Next: scripts/sh_01_phase0_local.sh"
