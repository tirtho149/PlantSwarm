#!/bin/bash
# ============================================================================
# scripts/sh_push_to_github.sh             LOCAL or NOVA → GitHub
# ============================================================================
# HARD push: stage EVERYTHING in the working tree (all files, every
# subdirectory, every new/modified/deleted path) and push.
#
# By default this does `git add -A`, which captures all tracked
# changes + all untracked files / new subdirectories, but still
# respects .gitignore. To also push files that .gitignore would
# normally exclude (e.g. data/, .bugwood_cache/, large checkpoints),
# set PATHOME_INCLUDE_IGNORED=1.
#
# Knobs
#   COMMIT_MSG               commit message (default: timestamped "hard push")
#   GIT_REMOTE               default origin
#   GIT_BRANCH               default main
#   PATHOME_INCLUDE_IGNORED  1 = also force-add .gitignore'd files
#                              under the repo root (uses `git add -A -f`).
#                              WARNING: this can stage caches, venvs,
#                              and other large directories. Use with care.
#   PATHOME_FORCE_PUSH       1 = use `git push --force-with-lease` instead
#                              of a plain push (only needed when local +
#                              remote have diverged). Refuses force-push
#                              if no upstream is set.
#   PATHOME_DRY_RUN          1 = print plan without staging / pushing
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

DEFAULT_MSG="hard push ($(date -u +%Y-%m-%dT%H:%MZ))"
COMMIT_MSG="${COMMIT_MSG:-$DEFAULT_MSG}"

# Build the `git add` invocation.
if [ "${PATHOME_INCLUDE_IGNORED:-0}" = "1" ]; then
  ADD_CMD=(git add -A -f .)
  ADD_MODE="ALL files (including .gitignore'd)"
else
  ADD_CMD=(git add -A .)
  ADD_MODE="ALL tracked + untracked (respects .gitignore)"
fi

# Build the `git push` invocation.
if [ "${PATHOME_FORCE_PUSH:-0}" = "1" ]; then
  PUSH_CMD=(git push --force-with-lease "$GIT_REMOTE" "$GIT_BRANCH")
  PUSH_MODE="force-with-lease"
else
  PUSH_CMD=(git push "$GIT_REMOTE" "$GIT_BRANCH")
  PUSH_MODE="plain (no force)"
fi

echo "================================================================="
echo " HARD push to GitHub"
echo "================================================================="
echo "  REPO_ROOT       : $REPO_ROOT"
echo "  GIT_REMOTE      : $GIT_REMOTE"
echo "  GIT_BRANCH      : $GIT_BRANCH"
echo "  COMMIT_MSG      : $COMMIT_MSG"
echo "  stage mode      : $ADD_MODE"
echo "  push mode       : $PUSH_MODE"
echo

# Show what'll be staged BEFORE staging, so you can ctrl-C if something
# looks wrong (e.g. an accidental venv or 50GB image cache).
echo "[1/4] preview: status before staging (top 30 paths)"
git status --short | head -n 30
total_untracked=$(git ls-files --others --exclude-standard | wc -l | tr -d ' ')
echo "  ... $(git status --short | wc -l | tr -d ' ') changed paths total"
echo "  ... $total_untracked untracked path(s) (will be added)"
if [ "${PATHOME_INCLUDE_IGNORED:-0}" = "1" ]; then
  total_ignored=$(git ls-files --others --ignored --exclude-standard | wc -l | tr -d ' ')
  echo "  ... $total_ignored ignored path(s) (will ALSO be force-added)"
fi
echo

if [ "${PATHOME_DRY_RUN:-0}" = "1" ]; then
  echo "PATHOME_DRY_RUN=1 — would run:"
  echo "  ${ADD_CMD[*]}"
  echo "  git commit -m \"$COMMIT_MSG\""
  echo "  ${PUSH_CMD[*]}"
  exit 0
fi

echo "[2/4] ${ADD_CMD[*]}"
"${ADD_CMD[@]}"

echo
echo "[3/4] git commit"
if git diff --cached --quiet; then
  echo "  nothing to commit (working tree clean) — pushing whatever's local"
else
  git commit -m "$COMMIT_MSG"
fi

echo
echo "[4/4] ${PUSH_CMD[*]}"
"${PUSH_CMD[@]}"

echo
echo "HARD push done. HEAD: $(git rev-parse --short HEAD)"
echo "remote $GIT_REMOTE/$GIT_BRANCH: $(git rev-parse --short "$GIT_REMOTE/$GIT_BRANCH" 2>/dev/null || echo '?')"
