#!/bin/bash
# ============================================================================
# smoke/run_phase0_full.sh
# ============================================================================
# Single-command, perfect-KB regenerate for any 2 crops in the smoke CSV.
# Default crops: Soybean + Corn. Override with SMOKE_CROPS="Crop1,Crop2".
#
# What runs (every stage end-to-end, default = MAXIMUM COVERAGE):
#
#   1. Filter the smoke CSV               → BugWood_Diseases_smoke_usable.csv
#   2. State-aware image cache top-up     → one image per (crop, disease, state)
#   3. Cross-region SAGE pipeline:
#        discovery (claude -p WebSearch)  → all candidate URLs per disease
#        extraction (claude -p)           → verbatim quotes + treatments
#        reconciliation (claude -p)       → canonical entries with treatments
#        → final_registry.json per crop
#   4. Per-state VLM observation (deltas-only, decision-tree style):
#        claude -p + Read tool looks at each cached Bugwood image
#        + canonical KB from step 3 as context
#        → list of deltas {field, canonical_says, image_shows, image_quote}
#          — only state-specific additions/contradictions, never a
#          parallel re-extraction of canonical fields
#        → embedded into final_registry.json (single source of truth)
#   5. Adapter merge → smoke/artifacts/pathome_seed/symptoms_seed.json
#
# Output schema:
#   SymptomProfile {
#     canonical: CanonicalDisease           # one block per disease (text)
#     regional_observations: {                # per-state image-grounded deltas
#       state: { image_ids, deltas:[{field, canonical_says,
#                                    image_shows, image_quote}] }
#     }
#   }
#
# Walltime / cost (full coverage):
#   ~45–90 min wall, ~$5–15 in claude -p OAuth quota
#
# Knobs:
#   FULL_QUICK=1         caps sources/states for fast iteration (~15-25 min, ~$1-3)
#   FULL_KEEP_CACHE=1    skip clearing the cached final_registry.json — reuse
#                        existing cross-region run (treatments may be missing
#                        if the cache predates the prompt update)
#   FULL_SKIP_SETUP=1    CSV already filtered
#   FULL_SKIP_CACHE=1    image cache already topped up
#   FULL_SKIP_KB=1       skip the python -m pathome_kb call (no-op smoke)
#
# Auth requirements:
#   - claude CLI on PATH and authenticated (`claude auth login`)
#   - ANTHROPIC_API_KEY optional (auto-falls-back to claude -p)
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW_CSV="smoke/BugWood_Diseases_smoke.csv"
USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
OUT="smoke/artifacts/pathome_seed/symptoms_seed.json"
CACHE_DIR="smoke/.bugwood_cache"

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  echo "Install: curl -fsSL https://claude.ai/install.sh | bash"
  echo "Auth:    claude auth login"
  exit 1
fi

step() {
  echo
  echo "================================================================="
  echo "  $1"
  echo "================================================================="
}

QUICK_ARG=()
if [ "${FULL_QUICK:-0}" = "1" ]; then
  QUICK_ARG=(--quick)
  echo "[mode] FULL_QUICK=1 — capping sources/states for fast iteration"
else
  echo "[mode] FULL coverage — every source URL, every state, every visual field"
fi

# ----------------------------------------------------------------------------
# 1. Filter the smoke CSV
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_SETUP:-0}" != "1" ]; then
  step "1. Setup — filter smoke CSV"
  python3 scripts/filter_bugwood_csv.py \
    --input "$RAW_CSV" \
    --output "$USABLE_CSV" \
    --threshold "${SMOKE_THRESHOLD:-15}" \
    --report smoke/bugwood_classes_smoke.tsv
fi

# ----------------------------------------------------------------------------
# 2. State-aware image cache top-up
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_CACHE:-0}" != "1" ]; then
  step "2. State-aware image cache top-up"
  python3 scripts/ensure_state_image_cache.py \
    --csv "$USABLE_CSV" \
    --cache-dir "$CACHE_DIR"
fi

# ----------------------------------------------------------------------------
# 3. Optionally drop stale per-crop registries so the new prompts run
#    (treatments was added to the extraction/reconciliation prompts; cached
#    registries from before that change won't have treatments populated).
# ----------------------------------------------------------------------------
if [ "${FULL_KEEP_CACHE:-0}" != "1" ]; then
  step "3a. Clearing stale registries to re-run with treatments prompt"
  for crop in $(echo "${SMOKE_CROPS:-Soybean,Corn}" | tr ',' ' '); do
    rm -f "artifacts/pathome_kb/$crop/raw_extractions.json" \
          "artifacts/pathome_kb/$crop/final_registry.json" \
          "artifacts/pathome_kb/$crop/registry.md" \
          "artifacts/pathome_kb/$crop/internet.xlsx"
  done
  echo "  (kept discovery_results.json — re-using cached URLs)"
fi

# ----------------------------------------------------------------------------
# 4 + 5. Cross-region SAGE + per-state VLM observation + merge
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_KB:-0}" != "1" ]; then
  step "3b. Cross-region SAGE pipeline + per-state VLM observation"

  # Auto-detect: if every target crop already has discovery_results.json on
  # disk, we can resume from the extraction stage and skip the (slow + costly)
  # WebSearch pass. Otherwise run discovery fresh.
  RESUME_ARG=()
  ALL_HAVE_DISCOVERY=1
  for crop in $(echo "${SMOKE_CROPS:-Soybean,Corn}" | tr ',' ' '); do
    if [ ! -f "artifacts/pathome_kb/$crop/discovery_results.json" ]; then
      ALL_HAVE_DISCOVERY=0
      break
    fi
  done
  if [ "$ALL_HAVE_DISCOVERY" = "1" ]; then
    echo "  [resume] discovery_results.json present for every crop — resuming from extraction"
    RESUME_ARG=(--resume-from extraction)
  else
    echo "  [fresh]  no cached discovery for at least one crop — running full discovery"
  fi

  python3 -m pathome_kb \
    --csv "$USABLE_CSV" \
    --out "$OUT" \
    --regional \
    --only-crops "${SMOKE_CROPS:-Soybean,Corn}" \
    "${RESUME_ARG[@]}" \
    "${QUICK_ARG[@]}"
fi

# ----------------------------------------------------------------------------
# Summary + push instructions
# ----------------------------------------------------------------------------
step "Done"
python3 -c "
import json
s = json.load(open('$OUT'))
profiles = s['profiles']
n_canon = sum(1 for p in profiles if (p.get('canonical') or {}).get('summary'))
n_reg   = sum(1 for p in profiles if p.get('regional_observations'))
n_blocks = sum(len(p.get('regional_observations') or {}) for p in profiles)
n_with_treat = sum(1 for p in profiles if (p.get('canonical') or {}).get('treatments'))
n_text = 0
n_deltas = 0
fields_hit = {}
for p in profiles:
    for f, cits in ((p.get('canonical') or {}).get('sources') or {}).items():
        for c in cits:
            n_text += 1 if c.get('grounding','text') == 'text' else 0
    for state, obs in (p.get('regional_observations') or {}).items():
        for d in (obs.get('deltas') or []):
            n_deltas += 1
            fields_hit[d.get('field','other')] = fields_hit.get(d.get('field','other'), 0) + 1
print(f'profiles total                   : {len(profiles)}')
print(f'profiles w/ canonical summary    : {n_canon}')
print(f'profiles w/ canonical treatments : {n_with_treat}')
print(f'profiles w/ regional observations: {n_reg}')
print(f'total per-state blocks           : {n_blocks}')
print(f'total state-specific deltas      : {n_deltas}')
print(f'text-grounded citations (canonical): {n_text}')
if fields_hit:
    print('deltas by canonical field:')
    for k, v in sorted(fields_hit.items(), key=lambda x: -x[1]):
        print(f'  {k:24s} {v}')
"

echo
echo "Push the seed to GitHub:"
echo "  git add -f $OUT \\"
echo "             $USABLE_CSV \\"
for crop in $(echo "${SMOKE_CROPS:-Soybean,Corn}" | tr ',' ' '); do
  echo "             artifacts/pathome_kb/$crop/{discovery_results,final_registry}.json \\"
done
echo "  git commit -m 'smoke: regenerate state-aware KB'"
echo "  git push origin main"
echo
echo "Then on Nova:"
echo "  ssh tirtho@hpc-login.iastate.edu"
echo "  cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main"
echo "  sbatch smoke/submit_smoke.sh"
