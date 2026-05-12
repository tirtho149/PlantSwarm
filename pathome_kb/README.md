# `pathome_kb/` — How the KB looks and how it's built

This module produces the **PathomeDB seed** — a single JSON document with one `SymptomProfile` per `(crop, disease)` class in your filtered Bugwood CSV. The seed is the terminal deliverable of the pipeline (Phase 1+ retired). It is built in two stages: a canonical KB pass (`claude -p` discovery → extraction → reconciliation) and a regional delta pass (`plantswarm/delta_pipeline.py` — the Qwen swarm reading canonical KB as context and emitting deltas).

The schema is a **decision tree**: canonical KB is the trunk, per-state regional observations are branches that *only* emit additions or contradictions vs the canonical text. The regional pass never re-extracts what canonical already owns.

```
SymptomProfile
   ├── canonical          ← cross-region, sourced from extension-service URLs
   │     · summary, diagnostic_features, look_alikes
   │     · treatments
   │     · affected_parts, pathogen_scientific_name, type_of_disease
   │     · sources : { field → [{ value, url, quote, grounding="text" }] }
   │
   └── regional_observations[state]   ← per-state, image-grounded DELTAS only
         · image_ids                  ← Bugwood photographs from this state
         · deltas                     ← list of structured records, each one is
         │     · field            ← which canonical field this delta refines
         │                          (lesion_morphology / severity / affected_organs /
         │                           spread_pattern / diagnostic_features / look_alikes /
         │                           treatments / other)
         │     · canonical_says   ← short quote from canonical, or "(not specified)"
         │     · image_shows      ← what THIS image adds or contradicts
         │     · image_quote      ← one-sentence visual evidence
         │     · image_id         ← bugwood::N (the witness)
         · (no parallel severity / lesion_morphology / spread_pattern fields —
            those slots live exclusively on canonical; regional only has deltas)
```

If the image confirms canonical exactly, `deltas` is empty for that state. The earlier schema duplicated canonical text per state and the one before it re-extracted parallel fields per state — both removed.

---

## What the KB looks like (worked example: `Soybean :: Charcoal Rot`)

Real entry from `artifacts/pathome_kb/Soybean/final_registry.json`. The canonical block is identical for every state; the two regional blocks below describe *different things in different photos* of the same disease.

```jsonc
{
  "disease_name": "Charcoal Rot",
  "pathogen_scientific_name": {
    "value": "Macrophomina phaseolina",
    "url":   "https://extension.umn.edu/soybean-pest-management/charcoal-rot-soybean",
    "quote": "Charcoal rot is caused by the soilborne fungus Macrophomina phaseolina."
  },
  "type_of_disease":  { "value": "Fungal", "url": "...", "quote": "..." },
  "affected_parts":   { "value": ["Foliar","Stem","Root","Seed","Pod","Vascular"], "url": "...", "quote": "..." },
  "visual_symptoms": {
    "summary":             { "value": "Brown-to-dark spots on cotyledons; circular-to-oblong reddish-brown lesions; premature wilt; lower stem and taproot streaked with charcoal-sprinkled microsclerotia.", "url": "...", "quote": "..." },
    "diagnostic_features": { "value": "Microsclerotia in pith and dry pods; cross-sections marbled with charcoal-sprinkled appearance.", "url": "...", "quote": "..." },
    "look_alikes":         { "value": [], "url": "", "quote": "" }
  },
  "treatments": {
    "value": ["drought-stress mitigation", "reduced tillage", "rotation to non-hosts",
              "reduced seeding rates", "cultivar selection", "fungicides not recommended"],
    "url": "...", "quote": "..."
  },

  "regional_observations": {

    "Alabama": {
      "image_ids": ["bugwood::1234567"],
      "deltas": [
        {
          "field":          "spread_pattern",
          "canonical_says": "(not specified)",
          "image_shows":    "Field-scale distribution: entire rows along the field edge collapsed and brown while adjacent rows remain green — patchy edge-of-field outbreak.",
          "image_quote":    "Foreground rows are uniformly tan-brown and prematurely senesced while rows on the right side of the field remain green and intact."
        },
        {
          "field":          "severity",
          "canonical_says": "infected plants may die prematurely and are often wilted and stunted",
          "image_shows":    "Severity at this Alabama site is advanced and stand-level — multiple contiguous rows show whole-plant death rather than scattered individuals.",
          "image_quote":    "Several full rows of soybean plants are brown, dried, and collapsed top-to-bottom with little remaining green tissue."
        },
        {
          "field":          "affected_organs",
          "canonical_says": "Foliar; Stem; Root; Seed; Pod; Vascular",
          "image_shows":    "From this distance only foliar canopy and overall plant form are visible; stem, root, pod, and vascular signs cannot be assessed.",
          "image_quote":    "The photograph is a wide field view showing browned canopies of whole rows; no close-up of stems, roots, or pods is visible."
        },
        {
          "field":          "look_alikes",
          "canonical_says": "(not specified)",
          "image_shows":    "At field-view scale, this row-level premature browning could be confused with drought desiccation, sudden death syndrome, or stem canker collapse.",
          "image_quote":    "The browning pattern is visible only as bulk row-level senescence with no diagnostic close-up of microsclerotia or stem streaking."
        }
      ]
    },

    "Kentucky": {
      "image_ids": ["bugwood::7654321"],
      "deltas": [
        {
          "field":          "diagnostic_features",
          "canonical_says": "cross-sections of lower stem and taproot streaked with gray hyphae and microsclerotia giving a marbled, charcoal-sprinkled appearance",
          "image_shows":    "Diagnosis here is performed by stripping the outer bark longitudinally rather than cross-sectioning, exposing a continuous silvery-gray inner surface densely peppered with fine black microsclerotia.",
          "image_quote":    "Each of the five stems has a long longitudinal strip of bark peeled back, revealing a smooth pale silvery surface flecked uniformly with very fine black specks."
        },
        {
          "field":          "severity",
          "canonical_says": "infected plants may die prematurely and are often wilted and stunted",
          "image_shows":    "All sampled Kentucky plants are fully dead and desiccated post-maturity with no remaining foliage — end-stage colonization rather than active wilting.",
          "image_quote":    "Five entirely leafless, dried, woody lower-stem-and-taproot specimens lie on cracked soil, with brittle dead lateral roots fanning out at the lower right."
        },
        {
          "field":          "affected_organs",
          "canonical_says": "Foliar; Stem; Root; Seed; Pod; Vascular",
          "image_shows":    "Only lower stem and taproot expression is documented in this frame; no cotyledon, leaf, pod, or seed tissue is included.",
          "image_quote":    "The image contains only stripped lower stems and taproots laid horizontally on bare soil, with no leaves, pods, or seed visible."
        }
      ]
    }

  }
}
```

Same disease, different visual contexts, different deltas. Canonical stays fixed. Regional contains **only** the local image's additions/contradictions; nothing in either delta block restates a canonical field unchanged.

---

## How it's built (5 stages)

```
                     YOU (laptop)                                              GitHub                  Nova
                     ───────────                                               ──────                  ────
BugWood_Diseases_usable.csv
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 1. DISCOVERY            claude -p WebSearch              │
│    one search per disease (parallel)                     │
│    → discovery_results.json                              │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 2. EXTRACTION           fetch URL + claude -p            │
│    per-source extraction with VERBATIM quotes from page  │
│    text. Captures `treatments` (mgmt section).           │
│    → raw_extractions.json (per crop)                     │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 3. RECONCILIATION       claude -p (no API key needed)    │
│    merge per-source records into ONE canonical entry per │
│    disease. Every field stays {value, url, quote}.       │
│    → final_registry.json (per crop, canonical-only)      │
│      → ❶ canonical block of every SymptomProfile         │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 4a. STATE-AWARE IMAGE CACHE TOP-UP                       │
│     scripts/ensure_state_image_cache.py downloads ONE    │
│     Bugwood image per (crop, disease, state) tuple.      │
│     → smoke/.bugwood_cache/<image_number>.jpg            │
│                                                           │
│ 4b. PER-STATE VLM OBSERVATION   claude -p + Read tool    │
│     For each (crop, disease, state) with a cached image: │
│       reads the image,                                    │
│       reads the canonical entry from step 3,              │
│       walks canonical like a decision tree,               │
│       emits ONLY deltas {field, canonical_says,           │
│                          image_shows, image_quote}.       │
│     → embeds the deltas back INTO final_registry.json    │
│       under each disease's `regional_observations` field. │
│       (No separate file — one unified registry per crop.) │
│       → ❷ regional_observations[state] of every profile  │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 5. ADAPTER + MERGE      symptoms_adapter.merge…          │
│    layers ❶ + ❷ into one SymptomProfile per (crop,disease)│
│    → smoke/artifacts/pathome_seed/symptoms_seed.json     │
└──────────────────────────────────────────────────────────┘
 │
 ▼ git push
                                                          ┌─────────┐    git pull
                                                          │  GitHub │ ─────────────►  Phase 0R: Qwen swarm
                                                          └─────────┘                 (vLLM on Nova A100)
```

Stages 1–3 are the **SAGE port** (`internet_pipeline.py`). Stage 4b is the **Pathome image-grounded delta extraction** (`regional_observation.py`). Stage 5 is the adapter (`symptoms_adapter.py`).

---

## Where each stage lives

| Stage | Code | Output | LLM input |
|---|---|---|---|
| 1 Discovery | `internet_pipeline._run_targeted_discovery` | `discovery_results.json` | (disease name) → `claude -p WebSearch` |
| 2 Extraction | `internet_pipeline._run_extraction` | `raw_extractions.json` | (URL page text) → `claude -p` (no tools) → fields with verbatim quotes + treatments |
| 3 Reconciliation | `internet_pipeline._run_reconciliation` | `final_registry.json` (canonical-only at this point) | (per-source records) → `claude -p` → canonical disease entry |
| 4a Image cache | `scripts/ensure_state_image_cache.py` | `smoke/.bugwood_cache/` | (no LLM) — pure URL fetch |
| 4b Regional observation | `regional_observation.run_regional_observation` | embedded into `final_registry.json` (`regional_observations[state].deltas`) | (image + canonical brief) → `claude -p --allowedTools Read` → deltas-only |
| 5 Merge | `symptoms_adapter.merge_registries_to_seed` | `symptoms_seed.json` | (no LLM) — pure assembly |

All per-crop artefacts land under `artifacts/pathome_kb/<Crop>/`. The merged seed lands at `smoke/artifacts/pathome_seed/symptoms_seed.json` (smoke) or `artifacts/pathome_seed/symptoms_seed.json` (production).

---

## What stage 4b actually puts in the LLM (the deltas-only VLM stage)

```
   ┌──────────────────────────────────────────────────────────┐
   │ Image:    /Users/.../.bugwood_cache/1234567.jpg          │
   │                                                           │
   │ Canonical KB (already populated — DO NOT repeat):         │
   │   - pathogen: Macrophomina phaseolina                     │
   │   - type_of_disease: Fungal                               │
   │   - affected_organs: Foliar; Stem; Root; Seed; Pod; …    │
   │   - lesion_morphology: brown-to-dark spots on cotyledons,│
   │       reddish-brown lesions, charcoal-sprinkled stem…    │
   │   - diagnostic_features: microsclerotia in pith, marbled │
   │       cross-sections…                                     │
   │   - look_alikes: (not specified)                          │
   │   - treatments: drought-stress mitigation, reduced …     │
   │                                                           │
   │ claude -p --allowedTools Read --max-turns 5               │
   │   → reads image at the given path                         │
   │   → walks canonical like a decision tree                  │
   │   → emits JSON:                                           │
   │     { "deltas": [                                         │
   │         { field, canonical_says, image_shows, image_quote},│
   │         …                                                 │
   │       ] }                                                 │
   │                                                           │
   │ If the image confirms canonical exactly → "deltas": []    │
   │ Restating canonical text is forbidden by prompt.          │
   └──────────────────────────────────────────────────────────┘
```

This is the only stage where the model is allowed to write text that isn't a verbatim quote — but every `image_shows` is anchored to a specific Bugwood `image_id` and accompanied by a one-sentence `image_quote` that points to literal visual evidence in that frame.

---

## Aggregate (smoke; Soybean, full coverage, real numbers)

After running `SMOKE_CROPS="Soybean" bash smoke/run_phase0_full.sh`:

```
profiles total                     : 17
profiles w/ canonical data         : 11
profiles w/ canonical treatments   : 11
profiles w/ regional observations  : 11
total per-state blocks             : 54
total state-specific deltas        : 211
text-grounded citations (canonical): ~120
deltas by canonical field          :
  spread_pattern        ~55
  severity              ~50
  affected_organs       ~45
  diagnostic_features   ~30
  look_alikes           ~20
  lesion_morphology     ~10
  other                 ~1
```

Each (crop, disease, state) tuple where Bugwood has a cached image gets one regional observation block; the rest of the seed is the shared canonical text. Average ~3.9 deltas per (disease, state).

---

## Files on disk after a run

```
artifacts/pathome_kb/Soybean/
  ├── discovery_results.json        candidate URLs (claude -p WebSearch)
  ├── raw_extractions.json          per-source extraction with verbatim quotes
  ├── final_registry.json           UNIFIED — canonical + embedded
  │                                 regional_observations[state].deltas per disease
  ├── final_registry.xlsx           single-sheet decision-tree view
  └── registry.md                   human-readable canonical summary

smoke/.bugwood_cache/               JPEGs (one per (crop, disease, state) tuple)
smoke/artifacts/pathome_seed/symptoms_seed.json    final assembled KB (terminal deliverable)
```

There is exactly **one JSON file per crop** holding both canonical and regional. The previous `regional_observations.json` standalone file has been retired.

Convert the unified registry to Excel:

```bash
python3 scripts/registry_to_excel.py \
    artifacts/pathome_kb/Soybean/final_registry.json \
    --out artifacts/pathome_kb/Soybean/final_registry.xlsx
```

One sheet, one row per disease, canonical fields on the left, a single `regional_deltas (decision-tree)` cell on the right grouped per state.

---

## CLI surface

```bash
# Single command — perfect-KB regenerate end-to-end
SMOKE_CROPS="Soybean,Tomato" bash smoke/run_phase0_full.sh

# Direct pathome_kb invocation
python -m pathome_kb \
  --csv BugWood_Diseases_usable.csv \
  --out artifacts/pathome_seed/symptoms_seed.json \
  --regional                                  # turn on stage 4b

# Restrict scope while iterating
--quick                            # smaller per-stage caps
--only-crops "Soybean,Tomato"      # crop allowlist
--limit-crops 5                    # first N crops alphabetically
--resume-from extraction           # use cached upstream artefacts
--no-cache                         # re-run even if final_registry.json exists
--regional-only                    # skip stages 1-3, just rerun stage 4b on
                                   # cached final_registry.json
```

The smoke wrapper:

```bash
SMOKE_CROPS="Soybean" bash smoke/run_phase0_full.sh         # full coverage, ~45-90 min, ~$5-15
FULL_QUICK=1 bash smoke/run_phase0_full.sh                  # fast, ~15-25 min, ~$1-3
FULL_KEEP_CACHE=1 bash smoke/run_phase0_full.sh             # reuse cached canonical
FULL_SKIP_KB=1 bash smoke/run_phase0_full.sh                # only cache top-up + setup
```

---

## Authentication

| What | Why | Failure mode |
|---|---|---|
| `claude` CLI on PATH | Stages 1, 2, 3, 4b all use `claude -p`; 4b uses the Read tool | `pathome_kb` exits with "claude CLI not on PATH" |
| `claude auth login` once | OAuth login for the CLI; sessions reuse the token | First `claude -p` blocks for browser sign-in |
| `ANTHROPIC_API_KEY` (optional) | Stage 3 reconciliation can use the SDK directly when present — slightly faster than subprocess. Without it, reconciliation falls back to `claude -p` automatically. | None — fallback is transparent |

Nova compute nodes can't run `claude auth login`, which is why Phase 0 runs locally and the seed file is shuttled through git.

---

## Cost & time (approximate)

| Run mode | LLM calls | Walltime | Cost (claude -p over OAuth) |
|---|---|---|---|
| Smoke `FULL_QUICK=1` (1 crop) | ~80 | ~15–25 min | ~$1–3 |
| Smoke full (1 crop) | ~150 | ~35–45 min | ~$5–10 |
| Production full (484 classes) | ~2500 | ~16–24 h | ~$60–180 |

(Production estimates assume `MAX_PARALLEL_EXTRACTIONS=4` in `config.py` and Sonnet 4.6 default. Bigger parallelism reduces wall but not cost; OAuth quota is the practical ceiling.)

---

## Consuming the KB downstream

```python
from pathome import PathomeDB

db = PathomeDB.load("artifacts/pathome_db/")

# Canonical-only context (no state)
prompt = db.symptom_context("Soybean", "Charcoal Rot")

# Canonical + this state's deltas, ready to drop into a prompt
prompt = db.symptom_context("Soybean", "Charcoal Rot", state="Alabama")
```

`SymptomProfile.context_for_state()` and `RegionalObservation.narrative()` are the supported entry points; agents/scripts should not reach into the dataclass fields directly.

---

## What changed vs the previous schema

| Two ago (rejected) | Previous (rejected) | Now |
|---|---|---|
| `SymptomProfile.visual` flat block + `regional_visuals[state]` duplicating canonical text | `regional_observations[state]` with parallel `severity` / `lesion_morphology` / `affected_organs` / `spread_pattern` + `variations_from_canonical[]` bullets | `regional_observations[state].deltas[]` — structured `{field, canonical_says, image_shows, image_quote}` records, deltas-only |
| Two stages (`regional_extraction.py` + `regional_image_fill.py`) | One stage emitting parallel fields | One stage (`regional_observation.py`) emitting deltas only |
| Two artefacts (`regional_registries.json`, `regional_image_fills.json`) | Separate `regional_observations.json` per crop | Embedded INTO `final_registry.json` — one unified registry per crop |
| Citation `grounding` field optional | `grounding="text"` (canonical) or `"image"` (regional) | unchanged — still required |
| Treatments field absent | Added to `extraction.py` and `reconciliation.py` | unchanged — still in canonical |
| Excel: 2 sheets (canonical, regional) | Excel: 2 sheets (canonical, regional) | Excel: 1 sheet, one row per disease, regional folded into a single decision-tree cell |
