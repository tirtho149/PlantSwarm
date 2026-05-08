# `pathome_kb/` — How the KB is generated

This module produces the **PathomeDB seed file** that Phase 1 (`scripts/build_pathome.py`) consumes. The seed file is a single JSON document containing one `SymptomProfile` per `(crop, disease)` class in your filtered Bugwood CSV, each with:

- a cross-region visual block (text from extension-service literature, with verbatim quotes)
- per-state regional blocks (same text re-scoped + image-grounded enum fields from the Bugwood photo for that state)
- citations tagged by **grounding** ∈ {`text`, `image`}

The whole pipeline runs locally (Nova compute nodes block claude OAuth). Output gets pushed to GitHub; Nova `git pull`s and consumes the seed via Phase 1 onward.

---

## 30-second mental model

```
                     YOU (laptop)                                              GitHub                  Nova
                     ───────────                                               ──────                  ────
BugWood_Diseases_usable.csv
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 1. DISCOVERY            claude -p WebSearch              │
│    one search per disease (parallel) → ~100 candidate URLs│
│    → discovery_results.json                              │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 2. EXTRACTION           fetch URL + claude -p            │
│    per-source extraction with VERBATIM quotes from page   │
│    text. Never invents content.                          │
│    → raw_extractions.json (per crop)                     │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 3. RECONCILIATION       claude -p (no API key needed)    │
│    merge per-source records into canonical per-disease    │
│    entries. Every field stays {value, url, quote}.       │
│    → final_registry.json (per crop)                      │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 4. REGIONAL EXTRACTION  claude -p over cached records    │
│    one call per (crop, disease, state) tuple where      │
│    Bugwood has images. Honest: when sources don't        │
│    mention the state, sets state_specific=false.         │
│    → regional_registries.json (per crop)                 │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 5. IMAGE-GROUNDED FILL  claude -p + Read tool            │
│    looks at the cached Bugwood image and fills *empty*   │
│    enum fields (color/shape/margin/texture/sporulation/  │
│    progression) with grounding="image" citations.        │
│    Text-grounded fields are LEFT UNTOUCHED.              │
│    → regional_image_fills.json (per crop)                │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 6. ADAPTER + MERGE      merge_registries_to_seed         │
│    layers all four artefacts into one SymptomProfile     │
│    per (crop, disease). Each citation tagged with        │
│    grounding ∈ {"text", "image"}.                       │
│    → symptoms_seed.json                                  │
└──────────────────────────────────────────────────────────┘
 │
 ▼ git push -f
                                                          ┌─────────┐    git pull
                                                          │  GitHub │ ─────────────►  PathomeDB Phase 1
                                                          └─────────┘                 (Nova A100)
```

Stages 1–3 are the **SAGE port** (`internet_pipeline.py`). Stages 4–5 are the **Pathome additions** (`regional_extraction.py`, `regional_image_fill.py`). Stage 6 is the adapter (`symptoms_adapter.py`).

---

## Where each stage lives

| Stage | Code | Output (per crop) | Reads | Writes |
|---|---|---|---|---|
| Setup | `scripts/filter_bugwood_csv.py` | `BugWood_Diseases_usable.csv` | raw IPMNet CSV | filtered CSV + per-class report |
| 1 Discovery | `internet_pipeline._run_targeted_discovery` | `discovery_results.json` | filtered CSV (disease names) | candidate URLs per disease |
| 2 Extraction | `internet_pipeline._run_extraction` | `raw_extractions.json` | discovery results | per-source records with verbatim quotes |
| 3 Reconciliation | `internet_pipeline._run_reconciliation` | `final_registry.json` | raw extractions | canonical disease entries (cross-region) |
| 4 Regional | `regional_extraction.run_regional_extraction` | `regional_registries.json` | raw extractions + state image map | per-(disease, state) records |
| 5 Image-fill | `regional_image_fill.run_regional_image_fill` | `regional_image_fills.json` | regional registries + cached Bugwood images | per-(disease, state) image-derived fields |
| 6 Merge | `symptoms_adapter.merge_registries_to_seed` | `symptoms_seed.json` | all of the above | the seed Phase 1 consumes |

All per-crop artefacts land under `artifacts/pathome_kb/<Crop>/`. The merged seed lands at `smoke/artifacts/pathome_seed/symptoms_seed.json` (smoke) or `artifacts/pathome_seed/symptoms_seed.json` (production).

---

## What each stage actually puts in the LLM

### Stage 1 — Discovery (`claude -p` WebSearch)

```
   ┌─────────────────────────────────────────┐
   │  Crop: Tomato                           │
   │  Disease: Early Blight                  │  → claude -p
   │  Tools: [WebSearch]                     │      WebSearch:
   │  Schema: {url, title, source_type, ...} │      "Tomato Early Blight extension"
   └─────────────────────────────────────────┘      "Tomato Early Blight UMN APS"
                                                    ...
                                              ◄─── 4-8 candidate URLs
```

One claude -p invocation per disease, parallel across 4 workers. Output: `candidate_sources` array per disease, deduplicated by URL.

### Stage 2 — Extraction (`claude -p` per URL)

```
   For each candidate URL:
   ┌─────────────────────────────────────────┐
   │ httpx.get(url) → page_text (≤30 KB)     │
   │ claude -p (no tools)                    │
   │   "Extract every disease mentioned in   │ → claude -p (text-only)
   │    this page. Use VERBATIM quotes from  │
   │    the page. Never invent content."     │
   │   Schema: {extractions: [...]}          │
   └─────────────────────────────────────────┘
                                              ◄─── disease records with
                                                   {value, url, quote}
                                                   per field
```

This is the **provenance backbone**. The LLM is forbidden from filling fields from its own knowledge; only verbatim sentences from the page text become evidence.

### Stage 3 — Reconciliation (`claude -p` over batched extractions)

```
   ┌─────────────────────────────────────────┐
   │ Group all per-source records for one    │
   │ disease across ~3-5 source URLs:        │
   │   [src1.early_blight,                   │ → claude -p (text-only)
   │    src2.early_blight,                   │
   │    src3.early_blight, ...]              │
   │ "Merge into one canonical entry. Keep   │
   │  per-field citations. Flag conflicts."  │
   │ Schema: FINAL_REGISTRY_SCHEMA           │
   └─────────────────────────────────────────┘
                                              ◄─── canonical Disease record:
                                                   {pathogen_scientific_name: {value,url,quote},
                                                    affected_parts: ...,
                                                    visual_symptoms: {summary, dx, look_alikes},
                                                    confidence, num_sources, conflicts}
```

### Stage 4 — Regional extraction (per-state filter on cached records)

```
   For each (crop, disease, state) where Bugwood has images:
   ┌─────────────────────────────────────────┐
   │ Load cached raw_extractions.json (Stage │
   │ 2 output). Filter to disease records.   │
   │ claude -p (no tools, no web)            │
   │   "Crop=Tomato Disease=Early Blight     │ → claude -p
   │    State=Alabama. Extract symptoms      │
   │    SPECIFIC TO Alabama using only       │
   │    verbatim quotes from the records'    │
   │    evidence fields."                    │
   │   Honest output: state_specific=false   │
   │   when no Alabama-mentioning text.      │
   └─────────────────────────────────────────┘
                                              ◄─── {state: "Alabama",
                                                    state_specific: bool,
                                                    summary, diagnostic_features,
                                                    affected_parts, look_alikes}
                                                   Each citation tagged with
                                                   bugwood::N image_id from this state.
```

In practice `state_specific=false` is common because extension factsheets are written US-wide. The pass still produces a record per state — the difference is the **image_id** carried forward, not the text.

### Stage 5 — Image-grounded fill (`claude -p` with Read tool)

```
   For each (crop, disease, state) with a cached Bugwood photo:
   ┌──────────────────────────────────────────────────────────┐
   │ Empty fields in regional_visuals[state]?                  │
   │ → typically: color, shape, margin, texture,               │
   │   sporulation, progression                                │
   │ claude -p --allowedTools "Read" --max-turns 5             │
   │   "Read /path/to/bugwood_NNN.jpg via Read tool, then       │
   │    fill these empty fields. State=Alabama Crop=Tomato      │ → claude -p (vision)
   │    Disease=Early Blight. Be specific to this photo;        │
   │    leave fields empty if you can't tell from the image."   │
   │   Schema: {color, shape, margin, ...} each with            │
   │           {value, quote: "<your visual description>"}      │
   └──────────────────────────────────────────────────────────┘
                                              ◄─── color: ["brown", "dark brown", "black"]
                                                   progression: "advanced defoliation"
                                                   shape: ""   ← honest empty
                                                   margin: ""  ← honest empty
                                                   each with quote = model's
                                                   one-sentence visual description
```

The model is allowed to refuse fields it can't determine from a single photo. The citations carry `grounding="image"` and the Bugwood `image_id`; `url` is empty since the image IS the witness.

### Stage 6 — Adapter merge → SymptomProfile

```
SymptomProfile {
  profile_id:  "Tomato::Early Blight",
  crop:        "Tomato",
  disease:     "Early Blight",

  # Cross-region (Stage 3 product)
  visual: {
    plant_parts: ["leaf", "stem", "fruit"],
    distinctive_signs: ["concentric rings on lesions", ...],
    confusion_diseases: ["Septoria leaf spot", ...],
    notes: "Small dark brown spots on lower leaves expand into ...",
    sources: {
      "plant_parts":   [{ value, url: ces.ncsu.edu/.../early-blight,
                          quote: "...verbatim from page...",
                          grounding: "text" }],
      ...
    }
  },

  # Per-state — this is what was missing in V1
  regional_visuals: {
    "Alabama": {
      plant_parts: ["leaf", "fruit", "stem"],            ←── Stage 4 (text)
      distinctive_signs: ["Brownish-black lesions ..."], ←── Stage 4 (text)
      color: ["brown", "dark brown", "blackened"],       ←── Stage 5 (image)
      texture: ["dry", "shriveled"],                     ←── Stage 5 (image)
      reference_image_ids: ["bugwood::1568038"],
      sources: {
        "plant_parts": [
          { value, url: ces.ncsu.edu/...,
            quote: "...", image_id: "bugwood::1568038",
            grounding: "text" }
        ],
        "color": [
          { value: "brown; dark brown; blackened",
            url: "",
            quote: "Foliage across the staked tomato row appears
                    uniformly brown to nearly black ...",
            image_id: "bugwood::1568038",
            grounding: "image" }
        ]
      }
    },
    "Connecticut": { ... different reference_image_ids, same structure ... }
  },

  # Empirical (filled later by build_pathome.py / enhance_pathome_from_traces.py)
  state_counts: {"Alabama": 3, ...},
  reference_ids: ["bugwood::..."],
  swarm_observations: null
}
```

**Two key invariants:**
1. Every `Citation` has a `grounding` field — `text` (URL+quote from a real page) or `image` (model description grounded in a Bugwood photo).
2. `image_id` is carried on EVERY regional citation regardless of grounding type. So a downstream consumer can always fetch the supporting Bugwood photograph.

---

## Worked example (real numbers from the smoke run)

For `Tomato::Early Blight / Alabama` after the full Phase 0 (stages 1–6):

| Field | grounding | source |
|---|---|---|
| `plant_parts: ["leaf", "fruit", "stem"]` | `text` | NC State Extension |
| `distinctive_signs: [".. concentric rings .."]` | `text` | NC State Extension |
| `confusion_diseases: ["Septoria leaf spot"]` | `text` | NC State Extension |
| `notes: "Small dark brown spots on lower leaves..."` | `text` | NC State Extension |
| `color: ["brown", "dark brown", "blackened"]` | `image` | bugwood::1568038 |
| `texture: ["dry", "shriveled"]` | `image` | bugwood::1568038 |
| `shape: ""` | _none_ | model honestly couldn't tell |
| `margin: ""` | _none_ | model honestly couldn't tell |

For the same disease in Connecticut, `reference_image_ids = ["bugwood::5559537"]` and the image-grounded fields will reflect what's visible in *that* photo (potentially smaller lesions on younger plants, different host cultivar, etc.). The text-grounded fields are the same because the extension service text is US-wide.

---

## Aggregate (smoke; Tomato + Soybean, 25 classes)

After running all six stages:

```
profiles total           : 25
profiles w/ visual data  : 17   (cross-region SAGE survived)
profiles w/ regional data: 19   (per-state blocks present)
per-state blocks         : 38

text-grounded citations  : 130
image-grounded citations :  61
```

The 8 still-empty profiles are mostly Soybean diseases whose canonical names didn't lex-match the disease records the LLM extracted (e.g. "Soybean Rust" vs "Phakopsora pachyrhizi"). Those need either tighter matching prompts or a second pass — out of scope for this run.

---

## Reusable helpers in this module

```python
from pathome_kb.shared import claude_query, claude_query_with_image
# claude_query: text-only claude -p with optional tools, JSON schema, system prompt.
# claude_query_with_image: same but auto-prepends the image path and whitelists Read.

from pathome_kb.regional_extraction import build_state_image_map
# (crop, disease, state) → [bugwood::N, ...] from the filtered CSV

from pathome_kb.symptoms_adapter import (
    disease_to_profile_dict, regional_record_to_visual_dict,
    merge_registries_to_seed,
)
# Adapter primitives — useful if you want to assemble a SymptomProfile dict
# from a custom registry source (paper-supplement bibliography, etc.)
```

---

## CLI surface

```bash
# Full Phase 0 from scratch
python -m pathome_kb \
  --csv BugWood_Diseases_usable.csv \
  --out artifacts/pathome_seed/symptoms_seed.json

# + per-state regional extraction (additive)
python -m pathome_kb ... --regional

# + image-grounded fill (additive)
python -m pathome_kb ... --regional --regional-image-fill

# Stages 4-5 only (assumes cached raw_extractions.json + final_registry.json)
python -m pathome_kb ... --regional-only
python -m pathome_kb ... --regional-image-only

# Restrict scope while iterating
python -m pathome_kb ... --quick                          # cap sources/extractions
python -m pathome_kb ... --only-crops "Tomato,Soybean"
python -m pathome_kb ... --limit-crops 5
python -m pathome_kb ... --resume-from extraction         # use cached upstream
python -m pathome_kb ... --no-cache                       # ignore cached registries
```

The smoke directory wraps the common patterns:

```bash
bash smoke/run_phase0_local.sh           # Stages 1-4 (cross-region + regional text)
bash smoke/run_phase0_regional_only.sh   # Stage 4 only (refresh per-state text)
bash smoke/run_phase0_image_fill.sh      # Stage 5 only (refresh image-grounded)
```

---

## Authentication

| What | Why | Failure mode |
|---|---|---|
| `claude` CLI on PATH | Stages 1, 2, 4, 5 use `claude -p`; Stage 5 uses the Read tool | `pathome_kb` exits with "claude CLI not on PATH" |
| `claude auth login` once | OAuth login for the CLI; CLI sessions reuse the token | First `claude -p` blocks for browser sign-in |
| `ANTHROPIC_API_KEY` in env or `.env` (optional) | Stage 3 reconciliation can use the SDK directly when available — slightly faster than subprocess. Without it, reconciliation falls back to `claude -p` automatically. | None — the fallback is transparent |

Nova compute nodes typically can't run `claude auth login`, which is why Phase 0 must run on a local machine and the seed file is shuttled through git.

---

## Cost & time (approximate)

| Run mode | LLM calls | Walltime | Cost (claude -p over OAuth) |
|---|---|---|---|
| Smoke `--quick` (2 crops) full | ~80 | ~10–15 min | ~$1–2 |
| Smoke regional-only | ~38 | ~3–5 min | ~$0.5 |
| Smoke image-fill only | ~16 | ~2 min | ~$0.5 |
| Production (484 classes) full | ~1500 | ~12–20 h | ~$50–150 |
| Production regional-only | ~700 | ~3–4 h | ~$10–20 |
| Production image-fill only | ~700 | ~3–4 h | ~$15–25 |

(Production estimates assume `MAX_PARALLEL_EXTRACTIONS=4` in `config.py` and Sonnet 4.6 default. Bigger parallelism reduces wall but not cost; OAuth quota is the practical ceiling.)
