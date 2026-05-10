# PathomeDB — sample KB entry (decision-tree shape)

This doc walks through one real disease (`Soybean :: Charcoal Rot`) as
an end-to-end example of the unified `final_registry.json` shape, so
you can see exactly what canonical owns, what regional adds, and how
the two compose without duplication.

> Source file: `artifacts/pathome_kb/Soybean/final_registry.json`
> Source crop: Soybean (smoke run, 2026-05-08)

---

## 1. The shape

There is **one JSON file per crop**. Inside, every disease entry has
two parts:

```
diseases[i] = {
    canonical KB              ← cross-region, web-sourced, text + URL + quote
    regional_observations: {  ← per-state, image-grounded deltas only
        <state>: {
            image_ids: [...],
            deltas: [
                { field, canonical_says, image_shows, image_quote }
            ]
        }
    }
}
```

The decision tree:

```
                        ┌────────────────────────────┐
                        │  CANONICAL KB (the trunk)  │
                        │  pathogen, type, organs,   │
                        │  summary, diagnostic feats,│
                        │  look-alikes, treatments   │
                        └─────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
        ━━ Alabama ━━           ━━ Kentucky ━━           ━━ South Dakota ━━
        (field-view photo)      (close-up specimens)     (canopy photo)
              │                       │                       │
        deltas[]                deltas[]                deltas[]
              │                       │                       │
   adds spread_pattern,      adds diagnostic_features  adds severity,
   refines severity,         narrowing, refines        refines spread_pattern
   narrows affected_organs,  severity to "post-mature
   adds look_alikes          end-stage"
```

Regional **never** restates canonical. Each delta names which canonical
field it refines, what canonical actually says about that field, what
the image adds, and a one-sentence visual quote that backs the claim.

---

## 2. The canonical KB block (Soybean :: Charcoal Rot)

Built once per disease from claude-headless web search of extension
service literature. Every field carries a URL + verbatim quote.

| Canonical field          | Value                                                                                                              |
|--------------------------|--------------------------------------------------------------------------------------------------------------------|
| `pathogen_scientific_name` | *Macrophomina phaseolina*                                                                                          |
| `type_of_disease`         | Fungal                                                                                                             |
| `affected_parts`          | Foliar; Stem; Root; Seed; Pod; Vascular                                                                            |
| `summary`                 | Brown-to-dark spots on cotyledons; circular-to-oblong reddish-brown lesions turning dark; premature wilt, stunting; taproot and lower stem streaked light gray with charcoal-sprinkled microsclerotia. |
| `diagnostic_features`     | Microsclerotia in vascular tissue, pith, dry pods; cross-sections of lower stem/taproot streaked with gray hyphae and microsclerotia (marbled, charcoal-sprinkled appearance); reddish-brown pith/vascular discoloration. |
| `look_alikes`             | *(not specified by canonical sources)*                                                                             |
| `treatments`              | Drought-stress mitigation; reduced tillage; rotation to non-hosts; reduced seeding rates; cultivar selection; fungicides not typically recommended. |

Sources (URL + quote) for every field above are stored alongside in
`final_registry.json`, e.g.:

```json
"pathogen_scientific_name": {
    "value": "Macrophomina phaseolina",
    "url":   "https://extension.umn.edu/soybean-pest-management/charcoal-rot-soybean",
    "quote": "Charcoal rot is caused by the soilborne fungus Macrophomina phaseolina."
}
```

This block is identical for every state. There is exactly one of it
per disease.

---

## 3. Regional deltas — `Alabama` (field-view photo)

Bugwood image: a wide field shot showing rows of soybean prematurely
brown along the field edge, with green rows further in.

The VLM sees the canonical KB above as context, and emits **only** the
state-specific additions and contradictions:

```json
{
  "image_ids": ["bugwood::1234567", ...],
  "deltas": [
    {
      "field":          "spread_pattern",
      "canonical_says": "(not specified)",
      "image_shows":    "Field-scale distribution showing entire rows along the field edge collapsed and brown while adjacent rows further into the field remain green, indicating a patchy edge-of-field outbreak rather than uniform spread.",
      "image_quote":    "Foreground rows are uniformly tan-brown and prematurely senesced while rows on the right side of the field remain green and intact."
    },
    {
      "field":          "severity",
      "canonical_says": "infected plants may die prematurely and are often wilted and stunted, with leaflets becoming small, wilting and turning brown",
      "image_shows":    "Severity at this Alabama site is advanced and stand-level — multiple contiguous rows show whole-plant death with full canopy desiccation rather than scattered individual plants.",
      "image_quote":    "Several full rows of soybean plants are brown, dried, and collapsed top-to-bottom with little remaining green tissue."
    },
    {
      "field":          "affected_organs",
      "canonical_says": "Foliar; Stem; Root; Seed; Pod; Vascular",
      "image_shows":    "From this distance only the foliar canopy and overall plant form are visible; stem, root, pod, and vascular signs cannot be assessed in this image.",
      "image_quote":    "The photograph is a wide field view showing browned canopies of whole rows; no close-up of stems, roots, or pods is visible."
    },
    {
      "field":          "look_alikes",
      "canonical_says": "(not specified)",
      "image_shows":    "At field-view scale, this row-level premature browning could be confused with drought desiccation, sudden death syndrome, or stem canker collapse, since no close-up signs are visible.",
      "image_quote":    "The browning pattern is visible only as bulk row-level senescence with no diagnostic close-up of microsclerotia or stem streaking."
    }
  ]
}
```

What the deltas tell us about Alabama, beyond canonical:

- **`spread_pattern`** — canonical was silent; the image documents an
  edge-of-field patchy outbreak.
- **`severity`** — canonical describes premature death generically; the
  image refines it to *advanced, stand-level*.
- **`affected_organs`** — canonical lists six organ classes; the image
  acknowledges only foliar evidence is **visible at this scale** (a
  scope contradiction, not a biology contradiction).
- **`look_alikes`** — canonical was silent; the image flags drought,
  SDS, stem canker as field-view confusables.

No restatement of pathogen, type, treatments, summary, or diagnostic
features — those stay in canonical, untouched.

---

## 4. Regional deltas — `Kentucky` (close-up specimens photo)

Bugwood image: five soybean lower-stem-and-taproot specimens lying on
cracked soil, each with bark stripped longitudinally exposing the
silvery-flecked inner surface.

Different image of the **same disease** — different deltas:

```json
{
  "deltas": [
    {
      "field":          "diagnostic_features",
      "canonical_says": "cross-sections of lower stem and taproot streaked with gray hyphae and microsclerotia giving a marbled, charcoal-sprinkled appearance",
      "image_shows":    "Diagnosis here is performed by stripping the outer bark longitudinally rather than cross-sectioning, exposing a continuous silvery-gray inner surface densely peppered with fine black microsclerotia along the entire lower stem.",
      "image_quote":    "Each of the five stems has a long longitudinal strip of bark peeled back, revealing a smooth pale silvery surface flecked uniformly with very fine black specks."
    },
    {
      "field":          "severity",
      "canonical_says": "infected plants may die prematurely and are often wilted and stunted, with leaflets becoming small, wilting and turning brown",
      "image_shows":    "All sampled Kentucky plants are fully dead and desiccated post-maturity with no remaining foliage, indicating end-stage colonization rather than active wilting.",
      "image_quote":    "Five entirely leafless, dried, woody lower-stem-and-taproot specimens lie on cracked soil, with brittle dead lateral roots fanning out at the lower right."
    },
    {
      "field":          "affected_organs",
      "canonical_says": "Foliar; Stem; Root; Seed; Pod; Vascular",
      "image_shows":    "Only lower stem and taproot expression is documented in this frame; no cotyledon, leaf, pod, or seed tissue is included in the sample.",
      "image_quote":    "The image contains only stripped lower stems and taproots laid horizontally on bare soil, with no leaves, pods, or seed visible anywhere in the frame."
    }
  ]
}
```

Same disease, completely different visual context, completely different
deltas. The canonical block is reused as-is; the regional deltas are
**only** what the local image surfaces.

---

## 5. What this fixes vs the previous schema

| Previous (rejected)                            | Now                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| Two JSON files per crop (canonical + regional) | One unified `final_registry.json` per crop                           |
| Two Excel sheets (canonical + regional)        | One sheet, one row per disease, regional folded into a single column |
| Regional re-extracted `severity`, `lesion_morphology`, `affected_organs`, `spread_pattern` per state | Regional emits **only** deltas; those slots live exclusively in canonical |
| `variations_from_canonical` was a flat string list | `deltas[]` is structured: `{field, canonical_says, image_shows, image_quote}` |
| VLM prompt said "describe what THIS image shows" (parallel pass) | VLM prompt says "walk canonical KB; emit ONLY adds or contradicts; if image confirms canonical exactly, return []" |

Net effect: canonical KB is the source of truth (the trunk), regional
is *adds-or-contradicts only* (the branches), and the workbook /
retrieval / VLM-grounding consumer reads exactly one file per crop.

---

## 6. Reproducing this entry

```bash
SMOKE_CROPS="Soybean" bash smoke/run_phase0_full.sh
python3 scripts/registry_to_excel.py \
    artifacts/pathome_kb/Soybean/final_registry.json \
    --out artifacts/pathome_kb/Soybean/final_registry.xlsx
```

End-to-end on a single crop with cached discovery: ~5 min for canonical
+ ~30–35 min for regional VLM grounding (54 (disease, state) tuples).
