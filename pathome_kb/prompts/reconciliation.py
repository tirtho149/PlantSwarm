RECONCILIATION_PROMPT = """\
You are a data reconciliation agent. You will receive:
1. Raw extraction data: disease records extracted from multiple web sources, each
   tagged with source URL, quotes, and evidence.
2. Taxonomy validation data: verified pathogen names with NCBI/MycoBank references.

YOUR TASK: Merge all extraction records into a single canonical disease registry.

RULES:
0. COMPLETENESS IS MANDATORY: You MUST output EVERY disease mentioned in the input,
   even if only one source mentions it. Never drop, skip, or omit a disease.
   Zero disease loss is required.
1. GROUP records by disease — match on disease_name + pathogen (fuzzy match, e.g.
   "Bacterial Blight" and "Bacterial blight" are the same disease).
2. For each field, SELECT the canonical value using this source authority ranking
   (highest first):
   - Rank 1: NCBI Taxonomy / MycoBank (for pathogen names only)
   - Rank 2: APS Compendium / peer-reviewed compendia
   - Rank 3: CABI datasheets
   - Rank 4: Peer-reviewed papers (PMC, journals)
   - Rank 5: University extension services (UMN, ISU, Purdue, etc.)
   - Rank 6: Industry sources (Bayer, Corteva, etc.)
3. If sources DISAGREE on a field, record BOTH values — the chosen canonical value
   AND the alternatives in a "conflicts" array.
4. Every non-null field MUST have at least one citation with:
   - source URL (the page it came from)
   - quote (the verbatim text from that page)
5. CONFIDENCE scoring:
   - "high": 2+ independent sources agree, with direct quotes
   - "medium": 1 authoritative source with direct quote
   - "low": 1 non-authoritative source or weak evidence
6. For pathogen_scientific_name: use the taxonomy-validated name as canonical.
   If the taxonomy validation found it was a synonym/outdated name, use the
   current accepted name and note the original in conflicts.
7. For visual_symptoms sub-fields: merge the BEST description from across sources.
   Prefer the most specific, visually descriptive quote. Each sub-field gets its
   own citation.
8. For treatments: merge the union of management / control measures the sources
   recommend. Deduplicate near-identical entries; preserve a citation for each.
9. Do NOT invent or infer any values. If no source provides a field, leave it null.

Raw extractions:
{extractions}

Taxonomy validations:
{taxonomy}

Output the final registry as JSON matching the required schema.
"""

# Citation schema used per-field in the final registry
# Use empty string "" instead of null to avoid union types (API limit: 16 unions)
_CITED_FIELD = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "description": "Field value, or empty string if unknown"},
        "url": {"type": "string", "description": "Source URL for this value, or empty string"},
        "quote": {"type": "string", "description": "Verbatim quote from source, or empty string"},
    },
    "required": ["value", "url", "quote"],
}

_CITED_FIELD_ARRAY = {
    "type": "object",
    "properties": {
        "value": {"type": "array", "items": {"type": "string"}},
        "url": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["value", "url", "quote"],
}

_CITED_VISUAL_SYMPTOMS = {
    "type": "object",
    "properties": {
        "summary": _CITED_FIELD,
        "diagnostic_features": _CITED_FIELD,
        "look_alikes": _CITED_FIELD_ARRAY,
    },
    "required": ["summary", "diagnostic_features", "look_alikes"],
}

_PATHOGEN_CITED_FIELD = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "url": {"type": "string"},
        "quote": {"type": "string"},
        "taxonomy_validated": {"type": "boolean"},
        "ncbi_url": {"type": "string"},
        "mycobank_url": {"type": "string"},
    },
    "required": ["value", "url", "quote", "taxonomy_validated"],
}

_CONFLICT = {
    "type": "object",
    "properties": {
        "field": {"type": "string"},
        "canonical_value": {"type": "string"},
        "alternative_value": {"type": "string"},
        "alternative_source_url": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["field", "canonical_value", "alternative_value"],
}

FINAL_REGISTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "crop": {"type": "string"},
        "generated_date": {"type": "string"},
        "diseases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "disease_name": {"type": "string"},
                    "pathogen_scientific_name": _PATHOGEN_CITED_FIELD,
                    "type_of_disease": _CITED_FIELD,
                    "affected_parts": _CITED_FIELD_ARRAY,
                    "visual_symptoms": _CITED_VISUAL_SYMPTOMS,
                    "treatments": _CITED_FIELD_ARRAY,
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "num_sources": {"type": "integer"},
                    "conflicts": {
                        "type": "array",
                        "items": _CONFLICT,
                    },
                },
                "required": [
                    "disease_name",
                    "pathogen_scientific_name",
                    "type_of_disease",
                    "affected_parts",
                    "visual_symptoms",
                    "treatments",
                    "confidence",
                    "num_sources",
                    "conflicts",
                ],
            },
        },
    },
    "required": ["crop", "generated_date", "diseases"],
}


# ─── Name Normalization (post-reconciliation dedup) ──────────────────────

NAME_NORMALIZATION_PROMPT = """\
You are a plant pathology nomenclature expert. Below is a JSON list of disease names
extracted from multiple sources. Some names refer to the SAME disease but use different
phrasing (e.g., "White Mold" vs "Sclerotinia Stem Rot (White Mold)", or "Soybean Rust"
vs "Asian Soybean Rust").

Your task: Group names that refer to the same disease, and pick ONE canonical name per
group using APS (American Phytopathological Society) common name conventions.

RULES:
1. Only group names that truly refer to the same disease caused by the same pathogen.
2. Do NOT merge diseases that are distinct even if they sound similar (e.g.,
   "Rhizoctonia damping-off" vs "Rhizoctonia aerial blight" are different diseases).
3. For the canonical name, prefer the APS standard common name.
4. Every input name must appear exactly once in the output.

Input disease names:
{disease_names}
"""

NAME_NORMALIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {
                        "type": "string",
                        "description": "The APS standard canonical name for this disease",
                    },
                    "original_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All input names that refer to this disease",
                    },
                },
                "required": ["canonical_name", "original_names"],
            },
        },
    },
    "required": ["groups"],
}
