# Evidence field schema — reused across all extracted fields
_EVIDENCE_FIELD = {
    "type": ["object", "null"],
    "properties": {
        "value": {"type": ["string", "null"]},
        "evidence": {
            "type": ["string", "null"],
            "description": "Verbatim quote from the source text supporting this value",
        },
    },
    "required": ["value", "evidence"],
}

_EVIDENCE_FIELD_ARRAY = {
    "type": ["object", "null"],
    "properties": {
        "value": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
        "evidence": {"type": ["string", "null"]},
    },
    "required": ["value", "evidence"],
}

_VISUAL_SYMPTOMS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": _EVIDENCE_FIELD,
        "diagnostic_features": _EVIDENCE_FIELD,
        "look_alikes": _EVIDENCE_FIELD_ARRAY,
    },
    "required": ["summary", "diagnostic_features", "look_alikes"],
}

EXTRACTION_PROMPT = """\
You are a data extraction agent for plant disease information.
Below is the text content of a web page. Extract disease entries from it.

CRITICAL RULES:
1. Extract ONLY information explicitly stated in the text below.
2. For every non-null field, include a VERBATIM QUOTE from the text as evidence.
3. If the text does not explicitly state a field's value, set BOTH value and evidence to null.
4. NEVER fill in fields from your own knowledge — only from the provided text.
5. If the text is ambiguous about a field, set it to null and move on.
6. Extract ALL diseases mentioned on the page. Do not skip or omit any disease.

For each disease found, extract:

IDENTITY:
- disease_name: common name of the disease
- pathogen_scientific_name: full binomial (genus species) or null if not stated
- type_of_disease: one of [Fungal, Bacterial, Viral, Nematode, Oomycete, Abiotic] — only if text states it
- affected_parts: list from [Foliar, Stem, Root, Seed, Pod, Vascular, Whole plant]

VISUAL SYMPTOMS (structured for image-based disease identification):
- summary: 1-2 sentence overview of visual symptoms
- diagnostic_features: what visually distinguishes THIS disease from similar ones
- look_alikes: other diseases this could be confused with

SOURCE: {url}
TITLE: {title}

--- PAGE TEXT ---
{page_text}
--- END PAGE TEXT ---

Return the extraction as JSON matching the required schema.
"""

PDF_PAGE_EXTRACTION_PROMPT = """\
You are a data extraction agent for plant disease information.
The attached PDF pages contain information about crop diseases for "{crop}".

The following diseases are known to be covered in this document:
{disease_list}

CONTEXT: An expert sheet already provides disease names, pathogens, and classification
for these diseases. The PDF is being read specifically to extract VISUAL SYMPTOMS and
EPIDEMIOLOGY that the expert sheet lacks. Focus your extraction on these fields.

CRITICAL RULES:
1. Extract ONLY information explicitly stated in the PDF pages.
2. For every non-null field, include a VERBATIM QUOTE from the PDF as evidence.
3. If the pages do not explicitly state a field's value, set BOTH value and evidence to null.
4. NEVER fill in fields from your own knowledge — only from the PDF content.
5. Extract ALL diseases found on these pages. If a disease matches one in the list above, use the EXACT name from the list. If it doesn't match any name in the list, use the name as written in the PDF.

For each disease found, extract:

IDENTITY (for matching purposes):
- disease_name: Use the EXACT name from the disease list if it matches, otherwise use the PDF's name (REQUIRED)
- pathogen_scientific_name: null (already known from expert sheet, skip)
- type_of_disease: null (already known from expert sheet, skip)
- affected_parts: list from [Foliar, Stem, Root, Seed, Pod, Vascular, Whole plant]

VISUAL SYMPTOMS — THIS IS THE PRIMARY GOAL:
- summary: 1-2 sentence overview of visual symptoms
- diagnostic_features: what visually distinguishes THIS disease from similar ones
- look_alikes: other diseases this could be confused with

SOURCE: pdf://{pdf_name}

Return the extraction as JSON matching the required schema.
"""


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "extractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_url": {"type": "string"},
                    "source_title": {"type": "string"},
                    "source_type": {"type": "string"},
                    "access_date": {"type": "string"},
                    "extracted_diseases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "disease_name": _EVIDENCE_FIELD,
                                "pathogen_scientific_name": _EVIDENCE_FIELD,
                                "type_of_disease": _EVIDENCE_FIELD,
                                "affected_parts": _EVIDENCE_FIELD_ARRAY,
                                "visual_symptoms": _VISUAL_SYMPTOMS_SCHEMA,
                            },
                            "required": [
                                "disease_name",
                                "pathogen_scientific_name",
                                "type_of_disease",
                                "affected_parts",
                                "visual_symptoms",
                            ],
                        },
                    },
                },
                "required": [
                    "source_url",
                    "source_title",
                    "source_type",
                    "access_date",
                    "extracted_diseases",
                ],
            },
        }
    },
    "required": ["extractions"],
}
