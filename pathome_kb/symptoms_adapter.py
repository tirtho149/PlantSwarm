"""
pathome_kb/symptoms_adapter.py
==============================
Convert SAGE-style ``final_registry.json`` records into the JSON format
``pathome.SymptomLibrary.load`` consumes.

SAGE registry shape (per disease):
    {
      "disease_name":             "Early Blight",
      "pathogen_scientific_name": {value, url, quote},
      "type_of_disease":          {value, url, quote},
      "affected_parts":           {value: [...], url, quote},
      "visual_symptoms": {
        "summary":             {value, url, quote},
        "diagnostic_features": {value, url, quote},
        "look_alikes":         {value: [...], url, quote},
      },
      "confidence": "high"|"medium"|"low",
      "num_sources": int,
      "conflicts": [...]
    }

Adapter mapping:
- ``affected_parts.value``                 → ``VisualSymptom.plant_parts``
- ``visual_symptoms.diagnostic_features``  → ``VisualSymptom.distinctive_signs``
- ``visual_symptoms.look_alikes.value``    → ``VisualSymptom.confusion_diseases``
- ``visual_symptoms.summary.value``        → ``VisualSymptom.notes``
- The structured tuples (color, shape, margin, texture, sporulation,
  progression) are left empty by this pipeline because the SAGE pipeline
  emits free-form prose; downstream Phase 2 routing reads those fields
  off the auto-built reobservation_prompt rather than from prose, so
  empty fields are valid.
- Every populated field is accompanied by a ``Citation`` in
  ``VisualSymptom.sources[<field>]``.

The adapter never invents content — if the SAGE record is empty, the
SymptomProfile is created with an empty visual block and no citations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(field: Any) -> Optional[Any]:
    """Pull the ``value`` out of a SAGE cited field, or return None."""
    if not isinstance(field, dict):
        return field if field else None
    v = field.get("value")
    if v in (None, "", []):
        return None
    return v


def _citation_record(field: Any, key_for_value: str = "") -> Optional[dict]:
    """Convert a SAGE cited field into a Citation-ready dict, or None."""
    if not isinstance(field, dict):
        return None
    v = field.get("value")
    if v in (None, "", []):
        return None
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v if x)
    url = (field.get("url") or "").strip()
    quote = (field.get("quote") or "").strip()
    if not (url or quote):
        # nothing to cite
        return None
    return {"value": str(v), "url": url, "quote": quote}


def _strs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


# ---------------------------------------------------------------------------
# One disease record → one symptoms-profile-shaped dict
# ---------------------------------------------------------------------------

def disease_to_profile_dict(
    crop: str,
    disease: str,
    record: dict,
) -> dict:
    """Map a SAGE registry entry to the dict shape SymptomProfile.from_dict accepts."""
    visual_section = record.get("visual_symptoms") or {}

    plant_parts = _strs(_val(record.get("affected_parts")))
    distinctive_signs_raw = _val(visual_section.get("diagnostic_features"))
    distinctive_signs = _strs(distinctive_signs_raw)
    look_alikes = _strs(_val(visual_section.get("look_alikes")))
    summary = _val(visual_section.get("summary"))
    notes = str(summary) if summary else ""

    sources: Dict[str, List[dict]] = {}

    cit = _citation_record(record.get("affected_parts"))
    if cit and plant_parts:
        sources["plant_parts"] = [cit]

    cit = _citation_record(visual_section.get("diagnostic_features"))
    if cit and distinctive_signs:
        sources["distinctive_signs"] = [cit]

    cit = _citation_record(visual_section.get("look_alikes"))
    if cit and look_alikes:
        sources["confusion_diseases"] = [cit]

    cit = _citation_record(visual_section.get("summary"))
    if cit and notes:
        sources["notes"] = [cit]

    pathogen_cit = _citation_record(record.get("pathogen_scientific_name"))
    if pathogen_cit:
        sources.setdefault("pathogen_scientific_name", []).append(pathogen_cit)

    type_cit = _citation_record(record.get("type_of_disease"))
    if type_cit:
        sources.setdefault("type_of_disease", []).append(type_cit)

    visual = {
        "plant_parts": plant_parts,
        "color": [],
        "shape": "",
        "margin": "",
        "texture": [],
        "sporulation": [],
        "distinctive_signs": distinctive_signs,
        "progression": "",
        "confusion_diseases": look_alikes,
        "notes": notes,
        "sources": sources,
    }

    return {
        "profile_id": f"{crop}::{disease}",
        "crop": crop,
        "disease": disease,
        "visual": visual,
        "state_counts": {},
        "aez_counts": {},
        "total_observations": 0,
        "reference_ids": [],
        "reobservation_prompt": "",
        "swarm_observations": None,
    }


# ---------------------------------------------------------------------------
# Merge per-crop registries → SymptomLibrary seed JSON
# ---------------------------------------------------------------------------

def merge_registries_to_seed(
    registries: Iterable[Tuple[str, dict]],
    expected_classes: Iterable[Tuple[str, str]],
    min_observations: int = 3,
) -> dict:
    """Merge crop→registry pairs into the seed JSON.

    For every (crop, disease) in ``expected_classes`` we emit one
    SymptomProfile. If the registry has data, we use it; otherwise we
    emit an empty profile so the build pass picks it up.
    """
    by_crop_disease: Dict[Tuple[str, str], dict] = {}
    for crop, registry in registries:
        if not isinstance(registry, dict):
            continue
        for d in registry.get("diseases", []) or []:
            disease = (d.get("disease_name") or "").strip()
            if not disease:
                continue
            by_crop_disease[(crop, disease)] = d

    profiles = []
    for crop, disease in expected_classes:
        record = by_crop_disease.get((crop, disease)) or {}
        profiles.append(disease_to_profile_dict(crop, disease, record))

    return {
        "min_observations": min_observations,
        "profiles": profiles,
    }


def write_seed_json(payload: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return p
