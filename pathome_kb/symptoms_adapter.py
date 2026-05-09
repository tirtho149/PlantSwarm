"""
pathome_kb/symptoms_adapter.py
==============================
Convert SAGE-style ``final_registry.json`` records and per-state
``regional_observations.json`` blocks into the JSON shape consumed by
``pathome.SymptomLibrary.load``.

Two outputs per profile:
- ``canonical`` (cross-region) ← from ``final_registry.json``
- ``regional_observations[state]`` ← from per-state VLM observations

The previous regional_text duplication stage has been retired; the
adapter no longer handles ``regional_visuals`` or ``visual`` schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils import save_json  # noqa: F401


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _val(field: Any) -> Optional[Any]:
    """Pull the ``value`` out of a SAGE cited field, or return None."""
    if not isinstance(field, dict):
        return field if field else None
    v = field.get("value")
    if v in (None, "", []):
        return None
    return v


def _strs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(x) for x in value if x)
    return str(value).strip()


def _citation(field_dict: Any, image_id: str = "", grounding: str = "text") -> Optional[dict]:
    """Convert a SAGE/regional cited field into a Citation-shaped dict, or None."""
    if not isinstance(field_dict, dict):
        return None
    v = field_dict.get("value")
    if v in (None, "", []):
        return None
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v if x)
    url = (field_dict.get("url") or "").strip()
    quote = (field_dict.get("quote") or "").strip()
    if grounding == "text" and not (url or quote):
        return None
    out = {
        "value": str(v),
        "url": url,
        "quote": quote,
        "image_id": image_id,
        "grounding": grounding,
    }
    return out


# ---------------------------------------------------------------------------
# canonical (cross-region) record → CanonicalDisease dict
# ---------------------------------------------------------------------------

def disease_to_canonical_dict(record: dict) -> dict:
    """Map a SAGE ``final_registry.json`` entry to a ``canonical`` dict."""
    visual = record.get("visual_symptoms") or {}
    summary = visual.get("summary")
    diagnostic = visual.get("diagnostic_features")
    look = visual.get("look_alikes")

    # ``treatments`` is a new-schema field; tolerate its absence in older
    # final_registry.json files produced before the prompt extension.
    treatments = record.get("treatments") or {}

    pathogen = record.get("pathogen_scientific_name") or {}
    type_field = record.get("type_of_disease") or {}
    affected = record.get("affected_parts") or {}

    canonical = {
        "summary": _str(_val(summary)),
        "diagnostic_features": _strs(_val(diagnostic)),
        "look_alikes": _strs(_val(look)),
        "treatments": _strs(_val(treatments)),
        "affected_parts": _strs(_val(affected)),
        "pathogen_scientific_name": _str(_val(pathogen)),
        "type_of_disease": _str(_val(type_field)),
        "notes": "",
        "sources": {},
    }

    sources: Dict[str, List[dict]] = {}
    for key, src in (
        ("summary", summary),
        ("diagnostic_features", diagnostic),
        ("look_alikes", look),
        ("treatments", treatments),
        ("affected_parts", affected),
        ("pathogen_scientific_name", pathogen),
        ("type_of_disease", type_field),
    ):
        cit = _citation(src)
        if cit:
            sources.setdefault(key, []).append(cit)
    canonical["sources"] = sources
    return canonical


# ---------------------------------------------------------------------------
# regional record → RegionalObservation dict
# ---------------------------------------------------------------------------

def observation_to_regional_dict(state: str, record: dict) -> dict:
    """Map a per-state regional_observations.json record to the dict
    consumed by SymptomProfile.from_dict.

    The record is the JSON the VLM stage emits per (profile, state). It
    typically has fields:
      severity, lesion_morphology, affected_organs, spread_pattern,
      variations_from_canonical (list of bullets),
      __image_ids__ (list of bugwood::N).
    Each populated field becomes an image-grounded Citation tied to
    the primary image_id.
    """
    image_ids = list(record.get("__image_ids__") or record.get("image_ids") or [])
    primary = image_ids[0] if image_ids else ""

    severity = _str(record.get("severity"))
    morphology = _str(record.get("lesion_morphology"))
    organs = _strs(record.get("affected_organs"))
    spread = _str(record.get("spread_pattern"))
    variations = _strs(record.get("variations_from_canonical"))

    sources: Dict[str, List[dict]] = {}

    def _vlm_cite(field_value: Any, quote: str) -> Optional[dict]:
        if field_value in (None, "", []):
            return None
        if isinstance(field_value, list):
            v = "; ".join(str(x) for x in field_value if x)
        else:
            v = str(field_value)
        return {
            "value": v,
            "url": "",
            "quote": str(quote or ""),
            "image_id": primary,
            "grounding": "image",
        }

    # Each field's "quote" lives on the structured record itself in the
    # newer prompt — record["severity_quote"] etc. Tolerate either a flat
    # string OR a {value, quote} sub-object per field.
    def _field_pair(key: str) -> Tuple[Any, str]:
        v = record.get(key)
        q_key = f"{key}_quote"
        q = record.get(q_key, "")
        if isinstance(v, dict):
            q = v.get("quote", q)
            v = v.get("value")
        return v, q

    pairs = {
        "severity": _field_pair("severity"),
        "lesion_morphology": _field_pair("lesion_morphology"),
        "affected_organs": _field_pair("affected_organs"),
        "spread_pattern": _field_pair("spread_pattern"),
        "variations_from_canonical": _field_pair("variations_from_canonical"),
    }
    for k, (val, quote) in pairs.items():
        cit = _vlm_cite(val, quote)
        if cit:
            sources.setdefault(k, []).append(cit)

    return {
        "state": state,
        "image_ids": image_ids,
        "severity": severity,
        "lesion_morphology": morphology,
        "affected_organs": organs,
        "spread_pattern": spread,
        "variations_from_canonical": variations,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------

def disease_to_profile_dict(crop: str, disease: str, record: dict) -> dict:
    """One SAGE registry entry → SymptomProfile JSON dict (canonical only)."""
    canonical = disease_to_canonical_dict(record)
    return {
        "profile_id": f"{crop}::{disease}",
        "crop": crop,
        "disease": disease,
        "canonical": canonical,
        "regional_observations": {},
        "state_counts": {},
        "aez_counts": {},
        "total_observations": 0,
        "reference_ids": [],
        "reobservation_prompt": "",
        "swarm_observations": None,
    }


# ---------------------------------------------------------------------------
# Top-level merge
# ---------------------------------------------------------------------------

def merge_registries_to_seed(
    registries: Iterable[Tuple[str, dict]],
    expected_classes: Iterable[Tuple[str, str]],
    regional_observations_by_crop: Optional[Dict[str, Dict[str, Dict[str, dict]]]] = None,
    min_observations: int = 3,
) -> dict:
    """Merge canonical (cross-region) + regional observations into seed JSON.

    ``registries`` is a list of (crop, registry_dict) pairs from the
    SAGE cross-region pipeline; each registry has a ``diseases`` list.

    ``regional_observations_by_crop`` maps:
        crop -> profile_id ("Crop::Disease") -> state -> observation_record
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

    regional_observations_by_crop = regional_observations_by_crop or {}

    profiles = []
    for crop, disease in expected_classes:
        record = by_crop_disease.get((crop, disease)) or {}
        prof = disease_to_profile_dict(crop, disease, record)

        crop_regional = regional_observations_by_crop.get(crop) or {}
        per_profile = crop_regional.get(prof["profile_id"]) or {}
        if per_profile:
            prof["regional_observations"] = {
                state: observation_to_regional_dict(state, rec)
                for state, rec in per_profile.items()
                if isinstance(rec, dict)
            }
        profiles.append(prof)

    return {
        "min_observations": min_observations,
        "profiles": profiles,
    }


def write_seed_json(payload: dict, path: str | Path) -> Path:
    """Persist the merged seed payload to disk and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return p
