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
    """Map a per-state VLM observation into the dict shape consumed by
    SymptomProfile.from_dict.

    The record is the JSON the VLM stage emits per (profile, state).
    The new schema is a deltas-only list:
        deltas: [{field, canonical_says, image_shows, image_quote}]
    Plus ``__image_ids__`` for image grounding. Older records with
    parallel fields (severity / lesion_morphology / ...) are folded into
    deltas via ``_legacy_record_to_deltas`` so existing JSON still loads.
    """
    image_ids = list(record.get("__image_ids__") or record.get("image_ids") or [])
    primary = image_ids[0] if image_ids else ""

    raw_deltas = record.get("deltas")
    if not isinstance(raw_deltas, list):
        raw_deltas = _legacy_record_to_deltas(record)

    deltas: List[dict] = []
    for d in raw_deltas:
        if not isinstance(d, dict):
            continue
        if not d.get("image_shows"):
            continue
        deltas.append({
            "field":          _str(d.get("field")) or "other",
            "canonical_says": _str(d.get("canonical_says")) or "(not specified)",
            "image_shows":    _str(d.get("image_shows")),
            "image_quote":    _str(d.get("image_quote")),
            "image_id":       _str(d.get("image_id")) or primary,
        })

    return {
        "state": state,
        "image_ids": image_ids,
        "deltas": deltas,
    }


def _legacy_record_to_deltas(record: dict) -> List[dict]:
    """Promote an older parallel-fields record into the deltas schema so
    cached final_registry.json files continue to load.

    Only the ``variations_from_canonical`` bullets are mapped — the
    parallel severity/morphology/etc fields are dropped because they
    duplicate canonical and that's the duplication this rewrite removes.
    """
    out: List[dict] = []
    for bullet in (record.get("variations_from_canonical") or []):
        if not isinstance(bullet, str) or not bullet.strip():
            continue
        out.append({
            "field": "other",
            "canonical_says": "(legacy bullet — canonical context not preserved)",
            "image_shows": bullet.strip(),
            "image_quote": "",
        })
    return out


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

    ``registries`` is a list of (crop, registry_dict) pairs. Each disease
    entry's ``regional_observations`` field (if present) is the
    canonical home for the per-state observations — the regional pass
    embeds them there. ``regional_observations_by_crop`` is still
    accepted as a fallback for callers that pass observations
    out-of-band, but is no longer required.
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

        # Prefer the embedded regional_observations on the disease entry.
        per_profile = record.get("regional_observations") or {}
        if not per_profile:
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
