"""
observe/prototypes.py
=====================
Build per-class text prototypes from PathomeDB seed JSON.

Each prototype is a 1-3 sentence description suitable for the SigLIP-2
text tower. The structure leans on every signal the KB carries:

  "A field photograph of {crop} affected by {disease} ({pathogen}, {type}).
   {canonical.summary}
   Diagnostic features: {diagnostic_features}.
   Look-alikes: {look_alikes}.
   Affected parts: {affected_parts}.
   Regional observations: {top-K regional deltas across states}."

For 'healthy' classes (which PathomeDB doesn't cover — Bugwood is
disease-only) a synthetic template is used:

  "A healthy {crop} leaf with no visible disease symptoms — uniform
   green color, no lesions, no spots, no wilting, no chlorosis."

Prototypes are stable, deterministic, and reproducible: same seed JSON
=> identical text strings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


HEALTHY_TEMPLATE = (
    "A healthy {crop} leaf with no visible disease symptoms — uniform "
    "green color, no lesions, no spots, no wilting, no chlorosis, no "
    "necrosis."
)


def _flatten_field(raw: Any) -> str:
    """Render a KB field value (str, list, dict-with-value) as plain text."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        raw = raw.get("value")
        if raw is None:
            return ""
    if isinstance(raw, list):
        return "; ".join(str(x) for x in raw if x is not None and str(x).strip())
    s = str(raw).strip()
    return s


def build_disease_prototype(
    *,
    crop: str,
    disease: str,
    canonical: Dict[str, Any],
    regional_observations: Optional[Dict[str, Any]] = None,
    top_k_regional: int = 3,
    max_chars: int = 1024,
) -> str:
    """Build a multi-sentence text prototype for one (crop, disease).

    ``canonical`` is the flattened SymptomProfile.canonical dict (already
    in plain-string shape). ``regional_observations`` is the
    SymptomProfile.regional_observations dict keyed by state — optional.
    """
    parts: List[str] = []

    # Lead sentence with identity.
    pathogen = _flatten_field(canonical.get("pathogen_scientific_name"))
    dtype    = _flatten_field(canonical.get("type_of_disease"))
    lead = f"A field photograph of {crop} affected by {disease}"
    if pathogen or dtype:
        tag = []
        if pathogen: tag.append(pathogen)
        if dtype:    tag.append(f"{dtype.lower()} disease")
        lead += " (" + ", ".join(tag) + ")"
    lead += "."
    parts.append(lead)

    # Canonical summary
    summary = _flatten_field(canonical.get("summary"))
    if summary:
        parts.append(summary if summary.endswith(".") else summary + ".")

    # Diagnostic features
    diag = _flatten_field(canonical.get("diagnostic_features"))
    if diag:
        parts.append(f"Diagnostic features: {diag}.")

    # Look-alikes
    la = _flatten_field(canonical.get("look_alikes"))
    if la:
        parts.append(f"May be confused with: {la}.")

    # Affected parts
    ap = _flatten_field(canonical.get("affected_parts"))
    if ap:
        parts.append(f"Affected parts: {ap}.")

    # Top-K regional deltas (high swarm_support + verified preferred)
    if regional_observations:
        deltas = _top_regional_deltas(regional_observations, top_k=top_k_regional)
        if deltas:
            parts.append(
                "Regional variations: " +
                "; ".join(deltas) +
                "."
            )

    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def build_healthy_prototype(crop: str) -> str:
    return HEALTHY_TEMPLATE.format(crop=crop)


def _top_regional_deltas(
    regional_observations: Dict[str, Any], top_k: int = 3,
) -> List[str]:
    """Pick the top K deltas across all states, preferring verified
    higher-support entries. Returns short phrases for prompt inclusion.
    """
    candidates: List[Dict[str, Any]] = []
    for state, ro in (regional_observations or {}).items():
        if not isinstance(ro, dict):
            continue
        for d in ro.get("deltas") or []:
            if not isinstance(d, dict):
                continue
            if not d.get("image_shows"):
                continue
            candidates.append({
                "state":       state,
                "image_shows": str(d.get("image_shows", "")).strip(),
                "field":       str(d.get("field", "other")),
                "support":     int(d.get("swarm_support") or d.get("support") or 0),
                "status":      str(d.get("verification_status") or "unverified"),
            })

    # Rank: verified > weakly_supported > others; then by support desc.
    rank = {"verified": 5, "weakly_supported": 4, "provisional": 3,
            "novel_plausible": 2, "unverified": 1, "contradictory": 0}
    candidates.sort(
        key=lambda d: (-rank.get(d["status"], 0), -d["support"]),
    )
    out: List[str] = []
    seen = set()
    for c in candidates:
        phrase = f"in {c['state']}, {c['image_shows'][:160]}"
        key = (c["field"], phrase[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Loader — convert symptoms_seed.json into a list of (label, prototype_text)
# ---------------------------------------------------------------------------

def load_seed_prototypes(
    seed_path: str | Path,
    *,
    crop_filter: Optional[str] = None,
    top_k_regional: int = 3,
) -> List[Dict[str, Any]]:
    """Load PathomeDB seed JSON and emit one prototype record per profile.

    Each record:
        {
          "label":     "<Crop>::<Disease>",
          "crop":      ...,
          "disease":   ...,
          "kind":      "disease",
          "prototype": "...",
        }
    """
    seed = json.loads(Path(seed_path).read_text())
    out: List[Dict[str, Any]] = []
    for p in seed.get("profiles") or []:
        crop    = p.get("crop") or ""
        disease = p.get("disease") or ""
        if crop_filter and crop != crop_filter:
            continue
        canonical = p.get("canonical") or {}
        if not (canonical.get("summary") or canonical.get("diagnostic_features")):
            # Skip profiles with no canonical content.
            continue
        regional = p.get("regional_observations") or {}
        out.append({
            "label":     f"{crop}::{disease}",
            "crop":      crop,
            "disease":   disease,
            "kind":      "disease",
            "prototype": build_disease_prototype(
                crop=crop, disease=disease,
                canonical=canonical,
                regional_observations=regional,
                top_k_regional=top_k_regional,
            ),
        })
    return out


def add_healthy_prototypes(
    records: List[Dict[str, Any]],
    crops: List[str],
) -> List[Dict[str, Any]]:
    """Append synthetic 'healthy' prototypes for the given crops."""
    for crop in crops:
        records.append({
            "label":     f"{crop}::healthy",
            "crop":      crop,
            "disease":   "healthy",
            "kind":      "healthy",
            "prototype": build_healthy_prototype(crop),
        })
    return records
