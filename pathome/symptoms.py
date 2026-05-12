"""
pathome/symptoms.py
===================
Schema definitions for the PathomeDB seed JSON.

After the Phase 1-5 retirement, ``pathome`` is purely the schema layer
for the JSON that ``pathome_kb`` produces. Each ``SymptomProfile`` has:

    SymptomProfile
      ├─ canonical: CanonicalDisease         text-grounded, URL + verbatim
      │                                       quote per field (Phase 0)
      ├─ regional_observations[state]:        image-grounded deltas
      │     RegionalObservation              (Phase 0R Qwen swarm)
      │       ├─ state                       US state name
      │       ├─ image_ids                   bugwood::N witnesses
      │       ├─ deltas: List[RegionalDelta] {field, canonical_says,
      │       │                               image_shows, image_quote,
      │       │                               image_id, support}
      │       └─ swarm_meta                  per-state telemetry from the
      │                                       N-trace swarm (paths,
      │                                       κ confidences, agreement
      │                                       counts) — opaque blob for
      │                                       downstream consumers
      ├─ state_counts / aez_counts            from the Bugwood CSV
      └─ reference_ids                        held-out witness images

Canonical owns the symptom slots; regional only emits deltas. No
parallel re-extraction. The geo prior (state_counts) and reference
holds (reference_ids) are stored verbatim from the CSV — no separate
Layer 3/5 tables.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Canonical (text-grounded, Phase 0)
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A single (value, url, quote) record supporting one CanonicalDisease field.

    Mirrors the SAGE/disease_registry provenance schema. ``grounding``
    distinguishes how the citation was produced:

        "text"  — verbatim quote from a source URL (Phase 0; default).
        "image" — VLM observation grounded in a Bugwood image (Phase 0R).
    """

    value: str = ""
    url: str = ""
    quote: str = ""
    image_id: str = ""
    grounding: str = "text"


@dataclass
class CanonicalDisease:
    """Canonical (cross-region) knowledge for one (crop, disease).

    Built once per disease by ``pathome_kb`` via Claude discovery →
    extraction → reconciliation. All fields text-grounded with verbatim
    quotes from source URLs; ``sources`` keys mirror the field names so
    every fact can be traced back.
    """

    summary: str = ""
    diagnostic_features: List[str] = field(default_factory=list)
    look_alikes: List[str] = field(default_factory=list)
    treatments: List[str] = field(default_factory=list)
    affected_parts: List[str] = field(default_factory=list)
    pathogen_scientific_name: str = ""
    type_of_disease: str = ""
    notes: str = ""
    sources: Dict[str, List[Citation]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any([
            self.summary, self.diagnostic_features, self.look_alikes,
            self.treatments, self.affected_parts,
            self.pathogen_scientific_name, self.type_of_disease, self.notes,
        ])

    def auto_reobservation_prompt(self) -> str:
        """Re-observation cue rendered from diagnostic_features +
        affected_parts. Kept as a convenience for downstream prompts."""
        if self.is_empty():
            return ""
        bits: List[str] = []
        if self.diagnostic_features:
            bits.append(f"check for {', '.join(self.diagnostic_features[:3])}")
        if self.affected_parts:
            bits.append(f"focus on the {', '.join(self.affected_parts)}")
        if self.look_alikes:
            bits.append(f"distinguish from {', '.join(self.look_alikes[:3])}")
        return "; ".join(bits)


# ---------------------------------------------------------------------------
# Regional deltas (image-grounded, Phase 0R)
# ---------------------------------------------------------------------------

@dataclass
class RegionalDelta:
    """One image-grounded observation that ADDS to or CONTRADICTS the
    canonical KB. The set of these per state IS the regional KB.

    ``support`` is the K-of-N agreement count from the Qwen swarm — how
    many of the N stochastic traces produced a delta that clustered into
    this one. ``support`` accumulates across iterative Phase 0R re-runs
    via the conservative merge in ``plantswarm.delta_pipeline``.
    """

    field: str = ""             # which canonical field this delta refines
    canonical_says: str = ""    # short quote from canonical, or "(not specified)"
    image_shows: str = ""       # what the image adds or contradicts
    image_quote: str = ""       # one-sentence visual evidence
    image_id: str = ""          # bugwood::N — primary witness
    support: int = 0            # agreement count (0 = legacy / pre-swarm)


@dataclass
class RegionalObservation:
    """Per-(crop, disease, state) deltas vs canonical.

    The Qwen swarm walks the canonical KB like a decision tree and emits
    ONLY the deltas — additions or contradictions backed by the Bugwood
    photo. If the image confirms canonical exactly, ``deltas`` is empty.

    ``swarm_meta`` is an opaque per-state telemetry blob (trace paths,
    κ confidences, backtrack counts, raw-per-run counts, merge stats).
    Downstream consumers can ignore it; the registry preserves it for
    audit and tuning.
    """

    state: str = ""
    image_ids: List[str] = field(default_factory=list)
    deltas: List[RegionalDelta] = field(default_factory=list)
    swarm_meta: Optional[Dict[str, Any]] = None

    def is_empty(self) -> bool:
        return not self.deltas

    def deltas_by_field(self) -> Dict[str, List[RegionalDelta]]:
        out: Dict[str, List[RegionalDelta]] = {}
        for d in self.deltas:
            out.setdefault(d.field or "other", []).append(d)
        return out

    def narrative(self, max_quote_chars: int = 240) -> str:
        """Render the per-state deltas as a prompt-ready text block."""
        if not self.deltas:
            return ""
        lines = [f"State: {self.state}"]
        for d in self.deltas:
            lines.append(f"  • [{d.field or 'other'}]"
                         + (f" (support={d.support})" if d.support else ""))
            if d.canonical_says:
                lines.append(f"      canonical: {d.canonical_says}")
            if d.image_shows:
                lines.append(f"      image:     {d.image_shows}")
            if d.image_quote:
                q = d.image_quote
                if len(q) > max_quote_chars:
                    q = q[:max_quote_chars].rstrip() + "…"
                lines.append(f"      evidence:  \"{q}\"")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class SymptomProfile:
    """One profile per (crop, disease) — canonical + regional deltas."""

    profile_id: str
    crop: str
    disease: str
    canonical: CanonicalDisease = field(default_factory=CanonicalDisease)
    regional_observations: Dict[str, RegionalObservation] = field(default_factory=dict)
    state_counts: Dict[str, int] = field(default_factory=dict)
    aez_counts: Dict[str, int] = field(default_factory=dict)
    total_observations: int = 0
    reference_ids: List[str] = field(default_factory=list)
    reobservation_prompt: str = ""

    @staticmethod
    def make_id(crop: str, disease: str) -> str:
        return f"{crop}::{disease}"

    def add_observation(self, state: Optional[str], aez_code: Optional[str]) -> None:
        if state:
            self.state_counts[state] = self.state_counts.get(state, 0) + 1
        if aez_code:
            self.aez_counts[aez_code] = self.aez_counts.get(aez_code, 0) + 1
        self.total_observations += 1

    def add_reference(self, ref_id: str) -> None:
        if ref_id and ref_id not in self.reference_ids:
            self.reference_ids.append(ref_id)

    def fraction_in_state(self, state: str) -> float:
        if not state or self.total_observations == 0:
            return 0.0
        return self.state_counts.get(state, 0) / self.total_observations

    def to_dict(self) -> dict:
        return asdict(self)

    def context_for_state(self, state: str = "") -> str:
        """Canonical KB + per-state deltas as one prompt-ready text block.

        If ``state`` is empty or has no observations, only canonical is
        rendered.
        """
        lines: List[str] = [f"Crop:    {self.crop}", f"Disease: {self.disease}"]
        c = self.canonical
        if c.pathogen_scientific_name:
            lines.append(f"Pathogen: {c.pathogen_scientific_name}")
        if c.type_of_disease:
            lines.append(f"Type: {c.type_of_disease}")
        if c.affected_parts:
            lines.append(f"Affected parts: {', '.join(c.affected_parts)}")
        if c.summary:
            lines.append(f"Summary: {c.summary}")
        if c.diagnostic_features:
            lines.append("Diagnostic features:")
            lines.extend(f"  - {f}" for f in c.diagnostic_features)
        if c.look_alikes:
            lines.append("Look-alikes:")
            lines.extend(f"  - {f}" for f in c.look_alikes)
        if c.treatments:
            lines.append("Treatments:")
            lines.extend(f"  - {t}" for t in c.treatments)
        if state:
            obs = self.regional_observations.get(state)
            if obs and not obs.is_empty():
                lines.append("")
                lines.append(f"State-specific deltas for {state}:")
                lines.append(obs.narrative())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Hydration (JSON → dataclass)
    # ------------------------------------------------------------------

    @staticmethod
    def _hydrate_sources(raw: Optional[dict]) -> Dict[str, List[Citation]]:
        out: Dict[str, List[Citation]] = {}
        for k, items in (raw or {}).items():
            if not isinstance(items, list):
                continue
            out[k] = [
                Citation(
                    value=str(it.get("value", "")),
                    url=str(it.get("url", "")),
                    quote=str(it.get("quote", "")),
                    image_id=str(it.get("image_id", "")),
                    grounding=str(it.get("grounding", "text")) or "text",
                )
                for it in items if isinstance(it, dict)
            ]
        return out

    @classmethod
    def _hydrate_canonical(cls, raw: Optional[dict]) -> CanonicalDisease:
        c = dict(raw or {})
        return CanonicalDisease(
            summary=str(c.get("summary", "")),
            diagnostic_features=list(c.get("diagnostic_features") or []),
            look_alikes=list(c.get("look_alikes") or []),
            treatments=list(c.get("treatments") or []),
            affected_parts=list(c.get("affected_parts") or []),
            pathogen_scientific_name=str(c.get("pathogen_scientific_name", "")),
            type_of_disease=str(c.get("type_of_disease", "")),
            notes=str(c.get("notes", "")),
            sources=cls._hydrate_sources(c.get("sources")),
        )

    @classmethod
    def _hydrate_regional(cls, state: str, raw: Optional[dict]) -> RegionalObservation:
        r = dict(raw or {})
        deltas: List[RegionalDelta] = []
        for d in (r.get("deltas") or []):
            if not isinstance(d, dict):
                continue
            try:
                support = int(d.get("support", d.get("__support__", 0)) or 0)
            except (TypeError, ValueError):
                support = 0
            deltas.append(RegionalDelta(
                field=str(d.get("field", "")),
                canonical_says=str(d.get("canonical_says", "")),
                image_shows=str(d.get("image_shows", "")),
                image_quote=str(d.get("image_quote", "")),
                image_id=str(d.get("image_id", "")),
                support=support,
            ))
        swarm_meta = r.get("swarm_meta") or r.get("__swarm_meta__")
        if not isinstance(swarm_meta, dict):
            swarm_meta = None
        return RegionalObservation(
            state=str(r.get("state") or state),
            image_ids=list(r.get("image_ids") or []),
            deltas=deltas,
            swarm_meta=swarm_meta,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SymptomProfile":
        canonical = cls._hydrate_canonical(
            d.get("canonical") if isinstance(d.get("canonical"), dict) else {}
        )
        regional: Dict[str, RegionalObservation] = {}
        ro_raw = d.get("regional_observations")
        if isinstance(ro_raw, dict):
            regional = {
                state: cls._hydrate_regional(state, blob)
                for state, blob in ro_raw.items()
                if isinstance(blob, dict)
            }
        return cls(
            profile_id=d["profile_id"],
            crop=d["crop"],
            disease=d["disease"],
            canonical=canonical,
            regional_observations=regional,
            state_counts=dict(d.get("state_counts") or {}),
            aez_counts=dict(d.get("aez_counts") or {}),
            total_observations=int(d.get("total_observations", 0)),
            reference_ids=list(d.get("reference_ids") or []),
            reobservation_prompt=str(d.get("reobservation_prompt") or ""),
        )


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class SymptomLibrary:
    """In-memory collection of SymptomProfiles loaded from the seed JSON.

    Simple key-value store keyed by ``profile_id = "<Crop>::<Disease>"``;
    callers query via ``get(crop, disease)`` or iterate.
    """

    def __init__(self, min_observations: int = 3):
        self.min_observations = min_observations
        self._profiles: Dict[str, SymptomProfile] = {}
        self._by_disease: Dict[str, List[str]] = defaultdict(list)

    def get_or_create(self, crop: str, disease: str) -> SymptomProfile:
        pid = SymptomProfile.make_id(crop, disease)
        prof = self._profiles.get(pid)
        if prof is None:
            prof = SymptomProfile(profile_id=pid, crop=crop, disease=disease)
            self._profiles[pid] = prof
            self._by_disease[disease].append(pid)
        return prof

    def add_or_update(self, profile: SymptomProfile) -> None:
        existing = self._profiles.get(profile.profile_id)
        if existing is None:
            self._profiles[profile.profile_id] = profile
            self._by_disease[profile.disease].append(profile.profile_id)
            return
        # Merge: keep curated canonical/regional, accumulate counts + refs.
        if not existing.canonical.is_empty():
            profile.canonical = existing.canonical
        if existing.regional_observations:
            for state, obs in existing.regional_observations.items():
                profile.regional_observations.setdefault(state, obs)
        for s, n in existing.state_counts.items():
            profile.state_counts[s] = profile.state_counts.get(s, 0) + n
        for a, n in existing.aez_counts.items():
            profile.aez_counts[a] = profile.aez_counts.get(a, 0) + n
        profile.total_observations += existing.total_observations
        for rid in existing.reference_ids:
            if rid not in profile.reference_ids:
                profile.reference_ids.append(rid)
        if not profile.reobservation_prompt and existing.reobservation_prompt:
            profile.reobservation_prompt = existing.reobservation_prompt
        self._profiles[profile.profile_id] = profile

    # -- query ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterable[SymptomProfile]:
        return iter(self._profiles.values())

    def get(self, crop: str, disease: str) -> Optional[SymptomProfile]:
        return self._profiles.get(SymptomProfile.make_id(crop, disease))

    def find_by_disease(self, disease: str) -> List[SymptomProfile]:
        return [self._profiles[pid] for pid in self._by_disease.get(disease, [])]

    def reobservation_prompt(self, crop: str, disease: str, default: str = "") -> str:
        prof = self.get(crop, disease)
        if prof is None:
            return default
        return prof.reobservation_prompt or prof.canonical.auto_reobservation_prompt() or default

    def context_for(self, crop: str, disease: str, state: str = "") -> str:
        prof = self.get(crop, disease)
        if prof is None:
            return ""
        return prof.context_for_state(state)

    # -- persistence ----------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_observations": self.min_observations,
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SymptomLibrary":
        with open(path) as fh:
            payload = json.load(fh)
        lib = cls(min_observations=int(payload.get("min_observations", 3)))
        for d in payload.get("profiles", []):
            prof = SymptomProfile.from_dict(d)
            lib._profiles[prof.profile_id] = prof
            lib._by_disease[prof.disease].append(prof.profile_id)
        return lib
