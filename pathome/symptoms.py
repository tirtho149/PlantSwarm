"""
pathome/symptoms.py
===================
Visual-symptom-centric core of PathomeDB.

Replaces the older Layer 1 (mechanistic pathways) / Layer 2 (cross-crop
manifestation) / Layer 3 (regional epidemiology) / Layer 4 (decision graph)
split. Everything Pathome needs to ground PlantSwarm/OBSERVE in is folded
into one structure per (crop, disease):

    SymptomProfile
      ├─ visual: VisualSymptom         what the disease LOOKS like
      ├─ state_counts: {state: n}      where it has been observed (geo-aware)
      ├─ aez_counts:   {aez:   n}      AEZ rollup of the same
      ├─ reference_ids: [ref_id, ...]  pointer into ReferenceLibrary
      └─ reobservation_prompt: str     what to re-examine on low-conf backtrack

The visual block is the *primary* knowledge unit. The geo block keeps the
"geo-aware" property the older Layer 3 was supposed to provide, but at the
resolution the Bugwood CSV actually offers (US state, with AEZ rolled up
from the state centroid). No month axis — the CSV has no capture date.

Visual features can be hand-curated via a JSON sidecar (``symptoms_path``
on ``PathomeDB.build_from_bugwood``); state/AEZ counts and reference IDs
are auto-derived from BugwoodRecords at build time.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Visual block
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A single (value, url, quote) record supporting one VisualSymptom field.

    Mirrors the SAGE/disease_registry provenance schema: the URL and the
    verbatim quote from that URL that supports the field's value. ``image_id``
    optionally ties the citation to a Bugwood reference image (``bugwood::N``)
    so downstream consumers can show the supporting field photograph next to
    the source quote.

    ``grounding`` distinguishes how the citation was produced:
        "text"  — verbatim quote from a source URL (extension factsheet etc.)
                  Default. Always carries (url, quote).
        "image" — VLM observation grounded in a specific Bugwood image. The
                  image_id is the primary witness; quote is a model-generated
                  description; url may be empty.
    """

    value: str = ""              # the extracted fact (string or "; "-joined list)
    url: str = ""                # source page / pdf://...
    quote: str = ""              # verbatim sentence supporting `value` (text) or model description (image)
    image_id: str = ""           # optional: Bugwood image ID grounding this citation
    grounding: str = "text"      # "text" | "image"


@dataclass
class CanonicalDisease:
    """Canonical (cross-region) knowledge about one (crop, disease).

    Built ONCE per disease from web research (claude -p WebSearch +
    extension-service URL extraction + reconciliation). All fields are
    text-grounded with verbatim quotes from source URLs; ``sources``
    keys mirror the field names so every fact can be traced back.

    Field choice mirrors what extension-service / APS / CABI factsheets
    publish: a one-paragraph summary, the diagnostic features a field
    diagnostician should look for, look-alike confusions, the recommended
    treatment(s), affected plant parts, and basic taxonomy.
    """

    summary: str = ""                                            # one-paragraph cross-region overview
    diagnostic_features: List[str] = field(default_factory=list) # what to look for to confirm
    look_alikes: List[str] = field(default_factory=list)         # diseases easily confused with this one
    treatments: List[str] = field(default_factory=list)          # management / control measures
    affected_parts: List[str] = field(default_factory=list)      # leaf, stem, fruit, root, flower, ...
    pathogen_scientific_name: str = ""                           # e.g. "Alternaria tomatophila"
    type_of_disease: str = ""                                    # Fungal | Bacterial | Viral | Oomycete | ...
    notes: str = ""
    sources: Dict[str, List[Citation]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any([
            self.summary, self.diagnostic_features, self.look_alikes,
            self.treatments, self.affected_parts,
            self.pathogen_scientific_name, self.type_of_disease, self.notes,
        ])

    def auto_reobservation_prompt(self) -> str:
        """Re-observation prompt for low-confidence routing.

        Drawn from diagnostic_features and affected_parts when populated.
        """
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


@dataclass
class RegionalDelta:
    """One image-grounded observation that ADDS to or CONTRADICTS the
    canonical KB. The set of these per state is the regional KB.
    Re-statements of canonical text are forbidden by construction —
    canonical is the source of truth for fields the VLM doesn't touch.
    """

    field: str = ""             # which canonical field this delta refines
    canonical_says: str = ""    # short quote from canonical, or "(not specified)"
    image_shows: str = ""       # what the image adds or contradicts
    image_quote: str = ""       # one-sentence visual evidence from the image
    image_id: str = ""          # bugwood::N — which image was the witness


@dataclass
class RegionalObservation:
    """Per-(crop, disease, state) deltas vs canonical.

    The VLM walks the canonical KB like a decision tree and emits ONLY
    the deltas — additions or contradictions backed by the Bugwood
    photo. There is no parallel re-extraction of severity / morphology /
    etc; canonical owns those slots, regional owns the deltas. If the
    image confirms canonical exactly, ``deltas`` is empty.
    """

    state: str = ""
    image_ids: List[str] = field(default_factory=list)
    deltas: List[RegionalDelta] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.deltas

    def deltas_by_field(self) -> Dict[str, List[RegionalDelta]]:
        """Group deltas by which canonical field they refine."""
        out: Dict[str, List[RegionalDelta]] = {}
        for d in self.deltas:
            out.setdefault(d.field or "other", []).append(d)
        return out

    def narrative(self, max_quote_chars: int = 240) -> str:
        """Render the per-state deltas as a prompt-ready text block.

        One bullet per delta, naming the canonical field, what canonical
        says, what THIS image adds/contradicts, and a trimmed visual
        quote. Empty if the image confirms canonical exactly.
        """
        if not self.deltas:
            return ""
        lines = [f"State: {self.state}"]
        for d in self.deltas:
            lines.append(f"  • [{d.field or 'other'}]")
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
# Swarm-derived enhancement block (filled by enhance_pathome_from_traces.py)
# ---------------------------------------------------------------------------

@dataclass
class SwarmObservations:
    """Per-(crop, disease) aggregates from PlantSwarm trace generation.

    Populated after trace generation by mining the routing-trace JSONL files.
    Lets the "before" (Claude-seed-only) and "after" (seed + traces) PathomeDB
    versions report what changed. All counts are over routing traces whose
    *ground-truth* class equals this profile's (crop, disease).
    """

    n_traces: int = 0
    avg_path_length: float = 0.0
    backtrack_rate: float = 0.0                                   # fraction of traces with >=1 backtrack
    high_confidence_rate: float = 0.0                             # fraction ending in confidence=high
    confusion_targets: Dict[str, int] = field(default_factory=dict)   # disease the swarm misroutes to → count
    common_lesion_terms: Dict[str, int] = field(default_factory=dict) # term → frequency in agent outputs
    common_signs: Dict[str, int] = field(default_factory=dict)        # signs/sporulation terms → frequency
    last_updated: str = ""                                            # ISO timestamp


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class SymptomProfile:
    """One profile per (crop, disease).

    Schema is split deliberately:
    - ``canonical`` is built once per disease from web research, has all
      text fields a field diagnostician needs (summary, diagnostic features,
      look-alikes, treatments, taxonomy). Sources are URL+verbatim quote.
    - ``regional_observations[state]`` is built per state from the
      Bugwood image(s) for that state. Stores ONLY visual phenotype
      observations and the deltas between what the image shows and what
      the canonical entry describes. Sources are image_id-grounded with
      no URL.

    This avoids the duplication that the previous flat ``visual`` +
    ``regional_visuals[state]`` schema produced (where each per-state
    block was largely a copy of the cross-region text).
    """

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
    swarm_observations: Optional[SwarmObservations] = None

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
        d = asdict(self)
        return d

    def context_for_state(self, state: str = "") -> str:
        """Render canonical KB + per-state deltas as a single prompt block.

        Used by routing/diagnosis agents that want the full decision-tree
        view for a (crop, disease, state) tuple: canonical is the trunk,
        deltas for the requested state are the branches. If ``state`` is
        empty or has no observations, only canonical is rendered.
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
        sources = cls._hydrate_sources(c.get("sources"))
        return CanonicalDisease(
            summary=str(c.get("summary", "")),
            diagnostic_features=list(c.get("diagnostic_features") or []),
            look_alikes=list(c.get("look_alikes") or []),
            treatments=list(c.get("treatments") or []),
            affected_parts=list(c.get("affected_parts") or []),
            pathogen_scientific_name=str(c.get("pathogen_scientific_name", "")),
            type_of_disease=str(c.get("type_of_disease", "")),
            notes=str(c.get("notes", "")),
            sources=sources,
        )

    @classmethod
    def _hydrate_regional(cls, state: str, raw: Optional[dict]) -> RegionalObservation:
        r = dict(raw or {})
        deltas: List[RegionalDelta] = []
        for d in (r.get("deltas") or []):
            if not isinstance(d, dict):
                continue
            deltas.append(RegionalDelta(
                field=str(d.get("field", "")),
                canonical_says=str(d.get("canonical_says", "")),
                image_shows=str(d.get("image_shows", "")),
                image_quote=str(d.get("image_quote", "")),
                image_id=str(d.get("image_id", "")),
            ))
        return RegionalObservation(
            state=str(r.get("state") or state),
            image_ids=list(r.get("image_ids") or []),
            deltas=deltas,
        )

    @classmethod
    def _legacy_visual_to_canonical(cls, raw: dict) -> CanonicalDisease:
        """Read OLD ``visual`` blob (plant_parts/distinctive_signs/notes/...)
        as a CanonicalDisease so older symptoms_seed.json files still load."""
        return CanonicalDisease(
            summary=str(raw.get("notes", "")),
            diagnostic_features=list(raw.get("distinctive_signs") or []),
            look_alikes=list(raw.get("confusion_diseases") or []),
            treatments=[],
            affected_parts=list(raw.get("plant_parts") or []),
            pathogen_scientific_name="",
            type_of_disease="",
            notes="",
            sources=cls._hydrate_sources(raw.get("sources")),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SymptomProfile":
        # Canonical: prefer the new "canonical" key; fall back to legacy "visual".
        if isinstance(d.get("canonical"), dict):
            canonical = cls._hydrate_canonical(d["canonical"])
        elif isinstance(d.get("visual"), dict):
            canonical = cls._legacy_visual_to_canonical(d["visual"])
        else:
            canonical = CanonicalDisease()

        # Regional: prefer the new "regional_observations"; fall back to legacy
        # "regional_visuals" (which had the same shape as visual).
        regional: Dict[str, RegionalObservation] = {}
        if isinstance(d.get("regional_observations"), dict):
            regional = {
                state: cls._hydrate_regional(state, blob)
                for state, blob in d["regional_observations"].items()
                if isinstance(blob, dict)
            }
        elif isinstance(d.get("regional_visuals"), dict):
            # Legacy regional_visuals blocks predate the deltas schema;
            # keep image_ids only (no synthesized deltas — there's no
            # canonical context to compare against here).
            for state, blob in d["regional_visuals"].items():
                if not isinstance(blob, dict):
                    continue
                regional[state] = RegionalObservation(
                    state=state,
                    image_ids=list(blob.get("reference_image_ids") or []),
                )

        sw_raw = d.get("swarm_observations")
        sw = SwarmObservations(**sw_raw) if isinstance(sw_raw, dict) else None
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
            swarm_observations=sw,
        )


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

class SymptomLibrary:
    """All SymptomProfiles, indexed by (crop, disease) and by disease alone.

    Geo-aware queries (P(disease | state) and ranked-by-state) come straight
    out of the per-profile ``state_counts``: there is no separate Layer 3
    table to maintain. ``min_observations`` controls when a state is treated
    as "dense enough" to compute a confident prior — below the threshold,
    callers should fall back to the global prior.
    """

    def __init__(self, min_observations: int = 3):
        self.min_observations = min_observations
        self._profiles: Dict[str, SymptomProfile] = {}                # profile_id -> profile
        self._by_disease: Dict[str, List[str]] = defaultdict(list)    # disease -> [profile_id]

    # -- mutate ---------------------------------------------------------

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

    def update_from_records(
        self,
        observation_records: Iterable,
        reference_records: Iterable = (),
    ) -> None:
        """Bulk-populate state/AEZ counts and reference IDs from BugwoodRecords.

        ``observation_records`` should be the trace split (drives the geo
        prior). ``reference_records`` is the held-out reference split (their
        IDs are stored on the profiles for downstream Layer-5 retrieval).
        """
        for r in observation_records:
            crop = getattr(r, "crop_species", None)
            disease = getattr(r, "disease_name", None)
            if not crop or not disease:
                continue
            prof = self.get_or_create(crop, disease)
            state = (getattr(r, "meta", {}) or {}).get("state")
            prof.add_observation(state, getattr(r, "aez_code", None))
        for r in reference_records:
            crop = getattr(r, "crop_species", None)
            disease = getattr(r, "disease_name", None)
            if not crop or not disease:
                continue
            prof = self.get_or_create(crop, disease)
            prof.add_reference(getattr(r, "image_id", ""))

    def finalize_reobservation_prompts(self) -> None:
        """Populate auto re-observation prompts where the curator left them blank."""
        for prof in self._profiles.values():
            if not prof.reobservation_prompt:
                prof.reobservation_prompt = prof.canonical.auto_reobservation_prompt()

    # -- query ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self):
        return iter(self._profiles.values())

    def get(self, crop: str, disease: str) -> Optional[SymptomProfile]:
        return self._profiles.get(SymptomProfile.make_id(crop, disease))

    def find_by_disease(self, disease: str) -> List[SymptomProfile]:
        return [self._profiles[pid] for pid in self._by_disease.get(disease, [])]

    def geo_prior(self, disease: str, state: Optional[str]) -> Optional[float]:
        """P(disease | state) computed across crops.

        Returns ``None`` when the state has < ``min_observations`` total
        records of the disease (sparse cell — caller should use
        ``global_prior``).
        """
        if not state:
            return None
        profs = self.find_by_disease(disease)
        if not profs:
            return None
        cell_count = sum(p.state_counts.get(state, 0) for p in profs)
        if cell_count < self.min_observations:
            return None
        # P(d|s) needs the marginal of any disease in that state.
        state_total = 0
        for p in self._profiles.values():
            state_total += p.state_counts.get(state, 0)
        if state_total == 0:
            return None
        return cell_count / state_total

    def global_prior(self, disease: str) -> float:
        profs = self.find_by_disease(disease)
        if not profs:
            return 0.0
        d_count = sum(p.total_observations for p in profs)
        total = sum(p.total_observations for p in self._profiles.values())
        return d_count / total if total else 0.0

    def is_sparse_in_state(self, disease: str, state: Optional[str]) -> bool:
        if not state:
            return True
        profs = self.find_by_disease(disease)
        cell_count = sum(p.state_counts.get(state, 0) for p in profs)
        return cell_count < self.min_observations

    def prevalent_in_state(self, state: str, top_k: int = 5) -> List[Tuple[str, int]]:
        """Most-observed diseases in a given state. Ties broken alphabetically."""
        counts: Counter = Counter()
        for prof in self._profiles.values():
            n = prof.state_counts.get(state, 0)
            if n:
                counts[prof.disease] += n
        return counts.most_common(top_k)

    def reobservation_prompt(self, crop: str, disease: str, default: str = "") -> str:
        prof = self.get(crop, disease)
        if prof is None:
            return default
        return prof.reobservation_prompt or prof.canonical.auto_reobservation_prompt() or default

    def context_for(self, crop: str, disease: str, state: str = "") -> str:
        """Canonical KB + per-state deltas as one prompt block.

        Empty string if the (crop, disease) profile is unknown.
        """
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
