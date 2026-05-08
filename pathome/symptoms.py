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
    verbatim quote from that URL that supports the field's value. Keeping
    citations on the profile lets the paper claim "every visual fact is
    traceable to a sourced extension-service or APS publication."
    """

    value: str = ""              # the extracted fact (string or "; "-joined list)
    url: str = ""                # source page / pdf://...
    quote: str = ""              # verbatim sentence supporting `value`


@dataclass
class VisualSymptom:
    """Structured description of what a disease looks like.

    All fields default to empty so a profile is valid even before a curator
    fills it in — geo + reference data still drive routing and retrieval in
    that case. Field names mirror the diagnostic vocabulary used by
    extension-service literature, not OBSERVE's internal label space.

    ``sources`` (optional) maps a field name (e.g. ``"distinctive_signs"``,
    ``"plant_parts"``, ``"notes"``) to a list of ``Citation`` records. The
    auto re-observation prompt and routing only read the typed fields; the
    citations are preserved on disk so downstream consumers (paper, audit
    UI, deployment dashboards) can show provenance.
    """

    plant_parts: List[str] = field(default_factory=list)         # leaf, stem, fruit, root, flower
    color: List[str] = field(default_factory=list)               # brown, yellow halo, black, ...
    shape: str = ""                                              # circular | angular | irregular | elongated
    margin: str = ""                                             # diffuse | sharp | halo
    texture: List[str] = field(default_factory=list)             # sunken, raised, powdery, downy
    sporulation: List[str] = field(default_factory=list)         # orange masses, white powder, salmon spores
    distinctive_signs: List[str] = field(default_factory=list)   # concentric rings, vein clearing, ...
    progression: str = ""                                        # expanding | systemic | static
    confusion_diseases: List[str] = field(default_factory=list)  # easily-confused diseases
    notes: str = ""
    sources: Dict[str, List[Citation]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any([
            self.plant_parts, self.color, self.shape, self.margin,
            self.texture, self.sporulation, self.distinctive_signs,
            self.progression, self.confusion_diseases, self.notes,
        ])

    def auto_reobservation_prompt(self) -> str:
        """Build a re-observation prompt from the populated visual fields.

        Returns an empty string when no fields are filled — caller should
        fall back to a generic retry message in that case.
        """
        if self.is_empty():
            return ""
        bits: List[str] = []
        if self.sporulation:
            bits.append(f"look for {', '.join(self.sporulation)}")
        if self.margin:
            bits.append(f"examine the lesion margin ({self.margin})")
        if self.distinctive_signs:
            bits.append(f"check for {', '.join(self.distinctive_signs)}")
        if self.color:
            bits.append(f"note color shifts toward {', '.join(self.color)}")
        if self.plant_parts:
            bits.append(f"focus on the {', '.join(self.plant_parts)}")
        return "; ".join(bits)


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
    profile_id: str
    crop: str
    disease: str
    visual: VisualSymptom = field(default_factory=VisualSymptom)
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

    @classmethod
    def from_dict(cls, d: dict) -> "SymptomProfile":
        v = dict(d.get("visual") or {})
        # sources: {field: [Citation-like dict, ...]} → rehydrate Citations.
        raw_sources = v.get("sources") or {}
        rehydrated: Dict[str, List[Citation]] = {}
        for k, items in raw_sources.items():
            if not isinstance(items, list):
                continue
            rehydrated[k] = [
                Citation(
                    value=str(it.get("value", "")),
                    url=str(it.get("url", "")),
                    quote=str(it.get("quote", "")),
                )
                for it in items if isinstance(it, dict)
            ]
        v["sources"] = rehydrated
        sw_raw = d.get("swarm_observations")
        sw = SwarmObservations(**sw_raw) if isinstance(sw_raw, dict) else None
        return cls(
            profile_id=d["profile_id"],
            crop=d["crop"],
            disease=d["disease"],
            visual=VisualSymptom(**v),
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
        # Merge: keep curated visual, accumulate counts and references.
        if not existing.visual.is_empty():
            profile.visual = existing.visual
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
                prof.reobservation_prompt = prof.visual.auto_reobservation_prompt()

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
        return prof.reobservation_prompt or prof.visual.auto_reobservation_prompt() or default

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
