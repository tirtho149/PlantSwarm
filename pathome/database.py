"""
pathome/database.py
===================
PathomeDB: a visual-symptom-centric, geo-aware knowledge base.

Two stores. That's it.

    db.symptoms : SymptomLibrary       what each (crop, disease) looks like,
                                       plus state/AEZ observation counts
    db.refs     : ReferenceLibrary     held-out images for visual retrieval

Older versions of this file split knowledge across five layers (mechanistic
pathway, cross-crop manifestation, regional epidemiology, decision graph,
reference library). That layering was more confusing than helpful given the
data we actually have — the Bugwood CSV doesn't carry the metadata the
mechanistic / decision-graph layers were designed for. The new design keeps
the two things the data does support — visual symptoms and geo-aware priors
— and folds them into a single profile per (crop, disease).

Build / load:

    from pathome import PathomeDB
    db = PathomeDB.build_from_bugwood(trace_records, reference_records)
    db.save("artifacts/pathome_v1/")

    db = PathomeDB.load("artifacts/pathome_v1/")
    prior = db.geo_prior(disease="Anthracnose", lat=35.6, lon=-79.8)
    refs  = db.retrieve_references(image, lat=35.6, lon=-79.8)

Query API kept stable for ``agents/base_agent.py`` and the run scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image

from .layer5_references import ReferenceImage, ReferenceLibrary, RetrievalHit
from .symptoms import (
    CanonicalDisease,
    RegionalObservation,
    SymptomLibrary,
    SymptomProfile,
)
from utils.geo import US_STATE_CENTROID, aez_lookup


# ---------------------------------------------------------------------------
# GeoPriorResult — return shape kept for API compatibility with old callers.
# ---------------------------------------------------------------------------

@dataclass
class GeoPriorResult:
    """What OBSERVE / agents consume when querying the geo-aware prior."""
    disease: str
    state: Optional[str]            # resolved from (lat, lon) via state-centroid match
    aez_code: Optional[str]
    prior: Optional[float]          # P(disease | state); None when sparse → use ``global_prior``
    global_prior: float
    is_sparse: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_state(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """Reverse-geocode (lat, lon) to the nearest US state name.

    The CSV ingest path uses state centroids, so this simply finds the
    closest centroid. Returns the human-readable state (Title Case).
    """
    if lat is None or lon is None:
        return None
    best_state: Optional[str] = None
    best_d = float("inf")
    for state, (slat, slon) in US_STATE_CENTROID.items():
        d = (slat - lat) ** 2 + (slon - lon) ** 2
        if d < best_d:
            best_d = d
            best_state = state
    return best_state.title() if best_state else None


# ---------------------------------------------------------------------------
# PathomeDB
# ---------------------------------------------------------------------------

class PathomeDB:
    """Symptom-centric, geo-aware knowledge base."""

    def __init__(
        self,
        symptoms: Optional[SymptomLibrary] = None,
        refs: Optional[ReferenceLibrary] = None,
        version: str = "v2.0",
    ):
        self.version = version
        self.symptoms: SymptomLibrary = symptoms if symptoms is not None else SymptomLibrary()
        self.refs: ReferenceLibrary = refs if refs is not None else ReferenceLibrary()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build_from_bugwood(
        cls,
        trace_records: Iterable,
        reference_records: Iterable,
        symptoms_path: Optional[str] = None,
        version: str = "v2.0",
        # Legacy kwargs accepted for backward compatibility with old call sites.
        # They do not influence the build — Layer 1/2/4 are no longer separate.
        layer1_path: Optional[str] = None,    # noqa: ARG003 (kept for compat)
        layer2_path: Optional[str] = None,    # noqa: ARG003 (kept for compat)
    ) -> "PathomeDB":
        """Build a fresh PathomeDB from the trace + reference Bugwood splits.

        ``symptoms_path`` is an optional JSON of curator-authored visual
        descriptions; auto-derived geo counts and reference IDs are layered
        on top so the curated visual block is preserved.
        """
        symptoms = (
            SymptomLibrary.load(symptoms_path)
            if symptoms_path else SymptomLibrary()
        )

        # Materialise once so we can iterate twice (records may be a generator).
        trace_list = list(trace_records)
        ref_list = list(reference_records)

        symptoms.update_from_records(trace_list, ref_list)
        symptoms.finalize_reobservation_prompts()

        refs = ReferenceLibrary()
        for r in ref_list:
            refs.add(ReferenceImage(
                ref_id=r.image_id,
                image_path=r.src_path,
                crop_species=r.crop_species,
                disease_name=r.disease_name,
                lat=r.lat, lon=r.lon, aez_code=r.aez_code,
            ))

        return cls(symptoms=symptoms, refs=refs, version=version)

    # ------------------------------------------------------------------
    # Query — kept stable for agents/base_agent.py and run scripts
    # ------------------------------------------------------------------

    def geo_prior(
        self,
        disease: str,
        lat: Optional[float],
        lon: Optional[float],
        month: Optional[int] = None,    # accepted for compat; ignored
    ) -> GeoPriorResult:
        """P(disease | state) where ``state`` is the nearest US state centroid.

        ``month`` is accepted for backward compatibility with the old
        signature but is unused — the CSV has no capture date and the
        symptom-centric library does not maintain a time axis.
        """
        state = _nearest_state(lat, lon)
        aez = aez_lookup(lat, lon) if (lat is not None and lon is not None) else None
        aez_code = aez.code if aez else None
        prior = self.symptoms.geo_prior(disease, state)
        return GeoPriorResult(
            disease=disease,
            state=state,
            aez_code=aez_code,
            prior=prior,
            global_prior=self.symptoms.global_prior(disease),
            is_sparse=self.symptoms.is_sparse_in_state(disease, state),
        )

    def retrieve_references(
        self,
        query_image: Image.Image,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        top_k: int = 3,
        constrain_disease: Optional[str] = None,
    ) -> List[RetrievalHit]:
        return self.refs.retrieve(
            query_image=query_image,
            query_lat=lat, query_lon=lon,
            top_k=top_k, constrain_disease=constrain_disease,
        )

    def reobservation_prompt(
        self, crop: str, disease: str, default: str = "",
    ) -> str:
        """Targeted re-examination text for low-confidence backtracks.

        Replaces the old ``db.layer4.root().reobservation_prompt`` lookup
        with a per-(crop, disease) prompt drawn from the symptom profile's
        visual block.
        """
        return self.symptoms.reobservation_prompt(crop, disease, default)

    def symptom_profile(self, crop: str, disease: str) -> Optional[SymptomProfile]:
        return self.symptoms.get(crop, disease)

    def symptom_context(self, crop: str, disease: str, state: str = "") -> str:
        """Canonical KB + per-state deltas, ready to drop into a prompt.
        Empty string if the (crop, disease) profile is unknown.
        """
        return self.symptoms.context_for(crop, disease, state)

    def prevalent_in_state(self, state: str, top_k: int = 5):
        return self.symptoms.prevalent_in_state(state, top_k=top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dirpath: str) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        self.symptoms.save(str(d / "symptoms.json"))
        self.refs.save(str(d / "refs"))
        with open(d / "version.txt", "w") as f:
            f.write(self.version + "\n")

    @classmethod
    def load(cls, dirpath: str) -> "PathomeDB":
        d = Path(dirpath)
        symptoms_path = d / "symptoms.json"
        refs_path = d / "refs"
        symptoms = (
            SymptomLibrary.load(str(symptoms_path))
            if symptoms_path.exists() else SymptomLibrary()
        )
        refs = (
            ReferenceLibrary.load(str(refs_path))
            if refs_path.exists() else ReferenceLibrary()
        )
        version = (
            (d / "version.txt").read_text().strip()
            if (d / "version.txt").exists() else "unknown"
        )
        return cls(symptoms=symptoms, refs=refs, version=version)
