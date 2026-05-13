"""
tests/test_observe.py
=====================
Light, no-GPU unit tests for the KB-augmented OOD classifier.

Covers the pure-Python pieces that don't need transformers / torch:
  - prototypes.build_disease_prototype : KB fields -> text prompt
  - prototypes.build_healthy_prototype : synthetic healthy template
  - prototypes._top_regional_deltas    : verified > weakly > others
  - prototypes.load_seed_prototypes    : symptoms_seed.json roundtrip
  - prototypes.add_healthy_prototypes  : appends per-crop healthy class
  - dataset.ClassIndex                  : bidirectional label <-> id
  - dataset._normalize_pv_folder_to_label
  - dataset.BugwoodTomatoDataset       : CSV + cache filtering + stats
  - dataset.PVFolderDataset            : PV folder layout parsing

The model + trainer require torch/transformers and are exercised by
``scripts/train_observe.py``; we don't unit-test them here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Prototypes
#
# Note: prototypes.py is pure Python, but observe/__init__.py eagerly
# imports the torch-dependent dataset / trainer modules, so any
# ``from observe.prototypes import ...`` still requires torch to be
# installed. Tests skip on machines without torch.
# ---------------------------------------------------------------------------

def test_build_disease_prototype_includes_kb_fields():
    pytest.importorskip("torch")
    from observe.prototypes import build_disease_prototype
    canonical = {
        "summary": "Soilborne fungus causing root + stem collapse.",
        "diagnostic_features": ["microsclerotia in stems"],
        "look_alikes": ["Sudden Death Syndrome"],
        "affected_parts": ["Stem", "Root"],
        "pathogen_scientific_name": "Macrophomina phaseolina",
        "type_of_disease": "Fungal",
        "notes": "", "sources": {}, "treatments": [],
    }
    text = build_disease_prototype(
        crop="Soybean", disease="Charcoal Rot",
        canonical=canonical, regional_observations=None,
    )
    assert "Soybean" in text
    assert "Charcoal Rot" in text
    assert "Macrophomina phaseolina" in text
    assert "microsclerotia" in text.lower()
    assert "Sudden Death Syndrome" in text


def test_build_healthy_prototype():
    pytest.importorskip("torch")
    from observe.prototypes import build_healthy_prototype
    t = build_healthy_prototype("Tomato")
    assert "Tomato" in t
    assert "healthy" in t.lower()
    assert "no visible disease" in t.lower()


def test_top_regional_deltas_prefers_verified_then_support():
    pytest.importorskip("torch")
    from observe.prototypes import _top_regional_deltas
    regional = {
        "Iowa": {
            "deltas": [
                {"field": "L", "image_shows": "low-support unverified item",
                 "swarm_support": 1, "verification_status": "unverified"},
                {"field": "L", "image_shows": "high-support unverified",
                 "swarm_support": 9, "verification_status": "unverified"},
            ],
        },
        "Alabama": {
            "deltas": [
                {"field": "L", "image_shows": "weakly-supported entry",
                 "swarm_support": 2, "verification_status": "weakly_supported"},
                {"field": "L", "image_shows": "verified entry",
                 "swarm_support": 4, "verification_status": "verified"},
            ],
        },
    }
    out = _top_regional_deltas(regional, top_k=3)
    assert out, "expected at least one delta"
    # verified > weakly_supported > unverified
    assert out[0].startswith("in Alabama,") and "verified entry" in out[0]
    assert any("weakly-supported entry" in s for s in out)
    # The low-support unverified item should not bump the high-support one out
    assert any("high-support unverified" in s for s in out)


def test_load_seed_prototypes_filters_and_skips_empty(tmp_path):
    pytest.importorskip("torch")
    from observe.prototypes import load_seed_prototypes
    seed = {
        "min_observations": 3,
        "profiles": [
            {
                "profile_id": "Tomato::Early Blight",
                "crop": "Tomato", "disease": "Early Blight",
                "canonical": {"summary": "Alternaria solani spots.",
                              "diagnostic_features": ["bullseye"],
                              "look_alikes": [], "treatments": [],
                              "affected_parts": ["Foliar"],
                              "pathogen_scientific_name": "Alternaria solani",
                              "type_of_disease": "Fungal",
                              "notes": "", "sources": {}},
                "regional_observations": {},
            },
            {
                "profile_id": "Tomato::EmptyOne",
                "crop": "Tomato", "disease": "EmptyOne",
                "canonical": {"summary": "", "diagnostic_features": [],
                              "look_alikes": [], "treatments": [],
                              "affected_parts": [],
                              "pathogen_scientific_name": "",
                              "type_of_disease": "",
                              "notes": "", "sources": {}},
                "regional_observations": {},
            },
            {
                "profile_id": "Soybean::Charcoal Rot",
                "crop": "Soybean", "disease": "Charcoal Rot",
                "canonical": {"summary": "Soilborne fungus.",
                              "diagnostic_features": ["microsclerotia"],
                              "look_alikes": [], "treatments": [],
                              "affected_parts": ["Stem"],
                              "pathogen_scientific_name": "Macrophomina phaseolina",
                              "type_of_disease": "Fungal",
                              "notes": "", "sources": {}},
                "regional_observations": {},
            },
        ],
    }
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(seed))
    out = load_seed_prototypes(p, crop_filter="Tomato")
    labels = [r["label"] for r in out]
    assert labels == ["Tomato::Early Blight"], (
        "EmptyOne should be dropped (no canonical), Soybean dropped (filtered)"
    )
    assert out[0]["kind"] == "disease"
    assert "Early Blight" in out[0]["prototype"]


def test_add_healthy_prototypes_appends_one_per_crop():
    pytest.importorskip("torch")
    from observe.prototypes import add_healthy_prototypes
    records = [{"label": "Tomato::Early Blight", "crop": "Tomato",
                "disease": "Early Blight", "kind": "disease",
                "prototype": "..."}]
    out = add_healthy_prototypes(records, ["Tomato"])
    assert any(r["label"] == "Tomato::healthy" for r in out)
    healthy = [r for r in out if r["kind"] == "healthy"][0]
    assert "healthy" in healthy["prototype"].lower()


# ---------------------------------------------------------------------------
# ClassIndex
# ---------------------------------------------------------------------------

def test_class_index_roundtrip():
    pytest.importorskip("torch")
    from observe.dataset import ClassIndex
    ci = ClassIndex(["Tomato::Early Blight", "Tomato::Late Blight",
                     "Tomato::healthy"])
    assert len(ci) == 3
    assert ci.to_id("Tomato::Early Blight") == 0
    assert ci.to_label(2) == "Tomato::healthy"
    assert ci.to_id("Tomato::Not In KB") is None


# ---------------------------------------------------------------------------
# PV folder normaliser
# ---------------------------------------------------------------------------

def test_normalize_pv_folder_to_label_handles_special_crops():
    pytest.importorskip("torch")
    from observe.dataset import _normalize_pv_folder_to_label
    assert _normalize_pv_folder_to_label("Tomato___Early_blight") == (
        "Tomato", "Early Blight",
    )
    assert _normalize_pv_folder_to_label("Tomato___healthy") == (
        "Tomato", "healthy",
    )
    crop, _ = _normalize_pv_folder_to_label("Pepper,_bell___Bacterial_spot")
    assert crop == "Bell Pepper"


# ---------------------------------------------------------------------------
# BugwoodTomatoDataset — CSV + cache-aware loader (no PIL decode needed
# beyond constructing the dataset, since __getitem__ requires actual files)
# ---------------------------------------------------------------------------

def _write_minimal_bugwood_csv(path: Path) -> None:
    fields = ["Image Number", "NormCrop", "NormDisease"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        # in scope: Tomato + Early Blight (mapped)
        w.writerow({"Image Number": "img_001", "NormCrop": "Tomato",
                    "NormDisease": "Early Blight"})
        # in scope but no image cached
        w.writerow({"Image Number": "img_002", "NormCrop": "Tomato",
                    "NormDisease": "Early Blight"})
        # wrong crop
        w.writerow({"Image Number": "img_003", "NormCrop": "Soybean",
                    "NormDisease": "Charcoal Rot"})
        # disease not in class index
        w.writerow({"Image Number": "img_004", "NormCrop": "Tomato",
                    "NormDisease": "Mystery Wilt"})


def test_bugwood_tomato_dataset_filters_and_counts(tmp_path):
    pytest.importorskip("torch")
    from observe.dataset import BugwoodTomatoDataset, ClassIndex

    csv_path = tmp_path / "bw.csv"
    _write_minimal_bugwood_csv(csv_path)

    cache = tmp_path / "cache"
    cache.mkdir()
    # one tiny PNG to satisfy the cache lookup for img_001 only
    png_path = cache / "img_001.jpg"
    png_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)  # not a real jpeg but >0 bytes is what the loader checks

    ci = ClassIndex(["Tomato::Early Blight", "Tomato::healthy"])
    ds = BugwoodTomatoDataset(
        csv_path=csv_path, cache_dirs=[cache], class_index=ci, crop="Tomato",
    )
    s = ds.stats()
    assert s["n_samples"] == 1
    assert s["skipped_no_image"] == 1
    assert s["skipped_no_class"] == 1
    assert s["skipped_wrong_crop"] == 1


# ---------------------------------------------------------------------------
# PVFolderDataset — folder layout + class index filtering
# ---------------------------------------------------------------------------

def test_pv_folder_dataset_filters_to_crop_and_index(tmp_path):
    pytest.importorskip("torch")
    from observe.dataset import ClassIndex, PVFolderDataset

    root = tmp_path / "pv"
    (root / "Tomato___Early_blight").mkdir(parents=True)
    (root / "Tomato___Late_blight").mkdir(parents=True)
    (root / "Potato___Early_blight").mkdir(parents=True)  # wrong crop
    # 2 jpgs in Tomato Early; 1 in Tomato Late
    (root / "Tomato___Early_blight" / "a.jpg").write_bytes(b"\x00" * 8)
    (root / "Tomato___Early_blight" / "b.jpg").write_bytes(b"\x00" * 8)
    (root / "Tomato___Late_blight" / "c.jpg").write_bytes(b"\x00" * 8)
    (root / "Potato___Early_blight" / "d.jpg").write_bytes(b"\x00" * 8)

    # Only Early Blight in the class index; Late Blight should be skipped.
    ci = ClassIndex(["Tomato::Early Blight"])
    ds = PVFolderDataset(root=root, class_index=ci, crop="Tomato")
    s = ds.stats()
    assert s["n_samples"] == 2  # only Early Blight
    assert s["skipped_no_class"] == 1   # Late Blight present but not in index
    assert s["skipped_other_crop"] == 1
    assert s["per_class"] == {"Tomato::Early Blight": 2}
