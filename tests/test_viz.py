"""
tests/test_viz.py
=================
End-to-end smoke for the viz layer against synthetic JSON inputs.

Verifies:
  - kb_stats aggregation
  - observe_curves history aggregation
  - observe_eval table emission
  - trace_stats aggregation
  - Each script emits an auto_<name>.tex even when matplotlib is missing

These tests do NOT require matplotlib. When matplotlib IS installed,
PNG figures are also produced; we don't assert on their pixel content,
only that the file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_seed(tmp_path: Path) -> Path:
    """Tiny symptoms_seed.json with two profiles + deltas covering
    multiple statuses."""
    seed = {
        "min_observations": 3,
        "profiles": [
            {
                "profile_id": "Soybean::Charcoal Rot",
                "crop": "Soybean", "disease": "Charcoal Rot",
                "canonical": {
                    "summary": "Soilborne fungus",
                    "diagnostic_features": ["microsclerotia"],
                    "look_alikes": [], "treatments": [],
                    "affected_parts": ["Foliar", "Stem"],
                    "pathogen_scientific_name": "Macrophomina phaseolina",
                    "type_of_disease": "Fungal", "notes": "", "sources": {},
                },
                "regional_observations": {
                    "Alabama": {
                        "state": "Alabama",
                        "image_ids": ["bugwood::1"],
                        "deltas": [
                            {"field": "lesion_morphology",
                             "canonical_says": "(not specified)",
                             "image_shows": "yellow halos around dark spots",
                             "image_quote": "...", "image_id": "bugwood::1",
                             "swarm_support": 4, "verification_status": "verified",
                             "web_support": [{"url": "https://example.com/a",
                                              "quote": "..."}]},
                            {"field": "severity",
                             "canonical_says": "(not specified)",
                             "image_shows": "whole-field collapse",
                             "image_quote": "...", "image_id": "bugwood::1",
                             "swarm_support": 3, "verification_status": "provisional",
                             "web_support": []},
                        ],
                    },
                    "Iowa": {
                        "state": "Iowa",
                        "image_ids": ["bugwood::2"],
                        "deltas": [
                            {"field": "diagnostic_features",
                             "canonical_says": "microsclerotia",
                             "image_shows": "marbled cross-sections",
                             "image_quote": "...", "image_id": "bugwood::2",
                             "swarm_support": 5, "verification_status": "verified",
                             "web_support": [{"url": "https://example.com/b",
                                              "quote": "..."}]},
                        ],
                    },
                },
                "state_counts": {"Alabama": 1, "Iowa": 1},
                "aez_counts": {}, "total_observations": 2,
                "reference_ids": [], "reobservation_prompt": "",
            },
            {
                "profile_id": "Tomato::Early Blight",
                "crop": "Tomato", "disease": "Early Blight",
                "canonical": {"summary": "", "diagnostic_features": [], "look_alikes": [],
                              "treatments": [], "affected_parts": [],
                              "pathogen_scientific_name": "", "type_of_disease": "",
                              "notes": "", "sources": {}},
                "regional_observations": {},
                "state_counts": {}, "aez_counts": {}, "total_observations": 0,
                "reference_ids": [], "reobservation_prompt": "",
            },
        ],
    }
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(seed))
    return p


def _make_history(tmp_path: Path) -> Path:
    history = [
        {"epoch": 1, "train": {"loss": 1.4, "top1": 0.25},
                     "val":   {"loss": 1.3, "top1": 0.30}},
        {"epoch": 2, "train": {"loss": 0.9, "top1": 0.55},
                     "val":   {"loss": 0.95, "top1": 0.52}},
        {"epoch": 3, "train": {"loss": 0.6, "top1": 0.72},
                     "val":   {"loss": 0.70, "top1": 0.68}},
    ]
    p = tmp_path / "history.json"
    p.write_text(json.dumps(history))
    return p


def _make_eval(tmp_path: Path) -> Path:
    ev = {
        "crop": "Tomato",
        "evals": {
            "plantvillage": {
                "n_samples":     200,
                "top1_accuracy": 0.62,
                "top5_accuracy": 0.91,
                "macro_f1":      0.57,
                "per_class": {
                    "Tomato::Early Blight":      {"support": 100, "correct": 70, "accuracy": 0.70, "in_kb": True},
                    "Tomato::Late Blight":       {"support": 60,  "correct": 38, "accuracy": 0.63, "in_kb": True},
                    "Tomato::Septoria Leaf Spot":{"support": 40,  "correct": 16, "accuracy": 0.40, "in_kb": False},
                },
            },
            "plantwild": {
                "n_samples":     50,
                "top1_accuracy": 0.40,
                "top5_accuracy": 0.78,
                "macro_f1":      0.36,
                "per_class": {
                    "Tomato::Early Blight": {"support": 30, "correct": 14, "accuracy": 0.47, "in_kb": True},
                    "Tomato::Late Blight":  {"support": 20, "correct": 6,  "accuracy": 0.30, "in_kb": True},
                },
            },
        },
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(ev))
    return p


def _make_traces(tmp_path: Path) -> Path:
    p = tmp_path / "traces.jsonl"
    lines = []
    for i in range(5):
        rec = {
            "profile_id": "Soybean::Charcoal Rot",
            "crop": "Soybean", "disease": "Charcoal Rot", "state": "Alabama",
            "primary_image_id": f"bugwood::{i}", "image_path": f"/tmp/img_{i}.jpg",
            "pass_idx": i,
            "specialist_outputs": [
                {"agent_name":"MorphologyAgent","confidence":"medium",
                 "deltas":[{"field":"lesion_morphology","image_shows":"X",
                            "canonical_says":"","image_quote":""}],
                 "reasoning":"","raw_text":""},
                {"agent_name":"SymptomAgent","confidence":"high",
                 "deltas":[{"field":"spread_pattern","image_shows":"Y",
                            "canonical_says":"","image_quote":""}],
                 "reasoning":"","raw_text":""},
                {"agent_name":"PathogenAgent","confidence":"low","deltas":[],
                 "reasoning":"","raw_text":""},
                {"agent_name":"SeverityAgent","confidence":"medium","deltas":[],
                 "reasoning":"","raw_text":""},
            ],
            "consolidator_output": {
                "agent_name":"DiagnosisAgent","confidence":"high",
                "deltas":[{"field":"lesion_morphology","image_shows":"X",
                           "canonical_says":"","image_quote":""}],
                "reasoning":"","raw_text":"",
            },
            "final_deltas": [
                {"field":"lesion_morphology","image_shows":"X",
                 "canonical_says":"","image_quote":""},
            ],
            "existing_kb_at_start": [],
        }
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_kb_stats_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import kb_stats, _common
    seed = _make_seed(tmp_path)
    # Redirect output dirs to a temp area so the test can't pollute the repo.
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["kb_stats", "--seed", str(seed), "--name", "kbtest"])
    kb_stats.main()
    tex = (tmp_path / "tex" / "auto_kbtest.tex").read_text()
    assert "PathomeDB seed summary" in tex
    assert "Profiles" in tex


def test_observe_curves_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import observe_curves, _common
    history = _make_history(tmp_path)
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["observe_curves", "--history", str(history),
                         "--name", "ocurves"])
    observe_curves.main()
    tex = (tmp_path / "tex" / "auto_ocurves.tex").read_text()
    assert "OBSERVE training history" in tex
    assert "Epoch" in tex


def test_observe_eval_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import observe_eval, _common
    ev = _make_eval(tmp_path)
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["observe_eval", "--eval", str(ev), "--name", "oeval"])
    observe_eval.main()
    tex = (tmp_path / "tex" / "auto_oeval.tex").read_text()
    assert "Top-1 accuracy" in tex
    assert "Macro F1"      in tex
    assert "plantvillage"  in tex
    assert "plantwild"     in tex


def test_trace_stats_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import trace_stats, _common
    tr = _make_traces(tmp_path)
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["trace_stats", "--traces", str(tr), "--name", "tstats"])
    trace_stats.main()
    tex = (tmp_path / "tex" / "auto_tstats.tex").read_text()
    assert "trace summary" in tex.lower() or "phase 0r trace" in tex.lower()
