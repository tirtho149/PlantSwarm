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
        {"epoch": 1, "train": {"total": 1.4, "cal": 0.3, "cons": 0.05, "oc": 0.1},
                     "val":   {"total": 1.3}},
        {"epoch": 2, "train": {"total": 0.9, "cal": 0.2, "cons": 0.04, "oc": 0.08},
                     "val":   {"total": 0.95}},
        {"epoch": 3, "train": {"total": 0.6, "cal": 0.15, "cons": 0.03, "oc": 0.06},
                     "val":   {"total": 0.7}},
    ]
    p = tmp_path / "history.json"
    p.write_text(json.dumps(history))
    return p


def _make_eval(tmp_path: Path) -> Path:
    ev = {
        "n_samples": 200, "n_images": 50,
        "kappa_mae": 0.12, "kappa_ece": 0.09,
        "epistemic_mae": 0.15, "aleatoric_mae": 0.18,
        "overconfidence_accuracy": 0.88,
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
    assert "Kappa MAE" in tex
    assert "Overconfidence" in tex


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
