"""
tests/test_swarm_logic.py
=========================
Unit tests for the swarm-layer logic that doesn't need vLLM or GPU:

  - parse_agent_output  (JSON, markdown fences, off-domain fields,
                         kappa coercion, handoff matching)
  - algorithm1_handoff  (paper Algorithm 1 over the kappa x backtrack
                         x coverage grid)
  - _agreement_filter   (K-of-N agreement clustering)
  - _merge_with_existing (conservative merge: existing preserved,
                          overlap bumps support, idempotency)
  - existing_deltas_for_state (extract prior regional deltas from a
                               final_registry.json record)
  - annotations_from_trace + load_phase0r_traces (OBSERVE training
                                                  data ingest)
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Agent output parser
# ---------------------------------------------------------------------------

def test_parse_agent_output_well_formed():
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [{
            "field": "lesion_morphology",
            "canonical_says": "(not specified)",
            "image_shows": "raised pustular lesions with chlorotic halos",
            "image_quote": "yellow rings around dark centers",
        }],
        "confidence": "high",
        "handoff_target": "PathogenAgent",
        "reasoning": "clear evidence",
    })
    deltas, conf, h, why = parse_agent_output(
        text=text,
        owned_fields=["lesion_morphology"],
        handoff_menu=["MorphologyAgent","SymptomAgent","PathogenAgent","SeverityAgent","DiagnosisAgent"],
    )
    assert len(deltas) == 1
    assert conf == "high"
    assert h == "PathogenAgent"
    assert "clear" in why


def test_parse_agent_output_markdown_fenced():
    from agents.base_agent import parse_agent_output
    text = '```json\n{"deltas": [], "confidence": "low", "handoff_target": "MorphologyAgent"}\n```'
    deltas, conf, h, _ = parse_agent_output(
        text=text, owned_fields=["severity"], handoff_menu=["MorphologyAgent","DiagnosisAgent"],
    )
    assert deltas == []
    assert conf == "low"
    assert h == "MorphologyAgent"


def test_parse_agent_output_off_domain_field_coerced():
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [{"field": "look_alikes", "image_shows": "X"}],
        "confidence": "medium", "handoff_target": "DiagnosisAgent",
    })
    deltas, _, _, _ = parse_agent_output(
        text=text, owned_fields=["severity"], handoff_menu=["DiagnosisAgent"],
    )
    # Severity doesn't own look_alikes; should be coerced to "other".
    assert deltas[0]["field"] == "other"


def test_parse_agent_output_empty_image_shows_dropped():
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [{"field": "severity", "image_shows": ""}],
        "confidence": "medium", "handoff_target": "DiagnosisAgent",
    })
    deltas, _, _, _ = parse_agent_output(
        text=text, owned_fields=["severity"], handoff_menu=["DiagnosisAgent"],
    )
    assert deltas == []


def test_parse_agent_output_kappa_word_boundary():
    """'highly uncertain' must not coerce to 'high' via substring match."""
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [], "confidence": "highly uncertain",
        "handoff_target": "DiagnosisAgent",
    })
    _, conf, _, _ = parse_agent_output(
        text=text, owned_fields=["severity"], handoff_menu=["DiagnosisAgent"],
    )
    assert conf == "medium"   # falls back to default


def test_parse_agent_output_handoff_substring_fallback():
    """Model writes 'next: SymptomAgent' instead of bare agent name."""
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [], "confidence": "medium",
        "handoff_target": "next: SymptomAgent",
    })
    _, _, h, _ = parse_agent_output(
        text=text,
        owned_fields=["spread_pattern"],
        handoff_menu=["MorphologyAgent","SymptomAgent","DiagnosisAgent"],
    )
    assert h == "SymptomAgent"


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kappa,b,max_b,run_count,current,expected", [
    # Rule 1: low + budget remaining → MorphologyAgent
    ("low",    0, 1, {"MorphologyAgent","SymptomAgent"},     "SymptomAgent",  "MorphologyAgent"),
    # Rule 1 with max=2 + b=1 → still backtracks
    ("low",    1, 2, {"MorphologyAgent","SymptomAgent"},     "SymptomAgent",  "MorphologyAgent"),
    # Rule 2: low + budget exhausted → default_forward
    ("low",    1, 1, {"MorphologyAgent","SymptomAgent"},     "SymptomAgent",  "PathogenAgent"),
    # Rule 2 with max=0 → never backtracks even at b=0
    ("low",    0, 0, {"MorphologyAgent","SymptomAgent"},     "SymptomAgent",  "PathogenAgent"),
    # Rule 3: high + all 4 ran → DiagnosisAgent
    ("high",   0, 1, {"MorphologyAgent","SymptomAgent","PathogenAgent","SeverityAgent"},
                                                              "SeverityAgent", "DiagnosisAgent"),
    # Rule 4: medium → model's choice
    ("medium", 0, 1, {"MorphologyAgent"},                     "MorphologyAgent","PathogenAgent"),
])
def test_algorithm1_rules(kappa, b, max_b, run_count, current, expected):
    from plantswarm.delta_pipeline import algorithm1_handoff
    nxt, _why = algorithm1_handoff(
        current_agent_name=current,
        model_handoff="PathogenAgent",
        confidence=kappa,
        backtrack_count=b,
        max_backtracks=max_b,
        specialists_run=run_count,
        default_forward="PathogenAgent",
    )
    assert nxt == expected


# ---------------------------------------------------------------------------
# Agreement filter
# ---------------------------------------------------------------------------

def test_agreement_filter_keeps_high_support_drops_singletons():
    from plantswarm.delta_pipeline import _agreement_filter
    per_run = [
        [{"field":"lesion_morphology","image_shows":"raised pustular lesions surrounded by chlorotic halos",
          "canonical_says":"(not specified)","image_quote":""}],
        [{"field":"lesion_morphology","image_shows":"small raised pustules surrounded by yellow halos",
          "canonical_says":"(not specified)","image_quote":""}],
        [{"field":"lesion_morphology","image_shows":"yellow halos around dark raised pustular lesions",
          "canonical_says":"(not specified)","image_quote":""}],
        [{"field":"diagnostic_features","image_shows":"carrot-shaped white fronds (hallucination)",
          "canonical_says":"","image_quote":""}],
        [],
    ]
    survivors = _agreement_filter(per_run, min_support=3, similarity_threshold=0.2)
    assert len(survivors) == 1
    assert survivors[0]["field"] == "lesion_morphology"
    assert survivors[0]["__support__"] == 3


def test_agreement_filter_floor_k_equals_one():
    from plantswarm.delta_pipeline import _agreement_filter
    per_run = [
        [{"field":"A","image_shows":"alpha","canonical_says":"","image_quote":""}],
        [{"field":"B","image_shows":"beta", "canonical_says":"","image_quote":""}],
    ]
    out = _agreement_filter(per_run, min_support=1, similarity_threshold=0.99)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Conservative merge
# ---------------------------------------------------------------------------

def test_merge_existing_preserved_and_bumps_support():
    from plantswarm.delta_pipeline import _merge_with_existing
    existing = [
        {"field":"L","image_shows":"raised pustular lesions w halos",
         "canonical_says":"","image_quote":"","__support__":5},
    ]
    new = [
        {"field":"L","image_shows":"yellow halos around raised pustular lesions",
         "canonical_says":"","image_quote":"","__support__":4},
        {"field":"P","image_shows":"new spread pattern, no existing in P",
         "canonical_says":"","image_quote":"","__support__":3},
    ]
    merged, counts = _merge_with_existing(
        existing=existing, new=new, similarity_threshold=0.3,
    )
    # Existing always preserved
    assert any(d["image_shows"].startswith("raised pustular") for d in merged)
    # L support bumped
    L = next(d for d in merged if d["field"] == "L")
    assert L["__support__"] == 9
    # P added
    assert any(d["field"] == "P" for d in merged)
    assert counts["n_added"] == 1
    assert counts["n_overlaps_bumped"] == 1


def test_merge_is_idempotent_on_shape():
    from plantswarm.delta_pipeline import _merge_with_existing
    existing = [
        {"field":"L","image_shows":"pustular lesions","canonical_says":"","image_quote":"","__support__":3},
    ]
    new = [
        {"field":"L","image_shows":"pustular lesions on leaves","canonical_says":"","image_quote":"","__support__":2},
    ]
    merged1, _ = _merge_with_existing(existing=existing, new=new, similarity_threshold=0.3)
    merged2, _ = _merge_with_existing(existing=merged1, new=new, similarity_threshold=0.3)
    # Same number of entries on re-run, support strictly grows.
    assert len(merged1) == len(merged2)
    s1 = next(d for d in merged1 if d["field"] == "L")["__support__"]
    s2 = next(d for d in merged2 if d["field"] == "L")["__support__"]
    assert s2 > s1


# ---------------------------------------------------------------------------
# existing_deltas_for_state
# ---------------------------------------------------------------------------

def test_existing_deltas_for_state_extracts_and_skips_empty():
    from plantswarm.delta_pipeline import existing_deltas_for_state
    rec = {
        "regional_observations": {
            "Alabama": {
                "deltas": [
                    {"field":"L","image_shows":"X","canonical_says":"Y","image_quote":"Z","image_id":"bugwood::1","support":7},
                    {"field":"S","image_shows":"","image_quote":""},   # dropped
                ],
            },
        },
    }
    out = existing_deltas_for_state(rec, "Alabama")
    assert len(out) == 1
    assert out[0]["__support__"] == 7
    # Cold start (state with no deltas) → empty.
    assert existing_deltas_for_state(rec, "Iowa") == []


# ---------------------------------------------------------------------------
# Trace JSONL ingest (OBSERVE training data)
# ---------------------------------------------------------------------------

def test_annotations_from_trace_per_step():
    pytest.importorskip("torch")
    from observe.trainer import annotations_from_trace, _AGENT_IDX
    trace = {
        "profile_id": "Soybean::Charcoal Rot",
        "crop": "Soybean", "disease": "Charcoal Rot", "state": "Alabama",
        "image_path": "/tmp/img.jpg",
        "primary_image_id": "bugwood::1",
        "run_idx": 0,
        "path": ["MorphologyAgent", "SymptomAgent", "DiagnosisAgent"],
        "decisions": ["model_choice", "alg1_high_kappa_all_covered_terminate"],
        "confidences": ["medium", "high"],
        "backtrack_count": 0,
        "early_terminated": True,
        "context_buffer": [
            {"agent_name":"MorphologyAgent",
             "deltas":[{"field":"lesion_morphology","image_shows":"X","canonical_says":"Y","image_quote":"Z"}],
             "confidence":"medium","handoff_target":"SymptomAgent","reasoning":"r1","raw_text":""},
            {"agent_name":"SymptomAgent",
             "deltas":[{"field":"spread_pattern","image_shows":"S","canonical_says":"","image_quote":""}],
             "confidence":"high","handoff_target":"DiagnosisAgent","reasoning":"r2","raw_text":""},
        ],
        "final_deltas": [
            {"field":"lesion_morphology","image_shows":"X","canonical_says":"Y","image_quote":"Z"},
            {"field":"spread_pattern","image_shows":"S","canonical_says":"","image_quote":""},
        ],
        "existing_kb_at_start": [],
    }
    anns = annotations_from_trace(trace)
    # Two context_buffer entries; the second's next_agent is DiagnosisAgent
    # which is a valid agent → both expand. (Last buf step has i+1 = len(path),
    # so it terminates and is skipped.) → expect 2 annotations.
    assert len(anns) == 2
    assert anns[0].current_agent == "MorphologyAgent"
    assert anns[0].next_agent == "SymptomAgent"
    assert anns[0].confidence == 0.6      # medium → 0.6
    assert anns[1].current_agent == "SymptomAgent"
    assert anns[1].next_agent == "DiagnosisAgent"
    assert anns[1].confidence == 0.9      # high → 0.9


def test_load_phase0r_traces_full_chain(tmp_path):
    pytest.importorskip("torch")
    from observe.trainer import load_phase0r_traces
    p = tmp_path / "phase0r_traces.jsonl"
    trace = {
        "profile_id": "X::Y", "crop": "X", "disease": "Y", "state": "Z",
        "image_path": "/tmp/img.jpg",
        "primary_image_id": "bugwood::1",
        "run_idx": 0,
        "path": ["MorphologyAgent", "DiagnosisAgent"],
        "decisions": ["model_choice"],
        "confidences": ["medium"],
        "backtrack_count": 0,
        "early_terminated": True,
        "context_buffer": [
            {"agent_name":"MorphologyAgent",
             "deltas":[{"field":"lesion_morphology","image_shows":"X","canonical_says":"","image_quote":""}],
             "confidence":"medium","handoff_target":"DiagnosisAgent","reasoning":"","raw_text":""},
        ],
        "final_deltas": [{"field":"lesion_morphology","image_shows":"X","canonical_says":"","image_quote":""}],
        "existing_kb_at_start": [],
    }
    p.write_text(json.dumps(trace) + "\n")
    anns = load_phase0r_traces(str(p))
    assert len(anns) == 1
    assert anns[0].next_agent == "DiagnosisAgent"


# ---------------------------------------------------------------------------
# Trainer collator (synthetic — uses Pillow but no GPU / no transformers)
# ---------------------------------------------------------------------------

def test_split_annotations_no_image_leak():
    pytest.importorskip("torch")
    from observe.trainer import TraceStepAnnotation, split_annotations
    anns = [
        TraceStepAnnotation(
            image_path=f"/tmp/img_{i}.jpg", crop="X", disease="Y", state="Z",
            step=0, current_agent="MorphologyAgent",
            context_text="...",
            next_agent="SymptomAgent", backtrack=False,
            confidence=0.6, epistemic=0.5, aleatoric=0.4, overconfidence=False,
            belief_state="", profile_id="X::Y", run_idx=0,
            n_deltas_at_step=0, n_deltas_final=0,
        )
        for i in range(10)
    ]
    s = split_annotations(anns, val_frac=0.2, held_frac=0.2, seed=42)
    # No image_path appears in more than one fold.
    train_imgs = {a.image_path for a in s["train"]}
    val_imgs   = {a.image_path for a in s["val"]}
    held_imgs  = {a.image_path for a in s["held"]}
    assert not (train_imgs & val_imgs)
    assert not (train_imgs & held_imgs)
    assert not (val_imgs   & held_imgs)
    # All annotations accounted for.
    assert len(s["train"]) + len(s["val"]) + len(s["held"]) == 10
