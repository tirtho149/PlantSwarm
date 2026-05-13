"""
tests/test_swarm_logic.py
=========================
Unit tests for the swarm-layer logic that doesn't need vLLM or GPU.

After Algorithm-1 routing removal:
  - parse_agent_output (JSON / markdown / off-domain / kappa coercion)
  - _agreement_filter (K-of-N agreement clustering)
  - _merge_with_existing (conservative merge: existing preserved,
                          overlap bumps support, idempotency,
                          verification status upgrades)
  - existing_deltas_for_state (extract prior regional deltas from a
                               final_registry.json record)

OBSERVE-side tests live in tests/test_observe.py — the trainer no
longer touches Phase 0R traces (OBSERVE is now a KB-augmented OOD
classifier trained on Bugwood images, not on swarm traces).
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Agent output parser (no handoff / routing fields)
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
        "reasoning": "clear evidence",
    })
    deltas, conf, why = parse_agent_output(
        text=text, owned_fields=["lesion_morphology"],
    )
    assert len(deltas) == 1
    assert conf == "high"
    assert "clear" in why


def test_parse_agent_output_markdown_fenced():
    from agents.base_agent import parse_agent_output
    text = '```json\n{"deltas": [], "confidence": "low"}\n```'
    deltas, conf, _ = parse_agent_output(
        text=text, owned_fields=["severity"],
    )
    assert deltas == []
    assert conf == "low"


def test_parse_agent_output_off_domain_field_coerced():
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [{"field": "look_alikes", "image_shows": "X"}],
        "confidence": "medium",
    })
    deltas, _, _ = parse_agent_output(
        text=text, owned_fields=["severity"],
    )
    assert deltas[0]["field"] == "other"


def test_parse_agent_output_empty_image_shows_dropped():
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [{"field": "severity", "image_shows": ""}],
        "confidence": "medium",
    })
    deltas, _, _ = parse_agent_output(
        text=text, owned_fields=["severity"],
    )
    assert deltas == []


def test_parse_agent_output_kappa_word_boundary():
    from agents.base_agent import parse_agent_output
    text = json.dumps({
        "deltas": [], "confidence": "highly uncertain",
    })
    _, conf, _ = parse_agent_output(
        text=text, owned_fields=["severity"],
    )
    assert conf == "medium"


# ---------------------------------------------------------------------------
# Agreement filter
# ---------------------------------------------------------------------------

def test_agreement_filter_keeps_high_support_drops_singletons():
    from plantswarm.delta_pipeline import _agreement_filter
    per_pass = [
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
    survivors = _agreement_filter(per_pass, min_support=3, similarity_threshold=0.2)
    assert len(survivors) == 1
    assert survivors[0]["field"] == "lesion_morphology"
    assert survivors[0]["__support__"] == 3


def test_agreement_filter_floor_k_equals_one():
    from plantswarm.delta_pipeline import _agreement_filter
    per_pass = [
        [{"field":"A","image_shows":"alpha","canonical_says":"","image_quote":""}],
        [{"field":"B","image_shows":"beta", "canonical_says":"","image_quote":""}],
    ]
    out = _agreement_filter(per_pass, min_support=1, similarity_threshold=0.99)
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
    assert any(d["image_shows"].startswith("raised pustular") for d in merged)
    L = next(d for d in merged if d["field"] == "L")
    assert L["__support__"] == 9
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
    assert len(merged1) == len(merged2)
    s1 = next(d for d in merged1 if d["field"] == "L")["__support__"]
    s2 = next(d for d in merged2 if d["field"] == "L")["__support__"]
    assert s2 > s1


def test_merge_with_existing_upgrades_verification_status():
    from plantswarm.delta_pipeline import _merge_with_existing
    existing = [{
        "field": "lesion_morphology",
        "image_shows": "raised pustular lesions with halos",
        "canonical_says": "", "image_quote": "",
        "swarm_support": 3, "verification_status": "unverified",
        "web_support": [],
    }]
    new = [{
        "field": "lesion_morphology",
        "image_shows": "yellow halos surround raised pustular lesions",
        "canonical_says": "", "image_quote": "",
        "swarm_support": 4, "verification_status": "verified",
        "web_support": [{"url": "https://example.com/a", "quote": "..."}],
    }]
    merged, counts = _merge_with_existing(
        existing=existing, new=new, similarity_threshold=0.3,
    )
    assert counts["n_overlaps_bumped"] == 1
    assert counts["n_upgraded"] == 1
    only = merged[0]
    assert only["verification_status"] == "verified"
    assert only["swarm_support"] == 7
    assert any(s["url"] == "https://example.com/a" for s in only["web_support"])


# ---------------------------------------------------------------------------
# existing_deltas_for_state
# ---------------------------------------------------------------------------

def test_existing_deltas_for_state_extracts_and_skips_empty():
    from plantswarm.delta_pipeline import existing_deltas_for_state
    rec = {
        "regional_observations": {
            "Alabama": {
                "deltas": [
                    {"field":"L","image_shows":"X","canonical_says":"Y","image_quote":"Z","image_id":"bugwood::42","support":7},
                    {"field":"S","image_shows":"","image_quote":""},
                ],
            },
        },
    }
    out = existing_deltas_for_state(rec, "Alabama")
    assert len(out) == 1
    assert out[0]["__support__"] == 7
    assert existing_deltas_for_state(rec, "Iowa") == []


