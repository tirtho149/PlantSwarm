"""
tests/test_visual_swarm.py
==========================
Sanity tests for the 24-specialist visual-symptom swarm.

These do not exercise vLLM — they assert structural properties of the
swarm (field vocabulary, agent roster, owned-field uniqueness, prompt
template integrity).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Roster + field vocabulary
# ---------------------------------------------------------------------------

def test_roster_has_24_specialists():
    from agents import SPECIALIST_AGENTS
    assert len(SPECIALIST_AGENTS) == 24


def test_each_specialist_owns_exactly_one_field():
    """The swarm is decomposed so each specialist asks ONE focused
    question and owns ONE delta field."""
    from agents import SPECIALIST_AGENTS
    for cls in SPECIALIST_AGENTS:
        assert isinstance(cls.OWNED_FIELDS, list), cls.__name__
        assert len(cls.OWNED_FIELDS) == 1, (
            f"{cls.__name__} owns {cls.OWNED_FIELDS!r} — expected exactly 1"
        )


def test_owned_fields_are_unique_across_specialists():
    """No two specialists may own the same delta field (avoids
    consolidator-side ambiguity about which agent's output wins)."""
    from agents import SPECIALIST_AGENTS
    seen = set()
    for cls in SPECIALIST_AGENTS:
        for f in cls.OWNED_FIELDS:
            assert f not in seen, f"{cls.__name__} duplicates owned field {f}"
            seen.add(f)


def test_every_specialist_field_is_in_vocabulary():
    from agents import SPECIALIST_AGENTS, ALLOWED_DELTA_FIELDS
    for cls in SPECIALIST_AGENTS:
        for f in cls.OWNED_FIELDS:
            assert f in ALLOWED_DELTA_FIELDS, (
                f"{cls.__name__} owns {f}, not in ALLOWED_DELTA_FIELDS"
            )


def test_vocabulary_is_visual_only_no_treatments_or_pathogen():
    """Phase 0 (Claude) owns non-visual KB; the swarm vocabulary must
    not include treatments / pathogen / type-of-disease."""
    from agents import ALLOWED_DELTA_FIELDS
    forbidden = ("treatments", "pathogen", "type_of_disease",
                 "cultural_control", "chemical_control")
    for f in forbidden:
        assert f not in ALLOWED_DELTA_FIELDS, (
            f"{f!r} is in ALLOWED_DELTA_FIELDS — that belongs to Claude, "
            f"not the visual swarm"
        )


# ---------------------------------------------------------------------------
# Agent metadata sanity
# ---------------------------------------------------------------------------

def test_every_specialist_has_focused_system_prompt_and_question():
    from agents import SPECIALIST_AGENTS
    for cls in SPECIALIST_AGENTS:
        assert cls.AGENT_NAME and cls.AGENT_NAME != "BaseAgent", cls.__name__
        assert cls.SYSTEM_PROMPT, cls.__name__
        assert cls.FOCUS_QUESTION, cls.__name__
        assert len(cls.SYSTEM_PROMPT) > 80, (
            f"{cls.__name__} SYSTEM_PROMPT is suspiciously short"
        )
        # Each system prompt mentions its agent's own name so the
        # model knows what specialty hat it's wearing.
        assert cls.AGENT_NAME in cls.SYSTEM_PROMPT, (
            f"{cls.__name__} SYSTEM_PROMPT does not self-identify"
        )


def test_consolidator_groups_cover_all_24_specialists():
    """VisualDiagnosisAgent's ORGAN_GROUPS must mention every
    specialist by AGENT_NAME so the consolidator prompt renders them
    in the right organ-family cluster."""
    from agents import SPECIALIST_AGENTS
    from agents.diagnosis_agent import ORGAN_GROUPS
    grouped = {a for v in ORGAN_GROUPS.values() for a in v}
    for cls in SPECIALIST_AGENTS:
        assert cls.AGENT_NAME in grouped, (
            f"{cls.AGENT_NAME} missing from ORGAN_GROUPS in diagnosis_agent.py"
        )


# ---------------------------------------------------------------------------
# parse_agent_output still handles the new field vocabulary
# ---------------------------------------------------------------------------

def test_parse_agent_output_accepts_new_field_name():
    """Smoke: an agent output naming a new visual field round-trips
    through parse_agent_output."""
    from agents.base_agent import parse_agent_output
    import json
    raw = json.dumps({
        "deltas": [{
            "field":          "stem_pith",
            "canonical_says": "(not specified)",
            "image_shows":    "split lower stem reveals white pith with a brown outer vascular ring",
            "image_quote":    "the cut stem shows white center surrounded by chocolate-brown vascular tissue",
        }],
        "confidence": "high",
        "reasoning":  "decisive SDS fork",
    })
    deltas, conf, why = parse_agent_output(raw, owned_fields=["stem_pith"])
    assert len(deltas) == 1
    assert deltas[0]["field"] == "stem_pith"
    assert conf == "high"
    assert "decisive" in why


def test_parse_agent_output_demotes_unknown_field_to_other():
    """Agent that hallucinates a field outside its owned list is
    coerced to 'other' rather than dropped."""
    from agents.base_agent import parse_agent_output
    import json
    raw = json.dumps({
        "deltas": [{
            "field":          "treatments",          # NOT in owned_fields
            "canonical_says": "(not specified)",
            "image_shows":    "label of a fungicide bottle visible",
            "image_quote":    "white bottle, blue cap, fungicide text",
        }],
        "confidence": "medium",
        "reasoning":  "off-topic",
    })
    deltas, _conf, _why = parse_agent_output(raw, owned_fields=["leaf_lesion_shape"])
    assert len(deltas) == 1
    assert deltas[0]["field"] == "other"
