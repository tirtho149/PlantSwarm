"""
agents/diagnosis_agent.py
=========================
DiagnosisAgent — the delta consolidator.

Takes the union of the four specialists' candidate deltas plus the FULL
canonical KB and the image, then:

  1. DEDUPES overlapping fields (e.g. diagnostic_features is claimed by
     both MorphologyAgent and SymptomAgent).
  2. DROPS deltas that restate canonical text — restating is forbidden.
  3. KEEPS deltas that add or contradict canonical with image evidence.

Returns the final consolidated list in the same delta schema the
specialists emit.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import (
    ALLOWED_DELTA_FIELDS,
    BaseAgent,
    _clean,
    _parse_delta_json,
)


CONSOLIDATOR_PROMPT = """\
You are the consolidator. Four specialist agents have looked at the
photograph and emitted candidate deltas. Produce the FINAL delta list by:

  1. Deduping overlapping fields — when two candidates target the same
     field with overlapping content, keep the more specific / better-
     grounded one.
  2. Dropping any candidate that just restates canonical text. Restating
     is forbidden.
  3. Keeping candidates that add or contradict canonical with image
     evidence.

Crop:    {crop}
Disease: {disease}
State:   {state}

FULL CANONICAL KB:
{canonical_full}

CANDIDATE DELTAS (from specialists):
{candidates}

Output STRICT JSON, no markdown fences, no preamble:
{{
  "deltas": [
    {{
      "field":          "<one of: {allowed_fields}>",
      "canonical_says": "<short quote from canonical above on this field, or '(not specified)'>",
      "image_shows":    "<state-specific addition or contradiction — one sentence>",
      "image_quote":    "<one-sentence visual evidence — what you see>"
    }}
  ]
}}

If every candidate is a redundant restatement, return {{"deltas": []}}.
"""


class DiagnosisAgent(BaseAgent):
    AGENT_NAME = "DiagnosisAgent"
    # The consolidator can emit deltas for any field — it just dedupes the
    # union of the specialists.
    OWNED_FIELDS = [f for f in ALLOWED_DELTA_FIELDS if f != "other"] + ["other"]

    SYSTEM_PROMPT = (
        "You are DiagnosisAgent, a plant pathology delta consolidator. "
        "Read the candidate deltas, dedupe overlapping fields, drop "
        "restatements of canonical, and return only deltas that add or "
        "contradict canonical with image evidence. Output strict JSON "
        "only — no prose, no markdown."
    )

    def extract_deltas(self, **_kwargs):  # noqa: D401
        raise NotImplementedError(
            "DiagnosisAgent uses consolidate(), not extract_deltas()."
        )

    def consolidate(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_b64: str,
        candidates: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        user_prompt = CONSOLIDATOR_PROMPT.format(
            crop=crop,
            disease=disease,
            state=state,
            canonical_full=self._format_canonical_full(canonical),
            candidates=self._format_candidates(candidates),
            allowed_fields=", ".join(ALLOWED_DELTA_FIELDS),
        )
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        }]
        text, _tokens = self.client.chat(
            messages=messages, system_prompt=self.SYSTEM_PROMPT
        )
        return _parse_delta_json(text, set(ALLOWED_DELTA_FIELDS))

    @staticmethod
    def _format_canonical_full(canonical: Dict[str, Any]) -> str:
        def _v(raw: Any) -> str:
            v = _clean(raw)
            return v or "(not specified)"
        return "\n".join([
            f"  pathogen:            {_v(canonical.get('pathogen_scientific_name'))}",
            f"  type_of_disease:     {_v(canonical.get('type_of_disease'))}",
            f"  affected_parts:      {_v(canonical.get('affected_parts'))}",
            f"  summary:             {_v(canonical.get('summary'))}",
            f"  diagnostic_features: {_v(canonical.get('diagnostic_features'))}",
            f"  look_alikes:         {_v(canonical.get('look_alikes'))}",
            f"  treatments:          {_v(canonical.get('treatments'))}",
        ])

    @staticmethod
    def _format_candidates(candidates: List[Dict[str, str]]) -> str:
        if not candidates:
            return "  (none)"
        lines: List[str] = []
        for i, d in enumerate(candidates, 1):
            lines.append(
                f"  [{i}] field={d.get('field', '')} | "
                f"canonical_says={d.get('canonical_says', '')!s} | "
                f"image_shows={d.get('image_shows', '')!s} | "
                f"image_quote={d.get('image_quote', '')!s}"
            )
        return "\n".join(lines)
