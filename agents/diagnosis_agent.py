"""
agents/diagnosis_agent.py
=========================
DiagnosisAgent — terminal aggregator for one routed trace.

Runs once at the end of every trace. Reads the full context buffer
(every specialist's deltas + confidence + reasoning), the full canonical
KB, the image, AND the existing regional KB observations for this
state. Returns the trace's final delta list — deduped, with restatements
of canonical or existing KB dropped.

Cross-run aggregation (N stochastic traces → final delta set) and
conservative merge with the existing KB are handled by
``plantswarm.delta_pipeline``, not by this agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base_agent import (
    ALLOWED_DELTA_FIELDS,
    AgentDeltaOutput,
    BaseAgent,
    _clean,
    parse_agent_output,
)


CONSOLIDATOR_PROMPT = """\
You are the consolidator for one routed swarm trace. Below is the
context buffer for this trace — every specialist that ran, with the
deltas it emitted and its confidence. Your job is to produce the FINAL
delta list for THIS trace by:

  (1) Deduping overlapping fields — when two specialists target the
      same field with overlapping content, keep the more specific /
      better-grounded one.
  (2) Dropping any candidate that just restates canonical text OR
      restates an existing KB observation. Both kinds of restatement
      are forbidden.
  (3) Keeping candidates that add or contradict canonical / existing KB
      with image evidence.

Crop:    {crop}
Disease: {disease}
State:   {state}

FULL CANONICAL KB:
{canonical_full}
{existing_kb_block}
TRACE CONTEXT BUFFER:
{context_buffer}

Output STRICT JSON, no markdown fences, no preamble:
{{
  "deltas": [
    {{
      "field":          "<one of: {allowed_fields}>",
      "canonical_says": "<short quote from canonical above, or '(not specified)'>",
      "image_shows":    "<state-specific addition or contradiction — one sentence>",
      "image_quote":    "<one-sentence visual evidence>"
    }}
  ],
  "confidence":     "high" | "medium" | "low",
  "handoff_target": null,
  "reasoning":      "<one-line summary of what survived>"
}}

If every candidate is a redundant restatement, return
{{"deltas": [], "confidence": "...", "handoff_target": null, "reasoning": "..."}}.
"""


class DiagnosisAgent(BaseAgent):
    AGENT_NAME = "DiagnosisAgent"
    OWNED_FIELDS = [f for f in ALLOWED_DELTA_FIELDS if f != "other"] + ["other"]
    HANDOFF_MENU: List[str] = []
    DEFAULT_FORWARD = ""

    SYSTEM_PROMPT = (
        "You are DiagnosisAgent, the terminal consolidator for one trace. "
        "Read the context buffer, dedupe overlapping fields, drop "
        "restatements of canonical AND existing KB, and return the final "
        "trace delta list. Output strict JSON only — no prose, no markdown."
    )

    def extract_with_routing(self, **_kwargs):  # noqa: D401
        raise NotImplementedError(
            "DiagnosisAgent uses consolidate(), not extract_with_routing()."
        )

    def consolidate(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_data_url: str,
        context_buffer: List[AgentDeltaOutput],
        existing_kb_deltas: Optional[List[Dict[str, Any]]] = None,
        seed: int = 0,
        temperature: float = 0.2,
    ) -> AgentDeltaOutput:
        existing_block = self._format_existing_kb(existing_kb_deltas or [], state)
        user_prompt = CONSOLIDATOR_PROMPT.format(
            crop=crop,
            disease=disease,
            state=state,
            canonical_full=self._format_canonical_full(canonical),
            existing_kb_block=existing_block,
            context_buffer=self._format_context_buffer(context_buffer),
            allowed_fields=", ".join(ALLOWED_DELTA_FIELDS),
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text",      "text":      user_prompt},
            ],
        }]
        text, _tokens = self.client.chat(
            messages=messages,
            system_prompt=self.SYSTEM_PROMPT,
            seed=seed,
            temperature=temperature,
        )
        deltas, confidence, _handoff, reasoning = parse_agent_output(
            text=text,
            owned_fields=list(ALLOWED_DELTA_FIELDS),
            handoff_menu=[],
        )
        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=deltas,
            confidence=confidence,
            handoff_target=None,
            reasoning=reasoning,
            raw_text=text,
        )

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
    def _format_context_buffer(buf: List[AgentDeltaOutput]) -> str:
        if not buf:
            return "  (empty)"
        lines: List[str] = []
        for step, out in enumerate(buf, 1):
            lines.append(f"  [{step}] {out.agent_name} (confidence={out.confidence})")
            if out.reasoning:
                lines.append(f"      reasoning: {out.reasoning}")
            if not out.deltas:
                lines.append("      (no deltas emitted)")
                continue
            for d in out.deltas:
                lines.append(
                    f"      delta[{d.get('field','')}]"
                    f" canonical={d.get('canonical_says','')!s}"
                    f" image={d.get('image_shows','')!s}"
                    f" evidence={d.get('image_quote','')!s}"
                )
        return "\n".join(lines)
