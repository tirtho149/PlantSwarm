"""
agents/diagnosis_agent.py
=========================
DiagnosisAgent — consolidator over the parallel specialists' outputs.

Receives the union of specialist deltas (Morphology, Symptom, Pathogen,
Severity) plus the canonical KB, existing regional KB, and the image.
Dedupes overlapping fields, drops restatements of canonical / existing
KB, returns the consolidated delta list for this pass.

There is no routing — this agent simply runs once per pass after the
four specialists complete in parallel.
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
You are the consolidator. The four specialist agents (Morphology,
Symptom, Pathogen, Severity) have each examined this photograph in
parallel and emitted candidate deltas. Produce the FINAL delta list
for THIS pass by:

  (1) Deduping overlapping fields. When two specialists target the same
      field with overlapping content, keep the more specific / better-
      grounded one.
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
SPECIALIST OUTPUTS (parallel):
{specialist_block}

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
  "confidence": "high" | "medium" | "low",
  "reasoning":  "<one-line summary of what survived>"
}}

If every candidate is a redundant restatement, return
{{"deltas": [], "confidence": "...", "reasoning": "..."}}.
"""


class DiagnosisAgent(BaseAgent):
    AGENT_NAME = "DiagnosisAgent"
    OWNED_FIELDS = [f for f in ALLOWED_DELTA_FIELDS if f != "other"] + ["other"]

    SYSTEM_PROMPT = (
        "You are DiagnosisAgent, the consolidator. Read the parallel "
        "specialist outputs, dedupe overlapping fields, drop restatements "
        "of canonical AND existing KB, and return the final pass delta "
        "list. Output strict JSON only — no prose, no markdown."
    )

    def extract_deltas(self, **_kwargs):  # noqa: D401
        raise NotImplementedError(
            "DiagnosisAgent uses consolidate(), not extract_deltas()."
        )

    extract_with_routing = extract_deltas      # backwards-compat shim

    def consolidate(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_data_url: str,
        specialist_outputs: List[AgentDeltaOutput],
        existing_kb_deltas: Optional[List[Dict[str, Any]]] = None,
        seed: int = 0,
        temperature: float = 0.2,
    ) -> AgentDeltaOutput:
        existing_block = self._format_existing_kb(existing_kb_deltas or [], state)
        user_prompt = CONSOLIDATOR_PROMPT.format(
            crop=crop, disease=disease, state=state,
            canonical_full=self._format_canonical_full(canonical),
            existing_kb_block=existing_block,
            specialist_block=self._format_specialist_outputs(specialist_outputs),
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
            messages=messages, system_prompt=self.SYSTEM_PROMPT,
            seed=seed, temperature=temperature,
        )
        deltas, confidence, reasoning = parse_agent_output(
            text=text, owned_fields=list(ALLOWED_DELTA_FIELDS),
        )
        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=deltas, confidence=confidence, reasoning=reasoning,
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
    def _format_specialist_outputs(buf: List[AgentDeltaOutput]) -> str:
        if not buf:
            return "  (empty)"
        lines: List[str] = []
        for out in buf:
            lines.append(f"  [{out.agent_name}] (confidence={out.confidence})")
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
