"""
agents/diagnosis_agent.py
=========================
VisualDiagnosisAgent — CoT-walking consolidator over the 24 visual
specialists' outputs.

The consolidator (a) groups specialist outputs by organ family, (b)
explicitly walks the look-alike decision graph (the docx CoT pattern:
"Is the lower stem pith white or brown? White → SDS, brown → BSR"),
(c) emits the FINAL deduplicated delta list for THIS pass, with a
``reasoning`` string that traces the CoT decisions.

Class name ``DiagnosisAgent`` is preserved as an alias for backward
compatibility with ``plantswarm.delta_pipeline``.
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


# How specialists are grouped in the consolidator prompt so the LLM
# sees the same organ-family clustering humans use when diagnosing.
ORGAN_GROUPS: Dict[str, List[str]] = {
    "LEAF FEATURES (8 specialists)": [
        "LeafLesionShapeAgent", "LeafLesionColorAgent", "LeafLesionTextureAgent",
        "LeafChlorosisAgent", "LeafNecrosisAgent", "LeafCurlAgent",
        "LeafVeinPatternAgent", "LeafGeometryAgent",
    ],
    "STEM FEATURES (4 specialists)": [
        "StemLesionAgent", "StemPithAgent", "StemSurfaceAgent",
        "StemDiscolorationAgent",
    ],
    "BELOW-GROUND (2 specialists)": [
        "RootAgent", "CrownCollarAgent",
    ],
    "REPRODUCTIVE (2 specialists)": [
        "FlowerAgent", "FruitAgent",
    ],
    "PATHOGEN SIGNS (1 specialist)": [
        "SporulationAgent",
    ],
    "WHOLE-PLANT PATTERNS (3 specialists)": [
        "WiltingAgent", "DefoliationAgent", "SpatialPatternAgent",
    ],
    "DIAGNOSTIC CROSS-CUTTERS (4 specialists)": [
        "ConcentricPatternAgent", "ColorPaletteAgent",
        "LookAlikeCoTAgent", "SeverityVisualAgent",
    ],
}


CONSOLIDATOR_PROMPT = """\
You are VisualDiagnosisAgent — the consolidator over a 24-specialist
visual swarm. Each specialist asked ONE focused question about this
photograph; their outputs are grouped below by organ family.

Crop:    {crop}
Disease: {disease}
State:   {state}

FULL CANONICAL KB:
{canonical_full}
{existing_kb_block}
SPECIALIST OUTPUTS (24 agents, grouped):
{specialist_block}

Walk a chain-of-thought over the specialist outputs in this order:

  STEP 1 — Triage. List which organs / structures are actually visible
           in this photograph (leaf? stem? cut stem? roots? fruits?).
           Specialists for invisible structures will have returned
           empty deltas with confidence "low"; skip them.

  STEP 2 — Decisive forks. Walk through diagnostic forks from the
           VISIBLE specialists' outputs. Examples from the look-alike
           CoT reference:
             - Stem pith color (white → SDS, brown → BSR)
             - Petiole-attached vs petioles-dropped defoliation
               (attached → SDS, both dropped → BSR / other)
             - Concentric rings present (target spot → Early Blight)
             - Visible cysts on roots (SCN), blue masses on taproot (SDS)
             - Bracts ≥3mm extending beyond tepals (Palmer amaranth)

  STEP 3 — Dedup + drop restatements. When two specialists target
           overlapping content, keep the more specific / better-
           grounded one. Drop anything that just restates canonical
           OR an existing KB observation.

  STEP 4 — Emit the FINAL delta list for this pass, plus a 1-2 line
           ``reasoning`` string that traces your CoT decisions.

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
  "reasoning":  "<CoT trace: visible organs -> decisive forks -> final deltas>"
}}

If every specialist output is a redundant restatement of canonical /
existing KB, return:
  {{"deltas": [], "confidence": "high", "reasoning": "all canonical confirmed"}}.
"""


class DiagnosisAgent(BaseAgent):
    """VisualDiagnosisAgent under the original class name (preserved
    for ``plantswarm.delta_pipeline`` imports and existing tests)."""

    AGENT_NAME = "VisualDiagnosisAgent"
    OWNED_FIELDS = [f for f in ALLOWED_DELTA_FIELDS if f != "other"] + ["other"]

    SYSTEM_PROMPT = (
        "You are VisualDiagnosisAgent, the consolidator. Read 24 "
        "parallel visual specialist outputs grouped by organ family, "
        "walk the look-alike decision-graph CoT, dedupe overlapping "
        "fields, drop restatements of canonical AND existing KB, and "
        "return the final pass delta list. Output strict JSON only — "
        "no prose, no markdown."
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
        # Visual-symptoms triad is what specialists compared against,
        # so emphasize those; pathogen/type for context only.
        return "\n".join([
            f"  pathogen:            {_v(canonical.get('pathogen_scientific_name'))}",
            f"  type_of_disease:     {_v(canonical.get('type_of_disease'))}",
            f"  summary:             {_v(canonical.get('summary'))}",
            f"  diagnostic_features: {_v(canonical.get('diagnostic_features'))}",
            f"  look_alikes:         {_v(canonical.get('look_alikes'))}",
        ])

    @staticmethod
    def _format_specialist_outputs(buf: List[AgentDeltaOutput]) -> str:
        if not buf:
            return "  (empty)"
        # Index specialist outputs by AGENT_NAME for grouped rendering.
        by_name: Dict[str, AgentDeltaOutput] = {o.agent_name: o for o in buf}
        lines: List[str] = []
        for group_name, agent_names in ORGAN_GROUPS.items():
            lines.append("")
            lines.append(f"  --- {group_name} ---")
            group_has_content = False
            for an in agent_names:
                out = by_name.get(an)
                if out is None:
                    lines.append(f"    [{an}] (not run)")
                    continue
                tag = f"(confidence={out.confidence})"
                lines.append(f"    [{an}] {tag}")
                if out.reasoning:
                    lines.append(f"        reasoning: {out.reasoning}")
                if not out.deltas:
                    lines.append("        (no deltas emitted)")
                    continue
                group_has_content = True
                for d in out.deltas:
                    lines.append(
                        f"        delta[{d.get('field','')}]"
                        f" canonical={d.get('canonical_says','')!s}"
                        f" image={d.get('image_shows','')!s}"
                        f" evidence={d.get('image_quote','')!s}"
                    )
            if not group_has_content:
                lines.append("    (group produced no deltas)")
        # Also append any specialists not in the canonical groups
        # (forward-compat for future agents).
        known = {a for v in ORGAN_GROUPS.values() for a in v}
        extras = [o for o in buf if o.agent_name not in known]
        if extras:
            lines.append("")
            lines.append("  --- UNGROUPED SPECIALISTS ---")
            for out in extras:
                lines.append(f"    [{out.agent_name}] (confidence={out.confidence})")
                for d in out.deltas:
                    lines.append(
                        f"        delta[{d.get('field','')}]"
                        f" image={d.get('image_shows','')!s}"
                    )
        return "\n".join(lines)
