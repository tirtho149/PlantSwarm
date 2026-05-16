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

import os
from typing import Any, Dict, List, Optional

from agents.base_agent import (
    ALLOWED_DELTA_FIELDS,
    AgentDeltaOutput,
    BaseAgent,
    _clean,
    parse_agent_output,
)


def _consolidator_max_new_tokens() -> int:
    """Token budget for the single consolidator call per image.

    The consolidator restates/dedupes deltas from ~22 specialists, so
    its JSON is far longer than any individual specialist's. At the
    shared specialist budget (``VLLM_MAX_NEW_TOKENS``, default 512) it
    truncates mid-string on leaf-heavy profiles and the parser then
    drops every delta for that image. Give it its own, larger budget.
    """
    try:
        return int(os.environ.get("VLLM_CONSOLIDATOR_MAX_NEW_TOKENS", "2048"))
    except (TypeError, ValueError):
        return 2048


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
real swarm with TWO rounds. Each specialist asked ONE focused question
about this photograph in ROUND 1 (independent observation), then ran
AGAIN in ROUND 2 with the full peer blackboard visible — and could
support, challenge, or withdraw against peers (cross_refs).

Crop:    {crop}
Disease: {disease}
State:   {state}

FULL CANONICAL KB:
{canonical_full}
{existing_kb_block}
ROUND-1 OUTPUTS (independent observation; 24 agents, grouped):
{round1_block}

ROUND-2 OUTPUTS (stigmergy round; same 24 agents AFTER reading the
round-1 blackboard — cross_refs declare support / challenge /
withdraw against peers):
{round2_block}

CROSS-REF DIGEST (peer interactions from round 2):
{cross_ref_block}

Walk a chain-of-thought over the swarm outputs in this order:

  STEP 1 — Triage. Which organs / structures are actually visible in
           this photograph (leaf? cut stem? roots? fruits?)?
           Specialists whose owned organ is not visible will have
           returned low-confidence empty outputs; skip them.

  STEP 2 — Decisive forks. Walk through diagnostic forks from the
           visible specialists' outputs. Use the round-2 refinements
           preferentially because peers have already cross-checked.
           Look-alike fork examples:
             - Stem pith color (white → SDS, brown → BSR)
             - Petiole-attached vs petioles-dropped defoliation
             - Concentric rings (target spot → Early Blight)
             - Visible cysts (SCN), blue masses on taproot (SDS)
             - Bract length / leaf petiole vs blade (Palmer vs waterhemp)

  STEP 3 — Adjudicate cross-refs. For each CHALLENGE in the cross-
           ref digest, decide who is right based on visual evidence.
           For each SUPPORT, raise confidence in the supported delta.
           For each WITHDRAW, drop the original delta.

  STEP 4 — Dedup + drop restatements. When two specialists target
           overlapping content, keep the more specific / better-
           grounded one. Drop anything that just restates canonical
           OR an existing KB observation.

  STEP 5 — Emit the FINAL delta list for this pass, plus a 1-2 line
           ``reasoning`` string that traces your CoT decisions
           (mention any decisive cross_refs that affected the
           outcome).

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
  "reasoning":  "<CoT trace: visible organs -> decisive forks -> cross-ref adjudication -> final deltas>"
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

        # Split specialist outputs by round. If the caller passed a
        # legacy flat list (round_idx defaults to 1 everywhere) the
        # round-2 block will be empty — backwards compatible.
        round1 = [o for o in specialist_outputs if o.round_idx == 1]
        round2 = [o for o in specialist_outputs if o.round_idx == 2]
        cross_refs = self._collect_cross_refs(round2)

        user_prompt = CONSOLIDATOR_PROMPT.format(
            crop=crop, disease=disease, state=state,
            canonical_full=self._format_canonical_full(canonical),
            existing_kb_block=existing_block,
            round1_block=self._format_specialist_outputs(round1)
                          if round1 else "  (no round-1 outputs)",
            round2_block=self._format_specialist_outputs(round2)
                          if round2 else "  (round 2 did not run — legacy single-round mode)",
            cross_ref_block=self._format_cross_refs(cross_refs),
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
            max_new_tokens=_consolidator_max_new_tokens(),
        )
        deltas, confidence, reasoning, _cross_refs = parse_agent_output(
            text=text, owned_fields=list(ALLOWED_DELTA_FIELDS),
        )
        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=deltas, confidence=confidence, reasoning=reasoning,
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Cross-ref aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_cross_refs(round2: List[AgentDeltaOutput]) -> List[Dict[str, str]]:
        """Flatten round-2 cross_refs into a single tagged list."""
        out: List[Dict[str, str]] = []
        for o in round2:
            for c in o.cross_refs or []:
                out.append({
                    "from":         o.agent_name,
                    "action":       c.get("action", ""),
                    "target_agent": c.get("target_agent", ""),
                    "rationale":    c.get("rationale", ""),
                })
        return out

    @staticmethod
    def _format_cross_refs(refs: List[Dict[str, str]]) -> str:
        if not refs:
            return "  (no cross_refs declared — round 2 added refinements but no peer challenges)"
        # Group by action for legibility.
        by_action: Dict[str, List[Dict[str, str]]] = {}
        for r in refs:
            by_action.setdefault(r["action"], []).append(r)
        lines: List[str] = []
        for action in ("challenge", "support", "withdraw"):
            items = by_action.get(action) or []
            if not items:
                continue
            lines.append(f"  --- {action.upper()} ({len(items)}) ---")
            for r in items:
                target = r["target_agent"] or "(self)"
                lines.append(
                    f"    {r['from']} -> {target}: {r['rationale']}"
                )
        return "\n".join(lines) if lines else "  (no cross_refs)"

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
