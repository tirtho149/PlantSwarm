"""
agents/base_agent.py
====================
Base class for the Qwen delta-extraction swarm — paper-faithful routing
edition with iterative KB context (PlantSwarm §4 / Algorithm 1, adapted
for deltas).

Each call this agent makes returns four things:

    - deltas          : list of {field, canonical_says, image_shows, image_quote}
                        for the canonical fields this agent owns
    - confidence (κ)  : "high" | "medium" | "low"
    - handoff_target  : the agent that should run next, or None to terminate
    - reasoning       : one-line free-text justification

The agent is shown three context blocks:

  1. CANONICAL KB (slice for this agent's owned fields)
  2. EXISTING KB OBSERVATIONS for this state (from prior Phase 0R runs)
  3. PRIOR TRACE CONTEXT (deltas emitted earlier in THIS trace)

It is instructed to emit ONLY ADDITIONS or CONTRADICTIONS — restating
canonical or restating already-captured KB observations is forbidden.

The orchestrator (``plantswarm.delta_pipeline``) overrides the model's
chosen handoff when paper Algorithm 1 dictates a different one.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Delta field vocabulary
# ---------------------------------------------------------------------------

ALLOWED_DELTA_FIELDS = (
    "lesion_morphology",
    "severity",
    "affected_organs",
    "spread_pattern",
    "diagnostic_features",
    "look_alikes",
    "treatments",
    "type_of_disease",
    "other",
)

CONFIDENCE_LEVELS = ("high", "medium", "low")

_DELTA_FIELD_TO_CANONICAL = {
    "lesion_morphology":   ("summary",),
    "affected_organs":     ("affected_parts",),
    "diagnostic_features": ("diagnostic_features",),
    "spread_pattern":      ("notes",),
    "look_alikes":         ("look_alikes",),
    "type_of_disease":     ("type_of_disease",),
    "severity":            (),
    "treatments":          ("treatments",),
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentDeltaOutput:
    agent_name: str
    deltas: List[Dict[str, str]] = field(default_factory=list)
    confidence: str = "medium"
    handoff_target: Optional[str] = None
    reasoning: str = ""
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _clean(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        return "; ".join(str(x) for x in s if x is not None and str(x).strip())
    return str(s).strip()


def _validate_delta(d: Any, allowed_fields: set) -> Optional[Dict[str, str]]:
    if not isinstance(d, dict):
        return None
    image_shows = _clean(d.get("image_shows"))
    if not image_shows:
        return None
    fld = _clean(d.get("field")).lower() or "other"
    if fld not in allowed_fields:
        fld = "other"
    return {
        "field":          fld,
        "canonical_says": _clean(d.get("canonical_says")) or "(not specified)",
        "image_shows":    image_shows,
        "image_quote":    _clean(d.get("image_quote")),
    }


def _coerce_confidence(c: Any) -> str:
    s = _clean(c).lower()
    if s in CONFIDENCE_LEVELS:
        return s
    tokens = re.findall(r"[a-z]+", s)
    for level in CONFIDENCE_LEVELS:
        if level in tokens:
            return level
    return "medium"


def _coerce_handoff(t: Any, menu: List[str]) -> Optional[str]:
    s = _clean(t)
    if not s or s.lower() in ("none", "null", "terminate"):
        return None
    for name in menu:
        if name.lower() == s.lower():
            return name
    for name in menu:
        if name.lower() in s.lower():
            return name
    return None


def parse_agent_output(
    text: str,
    owned_fields: List[str],
    handoff_menu: List[str],
) -> Tuple[List[Dict[str, str]], str, Optional[str], str]:
    """Return (deltas, confidence, handoff_target, reasoning)."""
    if not text:
        return [], "medium", None, ""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    obj: Any = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return [], "medium", None, ""
    allowed_fields = set(owned_fields) | {"other"}
    deltas: List[Dict[str, str]] = []
    for d in obj.get("deltas") or []:
        v = _validate_delta(d, allowed_fields)
        if v is not None:
            deltas.append(v)
    confidence = _coerce_confidence(obj.get("confidence"))
    handoff    = _coerce_handoff(obj.get("handoff_target"), handoff_menu)
    reasoning  = _clean(obj.get("reasoning"))
    return deltas, confidence, handoff, reasoning


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

DELTA_USER_PROMPT = """\
You are looking at one Bugwood Network field photograph of a plant
disease. The canonical knowledge base for this disease is given below,
along with any observations already captured for THIS state in prior
runs.

Crop:    {crop}
Disease: {disease}
State:   {state}

CANONICAL KB (slice for {agent_name} — do NOT re-describe these contents):
{canonical_slice}
{existing_kb_block}{prior_context}\
Your job has two parts:

(1) DELTAS — Look at the IMAGE and compare it to BOTH the canonical KB
    above AND the existing KB observations. For each of YOUR owned
    fields, decide:
      · Does the image show something canonical AND existing KB do NOT
        capture?                                                  → emit a delta
      · Does the image CONTRADICT canonical or existing KB?
                                                                  → emit a delta
      · Does the image CONFIRM what's already captured?            → do NOT emit
    Each delta MUST be supported by something visible in this photo.
    Restating canonical or already-captured KB text is forbidden.

(2) ROUTING — Report your confidence and pick the next agent:
      · confidence: "high" if your deltas are well-grounded; "medium"
        if some are uncertain; "low" if the image is ambiguous.
      · handoff_target: pick ONE of {handoff_menu_str}.

You own these delta fields: {owned_fields}

Output STRICT JSON, no markdown fences, no preamble:
{{
  "deltas": [
    {{
      "field":          "<one of: {owned_fields}>",
      "canonical_says": "<short quote from canonical above on this field, or '(not specified)'>",
      "image_shows":    "<state-specific addition or contradiction — one sentence>",
      "image_quote":    "<one-sentence visual evidence — what you literally see>"
    }}
  ],
  "confidence":     "high" | "medium" | "low",
  "handoff_target": "<one of {handoff_menu_str}>",
  "reasoning":      "<one-line justification for confidence + handoff>"
}}

If the image confirms canonical AND existing KB exactly, return
{{"deltas": [], "confidence": "high", "handoff_target": "DiagnosisAgent", "reasoning": "..."}}.
"""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """Subclasses must set AGENT_NAME, OWNED_FIELDS, HANDOFF_MENU,
    DEFAULT_FORWARD, SYSTEM_PROMPT.
    """

    AGENT_NAME: str = "BaseAgent"
    OWNED_FIELDS: List[str] = []
    HANDOFF_MENU: List[str] = []
    DEFAULT_FORWARD: str = "DiagnosisAgent"
    SYSTEM_PROMPT: str = (
        "You are a plant pathology vision agent. Output strict JSON only — "
        "no prose, no markdown, no preamble."
    )

    def __init__(self, client: VLLMClient):
        self.client = client

    def extract_with_routing(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_data_url: str,
        prior_context: List["AgentDeltaOutput"],
        existing_kb_deltas: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AgentDeltaOutput:
        """One stochastic step of the routed swarm.

        ``existing_kb_deltas`` is the persisted regional KB for THIS
        state from prior Phase 0R runs (empty on cold start). The agent
        is told to NOT restate them — only to add new observations or
        flag contradictions.
        """
        canonical_slice = self._format_canonical_slice(canonical)
        existing_block = self._format_existing_kb(existing_kb_deltas or [], state)
        prior_block = self._format_prior_context(prior_context)
        owned_list = ", ".join(self.OWNED_FIELDS) or "other"
        menu_str = ", ".join(self.HANDOFF_MENU) or "DiagnosisAgent"

        user_prompt = DELTA_USER_PROMPT.format(
            crop=crop,
            disease=disease,
            state=state,
            agent_name=self.AGENT_NAME,
            canonical_slice=canonical_slice,
            existing_kb_block=existing_block,
            prior_context=prior_block,
            owned_fields=owned_list,
            handoff_menu_str=menu_str,
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

        deltas, confidence, handoff, reasoning = parse_agent_output(
            text=text,
            owned_fields=self.OWNED_FIELDS,
            handoff_menu=self.HANDOFF_MENU,
        )

        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=deltas,
            confidence=confidence,
            handoff_target=handoff,
            reasoning=reasoning,
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Context rendering
    # ------------------------------------------------------------------

    def _format_canonical_slice(self, canonical: Dict[str, Any]) -> str:
        lines: List[str] = []
        if canonical.get("pathogen_scientific_name"):
            lines.append(f"  pathogen: {_clean(canonical['pathogen_scientific_name'])}")
        if (
            canonical.get("type_of_disease")
            and "type_of_disease" not in self.OWNED_FIELDS
        ):
            lines.append(f"  type: {_clean(canonical['type_of_disease'])}")
        for owned in self.OWNED_FIELDS:
            canon_keys = _DELTA_FIELD_TO_CANONICAL.get(owned, ())
            value = ""
            for key in canon_keys:
                raw = canonical.get(key)
                if raw:
                    value = _clean(raw)
                    if value:
                        break
            lines.append(f"  {owned}: {value or '(not specified)'}")
        return "\n".join(lines) if lines else "  (canonical not available)"

    @staticmethod
    def _format_existing_kb(
        existing: List[Dict[str, Any]],
        state: str,
    ) -> str:
        """Render existing regional KB deltas for THIS state.

        Returns a block with a leading blank line and a trailing newline so
        prompt concatenation stays readable. Empty when there are no prior
        observations (cold start).
        """
        if not existing:
            return ""
        lines = ["", f"EXISTING KB OBSERVATIONS for {state} "
                     f"(from prior runs — preserve, do NOT restate):"]
        for d in existing:
            fld = d.get("field", "other")
            support = d.get("__support__") or d.get("support") or 0
            tag = f" (support={support})" if support else ""
            shows = d.get("image_shows", "")
            if len(shows) > 220:
                shows = shows[:220].rstrip() + "…"
            lines.append(f"  • [{fld}]{tag}")
            if d.get("canonical_says") and d["canonical_says"] != "(not specified)":
                lines.append(f"      canonical: {d['canonical_says']}")
            lines.append(f"      previously observed: {shows}")
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_prior_context(prior: List["AgentDeltaOutput"]) -> str:
        if not prior:
            return ""
        lines: List[str] = ["", "PRIOR TRACE CONTEXT (this run, most recent last):"]
        for step, out in enumerate(prior, 1):
            lines.append(f"  [{step}] {out.agent_name} (confidence={out.confidence})")
            if out.reasoning:
                lines.append(f"      reasoning: {out.reasoning}")
            if out.deltas:
                for d in out.deltas:
                    img = d.get("image_shows", "")
                    if len(img) > 200:
                        img = img[:200].rstrip() + "…"
                    lines.append(f"      delta[{d.get('field','')}]: {img}")
            else:
                lines.append(f"      (no deltas emitted)")
        lines.append("")
        return "\n".join(lines) + "\n"
