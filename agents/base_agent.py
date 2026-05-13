"""
agents/base_agent.py
====================
Base class for the Qwen delta-extraction swarm.

Each specialist agent owns a slice of canonical KB fields. Given the
canonical KB block for one (crop, disease), the existing regional KB
deltas for the target state, and a Bugwood photograph, the agent
emits structured deltas — additions or contradictions backed by image
evidence — for the fields it owns. Restating canonical or existing-KB
text is forbidden.

Algorithm 1 routing was removed: specialists are now invoked in
PARALLEL, their outputs are consolidated by DiagnosisAgent, the union
is filtered by K-of-N cross-pass agreement, validated by a Claude
web-search verifier, and conservatively merged into the KB.

Delta schema (matches pathome.RegionalDelta):
    {
      "field":          str,   # one of ALLOWED_DELTA_FIELDS
      "canonical_says": str,   # short quote, or "(not specified)"
      "image_shows":    str,   # state-specific addition or contradiction
      "image_quote":    str,   # one-sentence visual evidence
    }
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


def parse_agent_output(
    text: str,
    owned_fields: List[str],
) -> Tuple[List[Dict[str, str]], str, str]:
    """Return (deltas, confidence, reasoning)."""
    if not text:
        return [], "medium", ""
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
        return [], "medium", ""
    allowed_fields = set(owned_fields) | {"other"}
    deltas: List[Dict[str, str]] = []
    for d in obj.get("deltas") or []:
        v = _validate_delta(d, allowed_fields)
        if v is not None:
            deltas.append(v)
    confidence = _coerce_confidence(obj.get("confidence"))
    reasoning  = _clean(obj.get("reasoning"))
    return deltas, confidence, reasoning


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
{existing_kb_block}\
Your task — emit deltas for the canonical fields YOU OWN, comparing the
IMAGE to BOTH canonical AND existing KB observations:

  - Image shows something canonical AND existing KB do NOT capture?
                                                          -> emit a delta
  - Image CONTRADICTS canonical or existing KB?           -> emit a delta
  - Image CONFIRMS what's already captured?               -> do NOT emit
Each delta MUST be supported by something visible in this photo.
Restating canonical or already-captured KB text is forbidden.

You own these delta fields: {owned_fields}

Also report your overall confidence in this output: "high" if your
deltas are well-grounded in clear visual evidence; "medium" if some are
uncertain; "low" if the image is ambiguous or you couldn't ground
anything.

Output STRICT JSON, no markdown fences, no preamble:
{{
  "deltas": [
    {{
      "field":          "<one of: {owned_fields}>",
      "canonical_says": "<short quote from canonical above, or '(not specified)'>",
      "image_shows":    "<state-specific addition or contradiction — one sentence>",
      "image_quote":    "<one-sentence visual evidence>"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "reasoning":  "<one-line justification for the confidence>"
}}

If the image confirms canonical AND existing KB exactly, return
{{"deltas": [], "confidence": "high", "reasoning": "..."}}.
"""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """Subclasses must set AGENT_NAME, OWNED_FIELDS, SYSTEM_PROMPT.

    Specialists are invoked in parallel by the orchestrator — no routing
    or handoff state lives on the agent itself.
    """

    AGENT_NAME: str = "BaseAgent"
    OWNED_FIELDS: List[str] = []
    SYSTEM_PROMPT: str = (
        "You are a plant pathology vision agent. Output strict JSON only — "
        "no prose, no markdown, no preamble."
    )

    def __init__(self, client: VLLMClient):
        self.client = client

    def extract_deltas(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_data_url: str,
        existing_kb_deltas: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AgentDeltaOutput:
        """One specialist pass — returns deltas + confidence + reasoning."""
        canonical_slice = self._format_canonical_slice(canonical)
        existing_block  = self._format_existing_kb(existing_kb_deltas or [], state)
        owned_list      = ", ".join(self.OWNED_FIELDS) or "other"

        user_prompt = DELTA_USER_PROMPT.format(
            crop=crop, disease=disease, state=state,
            agent_name=self.AGENT_NAME,
            canonical_slice=canonical_slice,
            existing_kb_block=existing_block,
            owned_fields=owned_list,
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
            text=text, owned_fields=self.OWNED_FIELDS,
        )
        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=deltas, confidence=confidence, reasoning=reasoning,
            raw_text=text,
        )

    # Backwards-compat alias for tests / external callers.
    extract_with_routing = extract_deltas

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
        if not existing:
            return ""
        lines = ["", f"EXISTING KB OBSERVATIONS for {state} "
                     f"(from prior runs — preserve, do NOT restate):"]
        for d in existing:
            fld = d.get("field", "other")
            support = d.get("__support__") or d.get("support") or d.get("swarm_support") or 0
            tag = f" (support={support})" if support else ""
            shows = d.get("image_shows", "")
            if len(shows) > 220:
                shows = shows[:220].rstrip() + "..."
            lines.append(f"  - [{fld}]{tag}")
            if d.get("canonical_says") and d["canonical_says"] != "(not specified)":
                lines.append(f"      canonical: {d['canonical_says']}")
            lines.append(f"      previously observed: {shows}")
        lines.append("")
        return "\n".join(lines) + "\n"
