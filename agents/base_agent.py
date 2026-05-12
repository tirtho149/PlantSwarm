"""
agents/base_agent.py
====================
Base class for the Qwen delta-extraction swarm.

Each specialist agent owns a slice of canonical KB fields. Given a
canonical KB block for a (crop, disease) and a single Bugwood field
photograph, the agent looks at the image, compares against the canonical
text for the fields it owns, and emits structured deltas — additions or
contradictions backed by the photo. It never restates canonical text.

The consolidator (DiagnosisAgent) takes the union of specialist deltas,
the full canonical block, and the image, then dedupes and drops
restatements to produce the final delta list.

Output delta schema (matches pathome.RegionalDelta):
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
from typing import Any, Dict, List, Optional

from utils.vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Delta field vocabulary (mirrors pathome_kb regional schema)
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


# Map each owned delta field → the canonical KB key(s) that describe it.
# Used by specialists to show their slice of canonical to the model.
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
# Helpers
# ---------------------------------------------------------------------------

def _clean(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        return "; ".join(str(x) for x in s if x is not None and str(x).strip())
    return str(s).strip()


def _validate_delta(d: Any, allowed_fields: set) -> Optional[Dict[str, str]]:
    """Normalize one model-emitted delta into the schema, or None to drop it."""
    if not isinstance(d, dict):
        return None
    image_shows = _clean(d.get("image_shows"))
    if not image_shows:
        return None
    field = _clean(d.get("field")).lower() or "other"
    if field not in allowed_fields:
        field = "other"
    return {
        "field":          field,
        "canonical_says": _clean(d.get("canonical_says")) or "(not specified)",
        "image_shows":    image_shows,
        "image_quote":    _clean(d.get("image_quote")),
    }


def _parse_delta_json(text: str, allowed_fields: set) -> List[Dict[str, str]]:
    """Strict-ish JSON parse → list of validated deltas. Returns []."""
    if not text:
        return []
    # Drop ``` fences
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    obj: Any
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return []
    raw = obj.get("deltas") if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for d in raw:
        v = _validate_delta(d, allowed_fields)
        if v is not None:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Prompt template (shared by all specialists)
# ---------------------------------------------------------------------------

DELTA_USER_PROMPT = """\
You are looking at one Bugwood Network field photograph of a plant
disease. The canonical knowledge base for this disease is given below.
Your job is to look at the IMAGE and compare it to the canonical KB for
the FIELDS YOU OWN — then emit ONLY structured deltas that ADD to or
CONTRADICT canonical with visual evidence from this specific photograph.

Crop:    {crop}
Disease: {disease}
State:   {state}

CANONICAL KB (slice for {agent_name}; DO NOT re-describe these contents):
{canonical_slice}

You own these delta fields: {owned_fields}

For each owned field, decide:
- Does the image show something the canonical text for that field does
  NOT capture?
- Does the image CONTRADICT the canonical text for that field?
If yes, emit a delta. If the image confirms canonical exactly for that
field, do not emit a delta for it.

Output STRICT JSON, no markdown fences, no preamble:
{{
  "deltas": [
    {{
      "field":          "<one of: {owned_fields}>",
      "canonical_says": "<short quote from canonical above on this field, or '(not specified)'>",
      "image_shows":    "<what THIS image adds or contradicts — one sentence, state-specific>",
      "image_quote":    "<one-sentence visual evidence — what you literally see in the photo>"
    }}
  ]
}}

Hard rules:
- Each delta MUST be supported by something visible in this photo.
- Do NOT restate canonical text. Restating is forbidden.
- "(not specified)" is the correct canonical_says when canonical is silent.
- If the image confirms canonical exactly, return {{"deltas": []}}.
- Stay strictly within your owned fields. Do not emit deltas for fields
  you do not own.
"""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """Subclasses must set:
        - AGENT_NAME    str
        - OWNED_FIELDS  List[str], subset of ALLOWED_DELTA_FIELDS
        - SYSTEM_PROMPT str
    """

    AGENT_NAME: str = "BaseAgent"
    OWNED_FIELDS: List[str] = []
    SYSTEM_PROMPT: str = (
        "You are a plant pathology vision agent. Output strict JSON only — "
        "no prose, no markdown, no preamble."
    )

    def __init__(self, client: VLLMClient):
        self.client = client

    # ------------------------------------------------------------------
    # Public call
    # ------------------------------------------------------------------

    def extract_deltas(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_b64: str,
    ) -> List[Dict[str, str]]:
        """Return the validated deltas this agent emits for the given image."""
        canonical_slice = self._format_canonical_slice(canonical)
        owned_list = ", ".join(self.OWNED_FIELDS) or "other"
        user_prompt = DELTA_USER_PROMPT.format(
            crop=crop,
            disease=disease,
            state=state,
            agent_name=self.AGENT_NAME,
            canonical_slice=canonical_slice,
            owned_fields=owned_list,
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
        allowed = set(self.OWNED_FIELDS) | {"other"}
        return _parse_delta_json(text, allowed)

    # ------------------------------------------------------------------
    # Canonical-slice rendering (owned-field view)
    # ------------------------------------------------------------------

    def _format_canonical_slice(self, canonical: Dict[str, Any]) -> str:
        """Render only the canonical fields this agent owns, plus pathogen
        identity for grounding."""
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
