from __future__ import annotations

"""Shared infrastructure for disease registry pipelines.

API helpers, JSON parsing, and Claude CLI wrappers used by both
local and internet pipelines.
"""

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .config import API_MODEL, API_MAX_TOKENS
from .utils import save_file


# ─── Anthropic API ──────────────────────────────────────────────────────────

_API_KEY: str | None = None
_ANTHROPIC_CLIENT = None
_API_KEY_PROBED = False


def _probe_api_key() -> str | None:
    """Look up ANTHROPIC_API_KEY without raising. Returns None when absent.

    Used by api_query() to decide whether to dispatch to the Anthropic SDK
    (key present) or to the claude -p subprocess fallback (key missing).
    """
    global _API_KEY, _API_KEY_PROBED
    if _API_KEY_PROBED:
        return _API_KEY
    _API_KEY_PROBED = True
    _API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    if not _API_KEY:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    _API_KEY = line.split("=", 1)[1].strip()
                    break
    return _API_KEY or None


def _get_api_key() -> str:
    """Get cached API key (loaded once from environment or .env file)."""
    key = _probe_api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env file")
    return key


def _get_api_client():
    """Get cached Anthropic client (instantiated once)."""
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        import anthropic
        _ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=_get_api_key())
    return _ANTHROPIC_CLIENT


def _add_additional_properties_false(schema: dict) -> dict:
    """Recursively add additionalProperties: false to all object types."""
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    schema_type = result.get("type")
    # Handle both "type": "object" and "type": ["object", "null"]
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if is_object:
        result["additionalProperties"] = False
        if "properties" in result:
            result["properties"] = {k: _add_additional_properties_false(v) for k, v in result["properties"].items()}
    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _add_additional_properties_false(result["items"])
    return result


def api_query(
    prompt: str,
    system_prompt: str,
    json_schema: dict | None = None,
    content_blocks: list | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """Call Anthropic API for tool-free structured tasks.

    When ANTHROPIC_API_KEY is set, hits the SDK directly (faster, ~1 round trip
    per call). When absent, falls back to ``claude -p`` so the pipeline runs
    on Claude Code's CLI auth alone — slower (~5×) but no API key required.

    ``content_blocks`` is only honoured by the SDK path (used by the local
    PDF track for native document blocks). The CLI fallback ignores it; the
    internet pipeline never passes content_blocks so this is fine in practice.
    """
    if _probe_api_key() is None:
        # CLI fallback — no API key available. Reuses claude_query so we
        # inherit its env-stripping, JSON-schema enforcement, and timeout.
        # max_turns=5 lets the model use a thinking turn before emitting the
        # structured JSON; max_turns=1 was too tight for the reconciliation
        # schema (~3 KB JSON, ~12 KB prompt) and produced empty results with
        # "Reached maximum number of turns" as the only error.
        return claude_query(
            prompt=prompt,
            system_prompt=system_prompt,
            json_schema=json_schema,
            max_turns=5,
            timeout_secs=300,
        )

    client = _get_api_client()

    if content_blocks:
        user_content = content_blocks + [{"type": "text", "text": prompt}]
    else:
        user_content = prompt

    kwargs = {
        "model": API_MODEL,
        "max_tokens": max_tokens or API_MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }

    if json_schema:
        kwargs["output_config"] = {
            "format": {"type": "json_schema", "schema": _add_additional_properties_false(json_schema)}
        }

    response = client.messages.create(**kwargs)
    return response.content[0].text if response.content else None


# ─── Claude CLI wrapper ─────────────────────────────────────────────────────
#
# Used for canonical KB build (Phase 0): discovery, extraction, reconciliation.
# The previous image-aware variant (`claude_query_with_image`) backed the old
# Claude-based regional observation pass; it has been retired in favour of the
# Qwen swarm in ``plantswarm/delta_pipeline.py``.


def claude_query(
    prompt: str,
    allowed_tools: list[str] | None = None,
    system_prompt: str | None = None,
    json_schema: dict | None = None,
    max_turns: int | None = None,
    timeout_secs: int = 600,
) -> str | None:
    """
    Run a single claude -p query and return the result text.
    Pipes prompt via stdin. Kills process group on timeout.
    """
    tmp_files = []

    try:
        prompt_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        prompt_file.write(prompt)
        prompt_file.close()
        tmp_files.append(prompt_file.name)

        cmd = [
            "claude", "-p", "Follow the instructions provided via stdin.",
            "--output-format", "json",
        ]

        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])

        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

        out_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        out_file.close()
        tmp_files.append(out_file.name)

        # Strip env vars that trigger nested-session detection or wrong credentials
        _strip_prefixes = ("CLAUDE", "CURSOR", "MCP_CONNECTION", "VSCODE", "ELECTRON")
        env = {
            k: v for k, v in os.environ.items()
            if not any(k.startswith(p) for p in _strip_prefixes)
        }
        env.pop("ANTHROPIC_API_KEY", None)  # use logged-in account, not API key

        print("  Running claude -p ...", flush=True)

        err_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".err", delete=False, encoding="utf-8"
        )
        err_file.close()
        tmp_files.append(err_file.name)

        with open(prompt_file.name, "r") as pf, open(out_file.name, "w") as of, open(err_file.name, "w") as ef:
            proc = subprocess.Popen(
                cmd, stdin=pf, stdout=of, stderr=ef, text=True,
                env=env, cwd=str(Path(__file__).parent.parent),
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=timeout_secs)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                print(f"  ERROR: claude -p timed out after {timeout_secs}s (killed)")
                return None

        if returncode != 0:
            stderr_text = Path(err_file.name).read_text().strip()
            print(f"  ERROR: claude -p failed (exit {returncode})")
            if stderr_text:
                print(f"  STDERR: {stderr_text[:500]}")
            return None

        with open(out_file.name, "r") as of:
            stdout = of.read().strip()

        if not stdout:
            print("  WARNING: claude -p returned empty output")
            return None

        try:
            envelope = json.loads(stdout)
            if "structured_output" in envelope and envelope["structured_output"]:
                return json.dumps(envelope["structured_output"])
            return envelope.get("result", stdout)
        except json.JSONDecodeError:
            return stdout

    except subprocess.TimeoutExpired:
        print(f"  ERROR: claude -p timed out after {timeout_secs}s")
        return None
    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


# ─── JSON parsing ───────────────────────────────────────────────────────────


def parse_json_result(raw: str | None, stage_name: str) -> dict:
    """Parse JSON from agent result, with helpful error on failure."""
    if raw is None:
        print(f"  WARNING: {stage_name} returned no result")
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        print(f"  WARNING: {stage_name} returned non-JSON result. Saving raw text.")
        save_file(f"{stage_name}_raw.txt", raw)
        return {}


# ─── Disease Name Matching ─────────────────────────────────────────────────

_MATCH_PROMPT = """\
You are given two lists:
1. INPUT diseases (the canonical folder names we must use):
{input_names}

2. EXTRACTED diseases (names found by the pipeline):
{extracted_names}

For each EXTRACTED disease, decide if it refers to the SAME BIOLOGICAL DISEASE \
as any INPUT disease. Match by biological identity, not lexical similarity. \
Use your plant-pathology knowledge to bridge:
- common name ↔ scientific name (e.g. "Pineapple disease" ↔ "Sett rot" — both \
  refer to Ceratocystis paradoxa infection of sugarcane setts; "White Mold" ↔ \
  "Sclerotinia Stem Rot" — both Sclerotinia sclerotiorum)
- acronym ↔ full name (e.g. "SCMV" ↔ "Sugarcane mosaic virus"; \
  "SCSMV" / "Streak Mosaic" ↔ "Sugarcane streak mosaic virus")
- alternative / regional names (e.g. "Black Sigatoka" ↔ "Mycosphaerella fijiensis leaf spot")
- abbreviated folder names (e.g. "Streak_Mosaic_Scsmv" or \
  "Bacterial_Wilt_Of_X" — strip underscores, expand acronyms before matching)

Be GENEROUS — if an extracted name plausibly refers to the same pathogen / \
condition as an input name, match it. Prefer a match over leaving it unmapped.

Return a mapping from each matching extracted name to its corresponding input name. \
Only include extracted diseases that match an input disease. Skip any that don't match.
"""

_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "mappings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "extracted_name": {"type": "string"},
                    "input_name": {"type": "string"},
                },
                "required": ["extracted_name", "input_name"],
            },
        }
    },
    "required": ["mappings"],
}


def match_names_to_folders(extracted_names: list[str],
                           folder_names: list[str]) -> dict[str, str]:
    """Use LLM to match extracted disease names to canonical folder names.

    Returns dict mapping extracted_name -> folder_name for matches found.
    """
    if not extracted_names or not folder_names:
        return {}

    prompt = _MATCH_PROMPT.format(
        input_names=json.dumps(folder_names, indent=2),
        extracted_names=json.dumps(extracted_names, indent=2),
    )
    raw = api_query(
        prompt=prompt,
        system_prompt="You are a plant pathology expert. Match disease names accurately. Output JSON only.",
        json_schema=_MATCH_SCHEMA,
    )
    result = parse_json_result(raw, "match_names")
    mappings = result.get("mappings", [])

    return {m["extracted_name"]: m["input_name"] for m in mappings}
