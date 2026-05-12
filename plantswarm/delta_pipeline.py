"""
plantswarm/delta_pipeline.py
============================
Qwen-swarm regional delta extraction.

Per (crop, disease, state, cached Bugwood image):
    1. Flatten the SAGE canonical KB record into a plain dict of values.
    2. Run the 4 specialist agents — each sees the canonical slice for
       its OWNED_FIELDS and the image, and emits candidate deltas.
    3. Run DiagnosisAgent as consolidator: it dedupes overlapping fields,
       drops restatements of canonical, and returns the final delta list.

Output (per state) matches what pathome_kb.symptoms_adapter expects:
    {
      "state":         "Alabama",
      "deltas":        [{field, canonical_says, image_shows, image_quote, image_id}, ...],
      "__image_ids__": ["bugwood::1568038", ...],
    }

Configuration
-------------
The vLLM endpoint is read from environment at client-build time:
    VLLM_BASE_URL   (default: http://localhost:8000/v1)
    VLLM_MODEL      (default: Qwen/Qwen2.5-VL-7B-Instruct)
    VLLM_TIMEOUT    (seconds, default: 180)
    VLLM_TEMPERATURE (default: 0.2 — deterministic-ish, JSON-friendly)

This module is import-light: it only constructs a client when
``run_for_state`` (or ``run_batch``) is actually called.
"""

from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agents.base_agent import BaseAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from agents.pathogen_agent import PathogenAgent
from agents.severity_agent import SeverityAgent
from agents.symptom_agent import SymptomAgent
from utils.vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def build_client_from_env() -> VLLMClient:
    """Build a VLLMClient from environment variables.

    The pipeline needs an OpenAI-compatible vLLM endpoint serving a
    vision-capable Qwen model. On Mac (no CUDA) point this at a remote
    endpoint via SSH tunnel; on Nova, run vLLM in the same SLURM job.
    """
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model    = os.environ.get("VLLM_MODEL",    "Qwen/Qwen2.5-VL-7B-Instruct")
    timeout  = int(os.environ.get("VLLM_TIMEOUT", "180"))
    try:
        temperature = float(os.environ.get("VLLM_TEMPERATURE", "0.2"))
    except ValueError:
        temperature = 0.2
    client = VLLMClient(
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    # Delta extraction does not need per-token logprobs.
    client.chat_request_logprobs = False
    return client


# ---------------------------------------------------------------------------
# Canonical flattener
# ---------------------------------------------------------------------------

def flatten_canonical(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a SAGE final_registry.json disease record to plain values.

    The registry stores each field as ``{"value": ..., "url": ..., "quote": ...}``
    (or a list of those). Here we keep only the ``value`` for prompt
    rendering — agents do not need the provenance metadata.
    """
    def _v(field: Any) -> Any:
        if not isinstance(field, dict):
            return field
        return field.get("value")

    visual = record.get("visual_symptoms") or {}
    return {
        "summary":                  _v(visual.get("summary"))               or "",
        "diagnostic_features":      _v(visual.get("diagnostic_features"))   or [],
        "look_alikes":              _v(visual.get("look_alikes"))           or [],
        "affected_parts":           _v(record.get("affected_parts"))        or [],
        "treatments":               _v(record.get("treatments"))            or [],
        "pathogen_scientific_name": _v(record.get("pathogen_scientific_name")) or "",
        "type_of_disease":          _v(record.get("type_of_disease"))       or "",
        "notes":                    _v(record.get("notes"))                 or "",
    }


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_b64(path: Path) -> str:
    """Read a cached Bugwood image and return its base64 payload (ASCII)."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Single-state runner
# ---------------------------------------------------------------------------

# Order matters only for log output; agents run independently.
SPECIALIST_CLASSES: Sequence[type] = (
    MorphologyAgent,
    SymptomAgent,
    PathogenAgent,
    SeverityAgent,
)


def run_for_state(
    *,
    crop: str,
    disease: str,
    state: str,
    canonical_record: Dict[str, Any],
    image_path: Path,
    primary_image_id: str,
    client: Optional[VLLMClient] = None,
    parallel_specialists: bool = True,
) -> Dict[str, Any]:
    """Run the swarm on one (crop, disease, state) tuple and return the
    record shape expected by symptoms_adapter.

    ``primary_image_id`` is stamped onto every emitted delta so the
    downstream consumer can attribute deltas to a witness image.
    """
    if client is None:
        client = build_client_from_env()

    canonical = flatten_canonical(canonical_record)
    image_b64 = _load_image_b64(image_path)

    specialists: List[BaseAgent] = [cls(client) for cls in SPECIALIST_CLASSES]

    def _run(ag: BaseAgent) -> List[Dict[str, str]]:
        try:
            return ag.extract_deltas(
                crop=crop,
                disease=disease,
                state=state,
                canonical=canonical,
                image_b64=image_b64,
            )
        except Exception as e:
            print(f"    [{ag.AGENT_NAME}] error: {type(e).__name__}: {e}")
            return []

    candidates: List[Dict[str, str]] = []
    if parallel_specialists:
        with ThreadPoolExecutor(max_workers=len(specialists)) as pool:
            for ds in pool.map(_run, specialists):
                candidates.extend(ds)
    else:
        for ag in specialists:
            candidates.extend(_run(ag))

    consolidator = DiagnosisAgent(client)
    try:
        final = consolidator.consolidate(
            crop=crop,
            disease=disease,
            state=state,
            canonical=canonical,
            image_b64=image_b64,
            candidates=candidates,
        )
    except Exception as e:
        print(f"    [DiagnosisAgent] consolidation failed ({type(e).__name__}: {e}); "
              f"falling back to specialist candidates")
        final = candidates

    for d in final:
        d.setdefault("image_id", primary_image_id)

    return {
        "state":         state,
        "deltas":        final,
        "__image_ids__": [primary_image_id],
    }


# ---------------------------------------------------------------------------
# Batch runner (used by pathome_kb.regional_observation)
# ---------------------------------------------------------------------------

# Work-item tuple: (profile_id, crop, disease, state, image_path, image_ids,
#                   canonical_record, primary_image_id)
WorkItem = Tuple[str, str, str, str, Path, List[str], Dict[str, Any], str]


def run_batch(
    work_items: Iterable[WorkItem],
    *,
    client: Optional[VLLMClient] = None,
    max_parallel: int = 4,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the swarm across many (profile, state) tuples.

    Returns ``{profile_id: {state: record}}`` where ``record`` is the
    dict produced by ``run_for_state``.
    """
    if client is None:
        client = build_client_from_env()

    items = list(work_items)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _worker(item: WorkItem) -> Tuple[str, str, Dict[str, Any]]:
        (profile_id, crop, disease, state, image_path,
         image_ids, canonical_record, primary_image_id) = item
        record = run_for_state(
            crop=crop,
            disease=disease,
            state=state,
            canonical_record=canonical_record,
            image_path=image_path,
            primary_image_id=primary_image_id,
            client=client,
        )
        # Preserve all witness image IDs (not just the primary one).
        record["__image_ids__"] = list(image_ids) or [primary_image_id]
        return profile_id, state, record

    completed = 0
    total = len(items)
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        futures = {pool.submit(_worker, it): it for it in items}
        for fut in as_completed(futures):
            try:
                profile_id, state, record = fut.result()
            except Exception as e:
                it = futures[fut]
                print(f"    ERROR on {it[0]} / {it[3]}: {type(e).__name__}: {e}")
                continue
            completed += 1
            n_deltas = len(record.get("deltas") or [])
            tag = "✓" if n_deltas else "·"
            print(f"    [{completed}/{total}] {tag} {profile_id} / {state}  "
                  f"deltas={n_deltas}")
            results.setdefault(profile_id, {})[state] = record

    return results
