"""
plantswarm/delta_pipeline.py
============================
Paper-faithful Qwen swarm for regional delta extraction (PlantSwarm §4 /
Algorithm 1, adapted for deltas) with iterative KB evolution.

Per (crop, disease, state, cached Bugwood image):

    load existing regional deltas for THIS state from final_registry.json
       ↓
    N stochastic routed traces  →  per-trace consolidated deltas
       (agents see canonical + existing KB deltas as context)
       ↓
    cross-run agreement filter (K-of-N Jaccard clusters)
       ↓
    conservative merge with existing:
       - existing deltas are preserved (idempotent re-runs)
       - new deltas added only if no existing same-field delta has
         Jaccard ≥ τ on image_shows
       - overlapping new deltas bump the existing's `__support__` counter
       ↓
    final regional deltas for THIS state

Algorithm 1 (κ = confidence ∈ {high, medium, low}, b = backtrack count):

    κ=low  AND b < max_backtracks         → MorphologyAgent (regrounding)
    κ=low  AND b >= max_backtracks        → default forward (loop guard)
    κ=high AND all specialists ran        → DiagnosisAgent (early terminate)
    otherwise                             → model's chosen handoff

Output (per state) matches what pathome_kb.symptoms_adapter expects:
    {
      "state":         "Alabama",
      "deltas":        [{field, canonical_says, image_shows, image_quote,
                          image_id, __support__, __cluster_size__}, ...],
      "__image_ids__": ["bugwood::1568038", ...],
      "__swarm_meta__": {n_runs, agreement_min, paths, merge_counts, ...},
    }

Configuration via env vars (read at client-build time):
    VLLM_BASE_URL          default http://localhost:8000/v1
    VLLM_MODEL             default Qwen/Qwen2.5-VL-7B-Instruct
    VLLM_TIMEOUT           seconds per HTTP call (default 180)
    VLLM_TEMPERATURE       per-call sampling temperature (default 0.8)
    VLLM_N_RUNS            stochastic traces per tuple (default 10)
    VLLM_AGREEMENT_MIN     min K-of-N agreement to keep a delta (default 3)
    VLLM_TMAX              max path length per trace (default 15)
    VLLM_MAX_BACKTRACKS    max backtracks per trace (default 1; honoured)
    VLLM_SIM_THRESHOLD     Jaccard threshold for delta clustering AND merge
                           dedup (default 0.4)
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from agents.base_agent import AgentDeltaOutput, BaseAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from agents.pathogen_agent import PathogenAgent
from agents.severity_agent import SeverityAgent
from agents.symptom_agent import SymptomAgent
from utils.vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

SPECIALIST_NAMES = ("MorphologyAgent", "SymptomAgent", "PathogenAgent", "SeverityAgent")

_AGENT_REGISTRY: Dict[str, type] = {
    "MorphologyAgent": MorphologyAgent,
    "SymptomAgent":    SymptomAgent,
    "PathogenAgent":   PathogenAgent,
    "SeverityAgent":   SeverityAgent,
    "DiagnosisAgent":  DiagnosisAgent,
}


def _make_agent(name: str, client: VLLMClient) -> BaseAgent:
    cls = _AGENT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown agent: {name}")
    return cls(client)


# ---------------------------------------------------------------------------
# Swarm config
# ---------------------------------------------------------------------------

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def build_client_from_env() -> VLLMClient:
    """Build a VLLMClient from environment variables."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model    = os.environ.get("VLLM_MODEL",    "Qwen/Qwen2.5-VL-7B-Instruct")
    timeout  = _int_env("VLLM_TIMEOUT", 180)
    temperature = _float_env("VLLM_TEMPERATURE", 0.8)
    client = VLLMClient(
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    client.chat_request_logprobs = False
    return client


# ---------------------------------------------------------------------------
# Canonical flattener
# ---------------------------------------------------------------------------

def flatten_canonical(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a SAGE final_registry.json disease record to plain values."""
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


def existing_deltas_for_state(
    record: Dict[str, Any],
    state: str,
) -> List[Dict[str, Any]]:
    """Pull existing regional deltas for THIS state from a SAGE
    final_registry.json disease record.

    Returns a list of dicts in the same shape the agents emit:
    ``{field, canonical_says, image_shows, image_quote, image_id,
       __support__?, __cluster_size__?}``.
    Empty list when there are no prior deltas (cold start).
    """
    ro = (record.get("regional_observations") or {}).get(state) or {}
    out: List[Dict[str, Any]] = []
    for d in ro.get("deltas") or []:
        if not isinstance(d, dict):
            continue
        if not d.get("image_shows"):
            continue
        entry: Dict[str, Any] = {
            "field":          str(d.get("field") or "other"),
            "canonical_says": str(d.get("canonical_says") or "(not specified)"),
            "image_shows":    str(d.get("image_shows") or "").strip(),
            "image_quote":    str(d.get("image_quote") or "").strip(),
            "image_id":       str(d.get("image_id") or ""),
        }
        # Preserve support telemetry if present (both bracket-style and clean keys).
        for k_src, k_dst in (("__support__", "__support__"),
                             ("support",      "__support__"),
                             ("__cluster_size__", "__cluster_size__"),
                             ("cluster_size",     "__cluster_size__")):
            if k_src in d:
                try:
                    entry[k_dst] = int(d[k_src])
                except (TypeError, ValueError):
                    pass
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Image loading (MIME-aware)
# ---------------------------------------------------------------------------

def _load_image_data_url(path: Path) -> str:
    """Return a ``data:<mime>;base64,...`` URL for the cached image.

    The Bugwood cache resolves .jpg / .jpeg / .png / .webp — using the
    file extension to drive MIME beats hardcoding image/jpeg for PNG /
    WEBP frames.
    """
    p = Path(path)
    mt, _ = mimetypes.guess_type(str(p))
    if not mt or not mt.startswith("image/"):
        mt = "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mt};base64,{b64}"


# ---------------------------------------------------------------------------
# Algorithm 1 — routing decision
# ---------------------------------------------------------------------------

def algorithm1_handoff(
    *,
    current_agent_name: str,
    model_handoff: Optional[str],
    confidence: str,
    backtrack_count: int,
    max_backtracks: int,
    specialists_run: set,
    default_forward: str,
) -> Tuple[Optional[str], str]:
    """Apply paper Algorithm 1 to override the model's chosen handoff.

    Returns (next_agent_or_None, decision_reason). ``None`` means
    terminate. Reason string is for trace logging.
    """
    # Rule 1: low confidence + budget remaining → MorphologyAgent (regrounding).
    if (
        confidence == "low"
        and backtrack_count < max_backtracks
        and current_agent_name != "MorphologyAgent"
    ):
        return "MorphologyAgent", "alg1_low_kappa_backtrack"

    # Rule 2: low confidence + budget exhausted → default forward (loop guard).
    if confidence == "low" and backtrack_count >= max_backtracks:
        return default_forward or "DiagnosisAgent", "alg1_loop_guard_forward"

    # Rule 3: high confidence + all specialists already contributed → terminate.
    if (
        confidence == "high"
        and len(specialists_run) >= len(SPECIALIST_NAMES)
        and current_agent_name != "DiagnosisAgent"
    ):
        return "DiagnosisAgent", "alg1_high_kappa_all_covered_terminate"

    # Rule 4: otherwise use the model's choice (or fall back to default forward).
    if model_handoff:
        return model_handoff, "model_choice"
    return default_forward or "DiagnosisAgent", "default_forward"


# ---------------------------------------------------------------------------
# One stochastic trace
# ---------------------------------------------------------------------------

def _run_single_trace(
    *,
    crop: str,
    disease: str,
    state: str,
    canonical: Dict[str, Any],
    image_data_url: str,
    existing_deltas: List[Dict[str, Any]],
    client: VLLMClient,
    run_idx: int,
    seed: int,
    temperature: float,
    entry_agent: str = "MorphologyAgent",
    Tmax: int = 15,
    max_backtracks: int = 1,
) -> Dict[str, Any]:
    """One routed traversal of the swarm. Returns a trace record."""
    context_buffer: List[AgentDeltaOutput] = []
    path: List[str] = []
    decisions: List[str] = []
    backtrack_count = 0
    specialists_run: set = set()

    current = entry_agent
    early_terminated = False

    while len(path) < Tmax:
        if current == "DiagnosisAgent":
            consolidator = DiagnosisAgent(client)
            out = consolidator.consolidate(
                crop=crop,
                disease=disease,
                state=state,
                canonical=canonical,
                image_data_url=image_data_url,
                context_buffer=context_buffer,
                existing_kb_deltas=existing_deltas,
                seed=seed + 1000,
                temperature=temperature,
            )
            path.append(current)
            context_buffer.append(out)
            decisions.append("terminate_diagnosis")
            early_terminated = True
            break

        agent = _make_agent(current, client)
        out = agent.extract_with_routing(
            crop=crop,
            disease=disease,
            state=state,
            canonical=canonical,
            image_data_url=image_data_url,
            prior_context=context_buffer,
            existing_kb_deltas=existing_deltas,
            seed=seed + len(path),
            temperature=temperature,
        )
        path.append(current)
        context_buffer.append(out)
        if current in SPECIALIST_NAMES:
            specialists_run.add(current)

        nxt, reason = algorithm1_handoff(
            current_agent_name=current,
            model_handoff=out.handoff_target,
            confidence=out.confidence,
            backtrack_count=backtrack_count,
            max_backtracks=max_backtracks,
            specialists_run=specialists_run,
            default_forward=agent.DEFAULT_FORWARD,
        )
        decisions.append(reason)

        if nxt is None:
            break

        if nxt == "MorphologyAgent" and current != "MorphologyAgent":
            backtrack_count += 1

        current = nxt

    # If we hit Tmax without DiagnosisAgent, force a terminal consolidation now.
    if not early_terminated:
        consolidator = DiagnosisAgent(client)
        out = consolidator.consolidate(
            crop=crop,
            disease=disease,
            state=state,
            canonical=canonical,
            image_data_url=image_data_url,
            context_buffer=context_buffer,
            existing_kb_deltas=existing_deltas,
            seed=seed + 1000,
            temperature=temperature,
        )
        path.append("DiagnosisAgent")
        context_buffer.append(out)
        decisions.append("tmax_forced_terminate")

    return {
        "run_idx":          run_idx,
        "path":             path,
        "context_buffer":   context_buffer,
        "final_deltas":     context_buffer[-1].deltas,
        "confidences":      [c.confidence for c in context_buffer],
        "decisions":        decisions,
        "backtrack_count":  backtrack_count,
        "early_terminated": early_terminated,
    }


# ---------------------------------------------------------------------------
# Similarity helpers (shared by agreement filter + merge)
# ---------------------------------------------------------------------------

def _tokenize(s: str) -> set:
    if not s:
        return set()
    out = set()
    for tok in s.lower().split():
        cleaned = "".join(ch for ch in tok if ch.isalnum())
        if cleaned:
            out.add(cleaned)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster_by_similarity(
    items: List[Tuple[int, Dict[str, str]]],
    threshold: float,
) -> List[List[Tuple[int, Dict[str, str]]]]:
    """Greedy any-member Jaccard clustering on image_shows tokens."""
    clusters: List[List[Tuple[int, Dict[str, str]]]] = []
    for run_idx, d in items:
        d_tokens = _tokenize(d.get("image_shows", ""))
        placed = False
        for cluster in clusters:
            for _, member in cluster:
                m_tokens = _tokenize(member.get("image_shows", ""))
                if _jaccard(d_tokens, m_tokens) >= threshold:
                    cluster.append((run_idx, d))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([(run_idx, d)])
    return clusters


def _agreement_filter(
    per_run_deltas: List[List[Dict[str, str]]],
    *,
    min_support: int,
    similarity_threshold: float = 0.4,
) -> List[Dict[str, str]]:
    """Keep deltas surviving K-of-N agreement."""
    all_with_run: List[Tuple[int, Dict[str, str]]] = []
    for run_idx, deltas in enumerate(per_run_deltas):
        for d in deltas or []:
            all_with_run.append((run_idx, d))

    by_field: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for run_idx, d in all_with_run:
        by_field[d.get("field", "other")].append((run_idx, d))

    survivors: List[Dict[str, str]] = []
    for fld, items in by_field.items():
        clusters = _cluster_by_similarity(items, threshold=similarity_threshold)
        for cluster in clusters:
            run_set = {ri for ri, _ in cluster}
            if len(run_set) >= min_support:
                rep = max((d for _, d in cluster),
                          key=lambda d: len(d.get("image_shows", "")))
                rep = dict(rep)
                rep["__support__"]      = len(run_set)
                rep["__cluster_size__"] = len(cluster)
                survivors.append(rep)
    return survivors


# ---------------------------------------------------------------------------
# Conservative merge with existing KB
# ---------------------------------------------------------------------------

def _merge_with_existing(
    *,
    existing: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
    similarity_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Merge ``new`` candidates into ``existing`` deltas conservatively.

    Policy:
      - Every existing delta is preserved (idempotent re-runs).
      - A new delta is added iff no existing delta in the same field has
        Jaccard ≥ ``similarity_threshold`` on ``image_shows``.
      - When a new delta overlaps with an existing one, the existing's
        ``__support__`` is incremented by the new's ``__support__``
        (or 1 if absent). The new delta itself is dropped.

    Returns (merged_list, counts).
    """
    merged: List[Dict[str, Any]] = [dict(e) for e in existing]
    by_field: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(merged):
        by_field[e.get("field", "other")].append(i)

    counts = {
        "n_existing":        len(existing),
        "n_new_candidates":  len(new),
        "n_added":           0,
        "n_overlaps_bumped": 0,
    }

    for n in new:
        n_field = n.get("field", "other")
        n_tokens = _tokenize(n.get("image_shows", ""))
        overlap_idx: Optional[int] = None
        for i in by_field[n_field]:
            e = merged[i]
            e_tokens = _tokenize(e.get("image_shows", ""))
            if _jaccard(n_tokens, e_tokens) >= similarity_threshold:
                overlap_idx = i
                break
        if overlap_idx is not None:
            e = merged[overlap_idx]
            bump = int(n.get("__support__", 1) or 1)
            e["__support__"] = int(e.get("__support__", 1) or 1) + bump
            counts["n_overlaps_bumped"] += 1
        else:
            n_copy = dict(n)
            n_copy.setdefault("__support__", 1)
            merged.append(n_copy)
            by_field[n_field].append(len(merged) - 1)
            counts["n_added"] += 1
    return merged, counts


# ---------------------------------------------------------------------------
# Trace persistence (training data for OBSERVE)
# ---------------------------------------------------------------------------

class _TraceWriter:
    """Append-mode JSONL writer with an fsync after every record.

    Thread-safe — used by ``run_for_state``'s parallel inner pool.
    When ``PATHOME_TRACE_DIR`` is set, per-trace records are persisted
    one line per (tuple, run) for downstream OBSERVE training.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass


def _trace_writer_from_env() -> Optional[_TraceWriter]:
    trace_dir = os.environ.get("PATHOME_TRACE_DIR")
    if not trace_dir:
        return None
    fname = os.environ.get("PATHOME_TRACE_FILE", "phase0r_traces.jsonl")
    return _TraceWriter(Path(trace_dir) / fname)


def _serialize_trace(
    *,
    tuple_meta: Dict[str, Any],
    trace: Dict[str, Any],
    existing_deltas: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Flatten one trace dict into a JSONL-serializable record for OBSERVE."""
    return {
        "ts":               time.time(),
        "profile_id":       tuple_meta["profile_id"],
        "crop":             tuple_meta["crop"],
        "disease":          tuple_meta["disease"],
        "state":            tuple_meta["state"],
        "primary_image_id": tuple_meta["primary_image_id"],
        "image_path":       tuple_meta["image_path"],
        "run_idx":          trace["run_idx"],
        "path":             trace["path"],
        "decisions":        trace["decisions"],
        "confidences":      trace["confidences"],
        "backtrack_count":  trace["backtrack_count"],
        "early_terminated": trace["early_terminated"],
        "context_buffer":   [asdict(c) for c in trace["context_buffer"]],
        "final_deltas":     trace["final_deltas"],
        "existing_kb_at_start": list(existing_deltas),
    }


# ---------------------------------------------------------------------------
# Top-level: one (crop, disease, state) tuple → final deltas
# ---------------------------------------------------------------------------

def run_for_state(
    *,
    crop: str,
    disease: str,
    state: str,
    canonical_record: Dict[str, Any],
    image_path: Path,
    primary_image_id: str,
    existing_deltas: Optional[List[Dict[str, Any]]] = None,
    profile_id: Optional[str] = None,
    client: Optional[VLLMClient] = None,
    n_runs: Optional[int] = None,
    agreement_min: Optional[int] = None,
    temperature: Optional[float] = None,
    Tmax: Optional[int] = None,
    max_backtracks: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    parallel_runs: bool = True,
    seed_base: int = 42,
    trace_writer: Optional[_TraceWriter] = None,
) -> Dict[str, Any]:
    """N stochastic routed traces → cross-run agreement → conservative
    merge with existing → final deltas for this (crop, disease, state).

    ``existing_deltas`` is the list of regional deltas already in the KB
    for this state (from prior runs). Empty list = cold start.
    All ``None`` knobs fall back to the corresponding ``VLLM_*`` env var
    or the documented default.
    """
    if client is None:
        client = build_client_from_env()

    # `is None` checks (not `or`) so a caller passing 0.0 / 0 is honoured.
    if n_runs              is None: n_runs              = _int_env  ("VLLM_N_RUNS",          10)
    if agreement_min       is None: agreement_min       = _int_env  ("VLLM_AGREEMENT_MIN",    3)
    if temperature         is None: temperature         = _float_env("VLLM_TEMPERATURE",      0.8)
    if Tmax                is None: Tmax                = _int_env  ("VLLM_TMAX",            15)
    if max_backtracks      is None: max_backtracks      = _int_env  ("VLLM_MAX_BACKTRACKS",   1)
    if similarity_threshold is None: similarity_threshold = _float_env("VLLM_SIM_THRESHOLD",  0.4)

    n_runs              = max(1, int(n_runs))
    agreement_min       = max(1, min(int(agreement_min), n_runs))
    Tmax                = max(1, int(Tmax))
    max_backtracks      = max(0, int(max_backtracks))
    similarity_threshold = max(0.0, min(1.0, float(similarity_threshold)))
    existing = list(existing_deltas or [])

    canonical = flatten_canonical(canonical_record)
    image_data_url = _load_image_data_url(image_path)

    def _one(i: int) -> Dict[str, Any]:
        return _run_single_trace(
            crop=crop,
            disease=disease,
            state=state,
            canonical=canonical,
            image_data_url=image_data_url,
            existing_deltas=existing,
            client=client,
            run_idx=i,
            seed=seed_base + i * 100,
            temperature=temperature,
            Tmax=Tmax,
            max_backtracks=max_backtracks,
        )

    traces: List[Dict[str, Any]] = []
    if parallel_runs and n_runs > 1:
        with ThreadPoolExecutor(max_workers=min(n_runs, 8)) as pool:
            for t in pool.map(_one, range(n_runs)):
                traces.append(t)
    else:
        for i in range(n_runs):
            traces.append(_one(i))

    # Persist per-trace records (OBSERVE training data) if a writer is wired.
    if trace_writer is not None:
        tuple_meta = {
            "profile_id":       profile_id or f"{crop}::{disease}",
            "crop":             crop,
            "disease":          disease,
            "state":            state,
            "primary_image_id": primary_image_id,
            "image_path":       str(image_path),
        }
        for t in traces:
            try:
                trace_writer.write(_serialize_trace(
                    tuple_meta=tuple_meta, trace=t, existing_deltas=existing,
                ))
            except Exception as e:
                print(f"    [trace_writer] error: {type(e).__name__}: {e}")

    # Cross-run agreement → candidates.
    per_run_final = [t["final_deltas"] for t in traces]
    candidates = _agreement_filter(
        per_run_final,
        min_support=agreement_min,
        similarity_threshold=similarity_threshold,
    )

    # Conservative merge with existing KB.
    merged, merge_counts = _merge_with_existing(
        existing=existing,
        new=candidates,
        similarity_threshold=similarity_threshold,
    )

    # Stamp image_id on any newly-added delta that doesn't already carry one.
    for d in merged:
        d.setdefault("image_id", primary_image_id)

    return {
        "state":          state,
        "deltas":         merged,
        "__image_ids__":  [primary_image_id],
        "__swarm_meta__": {
            "n_runs":               n_runs,
            "agreement_min":        agreement_min,
            "temperature":          temperature,
            "Tmax":                 Tmax,
            "max_backtracks":       max_backtracks,
            "similarity_threshold": similarity_threshold,
            "paths":                [t["path"] for t in traces],
            "path_lengths":         [len(t["path"]) for t in traces],
            "backtrack_counts":     [t["backtrack_count"] for t in traces],
            "early_terminated":     [t["early_terminated"] for t in traces],
            "n_raw_per_run":        [len(t["final_deltas"]) for t in traces],
            "n_after_agreement":    len(candidates),
            "merge":                merge_counts,
        },
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class WorkItem(NamedTuple):
    """One (crop, disease, state) work unit for the batch runner."""
    profile_id:       str
    crop:             str
    disease:          str
    state:            str
    image_path:       Path
    image_ids:        List[str]
    canonical_record: Dict[str, Any]
    primary_image_id: str
    existing_deltas:  List[Dict[str, Any]]


def run_batch(
    work_items: Iterable[WorkItem],
    *,
    client: Optional[VLLMClient] = None,
    max_parallel: int = 4,
    trace_writer: Optional[_TraceWriter] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the swarm across many (profile, state) tuples.

    Returns ``{profile_id: {state: record}}`` where ``record`` already
    contains the merged-with-existing delta list for that state.

    When ``PATHOME_TRACE_DIR`` is set in the environment (or
    ``trace_writer`` is passed), per-trace records are appended to
    ``<dir>/phase0r_traces.jsonl`` for downstream OBSERVE training.
    """
    if client is None:
        client = build_client_from_env()
    if trace_writer is None:
        trace_writer = _trace_writer_from_env()
        if trace_writer is not None:
            print(f"    [trace_writer] writing to {trace_writer.path}")

    items = list(work_items)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _worker(it: WorkItem) -> Tuple[str, str, Dict[str, Any]]:
        record = run_for_state(
            crop=it.crop,
            disease=it.disease,
            state=it.state,
            canonical_record=it.canonical_record,
            image_path=it.image_path,
            primary_image_id=it.primary_image_id,
            existing_deltas=it.existing_deltas,
            profile_id=it.profile_id,
            client=client,
            trace_writer=trace_writer,
        )
        record["__image_ids__"] = list(it.image_ids) or [it.primary_image_id]
        return it.profile_id, it.state, record

    completed = 0
    total = len(items)
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        futures = {pool.submit(_worker, it): it for it in items}
        for fut in as_completed(futures):
            try:
                profile_id, state, record = fut.result()
            except Exception as e:
                it = futures[fut]
                print(f"    ERROR on {it.profile_id} / {it.state}: "
                      f"{type(e).__name__}: {e}")
                continue
            completed += 1
            meta = record.get("__swarm_meta__", {})
            mg = (meta.get("merge") or {})
            n_deltas = len(record.get("deltas") or [])
            tag = "✓" if n_deltas else "·"
            print(
                f"    [{completed}/{total}] {tag} {profile_id} / {state}  "
                f"deltas={n_deltas} (N={meta.get('n_runs')}, "
                f"K≥{meta.get('agreement_min')}, "
                f"existing={mg.get('n_existing', 0)}, "
                f"added={mg.get('n_added', 0)}, "
                f"bumped={mg.get('n_overlaps_bumped', 0)})"
            )
            results.setdefault(profile_id, {})[state] = record

    return results
