"""
plantswarm/delta_pipeline.py
============================
Qwen swarm + Claude web-search verifier for regional delta extraction.

Per (crop, disease, state, cached Bugwood image):

    load existing regional deltas for THIS state (from final_registry.json)
       v
    N stochastic passes — each one is:
       1) the 4 specialist agents run in PARALLEL on
          (image, canonical, existing KB) producing per-specialist
          AgentDeltaOutputs
       2) DiagnosisAgent consolidates the union of specialist deltas
          -> per-pass final deltas + consolidator kappa
       v
    K-of-N cross-pass agreement filter (Jaccard clusters on image_shows)
       v
    Claude headless WebSearch verifier - retrieval-grounded validation
       v
    Conservative merge with existing KB - existing always preserved,
    overlap bumps swarm_support, stronger evidence upgrades
    verification_status

Algorithm 1 routing was removed: there is no kappa-gated handoff, no
backtrack, no DiagnosisAgent-as-terminal-state. Validation is done by
the Claude verifier, not by self-consistent routing.

Per-pass record (one line per (tuple, pass) in trace JSONL):
    {
      "profile_id", "crop", "disease", "state", "primary_image_id",
      "image_path", "pass_idx", "ts",
      "specialist_outputs": [
        {agent_name, deltas, confidence, reasoning, raw_text}, ...   (4 entries)
      ],
      "consolidator_output": {
        agent_name, deltas, confidence, reasoning, raw_text
      },
      "final_deltas": [...],                       # == consolidator deltas
      "existing_kb_at_start": [...]
    }

Configuration via env vars:
    VLLM_BASE_URL          default http://localhost:8000/v1
    VLLM_MODEL             default Qwen/Qwen2.5-VL-7B-Instruct
    VLLM_TIMEOUT           seconds per HTTP call (default 180)
    VLLM_TEMPERATURE       sampling temperature (default 0.8)
    VLLM_N_RUNS            stochastic passes per tuple (default 10)
    VLLM_AGREEMENT_MIN     K-of-N agreement floor (default 3)
    VLLM_SIM_THRESHOLD     Jaccard tau for clustering AND merge dedup (0.4)
    PATHOME_USE_VERIFIER   1 enable Claude+WebSearch verifier (default 1)
    PATHOME_VERIFIER_TIMEOUT 600
    PATHOME_VERIFIER_MAX_TURNS 30
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

from agents import SPECIALIST_AGENTS, VISUAL_GROUP_AGENTS
from agents.organ_router import OrganDetectionAgent, route_for_organ
from agents.base_agent import AgentDeltaOutput, BaseAgent
from agents.diagnosis_agent import DiagnosisAgent
from utils.vllm_client import VLLMClient
from utils.vllm_inproc import InProcessVLLMClient, get_inproc_client


# Roster selection. Default 'grouped' = 5 generalized VISUAL-SYMPTOM
# group agents (LeafSymptomAgent, StemRootSymptomAgent,
# FruitFlowerSignAgent, WholePlantSymptomAgent, DiagnosticVisualAgent),
# each comparing the photo to the canonical visual_symptoms KB. They
# run through the SAME machinery as the legacy specialists: parallel
# fan-out → shared blackboard → round 2 → DiagnosisAgent consolidator.
#
# 'specialists' restores the legacy 24 single-feature specialists.
SPECIALIST_CLASSES: Tuple[type, ...] = SPECIALIST_AGENTS


def _swarm_granularity() -> str:
    """Roster mode:
      'routed'      (default) — OrganDetectionAgent picks the organ,
                    then only that organ's deep single-feature
                    specialists + always-on cross-cutters fire
                    (DR.Arti decision tree).
      'grouped'     — 5 visual-symptom group agents (all run).
      'specialists' — legacy 24 single-feature specialists (all run).
    """
    v = os.environ.get("SWARM_GRANULARITY", "routed").strip().lower()
    if v in ("specialists", "specialist", "24"):
        return "specialists"
    if v in ("grouped", "group", "5"):
        return "grouped"
    return "routed"


def _active_roster() -> Tuple[type, ...]:
    """Non-routed rosters (organ routing is resolved per-image inside
    ``_run_single_pass``, so it is not expressible here)."""
    if _swarm_granularity() == "specialists":
        return SPECIALIST_AGENTS
    return VISUAL_GROUP_AGENTS


# ---------------------------------------------------------------------------
# Swarm config / env helpers
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


def build_client_from_env():
    """Build a vLLM client (in-process by default; HTTP only as opt-out).

    Phase 0R runs on a single GPU node — an HTTP boundary inside the
    same job adds zero value and historically masked silent 400s that
    wiped every (crop, disease, state) tuple. The in-process path loads
    ``vllm.LLM`` directly via :mod:`utils.vllm_inproc`.

    Set ``VLLM_INPROCESS=0`` to fall back to the legacy ``VLLMClient``
    talking to an externally-launched ``vllm serve`` (used only for
    debugging the HTTP path).
    """
    if os.environ.get("VLLM_INPROCESS", "1") != "0":
        return get_inproc_client()

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model    = os.environ.get("VLLM_MODEL",    "Qwen/Qwen2.5-VL-7B-Instruct")
    timeout  = _int_env("VLLM_TIMEOUT", 180)
    temperature = _float_env("VLLM_TEMPERATURE", 0.8)
    client = VLLMClient(
        base_url=base_url, model=model,
        temperature=temperature, timeout=timeout,
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
    record: Dict[str, Any], state: str,
) -> List[Dict[str, Any]]:
    """Pull existing regional deltas for THIS state from a SAGE
    final_registry.json disease record."""
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
        for k_src in ("__support__", "support", "swarm_support",
                      "__cluster_size__", "cluster_size"):
            if k_src in d:
                try:
                    entry[k_src.replace("cluster_size", "__cluster_size__")
                              .replace("support", "__support__")
                              .replace("swarm___support__", "__support__")
                              .replace("____", "__")] = int(d[k_src])
                except (TypeError, ValueError):
                    pass
        if "verification_status" in d:
            entry["verification_status"] = str(d["verification_status"])
        if isinstance(d.get("web_support"), list):
            entry["web_support"] = d["web_support"]
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Image loading (MIME-aware)
# ---------------------------------------------------------------------------

def _load_image_data_url(path: Path) -> str:
    p = Path(path)
    mt, _ = mimetypes.guess_type(str(p))
    if not mt or not mt.startswith("image/"):
        mt = "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mt};base64,{b64}"


# ---------------------------------------------------------------------------
# One stochastic pass — parallel specialists + consolidator
# ---------------------------------------------------------------------------

def _run_single_pass(
    *,
    crop: str,
    disease: str,
    state: str,
    canonical: Dict[str, Any],
    image_data_url: str,
    existing_deltas: List[Dict[str, Any]],
    client: VLLMClient,
    pass_idx: int,
    seed: int,
    temperature: float,
    parallel_specialists: bool = True,
    swarm_rounds: int = 2,
) -> Dict[str, Any]:
    """One pass through the REAL swarm:

      Round 1  : 24 specialists run in parallel, each on (image,
                 canonical, existing KB). No inter-agent visibility.
      Blackboard: built from round-1 outputs.
      Round 2  : 24 specialists run AGAIN in parallel — each now sees
                 the full blackboard (every peer's round-1 output)
                 and may REFINE its delta, emit a NEW delta prompted
                 by peer observations, or declare SUPPORT / CHALLENGE
                 / WITHDRAW cross_refs against peers.
      Consolidator: VisualDiagnosisAgent sees BOTH rounds and walks
                 the look-alike decision-graph CoT.

    ``swarm_rounds`` controls how many rounds run. Setting it to 1
    falls back to the legacy single-round parallel-ensemble. Default 2
    enables the stigmergy round; can be overridden via
    ``VLLM_SWARM_ROUNDS`` env var in ``run_for_state``.
    """
    # DR.Arti decision tree: detect the organ first, then activate
    # ONLY that organ's deep specialists. One image = one organ, so
    # this avoids running (and paying for) agents that can only ever
    # say "not visible".
    detected_organ: Optional[Dict[str, str]] = None
    if _swarm_granularity() == "routed":
        detected_organ = OrganDetectionAgent(client).detect(
            image_data_url, seed=seed, temperature=temperature,
        )
        roster = route_for_organ(detected_organ["organ"])
    else:
        roster = _active_roster()

    specialists: List[BaseAgent] = [cls(client) for cls in roster]
    n_specialists = len(specialists)

    def _run_round(
        round_idx: int,
        blackboard: Optional[Dict[str, AgentDeltaOutput]],
    ) -> List[AgentDeltaOutput]:
        def _run_one(idx_agent: Tuple[int, BaseAgent]) -> AgentDeltaOutput:
            i, ag = idx_agent
            try:
                return ag.extract_deltas(
                    crop=crop, disease=disease, state=state,
                    canonical=canonical, image_data_url=image_data_url,
                    existing_kb_deltas=existing_deltas,
                    # Round 2 uses a different seed offset so the
                    # model genuinely re-thinks, doesn't just emit
                    # the cached round-1 answer.
                    seed=seed + i + (10_000 if round_idx == 2 else 0),
                    temperature=temperature,
                    blackboard=blackboard,
                    round_idx=round_idx,
                )
            except Exception as e:
                print(f"    [round {round_idx}] [{ag.AGENT_NAME}] "
                      f"error: {type(e).__name__}: {e}")
                return AgentDeltaOutput(
                    agent_name=ag.AGENT_NAME, round_idx=round_idx,
                )

        pairs = list(enumerate(specialists))
        if parallel_specialists and n_specialists > 1:
            with ThreadPoolExecutor(max_workers=n_specialists) as pool:
                return list(pool.map(_run_one, pairs))
        return [_run_one(p) for p in pairs]

    # --- Round 1: independent observation ----------------------------------
    round1_outputs = _run_round(1, blackboard=None)

    # --- Build the shared blackboard ---------------------------------------
    blackboard: Dict[str, AgentDeltaOutput] = {
        o.agent_name: o for o in round1_outputs if o.agent_name
    }

    # --- Round 2 (default): stigmergy refinement ---------------------------
    round2_outputs: List[AgentDeltaOutput] = []
    if swarm_rounds >= 2:
        round2_outputs = _run_round(2, blackboard=blackboard)

    # --- Pick which set the consolidator sees ------------------------------
    # The consolidator gets BOTH rounds — it can read each agent's
    # round-1 baseline + round-2 refinement + cross_refs and pick the
    # better-grounded one per field. For backwards-compat callers that
    # expect a flat list, we also expose `specialist_outputs` as
    # round-2 (if it ran) else round-1.
    all_specialist_outputs: List[AgentDeltaOutput] = (
        round1_outputs + round2_outputs
    )
    primary_outputs = round2_outputs if round2_outputs else round1_outputs

    consolidator = DiagnosisAgent(client)
    try:
        consolidator_output = consolidator.consolidate(
            crop=crop, disease=disease, state=state,
            canonical=canonical, image_data_url=image_data_url,
            specialist_outputs=all_specialist_outputs,
            existing_kb_deltas=existing_deltas,
            seed=seed + 100_000, temperature=temperature,
        )
    except Exception as e:
        print(f"    [DiagnosisAgent] consolidation failed "
              f"({type(e).__name__}: {e}); using primary-round union")
        union: List[Dict[str, str]] = []
        for s in primary_outputs:
            union.extend(s.deltas)
        consolidator_output = AgentDeltaOutput(
            agent_name="DiagnosisAgent",
            deltas=union, confidence="low",
            reasoning="consolidator failed; using primary-round union",
        )

    return {
        "pass_idx":            pass_idx,
        "swarm_rounds":        swarm_rounds,
        "round1_outputs":      round1_outputs,
        "round2_outputs":      round2_outputs,
        # Back-compat: tests + traces expect a flat list of all
        # specialist outputs from this pass. Includes both rounds.
        "specialist_outputs":  all_specialist_outputs,
        "consolidator_output": consolidator_output,
        "final_deltas":        consolidator_output.deltas,
        "detected_organ":      detected_organ,
        "active_agents":       [a.AGENT_NAME for a in specialists],
    }


# ---------------------------------------------------------------------------
# Similarity helpers (agreement filter + merge dedup)
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
    per_pass_deltas: List[List[Dict[str, str]]],
    *,
    min_support: int,
    similarity_threshold: float = 0.4,
) -> List[Dict[str, str]]:
    """K-of-N agreement clustering across stochastic passes."""
    all_with_pass: List[Tuple[int, Dict[str, str]]] = []
    for pass_idx, deltas in enumerate(per_pass_deltas):
        for d in deltas or []:
            all_with_pass.append((pass_idx, d))

    by_field: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for pass_idx, d in all_with_pass:
        by_field[d.get("field", "other")].append((pass_idx, d))

    survivors: List[Dict[str, str]] = []
    for fld, items in by_field.items():
        clusters = _cluster_by_similarity(items, threshold=similarity_threshold)
        for cluster in clusters:
            pass_set = {pi for pi, _ in cluster}
            if len(pass_set) >= min_support:
                rep = max((d for _, d in cluster),
                          key=lambda d: len(d.get("image_shows", "")))
                rep = dict(rep)
                rep["__support__"]      = len(pass_set)
                rep["__cluster_size__"] = len(cluster)
                survivors.append(rep)
    return survivors


# ---------------------------------------------------------------------------
# Conservative merge with existing KB
# ---------------------------------------------------------------------------

def _support_of(d: Dict[str, Any]) -> int:
    for k in ("swarm_support", "__support__", "support"):
        if k in d:
            try:
                return int(d[k] or 0)
            except (TypeError, ValueError):
                pass
    return 1


def _set_support(d: Dict[str, Any], val: int) -> None:
    d["swarm_support"] = int(val)
    d["__support__"]   = int(val)


def _merge_with_existing(
    *,
    existing: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
    similarity_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Conservative merge — existing preserved, overlap bumps support,
    stronger verification status upgrades existing."""
    merged: List[Dict[str, Any]] = [dict(e) for e in existing]
    for e in merged:
        _set_support(e, _support_of(e))

    by_field: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(merged):
        by_field[e.get("field", "other")].append(i)

    counts = {
        "n_existing":        len(existing),
        "n_new_candidates":  len(new),
        "n_added":           0,
        "n_overlaps_bumped": 0,
        "n_upgraded":        0,
    }

    _STATUS_RANK = {
        "verified": 5, "weakly_supported": 4, "provisional": 3,
        "novel_plausible": 2, "unverified": 1, "contradictory": 0,
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
            _set_support(e, _support_of(e) + _support_of(n))
            counts["n_overlaps_bumped"] += 1
            n_status = n.get("verification_status", "unverified")
            e_status = e.get("verification_status", "unverified")
            if _STATUS_RANK.get(n_status, 0) > _STATUS_RANK.get(e_status, 0):
                e["verification_status"] = n_status
                counts["n_upgraded"] += 1
            existing_urls = {(s or {}).get("url", "") for s in (e.get("web_support") or [])}
            for s in n.get("web_support") or []:
                if (s or {}).get("url", "") not in existing_urls:
                    e.setdefault("web_support", []).append(s)
                    existing_urls.add(s.get("url", ""))
        else:
            n_copy = dict(n)
            _set_support(n_copy, _support_of(n_copy))
            n_copy.setdefault("verification_status", "unverified")
            n_copy.setdefault("web_support", [])
            merged.append(n_copy)
            by_field[n_field].append(len(merged) - 1)
            counts["n_added"] += 1
    return merged, counts


# ---------------------------------------------------------------------------
# Trace persistence (training data for OBSERVE)
# ---------------------------------------------------------------------------

class _TraceWriter:
    """Append-mode JSONL writer with fsync."""

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


def _serialize_pass(
    *,
    tuple_meta: Dict[str, Any],
    pass_record: Dict[str, Any],
    existing_deltas: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Flatten one pass record into a JSONL-serializable line for OBSERVE."""
    return {
        "ts":               time.time(),
        "profile_id":       tuple_meta["profile_id"],
        "crop":             tuple_meta["crop"],
        "disease":          tuple_meta["disease"],
        "state":            tuple_meta["state"],
        "primary_image_id": tuple_meta["primary_image_id"],
        "image_path":       tuple_meta["image_path"],
        "pass_idx":         pass_record["pass_idx"],
        "specialist_outputs": [asdict(s) for s in pass_record["specialist_outputs"]],
        "consolidator_output": asdict(pass_record["consolidator_output"]),
        "final_deltas":        pass_record["final_deltas"],
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
    similarity_threshold: Optional[float] = None,
    parallel_passes: bool = True,
    parallel_specialists: bool = True,
    seed_base: int = 42,
    trace_writer: Optional[_TraceWriter] = None,
) -> Dict[str, Any]:
    """N stochastic passes -> K-of-N agreement -> verifier -> merge."""
    if client is None:
        client = build_client_from_env()

    # vLLM must be constructed on the MAIN thread, ONCE, before any
    # ThreadPoolExecutor (passes / specialists) spawns — building it
    # lazily inside a worker thread caused the EngineCore CUDA-init
    # failure + retry-storm. Warm it here, on the calling thread.
    _warmup = getattr(client, "warmup", None)
    if callable(_warmup):
        _warmup()

    if n_runs              is None: n_runs              = _int_env  ("VLLM_N_RUNS",          10)
    if agreement_min       is None: agreement_min       = _int_env  ("VLLM_AGREEMENT_MIN",    3)
    if temperature         is None: temperature         = _float_env("VLLM_TEMPERATURE",      0.8)
    if similarity_threshold is None: similarity_threshold = _float_env("VLLM_SIM_THRESHOLD",  0.4)
    # 2-round real-swarm protocol by default. Set VLLM_SWARM_ROUNDS=1
    # to fall back to the legacy parallel-ensemble single-round mode.
    swarm_rounds = max(1, _int_env("VLLM_SWARM_ROUNDS", 2))

    n_runs              = max(1, int(n_runs))
    agreement_min       = max(1, min(int(agreement_min), n_runs))
    similarity_threshold = max(0.0, min(1.0, float(similarity_threshold)))
    existing = list(existing_deltas or [])

    canonical = flatten_canonical(canonical_record)
    image_data_url = _load_image_data_url(image_path)

    granularity = _swarm_granularity()

    def _one(i: int) -> Dict[str, Any]:
        return _run_single_pass(
            crop=crop, disease=disease, state=state,
            canonical=canonical, image_data_url=image_data_url,
            existing_deltas=existing, client=client,
            pass_idx=i, seed=seed_base + i * 100,
            temperature=temperature,
            parallel_specialists=parallel_specialists,
            swarm_rounds=swarm_rounds,
        )

    passes: List[Dict[str, Any]] = []
    if parallel_passes and n_runs > 1:
        with ThreadPoolExecutor(max_workers=min(n_runs, 8)) as pool:
            for p in pool.map(_one, range(n_runs)):
                passes.append(p)
    else:
        for i in range(n_runs):
            passes.append(_one(i))

    # Persist per-pass records for OBSERVE training, if requested.
    if trace_writer is not None:
        tuple_meta = {
            "profile_id":       profile_id or f"{crop}::{disease}",
            "crop":             crop, "disease": disease, "state": state,
            "primary_image_id": primary_image_id, "image_path": str(image_path),
        }
        for p in passes:
            try:
                trace_writer.write(_serialize_pass(
                    tuple_meta=tuple_meta, pass_record=p,
                    existing_deltas=existing,
                ))
            except Exception as e:
                print(f"    [trace_writer] error: {type(e).__name__}: {e}")

    # Cross-pass agreement.
    per_pass_final = [p["final_deltas"] for p in passes]
    candidates = _agreement_filter(
        per_pass_final, min_support=agreement_min,
        similarity_threshold=similarity_threshold,
    )

    # Claude web-search verifier.
    use_verifier = os.environ.get("PATHOME_USE_VERIFIER", "1") not in ("0", "false", "False")
    verifier_meta: Dict[str, Any] = {"enabled": use_verifier}
    if use_verifier and candidates:
        from pathome_kb.verifier import verify_candidates
        v_timeout = _int_env("PATHOME_VERIFIER_TIMEOUT", 600)
        v_turns   = _int_env("PATHOME_VERIFIER_MAX_TURNS", 30)
        try:
            verdict = verify_candidates(
                crop=crop, disease=disease, state=state,
                canonical=canonical, existing_kb_deltas=existing,
                candidates=candidates, primary_image_id=primary_image_id,
                timeout_secs=v_timeout, max_turns=v_turns,
            )
        except Exception as e:
            print(f"    [verifier] failure ({type(e).__name__}: {e}); "
                  f"passing candidates through as 'unverified'")
            verdict = {
                "verified":               [],
                "provisional":            [dict(c, verification_status="unverified",
                                                  web_support=[]) for c in candidates],
                "contradictory":          [],
                "duplicates_of_existing": [],
                "accepted":               [dict(c, verification_status="unverified",
                                                  web_support=[]) for c in candidates],
            }
        new_for_merge = verdict.get("accepted", [])
        verifier_meta.update({
            "n_verified":           len(verdict.get("verified", [])),
            "n_provisional":        len(verdict.get("provisional", [])),
            "n_contradictory":      len(verdict.get("contradictory", [])),
            "n_duplicates_existing": len(verdict.get("duplicates_of_existing", [])),
        })
    else:
        new_for_merge = []
        for c in candidates:
            cc = dict(c)
            cc.setdefault("swarm_support", cc.get("__support__", 1))
            cc.setdefault("verification_status", "unverified")
            cc.setdefault("web_support", [])
            new_for_merge.append(cc)

    # Conservative merge.
    merged, merge_counts = _merge_with_existing(
        existing=existing, new=new_for_merge,
        similarity_threshold=similarity_threshold,
    )
    for d in merged:
        d.setdefault("image_id", primary_image_id)

    return {
        "state":          state,
        "deltas":         merged,
        "__image_ids__":  [primary_image_id],
        "__swarm_meta__": {
            "granularity":          granularity,
            "n_runs":               n_runs,
            "agreement_min":        agreement_min,
            "temperature":          temperature,
            "similarity_threshold": similarity_threshold,
            "kappa_per_pass":       [p["consolidator_output"].confidence for p in passes],
            "n_raw_per_pass":       [len(p["final_deltas"]) for p in passes],
            "detected_organ_per_pass": [
                (p.get("detected_organ") or {}).get("organ") for p in passes
            ],
            "n_active_agents_per_pass": [
                len(p.get("active_agents") or []) for p in passes
            ],
            "n_after_agreement":    len(candidates),
            "verifier":             verifier_meta,
            "merge":                merge_counts,
        },
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class WorkItem(NamedTuple):
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
    if client is None:
        client = build_client_from_env()
    if trace_writer is None:
        trace_writer = _trace_writer_from_env()
        if trace_writer is not None:
            print(f"    [trace_writer] writing to {trace_writer.path}")

    # Build the vLLM engine HERE — on the main thread, once, before the
    # batch ThreadPoolExecutor below submits run_for_state into worker
    # threads. vLLM's engine core must not be constructed off the main
    # thread (that triggered the CUDA-init failure + retry-storm).
    _warmup = getattr(client, "warmup", None)
    if callable(_warmup):
        print("    [vllm_inproc] warming engine on main thread ...")
        _warmup()
        print("    [vllm_inproc] engine ready")

    items = list(work_items)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _worker(it: WorkItem) -> Tuple[str, str, Dict[str, Any]]:
        record = run_for_state(
            crop=it.crop, disease=it.disease, state=it.state,
            canonical_record=it.canonical_record,
            image_path=it.image_path,
            primary_image_id=it.primary_image_id,
            existing_deltas=it.existing_deltas,
            profile_id=it.profile_id,
            client=client, trace_writer=trace_writer,
        )
        record["__image_ids__"] = list(it.image_ids) or [it.primary_image_id]
        return it.profile_id, it.state, record

    total = len(items)
    started = 0
    completed = 0
    total_added = 0
    start_lock = threading.Lock()

    def _p(msg: str) -> None:
        # flush so progress shows in real time through the tee'd log
        print(msg, flush=True)

    def _worker_logged(it: WorkItem):
        nonlocal started
        with start_lock:
            started += 1
            s = started
        _p(f"    [start {s}/{total} | left {total - s}] "
           f"{it.profile_id} / {it.state}  "
           f"(KB context: existing_deltas={len(it.existing_deltas or [])}, "
           f"canonical_visual_symptoms=loaded)  image={it.primary_image_id}")
        return _worker(it)

    _p(f"  processing {total} (crop,disease,state) image tuples "
       f"(max_parallel={max_parallel}); each runs the real swarm vs the "
       f"generated KB and emits ADDITIONAL deltas only")
    t_batch = time.time()
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        futures = {pool.submit(_worker_logged, it): it for it in items}
        for fut in as_completed(futures):
            try:
                profile_id, state, record = fut.result()
            except Exception as e:
                it = futures[fut]
                completed += 1
                _p(f"    [done {completed}/{total} | left {total - completed}]"
                   f" ERROR {it.profile_id} / {it.state}: "
                   f"{type(e).__name__}: {e}")
                continue
            completed += 1
            meta = record.get("__swarm_meta__", {})
            mg = (meta.get("merge") or {})
            vf = (meta.get("verifier") or {})
            n_deltas = len(record.get("deltas") or [])
            n_added = mg.get("n_added", 0)
            total_added += int(n_added or 0)
            tag = "OK " if n_deltas else "---"
            v_summary = (
                f"vfy={vf.get('n_verified', 0)}/{vf.get('n_provisional', 0)}/"
                f"{vf.get('n_contradictory', 0)}, "
                if vf.get("enabled") else ""
            )
            _p(
                f"    [done {completed}/{total} | left {total - completed}] "
                f"{tag} {profile_id} / {state}  "
                f"deltas={n_deltas} (N={meta.get('n_runs')}, "
                f"K>={meta.get('agreement_min')}, "
                f"organ={meta.get('detected_organ_per_pass')}, {v_summary}"
                f"existing={mg.get('n_existing', 0)}, "
                f"ADDED={n_added}, "
                f"bumped={mg.get('n_overlaps_bumped', 0)})  "
                f"[run total added so far: {total_added}]"
            )
            results.setdefault(profile_id, {})[state] = record

    _p(f"  BATCH DONE: {completed}/{total} tuples in "
       f"{time.time() - t_batch:.0f}s; NEW delta-KB found this run = "
       f"{total_added} (additional observations beyond the generated KB)")
    return results
