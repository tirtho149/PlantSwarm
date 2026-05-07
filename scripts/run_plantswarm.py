"""
scripts/run_plantswarm.py
=========================
Main entry: PlantSwarm pipeline (paper Algorithm) on PlantDiagBench.

Usage:
    python scripts/run_plantswarm.py --config configs/default.yaml [--subset N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
from requests.exceptions import RequestException
import yaml
from tqdm import tqdm

from calibration.temperature_scaling import TemperatureScaler
from data.loader import PlantDiagBenchLoader
from utils.env import load_project_dotenv
from plantswarm.autogen_pipeline import AutoGenPlantSwarmPipeline, run_local_qwen_text_swarm_demo
from plantswarm.entropy_pipeline import EntropyPlantSwarmPipeline
from plantswarm.hf_pipeline import HFDirectPipeline
from utils.metrics import macro_f1, tpcp
from utils.routing_trace import append_trace, existing_trace_ids, save_traces
from utils.vllm_client import (
    VLLMClient,
    configure_vllm_client_from_yaml,
    validate_model_server_matches_config,
)
from utils.hf_client import HFClient


GT_ATTR = {
    "T1": "symptom_type",
    "T2": "pathogen_class",
    "T3": "disease_name",
    "T4": "severity_class",
    "T5": "crop_species",
}


def _canonical_benchmark(meta: dict, col: str) -> str | None:
    """Map a parquet/meta benchmark label to plantswarm_metrics.by_benchmark keys."""
    raw = meta.get(col)
    if raw is None:
        return None
    s = str(raw).strip().lower().replace(" ", "_")
    if "village" in s or s in ("pv", "plantvillage"):
        return "plantvillage"
    if "wild" in s or s in ("plantwild", "in_the_wild"):
        return "plantwild"
    if "doc" in s or s in ("pd", "plantdoc"):
        return "plantdoc"
    if "leafbench" in s or (s.startswith("leaf") and "bench" in s) or s == "lb":
        return "leafbench"
    return None


def _macro_f1_subset(
    records,
    traces,
    task_id: str,
    label_space: dict,
    indices: list,
    bootstrap_n: int,
):
    """Macro-F1 for a subset of (record, trace) pairs (by index into records/traces)."""
    gt_attr = GT_ATTR[task_id]
    labels = label_space[task_id]
    preds, gts = [], []
    for i in indices:
        record = records[i]
        trace = traces[i]
        gt = getattr(record, gt_attr, None)
        if gt is None:
            continue
        pred = trace.final_predictions.get(task_id, labels[0])
        preds.append(pred)
        gts.append(gt)
    if not preds:
        return None
    f1, _ = macro_f1(preds, gts, labels, bootstrap_n=bootstrap_n)
    return float(f1)


def _metrics_by_benchmark(cfg: dict, records, all_traces, label_space: dict) -> dict:
    """Per-benchmark T2/T3 macro-F1 blocks for LaTeX wide table (optional)."""
    col = (cfg.get("data") or {}).get("benchmark_col")
    if not col or not records:
        return {}
    buckets: dict[str, list] = {
        "plantvillage": [],
        "plantdoc": [],
        "plantwild": [],
        "leafbench": [],
    }
    for i, record in enumerate(records):
        b = _canonical_benchmark(record.meta, col)
        if b:
            buckets[b].append(i)
    out: dict = {}
    bn = cfg["eval"]["bootstrap_n"]
    for name, indices in buckets.items():
        if not indices:
            continue
        block = {}
        for tid in ("T2", "T3"):
            f1 = _macro_f1_subset(records, all_traces, tid, label_space, indices, bn)
            if f1 is not None:
                block[tid] = {"macro_f1": f1, "n": len(indices)}
        if block:
            out[name] = block
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run PlantSwarm pipeline on PlantDiagBench")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--orchestrator",
        choices=["autogen_swarm", "entropy_routing", "classic", "hf_direct"],
        default=None,
        help=(
            "autogen_swarm (default, needs vLLM server + autogen), "
            "hf_direct (single-GPU, no server required), "
            "entropy_routing (vLLM logprob entropy routing), "
            "or classic (rejected)."
        ),
    )
    parser.add_argument(
        "--local-qwen-text-demo",
        action="store_true",
        help=(
            "Run the notebook-style text-only Swarm (triage/specialist/reviewer) with local "
            "Hugging Face Qwen — see plantswarm/autogen_pipeline.py (no PlantDiagBench, no vLLM)."
        ),
    )
    parser.add_argument(
        "--local-qwen-query",
        type=str,
        default="A tomato leaf shows yellow spots and curling. What disease is this?",
        help="User task string for --local-qwen-text-demo.",
    )
    parser.add_argument(
        "--local-qwen-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model id for --local-qwen-text-demo (text-only causal LM).",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _check_openai_compatible_api(base_url: str, timeout: float = 5.0) -> None:
    """
    PlantSwarm talks to whatever is at model.vllm_base_url via AutoGen's OpenAI client
    (typically vLLM, but any OpenAI-compatible server works). Dataset loading is separate
    (TFDS / files) and does not need this server.
    """
    bu = base_url.rstrip("/")
    models_url = f"{bu}/models" if bu.endswith("/v1") else f"{bu}/v1/models"
    try:
        r = requests.get(models_url, timeout=timeout)
        if r.status_code >= 400:
            print(
                f"WARNING: {models_url} returned HTTP {r.status_code}. "
                "Inference may fail or give meaningless metrics."
            )
        else:
            print(f"  OpenAI-compatible API reachable at {base_url} (GET /v1/models OK).")
    except RequestException as e:
        print(
            f"WARNING: Cannot reach OpenAI-compatible API at {base_url} ({e!r}). "
            "The swarm needs a running server (e.g. vLLM) at that URL; "
            "only dataset loading will work without it."
        )


def main():
    load_project_dotenv()
    args = parse_args()
    if args.local_qwen_text_demo:
        import asyncio

        asyncio.run(
            run_local_qwen_text_swarm_demo(
                args.local_qwen_query,
                model_name=args.local_qwen_model,
            )
        )
        return

    cfg = load_config(args.config)

    orchestrator = args.orchestrator or cfg.get("routing", {}).get("orchestrator", "autogen_swarm")
    if orchestrator == "classic":
        raise ValueError(
            "Orchestrator 'classic' is no longer supported. Use AutoGen AgentChat Swarm: "
            "routing.orchestrator: autogen_swarm (see plantswarm/autogen_pipeline.py)."
        )

    results_dir = args.output_dir or cfg["output"]["results_dir"]
    traces_dir = cfg["output"]["traces_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    print("Loading dataset (directory tree, Hugging Face, TFDS, PlantDoc, or parquet)...")
    loader_test = PlantDiagBenchLoader(cfg["data"], split="test")
    loader_cal = PlantDiagBenchLoader(
        cfg["data"],
        split="calibration",
        top_k_diseases=loader_test.top_k_diseases,
    )
    label_space = loader_test.label_space
    print(f"  Test: {len(loader_test)} images | Cal: {len(loader_cal)} images")

    r_cfg = cfg.get("routing", {})

    if orchestrator == "hf_direct":
        # Single-GPU in-process mode — no vLLM server, no AutoGen required
        # Pick a Qwen2.5-VL model: prefer one with "Qwen" and "VL" in the name
        replica_backbones = cfg["model"].get("replica_backbones", [])
        hf_model = next(
            (m for m in replica_backbones if "Qwen" in m and "VL" in m),
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )
        print(f"  HF direct mode: loading {hf_model} in-process (no server needed)...")
        client = HFClient(
            model=hf_model,
            temperature=cfg["model"]["temperature"],
            seed=cfg["model"]["seed"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
        )
        pipeline = HFDirectPipeline(
            client=client,
            label_space=label_space,
            Tmax=r_cfg["Tmax"],
            confidence_weights=r_cfg["confidence_weights"],
        )
    else:
        client = VLLMClient(
            base_url=cfg["model"]["vllm_base_url"],
            model=cfg["model"]["backbone"],
            temperature=cfg["model"]["temperature"],
            seed=cfg["model"]["seed"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
        )
        configure_vllm_client_from_yaml(client, cfg.get("model"), orchestrator=orchestrator)

        _check_openai_compatible_api(cfg["model"]["vllm_base_url"])
        validate_model_server_matches_config(cfg)

        if orchestrator == "entropy_routing":
            pipeline = EntropyPlantSwarmPipeline(
                client=client,
                label_space=label_space,
                Tmax=r_cfg["Tmax"],
                confidence_weights=r_cfg["confidence_weights"],
                delta1=float(r_cfg.get("entropy_delta1", 0.05)),
                delta2=float(r_cfg.get("entropy_delta2", 0.35)),
            )
        elif orchestrator == "autogen_swarm":
            pipeline = AutoGenPlantSwarmPipeline(
                client=client,
                label_space=label_space,
                Tmax=r_cfg["Tmax"],
                confidence_weights=r_cfg["confidence_weights"],
            )
        else:
            raise ValueError(f"Unknown orchestrator {orchestrator!r}.")

    all_traces = []
    predictions_output = []
    records = list(loader_test)
    if args.subset:
        records = records[: args.subset]

    # Resume support: skip image_ids already persisted to plantswarm_traces.jsonl.
    # Each successful trace is appended + fsynced inside the loop, so a SLURM
    # walltime kill or crash leaves a usable partial result.
    traces_filename = "plantswarm_traces.jsonl"
    preds_path = os.path.join(results_dir, "plantswarm_predictions.jsonl")
    already_done = existing_trace_ids(traces_dir, traces_filename)
    if already_done:
        print(f"  Resuming: {len(already_done)} traces already on disk — skipping those.")
        records = [r for r in records if r.image_id not in already_done]

    print(f"Running PlantSwarm ({orchestrator}) on {len(records)} images...")
    n_failed = 0
    for record in tqdm(records):
        try:
            trace = pipeline.run(record.image_id, record.image_b64)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            tqdm.write(f"  [skip] {record.image_id}: {type(e).__name__}: {e}")
            continue

        # Persist trace immediately (fsync inside) so SIGKILL doesn't lose it.
        try:
            append_trace(trace, traces_dir, traces_filename)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  [warn] failed to persist trace for {record.image_id}: {e}")

        pred_record = {
            "image_id": record.image_id,
            "predictions": trace.final_predictions,
            "ground_truth": {
                "T1": record.symptom_type,
                "T2": record.pathogen_class,
                "T3": record.disease_name,
                "T4": record.severity_class,
                "T5": record.crop_species,
            },
            "path": trace.path,
            "total_tokens": trace.total_tokens,
            "early_terminated": trace.early_terminated,
        }
        try:
            with open(preds_path, "a") as pf:
                pf.write(json.dumps(pred_record) + "\n")
                pf.flush()
                os.fsync(pf.fileno())
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  [warn] failed to persist prediction for {record.image_id}: {e}")

        all_traces.append(trace)
        predictions_output.append(pred_record)

    if n_failed:
        print(f"  {n_failed} images skipped due to runtime errors (see warnings above).")

    print("\nComputing metrics...")
    metrics = {}

    for task_id in ["T1", "T2", "T3", "T4", "T5"]:
        gt_attr = GT_ATTR[task_id]
        labels = label_space[task_id]

        preds, gts, tokens_list, correct_list = [], [], [], []
        probs_list = []

        for record, trace in zip(records, all_traces):
            gt = getattr(record, gt_attr, None)
            if gt is None:
                continue
            pred = trace.final_predictions.get(task_id, labels[0])
            preds.append(pred)
            gts.append(gt)
            tokens_list.append(trace.total_tokens)
            correct_list.append(int(pred == gt))

            task_probs = trace.ensemble_probs.get(task_id, {})
            probs_list.append([task_probs.get(lbl, 1.0 / len(labels)) for lbl in labels])

        if not preds:
            continue

        f1, (ci_lo, ci_hi) = macro_f1(preds, gts, labels, bootstrap_n=cfg["eval"]["bootstrap_n"])
        tpcp_val = tpcp(tokens_list, correct_list)

        from calibration.ece import compute_ece_from_probs

        probs_matrix = np.array(probs_list)
        ece_val, rel_diag = compute_ece_from_probs(
            probs_matrix, np.array(gts), labels, n_bins=cfg["calibration"]["ece_bins"]
        )

        metrics[task_id] = {
            "macro_f1": f1,
            "macro_f1_ci": [ci_lo, ci_hi],
            "ece": ece_val,
            "tpcp": tpcp_val,
            "n": len(preds),
            "n_correct": sum(correct_list),
        }

        with open(os.path.join(results_dir, f"reliability_diagram_{task_id}.json"), "w") as f:
            json.dump(rel_diag, f, indent=2)

        print(
            f"  {task_id}: F1={f1:.1f} [{ci_lo:.1f},{ci_hi:.1f}] "
            f"ECE={ece_val:.4f} TPCP={tpcp_val:.1f}"
        )

        if cfg["calibration"]["temperature_scaling"]:
            cal_probs_list, cal_gts = [], []
            for cal_record in loader_cal:
                gt_c = getattr(cal_record, gt_attr, None)
                if gt_c:
                    cal_gts.append(gt_c)
                    cal_trace = pipeline.run(cal_record.image_id, cal_record.image_b64)
                    cal_task_probs = cal_trace.ensemble_probs.get(task_id, {})
                    cal_probs_list.append(
                        [cal_task_probs.get(lbl, 1.0 / len(labels)) for lbl in labels]
                    )

            if cal_probs_list:
                scaler = TemperatureScaler()
                cal_matrix = np.array(cal_probs_list)
                scaler.fit(np.log(cal_matrix + 1e-10), cal_gts, labels)
                ts_report = scaler.report_ece(np.log(probs_matrix + 1e-10), gts, labels)
                metrics[task_id]["temperature_scaling"] = ts_report
                print(
                    f"  {task_id} TS: T*={ts_report['T_star']:.3f} "
                    f"ECE before={ts_report['ece_before']:.4f} "
                    f"after={ts_report['ece_after']:.4f}"
                )

    bb = _metrics_by_benchmark(cfg, records, all_traces, label_space)
    if bb:
        metrics["by_benchmark"] = bb
        print(f"\nPer-benchmark T2/T3 slices (data.benchmark_col): {sorted(bb.keys())}")

    with open(os.path.join(results_dir, "plantswarm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions and traces are appended inside the loop with fsync, so we do
    # NOT rewrite the JSONL files here — that would clobber resumed entries
    # from a previous partial run. The on-disk JSONLs are already authoritative.

    print(f"\nDone. Results saved to {results_dir}")
    return metrics


if __name__ == "__main__":
    main()
