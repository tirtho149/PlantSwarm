"""
scripts/run_ablations.py
=========================
Run all six factorial ablation variants (§6 RQ3, Table 3).

Usage:
    python scripts/run_ablations.py --config configs/default.yaml [--subset N]

Outputs:
    results/ablation_results.csv    — Table 3 equivalent with McNemar's test
    results/ablation_metrics.json   — full metrics per variant
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from ablations.runner import run_all_ablations
from data.loader import PlantDiagBenchLoader
from utils.vllm_client import VLLMClient, configure_vllm_client_from_yaml, validate_model_server_matches_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--task", default="T1", help="Primary task for comparison (default T1)")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    results_dir = args.output_dir or cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("Loading PlantDiagBench...")
    loader = PlantDiagBenchLoader(cfg["data"], split="test")
    label_space = loader.label_space
    records = list(loader)
    if args.subset:
        records = records[:args.subset]
    print(f"  {len(records)} images loaded")

    client = VLLMClient(
        base_url=cfg["model"]["vllm_base_url"],
        model=cfg["model"]["backbone"],
        temperature=cfg["model"]["temperature"],
        seed=cfg["model"]["seed"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
    )
    configure_vllm_client_from_yaml(client, cfg.get("model"), orchestrator="autogen_swarm")
    validate_model_server_matches_config(cfg)

    rw = cfg.get("routing") or {}
    print(f"\nRunning factorial ablations (Table 3) on task {args.task}...")
    df = run_all_ablations(
        client=client,
        records=records,
        label_space=label_space,
        task_id=args.task,
        results_dir=results_dir,
        bonferroni_correct=cfg["eval"]["bonferroni_correct"],
        Tmax=int(rw.get("Tmax", 15)),
        confidence_weights=rw.get("confidence_weights"),
    )

    print("\nAblation results (Table 3 equivalent):")
    print(df.to_string(index=False))

    # Also run T2 for completeness
    print(f"\nRunning ablations on T2...")
    df_t2 = run_all_ablations(
        client=client,
        records=records,
        label_space=label_space,
        task_id="T2",
        results_dir=results_dir,
        bonferroni_correct=cfg["eval"]["bonferroni_correct"],
        Tmax=int(rw.get("Tmax", 15)),
        confidence_weights=rw.get("confidence_weights"),
    )

    print(f"\nAblation results saved to {results_dir}/ablation_results*.csv")


if __name__ == "__main__":
    main()
