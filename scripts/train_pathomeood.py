"""
scripts/train_pathomeood.py
========================
Thin wrapper around BioCAP's ``open_clip_train.main`` that:
  1. Resolves a variant tag (see ``scripts/pathomeood_variants.sh``) into
     the right caption-strategy / projector / epoch knobs.
  2. Constructs shard glob strings for the train and val splits.
  3. Builds the torchrun command line and execs it (or echoes for dry
     runs).

The captions and shards must already exist at:
    data/bugwood_captions/<crop>_<strategy>.parquet
    data/wds_shards/<crop>_<strategy>/{train,val}/shard-*.tar

Run scripts/build_pathomeood_captions.py and scripts/build_pathomeood_shards.py
beforehand (or use scripts/e2e_nova.sh which chains them).

Usage:
    python scripts/train_pathomeood.py --variant T04 [--crop Tomato]
    python scripts/train_pathomeood.py --variant T04 --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from glob import glob
from pathlib import Path


# Mirror of scripts/pathomeood_variants.sh (kept in sync by review, not by code).
VARIANTS = {
    # tag : (strategy, proj, epochs, subset, paper_tables)
    "T01": ("label_only",         "dual",   50, "all",         "T3"),
    "T02": ("summary_only",       "dual",   50, "all",         "T3"),
    "T03": ("canonical_full",     "dual",   50, "all",         "T3"),
    "T04": ("canonical_deltas_3", "dual",   50, "all",         "T1,T3,T17,T18,T19,T20"),
    "T05": ("canonical_deltas_1", "dual",   50, "all",         "T6"),
    "T06": ("canonical_deltas_5", "dual",   50, "all",         "T6"),
    "T07": ("canonical_deltas_7", "dual",   50, "all",         "T6"),
    "T08": ("canonical_deltas_3", "single", 50, "all",         "Fig3"),
    "T09": ("canonical_deltas_3", "dual",  100, "all",         "Fig3"),
    "T10": ("canonical_deltas_3", "dual",   50, "covered",     "T4"),
    "T11": ("canonical_deltas_3", "dual",   50, "non_covered", "T4"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS),
                   help="variant tag from scripts/pathomeood_variants.sh")
    p.add_argument("--crop", default="Tomato",
                   help="crop tag to find shards under data/wds_shards/<crop>_<strategy>/")
    p.add_argument("--shards-root", default="data/wds_shards")
    p.add_argument("--save-root", default="train_and_eval/checkpoints")
    p.add_argument("--model", default="hf-hub:imageomics/bioclip",
                   help="Warm-start from BioCLIP (TreeOfLife-10M pretraining, no "
                        "caption supervision — non-tautological starting point for "
                        "demonstrating that KB captions add value). Pass ViT-B-16 + "
                        "--pretrained openai for from-scratch.")
    p.add_argument("--pretrained", default="",
                   help="Empty when --model is an hf-hub: path; 'openai' for "
                        "ViT-B-16 from-scratch.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="per-GPU batch size — locked encoders let us fit ~512 on one A100")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--nproc-per-node", type=int, default=1,
                   help="GPUs on this node (torchrun --nproc_per_node)")
    p.add_argument("--full-finetune", action="store_true",
                   help="UNFREEZE the backbones (~86M trainable). Default is "
                        "projectors-only training (--lock-image --lock-text, ~800K "
                        "trainable) since BioCLIP already supplies bio-vocab features.")
    p.add_argument("--dry-run", action="store_true",
                   help="echo the command, don't execute")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    strategy, proj, epochs, subset, paper_tables = VARIANTS[args.variant]

    repo_root  = Path(__file__).parent.parent.resolve()
    shards_dir = (repo_root / args.shards_root / f"{args.crop}_{strategy}").resolve()
    train_dir  = shards_dir / "train"
    val_dir    = shards_dir / "val"

    train_shards = sorted(glob(str(train_dir / "shard-*.tar")))
    val_shards   = sorted(glob(str(val_dir   / "shard-*.tar")))
    if not train_shards:
        raise SystemExit(
            f"no train shards under {train_dir}. Did "
            f"scripts/build_pathomeood_shards.py run for "
            f"{args.crop}/{strategy}?"
        )

    train_glob = f"{train_dir}/shard-{{{train_shards[0].split('-')[-1].split('.')[0]}..{train_shards[-1].split('-')[-1].split('.')[0]}}}.tar"
    val_glob   = (
        f"{val_dir}/shard-{{{val_shards[0].split('-')[-1].split('.')[0]}..{val_shards[-1].split('-')[-1].split('.')[0]}}}.tar"
        if val_shards else ""
    )

    save_dir = (repo_root / args.save_root / args.variant).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m", "open_clip_train.main",
        "--train-data",     train_glob,
        "--dataset-type",   "webdataset",
        "--pretrained",     args.pretrained,
        "--text-type",      "random",
        "--model",          args.model,
        "--batch-size",     str(args.batch_size),
        "--lr",             str(args.lr),
        "--warmup",         str(args.warmup),
        "--workers",        str(args.workers),
        "--epochs",         str(epochs),
        "--log-every-n-steps", "20",
        "--save-frequency", "5",
        "--logs",           str(save_dir),
        "--name",           args.variant,
        "--dataset-resampled",
        "--grad-checkpointing",
    ]
    if val_shards:
        cmd += ["--val-data", val_glob]
    if proj == "dual":
        cmd += ["--dual-projector"]
    # Single-projector is the absence of --dual-projector.
    if not args.full_finetune:
        # Projectors-only training. transformer.py::lock keeps proj and
        # caption_proj trainable; the rest of the visual + text towers are
        # frozen. ~800K trainable params vs ~86M for full fine-tune.
        # Default ON because BioCLIP already supplies strong bio-features.
        cmd += ["--lock-image", "--lock-text"]

    # The training module lives at train_and_eval/open_clip_train/ -
    # cd in so the `-m open_clip_train.main` import works exactly like
    # biocap/slurm/train.sh does. Shard + log paths are absolute so the
    # cd does not invalidate them.
    cwd = (repo_root / "train_and_eval").resolve()

    print("=== train_pathomeood ===")
    print(f"  variant       : {args.variant} ({paper_tables})")
    print(f"  strategy      : {strategy}")
    print(f"  projector     : {proj}")
    print(f"  epochs        : {epochs}")
    print(f"  model init    : {args.model} pretrained={args.pretrained or '(default)'}")
    print(f"  trainable     : {'FULL FINE-TUNE (~86M params)' if args.full_finetune else 'PROJECTORS-ONLY (~800K params)'}")
    print(f"  shards (train): {len(train_shards)}  in {train_dir}")
    print(f"  shards (val)  : {len(val_shards)}    in {val_dir}")
    print(f"  save_dir      : {save_dir}")
    print(f"  cwd           : {cwd}")
    print(f"  cmd           : {' '.join(cmd)}")
    if args.dry_run:
        print("  [dry-run] not executing")
        return
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
