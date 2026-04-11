# PlantSwarm: Entropy-Gated Emergent Routing in Multi-Agent VLM Swarms

**Paper:** *PlantSwarm: Entropy-Gated Emergent Routing in Multi-Agent VLM Swarms for Calibrated Plant Disease Diagnosis* (EMNLP-style sources: `plantswarm/latex/`, main file `acl_latex.tex`).

Evaluation uses **four** benchmarks: PlantVillage, PlantDoc, PlantWild, and LeafBench. LaTeX table bodies are generated from `results/plantswarm_metrics.json` via `scripts/sync_latex_metrics.py` (see **Outputs** and **LaTeX sync**).

## Repository structure

Repository root (your clone may be named e.g. `socioswarm` or `PlantSwarm`):

```
├── configs/                    # YAML: data paths, orchestrator, label spaces
│   ├── default.yaml
│   ├── cyag_directory.yaml     # Directory tree dataset (no parquet)
│   ├── smoke_100_autogen.yaml  # Small-run smoke / Slurm
│   ├── plant_village_tfds.yaml
│   ├── plantdoc_github.yaml
│   └── leafbench_hf.yaml
├── data/                       # Loaders, stratification, TFDS / HF helpers
├── agents/                     # Morphology, Symptom, Pathogen, Severity, Diagnosis
├── plantswarm/
│   ├── pipeline.py
│   ├── autogen_pipeline.py     # Default AutoGen Swarm runtime
│   ├── entropy_pipeline.py     # Entropy-driven routing (vLLM logprobs)
│   └── latex/                  # ACL paper: acl_latex.tex, plantswarm.bib, auto_*.tex (generated)
├── baselines/
├── ablations/
├── calibration/
├── bias/
├── utils/
├── results/                    # Default experiment output (metrics JSON, PDF copy)
└── scripts/
    ├── run_plantswarm.py       # Main pipeline entry
    ├── run_baselines.py
    ├── run_ablations.py
    ├── run_calibration.py
    ├── run_routing_analysis.py
    ├── run_bias_analysis.py
    ├── sync_latex_metrics.py  # results/*.json → plantswarm/latex/auto_*.tex
    ├── build_latex_pdf.sh      # acl_latex.tex → PDF (+ optional copy to --results-dir)
    ├── run_experiment_bundle.sh
    └── smoke_test.sh           # compileall + sync + PDF (local / CI)
```

Optional Slurm drivers live under `scripts/slurm/` (paths inside those files are cluster-specific).

If you maintain a second copy of the paper tree under `PlantSwarm/latex/`, it may be hard-linked to `plantswarm/latex/` on the same filesystem—edit one canonical path or verify with `ls -li`.

## Setup

```bash
pip install -r requirements.txt
```

Recommended (project-local venv):

```bash
python -m venv .venv311
source .venv311/bin/activate   # Windows: .venv311\Scripts\activate
pip install -r requirements.txt
```

## Smoke test (recommended before a full run)

Runs Python bytecode check, LaTeX metric sync, and PDF build (same stack as CI):

```bash
bash scripts/smoke_test.sh
```

With a specific interpreter:

```bash
PYTHON_BIN=.venv311/bin/python bash scripts/smoke_test.sh
```

This expects `results/` (at least `plantswarm_metrics.json` may be partial) and writes/updates `plantswarm/latex/auto_*.tex`, then builds `plantswarm/latex/acl_latex.pdf` and copies it to `results/paper_acl_latex.pdf` when successful.

## Google Colab (full pipeline)

`colab/PlantSwarm_full_pipeline.ipynb` is ordered as: **Drive + HF/TFDS caches** (and optional **`HF_TOKEN`** from Colab Secrets) → **one pip-install cell** → **clone** (defaults to `github.com/tirtho149/PlantSwarm`) → **§4: model IDs + resumable `snapshot_download` + Transformers smoke load** → **§5 YAML** (requires §4 globals) → **§6 scripts** (guard if cells were skipped). Use a **GPU** runtime; run cells **top to bottom**. You still need a reachable **OpenAI-compatible** server at `vllm_base_url` for HTTP inference steps. Upload to Colab or open from GitHub.

## Quick start

**vLLM (Linux + NVIDIA GPU):** Inference is an OpenAI-compatible HTTP server (default `http://localhost:8000/v1`). PlantSwarm needs a **vision-language** model (e.g. Qwen2.5-VL), not text-only Qwen2.5-3B-Instruct. For a 5-image smoke test with **Qwen2.5-VL-3B**, start the server then run:

```bash
bash scripts/serve_vllm_qwen25_vl_3b.sh   # GPU machine; see script header for pip install
python scripts/run_plantswarm.py --config configs/qwen25_vl_3b_smoke.yaml --subset 5
```

Results go to `results/qwen25_vl_3b_n5/`. On macOS, run vLLM on a remote GPU host and use SSH port forwarding (`-L 8000:localhost:8000`) so `localhost:8000` still works.

**Local text-only Swarm (no vLLM, notebook-style):** the same AutoGen Swarm pattern with a shared Hugging Face `Qwen2.5-3B-Instruct` and per-turn entropy is implemented in `plantswarm/autogen_pipeline.py` (`LocalQwenChatCompletionClient`, `run_local_qwen_text_swarm_demo`). Run:

```bash
python scripts/run_plantswarm.py --local-qwen-text-demo
```

Requires `torch`, `transformers`, and `accelerate` in the active venv (`pip install -r requirements.txt` or `pip install torch transformers accelerate`). This does **not** run PlantDiagBench images (use vLLM + `autogen_swarm` for that).

**Folder dataset (no parquet):** set `data.directory_root` to the image root and `data.parquet_path: null`. Labels are inferred from subdirectory names (e.g. `Crop/Disease/image.jpg` → T5/T3). See `data/directory_index.py` and `configs/cyag_directory.yaml`.

```bash
# Full PlantSwarm pipeline (AutoGen AgentChat Swarm; default in YAML)
python scripts/run_plantswarm.py --config configs/default.yaml

python scripts/run_baselines.py --config configs/default.yaml
python scripts/run_ablations.py --config configs/default.yaml
python scripts/run_bias_analysis.py --config configs/default.yaml
```

## Full run flow (end-to-end)

Default orchestration is **AutoGen Swarm** (`--orchestrator autogen_swarm`). For **entropy-driven routing** from vLLM chat logprobs, use `--orchestrator entropy_routing` (see `plantswarm/entropy_pipeline.py`). The value `classic` is rejected.

1. **Manual step-by-step** (good for debugging each stage)
2. **Bundled full run** (recommended for reproducible experiments and Slurm)

### A) Manual step-by-step flow

Run in this order (same `--config`):

```bash
python scripts/run_plantswarm.py --config configs/cyag_directory.yaml
python scripts/run_baselines.py --config configs/cyag_directory.yaml
python scripts/run_ablations.py --config configs/cyag_directory.yaml
python scripts/run_calibration.py --config configs/cyag_directory.yaml
python scripts/run_routing_analysis.py --config configs/cyag_directory.yaml
python scripts/run_bias_analysis.py --config configs/cyag_directory.yaml
python scripts/sync_latex_metrics.py --results-dir results/<run_dir> --latex-dir plantswarm/latex --subset-hint full
bash scripts/build_latex_pdf.sh --latex-dir plantswarm/latex --main-tex acl_latex.tex --results-dir results/<run_dir>
```

### B) Bundled full run (recommended)

`scripts/run_experiment_bundle.sh` runs: PlantSwarm → baselines → ablations → calibration → routing analysis → bias analysis → **sync LaTeX** → **build PDF**. Per-step logs go to `RESULTS_DIR/step_logs/`.

Required environment variables: `PYTHON_BIN`, `CONFIG_PATH`, `RESULTS_DIR`.

```bash
PYTHON_BIN=.venv311/bin/python \
CONFIG_PATH=configs/cyag_directory.yaml \
RESULTS_DIR=results/full_run \
ORCHESTRATOR=autogen_swarm \
SUBSET=0 \
ROUTING_SUBSET=0 \
bash scripts/run_experiment_bundle.sh
```

Optional:

- `SKIP_LATEX_SYNC=1` — skip step 07 (sync only once from a chosen directory after array jobs).
- `BUILD_LATEX_PDF=0` — skip PDF build.
- `STRICT_PDF_BUILD=1` — fail the bundle if PDF build fails (default is to warn and continue).

Notes:

- `SUBSET` empty or `0` means full data (see `run_plantswarm.py` / config).
- `ROUTING_SUBSET` defaults to `500` when `SUBSET` is unset; set explicitly for routing analysis subset size.

## Data configuration flow

### Directory-based dataset (no parquet/csv)

**Default `configs/cyag_directory.yaml`:** loads **Plant Village** from TensorFlow Datasets (`data.tfds_name: plant_village`, `image_col: image_bytes`). Install TFDS **on its own line** (shell comments break `pip` if pasted on the same line):

```bash
pip install -r requirements-tfds.txt
```

The first run downloads and prepares Plant Village (~827 MiB; cache under `~/tensorflow_datasets` by default). If you see `No module named 'importlib_resources'`, re-run the command above (that file lists `importlib_resources`).

TensorFlow may pin `protobuf` to v4 while `autogen-core` prefers v5; if AutoGen imports fail after installing TFDS, try `pip install 'protobuf>=5.29.3,<6'` and re-test (or use a separate venv for TFDS-only experiments).

For a **folder tree** on disk (e.g. CyAg on a cluster), use `configs/cyag_directory_cluster.yaml` or copy it and set:

- `data.parquet_path: null`
- `data.directory_root: /path/to/Curated_Dataset/Images`
- `data.tfds_name: null` (or omit TFDS keys) so the loader uses the directory branch
- optional `data.image_root` or `CYAG_IMAGE_ROOT` env var

Expected folder format (example):

```text
<root>/<crop>/<disease>/<image>.jpg
```

The loader infers task labels from path segments using `data/directory_index.py`.

### Plant Village (TensorFlow Datasets)

The [TFDS `plant_village`](https://www.tensorflow.org/datasets/catalog/plant_village) builder is supported without a local image tree or parquet file.

1. Install: `pip install tensorflow tensorflow-datasets` (see commented lines in `requirements.txt`).
2. Use `configs/plant_village_tfds.yaml`: set `data.tfds_name: plant_village`, `data.image_col: image_bytes`, and optional `data.tfds_max_examples`.
3. Optional: `data.benchmark_col: benchmark` so runs tag rows with `plantvillage` for LaTeX `by_benchmark` metrics.
4. Smoke test without the VLM: `python scripts/verify_tfds_plant_village.py --max-examples 16`

Implementation: `data/tfds_plant_village.py` builds an in-memory DataFrame consumed by the loader.

### LeafBench (Hugging Face)

The paper refers to the benchmark as **LeafBench**. Example config: `configs/leafbench_hf.yaml` with `data.hf_dataset_id` (e.g. `enalis/LeafBench`), `data.leafbench_question_types`, `data.image_col: image_bytes`. Use `datasets` + `HF_TOKEN` for gated access; copy `.env.example` to `.env` if present. Loader: `data/leafbench_hf.py`.

### PlantDoc (GitHub Cropped dataset)

The official Cropped **PlantDoc** release: [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) ([paper](https://doi.org/10.1145/3371158.3371196), CC BY 4.0). After cloning:

```bash
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git /path/to/PlantDoc-Dataset
python scripts/verify_plantdoc_repo.py /path/to/PlantDoc-Dataset --split train
```

Use `configs/plantdoc_github.yaml`: set `data.plantdoc_repo_root`, `data.plantdoc_split`, and `data.benchmark_col: benchmark` so metrics sync tags **`plantdoc`**. Loader: `data/plantdoc_github.py`.

## Slurm full flow

Cluster scripts share one **Nova / ISU scratch** template: `nodes=1`, `cpus-per-task=4`, `mem=32G`, `time=24:00:00`, `gres=gpu:1`, `partition=nova`, logs under `/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm/logs/`, `chdir` to that `PlantSwarm` tree, and mail to `tirtho@iastate.edu`. Copy from `scripts/run_all_tests.slurm` or any `scripts/slurm/*.slurm` when adding jobs; only `--job-name`, optional `--array`, and `%j` vs `%A_%a` in log names differ.

Submit examples:

```bash
sbatch scripts/run_all_tests.slurm
sbatch scripts/slurm/run_partial_100.slurm
sbatch scripts/slurm/run_partial_500.slurm
sbatch scripts/slurm/run_matrix_array.slurm
```

## Outputs and where to check

For each run directory (`results/<run_name>`), expect:

| Artifact | Role |
|----------|------|
| `plantswarm_metrics.json` | Main metrics; optional `by_benchmark` for wide table columns |
| `plantswarm_predictions.jsonl` | Per-image predictions |
| `traces/plantswarm_traces.jsonl` | Routing traces |
| `baseline_results.json` | Baseline rows (when run) |
| `ablation_metrics_T3.json` (and related) | Ablation table |
| `routing_analysis.json` | P1–P4 / mechanism / hedge statistics |
| `budget_sensitivity.json` | Optional; backtrack budget table |
| `bias_analysis*.json` | Bias / demographics analysis |
| `experiment_summary.json` | Written by `sync_latex_metrics.py` |
| `paper_acl_latex.pdf` | Copied from the LaTeX build when using `build_latex_pdf.sh --results-dir` |
| `step_logs/*.log` | From `run_experiment_bundle.sh` |

LaTeX **generated** fragments under `plantswarm/latex/` (regenerate with sync; do not hand-edit for numbers):

- `auto_metrics.tex` — inline macros from metrics / routing JSON
- `auto_table_main_results.tex` — main table (four benchmarks × T2/T3 + ECE + TPCP)
- `auto_table_predictions.tex`, `auto_table_ablation_results.tex`, `auto_table_mechanisms.tex`, `auto_table_budget.tex`

**Per-benchmark columns** in the main table fill when `data.benchmark_col` is set and values map to `plantvillage`, `plantdoc`, `plantwild`, `leafbench` (see `scripts/run_plantswarm.py`). Otherwise benchmark cells may show `---` while pooled ECE/TPCP can still sync.

## Re-run and sync policy

1. Re-run the experiment stages you need.
2. Run `scripts/sync_latex_metrics.py` with the intended `--results-dir` (and `--latex-dir plantswarm/latex` if not using the default).
3. Build the PDF with `scripts/build_latex_pdf.sh`.

For array jobs, pick one canonical `results/<dir>` before syncing.

## LaTeX sync (paper numbers)

```bash
python scripts/sync_latex_metrics.py --results-dir results/<run_dir> --latex-dir plantswarm/latex --subset-hint full
```

`--subset-hint` is embedded in `auto_metrics.tex` for traceability (e.g. Slurm `SUBSET`).

Optional **budget table**: add `results/<run>/budget_sensitivity.json` with `{"rows": [{"label": "...", "t3_f1": ..., "t2_f1": ..., "ece": ..., "mean_L": ..., "tpcp": ...}, ...]}` to populate `auto_table_budget.tex`.

## PDF build (TeX Live / TinyTeX)

- From repo root: `bash scripts/build_latex_pdf.sh --latex-dir plantswarm/latex --results-dir results/<run>` copies `acl_latex.pdf` to `results/paper_acl_latex.pdf` on success.
- The script exports `TEXINPUTS` / `BIBINPUTS` from your TeX tree when possible (helps TinyTeX and CI find `caption.sty` and other packages). If `kpsewhich` is unavailable, it falls back to common paths such as `~/Library/TinyTeX/texmf-dist`.
- ACL is **two-column**; the dataset licence appendix uses `table*` + `tabular` (not `longtable`).
- If `latexmk` or `pdflatex` is missing, install a minimal TeX distribution (e.g. TinyTeX) and ensure packages: `algorithms`, `algorithmicx`, `caption`, `booktabs`, `natbib` (ACL style files `acl.sty`, `acl_natbib.bst` ship beside `acl_latex.tex`).

## Troubleshooting

- **No images / wrong labels**: verify `data.directory_root` and folder depth; test with `--subset` or `SUBSET` first.
- **Image open errors**: ensure files are real images (not empty placeholders).
- **PDF build fails**: run `bash scripts/build_latex_pdf.sh ...` and read `plantswarm/latex/acl_latex.log`. Run `bash scripts/smoke_test.sh` after fixing TeX packages.
- **Citations undefined**: ensure `\cite` keys in `acl_latex.tex` exist in `plantswarm/latex/plantswarm.bib` (single `.bib` database; duplicate keys break BibTeX).
- **Tables show `---`**: rerun experiments with `by_benchmark` data or add `baseline_results.json` / ablation JSON as appropriate, then `sync_latex_metrics.py`.
- **Mismatch between JSON and paper**: confirm `--results-dir` points at the run you intend; rerun sync after new metrics.
- **TFDS `FileExistsError` / stuck Plant Village**: the loader clears `incomplete.*` temp dirs and ignores a rename conflict when the version folder already exists. If `as_dataset` still fails, reset the cache: `rm -rf ~/tensorflow_datasets/plant_village` and run again.

## Citation

Update the BibTeX entry in `plantswarm.bib` to match camera-ready venue metadata when available.
