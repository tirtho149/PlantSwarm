# PlantSwarm + PathomeDB + OBSERVE — Train on the Wild

**Paper:** *Train on the Wild: Geospatial Multi-Agent Routing for Cross-Crop Plant Disease Diagnosis from Ten Field Images* (EMNLP 2026, anonymous submission). Source: `plantswarm/latex/acl_latex.tex`.

---

## TL;DR

A three-step loop that decouples *what a disease looks like* from *what a multi-agent VLM swarm sees when routed against it*:

1. **Seed** the visual content of the knowledge base by shelling out to Claude headless (one prompt per (crop, disease)).
2. **Trace** with PlantSwarm — 5 agents over Qwen2.5-VL-7B, 30 stochastic runs per Bugwood image — against the seeded knowledge base.
3. **Enhance** the knowledge base with per-class aggregates from those traces (path length, backtrack rate, confusion targets).

Then train OBSERVE twice — once on the seed-only DB, once on the enhanced DB — and report the seed→enhanced delta on the full PlantVillage and PlantWild benchmarks. The headline result is the delta.

```
  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
  │  Phase 0 SEED  │  →   │ Phase 1 BUILD  │  →   │ Phase 2 TRACES │
  │ Claude headless│      │ symptoms.json +│      │ Qwen2.5-VL-7B  │
  │ writes 484     │      │ state/AEZ geo +│      │ × 5 agents     │
  │ VisualSymptom  │      │ 1,452 refs from│      │ × 30 runs      │
  │ blocks         │      │ Bugwood CSV    │      │ = 101k traces  │
  └────────────────┘      └────────────────┘      └────────┬───────┘
                                                            │
  ┌────────────────┐      ┌────────────────┐      ┌────────▼───────┐
  │ Phase 5 COMPARE│  ←   │ Phase 4 TRAIN  │  ←   │ Phase 3 ENHANCE│
  │ before / after │      │ OBSERVE × 2    │      │ mine traces →  │
  │ ΔT3 F1, ΔECE,  │      │ (seed DB,      │      │ SwarmObserva-  │
  │ ΔPathLen, …    │      │  enhanced DB)  │      │ tions per class│
  └────────────────┘      └────────────────┘      └────────────────┘
```

PathomeDB is two stores: `db.symptoms` (`SymptomLibrary`) and `db.refs` (`ReferenceLibrary`). The earlier 5-layer split (mechanistic pathway, cross-crop manifestation, regional epidemiology, decision graph, references) was retired in the post-CSV migration — see [`MIGRATION.md`](MIGRATION.md).

---

## Repository layout

```
PlantSwarm/
├── BugWood_Diseases.csv              raw IPMNet export (19,749 rows; pulled from Bugwood)
├── BugWood_Diseases_usable.csv       filtered subset (484 classes; produced by setup phase)
├── bugwood_classes_report.tsv        per-class candidate counts
├── configs/
│   ├── bugwood_pathome.yaml          training config (single source of truth)
│   ├── plantvillage_full_eval.yaml   held-out PV eval
│   └── plantwild_full_eval.yaml      held-out PW eval
├── data/bugwood_loader.py            CSV → BugwoodRecord stream
├── pathome/
│   ├── database.py                   PathomeDB orchestrator
│   ├── symptoms.py                   SymptomLibrary, SymptomProfile, VisualSymptom, SwarmObservations
│   └── layer5_references.py          ReferenceLibrary (CLIP + FAISS)
├── plantswarm/                       multi-agent routing pipelines (vLLM + hf_direct)
├── observe/                          OBSERVE student model (Qwen2.5-VL-7B + LoRA + DT + GRPO)
├── agents/                           5 routing agents (Morph / Symptom / Pathogen / Severity / Diagnosis)
├── utils/                            geo (state centroid + AEZ), trace I/O, vLLM/HF clients
├── scripts/
│   ├── filter_bugwood_csv.py         setup: CSV → filtered usable CSV
│   ├── seed_pathome_with_claude.py   phase 0: Claude headless seed
│   ├── build_pathome.py              phase 1: build PathomeDB
│   ├── run_pathome_traces.py         phase 2: PlantSwarm trace generation
│   ├── enhance_pathome_from_traces.py phase 3: trace → SwarmObservations
│   ├── train_observe_pathome.py      phase 4: DT + GRPO
│   ├── evaluate_pathome.py           phase 5a: held-out eval
│   ├── compare_pathome_versions.py   phase 5b: emits comparison.{json,md,tex}
│   ├── sync_pathome_metrics.py       LaTeX macro emitter
│   └── submit_pathome_*.sh           Nova SLURM scripts (one per phase + chain)
├── artifacts/                        pipeline outputs (gitignored)
├── results/                          eval JSONs + comparison artefacts (gitignored)
├── plantswarm/latex/acl_latex.tex    the paper
├── MIGRATION.md                      what changed across the symptom-centric refactor
└── README.md                         (this file)
```

---

## One-time prerequisites

### 1. Nova SSH + workspace

```bash
# Local machine
ssh tirtho@hpc-login.iastate.edu

# Nova login node
cd /work/mech-ai-scratch/tirtho/
git clone https://github.com/tirtho149/PlantSwarm.git    # or `git pull` if already present
cd PlantSwarm
mkdir -p logs artifacts results
```

### 2. Python environment

```bash
module load python cuda/11.8

# Create the venv (once)
python -m venv .venv
source .venv/bin/activate

# Install
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tfds.txt   # for the held-out PV eval
```

### 3. Claude Code CLI (Phase 0 dependency)

Phase 0 shells out to `claude -p` for ~484 calls. Install the CLI on the compute node and authenticate:

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --version
claude auth login         # opens an OAuth browser flow on first use
```

The CLI must be on `$PATH` from inside the SLURM allocation; the seed script bails out otherwise.

### 4. Bugwood IPMNet CSV

Download `BugWood_Diseases.csv` from the [Bugwood IPMNet](https://www.bugwood.org/ipmnet) export tool and place it at the repo root. The setup phase reads it and produces `BugWood_Diseases_usable.csv`.

### 5. (Optional) FAO GAEZ shapefile

To get the full ~17-zone AEZ resolution instead of the 2-zone coarse fallback, download the FAO GAEZ shapefile and:

```bash
export PATHOME_AEZ_SHAPEFILE=/path/to/FAO_AEZv4_50K.shp
```

Add this to your `~/.bashrc` so it's picked up by every SLURM allocation. Without it, the geo prior collapses to TMP / STM and Layer-3 priors are nearly degenerate (see [`MIGRATION.md`](MIGRATION.md) caveats).

---

## Quick start (chain everything)

```bash
# On Nova login node
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main
bash scripts/submit_pathome_all.sh
```

This queues seven jobs with `--dependency=afterok` chains so each phase waits for the previous to finish:

```
Setup    (CPU,            ~30 s)   →  Phase 0
Phase 0  (CPU,            15-30 min)  →  Phase 1
Phase 1  (CPU+net,        ~30 min)    →  Phase 2
Phase 2  (A100+vLLM,      ~36-50 h)   →  Phase 3
Phase 3  (CPU,            ~5 min)     →  Phase 4
Phase 4  (A100,           ~24 h)      →  Phase 5
Phase 5  (A100+CPU,       ~6-8 h)     →  comparison.md
```

Skip steps that are already done:

```bash
PATHOME_SKIP="setup,0"      bash scripts/submit_pathome_all.sh   # CSV + seed already on disk
PATHOME_FROM_PHASE=4        bash scripts/submit_pathome_all.sh   # restart from training
```

Monitor:

```bash
squeue -u $USER
tail -f logs/pathome_*-*.out
```

---

## Step-by-step

Each step has its own SLURM script under `scripts/`. Submit individually with `sbatch <script>` and override knobs via env vars (see each script's header).

### Setup — Filter Bugwood CSV

`scripts/submit_pathome_setup_filter.sh`

| | |
|---|---|
| **Purpose** | Normalise the raw IPMNet export (`BugWood_Diseases.csv`, 19,749 rows) into the per-class-thresholded subset the pipeline trains on. |
| **Compute** | 2 CPUs, 4 GB RAM, no GPU |
| **Walltime** | ~30 s |
| **Inputs** | `BugWood_Diseases.csv` (raw export at repo root) |
| **Outputs** | `BugWood_Diseases_usable.csv` (~11,513 rows / 484 classes), `bugwood_classes_report.tsv` |
| **Knobs** | `PATHOME_THRESHOLD` (default `10` rows/class; `15`→263 classes, `5`→982) |

```bash
sbatch scripts/submit_pathome_setup_filter.sh
# tighter subset:
PATHOME_THRESHOLD=15 sbatch scripts/submit_pathome_setup_filter.sh
```

### Phase 0 — Build the seed PathomeDB knowledge base (provenance-tracked)

`scripts/submit_pathome_phase0_seed.sh` → `python -m pathome_kb`

This is the **SAGE-ported `disease_registry` internet track**, adapted to the 484 (crop, disease) classes from the filtered CSV. Three stages per crop:

```
discovery       claude -p WebSearch per disease (parallel)  →  candidate URLs
                          │
                          ▼
extraction      fetch each URL  →  claude -p extracts disease records with
                VERBATIM QUOTES from the page text (never invents content)
                          │
                          ▼
reconciliation  merge per-source records into a canonical entry per disease.
                Every field stored as {value, url, quote}, so each visual
                fact in the KB is traceable to the exact sentence on the
                exact source page that supports it.
```

The orchestrator groups the 484 classes by crop, runs the internet track once per crop (so the discovery search can focus on one crop's disease catalogue), and merges per-crop registries into a single `symptoms_seed.json` consumable by Phase 1.

| | |
|---|---|
| **Purpose** | Build a provenance-tracked seed KB with `{value, url, quote}` per visual field. |
| **Compute** | 8 CPUs, 16 GB RAM, no GPU, outbound HTTPS for `api.anthropic.com` + every per-source page fetch |
| **Walltime** | 24 h budget. Quick mode (3 sources/crop): ~30 min. Full run (197 crops × ~5–15 sources each): typically 12–20 h depending on parallelism + page-fetch latency. |
| **Inputs** | `BugWood_Diseases_usable.csv`, authenticated `claude` CLI on PATH, `ANTHROPIC_API_KEY` in environment or `.env` |
| **Outputs** | `artifacts/pathome_kb/<Crop>/{discovery_results,raw_extractions,final_registry}.json` + `registry.md` + `internet.xlsx` per crop, plus the merged `artifacts/pathome_seed/symptoms_seed.json` for Phase 1 |
| **Knobs** | `PATHOME_SEED_QUICK=1` (smoke), `PATHOME_SEED_LIMIT=N` (first N crops), `PATHOME_SEED_ONLY_CROPS="Tomato,Soybean"`, `PATHOME_SEED_RESUME=discovery\|extraction\|reconciliation`, `PATHOME_SEED_NO_CACHE=1` |
| **Resume** | Yes, two levels. (1) Per-crop: any crop with an existing `final_registry.json` is skipped on re-run unless `PATHOME_SEED_NO_CACHE=1`. (2) Per-stage within a crop: `--resume-from extraction` reuses cached `discovery_results.json` etc. |
| **Cost** | ~$50–150 in Anthropic API spend for a full run; quick mode is ~$5. Per-source extraction is the dominant cost. |

```bash
# Smoke test on the top 3 crops (Tomato, Soybean, Corn) to validate auth + plumbing
PATHOME_SEED_QUICK=1 PATHOME_SEED_ONLY_CROPS="Tomato,Soybean,Corn" \
  sbatch scripts/submit_pathome_phase0_seed.sh

# Full run (12-20 h)
sbatch scripts/submit_pathome_phase0_seed.sh

# Resume only the reconciliation stage across crops that already have
# raw_extractions.json on disk:
PATHOME_SEED_RESUME=reconciliation sbatch scripts/submit_pathome_phase0_seed.sh

# Force every crop to re-run from scratch:
PATHOME_SEED_NO_CACHE=1 sbatch scripts/submit_pathome_phase0_seed.sh

# Run locally (no Nova) — same flags, same outputs:
python -m pathome_kb --csv BugWood_Diseases_usable.csv \
  --out artifacts/pathome_seed/symptoms_seed.json \
  --quick --only-crops "Tomato,Soybean"
```

**What lands in `symptoms_seed.json`** (per profile):

```json
{
  "profile_id": "Tomato::Early Blight",
  "crop": "Tomato",
  "disease": "Early Blight",
  "visual": {
    "plant_parts": ["leaf", "stem", "fruit"],
    "distinctive_signs": ["concentric rings on lesions", "yellow halo"],
    "confusion_diseases": ["Septoria leaf spot", "Late blight"],
    "notes": "Lesions begin on older leaves as small brown flecks ...",
    "sources": {
      "plant_parts":       [{"value": "leaf, stem, fruit", "url": "https://extension.../early-blight", "quote": "..."}],
      "distinctive_signs": [{"value": "concentric rings ... yellow halo", "url": "...", "quote": "..."}],
      "confusion_diseases":[{"value": "Septoria leaf spot; Late blight", "url": "...", "quote": "..."}],
      "notes":             [{"value": "Lesions begin ...", "url": "...", "quote": "..."}]
    }
  },
  ...
}
```

The structured tuples (color, shape, margin, texture, sporulation, progression) are intentionally left empty by this pipeline — the SAGE pipeline emits free-form prose with citations rather than discrete enums, and the auto re-observation prompt builds itself from whichever fields *are* populated. If you want the discrete tuples too, you can run the legacy schema-driven seeder afterward as an additive pass:

```bash
python scripts/seed_pathome_with_claude.py --workers 4 --model sonnet
```

That path skips profiles whose visual block is already populated, so it only fills the still-empty discrete fields without overwriting the SAGE-sourced content.

### Phase 1 — Build PathomeDB v1_seed

`scripts/submit_pathome_phase1_build.sh`

| | |
|---|---|
| **Purpose** | Layer the Claude seed JSON over the filtered CSV. Produces a `SymptomLibrary` (visual + per-state + per-AEZ counts + reference IDs) and a `ReferenceLibrary` (1,452 held-out images, lazily CLIP-indexed on first retrieval). |
| **Compute** | 8 CPUs, 32 GB RAM, no GPU, network for first-time Bugwood image downloads |
| **Walltime** | 6 h budget; ~30 min on first run (downloads ~600 MB to `.bugwood_cache/`), instant on subsequent builds (cache hit) |
| **Inputs** | `configs/bugwood_pathome.yaml`, `BugWood_Diseases_usable.csv`, `artifacts/pathome_seed/symptoms_seed.json` |
| **Outputs** | `artifacts/pathome_v1_seed/{symptoms.json, refs/, version.txt, build_summary.json}` |
| **Knobs** | `PATHOME_CONFIG`, `PATHOME_SEED_FILE`, `PATHOME_OUT_DIR` |

```bash
sbatch scripts/submit_pathome_phase1_build.sh
# alternate config / seed file:
PATHOME_CONFIG=configs/bugwood_pathome.yaml \
PATHOME_SEED_FILE=artifacts/pathome_seed/symptoms_seed.json \
PATHOME_OUT_DIR=artifacts/pathome_v1_seed \
  sbatch scripts/submit_pathome_phase1_build.sh
```

### Phase 2 — Generate PlantSwarm traces

`scripts/submit_pathome_phase2_traces.sh`

| | |
|---|---|
| **Purpose** | Run the 5-agent swarm (Morphology / Symptom / Pathogen / Severity / Diagnosis) over Qwen2.5-VL-7B against the seeded PathomeDB. 3,388 trace seeds × 30 stochastic runs at T=0.9 = **101,640 traces**. |
| **Compute** | 1× A100-80GB, 8 CPUs, 64 GB RAM; vLLM booted in-job |
| **Walltime** | 72 h budget; typical ~36–50 h for the full pass |
| **Inputs** | `artifacts/pathome_v1_seed/`, `BugWood_Diseases_usable.csv` (loader pulls trace split), Qwen2.5-VL-7B weights (HF cache or vLLM auto-download) |
| **Outputs** | `results/bugwood_seed/traces/plantswarm_traces.jsonl` (one JSON per trace, fsynced) |
| **Knobs** | `PATHOME_DB_DIR` (which DB the agents read from), `PATHOME_OUT_DIR` (where traces land) |
| **Resume** | Yes — already-persisted `image_id`s are skipped on resubmit. A walltime kill is recoverable. |

```bash
sbatch scripts/submit_pathome_phase2_traces.sh
# point at a different DB version:
PATHOME_DB_DIR=artifacts/pathome_v1_seed \
PATHOME_OUT_DIR=results/bugwood_seed \
  sbatch scripts/submit_pathome_phase2_traces.sh
```

If vLLM fails to start (driver mismatch, OOM during weight load), the loader falls back to `hf_direct` mode automatically — slower but proven memory-safe after the recent allocator fix.

### Phase 3 — Enhance DB from traces

`scripts/submit_pathome_phase3_enhance.sh`

| | |
|---|---|
| **Purpose** | Mine the 101,640 traces into per-class `SwarmObservations` (n_traces, avg_path_length, backtrack_rate, high_confidence_rate, confusion_targets) and attach onto the matching `SymptomProfile`. |
| **Compute** | 4 CPUs, 16 GB RAM, no GPU |
| **Walltime** | 1 h budget; ~5 min in practice |
| **Inputs** | `artifacts/pathome_v1_seed/` (the seed DB), `results/bugwood_seed/traces/plantswarm_traces.jsonl` |
| **Outputs** | `artifacts/pathome_v1_enhanced/{symptoms.json, refs/, version.txt, enhancement_summary.json}` |
| **Knobs** | `PATHOME_SEED_DB`, `PATHOME_TRACES`, `PATHOME_OUT_DIR` |

```bash
sbatch scripts/submit_pathome_phase3_enhance.sh
```

The visual block from Phase 0 is left untouched. Enhancement is strictly additive on the empirical fields — that's what makes the seed-vs-enhanced ablation clean.

### Phase 4 — Train OBSERVE × 2

`scripts/submit_pathome_phase4_train.sh`

| | |
|---|---|
| **Purpose** | Train OBSERVE twice on the same trace set, differing only in which PathomeDB the agents read from at training time. Each run does Phase A (Decision Transformer) + Phase B (GRPO refinement). |
| **Compute** | 1× A100-80GB, 8 CPUs, 128 GB RAM |
| **Walltime** | 24 h budget; ~10–14 h DT + ~6–8 h GRPO per checkpoint, sequential. |
| **Inputs** | `artifacts/pathome_v1_seed/`, `artifacts/pathome_v1_enhanced/`, traces from Phase 2, `configs/bugwood_pathome.yaml` |
| **Outputs** | `observe/checkpoints/seed/observe_grpo_epoch_*.pt`, `observe/checkpoints/enhanced/observe_grpo_epoch_*.pt`, training-history JSONs |
| **Knobs** | `PATHOME_SEED_DB`, `PATHOME_ENHANCED_DB`, `PATHOME_CONFIG` |

```bash
sbatch scripts/submit_pathome_phase4_train.sh
```

### Phase 5 — Eval × 4 + before/after compare

`scripts/submit_pathome_phase5_eval.sh`

| | |
|---|---|
| **Purpose** | Evaluate both checkpoints on full PlantVillage (with seen/unseen slice) and full PlantWild, then run `compare_pathome_versions.py` to emit the headline before/after artefact. |
| **Compute** | 1× A100-80GB, 8 CPUs, 64 GB RAM; one vLLM instance reused across all four evaluations |
| **Walltime** | 8 h budget; typical ~6 h |
| **Inputs** | both OBSERVE checkpoints, `configs/plantvillage_full_eval.yaml`, `configs/plantwild_full_eval.yaml`, traces from Phase 2 (for the trace-quality side of the comparison) |
| **Outputs** | `results/pathome_compare/{seed,enhanced}/{pv,pw}/pathome_eval.json`, `results/pathome_compare/comparison.{json,md,tex}` |
| **Knobs** | `PATHOME_SEED_CKPT`, `PATHOME_ENHANCED_CKPT`, `PATHOME_PV_CONFIG`, `PATHOME_PW_CONFIG`, `PATHOME_UNSEEN_CLASSES` (CSV string), `PATHOME_RESULTS_BASE` |

```bash
sbatch scripts/submit_pathome_phase5_eval.sh
# pin checkpoints / unseen slice explicitly:
PATHOME_SEED_CKPT=observe/checkpoints/seed/observe_grpo_epoch_10.pt \
PATHOME_ENHANCED_CKPT=observe/checkpoints/enhanced/observe_grpo_epoch_10.pt \
PATHOME_UNSEEN_CLASSES="Tomato Spotted Wilt Virus,Late Blight" \
  sbatch scripts/submit_pathome_phase5_eval.sh
```

The `comparison.tex` file emits LaTeX macros (`\PathomeDeltaTthreeF`, `\PathomeDeltaTthreeECE`, `\PathomeDeltaPathLen`, …) which the paper picks up via `\input{auto_pathome_metrics}` near the headline before/after table.

---

## Configuration

The single source of truth is `configs/bugwood_pathome.yaml`. The most-tweaked knobs:

```yaml
data:
  csv_path: "BugWood_Diseases_usable.csv"
  per_class: 10                # max images per (crop, disease) — admit budget
  trace_split: 7               # first N → trace seeds; remainder → references
  min_per_class: 10            # drop classes below this row count

routing:
  orchestrator: "autogen_swarm"  # or "hf_direct" for single-GPU fallback
  Tmax: 15                       # max path length per trace
  runs_per_image: 30             # stochastic runs per Bugwood seed image

model:
  backbone: "Qwen/Qwen2.5-VL-7B-Instruct"
  temperature: 0.9
  vllm_base_url: "http://localhost:8000/v1"

observe:
  backbone: "Qwen/Qwen2.5-VL-7B-Instruct"
  oc_threshold: 0.55             # paper §7.2 overconfidence cutoff
  decision_transformer:
    epochs: 50
    patience: 5
  grpo:
    epochs: 10
    rollouts_per_instance: 8
    beta_kl: 0.04
```

The two eval configs (`plantvillage_full_eval.yaml`, `plantwild_full_eval.yaml`) override `data.*` and `output.results_dir` only — model + routing settings are inherited.

---

## Troubleshooting

### vLLM fails to boot

Most common cause is a CUDA driver / vLLM version mismatch. The vLLM stderr is in `logs/vllm-<JOB>.log`. To force the HF-direct fallback (slower, but always works):

```bash
PLANTSWARM_MODE=hf_direct sbatch scripts/submit_pathome_phase2_traces.sh
```

### CUDA OOM mid-run (HF direct only)

The HFClient is patched to release reserved-but-unallocated GPU memory after every generation. If you still see OOM:

1. Confirm the SLURM script exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already added in current scripts).
2. Drop `model.max_new_tokens` from `512` → `256` in `configs/bugwood_pathome.yaml`.
3. Drop the image cap in `utils/hf_client.py:_MAX_IMAGE_SIDE` from `1024` → `768`.
4. Last resort: switch to vLLM which has paged attention.

### Walltime kill mid-trace-generation

Trace JSONL is appended with fsync after each trace, so already-persisted `image_id`s are skipped on resubmit. Just `sbatch` the script again — no manual cleanup needed.

### Phase 0 "claude CLI not found"

The `claude` binary needs to be on `$PATH` from inside the SLURM allocation. Install on the compute node:

```bash
ssh nova-compute-XX     # or run inside an interactive sbatch
curl -fsSL https://claude.ai/install.sh | bash
claude auth login
```

Then resubmit Phase 0.

### Layer-3 prior is degenerate

The coarse-fallback AEZ table maps the entire US footprint into 2 zones (TMP, STM). Set `PATHOME_AEZ_SHAPEFILE` to a real FAO GAEZ shapefile to recover the ~17-zone resolution. State-resolution priors via `state_counts` work either way.

---

## Output directory map

After a complete run:

```
PlantSwarm/
├── BugWood_Diseases_usable.csv               (Setup)
├── bugwood_classes_report.tsv                (Setup)
├── artifacts/
│   ├── pathome_seed/
│   │   ├── symptoms_seed.json                (Phase 0)
│   │   └── failed.jsonl                       (Phase 0)
│   ├── pathome_v1_seed/                      (Phase 1)
│   │   ├── symptoms.json
│   │   ├── refs/
│   │   ├── version.txt
│   │   └── build_summary.json
│   └── pathome_v1_enhanced/                  (Phase 3)
│       ├── symptoms.json
│       ├── refs/
│       └── enhancement_summary.json
├── results/
│   ├── bugwood_seed/
│   │   └── traces/plantswarm_traces.jsonl    (Phase 2)
│   └── pathome_compare/
│       ├── seed/{pv,pw}/pathome_eval.json    (Phase 5)
│       ├── enhanced/{pv,pw}/pathome_eval.json (Phase 5)
│       ├── comparison.json                    (Phase 5)
│       ├── comparison.md                      (Phase 5 — main output)
│       └── comparison.tex                     (Phase 5 — paper macros)
├── observe/checkpoints/
│   ├── seed/observe_grpo_epoch_*.pt          (Phase 4)
│   └── enhanced/observe_grpo_epoch_*.pt      (Phase 4)
└── logs/pathome_*-*.{out,err}                SLURM stdout/stderr
```

Both `artifacts/` and `results/` are gitignored — sync via the workflow below if you want them mirrored locally.

---

## Two-way sync workflow (Local ↔ GitHub ↔ Nova)

```
┌──────────┐  push code  ┌────────┐   git pull   ┌──────┐
│  Local   │──────────→  │ GitHub │ ───────────→ │ Nova │
│          │  ←──────── │        │ ←─────────── │      │
└──────────┘  pull res.  └────────┘  push res.   └──────┘
```

**Code: Local → Nova**
```bash
# Local
git add <files> && git commit -m "..." && git push origin main

# Nova
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
```

**Results: Nova → Local**
```bash
# Nova
git add MIGRATION.md README.md   # if updated
# results/ and artifacts/ are gitignored — pull via rsync if desired:
git push origin main             # for paper edits etc.

# Local
rsync -avz nova-login:/work/mech-ai-scratch/tirtho/PlantSwarm/results/ ./results/
rsync -avz nova-login:/work/mech-ai-scratch/tirtho/PlantSwarm/artifacts/ ./artifacts/
git pull origin main
```

If you want pipeline outputs in git too (not recommended for the trace JSONL — it's hundreds of MB), drop the relevant entries from `.gitignore` and stage explicitly.

---

## Compile the paper

```bash
cd plantswarm/latex
latexmk -pdf acl_latex.tex
```

If you've run Phase 5, `\input{auto_pathome_metrics}` near the headline table picks up the `\PathomeDelta*` macros emitted by `compare_pathome_versions.py` and the table fills in automatically.

The paper in this repo currently describes the symptom-centric PathomeDB construction in §6 plus the original 5-layer narrative in some legacy sections — see [`MIGRATION.md`](MIGRATION.md) for which sections are reconciled to current code and which are intentionally kept verbatim from the prior draft.

---

## Known limitations

- The Bugwood IPMNet CSV is **US-only** at state granularity. International deployment requires either a different export with finer GPS or a separate regional knowledge base.
- The coarse-fallback FAO AEZ table resolves the US footprint into 2 zones; the full ~17-zone resolution requires `PATHOME_AEZ_SHAPEFILE` pointing at a real GAEZ shapefile.
- About half of the 484 admitted classes appear in only one state, contributing no spatial-variance signal; the geo prior is informative on the multi-state subset only.
- The earlier formulation's monthly AEZ priors and EPPO Pearson-r validation are unsupported by the IPMNet export and have been dropped from the methodology.
- `claude -p` seed quality varies with disease prevalence in Claude's training data; rare or recently-described diseases may return empty visual fields.

---

## Citations

If you use this code or the released artefacts, please cite:

```bibtex
@inproceedings{plantswarm2026,
  title     = {Train on the Wild: Geospatial Multi-Agent Routing for
               Cross-Crop Plant Disease Diagnosis from Ten Field Images},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```

Bugwood IPMNet images are publicly available under academic and extension-service terms — see [bugwood.org](https://www.bugwood.org/) for citation expectations on individual images.
