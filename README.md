# PlantSwarm + PathomeDB + OBSERVE — Train on the Wild

**Paper:** *Train on the Wild: Geospatial Multi-Agent Routing for Cross-Crop Plant Disease Diagnosis from Ten Field Images* (EMNLP 2026, anonymous submission). Source: `plantswarm/latex/acl_latex.tex`.

---

## TL;DR

A three-step loop that decouples *what a disease looks like* from *what a multi-agent VLM swarm sees when routed against it*:

1. **Seed** the visual content of the knowledge base by running the SAGE-ported `pathome_kb` pipeline (Claude headless web discovery → URL extraction with verbatim quotes → reconciliation → state-aware image cache → per-state VLM **delta** extraction). One unified `final_registry.json` per crop holds canonical text plus image-grounded per-state deltas.
2. **Trace** with PlantSwarm — 5 agents over Qwen2.5-VL-7B, 30 stochastic runs per Bugwood image — against the seeded KB.
3. **Enhance** the KB by mining those traces into per-class `SwarmObservations` (path length, backtrack rate, confusion targets).

Then train OBSERVE twice — once on the seed-only KB, once on the enhanced KB — and report the seed→enhanced delta on the full PlantVillage and PlantWild benchmarks. **The headline result is the delta.**

```
  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
  │  Phase 0 SEED  │  →   │ Phase 1 BUILD  │  →   │ Phase 2 TRACES │
  │ pathome_kb     │      │ symptoms.json +│      │ Qwen2.5-VL-7B  │
  │ (5 stages,     │      │ state/AEZ geo +│      │ × 5 agents     │
  │  LOCAL only —  │      │ refs from      │      │ × 30 runs      │
  │  Claude OAuth) │      │ Bugwood CSV    │      │ = ~100k traces │
  └────────────────┘      └────────────────┘      └────────┬───────┘
                                                            │
  ┌────────────────┐      ┌────────────────┐      ┌────────▼───────┐
  │ Phase 5 COMPARE│  ←   │ Phase 4 TRAIN  │  ←   │ Phase 3 ENHANCE│
  │ before / after │      │ OBSERVE × 2    │      │ mine traces →  │
  │ ΔT3 F1, ΔECE,  │      │ (seed DB,      │      │ SwarmObserva-  │
  │ ΔPathLen, …    │      │  enhanced DB)  │      │ tions per class│
  └────────────────┘      └────────────────┘      └────────────────┘
```

PathomeDB is two stores: `db.symptoms` (`SymptomLibrary`) and `db.refs` (`ReferenceLibrary`). Each `SymptomProfile` splits into a `CanonicalDisease` block (text-grounded, URL+verbatim quote per field) and `regional_observations[state].deltas[]` (image-grounded `{field, canonical_says, image_shows, image_quote}` records). Canonical owns the symptom slots; regional only emits state-specific additions or contradictions — a decision-tree shape rather than parallel re-extraction.

---

## Where each phase runs

The pipeline splits across two machines:

```
   ┌──────────────────────────┐                 ┌────────────────────────┐
   │       LOCAL machine      │     GitHub      │     Nova compute       │
   │  (laptop / workstation)  │      git        │  (SLURM-scheduled GPU) │
   └──────────────────────────┘                 └────────────────────────┘

   Phase 0  pathome_kb         ────push──→     git pull
   KB build (~45 min for                          ↓
   2 crops, ~24 h for 484)                    Setup    Filter Bugwood CSV
       ↓                                      Phase 1  Build PathomeDB
   symptoms_seed.json                         Phase 2  PlantSwarm traces  (A100)
                                              Phase 3  Enhance from traces
                                              Phase 4  Train OBSERVE × 2  (A100)
                                              Phase 5  Eval × 4 + compare
```

**Why the split.** Phase 0 needs the `claude` CLI's OAuth login flow, which Nova compute nodes don't allow. Everything else is pure compute (Python + Qwen2.5-VL-7B) and runs as ordinary SLURM jobs.

**Handoff.** Phase 0 produces a single JSON file (`smoke/artifacts/pathome_seed/symptoms_seed.json` for the smoke flow, or `artifacts/pathome_seed/symptoms_seed.json` for production). You `git add -f` + push it from your laptop and `git pull` it on Nova.

---

## Repository layout

```
PlantSwarm/
├── BugWood_Diseases.csv                  raw IPMNet export (committed)
├── BugWood_Diseases_usable.csv           filtered subset (committed; regenerable via Setup)
├── bugwood_classes_report.tsv            per-class candidate counts
│
├── configs/
│   ├── bugwood_pathome.yaml              training config (single source of truth)
│   ├── plantvillage_full_eval.yaml       held-out PV eval
│   └── plantwild_full_eval.yaml          held-out PW eval
│
├── pathome_kb/                           Phase 0 — SAGE-ported KB build (LOCAL)
│   ├── pipeline.py                       per-crop orchestrator + seed merge
│   ├── internet_pipeline.py              discovery → extraction → reconciliation
│   ├── regional_observation.py           per-(crop,disease,state) image-grounded deltas
│   ├── symptoms_adapter.py               registry → SymptomProfile JSON
│   ├── prompts/                          discovery / extraction / reconciliation
│   ├── shared.py, utils.py, config.py
│   └── README.md                         schema diagram + worked example
│
├── pathome/                              Phase 1+ — PathomeDB stores
│   ├── database.py                       PathomeDB orchestrator
│   ├── symptoms.py                       SymptomLibrary, SymptomProfile,
│   │                                     CanonicalDisease, RegionalObservation,
│   │                                     RegionalDelta, Citation, SwarmObservations
│   └── layer5_references.py              ReferenceLibrary (CLIP + FAISS)
│
├── data/bugwood_loader.py                CSV → BugwoodRecord stream
├── plantswarm/                           multi-agent routing pipelines
├── observe/                              OBSERVE student (Qwen2.5-VL-7B + LoRA + DT + GRPO)
├── agents/                               5 routing agents
├── utils/                                geo (state centroid + AEZ), trace I/O, vLLM/HF
├── calibration/                          ECE, temperature scaling, conformal
│
├── scripts/
│   ├── filter_bugwood_csv.py             setup: CSV → filtered usable CSV
│   ├── ensure_state_image_cache.py       per-(crop,disease,state) Bugwood image cache
│   ├── registry_to_excel.py              final_registry.json → 1-sheet decision-tree xlsx
│   ├── build_pathome.py                  Phase 1 — build PathomeDB
│   ├── run_pathome_traces.py             Phase 2 — PlantSwarm trace generation
│   ├── enhance_pathome_from_traces.py    Phase 3 — trace mining
│   ├── train_observe_pathome.py          Phase 4 — DT + GRPO
│   ├── evaluate_pathome.py               Phase 5a — held-out eval
│   ├── compare_pathome_versions.py       Phase 5b — comparison.{json,md,tex}
│   ├── sync_pathome_metrics.py           LaTeX macro emitter
│   ├── submit_pathome_setup_filter.sh    NOVA: filter CSV (~30 s, CPU)
│   ├── submit_pathome_phase1_build.sh    NOVA: build DB (~30 min, CPU+net)
│   ├── submit_pathome_phase2_traces.sh   NOVA: traces (~36–50 h, A100+vLLM)
│   ├── submit_pathome_phase3_enhance.sh  NOVA: enhance (~5 min, CPU)
│   ├── submit_pathome_phase4_train.sh    NOVA: OBSERVE × 2 (~24 h, A100)
│   ├── submit_pathome_phase5_eval.sh     NOVA: eval+compare (~6–8 h, A100)
│   └── submit_pathome_all.sh             NOVA: chain Setup + Phases 1–5
│
├── smoke/                                two-crop end-to-end runner (the documented happy path)
│   ├── run_phase0_full.sh                LOCAL: full Phase 0 for 2 crops
│   ├── submit_smoke.sh                   NOVA: Phase 1–5 in one A100 job
│   ├── BugWood_Diseases_smoke.csv        2-crop subset of the IPMNet export
│   └── README.md                         smoke specifics
│
├── artifacts/                            pipeline outputs (gitignored; seed pushed via -f)
├── results/                              eval JSONs + comparison artefacts (gitignored)
│
├── plantswarm/latex/acl_latex.tex        the paper
└── README.md                             (this file)
```

---

## One-time prerequisites

### On your local machine

```bash
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For Phase 0 you also need:

```bash
# Claude Code CLI (used for the discovery WebSearch + extraction + regional VLM stages)
curl -fsSL https://claude.ai/install.sh | bash
claude auth login        # OAuth in browser

# Anthropic SDK key (optional — slightly faster reconciliation; falls back to claude -p without it)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

The Bugwood IPMNet CSV (`BugWood_Diseases.csv`) is committed.

### On Nova

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/
git clone https://github.com/tirtho149/PlantSwarm.git    # first time only
cd PlantSwarm
mkdir -p logs

module load python cuda/11.8
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tfds.txt        # for the held-out PV eval
```

Optional: get the FAO GAEZ shapefile to upgrade the geo prior from the 2-zone coarse fallback to the full ~17-zone resolution. Add to `~/.bashrc` so SLURM sees it:

```bash
export PATHOME_AEZ_SHAPEFILE=/path/to/FAO_AEZv4_50K.shp
```

---

## Full run — two crops (the documented happy path)

The canonical end-to-end flow is **two crops** (default Soybean + Tomato; ~25 disease classes; ~50 (disease, state) tuples). It exercises every stage of the pipeline at a fraction of the time and cost of a 484-class production run, so all plumbing issues surface before you commit to the full spend.

### Phase 0 — LOCAL (~45 min, ~$5–15)

```bash
# default: Soybean + Tomato
SMOKE_CROPS="Soybean,Tomato" bash smoke/run_phase0_full.sh
```

The script runs five stages per crop and prints a summary at the end:

```
  1. Filter the smoke CSV               → BugWood_Diseases_smoke_usable.csv
  2. State-aware image cache top-up     → smoke/.bugwood_cache/
  3. Cross-region SAGE pipeline         → final_registry.json (canonical-only)
       discovery (claude -p WebSearch)
       extraction (claude -p)            → verbatim quotes + treatments
       reconciliation (claude -p)        → canonical entries
  4. Per-state VLM delta extraction     → embedded into final_registry.json
       claude -p + Read tool reads each cached Bugwood image and the
       canonical KB; emits ONLY structured deltas
       {field, canonical_says, image_shows, image_quote}
  5. Adapter merge                      → smoke/artifacts/pathome_seed/symptoms_seed.json
```

After the run:

```
artifacts/pathome_kb/<Crop>/
  ├── discovery_results.json            URL cache
  ├── raw_extractions.json              per-source quotes
  ├── final_registry.json               UNIFIED — canonical + per-disease
  │                                     regional_observations[state].deltas
  ├── final_registry.xlsx               1-sheet decision-tree view
  └── registry.md                       human-readable canonical summary

smoke/artifacts/pathome_seed/symptoms_seed.json    final assembled KB
```

Convert the unified registry to Excel for inspection:

```bash
python3 scripts/registry_to_excel.py \
    artifacts/pathome_kb/Soybean/final_registry.json \
    --out artifacts/pathome_kb/Soybean/final_registry.xlsx
```

Knobs (env vars on `smoke/run_phase0_full.sh`):

```bash
SMOKE_CROPS="Soybean,Tomato"   # default; any 2+ crops in the smoke CSV
FULL_QUICK=1                    # cap sources/states for fast iteration (~15-25 min, ~$1-3)
FULL_KEEP_CACHE=1               # reuse cached final_registry.json (skip re-running canonical)
FULL_SKIP_SETUP=1               # CSV already filtered
FULL_SKIP_CACHE=1               # image cache already topped up
FULL_SKIP_KB=1                  # skip pathome_kb (no-op smoke)
```

Push the seed to GitHub:

```bash
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           smoke/BugWood_Diseases_smoke_usable.csv \
           artifacts/pathome_kb/Soybean/{discovery_results,final_registry}.json \
           artifacts/pathome_kb/Tomato/{discovery_results,final_registry}.json
git commit -m "Phase 0: regenerate two-crop seed"
git push origin main
```

### Phases 1–5 — NOVA (~60–90 min for two crops on a single A100)

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main
sbatch smoke/submit_smoke.sh
tail -f logs/pathome_smoke-*.out
```

A single A100 job runs Setup → Phase 1 → 2 → 3 → 4 → 5 sequentially. Final output drops at `results/pathome_compare/comparison.md`.

See [`smoke/README.md`](smoke/README.md) for what's downscaled (per_class images, agent prompts, training epochs), skip/resume knobs, and expected outputs.

---

## Going to production (484 classes)

For the full run, swap the smoke wrappers for the per-phase SLURM submitters:

```bash
# === LOCAL (Phase 0, ~16–24 h, ~$60–180) ===
python -m pathome_kb \
  --csv BugWood_Diseases_usable.csv \
  --out artifacts/pathome_seed/symptoms_seed.json \
  --regional

git add -f artifacts/pathome_seed/symptoms_seed.json
git add -f artifacts/pathome_kb/                    # optional audit trail
git commit -m "Phase 0 seed (484 classes)"
git push origin main

# === NOVA ===
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
bash scripts/submit_pathome_all.sh
# Setup → Phase 1 → 2 → 3 → 4 → 5 (sbatch dependency chain)
```

Skip steps that are already done:

```bash
PATHOME_SKIP="setup"     bash scripts/submit_pathome_all.sh   # CSV already filtered
PATHOME_FROM_PHASE=4     bash scripts/submit_pathome_all.sh   # restart at training
```

Monitor:

```bash
squeue -u $USER
tail -f logs/pathome_*-*.out
```

Final output drops at `results/pathome_compare/comparison.md`.

---

## Step-by-step

### Phase 0 — Build the seed PathomeDB knowledge base (LOCAL only)

`smoke/run_phase0_full.sh` (two crops) or `python -m pathome_kb` (any subset)

> ⚠ **Runs on your local machine, not on Nova.** Nova compute nodes block the OAuth login flow that `claude` headless needs.

| | |
|---|---|
| **Where it runs** | LOCAL machine |
| **Compute** | CPU; outbound HTTPS for `api.anthropic.com` + per-source page fetches |
| **Walltime** | Smoke (2 crops): ~45 min full / ~15-25 min `FULL_QUICK=1`. Production (197 crops × ~5–15 sources each): 16–24 h |
| **Inputs** | `BugWood_Diseases.csv` (or smoke CSV), authenticated `claude` CLI, optional `ANTHROPIC_API_KEY` |
| **Outputs** | `artifacts/pathome_kb/<Crop>/{discovery_results,raw_extractions,final_registry}.json` + merged `symptoms_seed.json` |
| **Handoff** | `git add -f symptoms_seed.json && git commit && git push` |
| **Cost** | Smoke: ~$5–15. Production: ~$60–180 in Anthropic API spend. |

See [`pathome_kb/README.md`](pathome_kb/README.md) for the full schema diagram, the deltas-only prompt, and a worked example (`Soybean :: Charcoal Rot` with Alabama field-view vs Kentucky close-up specimens).

### Setup — Filter Bugwood CSV (Nova)

`scripts/submit_pathome_setup_filter.sh`

| | |
|---|---|
| **Purpose** | Normalise the raw IPMNet export into the per-class-thresholded subset the pipeline trains on. |
| **Compute** | 2 CPUs, 4 GB RAM, no GPU |
| **Walltime** | ~30 s |
| **Inputs** | `BugWood_Diseases.csv` |
| **Outputs** | `BugWood_Diseases_usable.csv` (~11,513 rows / 484 classes), `bugwood_classes_report.tsv` |
| **Knobs** | `PATHOME_THRESHOLD` (default 10 rows/class; `15`→263 classes, `5`→982) |

```bash
sbatch scripts/submit_pathome_setup_filter.sh
PATHOME_THRESHOLD=15 sbatch scripts/submit_pathome_setup_filter.sh
```

### Phase 1 — Build PathomeDB v1_seed (Nova)

`scripts/submit_pathome_phase1_build.sh`

| | |
|---|---|
| **Purpose** | Layer the seed JSON over the filtered CSV. Produces `SymptomLibrary` (canonical + regional deltas + per-state + per-AEZ counts + reference IDs) and `ReferenceLibrary` (1,452 held-out images, lazily CLIP-indexed on first retrieval). |
| **Compute** | 8 CPUs, 32 GB RAM, no GPU, network for first-time Bugwood image downloads |
| **Walltime** | 6 h budget; ~30 min on first run, instant on subsequent (cache hit) |
| **Inputs** | `configs/bugwood_pathome.yaml`, `BugWood_Diseases_usable.csv`, `artifacts/pathome_seed/symptoms_seed.json` |
| **Outputs** | `artifacts/pathome_v1_seed/{symptoms.json, refs/, version.txt, build_summary.json}` |

### Phase 2 — Generate PlantSwarm traces (Nova)

`scripts/submit_pathome_phase2_traces.sh`

| | |
|---|---|
| **Purpose** | Run the 5-agent swarm over Qwen2.5-VL-7B against the seeded PathomeDB. ~3,388 trace seeds × 30 stochastic runs at T=0.9. |
| **Compute** | 1× A100-80GB, 8 CPUs, 64 GB RAM; vLLM booted in-job |
| **Walltime** | 72 h budget; typical ~36–50 h |
| **Inputs** | `artifacts/pathome_v1_seed/`, `BugWood_Diseases_usable.csv`, Qwen weights (HF cache) |
| **Outputs** | `results/bugwood_seed/traces/plantswarm_traces.jsonl` |
| **Resume** | Yes — already-persisted `image_id`s are skipped on resubmit. |

### Phase 3 — Enhance DB from traces (Nova)

`scripts/submit_pathome_phase3_enhance.sh`

| | |
|---|---|
| **Purpose** | Mine traces into per-class `SwarmObservations` (n_traces, avg_path_length, backtrack_rate, high_confidence_rate, confusion_targets) attached to the matching `SymptomProfile`. Canonical and regional blocks left untouched — enhancement is strictly additive. |
| **Compute** | 4 CPUs, 16 GB RAM, no GPU |
| **Walltime** | 1 h budget; ~5 min in practice |
| **Outputs** | `artifacts/pathome_v1_enhanced/{symptoms.json, refs/, enhancement_summary.json}` |

### Phase 4 — Train OBSERVE × 2 (Nova)

`scripts/submit_pathome_phase4_train.sh`

| | |
|---|---|
| **Purpose** | Train OBSERVE twice on the same trace set, differing only in which PathomeDB the agents read from at training time. Each run does Decision Transformer + GRPO. |
| **Compute** | 1× A100-80GB, 8 CPUs, 128 GB RAM |
| **Walltime** | 24 h budget; ~10–14 h DT + ~6–8 h GRPO per checkpoint, sequential |
| **Outputs** | `observe/checkpoints/{seed,enhanced}/observe_grpo_epoch_*.pt` |

### Phase 5 — Eval + before/after compare (Nova)

`scripts/submit_pathome_phase5_eval.sh`

| | |
|---|---|
| **Purpose** | Evaluate both checkpoints on full PV (with seen/unseen slice) and full PW; emit the headline before/after artefact via `compare_pathome_versions.py`. |
| **Compute** | 1× A100-80GB, 8 CPUs, 64 GB RAM |
| **Walltime** | 8 h budget; typical ~6 h |
| **Outputs** | `results/pathome_compare/{seed,enhanced}/{pv,pw}/pathome_eval.json`, `results/pathome_compare/comparison.{json,md,tex}` |

The `comparison.tex` file emits LaTeX macros (`\PathomeDeltaTthreeF`, `\PathomeDeltaTthreeECE`, `\PathomeDeltaPathLen`, …) which the paper picks up via `\input{auto_pathome_metrics}`.

---

## Configuration

Single source of truth: `configs/bugwood_pathome.yaml`. Most-tweaked knobs:

```yaml
data:
  csv_path: "BugWood_Diseases_usable.csv"
  per_class: 10              # max images per (crop, disease)
  trace_split: 7             # first N → trace seeds; remainder → references
  min_per_class: 10          # drop classes below this row count

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

---

## Consuming the KB downstream

```python
from pathome import PathomeDB

db = PathomeDB.load("artifacts/pathome_v1_seed/")

# Canonical-only context (no state)
prompt = db.symptom_context("Soybean", "Charcoal Rot")

# Canonical + this state's image-grounded deltas, ready to drop into a prompt
prompt = db.symptom_context("Soybean", "Charcoal Rot", state="Alabama")
```

`SymptomProfile.context_for_state()` is the supported entry point; agents/scripts should not reach into the dataclass fields directly.

---

## Troubleshooting

### Phase 0 errors

| Symptom | Fix |
|---|---|
| `claude CLI not on PATH` | `curl -fsSL https://claude.ai/install.sh \| bash` then `claude auth login` |
| `claude -p timed out` | A specific source page is slow. Re-run; that source is now cached. |
| Regional pass returns empty deltas for a state | The cached image may be a thumbnail or wrong-disease photo. Inspect `smoke/.bugwood_cache/<id>.jpg`. |
| Want to re-run regional only without redoing discovery/extraction | `python -m pathome_kb --regional-only --only-crops "Soybean,Tomato" --csv ...` |

### Phase 2 / 4 / 5 — vLLM fails to boot

`logs/vllm-<JOB>.log` has the stderr. To force the HF-direct fallback (slower but memory-safe):

```bash
PLANTSWARM_MODE=hf_direct sbatch scripts/submit_pathome_phase2_traces.sh
```

### CUDA OOM mid-run (HF direct only)

The HFClient releases reserved-but-unallocated GPU memory after every generation. If you still see OOM:

1. Confirm the SLURM script exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in current scripts).
2. Drop `model.max_new_tokens` from `512` → `256` in `configs/bugwood_pathome.yaml`.
3. Drop the image cap in `utils/hf_client.py:_MAX_IMAGE_SIDE` from `1024` → `768`.
4. Last resort: switch to vLLM (paged KV cache).

### Walltime kill mid-trace-generation

Trace JSONL is appended with fsync after each trace. Already-persisted `image_id`s are skipped on resubmit; just `sbatch` again.

### Geo prior is degenerate

The coarse-fallback AEZ table maps the entire US footprint into 2 zones (TMP, STM). Set `PATHOME_AEZ_SHAPEFILE` to a real FAO GAEZ shapefile to recover ~17-zone resolution. State-level priors via `state_counts` work either way.

---

## Output directory map

After a complete two-crop run:

```
PlantSwarm/
├── BugWood_Diseases_usable.csv               (Setup)
├── bugwood_classes_report.tsv                (Setup)
│
├── artifacts/                                 [gitignored except seed]
│   ├── pathome_kb/<Crop>/                    (LOCAL, Phase 0 — per-crop audit)
│   │   ├── discovery_results.json
│   │   ├── raw_extractions.json
│   │   ├── final_registry.json              ← UNIFIED canonical + regional deltas
│   │   ├── final_registry.xlsx              ← decision-tree view
│   │   └── registry.md
│   ├── pathome_v1_seed/                      (NOVA, Phase 1)
│   ├── pathome_v1_enhanced/                  (NOVA, Phase 3)
│   └── pathome_seed/symptoms_seed.json       (production; smoke uses smoke/artifacts/)
│
├── smoke/artifacts/pathome_seed/             (smoke seed — pushed via git -f)
│   └── symptoms_seed.json
│
├── results/                                   [gitignored]
│   ├── bugwood_seed/traces/plantswarm_traces.jsonl    (NOVA, Phase 2)
│   └── pathome_compare/
│       ├── seed/{pv,pw}/pathome_eval.json    (NOVA, Phase 5)
│       ├── enhanced/{pv,pw}/pathome_eval.json (NOVA, Phase 5)
│       ├── comparison.json
│       ├── comparison.md                     ← main output
│       └── comparison.tex                    ← paper macros
│
├── observe/checkpoints/                       [gitignored]
│   ├── seed/observe_grpo_epoch_*.pt          (NOVA, Phase 4)
│   └── enhanced/observe_grpo_epoch_*.pt      (NOVA, Phase 4)
│
└── logs/pathome_*-*.{out,err}                SLURM stdout/stderr
```

---

## Sync workflow

```
   ┌──────────┐  Phase 0 push  ┌────────┐  git pull   ┌──────┐
   │  Local   │───────────────→│ GitHub │────────────→│ Nova │
   │          │                │        │             │      │
   │          │   results pull │        │ Phase 5 push│      │
   │          │←───(rsync)─────│        │←────────────│      │
   └──────────┘                └────────┘             └──────┘
```

**Phase 0 push (Local → Nova)** is what `smoke/run_phase0_full.sh` prints at the end — copy-paste those `git add -f` / `commit` / `push` lines.

**Results pull (Nova → Local):**

```bash
# Pull large artefacts via rsync (results/ is gitignored)
rsync -avz nova-login:/work/mech-ai-scratch/tirtho/PlantSwarm/results/ ./results/

# OR commit just the comparison artefacts:
ssh nova-login "cd /work/.../PlantSwarm && \
  git add -f results/pathome_compare/comparison.{json,md,tex} && \
  git commit -m 'Phase 5 results' && git push"
git pull
cat results/pathome_compare/comparison.md
```

---

## Compile the paper

```bash
cd plantswarm/latex
tectonic acl_latex.tex     # or: latexmk -pdf acl_latex.tex
```

If you've run Phase 5, `\input{auto_pathome_metrics}` near the headline table picks up the `\PathomeDelta*` macros emitted by `compare_pathome_versions.py` and the table fills in automatically.

---

## Known limitations

- **US-only data.** The Bugwood IPMNet CSV is US-only at state granularity. International deployment requires a different export with finer GPS or a separate regional KB.
- **2-zone AEZ fallback.** Coarse FAO AEZ table maps the US footprint into 2 zones; full ~17-zone resolution needs `PATHOME_AEZ_SHAPEFILE` pointing at a real GAEZ shapefile.
- **Half of classes are single-state.** ~248 of 484 admitted classes appear in only one state; the geo prior is informative on the multi-state subset only.
- **No monthly priors.** The IPMNet CSV has no capture date.
- **Phase 0 cost variance.** `claude -p` quality varies with disease prevalence in Claude's training data; rare diseases may return empty canonical fields and few/no deltas.
- **Phase 0 isn't on Nova.** The local→GitHub→Nova handoff means a fresh full Phase 0 commits a few-MB seed file (and optionally tens of MB of per-crop registry artefacts) to git history.

---

## Citations

```bibtex
@inproceedings{plantswarm2026,
  title     = {Train on the Wild: Geospatial Multi-Agent Routing for
               Cross-Crop Plant Disease Diagnosis from Ten Field Images},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```
