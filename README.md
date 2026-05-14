# PlantSwarm + PathomeDB — Canonical KB + Qwen-Swarm Regional Deltas + PathomeOOD

A three-stage pipeline that produces an image-grounded plant disease
knowledge base for the 484 Bugwood IPMNet classes, plus a full BioCAP
([Zhang et al., arXiv:2510.20095](https://arxiv.org/abs/2510.20095))
two-projector CLIP trained on Bugwood images with **KB-derived hybrid
captions** (canonical KB text + per-state regional deltas):

1. **Phase 0  — canonical KB** (LOCAL, Claude). Discovery →
   extraction → reconciliation produces a text-grounded
   `CanonicalDisease` block per (crop, disease) with URL + verbatim
   quote per field. Identical to the previous SAGE-ported Phase 0.
2. **Phase 0R — regional deltas** (24-agent real swarm). For each
   (crop, disease, state) tuple with a cached Bugwood photograph, a
   **24-specialist Qwen2.5-VL-7B swarm** runs a **2-round protocol**
   over the photograph and canonical KB. Round 1: each specialist
   asks ONE focused visual question independently. Round 2: every
   specialist reads a shared blackboard of round-1 outputs and may
   `SUPPORT` / `CHALLENGE` / `WITHDRAW` against peers (the
   stigmergy + cross-talk that makes it a real swarm). The
   `VisualDiagnosisAgent` consolidator walks a 5-step CoT from
   `DR.Arti.docx` over both rounds and emits state-specific deltas —
   additions or contradictions backed by image evidence. Deltas never
   restate canonical. Claude+WebSearch verifier validates each delta
   against extension / APS / CABI before merge.
3. **Phase PathomeOOD — KB-grounded CLIP training**
   ([BioCAP paper](https://arxiv.org/abs/2510.20095)). The KB seed +
   per-image state-specific deltas are rendered into captions by
   `plantswarm/captioning.py::build_disease_caption`. Bugwood images +
   those captions are packaged as WebDataset shards and used to train
   an OpenCLIP fork with **two visual projectors** — one aligned to the
   short label text ("Tomato Early Blight"), one to the long
   descriptive caption. Evaluation reproduces every reproducible BioCAP
   paper table on PlantVillage / PlantWild / PlantDoc + a Bugwood
   held-out retrieval bench. See [PIPELINE.md](PIPELINE.md) for the
   paper-table → PlantSwarm artifact map.

The KB-side terminal deliverable is `artifacts/pathome_kb/<crop>/final_registry.json`
(canonical text + image-grounded deltas per state). The PathomeOOD
terminal deliverable is `results/pathomeood_report.md` — a paper-style
markdown report reproducing Tables 1, 2, 3, 4, 6, 8, 13, 17, 18, 19, 20
and Figure 3 on Bugwood data.

```
  ┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
  │ Phase 0  CANONICAL  │  →    │ Phase 0R REGIONAL   │  →    │ artifacts/          │
  │ pathome_kb          │       │ plantswarm/delta_   │       │   pathome_kb/Crop/  │
  │ claude -p:          │       │ pipeline.py         │       │   final_registry    │
  │   discovery         │       │ qwen2.5-vl-7b:      │       │   .json             │
  │   extraction        │       │   24 specialists    │       │ canonical +         │
  │   reconciliation    │       │   round 1 + round 2 │       │ regional_observa-   │
  │ (NON-visual KB:     │       │   + blackboard      │       │ tions[state]        │
  │  pathogen, parts,   │       │   + consolidator    │       │   .deltas[] with    │
  │  treatments)        │       │ (visual-only)       │       │   support + status  │
  └─────────────────────┘       └─────────────────────┘       └─────────────────────┘
   text-grounded, URL +           image-grounded, one              { field,
   verbatim quote per field       delta per state-specific            canonical_says,
                                  addition / contradiction            image_shows,
                                                                      image_quote }
```

PathomeDB is one store now: `SymptomLibrary`. Each `SymptomProfile`
splits into a `CanonicalDisease` block (text-grounded) and
`regional_observations[state].deltas[]` (image-grounded). Canonical owns
the symptom slots; regional only emits state-specific additions or
contradictions — a decision-tree shape rather than parallel re-
extraction.

---

## Quickstart (Nova)

```bash
# 0. one-time GPU-host install (see "GPU host only" section at the bottom
#    of requirements.txt for the full list of CUDA-side deps)
pip install -r requirements.txt
pip install torch open_clip_torch webdataset huggingface_hub

# 1. (prereq) Phase 0R must populate regional_observations in the KB so
#    delta-based captions work for KB-covered classes.
sbatch --wait scripts/submit_phase0r_regional.sh

# 2. Build captions + WebDataset shards for every variant strategy.
#    --crop is omitted on purpose: captions span ALL Bugwood crops, with
#    KB-rich text for the 25 KB-covered classes and a minimal fallback
#    ("A field photograph of {crop} affected by {disease}.") for the rest.
for s in label_only summary_only canonical_full \
         canonical_deltas_1 canonical_deltas_3 \
         canonical_deltas_5 canonical_deltas_7; do
  python scripts/build_pathomeood_captions.py --strategy "$s"
  python scripts/build_pathomeood_shards.py \
    --captions data/bugwood_captions/all_${s}.parquet \
    --out-dir  data/wds_shards/all_${s}
done

# 3. Train the 11-variant matrix (T01..T11). Each variant warm-starts from
#    BioCLIP, trains projectors only, 50 epochs, ~30-60 min per variant
#    on one A100.
CROP=all bash scripts/submit_pathomeood_matrix.sh

# 4. Eval suite on PV/PD/PW + Bugwood held-out retrieval + few-shot.
#    Also pulls 5 off-shelf baselines (CLIP, SigLIP, FG-CLIP, BioTrove-CLIP,
#    BioCLIP, BioCLIP-2) for comparison. imageomics/biocap is intentionally
#    excluded — see scripts/fetch_baselines.py for the reasoning.
python scripts/setup_plantdoc.py
python scripts/fetch_baselines.py
bash scripts/e2e_nova.sh   # phases 5-7 (eval + table aggregation + push)
```

The master report lands at `results/pathomeood_report.md`. Skipped paper
tables (5, 7, 11, 14, 15, 16, 21) are listed there with reasons.

---

## Where each phase runs

| Phase  | Host             | What it needs                                       | Compute        |
|--------|------------------|-----------------------------------------------------|----------------|
| Setup  | LOCAL or Nova    | `BugWood_Diseases.csv`                              | CPU, < 1 min    |
| 0      | LOCAL only       | `claude` CLI (OAuth) — Nova compute blocks the flow | CPU + network  |
| 0R     | Nova (or any GPU host) | vLLM serving Qwen2.5-VL-7B on `VLLM_BASE_URL` | A100 + network |

**Handoff.** Phase 0 produces canonical-only `final_registry.json` files
under `artifacts/pathome_kb/<Crop>/`; you `git add -f` + push them, then
on the GPU host you `git pull` and run Phase 0R against the same files.
Phase 0R writes the deltas back into the same `final_registry.json`
under each disease's `regional_observations` field.

---

## Repository layout

```
PlantSwarm/
├── BugWood_Diseases.csv                  raw IPMNet export (committed)
├── BugWood_Diseases_usable.csv           filtered subset (committed; regen via Setup)
├── bugwood_classes_report.tsv            per-class candidate counts
│
├── configs/
│   └── bugwood_pathome.yaml              swarm / model knobs
│
├── pathome_kb/                           Phase 0 + Phase 0R orchestration
│   ├── pipeline.py                       per-crop orchestrator + seed merge
│   ├── internet_pipeline.py              Claude discovery → extraction → reconciliation
│   ├── regional_observation.py           per-(crop, disease, state) Qwen-swarm pass
│   ├── symptoms_adapter.py               registry → SymptomProfile JSON
│   ├── prompts/                          canonical prompts only
│   └── shared.py, utils.py, config.py
│
├── plantswarm/                           Qwen-swarm regional delta extraction
│   ├── delta_pipeline.py                 2-round real swarm: 24 specialists × 2 rounds + consolidator
│   └── latex/                            EMNLP 2026 paper sources
│
├── agents/                               5 delta-extraction agents
│   ├── base_agent.py                     shared prompt scaffolding + Blackboard +
│   │                                     CROSS_REF_ACTIONS + DELTA_USER_PROMPT_R2
│   ├── leaf_agents.py                    8 leaf specialists (lesion shape/color/
│   │                                     texture, chlorosis, necrosis, curl, vein, geometry)
│   ├── stem_agents.py                    4 stem specialists (lesion, pith, surface,
│   │                                     discoloration) — pith is the decisive SDS/BSR fork
│   ├── root_agents.py                    Root + CrownCollar
│   ├── reproductive_agents.py            Flower + Fruit
│   ├── sign_agents.py                    Sporulation (mycelium / spores / ooze)
│   ├── pattern_agents.py                 Wilting + Defoliation + SpatialPattern
│   ├── diagnostic_agents.py              ConcentricPattern + ColorPalette (color
│   │                                     encoder) + LookAlikeCoT (decision-graph)
│   │                                     + SeverityVisual
│   └── diagnosis_agent.py                VisualDiagnosisAgent — 5-step CoT consolidator
│
├── train_and_eval/                       PathomeOOD CLIP training + eval
│   ├── open_clip/                        forked openclip with TWO visual projectors
│   ├── open_clip_train/                  torchrun entry (data.py + train.py adapted
│   │                                     to 2-field shards: taxon + caption)
│   ├── evaluation/                       zero_shot_iid + retrieval_openclip + metrics
│   └── imageomics/                       naming_eval + disk + helpers (minimum subset)
│
├── pathome/                              schema for the KB
│   └── symptoms.py                       SymptomLibrary / SymptomProfile /
│                                         CanonicalDisease / RegionalObservation /
│                                         RegionalDelta / Citation
│
├── utils/
│   ├── vllm_client.py                    OpenAI-compatible vLLM client
│   ├── geo.py                            state centroid + AEZ lookup (Setup)
│   └── env.py                            .env loader
│
├── data/bugwood_loader.py                crop / disease normalisation helpers (Setup)
│
├── scripts/
│   ├── filter_bugwood_csv.py             Setup: CSV → filtered usable CSV
│   ├── ensure_state_image_cache.py       per-(crop, disease, state) image cache
│   ├── registry_to_excel.py              final_registry.json → 1-sheet xlsx
│   ├── run_phase0_local.sh               LOCAL: canonical-only Phase 0 (Claude)
│   ├── submit_pathome_setup_filter.sh    Nova: filter CSV (~30 s, CPU)
│   ├── submit_phase0r_regional.sh        Nova: vLLM + 24-agent real swarm + verifier
│   │                                     (~10–24 h prod, A100)
│   │   ----- PathomeOOD pipeline ----------------------------------------------
│   ├── build_pathomeood_captions.py      KB → per-image (taxon, caption) parquet
│   ├── build_pathomeood_shards.py        parquet → WebDataset tar shards
│   ├── pathomeood_variants.sh            T01..T11 variant matrix (canonical source)
│   ├── train_pathomeood.py               wrapper around open_clip_train.main
│   ├── submit_pathomeood_train.sh        SLURM: one variant
│   ├── submit_pathomeood_matrix.sh       SLURM: sbatch all 11 variants
│   ├── fetch_baselines.py                cache 5 off-shelf CLIP baselines
│   ├── setup_plantdoc.py                 clone PlantDoc to data/eval/PlantDoc/
│   ├── evaluate_pathomeood.py            zero-shot eval on PV/PD/PW
│   ├── evaluate_pathomeood_retrieval.py  Bugwood held-out R@k
│   ├── evaluate_pathomeood_fewshot.py    prototype-mean K-shot
│   ├── aggregate_pathomeood_tables.py    results JSONs → paper-style table .md
│   ├── e2e_local.sh / e2e_nova.sh /      umbrellas — see "End-to-end pipeline" §
│   │   e2e_visualize.sh / e2e_full.sh
│   └── build_latex_pdf.sh                paper compile helper
│
├── smoke/                                two-crop happy path (Soybean + Tomato)
│   ├── run_phase0_full.sh                LOCAL Phase 0 + (LOCAL-or-tunneled) Phase 0R
│   ├── run_phase0_local.sh               LOCAL canonical-only Phase 0
│   ├── bugwood_pathome_smoke.yaml        smaller knobs
│   ├── BugWood_Diseases_smoke.csv        2-crop subset
│   └── README.md                         smoke specifics
│
└── artifacts/                            outputs (gitignored)
    └── pathome_kb/<Crop>/                Phase 0 + 0R KB per crop
```

---

## Run the whole pipeline (step-by-step runbook)

This is the master walkthrough — a fresh user clones the repo and
follows these commands top to bottom to go from raw IPMNet CSV to a
compiled paper PDF with figures and tables filled in. If you already
have `e2e_full.sh` configured you can skip to the **One-command path**
at the end of this section.

### 0. What you need

| | LOCAL | GPU host (Nova) |
|---|---|---|
| Python 3.10+ | yes | yes |
| `git` | yes | yes |
| `claude` CLI (OAuth) | yes | no |
| `ANTHROPIC_API_KEY` | optional, ~5x faster | yes (for verifier) |
| CUDA + A100-class GPU | no | yes |
| vLLM + Qwen2.5-VL-7B | no | yes |
| `latexmk` or `pdflatex` | yes (for paper) | no |

### 1. LOCAL — clone + install + auth Claude

```bash
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install matplotlib                     # for visualization PNGs

# Claude CLI for Phase 0 canonical KB + Phase 0R verifier
curl -fsSL https://claude.ai/install.sh | bash
claude auth login                           # OAuth in browser
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env  # optional but recommended
```

Place the raw IPMNet export at `BugWood_Diseases.csv` (already in the
repo; pull a fresh export from Bugwood if needed).

### 2. LOCAL — Phase 0 canonical KB (Claude)

```bash
# Filter the CSV + top up the per-(crop, disease, state) image cache
# + run Claude discovery / extraction / reconciliation + git push.
bash scripts/e2e_local.sh
```

What this does:
1. `scripts/filter_bugwood_csv.py` → `BugWood_Diseases_usable.csv` (484 classes at threshold ≥ 10).
2. `scripts/setup_image_cache.sh` → `.bugwood_cache/<image>.jpg`.
3. `scripts/run_phase0_local.sh` → `artifacts/pathome_kb/<Crop>/final_registry.json` per crop (canonical only).
4. `git add -f` canonical artefacts + `git commit` + `git push origin main`.

**Smoke variant** (2 crops, ~30 min, ~$2–5 in API quota):
```bash
SMOKE_CROPS="Soybean,Tomato" bash smoke/run_phase0_local.sh
git add -f artifacts/pathome_kb/{Soybean,Tomato}/final_registry.json
git commit -m "smoke: phase 0 canonical" && git push
```

### 3. GPU host (Nova) — one-time install

```bash
ssh you@hpc-login.iastate.edu
cd /work/<your-scratch>/
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm
mkdir -p logs

module load python cuda/11.8
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install "vllm>=0.4.0" "transformers>=4.40.0" "torch>=2.1.0" \
            "peft>=0.4.0" "accelerate>=0.30.0"
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # for the verifier
```

### 4. Nova — Phase 0R + BioCAP matrix + paper-table reproduction

```bash
# back on Nova
cd /work/<your-scratch>/PlantSwarm
git pull origin main

# Run the full GPU-host pipeline (8 phases chained with sbatch --wait).
bash scripts/e2e_nova.sh
```

What this does (see `scripts/e2e_nova.sh` for full source):
1. `git pull` — fetches canonical artefacts you pushed in step 2.
2. `sbatch --wait scripts/submit_phase0r_regional.sh` — Phase 0R: vLLM
   boot, 24-agent Qwen swarm (2 rounds + blackboard + consolidator), K-of-N agreement
   filter, Claude+WebSearch verifier, conservative merge. Updates
   `final_registry.json` with `regional_observations.<state>.deltas[]`.
3. **BioCAP captions + shards** — for each unique strategy in the
   variant matrix (`scripts/pathomeood_variants.sh`), runs
   `scripts/build_pathomeood_captions.py` + `scripts/build_pathomeood_shards.py`.
4. **BioCAP training matrix** — 11 variants (T01–T11) covering caption
   ablation (Table 3), #-deltas ablation (Table 6), covered/non-covered
   split (Table 4), and projector / epoch ablation (Fig 3). Each variant
   sbatches a separate ViT-B/16 dual-projector run from OpenAI init.
5. **Baseline cache** — `scripts/fetch_baselines.py` warms the HF hub
   cache for the 7 off-shelf models in Tables 1, 2, 17, 18, 19, 20.
6. **Eval suite** — for every variant + baseline, runs
   `evaluate_pathomeood.py` (zero-shot PV/PW/PlantDoc), `evaluate_pathomeood_retrieval.py`
   (Bugwood held-out R@k), `evaluate_pathomeood_fewshot.py` (1- and 5-shot).
7. **Aggregate paper tables** — `scripts/aggregate_pathomeood_tables.py`
   walks `results/pathomeood_eval/<run_id>/*.json` and writes 11 paper-table
   markdown files under `results/tables/` plus a master `results/pathomeood_report.md`.
8. `git add -f` + `git commit` + `git push origin main`.

**Smoke variant** (2 crops):
```bash
PATHOME_ONLY_CROPS="Soybean,Tomato" \
  PATHOME_SEED_QUICK=1 \
  PATHOME_USABLE_CSV=smoke/BugWood_Diseases_smoke_usable.csv \
  bash scripts/e2e_nova.sh
```

Walltime: smoke ~30–60 min, production ~16–30 h on one A100.

### 5. LOCAL — pull results + visualize + build PDF

```bash
# back on your LOCAL machine
cd /path/to/PlantSwarm
bash scripts/e2e_visualize.sh
```

What this does:
1. `git pull origin main` (gets the results Nova pushed in step 4).
2. `scripts/viz_kb.sh` → KB summary figures + `auto_kb_stats.tex`.
3. `scripts/viz_traces.sh` → trace stats + `auto_trace_stats.tex`.
4. `scripts/aggregate_pathomeood_tables.py` → BioCAP paper-table reproduction at `results/pathomeood_report.md` + per-table markdown under `results/tables/`.
5. `scripts/build_latex_pdf.sh` → `plantswarm/latex/acl_latex.pdf` with auto-generated tables and figures included.

Open the result:
```bash
open plantswarm/latex/acl_latex.pdf            # macOS
xdg-open plantswarm/latex/acl_latex.pdf        # Linux
```

### 6. One-command path

If you've set `PATHOME_NOVA_HOST` and `PATHOME_NOVA_REPO` and your
local machine has key-based SSH to Nova, run the whole pipeline in
one shot:

```bash
export PATHOME_NOVA_HOST=tirtho@hpc-login.iastate.edu
export PATHOME_NOVA_REPO=/work/mech-ai-scratch/tirtho/PlantSwarm
bash scripts/e2e_full.sh
```

`e2e_full.sh` runs `e2e_local.sh`, then SSHs to Nova and runs
`e2e_nova.sh` there, then comes back and runs `e2e_visualize.sh`.

### 7. Re-running only some phases

Every umbrella respects skip-knobs:

```bash
PATHOME_SKIP_NOVA=1        bash scripts/e2e_full.sh        # local-only
PATHOME_SKIP_VIZ=1         bash scripts/e2e_full.sh        # no paper rebuild
PATHOME_SKIP_PUSH=1        bash scripts/e2e_local.sh       # commit but no push
PATHOME_SKIP_PHASE0R=1     bash scripts/e2e_nova.sh        # captions + train + eval only
PATHOME_SKIP_CAPTIONS=1    bash scripts/e2e_nova.sh        # skip caption + shard rebuild
PATHOME_SKIP_TRAIN=1       bash scripts/e2e_nova.sh        # eval only
PATHOME_SKIP_EVAL=1        bash scripts/e2e_nova.sh        # train but no eval
PATHOME_SKIP_BASELINES=1   bash scripts/e2e_nova.sh        # no off-shelf baseline cache fill
PATHOME_USE_VERIFIER=0     bash scripts/e2e_nova.sh        # skip web verifier in Phase 0R
PATHOME_SKIP_PDF=1         bash scripts/e2e_visualize.sh   # figures only
```

### 8. Where the outputs end up

```
artifacts/pathome_kb/<Crop>/final_registry.json        canonical + regional KB (terminal KB deliverable)
data/bugwood_captions/<crop>_<strategy>.parquet        per-image (taxon, caption) text rows
data/wds_shards/<crop>_<strategy>/{train,val,holdout}/ WebDataset .tar shards for BioCAP training
train_and_eval/checkpoints/<VARIANT>/                   one trained ViT-B/16 per variant (T01..T11)
results/pathomeood_eval/<run_id>/{plantvillage,plantwild,plantdoc,retrieval,fewshot_*}.json
                                                       raw per-eval JSON for every (variant, dataset) cell
results/tables/table_{01,02,03,04,06,08,13,17,18,19,20}.md
                                                       paper-table markdown (one file per reproduced table)
results/tables/figure_03.md                            recipe-ablation bar table (Figure 3)
results/pathomeood_report.md                               master report — paper-style summary
results/figures/*.png                                  KB + trace figures for the paper
plantswarm/latex/auto_*.tex                            LaTeX snippets the paper \input{}s
plantswarm/latex/acl_latex.pdf                         compiled paper PDF
```

---

## One-time prerequisites

### LOCAL

```bash
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Phase 0 needs the Claude CLI:

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude auth login                                  # OAuth in browser
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env         # optional, faster
```

### GPU host (Nova or similar)

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/
git clone https://github.com/tirtho149/PlantSwarm.git    # first time
cd PlantSwarm
mkdir -p logs

module load python cuda/11.8
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install vLLM and Qwen prerequisites on the GPU host:
pip install "vllm>=0.4.0" "transformers>=4.40.0" "torch>=2.1.0"
```

---

## Two-crop smoke run

```bash
# === LOCAL ===
# Canonical KB only (Claude). No vLLM needed.
bash smoke/run_phase0_local.sh

# Or canonical + regional in one shot (needs a reachable vLLM endpoint).
# On a Mac, point at a remote vLLM via SSH tunnel:
#   ssh -L 8000:localhost:8000 nova-login
bash smoke/run_phase0_full.sh
```

After Phase 0:

```
artifacts/pathome_kb/<Crop>/
  ├── discovery_results.json            URL cache
  ├── raw_extractions.json              per-source quotes
  ├── final_registry.json               canonical + (after Phase 0R) regional deltas
  ├── final_registry.xlsx               1-sheet decision-tree view
  └── registry.md                       human-readable canonical summary
```

For canonical-only smoke (no GPU needed), push the canonical artefacts and
run Phase 0R on Nova:

```bash
git add -f smoke/BugWood_Diseases_smoke_usable.csv \
           artifacts/pathome_kb/{Soybean,Tomato}/{discovery_results,final_registry}.json
git commit -m "Phase 0: canonical KB (smoke)"
git push origin main

# Then on Nova:
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main
PATHOME_ONLY_CROPS="Soybean,Tomato" PATHOME_SEED_QUICK=1 \
  sbatch scripts/submit_phase0r_regional.sh
tail -f logs/pathome_phase0r-*.out
```

The submitter boots vLLM in the same job, waits for `/v1/models` to
respond, then runs `python -m pathome_kb --regional-only` against the
cached canonical registries. The 24-agent 2-round swarm writes the
regional deltas back into `artifacts/pathome_kb/<Crop>/final_registry.json`
under each disease's `regional_observations[<state>]` field.

---

## Production run (484 classes, ~10K Bugwood images)

```bash
# === LOCAL (Phase 0 only, ~16–24 h, ~$60–180 in Anthropic API spend) ===
python -m pathome_kb \
  --csv BugWood_Diseases_usable.csv
# Writes canonical-only artifacts/pathome_kb/<Crop>/final_registry.json
# files. The KB is per-crop, no merged seed file needed.

git add -f artifacts/pathome_kb/*/final_registry.json
git commit -m "Phase 0 canonical (484 classes)"
git push origin main

# === Nova (Phase 0R + PathomeOOD, ~10–24 h on a single A100) ===
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main

# Full e2e: Phase 0R -> captions -> shards -> 11-variant matrix ->
#           baselines -> eval -> paper-table aggregation -> push
bash scripts/e2e_nova.sh
```

`submit_phase0r_regional.sh` (part of e2e_nova.sh):
- boots `vllm.entrypoints.openai.api_server` serving
  `Qwen/Qwen2.5-VL-7B-Instruct` on `:8000`
- waits up to 10 min for `/v1/models` to respond
- runs `python -m pathome_kb --regional-only` against the cached
  canonical registries (this drives the 24-agent 2-round swarm
  through `plantswarm/delta_pipeline.run_for_state`)
- tears down vLLM on exit

Override knobs at submit time:
```bash
PATHOME_ONLY_CROPS="Soybean,Tomato,Corn"   # crop allowlist
PATHOME_SEED_QUICK=1                       # cap states per disease
PATHOME_USABLE_CSV=other.csv               # override input CSV
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct     # override served model
VLLM_SWARM_ROUNDS=2                        # real-swarm rounds (set 1 to disable)
VLLM_N_RUNS=10                             # stochastic passes per tuple
VLLM_AGREEMENT_MIN=3                       # K-of-N agreement floor
CROP=all                                   # crop tag for PathomeOOD captions/shards
PATHOME_TRACE_DIR=artifacts/swarm_traces   # set to capture per-pass traces
```

Final outputs:
- `artifacts/pathome_kb/<Crop>/final_registry.json` — KB (canonical + per-state deltas)
- `train_and_eval/checkpoints/<VARIANT>/` — one trained ViT-B/16 per variant
- `results/pathomeood_eval/<run_id>/*.json` — per-eval-cell raw JSON
- `results/tables/*.md` + `results/pathomeood_report.md` — paper-style table reproduction

---

## Step-by-step

### Phase 0 — Canonical KB (LOCAL only)

`python -m pathome_kb` or `scripts/run_phase0_local.sh`.

|                  |                                                                                          |
|------------------|------------------------------------------------------------------------------------------|
| Where it runs    | LOCAL machine                                                                            |
| Compute          | CPU; outbound HTTPS for `api.anthropic.com` + per-source page fetches                    |
| Walltime         | Smoke (2 crops): ~30 min full / ~10 min `--quick`. Production: 16–24 h                   |
| Inputs           | `BugWood_Diseases_usable.csv`, authenticated `claude` CLI, optional `ANTHROPIC_API_KEY`  |
| Outputs          | `artifacts/pathome_kb/<Crop>/{discovery_results,raw_extractions,final_registry}.json`    |
| Cost             | Smoke: ~$2–5. Production: ~$60–180.                                                      |

See [`pathome_kb/README.md`](pathome_kb/README.md) for the canonical
schema, the SAGE prompts, and a worked example.

### Phase 0R — Regional deltas via Qwen swarm

`python -m pathome_kb --regional-only` or
`scripts/submit_phase0r_regional.sh`.

|                  |                                                                                                 |
|------------------|-------------------------------------------------------------------------------------------------|
| Where it runs    | Any host with vLLM + a GPU (Nova A100 in practice)                                              |
| Compute          | 1× A100-80GB, 8 CPUs, 64 GB RAM, vLLM booted in-job                                              |
| Walltime         | Smoke (~50 tuples): ~20 min. Production (~3,000+ tuples): 6–10 h.                                |
| Inputs           | `artifacts/pathome_kb/<Crop>/final_registry.json` (canonical, from Phase 0), `BugWood_..._usable.csv`, `.bugwood_cache/` |
| Outputs          | regional deltas embedded into `final_registry.json[*].regional_observations[state]`; per-pass trace JSONL if `PATHOME_TRACE_DIR` is set  |

Inside one (crop, disease, state) call (24-agent real swarm,
2-round protocol, then K-of-N + verifier + merge):

1. **Load existing regional deltas** for THIS state from
   `final_registry.json` (`existing_deltas_for_state()`). On cold start
   this is empty; on re-runs it's whatever the previous Phase 0R
   committed. The agents see these in their context.
2. `flatten_canonical()` reduces the SAGE-shaped record to plain values.
3. **N stochastic passes** run independently (default N=10, T=0.8;
   smoke uses N=5). Each pass runs the **2-round real swarm**:

   **Round 1 — independent observation.** All 24 single-feature
   visual specialists (grouped into 7 organ families: 8 LEAF, 4 STEM,
   2 BELOW-GROUND, 2 REPRODUCTIVE, 1 PATHOGEN SIGNS, 3 WHOLE-PLANT
   PATTERNS, 4 DIAGNOSTIC CROSS-CUTTERS) run in **parallel** on
   `(image, canonical, existing KB)`. Each asks ONE laser-focused
   question (e.g. "is the lower stem pith white or brown?") and emits
   `{deltas, confidence (κ), reasoning}` for the single field it owns.
   No peer visibility yet.

   **Blackboard.** All round-1 outputs are collected into a shared
   dict keyed by `AGENT_NAME`. This is the stigmergy substrate that
   makes the swarm a real swarm (not just a parallel ensemble).

   **Round 2 — cross-talk refinement (the "real swarm" round).**
   The same 24 specialists run AGAIN in parallel, but each now sees
   the full blackboard rendered into its prompt. Each may:
     - `REFINE` its own round-1 delta given peer evidence
     - emit a `NEW` delta prompted by a peer observation
     - `SUPPORT` a peer with a `cross_ref` (raises peer confidence)
     - `CHALLENGE` a peer with a `cross_ref` (consolidator adjudicates)
     - `WITHDRAW` its own round-1 delta with a self-targeted cross_ref

   **VisualDiagnosisAgent (consolidator)** sees BOTH rounds rendered
   grouped by organ family + a cross-ref digest grouped by action,
   and walks a **5-step CoT** (1 triage which organs are visible; 2
   decisive forks from `DR.Arti.docx` — e.g. white pith → SDS, bare
   petioles → SDS, blue masses → SDS; 3 adjudicate cross_refs; 4
   dedup; 5 emit final deltas with a CoT trace).

   The consolidator's output is the pass's final delta list.
   Passes are stochastic but the agent graph is fixed.

   *Per-pass cost*: 24 (round 1) + 24 (round 2) + 1 consolidator
   = **49 vLLM calls**. Set `VLLM_SWARM_ROUNDS=1` to fall back to
   the legacy 25-call single-round mode.

4. **K-of-N cross-pass agreement filter**: deltas from the N passes are
   clustered by (`field`, Jaccard similarity over `image_shows` tokens).
   Clusters whose support covers ≥ K distinct passes (default K=3;
   smoke K=2) are kept; the rest are dropped as likely hallucination.
   K-of-N is a *proposal-confidence prior*, not a truth criterion — it
   filters one-off noise so the verifier doesn't waste API spend on
   weak candidates.
5. **Web-grounded verifier** (Claude headless + WebSearch): every
   surviving candidate is sent to `pathome_kb/verifier.py`, which
   searches extension / APS / CABI / peer-reviewed sources for
   evidence and tags each delta with a `verification_status` in
   `{verified, weakly_supported, provisional, novel_plausible,
   contradictory, duplicate_existing}` plus a `web_support` list of
   `(url, quote)` citations. Verified + provisional pass into the
   merge; contradictory ones are dropped (with audit trail);
   duplicates bump the existing entry's support instead of adding a
   row. Opt-out via `PATHOME_USE_VERIFIER=0`.
6. **Conservative merge with existing KB**:
   - Every existing delta is preserved (idempotent re-runs).
   - A new delta is added iff no existing same-field delta has Jaccard
     ≥ τ on `image_shows`.
   - When a new delta overlaps with an existing one, the existing's
     `swarm_support` counter is bumped (not duplicated), and its
     `verification_status` is upgraded if the new candidate has stronger
     external support (e.g. `unverified → verified`). The new delta's
     `web_support` citations are merged in (dedupe by URL).
   - Contradictions (same field, low-Jaccard `image_shows`) are kept as
     additional entries — downstream consumers see both and can weigh them.

The merged result is what lands in `regional_observations[state]`.
States not processed this run are preserved verbatim.

Specialist roster (24 agents, one field each):

**LEAF (8)** — `LeafLesionShape, LeafLesionColor, LeafLesionTexture,
LeafChlorosis, LeafNecrosis, LeafCurl, LeafVeinPattern, LeafGeometry`
**STEM (4)** — `StemLesion, StemPith (decisive SDS/BSR fork),
StemSurface, StemDiscoloration`
**BELOW-GROUND (2)** — `Root (cysts → SCN; blue masses → SDS),
CrownCollar`
**REPRODUCTIVE (2)** — `Flower, Fruit`
**PATHOGEN SIGNS (1)** — `Sporulation` (signs vs symptoms — mycelium,
spores, fruiting bodies, bacterial ooze, rust pustules)
**PATTERNS (3)** — `Wilting (whole/hemispheric/branch),
Defoliation (bare-petiole attachment is the SDS fork), SpatialPattern`
**DIAGNOSTIC CROSS-CUTTERS (4)** — `ConcentricPattern,
ColorPalette (color encoder), LookAlikeCoT (decision-graph CoT),
SeverityVisual`

The vLLM endpoint and swarm knobs are read from env at client-build time:

```bash
VLLM_BASE_URL       default http://localhost:8000/v1
VLLM_MODEL          default Qwen/Qwen2.5-VL-7B-Instruct
VLLM_TIMEOUT        seconds per HTTP call (default 180)
VLLM_TEMPERATURE    per-call sampling temp (default 0.8; paper §5.3 used 0.9)
VLLM_N_RUNS         stochastic passes per tuple (default 10; smoke 5)
VLLM_SWARM_ROUNDS   real-swarm rounds per pass (default 2; set to 1 for legacy single-round)
VLLM_AGREEMENT_MIN  K-of-N agreement to keep a delta (default 3; smoke 2)
VLLM_SIM_THRESHOLD  Jaccard threshold for delta clustering AND merge dedup (default 0.4)
PATHOME_USE_VERIFIER     enable Claude web-search verifier (default 1; 0 = skip)
PATHOME_VERIFIER_TIMEOUT verifier claude -p timeout in seconds (default 600)
PATHOME_VERIFIER_MAX_TURNS verifier max turns for WebSearch (default 30)
PATHOME_IMAGE_CACHE_DIR  optional override prepended to the cache search path
```

### Phase PathomeOOD — KB-grounded CLIP training on Bugwood

BioCAP ([arXiv:2510.20095](https://arxiv.org/abs/2510.20095)) is an
OpenCLIP fork that adds **two visual projectors** on top of a shared
visual encoder: one is contrastively aligned to short label text
(`"Tomato Early Blight"`), the other to long descriptive captions. We
adapt BioCAP to Bugwood by synthesising the long captions from
PathomeDB — `plantswarm/captioning.py::build_disease_caption` packs the
canonical visual_symptoms summary + diagnostic features + look-alikes +
affected parts + the top-K state-specific regional deltas into a
multi-sentence caption per image.

The variant matrix (`scripts/pathomeood_variants.sh`) trains 11 models
(T01–T11) that reproduce every reproducible BioCAP paper table on
Bugwood. See [PIPELINE.md](PIPELINE.md) for the paper-table → variant map.

|                  |                                                                                                |
|------------------|------------------------------------------------------------------------------------------------|
| Where it runs    | GPU host with CUDA (A100/H100-class)                                                           |
| Compute          | 1× A100 per variant, 8 CPUs, 64 GB RAM                                                         |
| Walltime         | ~2–4 h per variant; ~30 GPU-h for the full 11-variant matrix                                   |
| Architecture     | ViT-B/16 dual-projector (single-projector ablation in T08); openai pretrained init             |
| Inputs           | KB seed (`artifacts/pathome_kb/<crop>/final_registry.json`), `BugWood_Diseases_usable.csv`, `.bugwood_cache/` |
| Outputs          | `train_and_eval/checkpoints/<VARIANT>/` per variant, `results/pathomeood_report.md` master report  |

**Build captions + shards** (one bundle per unique caption strategy):
```bash
python scripts/build_pathomeood_captions.py --strategy canonical_deltas_3 --crop Tomato
python scripts/build_pathomeood_shards.py \
    --captions data/bugwood_captions/Tomato_canonical_deltas_3.parquet \
    --out-dir  data/wds_shards/Tomato_canonical_deltas_3
```

**Train one variant**:
```bash
# Single variant — locally with one GPU
python scripts/train_pathomeood.py --variant T04 --crop Tomato
# Or under SLURM:
VARIANT=T04 sbatch scripts/submit_pathomeood_train.sh
# All 11 variants in one matrix:
bash scripts/submit_pathomeood_matrix.sh
```

**Evaluate** (zero-shot classification + retrieval + few-shot):
```bash
python scripts/evaluate_pathomeood.py \
    --model train_and_eval/checkpoints/T04/.../epoch_50.pt \
    --pv-root /path/to/PlantVillage \
    --pw-root /path/to/PlantWild \
    --plantdoc-root data/eval/PlantDoc/test \
    --crop Tomato --out-dir results/pathomeood_eval/T04
```

**Aggregate paper tables** (after eval is done):
```bash
python scripts/aggregate_pathomeood_tables.py
# -> results/tables/{table_01,...,figure_03}.md + results/pathomeood_report.md
```

**Skipped paper tables** (and why): Tables 5, 11, 21 (human raters
needed), Table 7 (MLLM-captioner ablation — user chose KB-only path),
Table 14 (CUB localization — Bugwood has no bounding boxes),
Tables 15, 16 (format-example ablations — N/A for KB path). See the
master report for the full list.

**Tests** — `pytest tests/` covers the parser, agreement filter,
conservative merge (incl. idempotency + status upgrades),
existing-deltas extraction, the Claude verifier (mocked),
`plantswarm/captioning.py` (all 7 strategies, delta-gate hard-fail),
the shard packager, and the visualization pipeline. None of these
require torch or open_clip locally; the train/eval scripts use lazy
imports so the test suite runs on the laptop.

---

## End-to-end pipeline (one command)

The pipeline splits across LOCAL (Claude OAuth) and a GPU host (vLLM
+ Qwen). Each phase has its own dedicated shell wrapper; three umbrella
scripts chain them; one master orchestrator drives the whole loop over
SSH.

```
LOCAL    e2e_local.sh          # Setup + image cache + Phase 0 canonical
   |                           # then git push canonical artefacts
   v   git push
GitHub
   |   git pull on Nova
   v
NOVA     e2e_nova.sh           # Phase 0R + BioCAP captions + shards
   |                           # + 11-variant training matrix + baseline
   |                           # cache + eval suite + paper tables;
   |                           # via sbatch --wait, then git push results
   v   git push
GitHub
   |   git pull on LOCAL
   v
LOCAL    e2e_visualize.sh      # KB + trace viz + paper-table aggregation
                               # + (optional) PDF build
```

Run the whole loop in one shot with:

```bash
export PATHOME_NOVA_HOST=tirtho@hpc-login.iastate.edu
export PATHOME_NOVA_REPO=/work/mech-ai-scratch/tirtho/PlantSwarm
bash scripts/e2e_full.sh
```

Or run each leg manually:

```bash
# LOCAL
bash scripts/e2e_local.sh

# NOVA (ssh in, then:)
bash scripts/e2e_nova.sh

# LOCAL again
bash scripts/e2e_visualize.sh
```

Outputs land at:
- `artifacts/pathome_kb/<Crop>/final_registry.json`  — KB (canonical + per-state deltas)
- `train_and_eval/checkpoints/<VARIANT>/`            — one trained CLIP per variant (T01..T11)
- `results/pathomeood_eval/<run_id>/*.json`              — per-eval JSON for every (model, dataset) cell
- `results/tables/*.md` + `results/pathomeood_report.md` — paper-table reproduction
- `results/figures/*.png`                            — KB + trace figures
- `plantswarm/latex/auto_*.tex`                      — paper LaTeX snippets

---

## Script reference

Every phase has one dedicated `.sh`. The umbrellas in the previous
section just chain these.

### Setup

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/submit_pathome_setup_filter.sh` | Filter the raw Bugwood CSV (Nova SBATCH) | `BugWood_Diseases.csv` | `BugWood_Diseases_usable.csv`, `bugwood_classes_report.tsv` |
| `scripts/setup_image_cache.sh` | Top up `.bugwood_cache/` per (crop, disease, state) | filtered CSV | per-image JPGs in `.bugwood_cache/` |

### Phase 0 — canonical KB (LOCAL, Claude)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/run_phase0_local.sh` | Run Claude discovery + extraction + reconciliation per crop | filtered CSV, Claude CLI | `artifacts/pathome_kb/<Crop>/final_registry.json` (canonical NON-visual KB) |

### Phase 0R — regional deltas (Nova, Qwen + Claude verifier)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/submit_phase0r_regional.sh` | Nova SBATCH: boot vLLM, run 24-agent 2-round real swarm (round 1 = independent fan-out, round 2 = blackboard cross-talk with SUPPORT/CHALLENGE/WITHDRAW, then VisualDiagnosisAgent 5-step CoT consolidator), K-of-N agreement filter, Claude web-search verifier, conservative merge | canonical KB, image cache | regional deltas merged into `final_registry.json[*].regional_observations[<state>].deltas[]`; `phase0r_traces.jsonl` if `PATHOME_TRACE_DIR` set |

### BioCAP — KB-grounded CLIP training (Nova, CUDA)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/build_pathomeood_captions.py` | Build per-image (taxon, caption) rows from KB + CSV for one caption strategy | `BugWood_Diseases_usable.csv`, `artifacts/pathome_kb/<crop>/final_registry.json` | `data/bugwood_captions/<crop>_<strategy>.parquet` |
| `scripts/build_pathomeood_shards.py` | Package (image, taxon.txt, caption.txt) tuples into WebDataset .tar shards | caption parquet, `.bugwood_cache/` | `data/wds_shards/<crop>_<strategy>/{train,val,holdout}/shard-*.tar` |
| `scripts/train_pathomeood.py` | Thin wrapper that resolves a variant tag → torchrun -m open_clip_train.main with the right flags | shards, variant tag (T01..T11) | `train_and_eval/checkpoints/<VARIANT>/...` |
| `scripts/submit_pathomeood_train.sh` | SLURM submitter for ONE variant (takes VARIANT env var) | shards + variant tag | per-variant checkpoint + logs |
| `scripts/submit_pathomeood_matrix.sh` | Build captions + shards once per strategy, sbatch all 11 variants | KB + CSV + image cache | per-variant checkpoints |
| `scripts/pathomeood_variants.sh` | Canonical 11-variant matrix definition (T01..T11) | — | — |
| `scripts/fetch_baselines.py` | Pre-cache the 7 off-shelf baselines (CLIP, SigLIP, FG-CLIP, BioTrove, BioCLIP, BioCLIP-2, BioCAP-HF) | — | warmed HF hub cache |
| `scripts/evaluate_pathomeood.py` | Zero-shot classification eval on PV / PW / PlantDoc folder structures, programmatic call to BioCAP's `evaluation.zero_shot_iid` | model ckpt, eval root | `results/pathomeood_eval/<run_id>/{plantvillage,plantwild,plantdoc}.json` |
| `scripts/evaluate_pathomeood_retrieval.py` | Bugwood held-out R@k retrieval (Table 2 analog) | model + captions parquet with `split=holdout` rows | `results/pathomeood_eval/<run_id>/retrieval.json` |
| `scripts/evaluate_pathomeood_fewshot.py` | Prototype-mean K-shot eval (Tables 18, 20) | model ckpt, eval roots | `results/pathomeood_eval/<run_id>/fewshot_*.json` |
| `scripts/setup_plantdoc.py` | One-shot clone of the public PlantDoc dataset (Table 19 source) | git clone access | `data/eval/PlantDoc/{train,test}/` |
| `scripts/aggregate_pathomeood_tables.py` | Walk all `results/pathomeood_eval/*/` JSONs → paper-table markdown | per-eval JSONs | `results/tables/{table_01..figure_03}.md`, `results/pathomeood_report.md` |

### Visualization (LOCAL)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/viz_kb.sh` | KB stats: per-status pie, field-count bar, support histogram, per-state coverage | `artifacts/pathome_kb/<Crop>/final_registry.json` | `results/figures/kb_*.png`, `plantswarm/latex/auto_kb_stats.tex` |
| `scripts/viz_traces.sh` | Phase 0R trace stats: path lengths, κ-by-agent | `phase0r_traces.jsonl` | `results/figures/trace_*.png`, `plantswarm/latex/auto_trace_stats.tex` |
| `scripts/viz_all.sh` | Run KB + trace viz scripts in sequence | — | — |
| `scripts/aggregate_pathomeood_tables.py` | Paper-table markdown summary (re-runnable any time) | `results/pathomeood_eval/*/` | `results/pathomeood_report.md` |
| `scripts/build_latex_pdf.sh` | Compile the paper PDF (`acl_latex.tex`) | the `auto_*.tex` snippets above | `plantswarm/latex/acl_latex.pdf` |

### Umbrellas

| Script | Drives |
|---|---|
| `scripts/e2e_local.sh` | Setup + image cache + Phase 0 canonical + git push |
| `scripts/e2e_nova.sh` | git pull + Phase 0R + BioCAP captions+shards + 11-variant train + baselines + eval suite + paper tables + git push |
| `scripts/e2e_visualize.sh` | git pull + KB/trace viz + paper-table aggregation + (optional) PDF |
| `scripts/e2e_full.sh` | The three above, with SSH for the Nova leg |

### Skipping legs

Every umbrella respects skip-knobs so you can re-run only the parts you
need:

```bash
PATHOME_SKIP_NOVA=1 bash scripts/e2e_full.sh        # local-only smoke
PATHOME_SKIP_VIZ=1  bash scripts/e2e_full.sh        # generate data only
PATHOME_SKIP_PUSH=1 bash scripts/e2e_local.sh       # commit but no push
PATHOME_SKIP_PHASE0R=1 bash scripts/e2e_nova.sh     # train + eval only
```

---

## Consuming the KB downstream

PathomeOOD reads `final_registry.json` directly via
`plantswarm.captioning.load_kb_profiles`. Legacy consumers that want
the merged `symptoms_seed.json` shape can still produce it via
`pathome_kb.symptoms_adapter.merge_registries_to_seed(...)`, but it is
off the critical path.

```python
# Recommended: read per-crop final_registry.json directly
from plantswarm.captioning import load_kb_profiles, caption_for_row

profiles = load_kb_profiles("artifacts/pathome_kb", crop_filter=["Tomato"])
# profiles is dict[(crop, disease) -> disease_record from final_registry.json]

caption, used_kb = caption_for_row(
    crop="Tomato", disease="Early Blight", state="CA",
    profiles=profiles, strategy="canonical_deltas_3",
)
# caption is a multi-sentence text combining canonical summary,
# diagnostic features, look-alikes, and the top-3 regional deltas
# for the given state.

# Legacy path: merged seed (no longer required by PathomeOOD)
from pathome import SymptomLibrary
lib = SymptomLibrary.load("artifacts/pathome_seed/symptoms_seed.json")
prompt = lib.context_for("Soybean", "Charcoal Rot", state="Alabama")
```

`SymptomProfile.context_for_state()` is the supported entry point.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `claude CLI not on PATH` | `curl -fsSL https://claude.ai/install.sh \| bash` then `claude auth login` |
| `claude -p timed out` (Phase 0) | A specific source page is slow. Re-run; that source is now cached. |
| `vLLM endpoint not reachable` (Phase 0R) | Confirm `VLLM_BASE_URL` is correct and `curl $VLLM_BASE_URL/models` succeeds. |
| Regional pass returns empty deltas for a state | The cached image may be a thumbnail or wrong-disease photo. Inspect `smoke/.bugwood_cache/<id>.jpg`. |
| Want to re-run regional only | `python -m pathome_kb --regional-only --csv ... --out ...` |
| vLLM OOM | Lower `--max-model-len` in the submitter, or set `VLLM_MAX_NEW_TOKENS=256`. |

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
