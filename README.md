# PlantSwarm + PathomeDB — Canonical KB + Qwen-Swarm Regional Deltas + OBSERVE OOD classifier

A three-stage pipeline that produces an image-grounded plant disease
knowledge base for the 484 Bugwood IPMNet classes, plus a small
KB-augmented classifier evaluated under heavy cross-domain shift:

1. **Phase 0  — canonical KB** (LOCAL, Claude). Discovery →
   extraction → reconciliation produces a text-grounded
   `CanonicalDisease` block per (crop, disease) with URL + verbatim
   quote per field. Identical to the previous SAGE-ported Phase 0.
2. **Phase 0R — regional deltas** (Qwen swarm). For each (crop,
   disease, state) tuple with a cached Bugwood photograph, a 5-agent
   Qwen2.5-VL-7B swarm reads the canonical KB as context, inspects the
   photograph, and emits state-specific deltas — additions or
   contradictions backed by image evidence. Deltas never restate
   canonical.
3. **Phase OBSERVE — KB-augmented OOD classifier**
   (SigLIP-2 + LoRA). Image -> SigLIP-2 vision tower (frozen base +
   LoRA q/k/v) -> embedding; class prototypes built from
   canonical + regional KB blocks (PathomeDB) are encoded once by the
   frozen SigLIP-2 text tower. Prediction is the argmax cosine
   similarity. Trained on Bugwood (field photos, Tomato by default),
   evaluated on PlantVillage (lab cutouts — easy OOD) and PlantWild
   (in-the-wild — hard OOD). Open-vocabulary: any disease with a KB
   prototype can be scored, including diseases never seen as training
   images.

The terminal deliverable from the KB side is **`symptoms_seed.json`**
(canonical text + image-grounded deltas per state). The OBSERVE
checkpoint at **`observe/checkpoints/observe_best.pt`** is the
secondary deliverable — a cheap text-conditioned image classifier
that benefits from PathomeDB's KB at zero extra training cost.

```
  ┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
  │ Phase 0  CANONICAL  │  →    │ Phase 0R REGIONAL   │  →    │   symptoms_seed.json│
  │ pathome_kb          │       │ plantswarm/delta_   │       │ canonical +         │
  │ claude -p:          │       │ pipeline.py         │       │ regional_observa-   │
  │   discovery         │       │ qwen2.5-vl-7b:      │       │ tions[state]        │
  │   extraction        │       │   4 specialists     │       │ .deltas[]           │
  │   reconciliation    │       │   + consolidator    │       │                     │
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
│   ├── delta_pipeline.py                 orchestrates 4 specialists + consolidator
│   └── latex/                            EMNLP 2026 paper sources
│
├── agents/                               5 delta-extraction agents
│   ├── base_agent.py                     shared delta-prompt scaffolding
│   ├── morphology_agent.py               lesion_morphology, affected_organs, diagnostic_features
│   ├── symptom_agent.py                  spread_pattern, diagnostic_features
│   ├── pathogen_agent.py                 look_alikes, type_of_disease
│   ├── severity_agent.py                 severity, treatments
│   └── diagnosis_agent.py                consolidator: dedupe + drop restatements
│
├── pathome/                              schema definitions for symptoms_seed.json
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
│   ├── ensure_state_image_cache.py       Phase 0R input: per-(crop, disease, state) image cache
│   ├── registry_to_excel.py              final_registry.json → 1-sheet decision-tree xlsx
│   ├── run_phase0_local.sh               LOCAL: canonical-only Phase 0
│   ├── submit_pathome_setup_filter.sh    Nova: filter CSV (~30 s, CPU)
│   ├── submit_phase0r_regional.sh        Nova: boot vLLM + run Phase 0R (~6–10 h, A100)
│   └── build_latex_pdf.sh                paper compile helper
│
├── smoke/                                two-crop happy path (Soybean + Tomato)
│   ├── run_phase0_full.sh                LOCAL Phase 0 + (LOCAL-or-tunneled) Phase 0R
│   ├── run_phase0_local.sh               LOCAL canonical-only Phase 0
│   ├── bugwood_pathome_smoke.yaml        smaller knobs
│   ├── BugWood_Diseases_smoke.csv        2-crop subset
│   └── README.md                         smoke specifics
│
└── artifacts/                            outputs (gitignored except symptoms_seed.json)
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
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           artifacts/pathome_kb/{Soybean,Tomato}/*.json
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

### 4. Nova — Phase 0R + OBSERVE train + OBSERVE eval

```bash
# back on Nova
cd /work/<your-scratch>/PlantSwarm
git pull origin main

# Run all three SBATCH-submitted Nova phases, chained via sbatch --wait.
bash scripts/e2e_nova.sh
```

What this does:
1. `git pull` (gets the canonical artefacts you pushed in step 2).
2. `sbatch --wait scripts/submit_phase0r_regional.sh` — boots vLLM, runs the Qwen swarm (N stochastic passes; 4 specialists run in parallel and DiagnosisAgent consolidates per pass), the K-of-N cross-pass agreement filter, the Claude+WebSearch verifier, and the conservative merge with existing KB. Updates `final_registry.json` and `symptoms_seed.json` with the regional deltas and their `verification_status` + `web_support` citations.
3. `sbatch --wait scripts/submit_observe_train.sh` — trains the OBSERVE classifier on Bugwood (Tomato by default). Uses the KB seed JSON for per-class text prototypes; the SigLIP-2 vision tower with LoRA on q/k/v is the only trainable part. Writes `observe/checkpoints/observe_best.pt` and `history.json`.
4. `sbatch --wait scripts/submit_evaluate_observe.sh` — runs the trained classifier on PlantVillage and/or PlantWild (set `PV_ROOT` / `PW_ROOT`). Writes `results/observe_eval.json` with per-dataset top-1, top-5, macro F1, and per-class accuracy split by KB-known vs zero-shot classes.
5. `git add -f` the results + `git commit` + `git push origin main`.

**Smoke variant** (2 crops):
```bash
PATHOME_ONLY_CROPS="Soybean,Tomato" \
  PATHOME_SEED_QUICK=1 \
  PATHOME_USABLE_CSV=smoke/BugWood_Diseases_smoke_usable.csv \
  PATHOME_SEED_FILE=smoke/artifacts/pathome_seed/symptoms_seed.json \
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
3. `scripts/viz_observe.sh` → training curves + eval bar + `auto_observe_{curves,eval}.tex`.
4. `scripts/viz_traces.sh` → trace stats + `auto_trace_stats.tex`.
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
PATHOME_SKIP_PHASE0R=1     bash scripts/e2e_nova.sh        # train + eval only
PATHOME_SKIP_TRAIN=1       bash scripts/e2e_nova.sh        # eval only
PATHOME_USE_VERIFIER=0     bash scripts/e2e_nova.sh        # skip web verifier
PATHOME_SKIP_PDF=1         bash scripts/e2e_visualize.sh   # figures only
```

### 8. Where the outputs end up

```
artifacts/pathome_kb/<Crop>/final_registry.json     canonical + regional KB
artifacts/pathome_seed/symptoms_seed.json           merged seed (terminal deliverable)
observe/checkpoints/observe_best.pt                 trained KB-augmented OOD classifier
observe/checkpoints/history.json                    training history
results/observe_eval.json                           PV / PW eval metrics
results/figures/*.png                               figures for the paper
plantswarm/latex/auto_*.tex                         LaTeX snippets the paper \input{}s
plantswarm/latex/acl_latex.pdf                      compiled paper PDF
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

smoke/artifacts/pathome_seed/symptoms_seed.json    merged seed
```

For canonical-only smoke (no GPU needed), push the canonical artefacts and
run Phase 0R on Nova:

```bash
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           smoke/BugWood_Diseases_smoke_usable.csv \
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
respond, then runs `python -m pathome_kb --regional-only`. Final
`symptoms_seed.json` lands at `smoke/artifacts/pathome_seed/`.

---

## Production run (484 classes)

```bash
# === LOCAL (Phase 0 only, ~16–24 h, ~$60–180 in Anthropic API spend) ===
python -m pathome_kb \
  --csv BugWood_Diseases_usable.csv \
  --out artifacts/pathome_seed/symptoms_seed.json

git add -f artifacts/pathome_seed/symptoms_seed.json
git add -f artifacts/pathome_kb/                    # optional audit trail
git commit -m "Phase 0 canonical (484 classes)"
git push origin main

# === Nova (Phase 0R, ~6–10 h on a single A100) ===
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
sbatch scripts/submit_phase0r_regional.sh
```

`submit_phase0r_regional.sh`:
- boots `vllm.entrypoints.openai.api_server` serving
  `Qwen/Qwen2.5-VL-7B-Instruct` on `:8000`
- waits up to 10 min for `/v1/models` to respond
- runs `python -m pathome_kb --regional-only` against the cached
  canonical registries
- tears down vLLM on exit

Override knobs at submit time:
```bash
PATHOME_ONLY_CROPS="Soybean,Tomato,Corn"   # crop allowlist
PATHOME_SEED_QUICK=1                       # cap states per disease
PATHOME_USABLE_CSV=other.csv               # override input CSV
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct     # override served model
```

Final output: `artifacts/pathome_seed/symptoms_seed.json`.

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
| Outputs          | regional deltas embedded into `final_registry.json[*].regional_observations[state]`; merged `symptoms_seed.json`  |

Inside one (crop, disease, state) call (parallel-swarm + verifier; no
routing):

1. **Load existing regional deltas** for THIS state from
   `final_registry.json` (`existing_deltas_for_state()`). On cold start
   this is empty; on re-runs it's whatever the previous Phase 0R
   committed. The agents see these in their context.
2. `flatten_canonical()` reduces the SAGE-shaped record to plain values.
3. **N stochastic passes** run independently (default N=10, T=0.8;
   smoke uses N=5). Each pass is:
   - The 4 specialists (Morphology, Symptom, Pathogen, Severity) run
     in **parallel** on `(image, canonical, existing KB)`. Each emits
     `{deltas, confidence (κ), reasoning}` for the fields it owns.
   - `DiagnosisAgent` consolidates the union of the 4 specialists'
     deltas, dropping restatements of canonical / existing KB and
     deduping overlapping fields.
   - The consolidator's output is the pass's final delta list.
   There is no routing, no κ-gated handoff, and no backtrack — passes
   are stochastic but the agent graph is fixed (parallel + consolidator).
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

Agent ownership (each agent's `OWNED_FIELDS`):
- **MorphologyAgent** — `lesion_morphology, affected_organs, diagnostic_features`
- **SymptomAgent** — `spread_pattern, diagnostic_features`
- **PathogenAgent** — `look_alikes, type_of_disease`
- **SeverityAgent** — `severity, treatments`

The vLLM endpoint and swarm knobs are read from env at client-build time:

```bash
VLLM_BASE_URL       default http://localhost:8000/v1
VLLM_MODEL          default Qwen/Qwen2.5-VL-7B-Instruct
VLLM_TIMEOUT        seconds per HTTP call (default 180)
VLLM_TEMPERATURE    per-call sampling temp (default 0.8; paper §5.3 used 0.9)
VLLM_N_RUNS         stochastic traces per tuple (default 10; smoke 5)
VLLM_AGREEMENT_MIN  K-of-N agreement to keep a delta (default 3; smoke 2)
VLLM_SIM_THRESHOLD  Jaccard threshold for delta clustering AND merge dedup (default 0.4)
PATHOME_USE_VERIFIER     enable Claude web-search verifier (default 1; 0 = skip)
PATHOME_VERIFIER_TIMEOUT verifier claude -p timeout in seconds (default 600)
PATHOME_VERIFIER_MAX_TURNS verifier max turns for WebSearch (default 30)
PATHOME_IMAGE_CACHE_DIR  optional override prepended to the cache search path
```

### Phase OBSERVE — KB-augmented OOD classifier

OBSERVE is a cheap SigLIP-2 + LoRA classifier whose class labels come
from PathomeDB text prototypes. Image -> SigLIP-2 vision tower (frozen
base + LoRA on vision q/k/v) -> embedding; class prototypes built
from canonical + regional KB blocks are encoded once by the frozen
SigLIP-2 text tower. Prediction is the argmax cosine similarity. The
classifier is trained on Bugwood (Tomato by default, ~600 field
photos, 14 classes that match the KB) and evaluated zero/few-shot on
PlantVillage and PlantWild.

|                  |                                                                                                |
|------------------|------------------------------------------------------------------------------------------------|
| Where it runs    | GPU host with CUDA (A100-class)                                                                |
| Compute          | 1× A100, 8 CPUs, 64 GB RAM                                                                     |
| Walltime         | ~30–90 min on Tomato (~600 images, 10 epochs)                                                  |
| Inputs           | `$PATHOME_SEED_JSON` (KB), `$PATHOME_BUGWOOD_CSV` (training rows), `$PATHOME_BUGWOOD_CACHE`    |
| Outputs          | `observe/checkpoints/{observe_best, observe_last}.pt`, `history.json`                          |

**Train**:
```bash
sbatch scripts/submit_observe_train.sh
# or directly:
python scripts/train_observe.py \
    --seed artifacts/pathome_seed/symptoms_seed.json \
    --bugwood-csv BugWood_Diseases_usable.csv \
    --cache-dir .bugwood_cache \
    --crop Tomato \
    --backbone google/siglip-base-patch16-224 \
    --include-healthy \
    --epochs 10 --batch-size 32 --lora-r 8
```

What the trainer does:
- Loads the seed JSON and builds one text prototype per KB profile for
  the requested crop. Each prototype packs canonical summary +
  diagnostic features + look-alikes + affected parts + top-K verified
  regional deltas into a single multi-sentence prompt for the SigLIP-2
  text tower.
- Optionally appends a synthetic `<crop>::healthy` prototype so the
  classifier can recognise non-disease leaves (PathomeDB itself
  doesn't cover healthy).
- Loads Bugwood field photos via the filtered usable CSV +
  `.bugwood_cache/<image_number>.jpg`, filters to the requested crop,
  drops rows whose disease isn't in the KB.
- Stratified train / val split (default 15% val).
- Per epoch: encode the class prototypes once with the frozen text
  tower, then train the LoRA-adapted vision tower to minimise softmax
  cross-entropy over cosine·temperature logits.
- Saves the best checkpoint by val top-1 plus full history.

**Inference**: `OBSERVEInference(ckpt).classify(image, topk=5)` returns
a `ClassificationResult` with top-k labels and softmax probabilities.

**Evaluation** — `scripts/evaluate_observe.py` (and
`scripts/submit_evaluate_observe.sh`) loads a checkpoint and runs it
against PlantVillage and/or PlantWild folder-per-class datasets:
```bash
OBSERVE_CKPT=observe/checkpoints/observe_best.pt \
  PV_ROOT=/path/to/PlantVillage \
  PW_ROOT=/path/to/PlantWild \
  sbatch scripts/submit_evaluate_observe.sh
```
The evaluator extends the trained class index with any PV/PW classes
not seen at training time, synthesising a minimal `"A field photograph
of <crop> affected by <disease>."` prototype for each. Per dataset it
reports top-1, top-5, macro F1, and per-class accuracy with a
KB-known vs zero-shot flag.

**Tests** — `pytest tests/` covers the parser, agreement filter,
conservative merge (incl. idempotency + status upgrades),
existing-deltas extraction, the Claude verifier (mocked), the
OBSERVE prototype + dataset machinery, and the visualization
pipeline. The OBSERVE tests require `torch` to be installed; the
swarm-logic tests don't.

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
NOVA     e2e_nova.sh           # Phase 0R + OBSERVE train + OBSERVE eval
   |                           # via sbatch --wait, then git push results
   v   git push
GitHub
   |   git pull on LOCAL
   v
LOCAL    e2e_visualize.sh      # all viz + paper figures + LaTeX snippets
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
- `artifacts/pathome_seed/symptoms_seed.json`  — KB
- `observe/checkpoints/observe_best.pt`        — student
- `results/observe_eval.json`                  — metrics
- `results/figures/*.png`                      — paper figures
- `plantswarm/latex/auto_*.tex`                — paper LaTeX snippets

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
| `scripts/run_phase0_local.sh` | Run Claude discovery + extraction + reconciliation per crop | filtered CSV, Claude CLI | `artifacts/pathome_kb/<Crop>/final_registry.json`, `symptoms_seed.json` |

### Phase 0R — regional deltas (Nova, Qwen + Claude verifier)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/submit_phase0r_regional.sh` | Nova SBATCH: boot vLLM, run Qwen swarm (4 specialists in parallel + DiagnosisAgent consolidator, N stochastic passes), K-of-N agreement filter, Claude web-search verifier, conservative merge | canonical KB, image cache | regional deltas merged into `final_registry.json`; `symptoms_seed.json`; `phase0r_traces.jsonl` if `PATHOME_TRACE_DIR` set |

### OBSERVE — KB-augmented OOD classifier (Nova, CUDA)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/submit_observe_train.sh` | Train SigLIP-2 + LoRA on Bugwood (Tomato by default) against KB text prototypes | `symptoms_seed.json`, filtered CSV, `.bugwood_cache/` | `observe/checkpoints/{observe_best,observe_last}.pt`, `history.json` |
| `scripts/submit_evaluate_observe.sh` | Eval on PV / PW: top-1 / top-5 / macro F1 + per-class accuracy, with KB-known vs zero-shot split | checkpoint, `PV_ROOT` and/or `PW_ROOT` | `results/observe_eval.json` |

### Visualization (LOCAL)

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scripts/viz_kb.sh` | KB stats: per-status pie, field-count bar, support histogram, per-state coverage | `symptoms_seed.json` | `results/figures/kb_*.png`, `plantswarm/latex/auto_kb_stats.tex` |
| `scripts/viz_observe.sh` | OBSERVE training loss + top-1 curves + PV / PW per-class bars | `history.json`, `observe_eval.json` | `results/figures/observe_*.png`, `plantswarm/latex/auto_observe_{curves,eval}.tex` |
| `scripts/viz_traces.sh` | Phase 0R trace stats: path lengths, κ-by-agent | `phase0r_traces.jsonl` | `results/figures/trace_*.png`, `plantswarm/latex/auto_trace_stats.tex` |
| `scripts/viz_all.sh` | Run all three viz scripts in sequence | — | — |
| `scripts/build_latex_pdf.sh` | Compile the paper PDF (`acl_latex.tex`) | the `auto_*.tex` snippets above | `plantswarm/latex/acl_latex.pdf` |

### Umbrellas

| Script | Drives |
|---|---|
| `scripts/e2e_local.sh` | Setup + image cache + Phase 0 canonical + git push |
| `scripts/e2e_nova.sh` | git pull + Phase 0R + OBSERVE train + OBSERVE eval + git push |
| `scripts/e2e_visualize.sh` | git pull + all viz + paper PDF |
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

## Consuming the seed JSON downstream

```python
from pathome import SymptomLibrary

lib = SymptomLibrary.load("artifacts/pathome_seed/symptoms_seed.json")

# Canonical-only context (no state)
prompt = lib.context_for("Soybean", "Charcoal Rot")

# Canonical + this state's image-grounded deltas
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
