# PlantSwarm + PathomeDB — Canonical KB + Qwen-Swarm Regional Deltas + OBSERVE student

A three-stage pipeline that produces an image-grounded plant disease
knowledge base for the 484 Bugwood IPMNet classes, plus a student model
that learns to imitate the swarm at inference:

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
3. **Phase OBSERVE — distilled student** (LoRA fine-tune of
   Qwen2.5-VL-7B). Trained on Phase 0R trace JSONL (per-step
   {state, action} pairs from the swarm). At inference, replaces the
   N-stochastic-traces swarm with a single forward pass — ~6× faster.

The terminal deliverable from the KB side is **`symptoms_seed.json`**
(canonical text + image-grounded deltas per state). The OBSERVE
checkpoint at **`observe/checkpoints/observe_best.pt`** is the
secondary deliverable — it can act as a single-pass swarm replacement
for new images.

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

Inside one (crop, disease, state) call (paper §4 / Algorithm 1, adapted,
with iterative KB evolution):

1. **Load existing regional deltas** for THIS state from
   `final_registry.json` (`existing_deltas_for_state()`). On cold start
   this is empty; on re-runs it's whatever the previous Phase 0R
   committed. The agents see these in their context.
2. `flatten_canonical()` reduces the SAGE-shaped record to plain values.
3. **N stochastic routed traces** run independently (default N=10, T=0.8;
   smoke uses N=5). Each trace is a sequential traversal:
   - Entry: `MorphologyAgent` (visual grounding).
   - Each agent emits `{deltas, confidence (κ), handoff_target, reasoning}`
     and sees three context blocks: the **canonical KB slice** for its
     owned fields, the **existing KB observations** for this state, and
     the **prior trace context** (deltas emitted earlier in this trace).
     Agents are instructed to NOT restate canonical or existing KB.
   - **Algorithm 1 routing** overrides the model's choice when:
     - κ=low + backtrack budget remaining → `MorphologyAgent` (regrounding)
     - κ=low + budget exhausted → default forward (loop guard)
     - κ=high + all 4 specialists ran → `DiagnosisAgent` (early terminate)
   - `Tmax=15` caps the path; if reached, force a terminal `DiagnosisAgent`
     call. `max_backtracks` (default 1) is honored as the actual cap.
   - Each trace ends with `DiagnosisAgent` consolidating its own context
     buffer (specialists' deltas) plus the canonical + existing-KB blocks
     into that trace's final delta list.
4. **Cross-run agreement filter**: deltas from the N traces are clustered
   by (`field`, Jaccard similarity over `image_shows` tokens). Clusters
   whose support covers ≥ K distinct runs (default K=3; smoke K=2) are
   kept; everything else is dropped as likely hallucination.
5. **Conservative merge with existing KB**:
   - Every existing delta is preserved (idempotent re-runs).
   - A new delta is added iff no existing same-field delta has Jaccard
     ≥ τ on `image_shows`.
   - When a new delta overlaps with an existing one, the existing's
     `__support__` counter is bumped (not duplicated).
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
VLLM_TMAX           max path length per trace (default 15; smoke 8)
VLLM_MAX_BACKTRACKS paper §5.3 (default 1)
VLLM_SIM_THRESHOLD  Jaccard threshold for delta clustering AND merge dedup (default 0.4)
PATHOME_IMAGE_CACHE_DIR  optional override prepended to the cache search path
```

### Phase OBSERVE — distilled student (LoRA fine-tune)

OBSERVE is a Qwen2.5-VL-7B + LoRA student that learns to imitate the
swarm. At inference, it replaces N stochastic traces with a single
forward pass.

|                  |                                                                                                |
|------------------|------------------------------------------------------------------------------------------------|
| Where it runs    | GPU host with CUDA (A100-class)                                                                |
| Compute          | 1× A100, 8 CPUs, 64 GB RAM                                                                     |
| Walltime         | ~4–8 h on Phase 0R trace JSONL (depends on N traces × tuples)                                  |
| Inputs           | `$PATHOME_TRACE_FILE` (default: `artifacts/observe_traces/phase0r_traces.jsonl`)               |
| Outputs          | `observe/checkpoints/{observe_best, observe_last}.pt`, `history.json`                          |

**Generate training data**: re-run Phase 0R with the trace writer enabled:
```bash
PATHOME_TRACE_DIR=artifacts/observe_traces \
  sbatch scripts/submit_phase0r_regional.sh
```
This appends one JSONL record per (tuple, run) to
`artifacts/observe_traces/phase0r_traces.jsonl` — every per-step
context, agent action, κ confidence, and final delta set is captured.

**Train**:
```bash
sbatch scripts/submit_observe_train.sh
# or directly:
python scripts/train_observe.py \
    --traces artifacts/observe_traces/phase0r_traces.jsonl \
    --save-dir observe/checkpoints/ \
    --epochs 5 --batch-size 4
```

Per-step supervision derived from each trace:
- `target_routing`     = next agent in `path`
- `target_backtrack`   = 1 iff the next step is `MorphologyAgent` after non-Morphology
- `target_confidence`  = κ ∈ {high, medium, low} → scalar {0.9, 0.6, 0.3}
- `target_epistemic`   = how many deltas appeared after this step vs final
- `target_aleatoric`   = 1 − κ scalar (low κ ↔ high irreducible noise)
- `target_overconfidence` = κ=high but the agent emitted 0 deltas

**Inference**: `observe.OBSERVEInference.predict(image, context)` returns
an `EpistemicAction` (next agent, backtrack, κ scalar, uncertainty,
belief text) — no swarm, no vLLM HTTP loop.

**Two-phase training** is supported:
- **Phase A** — Decision Transformer (`observe/decision_transformer.py`)
  with delta-mode reward `r_T = routing_acc * (1 - kappa_ece)`. Same BC
  loss as `OBSERVETrainer` but with return-conditioned per-trace
  weighting and early stopping on val total loss.
- **Phase B** — GRPO (`observe/grpo.py`) with reward
  `r = routing_acc * (1 - kappa_ece) - lambda_len * len(path) / Tmax`,
  clipped policy ratio update against a frozen Phase-A reference
  policy, KL-anchored.

**Evaluation** — `scripts/evaluate_observe.py` (and
`scripts/submit_evaluate_observe.sh`) loads a checkpoint and reports
routing accuracy, backtrack F1, κ MAE/ECE, and OC accuracy on the
held-out trace fold. The split is image-grouped, so no leakage across
folds.

**Tests** — `pytest tests/` covers the parser, Algorithm 1 routing
grid, agreement filter, conservative merge (incl. idempotency),
existing-deltas extraction, and the OBSERVE annotation chain. 17
tests, no GPU required for the swarm-logic tests.

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
