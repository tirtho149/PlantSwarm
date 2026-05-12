# PlantSwarm — End-to-End Flow

Submission-ready overview of the current pipeline. All flowcharts are
Mermaid (renderable on GitHub, mermaid.live, and any standard Markdown
viewer); data shapes are ASCII for stable layout.

Sections
1. [Top-level pipeline](#1-top-level-pipeline)
2. [Phase 0 — canonical KB (Claude)](#2-phase-0--canonical-kb-claude)
3. [Phase 0R — regional deltas (Qwen swarm)](#3-phase-0r--regional-deltas-qwen-swarm)
   - 3a. [Per-tuple flow (iterative KB loop)](#3a-per-tuple-flow-iterative-kb-loop)
   - 3b. [Inside one trace (routed swarm)](#3b-inside-one-trace-routed-swarm)
   - 3c. [Algorithm 1 routing decision](#3c-algorithm-1-routing-decision)
   - 3d. [Cross-run K-of-N agreement filter](#3d-cross-run-k-of-n-agreement-filter)
   - 3e. [Conservative merge with existing KB](#3e-conservative-merge-with-existing-kb)
4. [Phase OBSERVE — distilled student](#4-phase-observe--distilled-student)
5. [Data shape evolution](#5-data-shape-evolution)
6. [File map](#6-file-map)
7. [Env var reference](#7-env-var-reference)
8. [Run-report line](#8-run-report-line)

---

## 1. Top-level pipeline

LOCAL machine → GitHub → GPU host. Three terminal deliverables.

```mermaid
flowchart TD
    SETUP[Setup<br/>filter_bugwood_csv.py<br/>raw CSV to filtered CSV]
    CACHE[Image cache<br/>ensure_state_image_cache.py<br/>per crop disease state photo]
    P0[Phase 0 - canonical KB<br/>pathome_kb via Claude<br/>discovery to extraction to reconciliation]
    PUSH([git push canonical artifacts])
    PULL([git pull on GPU host])
    VLLM[vLLM serves Qwen2.5-VL-7B-Instruct<br/>booted in-job]
    P0R[Phase 0R - regional deltas<br/>Qwen swarm<br/>load existing then N traces then agreement then merge]
    ADAPT[Adapter merge<br/>symptoms_adapter.py]
    SEED([symptoms_seed.json<br/>KB deliverable])
    TRACES[(phase0r_traces.jsonl<br/>per-trace records<br/>written when PATHOME_TRACE_DIR is set)]
    TRAIN[Train OBSERVE<br/>Qwen2.5-VL-7B plus LoRA<br/>per-step behavioral cloning]
    CKPT([observe_best.pt<br/>model deliverable])

    SETUP --> CACHE --> P0
    P0 --> PUSH --> PULL --> P0R
    VLLM -.serves.-> P0R --> ADAPT --> SEED
    P0R -.persists.-> TRACES
    TRACES --> TRAIN --> CKPT

    classDef local fill:#dff,stroke:#066,stroke-width:1px
    classDef gpu fill:#fde,stroke:#a06,stroke-width:1px
    classDef student fill:#eef,stroke:#33a,stroke-width:1px
    classDef terminal fill:#efe,stroke:#060,stroke-width:2px
    class SETUP,CACHE,P0 local
    class VLLM,P0R,ADAPT gpu
    class TRACES,TRAIN student
    class SEED,CKPT terminal
```

| Stage | Host | Compute | Walltime |
|---|---|---|---|
| Setup | LOCAL or Nova | CPU, &lt; 1 min | trivial |
| Image cache | LOCAL or Nova | network only | smoke ~2 min |
| Phase 0 (Claude) | LOCAL only (OAuth) | CPU + Anthropic API | smoke ~30 min / prod 16-24 h |
| Phase 0R (Qwen) | GPU host with vLLM | 1x A100-80GB | smoke ~20-40 min / prod 10-20 h |
| Adapter merge | Same as Phase 0R | CPU, seconds | trivial |
| Phase OBSERVE (train) | GPU host with CUDA | 1x A100 | ~4-8 h on Phase 0R traces |

---

## 2. Phase 0 — canonical KB (Claude)

Run via `python -m pathome_kb`. Three Claude-driven stages per crop, all
text-grounded (URL + verbatim quote per field). No images touched here.

```mermaid
flowchart LR
    CSV[(BugWood_Diseases_usable.csv)]
    D[Stage 1 Discovery<br/>internet_pipeline.py<br/>claude -p with WebSearch<br/>per-disease URL list]
    E[Stage 2 Extraction<br/>internet_pipeline.py<br/>claude -p per URL<br/>verbatim quotes plus treatments]
    R[Stage 3 Reconciliation<br/>internet_pipeline.py<br/>Anthropic SDK or claude -p<br/>per-field merge with citations]
    REG[(final_registry.json<br/>canonical only<br/>per crop)]

    CSV --> D --> E --> R --> REG

    classDef claude fill:#fef,stroke:#606,stroke-width:1px
    classDef file fill:#ffd,stroke:#660,stroke-width:1px
    class D,E,R claude
    class CSV,REG file
```

Output shape (one disease entry):

```jsonc
{
  "disease_name": "Charcoal Rot",
  "pathogen_scientific_name": {
    "value": "Macrophomina phaseolina",
    "url":   "https://extension.umn.edu/.../charcoal-rot-soybean",
    "quote": "Charcoal rot is caused by the soilborne fungus..."
  },
  "type_of_disease":  { "value": "Fungal",  "url": "...", "quote": "..." },
  "affected_parts":   { "value": ["Foliar","Stem","Root","Pod"], "url": "...", "quote": "..." },
  "visual_symptoms": {
    "summary":             { "value": "...", "url": "...", "quote": "..." },
    "diagnostic_features": { "value": "...", "url": "...", "quote": "..." },
    "look_alikes":         { "value": [], "url": "", "quote": "" }
  },
  "treatments":         { "value": [], "url": "...", "quote": "..." },
  "regional_observations": {}
}
```

---

## 3. Phase 0R — regional deltas (Qwen swarm)

Run via `python -m pathome_kb --regional-only`. The orchestrator is
`plantswarm.delta_pipeline.run_for_state`, called once per
(crop, disease, state, cached image) tuple.

### 3a. Per-tuple flow (iterative KB loop)

```mermaid
flowchart TD
    INPUT[crop, disease, state, cached image]
    LOAD[Load existing KB<br/>existing_deltas_for_state<br/>empty on cold start]
    FLAT[flatten_canonical<br/>plain values for prompt]
    URL[load_image_data_url<br/>MIME detected from extension]
    T1[Trace 1<br/>seed=42]
    T2[Trace 2<br/>seed=142]
    TN[Trace N<br/>seed=42+N*100]
    AGR[Cross-run agreement filter<br/>cluster by field plus Jaccard ge tau<br/>keep clusters covering ge K distinct runs]
    MERGE[Conservative merge with existing<br/>existing preserved<br/>new added if no Jaccard overlap<br/>overlap bumps existing support]
    OUT[Merged record<br/>state, deltas, image_ids, swarm_meta]

    INPUT --> LOAD
    INPUT --> FLAT
    INPUT --> URL
    LOAD -.context.-> T1
    LOAD -.context.-> T2
    LOAD -.context.-> TN
    FLAT -.context.-> T1
    FLAT -.context.-> T2
    FLAT -.context.-> TN
    URL -.image.-> T1
    URL -.image.-> T2
    URL -.image.-> TN
    T1 --> AGR
    T2 --> AGR
    TN --> AGR
    LOAD --> MERGE
    AGR --> MERGE
    MERGE --> OUT

    classDef input fill:#ffd,stroke:#660
    classDef ctx fill:#dff,stroke:#066
    classDef trace fill:#fde,stroke:#a06
    classDef agg fill:#eef,stroke:#33a
    classDef out fill:#efe,stroke:#060,stroke-width:2px
    class INPUT input
    class LOAD,FLAT,URL ctx
    class T1,T2,TN trace
    class AGR,MERGE agg
    class OUT out
```

After every tuple finishes, `_embed_into_registry` merges its per-state
record back into the disease's `regional_observations` dict — **states
not processed this run are preserved verbatim**.

### 3b. Inside one trace (routed swarm)

Each of the N traces is a sequential traversal of the 5-agent graph,
starting at MorphologyAgent. Each agent emits
`{deltas, confidence (κ), handoff_target, reasoning}` and sees the
canonical slice plus existing KB plus prior trace context as input.

```mermaid
flowchart TD
    ENTRY([entry: MorphologyAgent])
    MA[MorphologyAgent<br/>owned: lesion_morphology<br/>affected_organs<br/>diagnostic_features]
    SA[SymptomAgent<br/>owned: spread_pattern<br/>diagnostic_features]
    PA[PathogenAgent<br/>owned: look_alikes<br/>type_of_disease]
    SV[SeverityAgent<br/>owned: severity<br/>treatments]
    DA[DiagnosisAgent<br/>consolidator: dedupe<br/>drop restatements]
    DONE([per-trace final deltas])

    ENTRY --> MA
    MA --> ROUTE[Algorithm 1<br/>kappa-gated]
    SA --> ROUTE
    PA --> ROUTE
    SV --> ROUTE
    ROUTE -->|backtrack| MA
    ROUTE -->|forward to specialist| SA
    ROUTE -->|forward to specialist| PA
    ROUTE -->|forward to specialist| SV
    ROUTE -->|terminate| DA
    DA --> DONE

    classDef agent fill:#fde,stroke:#a06
    classDef router fill:#eef,stroke:#33a
    classDef done fill:#efe,stroke:#060,stroke-width:2px
    class MA,SA,PA,SV,DA agent
    class ROUTE router
    class DONE done
```

### 3c. Algorithm 1 routing decision

Four rules applied in order. Returns the next agent to call, or
DiagnosisAgent to terminate.

```mermaid
flowchart TD
    CALL[Agent A emits deltas, kappa, model_handoff, reasoning]
    R1{kappa = low<br/>AND backtrack_count &lt; max_backtracks<br/>AND A not MorphologyAgent}
    R2{kappa = low<br/>AND backtrack_count &ge; max_backtracks}
    R3{kappa = high<br/>AND all 4 specialists ran<br/>AND A not DiagnosisAgent}
    R4{model_handoff set}
    BACK[Next = MorphologyAgent<br/>regrounding]
    FWD[Next = default_forward<br/>loop guard]
    TERM[Next = DiagnosisAgent<br/>early terminate]
    MODEL[Next = model_handoff]
    DEF[Next = default_forward]

    CALL --> R1
    R1 -->|yes| BACK
    R1 -->|no| R2
    R2 -->|yes| FWD
    R2 -->|no| R3
    R3 -->|yes| TERM
    R3 -->|no| R4
    R4 -->|yes| MODEL
    R4 -->|no| DEF

    classDef decision fill:#fef,stroke:#606
    classDef action fill:#eef,stroke:#33a
    class R1,R2,R3,R4 decision
    class BACK,FWD,TERM,MODEL,DEF action
```

Per-agent routing menus:

| Agent | DEFAULT_FORWARD | HANDOFF_MENU (model may pick from) |
|---|---|---|
| MorphologyAgent | SymptomAgent | Symptom, Severity, Diagnosis |
| SymptomAgent | PathogenAgent | Morphology, Pathogen, Severity, Diagnosis |
| PathogenAgent | SeverityAgent | Morphology, Symptom, Severity, Diagnosis |
| SeverityAgent | DiagnosisAgent | Morphology, Diagnosis |

### 3d. Cross-run K-of-N agreement filter

After all N traces complete, per-trace final-delta lists are pooled,
grouped by field, and clustered greedily on `image_shows` Jaccard. Only
clusters covering at least K distinct run-indices survive.

```
Trace 0 final_deltas    [d_00, d_01]
Trace 1 final_deltas    [d_10]
Trace 2 final_deltas    [d_20, d_21, d_22]
                ...
Trace N-1 final_deltas  [...]
                  |
                  |  group by field
                  v
       +--------------------------+
       | lesion_morphology:       |
       |   (0, d_00) (2, d_20)    |
       |   (5, d_50)              |
       | severity:                |
       |   (0, d_01) (1, d_10)    |
       |   ...                    |
       +-------------+------------+
                     |
                     |  greedy Jaccard cluster within each field
                     v
       +-------------------------------------------+
       | lesion_morphology Cluster A:              |
       |   (0, "pustular lesions w/ halos")        |
       |   (2, "halos around pustules")            |
       |   (5, "pustules surrounded by yellow")    |
       |   distinct_runs = {0, 2, 5}               |
       |   support = 3                             |   keep (>= K)
       |                                           |
       | severity Cluster B:                       |
       |   (0, "carrot-shaped fronds")             |
       |   distinct_runs = {0}                     |
       |   support = 1                             |   drop  (< K)
       +-------------------------------------------+
                     |
                     v
       candidates (K-of-N survivors), each tagged
       with __support__ and __cluster_size__
```

### 3e. Conservative merge with existing KB

Candidates from agreement are merged into the **existing** regional
deltas for this state. Existing is never wiped.

```
existing  = [E0 (field=L, support=5),
             E1 (field=S, support=3)]
candidates = [C0 (field=L, image_shows close to E0: Jaccard >= tau),
              C1 (field=P, image_shows, no existing in field P),
              C2 (field=S, image_shows, contradicts E1: Jaccard < tau)]
                |
                |  for each candidate C:
                |    if exists E with same field AND Jaccard >= tau:
                |        E.support += C.support
                |        drop C
                |    else:
                |        append C (support default 1)
                v
merged = [E0 (support = 5 + C0.support = 8),
          E1 (support = 3),
          C1 (support = 1),
          C2 (support = 1)]

counts = {n_existing: 2, n_new_candidates: 3,
          n_added: 2, n_overlaps_bumped: 1}
```

Properties:
- **Idempotent on shape**: re-running with the same candidates against
  the same existing list adds no entries; only bumps support.
- **Existing always preserved**: prior Phase 0R deltas are never
  overwritten.
- **Contradictions kept**: low-Jaccard same-field deltas are added as
  separate entries; downstream consumers see all observations.

---

## 4. Phase OBSERVE — distilled student

Trained on Phase 0R trace JSONL. At inference, replaces the
N-stochastic-traces swarm with a single forward pass.

```mermaid
flowchart TD
    P0R[Phase 0R run<br/>with PATHOME_TRACE_DIR set]
    JSONL[(phase0r_traces.jsonl<br/>one line per tuple-run<br/>profile_id, path, decisions<br/>context_buffer, final_deltas<br/>existing_kb_at_start)]
    EXPAND[load_phase0r_traces<br/>expand per-step<br/>TraceStepAnnotation]
    SPLIT[split_annotations<br/>group by image_path<br/>train / val / held]
    MODEL[OBSERVE model<br/>Qwen2.5-VL-7B plus LoRA r=16<br/>heads: routing 5-class, backtrack,<br/>epsilon, alpha, c, OC]
    TRAIN[OBSERVETrainer<br/>multi-task loss<br/>L = L_rt + 0.4 L_cal<br/>+ 0.2 L_cons + 0.3 L_OC]
    CKPT[(observe_best.pt)]
    INFER[OBSERVEInference predict<br/>EpistemicAction:<br/>next_agent, backtrack,<br/>kappa, uncertainty, belief]

    P0R --> JSONL --> EXPAND --> SPLIT --> TRAIN
    MODEL --> TRAIN --> CKPT
    CKPT --> INFER

    classDef src fill:#fde,stroke:#a06
    classDef stage fill:#eef,stroke:#33a
    classDef model fill:#efd,stroke:#060
    classDef deliv fill:#efe,stroke:#060,stroke-width:2px
    class P0R,JSONL src
    class EXPAND,SPLIT,TRAIN stage
    class MODEL model
    class CKPT,INFER deliv
```

Per-step supervision derived from each trace's context buffer:

| Target | Source |
|---|---|
| target_routing | path[i+1] — which agent the swarm called next |
| target_backtrack | 1 iff path[i+1] is MorphologyAgent AND path[i] is not MorphologyAgent |
| target_confidence | kappa in {high, medium, low} mapped to {0.9, 0.6, 0.3} |
| target_epistemic | (n_final - n_at_step) / max(1, n_final) |
| target_aleatoric | 1 - kappa_scalar |
| target_overconfidence | 1 iff kappa is high AND len(deltas at step) == 0 |
| target_belief | reasoning string the agent emitted |

**NOTE.** Decision Transformer (`observe/decision_transformer.py`) and
GRPO (`observe/grpo.py`) are restored from the paper but **not yet
ported** to delta-mode reward — the reward signal
`delta_set_F1 + (1 - ECE)` needs wiring. The behavioral-cloning trainer
(`OBSERVETrainer`) is the v1 path.

---

## 5. Data shape evolution

What lives where, and what gets preserved between layers.

```
                  artifacts/pathome_kb/<Crop>/final_registry.json
                  +-------------------------------------------+
   Phase 0    >   | {                                         |
                  |   "crop": "Soybean",                      |
                  |   "diseases": [{                          |
                  |     "disease_name": "Charcoal Rot",       |
                  |     "pathogen_scientific_name": {...},    |
                  |     "visual_symptoms": {...},             |
                  |     "treatments": {...},                  |
   Phase 0R   >   |     "regional_observations": {            |
                  |       "Alabama": {                        |
                  |         "state": "Alabama",               |
                  |         "image_ids": [...],               |
                  |         "deltas": [                       |
                  |           { field, canonical_says,        |
                  |             image_shows, image_quote,     |
                  |             image_id,                     |
                  |             __support__,                  |
                  |             __cluster_size__ }, ...       |
                  |         ],                                |
                  |         "__swarm_meta__": {...}           |
                  |       }, ...                              |
                  |     }                                     |
                  |   }, ...]                                 |
                  | }                                         |
                  +-------------------------------------------+
                                  |
                                  v   symptoms_adapter.py
                                  |
                      artifacts/pathome_seed/symptoms_seed.json
                      +-------------------------------------------+
                      | {                                         |
                      |   "min_observations": 3,                  |
                      |   "profiles": [{                          |
                      |     "profile_id": "Soybean::Charcoal Rot",|
                      |     "crop": "Soybean",                    |
                      |     "disease": "Charcoal Rot",            |
                      |     "canonical": {...},                   |
                      |     "regional_observations": {            |
                      |       "Alabama": {                        |
                      |         state, image_ids,                 |
                      |         deltas: [{                        |
                      |           field, canonical_says,          |
                      |           image_shows, image_quote,       |
                      |           image_id,                       |
                      |           support,         <- __support__ |
                      |           cluster_size                    |
                      |         }],                               |
                      |         swarm_meta: {...}  <- __swarm__   |
                      |       }                                   |
                      |     },                                    |
                      |     state_counts, aez_counts,             |
                      |     reference_ids                         |
                      |   }, ...]                                 |
                      | }                                         |
                      +-------------------------------------------+
                                  |
                                  v   pathome.SymptomLibrary.load()
                                  |
                                consumers
```

The adapter strips the `__` prefix from telemetry keys but preserves
the content — consumers see `support`, `cluster_size`, `swarm_meta` as
clean keys.

When `PATHOME_TRACE_DIR` is set, Phase 0R also writes per-trace records
for OBSERVE training:

```
              $PATHOME_TRACE_DIR/phase0r_traces.jsonl   (append-mode)
              +------------------------------------------+
              | {                                        |   one line
              |   "ts": 1715520000.123,                  |   per tuple-run
              |   "profile_id": "Soybean::Charcoal Rot", |
              |   "crop": "Soybean", "disease": "...",   |
              |   "state": "Alabama",                    |
              |   "primary_image_id": "bugwood::1568038",|
              |   "image_path": ".../bugwood_cache/..",  |
              |   "run_idx": 0,                          |
              |   "path": ["MorphologyAgent",            |
              |            "SymptomAgent", ...,          |
              |            "DiagnosisAgent"],            |
              |   "decisions": ["model_choice", ...],    |
              |   "confidences": ["medium","high",...],  |
              |   "backtrack_count": 1,                  |
              |   "early_terminated": true,              |
              |   "context_buffer": [                    |
              |     { agent_name, deltas, confidence,    |
              |       handoff_target, reasoning,         |
              |       raw_text },                        |
              |     ...                                  |
              |   ],                                     |
              |   "final_deltas": [...],                 |
              |   "existing_kb_at_start": [...]          |
              | }                                        |
              +------------------------------------------+
```

This is the source the OBSERVE trainer reads.

---

## 6. File map

```
PlantSwarm/
|-- README.md                              narrative + commands
|-- FLOW.md                                this file
|
|-- BugWood_Diseases.csv                   raw IPMNet export
|-- BugWood_Diseases_usable.csv            filtered (Setup output)
|
|-- configs/bugwood_pathome.yaml           swarm + model knobs
|
|-- pathome_kb/                            Phase 0 + Phase 0R orchestration
|   |-- pipeline.py                        per-crop orchestrator (CLI)
|   |-- internet_pipeline.py               Claude discovery + extraction + reconciliation
|   |-- regional_observation.py            per-tuple Qwen-swarm caller
|   |-- symptoms_adapter.py                registry to SymptomProfile JSON
|   |-- prompts/                           canonical-stage prompts
|   `-- shared.py / utils.py / config.py
|
|-- plantswarm/                            Qwen swarm
|   |-- delta_pipeline.py                  run_for_state, run_batch,
|   |                                       algorithm1_handoff,
|   |                                       _merge_with_existing,
|   |                                       _agreement_filter,
|   |                                       existing_deltas_for_state,
|   |                                       _TraceWriter (PATHOME_TRACE_DIR)
|   `-- latex/                             EMNLP 2026 paper sources
|
|-- observe/                               Phase OBSERVE distilled student
|   |-- model.py                           Qwen2.5-VL-7B + LoRA + 6 heads
|   |-- trainer.py                         RoutingTraceDataset (Phase 0R JSONL),
|   |                                       TraceStepAnnotation,
|   |                                       OBSERVETrainer,
|   |                                       split_annotations
|   |-- loss.py                            multi-task L_rt + L_cal + L_cons + L_OC + L_bel
|   |-- inference.py                       OBSERVEInference single-pass
|   |-- decision_transformer.py            Phase A (NOT yet ported)
|   |-- grpo.py                            Phase B (NOT yet ported)
|   `-- active_learning.py                 epsilon-aware sample selection
|
|-- agents/                                5 delta-extraction agents
|   |-- base_agent.py                      DELTA_USER_PROMPT,
|   |                                       parse_agent_output,
|   |                                       AgentDeltaOutput,
|   |                                       _format_existing_kb,
|   |                                       _format_prior_context
|   |-- morphology_agent.py                lesion_morphology, affected_organs, diagnostic_features
|   |-- symptom_agent.py                   spread_pattern, diagnostic_features
|   |-- pathogen_agent.py                  look_alikes, type_of_disease
|   |-- severity_agent.py                  severity, treatments
|   `-- diagnosis_agent.py                 per-trace consolidator
|
|-- pathome/                               schema for symptoms_seed.json
|   `-- symptoms.py                        SymptomLibrary, SymptomProfile,
|                                           CanonicalDisease, RegionalObservation,
|                                           RegionalDelta, Citation
|
|-- utils/
|   |-- vllm_client.py                     OpenAI-compatible vLLM client
|   |                                       (per-call seed + temperature,
|   |                                        thread-safe guided fallback)
|   `-- geo.py                             state centroid + AEZ (Setup)
|
|-- data/bugwood_loader.py                 _clean_disease + _map_crop (Setup)
|
|-- scripts/
|   |-- filter_bugwood_csv.py              Setup
|   |-- ensure_state_image_cache.py        image cache
|   |-- registry_to_excel.py               final_registry.json to xlsx
|   |-- run_phase0_local.sh                LOCAL canonical-only Phase 0
|   |-- submit_pathome_setup_filter.sh     Nova filter CSV
|   |-- submit_phase0r_regional.sh         Nova boot vLLM + Phase 0R
|   |-- train_observe.py                   train OBSERVE on Phase 0R traces
|   `-- submit_observe_train.sh            Nova train OBSERVE
|
`-- smoke/                                 2-crop happy path
    |-- run_phase0_full.sh                 LOCAL P0 + tunneled P0R
    |-- run_phase0_local.sh                LOCAL canonical-only P0
    |-- bugwood_pathome_smoke.yaml         smaller N + Tmax
    `-- README.md
```

---

## 7. Env var reference

| Env var | Default | Controls |
|---|---|---|
| VLLM_BASE_URL | http://localhost:8000/v1 | OpenAI-compatible vLLM endpoint |
| VLLM_MODEL | Qwen/Qwen2.5-VL-7B-Instruct | Served model id |
| VLLM_TIMEOUT | 180 | Per-HTTP-call timeout (s) |
| VLLM_TEMPERATURE | 0.8 | Per-call sampling temperature |
| VLLM_N_RUNS | 10 (smoke: 5) | Stochastic traces per tuple |
| VLLM_AGREEMENT_MIN | 3 (smoke: 2) | K-of-N agreement floor |
| VLLM_TMAX | 15 (smoke: 8) | Max path length per trace |
| VLLM_MAX_BACKTRACKS | 1 | Max backtracks (actually honored) |
| VLLM_SIM_THRESHOLD | 0.4 | Jaccard threshold for clustering + merge |
| PATHOME_IMAGE_CACHE_DIR | — | Prepended to default cache search path |
| PATHOME_TRACE_DIR | — | When set, Phase 0R appends per-trace records to `<dir>/phase0r_traces.jsonl` |
| PATHOME_TRACE_FILE | phase0r_traces.jsonl | Trace JSONL filename within `PATHOME_TRACE_DIR` |
| OBSERVE_EPOCHS | 5 | Training epochs |
| OBSERVE_BATCH | 4 | Training batch size |
| OBSERVE_LR | 1e-4 | AdamW learning rate |
| OBSERVE_LORA_R / OBSERVE_LORA_ALPHA | 16 / 32 | LoRA config |
| OBSERVE_SAVE_DIR | observe/checkpoints/ | Checkpoint output |
| ANTHROPIC_API_KEY | — (optional) | Speeds up Phase 0 reconciliation; falls back to `claude -p` |
| PATHOME_ONLY_CROPS | — | Comma-separated crop allowlist |
| PATHOME_USABLE_CSV | BugWood_Diseases_usable.csv | Filtered CSV path |
| PATHOME_SEED_FILE | artifacts/pathome_seed/symptoms_seed.json | Output seed JSON path |
| PATHOME_SEED_QUICK | 0 | Cap states per disease for fast iteration |

---

## 8. Run-report line

One line per (crop, disease, state) tuple printed by `run_batch`:

```
[7/50] OK  Soybean::Charcoal Rot / Alabama  deltas=8 (N=10, K>=3, existing=4, added=2, bumped=3)
        |   |              |      |          |     |       |          |          |
        |   |              |      |          |     |       |          |          +-- overlap-bumped candidates
        |   |              |      |          |     |       |          +------------- net-new this run
        |   |              |      |          |     |       +------------------------ prior deltas loaded
        |   |              |      |          |     +-------------------------------- K = agreement floor
        |   |              |      |          +-------------------------------------- N = stochastic traces
        |   |              |      +------------------------------------------------- final merged count
        |   |              +-------------------------------------------------------- state
        |   +----------------------------------------------------------------------- crop::disease
        +--------------------------------------------------------------------------- progress
```

Reading examples:

- `existing=0, added=8` → cold start; swarm produced 8 new agreed deltas
- `existing=4, added=2, bumped=3` → iterative re-run; 4 prior preserved,
  2 net-new, 3 candidates already known (support incremented)
- `existing=4, added=0, bumped=0` → swarm produced no new info; KB
  stable for this state
