# PlantSwarm + PathomeDB — KB build + dual-track encoder evaluation

Six-step pipeline. Each step runs in a fixed place (LOCAL or NOVA) and
hands off to the next step via `git push` / `git pull`:

```
                 ┌──────────────── STEP 0 ─ LOCAL  ─────────────────┐
                 │ scripts/sh_00_setup_local.sh                     │
                 │   filter raw Bugwood CSV by per-class threshold  │
                 │   + Claude 2-layer label judge                   │
                 │     LAYER 1: crop decision tree                  │
                 │     LAYER 2: per-disease CORRECT/INCORRECT/      │
                 │              QUESTIONABLE                        │
                 │   drops INVALID/NON_CROP crops + INCORRECT       │
                 │   diseases; canonicalises MISSPELLED crops       │
                 │   → BugWood_Diseases_usable.csv + git push       │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 1 ─ LOCAL  ─────────────────┐
                 │ scripts/sh_01_phase0_local.sh                    │
                 │   Claude Phase 0 canonical KB build              │
                 │   (NON-visual: pathogen, type, parts, treatments)│
                 │   → git push                                     │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 2 ─ NOVA  ──────────────────┐
                 │ scripts/sh_02_swarm_nova.sh                      │
                 │   git pull                                       │
                 │   24-agent 2-round Qwen2.5-VL real swarm         │
                 │   (visual symptoms ONLY; verifier OFF here)      │
                 │   → git push  (deltas tagged "unverified")       │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 3 ─ LOCAL  ─────────────────┐
                 │ scripts/sh_03_validate_local.sh                  │
                 │   git pull                                       │
                 │   Claude + WebSearch verifier over each delta    │
                 │   (extension / APS / CABI / peer-reviewed)       │
                 │   → git push  (deltas tagged verified /          │
                 │                provisional / contradictory etc.) │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 4 ─ NOVA  ──────────────────┐
                 │ scripts/sh_04_train_encoder_nova.sh   (NEW)      │
                 │   git pull verified KB                           │
                 │   build KB-grounded captions + WebDataset shards │
                 │   train BioCAP-style ViT-B/16 dual-projector     │
                 │     CLIP encoder warm-started from BioCLIP       │
                 │   (default: ONE variant T04; TRAIN_FULL_MATRIX=1 │
                 │    sbatches all 11 BioCAP-style training         │
                 │    variants)                                     │
                 │   → epoch_50.pt checkpoint (~600 MB)             │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 5 ─ LOCAL ──────────────────┐
                 │ scripts/sh_05_tabpfn_local.sh                    │
                 │   git pull verified KB + step-4 checkpoint       │
                 │   build captions (per strategy)                  │
                 │   FROZEN encoder forward (BioCLIP / BioCLIP-2 /  │
                 │     CLIP / SigLIP / FG-CLIP / BioTrove / ours-T15│
                 │     = the step-4 trained encoder) over Bugwood + │
                 │     PV + PD + PW                                 │
                 │   feature vec = [image_emb | caption_emb |       │
                 │                  crop_text_emb | state_text_emb] │
                 │   TabPFN classifier over 15-variant feature      │
                 │     ablation matrix (zero trained params on      │
                 │     visual side; TabPFN is a meta-learned        │
                 │     tabular foundation model)                    │
                 │   Grad-CAM (BioCAP §C.3 reproduction; energy-    │
                 │     pointing-game if bbox CSV provided)          │
                 │   eval on PV + PD + PW                           │
                 │   aggregate paper-style tables                   │
                 │   → git push results                             │
                 └──────────────────────────────────────────────────┘
```

The split is deliberate. **Nova has the GPU** but no `claude` CLI;
**LOCAL has Claude** but no A100. Each step runs on the host that has
the right tool. Steps 0, 1, 3 need Claude (LOCAL); steps 2 & 4 need
GPUs (NOVA); step 5 is small enough to run on LOCAL or any single-GPU
host.

Two command sets are documented below. They differ only by which crops
are processed:

| Set | Crops | Wall-clock | API spend | Use case |
|---|---|---|---|---|
| **A. 2-crop (smoke)** | Soybean + Tomato | ~6-10 h end-to-end | ~$5-15 | first-time run, validates the pipeline, fits in a day |
| **B. all-crop (production)** | All 484 (crop, disease) pairs in `BugWood_Diseases_usable.csv` | ~5-8 days end-to-end | ~$80-300 | the real paper run |

---

## Set A — 2-crop smoke (Soybean + Tomato)

Start here. End-to-end in under a day; ~$5-15 in Claude API spend.

```bash
# ============================================================
# STEP 0 — LOCAL (filter raw CSV + Claude label judge)
# ============================================================
cd ~/Desktop/PlantSwarm
JUDGE_LABELS=1 bash scripts/sh_00_setup_local.sh
# ≈ 10-30 min. Filters BugWood_Diseases.csv into the threshold-
# satisfying BugWood_Diseases_usable.csv, then runs the Claude
# two-layer judge over the surviving (NormCrop, NormDisease) pairs
# and rewrites the CSV without INVALID/NON_CROP crops or INCORRECT
# diseases (MISSPELLED crops are canonicalised in place). Sidecar
# JSON report at artifacts/bugwood_judgement.json is resume-keyed.
# Skip the judge: JUDGE_LABELS=0 bash scripts/sh_00_setup_local.sh

# ============================================================
# STEP 1 — LOCAL (Phase 0 canonical KB via Claude)
# ============================================================
cd ~/Desktop/PlantSwarm
CROPS=smoke bash scripts/sh_01_phase0_local.sh
# ≈ 30-45 min. Writes artifacts/pathome_kb/{Soybean,Tomato}/final_registry.json
# then commits + pushes to origin/main.

# ============================================================
# STEP 2 — NOVA (24-agent 2-round Qwen swarm; verifier OFF)
# ============================================================
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
CROPS=smoke bash scripts/sh_02_swarm_nova.sh
# ≈ 3-6 h. sbatch one Phase 0R job (vLLM + 24-agent 2-round swarm),
# blocks until done, then pushes the unverified-deltas KB back to GitHub.
# Tip: set PATHOME_TRACE_DIR=artifacts/swarm_smoke to capture per-pass
# JSONL traces (round1_outputs, round2_outputs, cross_refs).

# ============================================================
# STEP 3 — LOCAL (Claude+WebSearch validation)
# ============================================================
# (back on your laptop)
cd ~/Desktop/PlantSwarm
git pull origin main
CROPS=smoke bash scripts/sh_03_validate_local.sh
# ≈ 30-60 min on smoke. Drives pathome_kb.verifier.verify_candidates
# tuple-by-tuple, fills in verification_status + web_support per delta,
# pushes verified KB back to GitHub.

# ============================================================
# STEP 4 — NOVA (BioCAP-style encoder fine-tune)
# ============================================================
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main
CROPS=smoke bash scripts/sh_04_train_encoder_nova.sh
# ≈ 30-60 min on one A100 (default = ONE variant T04, 50 epochs,
# projectors-only). Writes
#   train_and_eval/checkpoints/T04/<run-id>/checkpoints/epoch_50.pt
# Set TRAIN_FULL_MATRIX=1 to sbatch all 11 BioCAP-style training
# variants (~5 GPU-h). Set PATHOME_PUSH_CHECKPOINT=1 to git-push
# the checkpoint (large — ~600 MB), or scp it back to LOCAL
# manually for the next step.

# ============================================================
# STEP 5 — LOCAL (frozen encoders + TabPFN classifier + Grad-CAM)
# ============================================================
# (back on your laptop / any small-GPU host)
cd ~/Desktop/PlantSwarm
git pull origin main
# point at your step-4 checkpoint (or rely on the default path)
export PATHOMEOOD_CKPT=train_and_eval/checkpoints/T04/T04/checkpoints/epoch_50.pt
CROPS=smoke bash scripts/sh_05_tabpfn_local.sh
# ≈ 1-2 h (frozen forward on Tomato images + PV/PD/PW for 7 encoders
# × 7 caption strategies + TabPFN inference for 15 variants + Grad-CAM
# for qualitative figures). No CLIP training in this step; TabPFN is
# meta-learned. Runs on a small GPU for the encoder forward + CPU
# for TabPFN. Pushes paper-style tables to GitHub.
```

Final outputs after Set A:

```
artifacts/pathome_kb/Soybean/final_registry.json    canonical + verified deltas
artifacts/pathome_kb/Tomato/final_registry.json     canonical + verified deltas
train_and_eval/checkpoints/T04/.../epoch_50.pt     your trained encoder (step 4)
data/bugwood_features/<encoder>_<strategy>.npz     frozen-encoder Bugwood features
data/eval_features/<encoder>_<strategy>_<set>.npz  frozen-encoder PV/PD/PW features
results/pathomeood_eval/<variant>/{plantvillage,plantdoc,plantwild}.json   TabPFN results
results/figures/gradcam/<encoder>/<set>/<class>/*.png  Grad-CAM triptychs
results/tables/{table_01,...,figure_03}.md          paper-style markdown
results/pathomeood_report.md                        master report
```

---

## Set B — all-crop production (484 classes)

The real run. ~5-8 days end-to-end; ~$80-300 in Claude API spend.
Recommended only after Set A has succeeded end-to-end.

```bash
# ============================================================
# STEP 0 — LOCAL (filter raw CSV + Claude label judge)
# ============================================================
cd ~/Desktop/PlantSwarm
JUDGE_LABELS=1 bash scripts/sh_00_setup_local.sh
# ≈ 1-2 h on full Bugwood (~$3-10 Claude spend for the judge).
# Produces the cleaned BugWood_Diseases_usable.csv consumed by step 1.

# ============================================================
# STEP 1 — LOCAL (Phase 0 canonical KB for ALL 197 crops)
# ============================================================
cd ~/Desktop/PlantSwarm
CROPS=all bash scripts/sh_01_phase0_local.sh
# ≈ 16-24 h. ~$60-180 in Anthropic API spend. Writes
# artifacts/pathome_kb/<Crop>/final_registry.json for every crop in
# BugWood_Diseases_usable.csv (197 of them).

# ============================================================
# STEP 2 — NOVA (24-agent swarm over ~2,000-3,000 image tuples)
# ============================================================
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
CROPS=all VLLM_N_RUNS=10 VLLM_AGREEMENT_MIN=3 \
  bash scripts/sh_02_swarm_nova.sh
# ≈ 24-48 h. ~2,000-3,000 (crop, disease, state) tuples × 25 (or 49 in
# 2-round mode) vLLM calls each. Set VLLM_SWARM_ROUNDS=1 to fall back
# to single-round mode if you want ~half the wall-clock.

# ============================================================
# STEP 3 — LOCAL (Claude+WebSearch validation over every unverified delta)
# ============================================================
cd ~/Desktop/PlantSwarm
git pull origin main
CROPS=all bash scripts/sh_03_validate_local.sh
# ≈ 1-3 days. ~$20-100 in Claude spend. Use MAX_TUPLES=N to cap if
# you want to bound spend (the leftover deltas stay tagged
# "unverified" and PathomeOOD will still use them via the fallback
# caption path).

# ============================================================
# STEP 4 — NOVA (BioCAP-style encoder fine-tune over full Bugwood)
# ============================================================
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main
CROPS=all TRAIN_FULL_MATRIX=1 \
  bash scripts/sh_04_train_encoder_nova.sh
# ≈ 5-10 GPU-h with TRAIN_FULL_MATRIX=1 (11 variants × ~30 min each
# on one A100). Drop TRAIN_FULL_MATRIX to train just T04, the main
# variant, in ~30-60 min. Outputs land under
#   train_and_eval/checkpoints/T*/<run-id>/checkpoints/epoch_50.pt
# scp the T04 checkpoint back to LOCAL for the next step, or set
# PATHOME_PUSH_CHECKPOINT=1 to commit + push it (large object).

# ============================================================
# STEP 5 — LOCAL (frozen encoders + TabPFN over full Bugwood)
# ============================================================
cd ~/Desktop/PlantSwarm
git pull origin main
export PATHOMEOOD_CKPT=train_and_eval/checkpoints/T04/T04/checkpoints/epoch_50.pt
CROPS=all bash scripts/sh_05_tabpfn_local.sh
# ≈ 4-8 h on a single small GPU. Encoder forward for ~12K Bugwood
# images + PV/PD/PW, for each of 7 encoders (6 off-shelf + your
# step-4-trained T15 encoder) × 7 caption strategies, then TabPFN
# inference for all 15 variants on CPU. TabPFN scales O(N²) in
# train rows; we cap at 10K via stratified subsample.
```

Final outputs after Set B:

```
artifacts/pathome_kb/*/final_registry.json          197 crop registries
train_and_eval/checkpoints/T*/*.pt                  your BioCAP-style trained encoder(s)
data/bugwood_features/*.npz                         frozen-encoder features (7 encoders)
data/eval_features/*.npz                            frozen-encoder eval features
results/pathomeood_eval/<variant>/*.json            TabPFN results (15 variants × 3 sets + 7 baselines)
results/figures/gradcam/<encoder>/<set>/<class>/*.png   Grad-CAM triptychs (BioCAP §C.3)
results/pathomeood_report.md                        paper-style master report
```

---

## What each step does, in one sentence

| # | Where | Script | What |
|---|---|---|---|
| 0 | LOCAL | `sh_00_setup_local.sh` | Filter raw Bugwood CSV by per-class threshold + Claude 2-layer label judge (drops INVALID/NON_CROP crops, INCORRECT diseases; canonicalises MISSPELLED crops) |
| 1 | LOCAL | `sh_01_phase0_local.sh` | Claude builds the canonical (text-grounded, NON-visual) KB per crop |
| 2 | NOVA | `sh_02_swarm_nova.sh` | 24-agent 2-round Qwen2.5-VL real swarm extracts image-grounded visual deltas (verifier OFF) |
| 3 | LOCAL | `sh_03_validate_local.sh` | Claude+WebSearch verifies each delta against extension / APS / CABI |
| 4 | NOVA | `sh_04_train_encoder_nova.sh` | BioCAP-style ViT-B/16 dual-projector CLIP encoder fine-tuned on Bugwood + KB-grounded captions (warm-started from BioCLIP) |
| 5 | LOCAL | `sh_05_tabpfn_local.sh` | 7 frozen encoders (6 off-shelf + your step-4 trained one) emit image_emb + KB-caption_emb + crop_text_emb + state_text_emb; TabPFN classifies; Grad-CAM; eval on PV / PD / PW |

### Manual hand-off helpers (push / pull)

Each `sh_NN` step script does its own `git pull --ff-only` at the
start and `git push` at the end, so a normal end-to-end run never
needs these helpers. They exist for manual hand-offs — e.g. after a
step crashes mid-run, when you want to refresh Nova before sbatching,
or when you want to sync the entire working tree (not just the per-
step artifacts) between hosts.

| Direction | Script | What |
|---|---|---|
| LOCAL → GitHub or NOVA → GitHub | `scripts/sh_push_to_github.sh` | **HARD push.** `git add -A .` stages every changed / new file under the repo (every subdir included), commits, and pushes. |
| GitHub → LOCAL or GitHub → NOVA | `scripts/sh_pull_from_github.sh` | `git fetch` + `git pull --ff-only`. Refuses to clobber uncommitted local edits (tells you to `commit` or `stash` first). |

**`sh_push_to_github.sh` is a hard push** — every file, every
subdirectory under the repo root goes up in one commit. Knobs:

| Env var | Default | Effect |
|---|---|---|
| `COMMIT_MSG` | timestamped "hard push" | Commit message |
| `PATHOME_INCLUDE_IGNORED` | `0` | `1` → also force-add `.gitignore`'d files (large caches, checkpoints, data dirs) with `git add -A -f` |
| `PATHOME_FORCE_PUSH` | `0` | `1` → use `git push --force-with-lease` (only when local + remote have diverged) |
| `PATHOME_DRY_RUN` | `0` | `1` → print plan + the (potentially huge) untracked / ignored counts without staging or pushing |
| `GIT_REMOTE` | `origin` | Remote name |
| `GIT_BRANCH` | `main` | Branch to push to |

Before staging, the script prints a preview of the working-tree
status and the count of untracked (and, with `PATHOME_INCLUDE_IGNORED=1`,
ignored) paths. Ctrl-C if it's about to grab something unintended.

```bash
# Hard-push everything (from either host) — every file, every subdir:
bash scripts/sh_push_to_github.sh

# Same, but with a specific commit message:
COMMIT_MSG="hand-off after partial swarm run" bash scripts/sh_push_to_github.sh

# Also push gitignored files (caches, large data, checkpoints):
PATHOME_INCLUDE_IGNORED=1 bash scripts/sh_push_to_github.sh

# Force-push if local and remote have diverged:
PATHOME_FORCE_PUSH=1 bash scripts/sh_push_to_github.sh

# Preview without staging or pushing:
PATHOME_DRY_RUN=1 bash scripts/sh_push_to_github.sh

# Pull on the other host (fast-forward; aborts on uncommitted edits):
bash scripts/sh_pull_from_github.sh
```

---

## Skip-knobs (re-run only some steps)

Each shell script reads env vars for partial runs:

```bash
# Step 0
THRESHOLD=10                     # min rows per (crop, disease) class
JUDGE_LABELS=0                   # 1 = also run Claude judge (default 1)
DROP_QUESTIONABLE=1              # also drop QUESTIONABLE diseases
PER_CLASS=200                    # optional cap on rows per class (0 = none)
PATHOME_RAW_CSV=BugWood_Diseases.csv

# Step 1
PATHOME_USABLE_CSV=other.csv     # override input CSV
PATHOME_SKIP_PUSH=1              # commit but don't push

# Step 2
PATHOME_TRACE_DIR=traces/        # capture per-pass JSONL traces
VLLM_N_RUNS=5                    # cheaper smoke (default 10)
VLLM_SWARM_ROUNDS=1              # disable round 2 (cheaper, less stigmergy)
VLLM_AGREEMENT_MIN=2             # K-of-N floor (default 3)

# Step 3
MAX_TUPLES=50                    # cap on (crop, disease, state) tuples
DRY_RUN=1                        # print plan without calling Claude

# Step 4 (BioCAP-style encoder train on NOVA)
TRAIN_VARIANT=T04                # single variant to train (default)
TRAIN_FULL_MATRIX=1              # sbatch all 11 training variants instead
PATHOME_SKIP_CAPTIONS=1          # captions already built
PATHOME_SKIP_SHARDS=1            # WebDataset shards already built
PATHOME_SKIP_TRAIN=1             # skip training (just push existing ckpts)
PATHOME_PUSH_CHECKPOINT=1        # git-push the checkpoint after training

# Step 5 (TabPFN + Grad-CAM on LOCAL; default)
PATHOMEOOD_CKPT=path/to/epoch_50.pt   # locally-trained encoder (T15)
PATHOME_SKIP_CAPTIONS=1          # captions already built
PATHOME_SKIP_FEATURES=1          # encoder forwards already cached
PATHOME_SKIP_TABPFN=1            # TabPFN matrix already run (re-aggregate only)
PATHOME_SKIP_GRADCAM=1           # skip Grad-CAM figures
PATHOME_SKIP_AGG=1               # skip aggregation
ENCODERS="bioclip,pathomeood_v1" # which encoders to extract features for
STRATEGIES="canonical_deltas_3"  # which caption strategies to extract
BBOX_CSV=path/to/bboxes.csv      # enables energy-pointing-game score
```

---

## Architecture overview

This section explains *what* is being built. For *how to run it*, use
the command sets above.

### Phase 0 — Canonical KB (Claude, LOCAL)

For each (crop, disease) pair in `BugWood_Diseases_usable.csv`:

1. **Discovery** — Claude searches extension / APS / CABI / peer-
   reviewed sources for the most authoritative descriptions.
2. **Extraction** — Claude pulls verbatim quotes for each canonical
   field (`pathogen_scientific_name`, `type_of_disease`,
   `affected_parts`, `visual_symptoms.summary`,
   `visual_symptoms.diagnostic_features`,
   `visual_symptoms.look_alikes`, `treatments`).
3. **Reconciliation** — `claude -p` (headless CLI, JSON-schema mode)
   merges the per-source extractions into one canonical record with
   URL + verbatim quote per field. No Anthropic API key path —
   everything runs on the user's Claude Code subscription.

Output: `artifacts/pathome_kb/<Crop>/final_registry.json` with the
top-level `diseases[]` array. `regional_observations` is empty at this
stage; Phase 0R fills it in.

### Phase 0R — 24-agent 2-round real swarm (Qwen2.5-VL, NOVA)

**The "real swarm" part.** Naive parallel-ensemble setups have
specialists run in isolation and a consolidator collects outputs. This
is a real swarm because it has **stigmergy** (a shared blackboard) and
**cross-talk** (specialists react to each other's findings):

```
Round 1 — independent observation
  └─ 24 specialists run in parallel on (image, canonical KB, existing KB)
  └─ each asks ONE laser-focused visual question
  └─ no peer visibility yet

Blackboard built from all round-1 outputs (dict[AGENT_NAME → output])

Round 2 — stigmergy refinement
  └─ same 24 specialists run AGAIN in parallel
  └─ each now sees the FULL blackboard rendered in its prompt
  └─ may emit cross_refs against peers:
       SUPPORT   — raises peer's effective confidence
       CHALLENGE — consolidator must adjudicate
       WITHDRAW  — self-cancel a round-1 delta

VisualDiagnosisAgent (consolidator)
  └─ sees BOTH rounds + cross-ref digest grouped by action
  └─ walks 5-step CoT (decision-graph from DR.Arti.docx):
       (1) triage which organs are visible
       (2) decisive forks
       (3) adjudicate cross_refs
       (4) dedup
       (5) emit final deltas + CoT trace
```

The 24 specialists are decomposed into 7 organ families:

| Family | Count | Specialists |
|---|---|---|
| LEAF | 8 | LeafLesionShape, LeafLesionColor, LeafLesionTexture, LeafChlorosis, LeafNecrosis, LeafCurl, LeafVeinPattern, LeafGeometry |
| STEM | 4 | StemLesion, **StemPith** (decisive SDS/BSR fork), StemSurface, StemDiscoloration |
| BELOW-GROUND | 2 | **Root** (cysts → SCN; blue masses → SDS), CrownCollar |
| REPRODUCTIVE | 2 | Flower, Fruit |
| PATHOGEN SIGNS | 1 | Sporulation (mycelium / spores / ooze) |
| WHOLE-PLANT PATTERNS | 3 | Wilting, **Defoliation** (bare-petiole SDS fork), SpatialPattern |
| DIAGNOSTIC CROSS-CUTTERS | 4 | ConcentricPattern, **ColorPalette** (color encoder), **LookAlikeCoT** (decision-graph), SeverityVisual |

Per-pass cost: 24 specialists × 2 rounds + 1 consolidator = **49 vLLM
calls**. N=10 stochastic passes per (crop, disease, state) tuple.

The swarm focuses **exclusively on visual symptoms**. Pathogen, type,
affected parts, treatments — those are all handled by Claude in Phase 0
and never re-emitted by the swarm.

### Phase 0R verification — Claude+WebSearch (LOCAL, step 3)

Nova writes deltas with `verification_status="unverified"`. Step 3
walks every unverified delta, sends it (with context) to
`pathome_kb.verifier.verify_candidates` which calls `claude -p` with
WebSearch. Each delta gets:

| `verification_status` | Meaning | Goes to KB? |
|---|---|---|
| `verified` | direct hit on multiple authoritative sources | ✓ |
| `weakly_supported` | one source agrees | ✓ |
| `provisional` | no direct support but biologically plausible | ✓ |
| `novel_plausible` | new observation, plausible mechanism | ✓ (flagged) |
| `contradictory` | sources contradict the claim | ✗ (dropped) |
| `duplicate_existing` | matches an existing delta | merged (support++) |

### Phase PathomeOOD — BioCAP-style encoder fine-tune (NOVA, step 4)

This step trains your OWN encoder, BioCAP-style, so that the next step
can compare it against 6 off-shelf encoders as the **T15** entry in the
encoder-importance ablation.

**Model**: ViT-B/16 dual-projector CLIP, warm-started from BioCLIP.
Two visual projectors are trained jointly: one against the class-name
text tower (label supervision), one against the KB-grounded caption
text tower (canonical visual_symptoms + per-state regional deltas).

**Default**: one variant `T04` (= `canonical_deltas_3` caption,
projectors-only, 50 epochs). One A100, ~30-60 min, one checkpoint.

**Full ablation**: `TRAIN_FULL_MATRIX=1` sbatches all 11 caption-
strategy / projector-mode / epoch-count variants
(`scripts/pathomeood_variants.sh::PATHOMEOOD_VARIANTS`). ~5 GPU-h
total. Useful when you want to reproduce the full BioCAP-paper-style
ablation by training rather than by post-hoc TabPFN.

**Output**: `train_and_eval/checkpoints/<VARIANT>/<run-id>/checkpoints/epoch_50.pt`.
Either set `PATHOME_PUSH_CHECKPOINT=1` to commit + push it to GitHub
(checkpoints are ~600 MB; large for git but workable for one or two),
or `scp` it back to LOCAL before step 5.

### Phase PathomeOOD — frozen encoders + TabPFN classifier + Grad-CAM (LOCAL, step 5)

For the small-data regime (~10–12K Bugwood images), step 5
**replaces full CLIP fine-tuning at evaluation time with a frozen-
encoder + tabular foundation classifier setup**. Zero trained
parameters on the visual side; the classifier is meta-learned
[TabPFN](https://arxiv.org/abs/2207.01848). The encoder pool includes
**your own step-4 trained encoder** alongside 6 off-shelf encoders.

**Feature vector per image** (all four blocks come from each encoder's
paired text tower — no one-hot anywhere):
```
x = [ image_emb       (frozen visual encoder)              ≈ 512–1024 dim
    | caption_emb     (text tower on the KB-derived
                       caption for this row)                ≈ 512      dim
    | crop_text_emb   (text tower on
                       "a photograph of a <crop> plant.")   ≈ 512      dim
    | state_text_emb  (text tower on
                       "a photograph taken in <state>.")    ≈ 512      dim ]
```
PCA-reduced before TabPFN, keeping it well under the TabPFNv2
feature-count limit. Text-embedded metadata generalizes to unseen
crops at test time, unlike one-hot.

**15-variant feature ablation matrix** (`scripts/tabpfn_eval.py::VARIANTS`):
T01–T07 vary the caption strategy (label_only → canonical_deltas_7);
T08–T12 swap the encoder (CLIP / SigLIP / BioCLIP-2 / FG-CLIP /
BioTrove); T13–T14 restrict the train set to KB-covered / non-covered
classes; **T15 = `pathomeood_v1` = your step-4 trained encoder**.
**Zero training per variant** — TabPFN does in-context learning over
the support set in one forward pass.

Plus 7 off-shelf zero-shot baselines (CLIP / SigLIP / BioCLIP /
BioCLIP-2 / FG-CLIP / BioTrove / `pathomeood_v1`) computed by straight
cosine-sim against class-name templates.

**Grad-CAM** (`scripts/gradcam_eval.py`, BioCAP §C.3 reproduction):
forward + backward hooks on the final ViT transformer block. The
cosine logit `s = (V·T) / (||V|| ||T||)` is backpropagated;
`ReLU(mean_grad · feature_map)` is upsampled to a 2-D heatmap. Saves
PNG triptychs (input | heatmap | overlay) per encoder × eval set ×
class. With `--bbox-csv` it also computes the energy-pointing-game
score for Table 14.

Eval: top-1 / top-5 on PlantVillage, PlantDoc, PlantWild — same three
test sets as the original BioCAP-style table reproduction. Outputs
written to `results/pathomeood_eval/<variant>/{plantvillage,plantdoc,plantwild}.json`
and aggregated into `results/pathomeood_report.md` (same paper-style
table set as BioCAP).

Master report: `results/pathomeood_report.md`.

For the full architectural deep-dive see [PIPELINE.md](PIPELINE.md);
for the end-to-end animated walkthrough see [FLOW.md](FLOW.md).

---

## One-time prerequisites

### LOCAL

```bash
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Claude CLI for Phase 0 + verifier
# (install from https://claude.com/code; run `claude` once interactively
# to authenticate)
#
# All Claude calls in this pipeline go through the headless `claude -p`
# CLI — there is no Anthropic API key path. Your Claude Code
# subscription is the only billing surface.

# ---- Build BugWood_Diseases_usable.csv (one-time) ----
# Plain (threshold-only) build:
python scripts/filter_bugwood_csv.py \
    --input  BugWood_Diseases.csv \
    --output BugWood_Diseases_usable.csv \
    --threshold 10

# Optional: also run the Claude two-layer label judge over the
# surviving (NormCrop, NormDisease) pairs and drop INVALID / NON_CROP
# crops and INCORRECT diseases (also canonicalises MISSPELLED crop
# names in place). One Claude call per crop + one per crop-disease
# block; ~$5-15 total on Bugwood, fully resumable.
python scripts/filter_bugwood_csv.py \
    --input  BugWood_Diseases.csv \
    --output BugWood_Diseases_usable.csv \
    --threshold 10 --judge
# Add --judge-drop-questionable to also drop QUESTIONABLE labels.
# Sidecar report (resume key): artifacts/bugwood_judgement.json
```

### NOVA (one-time GPU-host install)

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/<your-scratch>/
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

# Standard deps:
pip install -r requirements.txt

# GPU-only deps (see requirements.txt's "GPU host only" section):
pip install vllm torch open_clip_torch webdataset huggingface_hub \
            transformers accelerate
```

---

## Repo layout

```
PlantSwarm/
├── README.md                              this file (run-it instructions)
├── PIPELINE.md                            architectural deep-dive
├── FLOW.md                                end-to-end flow + GIF
├── DR.Arti.docx                           reference doc with look-alike CoT
│                                          decision graphs (SDS↔BSR etc.)
│
├── BugWood_Diseases.csv                   raw IPMNet export
├── BugWood_Diseases_usable.csv            filtered (Setup output)
│
├── disease_label_judge.py                 Claude 2-layer crop / disease
│                                          label judge (LAYER 1 = crop
│                                          decision tree; LAYER 2 = per-
│                                          disease CORRECT / INCORRECT /
│                                          QUESTIONABLE). Importable +
│                                          standalone CLI; wired into
│                                          filter_bugwood_csv.py via --judge.
│
├── agents/                                24-specialist visual-symptom swarm
│   ├── base_agent.py                      Blackboard, CROSS_REF_ACTIONS,
│   │                                      DELTA_USER_PROMPT (R1 + R2)
│   ├── leaf_agents.py                     8 leaf specialists
│   ├── stem_agents.py                     4 stem specialists
│   ├── root_agents.py                     Root + CrownCollar
│   ├── reproductive_agents.py             Flower + Fruit
│   ├── sign_agents.py                     Sporulation (signs vs symptoms)
│   ├── pattern_agents.py                  Wilting + Defoliation + Spatial
│   ├── diagnostic_agents.py               Concentric + ColorPalette +
│   │                                      LookAlikeCoT + Severity
│   └── diagnosis_agent.py                 VisualDiagnosisAgent CoT consolidator
│
├── train_and_eval/                        BioCAP-style dual-projector CLIP code
│                                          used by STEP 4 (sh_04_train_encoder)
│                                          to fine-tune your own encoder
│                                          (warm-start from BioCLIP).
│   ├── open_clip/                         model + two visual projectors
│   ├── open_clip_train/                   torchrun entry (data + train adapted)
│   ├── evaluation/                        zero_shot_iid + retrieval + metrics
│   ├── imageomics/                        naming_eval + disk + helpers
│   └── checkpoints/<VARIANT>/.../epoch_50.pt    step-4 trained encoder(s)
│
├── plantswarm/                            swarm orchestrator + captioner
│   ├── delta_pipeline.py                  2-round real swarm: run_for_state,
│   │                                      run_batch, _agreement_filter,
│   │                                      _merge_with_existing
│   └── captioning.py                      build_disease_caption (7 strategies),
│                                          _top_regional_deltas (state-aware),
│                                          load_kb_profiles, caption_for_row
│
├── paper/                                 paper sources (renamed from
│   │                                       plantswarm/latex/)
│   ├── plantswarm_paper.tex                main paper (renamed from acl_latex.tex)
│   ├── plantswarm_paper_lualatex.tex       lualatex variant
│   ├── auto_*.tex                          viz-script-emitted snippets
│   ├── appendix_dataset_licenses.tex       dataset licensing appendix
│   ├── plantswarm.bib / pathome3.bib       bibliographies
│   └── acl.sty / acl_natbib.bst            ACL style (vendored)
│
├── pathome_kb/                            Phase 0 + verifier
│   ├── pipeline.py                        per-crop orchestrator (CLI)
│   ├── internet_pipeline.py               Claude discovery + extraction +
│   │                                      reconciliation
│   ├── regional_observation.py            per-tuple Qwen-swarm caller
│   ├── verifier.py                        Claude web-search verifier
│   ├── symptoms_adapter.py                (legacy) merged-seed adapter
│   └── prompts/                           canonical-stage prompts
│
├── pathome/                               KB schema
│   └── symptoms.py                        SymptomLibrary, SymptomProfile,
│                                          CanonicalDisease, RegionalObservation,
│                                          RegionalDelta, Citation
│
├── utils/
│   ├── vllm_client.py                     OpenAI-compatible vLLM client
│   ├── geo.py                             state centroid + AEZ
│   └── env.py                             .env loader
│
├── data/bugwood_loader.py                 crop / disease normalization (Setup)
│
├── scripts/
│   ├── sh_00_setup_local.sh               STEP 0 — LOCAL: filter raw CSV
│   │                                       + Claude 2-layer label judge + push
│   ├── sh_01_phase0_local.sh              STEP 1 — LOCAL: Phase 0 + push
│   ├── sh_02_swarm_nova.sh                STEP 2 — NOVA: swarm + push
│   ├── sh_03_validate_local.sh            STEP 3 — LOCAL: validate + push
│   ├── sh_04_train_encoder_nova.sh        STEP 4 — NOVA: BioCAP-style
│   │                                       encoder fine-tune
│   ├── sh_05_tabpfn_local.sh              STEP 5 — LOCAL: frozen-encoder
│   │                                       forward + TabPFN + Grad-CAM + push
│   ├── sh_push_to_github.sh               manual hand-off: LOCAL/NOVA → GitHub
│   ├── sh_pull_from_github.sh             manual hand-off: GitHub → LOCAL/NOVA
│   ├── validate_kb.py                     step-3 driver (Claude verifier)
│   │
│   ├── build_pathomeood_captions.py       KB → captions parquet
│   ├── build_pathomeood_shards.py         parquet → WebDataset shards (step 4)
│   ├── pathomeood_variants.sh             T01..T11 training-matrix definition
│   ├── train_pathomeood.py                wrapper around open_clip_train.main
│   ├── submit_pathomeood_train.sh         SLURM: one training variant
│   ├── submit_pathomeood_matrix.sh        SLURM: sbatch all 11 training variants
│   ├── build_features.py                  frozen encoder forward → image_emb
│   │                                       + caption_emb + crop_text_emb +
│   │                                       state_text_emb npz (7 encoders incl.
│   │                                       your step-4 trained pathomeood_v1)
│   ├── tabpfn_eval.py                     TabPFN classifier over the 15-variant
│   │                                       feature ablation matrix + 7 baselines
│   ├── gradcam_eval.py                    Grad-CAM (BioCAP §C.3 reproduction);
│   │                                       --bbox-csv enables energy-pointing
│   ├── aggregate_pathomeood_tables.py     result JSONs → paper-style table .md
│   │
│   ├── evaluate_pathomeood.py             (optional) zero-shot eval on PV/PD/PW
│   ├── evaluate_pathomeood_retrieval.py   (optional) Bugwood held-out R@k
│   ├── evaluate_pathomeood_fewshot.py     (optional) prototype-mean K-shot
│   ├── fetch_baselines.py                 cache 5 off-shelf CLIP baselines
│   ├── setup_plantdoc.py                  clone PlantDoc to data/eval/
│   │
│   ├── filter_bugwood_csv.py              raw CSV → filtered usable CSV
│   ├── ensure_state_image_cache.py        per-(crop, disease, state) image cache
│   ├── submit_pathome_setup_filter.sh     Nova SBATCH: filter CSV
│   ├── setup_image_cache.sh               LOCAL/Nova: image cache
│   ├── submit_phase0r_regional.sh         Nova SBATCH: vLLM + Phase 0R swarm
│   │
│   ├── viz_kb.sh / viz_traces.sh / viz_all.sh   KB + trace visualizations
│   ├── build_latex_pdf.sh                 paper compile helper
│   └── viz/                               Python visualizers
│
└── smoke/                                 (legacy 2-crop happy path)
```

---

## Tests

```bash
pytest tests/ -q
# 59 tests covering: agent parser, agreement filter, conservative
# merge, Blackboard + 2-round protocol, captioner (7 strategies +
# fallback + delta guard), shard packager.
```

All tests pass without GPU dependencies — Phase 0R / PathomeOOD code
uses lazy imports for torch / vLLM / open_clip.

---

## Skipping legs

If you've already done step N for the same crops, just re-run from
step N+1. Each script does `git pull --ff-only` at start, so as long
as you `git push` between hosts the next step will pick up the
correct state.

```bash
# Re-validate only (re-pulls Nova's deltas, re-runs verifier):
CROPS=smoke bash scripts/sh_03_validate_local.sh

# Re-train encoder only (skip captions + shards if already built):
ssh tirtho@hpc-login.iastate.edu
PATHOME_SKIP_CAPTIONS=1 PATHOME_SKIP_SHARDS=1 \
  bash scripts/sh_04_train_encoder_nova.sh

# Re-run TabPFN matrix only (encoder features already cached):
PATHOME_SKIP_CAPTIONS=1 PATHOME_SKIP_FEATURES=1 \
  bash scripts/sh_05_tabpfn_local.sh

# Just re-aggregate tables (TabPFN results already on disk):
PATHOME_SKIP_CAPTIONS=1 PATHOME_SKIP_FEATURES=1 \
PATHOME_SKIP_TABPFN=1 PATHOME_SKIP_GRADCAM=1 \
  bash scripts/sh_05_tabpfn_local.sh
```

---

## Consuming the KB downstream

PathomeOOD reads `final_registry.json` directly. Other consumers can
do the same:

```python
from plantswarm.captioning import load_kb_profiles, caption_for_row

profiles = load_kb_profiles("artifacts/pathome_kb", crop_filter=["Tomato"])
# dict[(crop, disease) -> disease_record from final_registry.json]

caption, used_kb = caption_for_row(
    crop="Tomato", disease="Early Blight", state="CA",
    profiles=profiles, strategy="canonical_deltas_3",
)
# multi-sentence text combining canonical summary, diagnostic features,
# look-alikes, and the top-3 regional deltas for the given state.
```

---

## Citation

See `CITATION.cff`.
