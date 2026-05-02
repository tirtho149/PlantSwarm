# PlantSwarm: Multi-Agent VLM Swarm for Plant Disease Diagnosis

**Paper:** *Why Ask When You Can Observe? A Vision-Language-Action Model for Epistemic Action Selection in Multi-Agent Crop Disease Diagnosis* (EMNLP 2026)

**Core Contribution:** PlantSwarm establishes that **routing behavior** (path length, backtrack decisions, contradiction events) predicts correctness far better than **self-declared confidence** in multi-agent VLM systems. OBSERVE operationalizes this as the first Vision-Language-Action model trained on routing traces, achieving 52% calibration improvement under domain shift with 6× lower inference cost.

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.10+
- NVIDIA GPU (for vLLM inference)
- 50GB disk (for TFDS Plant Village cache)

### Install & Test
```bash
# 1. Clone and setup
cd ObservePlantSwarm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install TFDS support (for Plant Village)
pip install -r requirements-tfds.txt

# 3. Verify imports
python -c "from agents import *; from plantswarm import *; print('✓ Ready')"
```

---

## 📋 Complete Workflow (5 Phases)

1. **Phase 1:** Generate PlantSwarm routing traces on PlantVillage
2. **Phase 2:** Run experimental comparisons (baselines, ablations, calibration, bias)
3. **Phase 3:** Train OBSERVE model on routing traces
4. **Phase 4:** Evaluate on PlantWild (OOD)
5. **Phase 5:** Build paper with auto-synced metrics

---

## ☁️ Google Colab Setup (Free GPU)

Run on Google Colab for free GPU access (T4 15GB) or upgrade to Pro for faster GPUs (V100/A100).

### Quick Start
```bash
# Option 1: Direct Colab Link
# Open in browser:
https://colab.research.google.com/github/tirtho149/PlantSwarm/blob/main/notebooks/plantswarm_colab.ipynb

# Option 2: VS Code + Colab Extension (Recommended)
# 1. Install: Extensions → Search "Colab" → Install "Colab" by Google
# 2. Open notebook in VS Code
# 3. Click "Open in Colab" button
# 4. Edit in VS Code, execute in Colab browser tab
```

**See [COLAB_SETUP.md](COLAB_SETUP.md) for full details, free vs Pro comparison, and troubleshooting.**

---

## 🖥️ First-Time Nova HPC Setup

### Step 1: SSH Access
```bash
# On your local machine
ssh tirtho@hpc-login.iastate.edu

# You'll be prompted for your ISU credentials
# If this is your first time, contact HPC support for account activation
```

### Step 2: Clone Repository
```bash
# On Nova login node
cd /work/mech-ai/tirtho/
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm
```

### Step 3: Load Modules & Create Virtual Environment
```bash
# Load required modules
module load python cuda/11.8

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Verify Python
python --version  # Should be 3.10+
```

### Step 4: Install Dependencies
```bash
# Install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install TFDS support for PlantVillage
pip install -r requirements-tfds.txt

# Verify installation
python -c "from agents import *; from plantswarm import *; print('✓ Ready')"
```

### Step 5: Create Required Directories
```bash
# Create directory structure
mkdir -p logs data results observe/checkpoints

# Set proper permissions
chmod -R 755 logs data results observe
```

### Step 6: Configure Git (One-time)
```bash
# Set up Git credentials for syncing
git config --global user.name "Your Name"
git config --global user.email "your.email@iastate.edu"

# Verify
git config --global --list
```

### Verification Checklist
```bash
# Verify all setup
python -c "import tensorflow_datasets; print('✓ TFDS')"
python -c "import torch; print('✓ PyTorch')"
python -c "from observe import OBSERVE; print('✓ OBSERVE')"
sbatch --version  # Should show SLURM version
```

---

## 🚀 Running on Nova HPC (SLURM Scripts)

For distributed GPU training on Nova HPC cluster, use the provided shell scripts. All scripts use SLURM with automatic dependency chaining.

### Quick Start: Run All 5 Phases (30 hours total)

```bash
# On Nova HPC login node
cd /work/mech-ai/tirtho/ObservePlantSwarm

# Step 1: Download PlantWild dataset (2-4 hours, one-time only)
sbatch scripts/submit_setup_plantwild.sh
# Monitor: tail -f logs/setup_plantwild-*.out

# Step 2: Submit all 5 phases with automatic dependency chaining
bash scripts/submit_all_phases.sh

# Step 3: Monitor jobs
squeue -u $USER
tail -f logs/phase*.out
```

### Individual Phase Scripts

#### Phase 1: Generate Routing Traces (12-18 hours)
```bash
sbatch scripts/submit_phase1_plantswarm.sh
# Generates: results/plant_village_tfds/traces/plantswarm_traces.jsonl
# Log: logs/phase1_plantswarm-{JOBID}.out
```

##### Phase 2: Experiments (2-3 hours)
```bash
sbatch scripts/submit_phase2_experiments.sh
# Runs all 4 sub-phases sequentially:
#   2a) Baselines (8 comparison models)
#   2b) Ablations (6 factorial variants)
#   2c) Calibration (ECE, temperature scaling, conformal)
#   2d) Routing Analysis (P1-P4 predictions)
# Log: logs/phase2_experiments-{JOBID}.out
```

#### Phase 3: Train OBSERVE (4-6 hours)
```bash
sbatch scripts/submit_phase3_observe_training.sh
# Outputs: observe/checkpoints/observe_final.pt
# Log: logs/phase3_observe_training-{JOBID}.out
```

#### Phase 4: OOD Evaluation (2-3 hours)
```bash
sbatch scripts/submit_phase4_ood_evaluation.sh
# Evaluates PlantSwarm + OBSERVE on PlantWild
# Log: logs/phase4_ood_eval-{JOBID}.out
```

#### Phase 5: LaTeX Sync (<1 minute)
```bash
sbatch scripts/submit_phase5_latex_sync.sh
# Syncs metrics to: plantswarm/latex/auto_*.tex
# Log: logs/phase5_latex_sync-{JOBID}.out
```

### Monitoring Jobs

```bash
# Check all your jobs
squeue -u $USER

# Monitor specific phase output (live)
tail -f logs/phase1_plantswarm-*.out

# Check for errors
tail -f logs/phase1_plantswarm-*.err

# See completed job info
sacct -j <JOBID>
```

### Output & Syncing Results

After jobs complete, sync results back to GitHub:

```bash
# On Nova HPC
git add results/ observe/checkpoints/ plantswarm/latex/auto_* logs/
git commit -m "Full pipeline results from Nova HPC"
git push origin main

# On local machine
git pull origin main
```

---

## 🔄 Two-Way Sync Workflow (Local ↔ GitHub ↔ Nova)

### Workflow Overview
```
Local Machine ──→ GitHub (code) ──→ Nova HPC (run jobs)
Local Machine ←── GitHub (results) ←── Nova HPC (push results)
```

### Step 1: Push Code Changes (Local → GitHub)

After making code changes locally:

```bash
# On local machine
git status                          # See changes
git add <file1> <file2> ...        # Stage specific files
git commit -m "Description of changes"
git push origin main               # Push to GitHub
```

### Step 2: Pull Code on Nova (GitHub → Nova)

Before running jobs on Nova:

```bash
# On Nova HPC
cd /work/mech-ai/tirtho/ObservePlantSwarm
git fetch origin                   # Get latest from GitHub
git pull origin main               # Update local clone
```

### Step 3: Submit Jobs and Wait

```bash
# On Nova HPC
bash scripts/submit_all_phases.sh  # Start all 5 phases
squeue -u $USER                    # Monitor jobs
tail -f logs/phase1_plantswarm-*.out  # Watch progress
```

### Step 4: Push Results Back (Nova → GitHub)

After jobs complete:

```bash
# On Nova HPC (in ObservePlantSwarm directory)
git add results/ observe/checkpoints/ plantswarm/latex/auto_* logs/
git commit -m "Results: Phase 1-5 pipeline (X hours, Y% accuracy)"
git push origin main               # Push results to GitHub
```

### Step 5: Pull Results Locally (GitHub → Local)

To get results on your local machine:

```bash
# On local machine
git fetch origin                   # Get latest from GitHub
git pull origin main               # Update with results
ls -lh results/plant_village_tfds/ # Check results downloaded
```

### Full Daily Workflow Example

```bash
# DAY 1: Local Development
# ========================
# On local machine (~/Desktop/ObservePlantSwarm)
git add configs/plant_village_tfds.yaml
git commit -m "Adjust temperature scaling parameters"
git push origin main

# DAY 1: Nova Setup & Submit
# ===========================
# SSH to Nova login node
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai/tirtho/ObservePlantSwarm
git pull origin main               # Get latest code
sbatch scripts/submit_setup_plantwild.sh  # One-time dataset download
# Wait 2-4 hours...

# DAY 2: Nova Pipeline Submission
# ================================
bash scripts/submit_all_phases.sh  # Submit all 5 phases (~30 hours)
squeue -u $USER                    # Track progress

# DAY 3-4: Nova Results Sync
# ==========================
# After jobs finish (30 hours later)
git add results/ observe/checkpoints/ plantswarm/latex/auto_* logs/
git commit -m "Full pipeline: 92.3% F1, 0.08 ECE on PlantVillage"
git push origin main

# DAY 4: Local Results Retrieval
# ===============================
# On local machine
git pull origin main
# Review results in results/plant_village_tfds/plantswarm_metrics.json
cat results/plant_village_tfds/plantswarm_metrics.json | python -m json.tool
```

### Common Sync Issues

**Issue:** "Your branch is ahead of origin"
```bash
# You have commits locally that aren't pushed
git push origin main
```

**Issue:** "Your branch is behind origin"
```bash
# Nova has pushed results you haven't pulled
git pull origin main
```

**Issue:** Merge conflict after pulling
```bash
# Edit the conflicted files manually
git add <resolved_files>
git commit -m "Resolve merge conflict"
git push origin main
```

**Issue:** Want to discard local changes and use GitHub version
```bash
git fetch origin
git reset --hard origin/main
```

### Best Practices

✅ **Do:**
- Commit frequently with clear messages
- Push after each significant change
- Pull before starting new work
- Include job logs (.out, .err) in results commits

❌ **Don't:**
- Push large binary files directly (except trained models)
- Force push to main (`git push --force`)
- Edit files in parallel on local + Nova without syncing
- Keep uncommitted changes for >1 day

---

### Phase 1: Generate Routing Traces (Training Data)

**Goal:** Run PlantSwarm on PlantVillage (~10,000 images) to generate routing traces for OBSERVE training.

#### Step 1: Submit PlantSwarm Job (Nova HPC)
```bash
# Submit Phase 1 job
sbatch scripts/submit_phase1_plantswarm.sh

# Monitor progress
tail -f logs/phase1_plantswarm-*.out
```

**Time:** 12-18 hours on single A100 GPU  
**Output:** `results/plant_village_tfds/`
- `plantswarm_metrics.json` — accuracy, ECE, TPCP metrics
- `plantswarm_predictions.jsonl` — per-image predictions
- `traces/plantswarm_traces.jsonl` — routing traces (training data for OBSERVE)

---

### Phase 2: Run Experimental Comparisons

#### Step 2a: Baselines (Single-agent, Fixed Chain, Debate, etc.)
```bash
python scripts/run_baselines.py --config configs/plant_village_tfds.yaml
```
Compares PlantSwarm against 8 baselines:
- Zero-shot single VLM
- Chain-of-thought
- Fixed chain (no routing)
- DeeR (two-stage exit)
- Multi-agent debate
- Random, Majority class

**Output:** `results/plant_village_tfds/baseline_results.json`

#### Step 2b: Ablations (Factorial Study)
```bash
python scripts/run_ablations.py --config configs/plant_village_tfds.yaml
```
Tests contribution of routing components (Table 3):
- Fixed Chain (baseline)
- +Context buffer
- +Free routing (no confidence gate)
- +Backtracking
- 3-agent swarm
- Full PlantSwarm

**Output:** `results/plant_village_tfds/ablation_metrics_*.json`

#### Step 2c: Calibration Analysis
Included in Phase 2 experiments script. Analyzes uncertainty quantification:
- ECE before/after temperature scaling
- Reliability diagrams
- Split conformal prediction
- κ calibration (confidence vs. correctness)

**Output:** `results/plant_village_tfds/calibration_report.json`

#### Step 2d: Routing Analysis
Included in Phase 2 experiments script. Tests falsifiable predictions (P1-P4):
- P1: Path length ↔ entropy correlation
- P2: Backtrack improves confidence
- P3: Early termination accuracy
- P4: OOD behavioral transfer

**Output:** `results/plant_village_tfds/routing_analysis.json`

---

### Phase 3: Train OBSERVE (Vision-Language-Action Model)

**Goal:** Train a lightweight epistemic action selector on PlantSwarm routing traces for 6× lower inference cost with 52% better calibration under domain shift.

#### Step 3a: Prepare Training Traces
Ensure you have routing traces from Phase 1:
```bash
# Check traces exist
ls -lh results/plant_village_tfds/traces/plantswarm_traces.jsonl
# Should contain 8,000-10,000 routing traces
```

#### Step 3b: Train OBSERVE Model (Nova HPC)
```bash
# Submit OBSERVE training job
sbatch scripts/submit_phase3_observe_training.sh

# Monitor progress
tail -f logs/phase3_observe_training-*.out
```

**Training Details:**
- **Time:** 4-6 hours on single A100 GPU
- **Architecture:** Qwen2.5-VL-3B with LoRA (r=16, α=32, ~56M trainable params)
- **Data:** 8,000-10,000 routing traces from PlantSwarm
- **Loss:** Weighted multi-task (routing 1.0 + calibration 0.4 + consistency 0.2 + belief 0.2)
- **Optimizer:** AdamW with lr=1e-4, warmup 500 steps, cosine decay

**Output:**
- `observe/checkpoints/observe_final.pt` — trained model weights
- `observe/checkpoints/training_history.json` — loss curves and metrics

#### Step 3c: Evaluate OBSERVE on PlantVillage (ID)
Evaluation runs automatically after training completes. Check results:
```bash
cat observe/checkpoints/training_history.json
cat results/plant_village_tfds/observe_evaluation.json
```

#### Step 3d: Inference with OBSERVE
```python
from observe import OBSERVEInference
from PIL import Image

# Load model
inference = OBSERVEInference("observe/checkpoints/observe_final.pt")

# Single image
image = Image.open("crop.jpg")
action = inference.predict(image, context_text="Prior observations: healthy leaf")

print(f"Next agent: {action.next_agent}")
print(f"Backtrack: {action.backtrack}")
print(f"Epistemic uncertainty: {action.epistemic_uncertainty:.3f}")
print(f"Aleatoric uncertainty: {action.aleatoric_uncertainty:.3f}")
print(f"Confidence: {action.confidence:.3f}")

# Get uncertainty decomposition with recommendations
decomp = inference.get_uncertainty_decomposition(action)
print(f"\nEpistemic: {decomp['epistemic']['recommendation']}")
print(f"Aleatoric: {decomp['aleatoric']['recommendation']}")

# Batch inference
images = [Image.open(f"crop_{i}.jpg") for i in range(10)]
actions = inference.predict_batch(images, batch_size=4)
```

---

### Phase 4: OOD Evaluation (PlantWild)

**Goal:** Evaluate PlantSwarm on wild (uncontrolled) images for domain shift assessment.

```bash
sbatch scripts/submit_phase4_ood_evaluation.sh
```

**Time:** 2-3 hours on single A100 GPU  
**Output:** `results/plantwild/`
- `plantswarm_metrics.json` — OOD accuracy, ECE (should be worse than PlantVillage)
- `traces/plantswarm_traces.jsonl` — routing traces for OBSERVE evaluation
- Validates robustness to controlled→wild domain shift

Evaluation automatically evaluates OBSERVE on PlantWild after PlantSwarm completes.  
Should show 52% ECE improvement over prompt-based baselines under domain shift.

---

### Phase 5: LaTeX Metrics Sync

**Goal:** Auto-sync all metrics to paper LaTeX files.

```bash
sbatch scripts/submit_phase5_latex_sync.sh
```

**Time:** <1 minute  
**Output:** Auto-generated TeX files synced to `plantswarm/latex/auto_*.tex`
- `auto_metrics.tex` — inline macro definitions (ECE, F1, etc.)
- `auto_table_main_results.tex` — Table 4 (PlantSwarm vs baselines)
- `auto_table_ablation_results.tex` — Table 3 (ablations)
- `auto_table_mechanisms.tex` — context buffer mechanisms

---

## 🔧 Configuration Guide

### configs/plant_village_tfds.yaml (Training)
```yaml
data:
  tfds_name: "plant_village"      # TensorFlow Datasets
  tfds_split: "train"
  tfds_max_examples: 10000        # ~54k available; use subset for testing
  image_col: "image_bytes"        # TFDS provides JPEG bytes

model:
  backbone: "Qwen/Qwen3-VL-8B-Instruct"
  vllm_base_url: "http://localhost:8000/v1"
  temperature: 0.0                # Deterministic routing

routing:
  orchestrator: "autogen_swarm"   # Microsoft AutoGen Swarm
  Tmax: 15                        # Max agents per image
  allow_backtrack: true

output:
  results_dir: "results/plant_village_tfds/"
  save_traces: true              # Required for OBSERVE training
```

### configs/plantwild_hf.yaml (OOD Evaluation)
```yaml
data:
  hf_dataset_id: "rashikahura/plantWild"  # HuggingFace dataset
  image_col: "image_bytes"
  n_images: 18000                 # Full wild dataset
```

---

## 📦 Advanced: DataLoader.py (30+ Datasets)

For comprehensive dataset curation across 30+ plant disease sources (Kaggle, Zenodo, HuggingFace):

**See [DATALOADER_GUIDE.md](DATALOADER_GUIDE.md)** for:
- Support for SBRD, MangoLeaf, BananaLeaf, Cucumber, PlantDoc, LeafNet, and 24+ more datasets
- Interactive sampling per class with stratification
- Excel multi-sheet reporting with validation
- Crop/disease name normalization across sources

**Note:** For PlantSwarm pipeline, use modular loaders:
- Training: `data/tfds_plant_village.py` (PlantVillage)
- OOD Eval: `data/plantwild_hf.py` (PlantWild)
- General: `data/loader.py` (unified dispatcher)

DataLoader.py is a legacy research tool for exploration and custom dataset integration.

---

## 📊 Understanding Results

### plantswarm_metrics.json
```json
{
  "T1": {"macro_f1": 87.5, "ece": 0.11, "tpcp": 720},
  "T2": {"macro_f1": 92.3, "ece": 0.08, "tpcp": 650},
  ...
  "by_benchmark": {
    "plantvillage": {"T2": {"macro_f1": 94.1}, "T3": {"macro_f1": 88.9}},
    "plantwild": {...}  // OOD results
  }
}
```

**Key Metrics:**
- `macro_f1`: F1 score (main accuracy metric)
- `ece`: Expected Calibration Error (0.0=perfect, 1.0=worst)
- `tpcp`: Tokens-per-correct-prediction (efficiency)

### plantswarm_traces.jsonl
Per-image routing trace (training data for OBSERVE):
```json
{
  "image_id": "plantvillage_00042",
  "path": ["MorphologyAgent", "SymptomAgent", "PathogenAgent", "SeverityAgent", "DiagnosisAgent"],
  "path_length": 5,
  "backtrack_count": 0,
  "early_terminated": false,
  "total_tokens": 2847,
  "final_predictions": {"T1": "Blight", "T2": "Fungal", "T3": "Late Blight", ...},
  "ground_truth": {"T1": "Blight", "T2": "Fungal", "T3": "Late Blight", ...}
}
```

---

## 🏗️ Architecture Overview

### 5-Agent Swarm (Routing Strategy)
```
MorphologyAgent (visual grounding only)
         ↓
SymptomAgent (T1: symptom classification)
         ↓
PathogenAgent (T2: pathogen, T3: disease name)
         ↓
SeverityAgent (T4: severity, T5: crop species)
         ↓
DiagnosisAgent (synthesis + final JSON)
```

**Routing Decisions (Algorithm 1):**
- **Low confidence + no backtrack:** → MorphologyAgent (regrounding)
- **High confidence + all tasks complete:** → DiagnosisAgent (early terminate)
- **Medium confidence or pending tasks:** → forward to next agent

### OBSERVE: Vision-Language-Action Model

**Architecture:**
```
Input: Image + Context Text
  ↓
Qwen2.5-VL-3B (frozen, 2.95B params)
  ↓
LoRA Adapter (r=16, α=32, ~50M trainable params)
  ↓
Shared Head (512-dim)
  ├→ Routing Head (5-class softmax)
  ├→ Backtrack Head (binary sigmoid)
  ├→ Epistemic Head (scalar ∈ [0,1])
  ├→ Aleatoric Head (scalar ∈ [0,1])
  ├→ Confidence Head (scalar ∈ [0,1])
  └→ Belief Text (autoregressive from decoder)
```

**Key Outputs:**
- **next_agent:** Which of 5 agents to route to next
- **backtrack:** Whether to backtrack to MorphologyAgent
- **epistemic_uncertainty:** Resolvable ambiguity (improved by more evidence)
- **aleatoric_uncertainty:** Irreducible difficulty (escalate to human)
- **confidence:** Calibrated confidence in prediction [0, 1]
- **belief_state:** Natural language belief about current situation

**Training:**
- **Data:** 8,000-10,000 routing traces from PlantSwarm
- **Loss:** Weighted multi-task (routing 1.0 + calibration 0.4 + consistency 0.2 + belief 0.2)
- **Optimizer:** AdamW with lr=1e-4
- **Time:** ~4-6 hours on single A100 GPU for 50 epochs
- **Hyperparams:** batch_size=8, weight_decay=0.01

**Performance:**
- **Cost:** 700 tokens vs 4,200 for full PlantSwarm (6× reduction)
- **ID Accuracy:** 92% on PlantVillage
- **OOD Calibration:** ECE 0.16 vs 0.33 for baselines (52% improvement)
- **Uncertainty Decomposition:** Actionable epistemic/aleatoric split with human escalation guidance

---

## 📦 Directory Structure

```
PlantSwarm/
├── agents/
│   ├── base_agent.py           # ABC for all agents
│   ├── morphology_agent.py     # Visual grounding
│   ├── symptom_agent.py        # T1
│   ├── pathogen_agent.py       # T2, T3
│   ├── severity_agent.py       # T4, T5
│   └── diagnosis_agent.py      # Synthesis
│
├── observe/                    # Vision-Language-Action model
│   ├── __init__.py             # Module exports
│   ├── model.py                # OBSERVE architecture + LoRA
│   ├── trainer.py              # Training pipeline
│   ├── inference.py            # Deployment/evaluation
│   └── checkpoints/            # Trained model weights (after training)
│
├── plantswarm/
│   ├── pipeline.py             # Core κ-routing orchestrator
│   ├── autogen_pipeline.py     # AutoGen Swarm runtime (default)
│   ├── entropy_pipeline.py     # Entropy-driven routing variant
│   └── latex/                  # Paper source + auto-generated tables
│
├── calibration/
│   ├── ensemble.py             # Confidence-weighted aggregation
│   ├── ece.py                  # Expected Calibration Error
│   ├── temperature_scaling.py  # Post-hoc calibration
│   └── conformal.py            # Prediction sets
│
├── utils/
│   ├── vllm_client.py          # OpenAI-compatible HTTP client
│   ├── metrics.py              # F1, TPCP, McNemar's test
│   ├── routing_trace.py        # Trace I/O & analysis
│   ├── sequence_entropy.py     # Token-level entropy
│   └── hedge_lexicon.py        # Uncertainty signals
│
├── data/
│   ├── loader.py               # Unified PlantDiagBenchLoader
│   ├── tfds_plant_village.py   # TFDS Plant Village backend
│   ├── plantwild_hf.py         # HuggingFace PlantWild backend
│   ├── directory_index.py      # Folder tree backend
│   └── stratifier.py           # Train/cal/test splits
│
├── baselines/                  # 8 baseline implementations
├── ablations/                  # 6 factorial ablation variants
├── bias/                       # Demographic parity analysis
├── scripts/
│   ├── run_plantswarm.py       # Main entry point
│   ├── run_baselines.py
│   ├── run_ablations.py
│   ├── run_calibration.py
│   ├── run_routing_analysis.py
│   ├── run_bias_analysis.py
│   ├── sync_latex_metrics.py   # JSON → LaTeX
│   └── build_latex_pdf.sh      # LaTeX → PDF
│
├── configs/                    # YAML experiment configs
├── setup.py
├── requirements.txt
└── README.md (this file)
```

---

## 🤖 OBSERVE Model Usage

### Training OBSERVE
```bash
# Submit training job (Phase 3)
sbatch scripts/submit_phase3_observe_training.sh

# Monitor progress
tail -f logs/phase3_observe_training-*.out

# Check training history after completion
cat observe/checkpoints/training_history.json
```

### Evaluating OBSERVE
```bash
# Evaluate on PlantVillage (ID) or PlantWild (OOD)
sbatch scripts/submit_evaluate_observe.sh

# Check results
cat results/plant_village_tfds/observe_evaluation.json
```

### Using OBSERVE for Inference (Python)
```python
from observe import OBSERVEInference
from PIL import Image

# Load trained model
inference = OBSERVEInference("observe/checkpoints/observe_final.pt")

# Single image prediction
image = Image.open("plant_crop.jpg")
context = "Symptoms: lesions on leaf"
action = inference.predict(image, context)

# Inspect results
print(f"Next agent: {action.next_agent}")
print(f"Confidence: {action.confidence:.3f}")
print(f"Epistemic uncertainty: {action.epistemic_uncertainty:.3f}")

# Get actionable recommendations
decomp = inference.get_uncertainty_decomposition(action)
print(decomp["epistemic"]["recommendation"])

# Batch inference
images = [Image.open(f"crop_{i}.jpg") for i in range(100)]
actions = inference.predict_batch(images, batch_size=4)
```

---

## 🧪 Troubleshooting

### vLLM Server Issues
```bash
# Check server is reachable
curl http://localhost:8000/v1/models

# Verify Qwen model loaded
# Output should include: "Qwen/Qwen3-VL-8B-Instruct"

# If memory error: reduce batch size or use smaller model
# In config: adjust image_size or use Qwen2.5-VL-7B
```

### TFDS Download Stuck
```bash
# Clear stale cache
rm -rf ~/tensorflow_datasets/plant_village

# Retry with subset
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 100
```

### LaTeX PDF Build Fails
```bash
# Install TeX Live (macOS)
brew install texlive

# Or use TinyTeX
curl -fsSL https://yihui.org/tinytex/install-bin-unix.sh | sh

# Then try build again
bash scripts/build_latex_pdf.sh --latex-dir plantswarm/latex/
```

### Out of Memory
```bash
# Run on smaller subset first
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 100

# Reduce calibration split
# In config: calibration_split_size: 100 (default 500)
```

---

## 📈 Performance Targets (Paper Results)

### PlantVillage (Controlled)
| Metric | T2 | T3 |
|--------|----|----|
| Macro-F1 | 92.3% | 88.9% |
| ECE | 0.08 | 0.11 |
| TPCP | 650 | 720 |

### PlantWild (OOD)
| Metric | T2 | T3 |
|--------|----|----|
| Macro-F1 | 85.1% | 79.4% |
| ECE | 0.16 | 0.19 |
| (52% ECE improvement vs. prompt-based baselines) |

---

## 📚 Citation

If you use this code, cite the paper:

```bibtex
@inproceedings{observe2026,
  title={Why Ask When You Can Observe? A Vision-Language-Action Model for Epistemic Action Selection in Multi-Agent Crop Disease Diagnosis},
  author={[Authors]},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

---

## 📝 License

This project is released under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

For bug reports, feature requests, or questions:
1. Check existing issues
2. Open a new issue with reproducible steps
3. For contributions, open a pull request

---

## 📞 Support

For questions about:
- **Paper/methodology:** See Section 2-6 of `plantswarm/latex/acl_latex.tex`
- **Code:** Check docstrings in respective modules
- **Results:** Refer to `results/*/` directories after running experiments

---

**Last Updated:** May 2026  
**Tested on:** Python 3.10+, CUDA 11.8+, vLLM 0.4+
