# Two-Way Sync Guide: Local ↔ GitHub ↔ Nova

Complete guide for syncing code and results bidirectionally between your local machine, GitHub, and Nova HPC.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Local Machine (Mac)                      │
│  • Edit code                                                     │
│  • Run tests                                                     │
│  • View results                                                  │
└────────────────┬────────────────────────────────────────────────┘
                 │ git push/pull
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Repository                           │
│  • Central code repository                                       │
│  • Single source of truth                                        │
│  • Stores all commits and history                                │
└────────────────┬────────────────────────────────────────────────┘
                 │ git pull/push
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│              Nova HPC Cluster (ISU)                              │
│  • Run experiments (GPU intensive)                               │
│  • Store results locally                                         │
│  • Push results back to GitHub                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 DataLoader Integration & Syncing

DataLoader.py is now integrated into the pipeline. Sync and use 30+ datasets:

### Using DataLoader Datasets

**Local machine:**
```bash
# Run with DataLoader config (30+ datasets)
python scripts/run_plantswarm.py --config configs/dataloader_example.yaml
```

**Nova HPC:**
```bash
# Pull latest code (includes DataLoader)
git pull origin main

# Submit DataLoader experiment
sbatch scripts/submit_phase1_plantswarm.sh  # Or custom DataLoader job
```

### Syncing DataLoader Results

**Nova → GitHub:**
```bash
# After DataLoader job completes
cd /work/mech-ai/tirtho/ObservePlantSwarm
git add results/dataloader_experiment/
git commit -m "DataLoader results: [dataset-name]"
git push origin main
```

**GitHub → Local:**
```bash
# Retrieve results
git pull origin main
cat results/dataloader_experiment/plantswarm_metrics.json
```

---

## Prerequisites

### Local Machine
```bash
# Install Git
brew install git

# Configure Git (if not done)
git config --global user.name "Your Name"
git config --global user.email "your.email@iastate.edu"

# Verify
git config --global --list
```

### GitHub
```bash
# Create free account: https://github.com/signup
# Create repository: https://github.com/new
# Name: ObservePlantSwarm
# Visibility: Public (easier) or Private
```

### Nova HPC
```bash
# SSH access configured
# HuggingFace CLI installed
# Python 3.10+ available
```

---

## Phase 0: Initial Setup (One-Time)

### 0.1 Create Local Repository
```bash
# On your Mac
cd ~/Desktop
git init ObservePlantSwarm
cd ObservePlantSwarm

# Copy all code files from this directory

# Initialize git
git add .
git commit -m "Initial commit: PlantSwarm + OBSERVE implementation"
```

### 0.2 Connect to GitHub
```bash
# Create repo on GitHub (https://github.com/new)
# Then:

git remote add origin https://github.com/yourusername/ObservePlantSwarm.git
git branch -M main
git push -u origin main

# Verify
git remote -v
# Should show: origin https://github.com/yourusername/ObservePlantSwarm.git
```

### 0.3 Clone to Nova
```bash
# On Nova login node
ssh tirtho@hpc-login.iastate.edu

cd /work/mech-ai/tirtho/
git clone https://github.com/yourusername/ObservePlantSwarm.git
cd ObservePlantSwarm

# Verify
git remote -v
# Should show: origin https://github.com/yourusername/ObservePlantSwarm.git
```

---

## Phase 1: Push Code to GitHub (Local → GitHub)

### Before Experiments
When you make changes on your local machine and want to share with Nova:

```bash
# 1. See what changed
cd ~/Desktop/ObservePlantSwarm
git status

# 2. Stage all changes
git add -A

# 3. Commit with descriptive message
git commit -m "Description of changes"
# Example: "Add OBSERVE training script"
# Example: "Fix bug in routing logic"
# Example: "Update configs for Nova"

# 4. Push to GitHub
git push origin main

# Verify
git log --oneline -5  # See last 5 commits
```

### Commit Message Best Practices
```bash
# GOOD commit messages:
git commit -m "Add OBSERVE inference module with uncertainty decomposition"
git commit -m "Fix routing logic in severity agent"
git commit -m "Update SLURM config for Nova A100"
git commit -m "Add LaTeX sync script for paper metrics"

# BAD commit messages:
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
```

---

## Phase 2: Pull Code on Nova (GitHub → Nova)

### Before Running Experiments
```bash
# On Nova login node
cd /work/mech-ai/tirtho/ObservePlantSwarm

# 1. Get latest code from GitHub
git pull origin main

# 2. Verify you have all files
ls -la scripts/submit_*.sh
ls -la observe/
ls -la README.md

# 3. Setup environment
module load python cuda/11.8
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Ready to run!
squeue -u $USER  # Check existing jobs
```

### What If There Are Conflicts?
```bash
# If you see "CONFLICT" when pulling:
git status  # See conflicted files

# Option A: Keep GitHub version (recommended)
git checkout --theirs .

# Option B: Keep your local version
git checkout --ours .

# Then:
git add -A
git commit -m "Resolved merge conflicts"
```

---

## Phase 3: Run Experiments on Nova

### Before Submitting Jobs
```bash
# 1. Make sure code is synced
git pull origin main

# 2. Create logs directory
mkdir -p logs

# 3. For dataset download
sbatch scripts/submit_setup_plantwild.sh
```

### SLURM Output Files
Each job creates two files:
- **`.out` file** → Standard output (progress, results)
- **`.err` file** → Error messages and warnings

Example:
```bash
logs/phase1_plantswarm-12345.out   # Progress output
logs/phase1_plantswarm-12345.err   # Any errors
logs/phase2_experiments-12346.out
logs/phase2_experiments-12346.err
# ... etc for all phases
```

### Monitor Output in Real-Time
```bash
# Watch as job runs
tail -f logs/phase1_plantswarm-12345.out

# See last 100 lines
tail -100 logs/phase1_plantswarm-12345.out

# Search for errors
grep -i "error" logs/phase1_plantswarm-12345.err
grep -i "error" logs/*.err  # All errors

# See full log
cat logs/phase1_plantswarm-12345.out | less
```

### Submit All Phases
```bash
# Download dataset first
sbatch scripts/submit_setup_plantwild.sh

# Check it completes
tail -f logs/setup_plantwild-*.out

# Then run pipeline
bash scripts/submit_all_phases.sh

# Monitor all jobs
watch -n 5 "squeue -u $USER"  # Update every 5 sec
```

---

## Phase 4: Push Results Back to GitHub (Nova → GitHub)

### After Experiments Complete

```bash
# On Nova
cd /work/mech-ai/tirtho/ObservePlantSwarm

# 1. Check what changed
git status
# Should show:
#   results/
#   observe/checkpoints/
#   plantswarm/latex/auto/
#   logs/

# 2. Stage results and logs
git add results/
git add observe/checkpoints/
git add plantswarm/latex/auto/
git add logs/  # IMPORTANT: Include error logs!

# 3. Commit with details
git commit -m "Pipeline results: Phase 1-5 complete

- Phase 1: PlantSwarm on 10K PlantVillage images (12h)
- Phase 2: Baselines, ablations, calibration (2h)
- Phase 3: OBSERVE training on 8K traces (5h)
- Phase 4: OOD evaluation on PlantWild (2h)
- Phase 5: LaTeX sync complete

Metrics:
- PlantSwarm T3 F1: 89.2%
- OBSERVE OOD ECE: 0.16 (52% improvement)
- All logs in logs/ directory

Job IDs: Phase1=12345, Phase2=12346, Phase3=12347"

# 4. Push to GitHub
git push origin main

# Verify
git log --oneline -3  # See recent commits
```

### What to Commit
**YES - Commit These:**
- ✅ `results/plantswarm_metrics.json` — Metrics
- ✅ `results/*/baseline_results.json` — Experiment results
- ✅ `observe/checkpoints/observe_final.pt` — Trained model
- ✅ `plantswarm/latex/auto_*.tex` — Synced paper tables
- ✅ `logs/phase*.out` — Execution logs
- ✅ `logs/phase*.err` — Error logs

**NO - Don't Commit These:**
- ❌ `results/plant_village_tfds/plantswarm_predictions.jsonl` — Too large (~500MB)
- ❌ `data/PlantVillage/` — Dataset (auto-downloaded)
- ❌ `data/PlantWild/` — Dataset (auto-downloaded)
- ❌ `.venv/` — Virtual environment

### If Results Are Too Large
```bash
# Use .gitignore to exclude large files
cat >> .gitignore << 'EOF'
# Datasets (auto-downloaded)
data/PlantVillage/
data/PlantWild/
data/tfds_cache/

# Large result files
results/**/plantswarm_predictions.jsonl
results/**/plantswarm_predictions_detailed.jsonl

# Virtual environment
.venv/
EOF

# Then commit gitignore
git add .gitignore
git commit -m "Add gitignore for large files and datasets"
git push origin main
```

---

## Phase 5: Pull Results on Local Machine (GitHub → Local)

### Get Experiments Results
```bash
# On your Mac
cd ~/Desktop/ObservePlantSwarm

# 1. Pull from GitHub
git pull origin main

# 2. Check results
ls -lh results/plant_village_tfds/
ls -lh observe/checkpoints/
ls -lh plantswarm/latex/auto/

# 3. View logs (from Nova)
cat logs/phase1_plantswarm-*.out | tail -50
cat logs/phase1_plantswarm-*.err  # Check for errors

# 4. Compile paper with synced metrics
cd plantswarm/latex
latexmk -pdf acl_latex.tex
open acl_latex.pdf  # View on Mac
```

### Analyze Results Locally
```bash
# View metrics
python << 'EOF'
import json

with open('results/plant_village_tfds/plantswarm_metrics.json') as f:
    metrics = json.load(f)
    
print("PlantSwarm Results:")
for task, data in metrics.items():
    if task.startswith('T'):
        print(f"{task}: F1={data.get('macro_f1', 'N/A'):.3f}, ECE={data.get('ece', 'N/A'):.4f}")

with open('observe/checkpoints/training_history.json') as f:
    history = json.load(f)
    
print(f"\nOBSERVE Training: {history['epochs'][-1]} epochs")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
EOF
```

---

## Two-Way Sync Workflow Summary

### Daily Workflow
```bash
# ┌─────────────────────────────────────────────────┐
# │ Morning: Prepare code on local machine          │
# └─────────────────────────────────────────────────┘
cd ~/Desktop/ObservePlantSwarm
git add -A
git commit -m "Update config for tomorrow's experiments"
git push origin main

# ┌─────────────────────────────────────────────────┐
# │ Afternoon: Pull on Nova and run experiments     │
# └─────────────────────────────────────────────────┘
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai/tirtho/ObservePlantSwarm
git pull origin main
bash scripts/submit_all_phases.sh

# ┌─────────────────────────────────────────────────┐
# │ Evening: Monitor jobs and check logs            │
# └─────────────────────────────────────────────────┘
squeue -u $USER
tail -f logs/phase*.out
grep -i error logs/*.err

# ┌─────────────────────────────────────────────────┐
# │ Next Morning: Results ready, push to GitHub     │
# └─────────────────────────────────────────────────┘
git add results/ observe/checkpoints/ plantswarm/latex/auto/ logs/
git commit -m "Results from Phase 1-5 pipeline"
git push origin main

# ┌─────────────────────────────────────────────────┐
# │ Pull on local and analyze                       │
# └─────────────────────────────────────────────────┘
cd ~/Desktop/ObservePlantSwarm
git pull origin main
# View results, compile paper, etc.
```

---

## Error Handling and Logging

### All Errors Captured
Every SLURM job creates error logs:

```bash
# View errors from specific job
cat logs/phase3_observe_training-12347.err

# Find all errors across all phases
grep -r "error\|Error\|ERROR" logs/

# Count errors per phase
grep -l "error" logs/phase*.err | while read f; do echo "$f:"; grep "error" "$f" | wc -l; done
```

### Common Error Patterns and Fixes

#### Out of Memory
```bash
# Error appears in .err file: "CUDA out of memory"
# Fix: Reduce batch size in script

# In Nova terminal:
cd /work/mech-ai/tirtho/ObservePlantSwarm
git pull origin main

# Edit the script
nano scripts/submit_phase3_observe_training.sh
# Change: --batch-size 8
# To:     --batch-size 4

# Commit and push
git add scripts/submit_phase3_observe_training.sh
git commit -m "Reduce batch size to 4 for OOM fix"
git push origin main

# Resubmit
sbatch scripts/submit_phase3_observe_training.sh
```

#### Dataset Not Found
```bash
# Error: "PlantWild not found at data/PlantWild"
# Check logs:
cat logs/phase4_ood_eval-12348.err

# On Nova:
ls -lh data/PlantWild/ | wc -l
# If empty or missing, resubmit dataset download:
sbatch scripts/submit_setup_plantwild.sh
```

### Push Error Logs Back to GitHub
```bash
# Always include logs in commits:
git add logs/
git commit -m "Add logs from failed run

Errors in phase3:
- See logs/phase3_observe_training-12347.err
- Issue: CUDA out of memory
- Fix: Reduced batch size to 4"
git push origin main
```

---

## Best Practices

### 1. Commit Frequently
```bash
# Don't wait to commit - do it after each change
git add -A && git commit -m "Description"
git push origin main
```

### 2. Pull Before Working
```bash
# Always start by pulling latest
git pull origin main
# Prevents conflicts
```

### 3. Clear Commit Messages
```bash
# Good: Describes what and why
git commit -m "Fix routing logic in PathogenAgent

PathogenAgent was incorrectly setting all_tasks_covered=True,
causing premature routing to DiagnosisAgent. Changed to False
since T4 and T5 are not covered at this stage.

Affects: agents/pathogen_agent.py
Tests: run_plantswarm.py passes with this fix"

# Bad: Vague
git commit -m "fix bug"
```

### 4. Include Logs with Results
```bash
# Always push logs with results
git add results/ logs/
git commit -m "Pipeline results + logs"
git push origin main
```

### 5. Branch for Major Changes (Optional)
```bash
# For experimental changes, use branches
git checkout -b experiment/new-feature
# Make changes
git add -A && git commit -m "Experimental feature X"
git push origin experiment/new-feature

# When ready, merge to main
git checkout main
git pull origin main
git merge experiment/new-feature
git push origin main
```

---

## Troubleshooting Sync Issues

### Lost Changes (Not Committed)
```bash
# See what you lost
git log --oneline -5

# Restore a previous version
git checkout <commit-hash> -- <filename>

# Or restore everything to last commit
git reset --hard HEAD
```

### Merge Conflicts
```bash
# Pull shows conflicts
git status  # See what conflicts

# Accept GitHub version (recommended)
git checkout --theirs .
git add -A
git commit -m "Resolved conflicts by accepting GitHub version"

# OR accept local version
git checkout --ours .
git add -A
git commit -m "Resolved conflicts by keeping local version"

git push origin main
```

### Accidental Push
```bash
# If you pushed something wrong, revert
git revert <commit-hash>
git push origin main

# Or if not yet pushed
git reset --soft HEAD~1
# Make changes
git commit -m "Fixed version"
git push origin main
```

### Large Files Blocking Push
```bash
# Remove file from history
git rm --cached <large-file>
git commit -m "Remove large file"

# Add to .gitignore
echo "<large-file>" >> .gitignore
git add .gitignore
git commit -m "Add large file to gitignore"

git push origin main
```

---

## Complete Workflow Checklist

### Before Submitting Jobs
- [ ] Latest code pulled from GitHub: `git pull origin main`
- [ ] Code committed locally: `git status` shows clean
- [ ] Environment activated: `source .venv/bin/activate`
- [ ] Dataset setup if needed: `sbatch scripts/submit_setup_plantwild.sh`

### During Experiments
- [ ] Logs being written: `ls -lh logs/`
- [ ] Monitor errors: `grep error logs/*.err`
- [ ] Check progress: `tail -f logs/phase*.out`
- [ ] Track job IDs: `squeue -u $USER`

### After Experiments
- [ ] All phases completed: `squeue -u $USER` shows no jobs
- [ ] Results exist: `ls results/plant_village_tfds/`
- [ ] Model saved: `ls observe/checkpoints/observe_final.pt`
- [ ] LaTeX synced: `ls plantswarm/latex/auto/`
- [ ] Errors reviewed: `cat logs/*.err`

### Before Pushing
- [ ] Add all results: `git add results/ observe/ plantswarm/latex/auto/ logs/`
- [ ] Verify commit: `git status`
- [ ] Meaningful message: `git commit -m "..."`
- [ ] Push to GitHub: `git push origin main`

### On Local Machine
- [ ] Pull results: `git pull origin main`
- [ ] Review metrics: Check `results/` files
- [ ] Check logs: `cat logs/*.out`
- [ ] Compile paper: `latexmk -pdf acl_latex.tex`

---

## Summary: Quick Reference

```bash
# LOCAL → GITHUB (Push code)
cd ~/Desktop/ObservePlantSwarm
git add -A && git commit -m "Your message"
git push origin main

# GITHUB → NOVA (Pull code)
cd /work/mech-ai/tirtho/ObservePlantSwarm
git pull origin main

# Run experiments
bash scripts/submit_all_phases.sh

# NOVA → GITHUB (Push results)
git add results/ observe/ plantswarm/latex/auto/ logs/
git commit -m "Results from experiments"
git push origin main

# GITHUB → LOCAL (Pull results)
cd ~/Desktop/ObservePlantSwarm
git pull origin main
```

---

**Last Updated:** 2026-05-01
