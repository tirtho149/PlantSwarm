# Google Colab Setup Guide

Run PlantSwarm + OBSERVE on Google Colab (free GPU access).

---

## Quick Start (5 minutes)

### Option 1: Direct Colab Link
Open this link in your browser:
```
https://colab.research.google.com/github/tirtho149/PlantSwarm/blob/main/notebooks/plantswarm_colab.ipynb
```

Then click **"Run all"** to start training.

### Option 2: VS Code + Colab Extension
For seamless notebook editing in VS Code:

1. **Install VS Code Extension**
   - Open VS Code
   - Extensions → Search "Colab" → Install "Colab" by Google
   - OR: Command Palette (⌘ + Shift + P) → `ext install googlecolab.colab`

2. **Open Notebook**
   - File → Open Folder → Select `PlantSwarm`
   - Open `notebooks/plantswarm_colab.ipynb`
   - Click "Open in Colab" button (top of editor)
   - Automatically opens in browser with VS Code sync

3. **Edit in VS Code, Execute in Colab**
   - Make edits in VS Code
   - Save (⌘ + S)
   - Switch to Colab browser tab to see updates
   - Execute cells in Colab
   - Results appear in VS Code

---

## Full Colab Workflow

### Setup (First Cell)
```python
# Mount Google Drive (for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -r requirements.txt
!pip install -r requirements-tfds.txt

# Clone repository (if not using notebook link)
!git clone https://github.com/tirtho149/PlantSwarm.git
%cd PlantSwarm
```

### Phase 1: Generate Routing Traces (12-18 hours)
```python
# Run on 10,000 PlantVillage images
!python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml
```

**Important:** Colab sessions timeout after 12 hours. Use checkpointing:
```python
# Run on subset first (5 hours)
!python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 5000
```

### Phase 3: Train OBSERVE (4-6 hours)
```python
# Verify traces exist
!ls -lh results/plant_village_tfds/traces/plantswarm_traces.jsonl

# Train OBSERVE
!python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output observe/checkpoints/observe_final.pt \
  --epochs 50 --batch-size 8 --device cuda
```

### Phase 4: OOD Evaluation (2-3 hours)
```python
# Evaluate on PlantWild
!python scripts/run_plantswarm.py --config configs/plantwild_hf.yaml --subset 5000
```

### Save Results to Drive
```python
# Copy results to persistent Google Drive storage
!cp -r results/ /content/drive/MyDrive/PlantSwarm_Results/
!cp -r observe/checkpoints/ /content/drive/MyDrive/PlantSwarm_Models/

print("✓ Results saved to Google Drive")
```

---

## Colab Free vs Pro

| Feature | Free | Pro ($10/mo) |
|---------|------|--------------|
| **GPU** | T4 (15GB) | V100/A100 (40GB) |
| **Session Time** | 12 hours max | 24 hours max |
| **Storage** | 5GB | 100GB |
| **Recommended For** | Testing, Phase 3-4 | Phase 1-5 full pipeline |

**Free Tier Recommendation:**
- Phase 1: Run locally or Nova (12-18h, exceeds 12h limit)
- Phase 3: Run on Colab Pro (4-6h fits in Pro session)
- Phase 4: Run on Colab Free (2-3h, within limit)

---

## Troubleshooting Colab

### Memory Issues
```python
# Reduce batch size
!python scripts/train_observe.py ... --batch-size 4

# Or reduce epochs for testing
!python scripts/train_observe.py ... --epochs 10
```

### Session Timeout
```python
# Save checkpoint frequently
import shutil
!cp observe/checkpoints/observe_final.pt /content/drive/MyDrive/observe_backup.pt
```

### GPU Not Available
```python
# Check GPU status
!nvidia-smi

# If no GPU: Runtime → Change runtime type → GPU (T4 or V100)
```

### TFDS Download Slow
```python
# Use smaller subset
!python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 1000

# Or skip TFDS, use HF directly
!python scripts/run_plantswarm.py --config configs/plantwild_hf.yaml
```

---

## VS Code Colab Extension Features

| Feature | Shortcut |
|---------|----------|
| Open in Colab | Click "Open in Colab" button |
| Sync edits | Auto-sync (save in VS Code) |
| Execute cell | Shift + Enter (in Colab) |
| View output | See in Colab browser tab |
| Download notebook | Right-click → Download |
| Push to GitHub | `git add` + `git commit` + `git push` |

---

## Complete Colab Notebook Cells

### Cell 1: Setup
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -r requirements.txt
!pip install -r requirements-tfds.txt

import os
os.chdir('/content/PlantSwarm')
```

### Cell 2: Phase 1 (Subset)
```python
!python scripts/run_plantswarm.py \
  --config configs/plant_village_tfds.yaml \
  --subset 5000
```

### Cell 3: Phase 3 - OBSERVE Training
```python
!python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output observe/checkpoints/observe_final.pt \
  --epochs 50 --batch-size 8 --device cuda
```

### Cell 4: Evaluation
```python
!python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output results/plant_village_tfds/observe_evaluation.json
```

### Cell 5: Save Results
```python
import shutil
shutil.copytree('results/', '/content/drive/MyDrive/PlantSwarm_Results', dirs_exist_ok=True)
shutil.copytree('observe/checkpoints/', '/content/drive/MyDrive/PlantSwarm_Models', dirs_exist_ok=True)
print("✓ Results saved to Google Drive")
```

---

## Tips for Best Performance

1. **Use Pro for Phase 1** (12-18h session time needed)
2. **Free tier for Phase 3-4** (fit within 12h limit)
3. **Save to Drive frequently** (persistent storage across sessions)
4. **Use VS Code extension** (better editor than Colab's default)
5. **Subset data for testing** (reduce iteration time)
6. **Monitor GPU memory** (`!nvidia-smi` after each phase)

---

**Last Updated:** May 2, 2026
