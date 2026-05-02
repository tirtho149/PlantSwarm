# DataLoader.py Guide

Comprehensive dataset curator for 30+ plant disease datasets (Kaggle, Zenodo, HuggingFace, TFDS).

---

## Overview

`DataLoader.py` is a unified pipeline that:
- Downloads datasets from Kaggle, Zenodo, HuggingFace, and TensorFlow Datasets
- Samples images per class with stratification
- Normalizes crop/disease names across sources
- Generates Excel reports with multi-sheet summaries
- Handles authentication (Kaggle credentials, HF tokens)

**Note:** For PlantSwarm + OBSERVE pipeline, use:
- Training: `data/tfds_plant_village.py` (PlantVillage via TFDS)
- OOD Eval: `data/plantwild_hf.py` (PlantWild via HuggingFace)
- General: `data/loader.py` (unified dispatcher)

DataLoader.py is a legacy comprehensive dataset manager for research/exploration.

---

## Supported Datasets (30+)

### Kaggle Datasets
- SBRD, MangoLeaf, SoybeanPNAS, BeanLeaf, YellowRust
- BananaLeaf, Lettuce, Cucumber, DurianLeaf, EggplantDisease
- StrawberryDiseaseDetection, VanillaDisease, SugarLeafIDN
- Cauliflower, NewPlantDiseases, FUSARIUM22
- RadyPlantDiseases, A2H0H0R1PlantDisease, AvinashPlantDisease
- SakethPlantDisease, VQAPlantDisease, BDCropVegetable

### Online Sources
- **PlantDoc** (GitHub)
- **PlantVillage** (TensorFlow Datasets)
- **PlantWild** (HuggingFace)
- **LeafNet** (HuggingFace)
- **CucumberZenodo** (Zenodo)
- **BugwoodMerged** (Local directory)
- **CDDM** (Custom)

### Local Datasets
- Custom local directory structure: `data/<Crop>/<Disease>/<images>`

---

## Installation

```bash
# Core dependencies
pip install kaggle Pillow pandas scikit-learn requests tqdm openpyxl

# Optional: For specific datasets
pip install datasets              # LeafNet (HuggingFace)
pip install kagglehub             # Banana Leaf
pip install tensorflow tensorflow_datasets  # PlantVillage
```

---

## Setup

### 1. Kaggle Credentials (Optional)
For Kaggle datasets, place `~/.kaggle/kaggle.json`:

```bash
# Download from Kaggle → Account → Create New API Token
# Save to ~/.kaggle/kaggle.json
# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 2. HuggingFace Token (Optional)
For private HF datasets:
```bash
pip install huggingface-hub
huggingface-cli login
# Or: export HF_TOKEN=<your_token>
```

### 3. Set Work Root (Optional)
```bash
# Default: /work/mech-ai-scratch/tirtho/CyAg
# Override:
export CYAG_WORK_ROOT=/your/work/path
python DataLoader.py
```

---

## Usage

### Basic
```bash
python DataLoader.py
# Interactive prompt: "How many images per class to download? (or 'all')"
# Generates: Curated_Dataset/
```

### Programmatic
```python
from DataLoader import (
    load_PlantVillage, load_PlantWild, load_PlantDoc,
    load_SBRD, load_MangoLeaf, load_BananaLeaf
)

# Load with sample size
train_df = load_PlantVillage(n=1000)  # 1000 images per class
test_df = load_PlantWild(n=500)

print(train_df.head())
# Columns: image, T1, T2, T3, T4, T5 (tasks), crop, disease, source, split
```

### Supported Functions
```python
# PlantVillage (TFDS)
load_PlantVillage(n)      # n images per class

# PlantWild (HF)
load_PlantWild(n)         # In-the-wild evaluation

# Kaggle datasets
load_SBRD(n)
load_MangoLeaf(n)
load_BananaLeaf(n)
load_Cucumber(n)
load_Lettuce(n)
load_EggplantDisease(n)
... (and 20+ more)

# Local directory
load_local_category(category_name, cls_map, n, source_root)
```

---

## Output Structure

### Directory: `Curated_Dataset/`
```
Curated_Dataset/
├── Images/
│   ├── Tomato/
│   │   ├── Early_Blight/
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │   └── ...
│   │   ├── Late_Blight/
│   │   └── ...
│   ├── Potato/
│   └── ...
├── registry.xlsx       # Multi-sheet Excel report
├── registry.csv        # Summary CSV
└── dataset_info.json   # Metadata
```

### Excel Report: `registry.xlsx`

**Sheet 1: By Crop & Disease**
- Rows per dataset → crop-disease combination
- Columns: Crop, Disease, Source Dataset, Images
- Alternating row colors for readability

**Sheet 2: Crop-Disease Pairs (Unique)**
- One row per unique crop-disease pair
- Aggregated across all sources
- Columns: #, Crop, Disease, # Sources, Source Datasets, Total Images

**Sheet 3: Image Sources**
- One row per image (up to 1,048,576 Excel limit)
- Columns: Crop, Disease, Source, Filename
- Use for validation/auditing

**Sheet 4: Datasets & Papers**
- Metadata per dataset
- Columns: Dataset Name, URL, Paper/Citation

---

## Excel Row Limit (FIXED)

**Issue:** Sheet 3 (Image Sources) can exceed Excel's 1,048,576 row limit if dataset is very large.

**Solution (Applied):**
- Pre-append row limit checks on all sheets
- Graceful truncation with warning messages
- Fallback to CSV export if Excel truncates

**Warning Message Example:**
```
⚠ WARNING: Sheet 3 (Image Sources) reached Excel row limit at 1048575 images. Truncating.
```

If truncated, use CSV instead:
```python
import pandas as pd
df = pd.read_csv("Curated_Dataset/registry.csv")
```

---

## Common Tasks

### Load Multiple Datasets
```python
from DataLoader import load_PlantVillage, load_PlantWild, load_PlantDoc

train = load_PlantVillage(n=5000)
ood_test = load_PlantWild(n=2000)
field = load_PlantDoc(n=1000)

# Combine
import pandas as pd
all_data = pd.concat([train, ood_test, field], ignore_index=True)
```

### Check Dataset Info
```python
from DataLoader import load_from_samples

# Load from already-sampled directory
df = load_from_samples("PlantVillage", classes=["Early Blight", "Late Blight"])

print(f"Rows: {len(df)}")
print(f"Crops: {df['crop'].nunique()}")
print(f"Diseases: {df['disease'].nunique()}")
print(f"Images per class: {df.groupby('disease').size()}")
```

### Export to Different Format
```python
import pandas as pd

# Load from registry
df = pd.read_csv("Curated_Dataset/registry.csv")

# Export to Parquet (better for large datasets)
df.to_parquet("curated_images.parquet", index=False)

# Export to HDF5
df.to_hdf("curated_images.h5", key="data", mode="w")
```

---

## Troubleshooting

### Kaggle Download Fails
```bash
# Verify credentials
cat ~/.kaggle/kaggle.json
# Should have: "username" and "key"

# Test Kaggle API
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate()"
```

### HuggingFace Token Issues
```bash
# Login to HF
huggingface-cli login

# Or set token in environment
export HF_TOKEN=<your_token>
python DataLoader.py
```

### Out of Disk Space
```bash
# Check available space
df -h /work/mech-ai/

# Use smaller sample
python DataLoader.py
# Enter: 100  (instead of "all")
```

### Memory Issues (Large Datasets)
```bash
# Reduce batch processing
# Edit line ~2000 in DataLoader.py:
# batch_size = 100  # Reduce from default 1000
python DataLoader.py
```

---

## Architecture

### Key Functions

**`_normalize_crop_name(name)`** (line 662)
- Standardizes crop names across datasets
- Handles: plurals, underscores, case variations
- Example: "tomato_plants" → "Tomato"

**`_normalize_disease_name(name)`** (line 813)
- Standardizes disease names
- Maps synonyms: "late blight" = "Phytophthora infestans"
- Example: "EarlyB" → "Early Blight"

**`collect_images_df(base_directory)`** (line 458)
- Walks directory tree and collects all images
- Returns DataFrame with columns: image, crop, disease, source

**`sample_per_class(data_df, classes, n)`** (line 572)
- Stratified sampling: n images per class
- Handles class imbalance
- Returns balanced DataFrame

**`generate_xlsx(all_datasets, OUTPUT_XLSX)`** (line 5550)
- Creates 4-sheet Excel report
- With cell styling, merged cells, formulas
- Row limit checks to prevent overflow (NEW)

---

## Performance Notes

- **Kaggle downloads:** 5-30 minutes per dataset (depends on size)
- **Image collection:** ~1000 images/sec (disk-dependent)
- **Sampling:** ~10,000 images/sec
- **Excel generation:** ~5000 images/sec
- **Total for 54K images:** ~30-60 minutes

**To speed up:**
1. Use SSD instead of HDD
2. Reduce `n` (sample size)
3. Run on machine with good internet (for downloads)

---

## Related Documentation

- **Current Data Loaders:** See `data/loader.py`, `data/tfds_plant_village.py`, `data/plantwild_hf.py`
- **PlantVillage Setup:** See `DATASET_SETUP.md`
- **PlantWild Setup:** See `DATASET_SETUP.md`

---

**Last Updated:** May 2, 2026  
**Author:** Comprehensive dataset curation system  
**Status:** Legacy reference; use modular loaders for PlantSwarm pipeline
