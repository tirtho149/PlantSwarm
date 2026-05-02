"""
DataLoader Integration Wrapper
================================
Bridge between DataLoader.py and PlantSwarm pipeline.
Enables seamless use of 30+ datasets within the modular loader system.

Usage:
    from data.dataloader_wrapper import load_dataloader_dataset

    df = load_dataloader_dataset(
        dataset_name="PlantVillage",
        n_images=5000,
        split="train"
    )
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path so we can import DataLoader.py
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import DataLoader functions
try:
    from DataLoader import (
        load_PlantVillage, load_PlantWild, load_PlantDoc,
        load_SBRD, load_MangoLeaf, load_BananaLeaf,
        load_Cucumber, load_Lettuce, load_EggplantDisease,
        load_StrawberryDiseaseDetection, load_VanillaDisease,
        load_BeanLeaf, load_YellowRust, load_BugwoodMerged,
        load_LeafNet, load_Cauliflower, load_NewPlantDiseases,
        load_PlantDoc, load_CDDM, load_RadyPlantDiseases,
    )
    DATALOADER_AVAILABLE = True
except ImportError as e:
    DATALOADER_AVAILABLE = False
    _import_error = str(e)


# Mapping of dataset names to loader functions
DATASET_LOADERS = {
    "plant_village": load_PlantVillage,
    "plantvillage": load_PlantVillage,
    "plant_wild": load_PlantWild,
    "plantwild": load_PlantWild,
    "plant_doc": load_PlantDoc,
    "plantdoc": load_PlantDoc,
    "sbrd": load_SBRD,
    "mango_leaf": load_MangoLeaf,
    "mangoleaf": load_MangoLeaf,
    "banana_leaf": load_BananaLeaf,
    "bananaleaf": load_BananaLeaf,
    "cucumber": load_Cucumber,
    "lettuce": load_Lettuce,
    "eggplant_disease": load_EggplantDisease,
    "eggplantdisease": load_EggplantDisease,
    "strawberry": load_StrawberryDiseaseDetection,
    "vanilla": load_VanillaDisease,
    "bean_leaf": load_BeanLeaf,
    "beanleaf": load_BeanLeaf,
    "yellow_rust": load_YellowRust,
    "yellowrust": load_YellowRust,
    "bugwood": load_BugwoodMerged,
    "leafnet": load_LeafNet,
    "cauliflower": load_Cauliflower,
    "new_plant_diseases": load_NewPlantDiseases,
    "newplantdiseases": load_NewPlantDiseases,
    "cddm": load_CDDM,
    "rady": load_RadyPlantDiseases,
} if DATALOADER_AVAILABLE else {}


def load_dataloader_dataset(dataset_name, n_images=None, split=None):
    """
    Load dataset from DataLoader.py.

    Args:
        dataset_name: Name of dataset (e.g., "PlantVillage", "PlantWild")
        n_images: Number of images per class (None = all)
        split: "train", "val", "test" (optional, depends on dataset)

    Returns:
        pandas.DataFrame with columns:
            - image: path or URL
            - T1, T2, T3, T4, T5: task labels
            - crop: crop species
            - disease: disease name
            - source: dataset source
            - split: train/val/test
            - benchmark: dataset benchmark name

    Raises:
        RuntimeError: If DataLoader.py not available
        ValueError: If dataset_name not recognized
    """
    if not DATALOADER_AVAILABLE:
        raise RuntimeError(
            f"DataLoader.py not available. Error: {_import_error}\n"
            "Install dependencies: pip install -r requirements.txt"
        )

    # Normalize dataset name
    dataset_key = dataset_name.lower().replace(" ", "_").replace("-", "_")

    if dataset_key not in DATASET_LOADERS:
        available = ", ".join(sorted(DATASET_LOADERS.keys()))
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized.\n"
            f"Available: {available}"
        )

    loader_fn = DATASET_LOADERS[dataset_key]

    print(f"[DataLoader] Loading {dataset_name}...")
    try:
        df = loader_fn(n=n_images)

        # Ensure required columns exist
        required_cols = ["image", "T3", "T5"]  # disease, crop
        for col in required_cols:
            if col not in df.columns:
                print(f"  WARNING: Column '{col}' not found. Adding empty column.")
                df[col] = None

        # Add benchmark column if missing
        if "benchmark" not in df.columns:
            df["benchmark"] = dataset_name.lower()

        print(f"  ✓ Loaded {len(df)} images from {dataset_name}")
        return df

    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        raise


def list_available_datasets():
    """List all available DataLoader datasets."""
    if not DATALOADER_AVAILABLE:
        return []
    return sorted(set(DATASET_LOADERS.keys()))


def get_dataloader_info(dataset_name):
    """Get info about a DataLoader dataset."""
    if not DATALOADER_AVAILABLE:
        raise RuntimeError("DataLoader.py not available")

    dataset_key = dataset_name.lower().replace(" ", "_")
    if dataset_key not in DATASET_LOADERS:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    # Try to load a small sample
    try:
        df = DATASET_LOADERS[dataset_key](n=10)
        return {
            "name": dataset_name,
            "available": True,
            "sample_size": len(df),
            "columns": list(df.columns),
            "crops": df.get("T5", df.get("crop", pd.Series())).nunique() if "T5" in df.columns or "crop" in df.columns else 0,
            "diseases": df.get("T3", df.get("disease", pd.Series())).nunique() if "T3" in df.columns or "disease" in df.columns else 0,
        }
    except Exception as e:
        return {
            "name": dataset_name,
            "available": False,
            "error": str(e)
        }


# ═══════════════════════════════════════════════════════════════════════════
#  INTEGRATION WITH MAIN LOADER
# ═══════════════════════════════════════════════════════════════════════════

def build_dataloader_dataframe(
    dataset_name,
    *,
    max_examples=None,
    image_col="image",
    seed=42,
    benchmark_col="benchmark",
):
    """
    Load DataLoader dataset compatible with main pipeline.

    Args:
        dataset_name: Name of DataLoader dataset
        max_examples: Limit to N examples (None = all)
        image_col: Column name for image paths
        seed: Random seed for sampling
        benchmark_col: Column name for benchmark label

    Returns:
        DataFrame with standardized columns for pipeline
    """
    df = load_dataloader_dataset(dataset_name, n_images=max_examples)

    # Rename columns to match pipeline standard
    df = df.rename(columns={
        image_col: "image" if image_col != "image" else "image",
    })

    # Ensure benchmark column
    if benchmark_col and benchmark_col not in df.columns:
        df[benchmark_col] = dataset_name.lower()

    # Limit examples if requested
    if max_examples and len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Quick test
    if DATALOADER_AVAILABLE:
        print("✓ DataLoader wrapper initialized")
        print(f"  Available datasets: {len(list_available_datasets())}")
        print(f"  Datasets: {', '.join(list_available_datasets()[:5])}...")

        # Try loading a small sample
        try:
            df = load_dataloader_dataset("plant_village", n_images=10)
            print(f"\n✓ Test load successful: {len(df)} images")
        except Exception as e:
            print(f"\n✗ Test load failed: {e}")
    else:
        print("✗ DataLoader not available (dependencies not installed)")
