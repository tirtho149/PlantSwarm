"""
observe/dataset.py
==================
Datasets for the KB-augmented OOD classifier.

  BugwoodTomatoDataset   training set, filtered to one crop (default Tomato)
                          from the Bugwood usable CSV + .bugwood_cache/.
                          One sample per cached image; label = class_id.

  PVFolderDataset        evaluation set on PlantVillage (folder per class).
  PWFolderDataset        evaluation set on PlantWild     (folder per class).

The classifier is trained on Bugwood and evaluated zero/few-shot on PV
and PW. Class id space is built externally from KB prototypes; the
loaders just translate `(crop, disease)` or PV/PW folder names into
integer ids.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Class index — maps prototype labels to integer ids
# ---------------------------------------------------------------------------

class ClassIndex:
    """Bidirectional mapping between prototype labels and integer class ids."""

    def __init__(self, labels: Sequence[str]):
        self.labels: List[str] = list(labels)
        self._lookup: Dict[str, int] = {lbl: i for i, lbl in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    def to_id(self, label: str) -> Optional[int]:
        return self._lookup.get(label)

    def to_label(self, idx: int) -> str:
        return self.labels[idx]


# ---------------------------------------------------------------------------
# Image cache lookup (matches pathome_kb's cache layout)
# ---------------------------------------------------------------------------

def _find_image(image_number: str, cache_dirs: Sequence[Path]) -> Optional[Path]:
    if not image_number:
        return None
    for d in cache_dirs:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = d / f"{image_number}{ext}"
            if p.is_file() and p.stat().st_size > 0:
                return p
    return None


# ---------------------------------------------------------------------------
# Bugwood training dataset
# ---------------------------------------------------------------------------

class BugwoodTomatoDataset(Dataset):
    """One sample per (image_number, label) row from the filtered CSV.

    Filters to a single crop (default Tomato). Skips rows whose image
    isn't in the cache yet. Label is the ClassIndex id; rows whose
    disease isn't in the class index are dropped.
    """

    def __init__(
        self,
        *,
        csv_path: str | Path,
        cache_dirs: Sequence[str | Path],
        class_index: ClassIndex,
        crop: str = "Tomato",
        transform: Optional[Callable] = None,
    ):
        self.crop = crop
        self.transform = transform
        self.cache_dirs = [Path(d) for d in cache_dirs]
        self.class_index = class_index

        self.samples: List[Tuple[Path, int]] = []
        self._skipped_no_image    = 0
        self._skipped_no_class    = 0
        self._skipped_wrong_crop  = 0

        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row_crop = (row.get("NormCrop") or "").strip()
                if row_crop != crop:
                    self._skipped_wrong_crop += 1
                    continue
                disease = (row.get("NormDisease") or "").strip()
                if not disease:
                    self._skipped_no_class += 1
                    continue
                label = f"{row_crop}::{disease}"
                cls_id = class_index.to_id(label)
                if cls_id is None:
                    self._skipped_no_class += 1
                    continue
                img_num = (row.get("Image Number") or "").strip()
                path = _find_image(img_num, self.cache_dirs)
                if path is None:
                    self._skipped_no_image += 1
                    continue
                self.samples.append((path, cls_id))

    def stats(self) -> Dict[str, int]:
        return {
            "n_samples":         len(self.samples),
            "skipped_no_image":  self._skipped_no_image,
            "skipped_no_class":  self._skipped_no_class,
            "skipped_wrong_crop": self._skipped_wrong_crop,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": label, "path": str(path)}


# ---------------------------------------------------------------------------
# PlantVillage folder dataset (Tomato slice by default)
# ---------------------------------------------------------------------------

def _normalize_pv_folder_to_label(folder_name: str) -> Optional[Tuple[str, str]]:
    """Parse a PV folder name into (crop, disease).

    PV uses `<Crop>___<Disease>` with underscores; e.g.
    `Tomato___Early_blight`. Returns (crop, disease_titlecase) or None.
    """
    parts = folder_name.split("___")
    if len(parts) != 2:
        return None
    crop_raw, disease_raw = parts
    # Crop normalization: "Pepper,_bell" -> "Bell Pepper"
    crop = crop_raw.replace("_", " ").strip()
    crop = re.sub(r"\s*\(.*?\)\s*", "", crop).strip()
    if crop.lower() == "pepper, bell":
        crop = "Bell Pepper"
    elif crop.lower() == "corn, maize" or crop.lower() == "corn":
        crop = "Corn"
    elif crop.lower() == "cherry, including sour" or crop.lower() == "cherry":
        crop = "Cherry"
    # Disease normalization: keep underscores out, title-case spelling.
    disease = disease_raw.replace("_", " ").strip()
    disease = re.sub(r"\s+", " ", disease)
    if disease.lower() == "healthy":
        disease = "healthy"
    else:
        disease = disease.title()
    return crop, disease


class PVFolderDataset(Dataset):
    """PlantVillage folder-per-class dataset, filtered to a single crop.

    ``root`` should contain folders named like ``Tomato___Early_blight``.
    Rows whose (crop, disease) doesn't appear in ``class_index`` are
    dropped (logged in stats).
    """

    def __init__(
        self,
        *,
        root: str | Path,
        class_index: ClassIndex,
        crop: str = "Tomato",
        transform: Optional[Callable] = None,
        limit_per_class: Optional[int] = None,
    ):
        self.crop = crop
        self.transform = transform
        self.class_index = class_index
        self.samples: List[Tuple[Path, int, str]] = []
        self._skipped_no_class = 0
        self._skipped_other_crop = 0
        self._counts: Dict[str, int] = {}

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"PV root not found: {root}")

        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            parsed = _normalize_pv_folder_to_label(sub.name)
            if parsed is None:
                continue
            folder_crop, folder_disease = parsed
            if folder_crop != crop:
                self._skipped_other_crop += 1
                continue
            label = f"{folder_crop}::{folder_disease}"
            cls_id = class_index.to_id(label)
            if cls_id is None:
                self._skipped_no_class += 1
                continue
            files = []
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
                files.extend(sub.glob(f"*{ext}"))
            files = sorted(files)
            if limit_per_class is not None:
                files = files[:limit_per_class]
            for f in files:
                self.samples.append((f, cls_id, label))
                self._counts[label] = self._counts.get(label, 0) + 1

    def stats(self) -> Dict[str, int]:
        return {
            "n_samples":          len(self.samples),
            "skipped_no_class":   self._skipped_no_class,
            "skipped_other_crop": self._skipped_other_crop,
            "per_class":          dict(self._counts),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        path, label_id, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img, "label": label_id,
            "label_str": label, "path": str(path),
        }


# ---------------------------------------------------------------------------
# PlantWild folder dataset (mirror of PVFolderDataset; PW folder layout
# is slightly different — typically <crop>_<disease> rather than ___).
# ---------------------------------------------------------------------------

class PWFolderDataset(Dataset):
    """PlantWild folder-per-class dataset.

    PW folder layout (paper-as-written) uses ``<crop>_<disease>``. We
    accept either ``<crop>___<disease>`` (PV style) or
    ``<crop>_<disease>`` and try to parse both.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        class_index: ClassIndex,
        crop: str = "tomato",
        transform: Optional[Callable] = None,
        limit_per_class: Optional[int] = None,
    ):
        self.crop = crop
        self.transform = transform
        self.class_index = class_index
        self.samples: List[Tuple[Path, int, str]] = []
        self._counts: Dict[str, int] = {}

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"PW root not found: {root}")

        target_crop = crop.lower().strip()
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name
            # Try PV-style first.
            parsed = _normalize_pv_folder_to_label(name)
            if parsed is None:
                # PW <crop>_<disease> fallback.
                m = re.match(r"^([^_]+)_(.+)$", name)
                if not m:
                    continue
                crop_part = m.group(1).replace("-", " ")
                disease_part = m.group(2).replace("_", " ")
                parsed = (crop_part.title(), disease_part.title())
            folder_crop, folder_disease = parsed
            if folder_crop.lower() != target_crop:
                continue
            label = f"{folder_crop}::{folder_disease}"
            cls_id = class_index.to_id(label)
            if cls_id is None:
                continue
            files = []
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
                files.extend(sub.glob(f"*{ext}"))
            files = sorted(files)
            if limit_per_class is not None:
                files = files[:limit_per_class]
            for f in files:
                self.samples.append((f, cls_id, label))
                self._counts[label] = self._counts.get(label, 0) + 1

    def stats(self) -> Dict[str, int]:
        return {
            "n_samples": len(self.samples),
            "per_class": dict(self._counts),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        path, label_id, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img, "label": label_id,
            "label_str": label, "path": str(path),
        }
