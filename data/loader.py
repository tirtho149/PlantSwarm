"""
data/loader.py
==============
PlantDiagBench DataLoader (PlantSwarm paper).

Data sources:
    * Parquet (``parquet_path``): file path or base64 column.
    * Directory tree (``directory_root``): labels from folder names; see ``data/directory_index.py``.
    * Hugging Face (``hf_dataset_id``): e.g. ``rashikahura/plantWild``; see ``data/plantwild_hf.py`` (set ``HF_TOKEN`` in ``.env``).
    * TFDS (``tfds_name: plant_village``): TensorFlow Datasets Plant Village; see ``data/tfds_plant_village.py``.
    * DataLoader (``dataloader_dataset_name``): 30+ datasets (PlantVillage, PlantWild, PlantDoc, etc.); see ``data/dataloader_wrapper.py``.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


@dataclass
class PlantRecord:
    """One PlantDiagBench image + labels + metadata."""

    image_id: str
    image: Image.Image
    image_b64: str

    symptom_type: Optional[str] = None       # T1
    pathogen_class: Optional[str] = None     # T2
    disease_name: Optional[str] = None      # T3
    severity_class: Optional[str] = None    # T4
    crop_species: Optional[str] = None      # T5

    img_quality: Optional[str] = None
    img_quality_score: Optional[float] = None
    complexity_edge_density: Optional[float] = None

    meta: Dict = field(default_factory=dict)


def build_top_k_labels(df: pd.DataFrame, col: str, k: int) -> List[str]:
    if col not in df.columns:
        return ["Other"]
    counts = df[col].value_counts()
    top = counts.head(k).index.tolist()
    return top + ["Other"]


class PlantDiagBenchLoader:
    """
    Load PlantDiagBench-style data for PlantSwarm experiments.

    Parameters
    ----------
    cfg : dict
        ``data`` section of the YAML config.
    split : {'test', 'calibration', 'all'}

    Use ``cfg['directory_root']``, ``cfg['hf_dataset_id']``, ``cfg['tfds_name']``,
    or ``cfg['parquet_path']``.
    Priority: directory > Hugging Face > TFDS > parquet.
    """

    IMAGE_SIZE = 448

    def __init__(
        self,
        cfg_or_path: dict | str | Path,
        cfg: Optional[dict] = None,
        split: str = "test",
        top_k_diseases: Optional[List[str]] = None,
    ):
        """
        ``PlantDiagBenchLoader(data_cfg, split=...)`` or legacy
        ``PlantDiagBenchLoader(parquet_path, data_cfg, split=...)``.
        """
        if cfg is None:
            if not isinstance(cfg_or_path, dict):
                raise TypeError("PlantDiagBenchLoader expects a data config dict as first argument.")
            cfg = cfg_or_path
        else:
            cfg = dict(cfg)
            if not cfg.get("directory_root") and cfg_or_path is not None:
                cfg["parquet_path"] = cfg_or_path

        self.cfg = cfg
        self.split = split
        self._from_directory = bool(cfg.get("directory_root"))
        self._from_hf = bool(cfg.get("hf_dataset_id"))
        self._from_tfds = bool(cfg.get("tfds_name"))
        self._from_dataloader = bool(cfg.get("dataloader_dataset_name"))
        self.parquet_path: Optional[Path] = None

        if self._from_directory:
            from data.directory_index import build_directory_dataframe

            self._df_full = build_directory_dataframe(cfg)
        elif self._from_hf:
            self._df_full = self._load_hf_dataframe()
        elif self._from_tfds:
            self._df_full = self._load_tfds_dataframe()
        elif self._from_dataloader:
            self._df_full = self._load_dataloader_dataframe()
        else:
            pq_path = cfg.get("parquet_path")
            if not pq_path:
                raise ValueError(
                    "Set data.directory_root, data.hf_dataset_id, data.tfds_name, "
                    "data.dataloader_dataset_name, or data.parquet_path."
                )
            self.parquet_path = Path(pq_path)
            if not self.parquet_path.is_file():
                raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")
            self._df_full = self._load_parquet()

        quality_col = cfg.get("quality_col")
        quality_val = cfg.get("quality_filter")
        if quality_col and quality_col in self._df_full.columns and quality_val is not None:
            self._df_full = self._df_full[self._df_full[quality_col] == quality_val].reset_index(
                drop=True
            )

        lc = cfg.get("label_cols", {})
        disease_col = lc.get("T3", "disease_name")
        if top_k_diseases is not None:
            self.top_k_diseases = top_k_diseases
        elif disease_col in self._df_full.columns:
            self.top_k_diseases = build_top_k_labels(
                self._df_full,
                col=disease_col,
                k=cfg.get("top_k_diseases", 50),
            )
            unk_mask = self._df_full[disease_col].astype(str).str.strip() == "Unknown"
            if unk_mask.any() and "Unknown" not in self.top_k_diseases:
                self.top_k_diseases = list(self.top_k_diseases) + ["Unknown"]
        else:
            self.top_k_diseases = ["Other"]

        self._df_test, self._df_cal = self._stratified_split()

        if split == "calibration":
            self._df = self._df_cal
        elif split == "test":
            self._df = self._df_test
        else:
            self._df = self._df_full

        self._records: Optional[List[PlantRecord]] = None

    def _load_parquet(self) -> pd.DataFrame:
        assert self.parquet_path is not None
        table = pq.read_table(self.parquet_path)
        return table.to_pandas()

    def _load_hf_dataframe(self) -> pd.DataFrame:
        from utils.env import load_project_dotenv

        load_project_dotenv()

        ds_id = (self.cfg.get("hf_dataset_id") or "").strip()
        if "plantwild" in ds_id.lower():
            from data.plantwild_hf import build_plantwild_dataframe

            result = build_plantwild_dataframe(
                hf_dataset_id=ds_id,
                split=str(self.cfg.get("hf_split", "test")),
                max_examples=self.cfg.get("hf_max_examples"),
                seed=int(self.cfg.get("hf_seed", 42)),
                image_col=self.cfg.get("image_col", "image_bytes"),
                jpeg_quality=int(self.cfg.get("hf_jpeg_quality", 95)),
            )
            rows_data = result["rows"]
            return pd.DataFrame(rows_data)
        raise ValueError(f"Unknown data.hf_dataset_id (no loader): {ds_id!r}")

    def _load_tfds_dataframe(self) -> pd.DataFrame:
        name = (self.cfg.get("tfds_name") or "").strip().lower()
        if name == "plant_village":
            from data.tfds_plant_village import build_plant_village_dataframe

            return build_plant_village_dataframe(
                max_examples=self.cfg.get("tfds_max_examples"),
                data_dir=self.cfg.get("tfds_data_dir"),
                split=self.cfg.get("tfds_split", "train"),
                seed=int(self.cfg.get("tfds_seed", 42)),
                image_col=self.cfg.get("image_col", "image_bytes"),
            )
        raise ValueError(f"Unknown data.tfds_name: {self.cfg.get('tfds_name')!r}")

    def _load_dataloader_dataframe(self) -> pd.DataFrame:
        """Load from DataLoader.py (30+ datasets)."""
        from data.dataloader_wrapper import build_dataloader_dataframe

        dataset_name = self.cfg.get("dataloader_dataset_name")
        if not dataset_name:
            raise ValueError("data.dataloader_dataset_name must be set")

        return build_dataloader_dataframe(
            dataset_name=dataset_name,
            max_examples=self.cfg.get("dataloader_max_examples"),
            image_col=self.cfg.get("image_col", "image"),
            seed=int(self.cfg.get("seed", 42)),
            benchmark_col=self.cfg.get("benchmark_col", "benchmark"),
        )

    def _load_plantdoc_dataframe(self) -> pd.DataFrame:
        from data.plantdoc_github import build_plantdoc_dataframe

        repo = self.cfg.get("plantdoc_repo_root")
        if not repo:
            raise ValueError("data.plantdoc_repo_root must be set")
        return build_plantdoc_dataframe(
            repo,
            split=str(self.cfg.get("plantdoc_split", "train")),
            image_col=self.cfg.get("image_col", "image_path"),
            id_col=self.cfg.get("id_col", "id"),
            label_cols=self.cfg.get("label_cols"),
            infer_crop_for_t5=bool(self.cfg.get("plantdoc_infer_crop_t5", True)),
        )

    def _stratified_split(self):
        from data.stratifier import stratified_split

        n_test = self.cfg.get("n_images", 10000)
        n_cal = self.cfg.get("calibration_split_size", 500)
        stratify_cols = self.cfg.get("stratify_by", ["crop_species", "severity_class"])
        actual_cols = []
        lc = self.cfg.get("label_cols", {})
        for c in stratify_cols:
            col = lc.get(c, c)
            if col in self._df_full.columns:
                actual_cols.append(col)
        return stratified_split(self._df_full, n_test=n_test, n_cal=n_cal, stratify_cols=actual_cols)

    def _resolve_image_id(self, row: pd.Series) -> str:
        id_col = self.cfg.get("id_col", "id")
        for key in (id_col, "image_id", "plant_id", "sample_id"):
            if key in row.index and row.get(key) is not None and str(row.get(key)).strip():
                return str(row.get(key))
        return str(row.name)

    def _resolve_image_path(self, candidate: str) -> str:
        """Use path as-is if it exists; else try data.image_root or CYAG_IMAGE_ROOT."""
        if os.path.isfile(candidate):
            return candidate
        root = (self.cfg.get("image_root") or os.environ.get("CYAG_IMAGE_ROOT", "") or "").strip()
        if root:
            joined = os.path.join(root, candidate.lstrip(os.sep))
            if os.path.isfile(joined):
                return joined
        return candidate

    def _load_image(self, row: pd.Series) -> tuple[Image.Image, str]:
        image_col = self.cfg.get("image_col", "image_path")

        if image_col in row.index and isinstance(row[image_col], str):
            candidate = self._resolve_image_path(row[image_col])
            if os.path.exists(candidate):
                img = Image.open(candidate).convert("RGB")
                img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return img, b64

        if image_col in row.index and isinstance(row[image_col], (bytes, str)):
            raw = row[image_col]
            if isinstance(raw, str):
                try:
                    raw_bytes = base64.b64decode(raw, validate=False)
                except Exception:
                    raw_bytes = raw.encode()
            else:
                raw_bytes = raw
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return img, b64

        raise FileNotFoundError(f"Cannot load image from row: {row.get(image_col)}")

    def _row_to_record(self, row: pd.Series) -> PlantRecord:
        lc = self.cfg.get("label_cols", {})

        img, b64 = self._load_image(row)

        dname = row.get(lc.get("T3", "disease_name"), None)
        if dname is not None and dname not in self.top_k_diseases:
            dname = "Other"

        dc = self.cfg.get("demographic_cols", {})

        return PlantRecord(
            image_id=self._resolve_image_id(row),
            image=img,
            image_b64=b64,
            symptom_type=row.get(lc.get("T1", "symptom_type"), None),
            pathogen_class=row.get(lc.get("T2", "pathogen_class"), None),
            disease_name=dname,
            severity_class=row.get(lc.get("T4", "severity_class"), None),
            crop_species=row.get(lc.get("T5", "crop_species"), None),
            img_quality=row.get(self.cfg.get("quality_col", "img_quality"), None),
            img_quality_score=row.get("img_quality_score", None),
            complexity_edge_density=row.get("complexity_edge_density", None),
            meta=row.to_dict(),
        )

    def load_all(self) -> List[PlantRecord]:
        if self._records is None:
            self._records = [self._row_to_record(row) for _, row in self._df.iterrows()]
        return self._records

    def __iter__(self) -> Iterator[PlantRecord]:
        for _, row in self._df.iterrows():
            yield self._row_to_record(row)

    def __len__(self) -> int:
        return len(self._df)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def _unique_labels_for_task(self, task_id: str) -> Optional[List[str]]:
        """Return sorted unique labels from data for this task, or None if column missing/empty."""
        lc = self.cfg.get("label_cols", {})
        field_map = {
            "T1": lc.get("T1", "symptom_type"),
            "T2": lc.get("T2", "pathogen_class"),
            "T3": lc.get("T3", "disease_name"),
            "T4": lc.get("T4", "severity_class"),
            "T5": lc.get("T5", "crop_species"),
        }
        col = field_map.get(task_id)
        if not col or col not in self._df_full.columns:
            return None
        vals = sorted(
            {
                str(x).strip()
                for x in self._df_full[col].dropna().unique()
                if str(x).strip() and str(x).strip().lower() != "nan"
            }
        )
        return vals or None

    @property
    def label_space(self) -> Dict[str, List[str]]:
        """Task label vocabularies (PlantSwarm Table: tasks)."""
        static = self.cfg.get("labels") or {}
        default_t1 = [
            "Lesion",
            "Blight",
            "Mosaic",
            "Wilt",
            "Rot",
            "Canker",
            "Rust",
            "Powdery mildew",
        ]
        default_t2 = [
            "Fungal",
            "Bacterial",
            "Viral",
            "Nutrient deficiency",
            "Pest damage",
        ]
        default_t4 = ["Healthy", "Early", "Moderate", "Severe"]
        default_t5 = [
            "Tomato",
            "Corn",
            "Potato",
            "Cassava",
            "Grape",
            "Apple",
            "Cherry",
            "Peach",
            "Pepper",
            "Squash",
            "Strawberry",
            "Orange",
            "Soybean",
            "Other",
        ]

        def resolve(task_id: str, default_list: List[str]) -> List[str]:
            if static.get(task_id) is not None:
                return list(static[task_id])
            if self._from_directory or self._from_hf:
                discovered = self._unique_labels_for_task(task_id)
                if discovered:
                    return list(discovered)
            return list(default_list)

        t1 = resolve("T1", default_t1)
        t2 = resolve("T2", default_t2)
        t4 = resolve("T4", default_t4)
        t5 = resolve("T5", default_t5)

        if static.get("T3") is not None:
            t3 = list(static["T3"])
        else:
            t3 = list(self.top_k_diseases)
        return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5}


