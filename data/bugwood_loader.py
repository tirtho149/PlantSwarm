"""
data/bugwood_loader.py
======================
Bugwood Host Name / Subject Display Name normalisation helpers for the
Setup phase (``scripts/filter_bugwood_csv.py``).

The previous full BugwoodLoader (image fetcher, BugwoodRecord dataclass,
per-class batching) was Phase 2-only and has been retired along with the
trace-generation pipeline.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Crop name canonicalisation
# ---------------------------------------------------------------------------

BUGWOOD_EXACT_CROP_MAP: Dict[str, str] = {
    "alfalfa": "Alfalfa", "apple": "Apple", "banana": "Banana",
    "bananas": "Banana", "basil": "Basil", "bean": "Bean",
    "bell pepper": "Bell Pepper", "bell_pepper": "Bell Pepper",
    "blueberry": "Blueberry", "broccoli": "Broccoli", "cabbage": "Cabbage",
    "carrot": "Carrot", "cashew": "Cashew", "cassava": "Cassava",
    "cauliflower": "Cauliflower", "celery": "Celery", "cherry": "Cherry",
    "chickpea": "Chickpea", "citrus": "Citrus", "coconut": "Coconut",
    "coconut palm": "Coconut", "coffee": "Coffee", "corn": "Corn",
    "common corn": "Corn", "field corn": "Corn", "sweet corn": "Corn",
    "cotton": "Cotton", "cucumber": "Cucumber", "durian": "Durian",
    "eggplant": "Eggplant", "garden tomato": "Tomato", "garlic": "Garlic",
    "ginger": "Ginger", "grape": "Grape", "grapevine": "Grape",
    "wine grape": "Grape", "lettuce": "Lettuce", "mango": "Mango",
    "maple": "Maple", "melon": "Melon", "watermelon": "Watermelon",
    "muskmelon": "Melon", "cantaloupe": "Melon",
    "common hop": "Hops", "hop": "Hops",
    "oak": "Oak", "orange": "Orange", "orange haunglongbing": "Orange",
    "orange huanglongbing": "Orange", "peach": "Peach", "pear": "Pear",
    "pepper": "Pepper", "plum": "Plum", "potato": "Potato",
    "sweetpotato": "Sweet Potato", "sweet potato": "Sweet Potato",
    "pumpkin": "Pumpkin", "raspberry": "Raspberry", "rice": "Rice",
    "rose": "Rose", "rye": "Rye", "soybean": "Soybean", "squash": "Squash",
    "squashes (general)": "Squash", "winter squash": "Squash",
    "summer squash": "Squash", "strawberry": "Strawberry",
    "sugarcane": "Sugarcane", "tea": "Tea", "tobacco": "Tobacco",
    "tomato": "Tomato", "vanilla": "Vanilla", "wheat": "Wheat",
    "common wheat": "Wheat", "durum wheat": "Wheat", "spring wheat": "Wheat",
    "winter wheat": "Wheat", "zucchini": "Zucchini",
}

# Bugwood taxonomy entries that are not crop hosts and should be dropped.
BUGWOOD_NON_CROP_KEYS = {
    "wood decay fungi", "wood decay fungus", "canker complex",
    "shelf fungi", "bark beetle", "powdery mildew", "downy mildew",
}


def _normalize_key(value: str) -> str:
    s = str(value or "").strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Normalise lookup keys once so callers don't need to think about whether the
# key in the literal map happened to contain parens / punctuation.
_CROP_LOOKUP: Dict[str, str] = {_normalize_key(k): v for k, v in BUGWOOD_EXACT_CROP_MAP.items()}
_NON_CROP_LOOKUP: set = {_normalize_key(k) for k in BUGWOOD_NON_CROP_KEYS}


def _map_crop(raw_crop: str) -> Optional[str]:
    """Canonicalise a Bugwood ``Host Name`` to a crop label, or None to drop."""
    key = _normalize_key(raw_crop)
    if not key or key in _NON_CROP_LOOKUP:
        return None
    if key in _CROP_LOOKUP:
        return _CROP_LOOKUP[key]
    # Fallback: title-case the raw host so unmapped-but-plausible entries
    # ("oak", "rose") still survive instead of being silently dropped.
    pretty = re.sub(r"\s+", " ", str(raw_crop or "").replace("_", " ")).strip().title()
    return pretty or None


# ---------------------------------------------------------------------------
# Disease label cleanup
# ---------------------------------------------------------------------------

_DISEASE_PAREN_RE = re.compile(r"\s*\(.*$")


def _clean_disease(raw_disease: str) -> str:
    """Strip the parenthetical scientific suffix from Subject Display Name.

    "Phytophthora blight (Phytophthora capsici Leonian)" -> "Phytophthora Blight"
    """
    if not raw_disease:
        return ""
    base = _DISEASE_PAREN_RE.sub("", str(raw_disease)).strip()
    return base.title() if base else ""
