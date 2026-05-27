from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_random_seed(seed: int) -> np.random.Generator:
    """Set Python and NumPy seeds and return a Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def standardize(X: np.ndarray) -> np.ndarray:
    """Column-standardize data, safely handling constant columns."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
