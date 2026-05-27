from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .baselines import run_baseline
from .metrics import count_accuracy, edge_precision_recall
from .notears import linear_notears
from .utils import ensure_dir, standardize


SACHS_NODE_NAMES = [
    "raf", "mek", "plcg", "pip2", "pip3", "erk", "akt", "pka", "pkc", "p38", "jnk"
]


def load_sachs_csv(path: str | Path) -> pd.DataFrame:
    """Load a Sachs-style CSV file.

    The repository does not ship the dataset. Put a CSV file at data/sachs.csv
    with 11 columns, preferably named like SACHS_NODE_NAMES, or pass a path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sachs data not found at {path}. Put the dataset as CSV there or pass --data-path."
        )
    df = pd.read_csv(path)
    if df.shape[1] < 11:
        raise ValueError("Sachs CSV should contain at least 11 measurement columns")
    if all(c in df.columns for c in SACHS_NODE_NAMES):
        return df[SACHS_NODE_NAMES]
    return df.iloc[:, :11]


def sachs_consensus_graph() -> np.ndarray:
    """Return a commonly used Sachs consensus graph adjacency.

    Different libraries encode slightly different versions of the Sachs consensus
    network. For a strict paper reproduction, replace this matrix with the exact
    consensus network used by your chosen benchmark source. This default is meant
    to make the pipeline executable and transparent, not to hide assumptions.
    """
    idx = {name: i for i, name in enumerate(SACHS_NODE_NAMES)}
    edges = [
        ("raf", "mek"), ("mek", "erk"), ("plcg", "pip2"), ("plcg", "pip3"),
        ("pip3", "akt"), ("pka", "raf"), ("pka", "mek"), ("pka", "erk"),
        ("pka", "akt"), ("pka", "p38"), ("pka", "jnk"), ("pkc", "raf"),
        ("pkc", "mek"), ("pkc", "p38"), ("pkc", "jnk"), ("p38", "jnk"),
        ("pip2", "pkc"), ("pip3", "pka"), ("erk", "akt"), ("akt", "raf"),
    ]
    B = np.zeros((len(SACHS_NODE_NAMES), len(SACHS_NODE_NAMES)), dtype=int)
    for u, v in edges:
        B[idx[u], idx[v]] = 1
    return B


def run_sachs_experiment(
    data_path: str | Path,
    out_dir: str | Path,
    lambda1: float = 0.1,
    w_threshold: float = 0.3,
    methods=("notears", "notears-l1", "ges", "pc", "lingam"),
) -> pd.DataFrame:
    out_dir = ensure_dir(out_dir)
    df_data = load_sachs_csv(data_path)
    X = standardize(df_data.to_numpy(dtype=float))
    B_true = sachs_consensus_graph()

    rows = []
    for method in methods:
        ml = method.lower()
        if ml in {"notears", "notears-l1", "notears_l1"}:
            lam = lambda1 if ml != "notears" else 0.0
            res = linear_notears(X, lambda1=lam, w_threshold=w_threshold)
            B_est = res.B_est
            success = res.success
            message = res.message
            pretty = "NOTEARS-L1" if lam > 0 else "NOTEARS"
        else:
            base = run_baseline(method, X)
            B_est = base.B_est
            success = base.success
            message = base.message
            pretty = base.name
        row = count_accuracy(B_true, B_est)
        row.update(edge_precision_recall(B_true, B_est))
        row.update({"method": pretty, "success": success, "message": message, "n": X.shape[0], "d": X.shape[1]})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "sachs_metrics.csv", index=False)
    return out
