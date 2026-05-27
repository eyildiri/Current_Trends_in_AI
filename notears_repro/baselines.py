from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .metrics import threshold_matrix


@dataclass
class BaselineResult:
    name: str
    B_est: np.ndarray
    W_est: Optional[np.ndarray] = None
    success: bool = True
    message: str = ""


def correlation_threshold_baseline(X: np.ndarray, threshold: float = 0.25) -> BaselineResult:
    """Very simple non-causal baseline: threshold absolute correlations.

    Because correlations are undirected, this orients edges by index i < j. This is
    intentionally naive and is useful as a minimum baseline for course projects.
    """
    X = np.asarray(X, dtype=float)
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    d = C.shape[0]
    B = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            if abs(C[i, j]) > threshold:
                B[i, j] = 1
    return BaselineResult(name="CorrThreshold", B_est=B, W_est=C, success=True)


def pc_baseline(X: np.ndarray, alpha: float = 0.05) -> BaselineResult:
    """PC baseline using causal-learn, if installed.

    The returned graph can contain mixed edge marks. This function extracts a
    directed adjacency when an arrow direction is clearly present, and ignores
    ambiguous/undirected marks.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except Exception as exc:
        return BaselineResult("PC", np.zeros((X.shape[1], X.shape[1])), success=False, message=f"causal-learn not installed: {exc}")

    try:
        cg = pc(np.asarray(X, dtype=float), alpha=alpha)
        G = np.asarray(cg.G.graph)
        d = G.shape[0]
        B = np.zeros((d, d), dtype=int)

        # Causal-learn graph marks are not as simple as an adjacency matrix.
        # Common convention: G[i, j] = -1 and G[j, i] = 1 means i -> j.
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if G[i, j] == -1 and G[j, i] == 1:
                    B[i, j] = 1
        return BaselineResult("PC", B, success=True)
    except Exception as exc:
        return BaselineResult("PC", np.zeros((X.shape[1], X.shape[1])), success=False, message=str(exc))


def ges_baseline(X: np.ndarray) -> BaselineResult:
    """GES baseline using causal-learn, if installed.

    This is the closest easy Python alternative to the paper's FGS/GES comparison.
    FGS from Tetrad is a separate Java ecosystem; this wrapper keeps the project
    reproducible in Python while documenting the limitation.
    """
    try:
        from causallearn.search.ScoreBased.GES import ges
    except Exception as exc:
        return BaselineResult("GES", np.zeros((X.shape[1], X.shape[1])), success=False, message=f"causal-learn GES not installed: {exc}")

    try:
        record = ges(np.asarray(X, dtype=float))
        graph = record["G"]
        G = np.asarray(graph.graph)
        d = G.shape[0]
        B = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if G[i, j] == -1 and G[j, i] == 1:
                    B[i, j] = 1
        return BaselineResult("GES", B, success=True)
    except Exception as exc:
        return BaselineResult("GES", np.zeros((X.shape[1], X.shape[1])), success=False, message=str(exc))


def lingam_baseline(X: np.ndarray, threshold: float = 0.3) -> BaselineResult:
    """DirectLiNGAM baseline, if the lingam package is installed."""
    try:
        import lingam
    except Exception as exc:
        return BaselineResult("LiNGAM", np.zeros((X.shape[1], X.shape[1])), success=False, message=f"lingam not installed: {exc}")

    try:
        model = lingam.DirectLiNGAM()
        model.fit(np.asarray(X, dtype=float))
        # lingam returns adjacency_matrix_[to, from], so transpose to W[from, to].
        W = np.asarray(model.adjacency_matrix_, dtype=float).T
        B = threshold_matrix(W, threshold)
        return BaselineResult("LiNGAM", B, W_est=W, success=True)
    except Exception as exc:
        return BaselineResult("LiNGAM", np.zeros((X.shape[1], X.shape[1])), success=False, message=str(exc))


def run_baseline(name: str, X: np.ndarray, **kwargs) -> BaselineResult:
    name_lower = name.lower()
    if name_lower in {"corr", "correlation", "corrthreshold"}:
        return correlation_threshold_baseline(X, threshold=kwargs.get("threshold", 0.25))
    if name_lower == "pc":
        return pc_baseline(X, alpha=kwargs.get("alpha", 0.05))
    if name_lower in {"ges", "fgs"}:
        return ges_baseline(X)
    if name_lower in {"lingam", "directlingam"}:
        return lingam_baseline(X, threshold=kwargs.get("threshold", 0.3))
    raise ValueError(f"Unknown baseline: {name}")
