from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

import numpy as np

Edge = Tuple[int, int]


def threshold_matrix(W: np.ndarray, threshold: float) -> np.ndarray:
    B = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(B, 0)
    return B


def is_dag(B: np.ndarray) -> bool:
    """Return True iff the binary adjacency matrix is acyclic."""
    B = (np.asarray(B) != 0).astype(int)
    d = B.shape[0]
    indeg = B.sum(axis=0).astype(int).tolist()
    queue = [i for i in range(d) if indeg[i] == 0]
    visited = 0
    while queue:
        i = queue.pop()
        visited += 1
        for j in np.where(B[i] != 0)[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    return visited == d


def _edge_set(B: np.ndarray) -> Set[Edge]:
    B = (np.asarray(B) != 0).astype(int)
    return set(zip(*np.where(B != 0)))


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> Dict[str, float]:
    """Compute common structure-learning metrics for directed graphs.

    Metrics:
        fdr: false discovery rate; reversed edges count as false discoveries.
        tpr: true positive rate.
        fpr: false positive rate.
        shd: directed structural Hamming distance; a reversed edge counts as 1.
        nnz: number of predicted directed edges.

    This function assumes both inputs are directed adjacency matrices, not CPDAGs.
    """
    B_true = (np.asarray(B_true) != 0).astype(int)
    B_est = (np.asarray(B_est) != 0).astype(int)
    if B_true.shape != B_est.shape:
        raise ValueError("B_true and B_est must have the same shape")
    np.fill_diagonal(B_true, 0)
    np.fill_diagonal(B_est, 0)

    d = B_true.shape[0]
    pred = _edge_set(B_est)
    cond = _edge_set(B_true)
    cond_reversed = {(j, i) for (i, j) in cond}
    pred_reversed = {(j, i) for (i, j) in pred}

    true_pos = pred & cond
    reverse = pred & cond_reversed
    false_pos = pred - cond - cond_reversed

    # Missing true edges that are neither correctly predicted nor predicted reversed.
    missing = cond - pred - pred_reversed

    fdr = (len(reverse) + len(false_pos)) / max(len(pred), 1)
    tpr = len(true_pos) / max(len(cond), 1)
    fpr_den = d * (d - 1) - len(cond)
    fpr = (len(reverse) + len(false_pos)) / max(fpr_den, 1)
    shd = len(missing) + len(reverse) + len(false_pos)

    return {
        "fdr": float(fdr),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "shd": float(shd),
        "nnz": float(len(pred)),
    }


def l2_weight_error(W_true: np.ndarray, W_est: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(W_true) - np.asarray(W_est)))


def edge_precision_recall(B_true: np.ndarray, B_est: np.ndarray) -> Dict[str, float]:
    pred = _edge_set(B_est)
    cond = _edge_set(B_true)
    tp = len(pred & cond)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(cond), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
