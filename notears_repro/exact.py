from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import threshold_matrix
from .notears import objective_value


@dataclass
class ExactOrderResult:
    W_est: np.ndarray
    B_est: np.ndarray
    score: float
    order: Tuple[int, ...]
    success: bool
    message: str = ""


def _fit_linear_parents(X: np.ndarray, child: int, parents: Sequence[int], lambda1: float = 0.0) -> np.ndarray:
    """Fit linear regression X_child ~ X_parents. Ridge-free least squares.

    This is an exact-search helper for tiny graphs. It is not a full GOBNILP
    replacement because it enumerates topological orders instead of parent sets
    with integer programming. It is useful for small sanity checks.
    """
    d = X.shape[1]
    coef = np.zeros(d)
    if not parents:
        return coef
    Xp = X[:, parents]
    y = X[:, child]
    beta, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    coef[list(parents)] = beta
    return coef


def exhaustive_order_search(
    X: np.ndarray,
    max_d: int = 8,
    w_threshold: float = 0.3,
) -> ExactOrderResult:
    """Brute-force over all topological orders, for very small d only.

    For each order, every earlier node is allowed as a parent of later nodes.
    This is a simple exact topological-order least-squares benchmark, not the
    exact GOBNILP program used in the paper.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    if d > max_d:
        return ExactOrderResult(
            W_est=np.zeros((d, d)), B_est=np.zeros((d, d), dtype=int), score=np.inf,
            order=tuple(), success=False, message=f"d={d} too large for exhaustive_order_search(max_d={max_d})"
        )

    best_score = np.inf
    best_W = np.zeros((d, d))
    best_order: Tuple[int, ...] = tuple()
    for order in permutations(range(d)):
        W = np.zeros((d, d))
        previous: List[int] = []
        for child in order:
            coef = _fit_linear_parents(X, child, previous)
            W[:, child] = coef
            previous.append(child)
        np.fill_diagonal(W, 0.0)
        score = objective_value(X, W, lambda1=0.0)
        if score < best_score:
            best_score = score
            best_W = W
            best_order = tuple(order)
    B = threshold_matrix(best_W, w_threshold)
    return ExactOrderResult(best_W, B, float(best_score), best_order, True)


def gobnilp_placeholder_instructions() -> str:
    return (
        "The paper's global optimum comparison uses GOBNILP, which is an external exact "
        "Bayesian-network solver and is not reimplemented here. To reproduce that part exactly, "
        "install GOBNILP/SCIP separately, export local scores, and compare its output with "
        "NOTEARS using notears_repro.metrics.count_accuracy and notears_repro.notears.objective_value."
    )
