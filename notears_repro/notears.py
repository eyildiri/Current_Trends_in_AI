from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt

from .metrics import threshold_matrix


@dataclass
class NotearsResult:
    W_est: np.ndarray
    B_est: np.ndarray
    history: List[Dict[str, float]] = field(default_factory=list)
    success: bool = True
    message: str = ""


def h_func(W: np.ndarray) -> Tuple[float, np.ndarray]:
    """Smooth acyclicity function and gradient.

    h(W) = tr(expm(W ∘ W)) - d.
    Its gradient is expm(W ∘ W)^T ∘ 2W.
    """
    W = np.asarray(W, dtype=float)
    d = W.shape[0]
    E = slin.expm(W * W)
    h = float(np.trace(E) - d)
    G_h = E.T * (2.0 * W)
    return h, G_h


def squared_loss(X: np.ndarray, W: np.ndarray) -> Tuple[float, np.ndarray]:
    """Least-squares SEM loss: 1/(2n)||X - XW||_F^2 and its gradient."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    n = X.shape[0]
    R = X - X @ W
    loss = 0.5 / n * np.sum(R * R)
    grad = -1.0 / n * X.T @ R
    return float(loss), grad


def _adj_from_split(w: np.ndarray, d: int) -> np.ndarray:
    W = (w[: d * d] - w[d * d :]).reshape(d, d)
    np.fill_diagonal(W, 0.0)
    return W


def _bounds_for_split(d: int):
    bounds = []
    for _ in range(2):
        for i in range(d):
            for j in range(d):
                if i == j:
                    bounds.append((0.0, 0.0))
                else:
                    bounds.append((0.0, None))
    return bounds


def linear_notears(
    X: np.ndarray,
    lambda1: float = 0.0,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
    verbose: bool = False,
    scipy_maxiter: int = 1000,
    return_history: bool = True,
) -> NotearsResult:
    """Learn a weighted DAG using the linear NOTEARS formulation.

    This implementation uses the standard variable splitting trick to support
    L1 regularization with L-BFGS-B:
        W = W_pos - W_neg, W_pos >= 0, W_neg >= 0.
    The L1 penalty becomes lambda1 * sum(W_pos + W_neg).

    Args:
        X: n x d data matrix.
        lambda1: L1 sparsity regularization strength.
        max_iter: Max outer augmented-Lagrangian iterations.
        h_tol: Target acyclicity tolerance.
        rho_max: Maximum quadratic penalty.
        w_threshold: Hard threshold for final adjacency.
        verbose: Print progress.
        scipy_maxiter: Max inner L-BFGS-B iterations.
        return_history: Keep per-outer-iteration diagnostics.
    """
    if lambda1 < 0:
        raise ValueError("lambda1 must be non-negative")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    n, d = X.shape
    w_est = np.zeros(2 * d * d, dtype=float)
    bounds = _bounds_for_split(d)

    rho = 1.0
    alpha = 0.0
    h = np.inf
    history: List[Dict[str, float]] = []
    last_result: Optional[sopt.OptimizeResult] = None

    def _func(w: np.ndarray, rho_value: float, alpha_value: float):
        W = _adj_from_split(w, d)
        loss, G_loss = squared_loss(X, W)
        h_value, G_h = h_func(W)
        smooth_obj = loss + 0.5 * rho_value * h_value * h_value + alpha_value * h_value
        obj = smooth_obj + lambda1 * np.sum(w)
        G_smooth = G_loss + (rho_value * h_value + alpha_value) * G_h
        G_plus = G_smooth + lambda1
        G_minus = -G_smooth + lambda1
        grad = np.concatenate([G_plus.ravel(), G_minus.ravel()])
        return float(obj), grad.astype(float)

    try:
        for outer_iter in range(max_iter):
            w_new = w_est.copy()
            h_new = np.inf
            inner_result = None

            while rho < rho_max:
                inner_result = sopt.minimize(
                    fun=lambda w: _func(w, rho, alpha),
                    x0=w_est,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={"maxiter": scipy_maxiter, "ftol": 1e-12, "gtol": 1e-8},
                )
                w_new = inner_result.x
                W_new = _adj_from_split(w_new, d)
                h_new, _ = h_func(W_new)

                # Augmented-Lagrangian penalty update as in the public NOTEARS code.
                if h_new > 0.25 * h:
                    rho *= 10.0
                    if verbose:
                        print(f"increase rho to {rho:.1e}; h={h_new:.3e}")
                else:
                    break

            w_est = w_new
            W_est = _adj_from_split(w_est, d)
            loss, _ = squared_loss(X, W_est)
            h, _ = h_func(W_est)
            alpha += rho * h
            last_result = inner_result

            row = {
                "outer_iter": float(outer_iter),
                "loss": float(loss),
                "h": float(h),
                "rho": float(rho),
                "alpha": float(alpha),
                "objective_inner": float(inner_result.fun) if inner_result is not None else np.nan,
                "inner_success": float(bool(inner_result.success)) if inner_result is not None else 0.0,
            }
            if return_history:
                history.append(row)
            if verbose:
                print(
                    f"iter={outer_iter:02d} loss={loss:.6f} h={h:.3e} "
                    f"rho={rho:.1e} alpha={alpha:.3e}"
                )
            if h <= h_tol or rho >= rho_max:
                break

        W_est = _adj_from_split(w_est, d)
        B_est = threshold_matrix(W_est, w_threshold)
        success = bool(h <= h_tol or rho < rho_max)
        msg = "ok"
        if last_result is not None and not last_result.success:
            msg = f"last inner optimizer warning: {last_result.message}"
        return NotearsResult(W_est=W_est, B_est=B_est, history=history, success=success, message=msg)

    except Exception as exc:
        W_est = _adj_from_split(w_est, d)
        B_est = threshold_matrix(W_est, w_threshold)
        return NotearsResult(W_est=W_est, B_est=B_est, history=history, success=False, message=str(exc))


def objective_value(X: np.ndarray, W: np.ndarray, lambda1: float = 0.0) -> float:
    loss, _ = squared_loss(X, W)
    return float(loss + lambda1 * np.sum(np.abs(W)))
