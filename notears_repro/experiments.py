from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baselines import run_baseline
from .data import simulate_dataset
from .metrics import count_accuracy, edge_precision_recall, is_dag, l2_weight_error
from .notears import linear_notears, objective_value
from .utils import ensure_dir, standardize


def _evaluate_one(B_true, W_true, X, method: str, lambda1: float, w_threshold: float, seed: int, baseline_kwargs=None):
    baseline_kwargs = baseline_kwargs or {}
    start = time.perf_counter()
    method_lower = method.lower()

    if method_lower in {"notears", "notears-l1", "notears_l1"}:
        lam = lambda1 if method_lower != "notears" else 0.0
        result = linear_notears(X, lambda1=lam, w_threshold=w_threshold, verbose=False)
        W_est = result.W_est
        B_est = result.B_est
        success = result.success
        message = result.message
        extra = {
            "h_final": result.history[-1]["h"] if result.history else np.nan,
            "outer_iters": len(result.history),
            "objective": objective_value(X, W_est, lambda1=lam),
        }
        pretty_method = "NOTEARS-L1" if lam > 0 else "NOTEARS"
    else:
        result = run_baseline(method, X, **baseline_kwargs)
        W_est = result.W_est if result.W_est is not None else np.zeros_like(W_true)
        B_est = result.B_est
        success = result.success
        message = result.message
        extra = {"h_final": np.nan, "outer_iters": np.nan, "objective": np.nan}
        pretty_method = result.name

    runtime = time.perf_counter() - start
    metrics = count_accuracy(B_true, B_est)
    metrics.update(edge_precision_recall(B_true, B_est))
    metrics.update(
        {
            "method": pretty_method,
            "runtime_sec": runtime,
            "success": bool(success),
            "message": message,
            "is_dag": bool(is_dag(B_est)),
            "weight_l2": l2_weight_error(W_true, W_est) if W_est is not None else np.nan,
            "seed": seed,
        }
    )
    metrics.update(extra)
    return metrics, W_est, B_est


def run_synthetic_experiment(
    out_dir: str | Path,
    d_values: Sequence[int] = (10, 20, 50, 100),
    n_values: Sequence[int] = (20, 1000),
    graph_types: Sequence[str] = ("ER", "SF"),
    sem_types: Sequence[str] = ("gauss", "exp", "gumbel"),
    edge_multipliers: Dict[str, int] | None = None,
    seeds: Sequence[int] = tuple(range(10)),
    methods: Sequence[str] = ("notears", "notears-l1", "corr"),
    lambda1: float = 0.1,
    w_threshold: float = 0.3,
    standardize_data: bool = True,
    save_matrices: bool = False,
) -> pd.DataFrame:
    """Run the synthetic reproduction grid.

    This covers the central experiment family from the paper: ER/SF graphs,
    Gaussian/Exponential/Gumbel noise, multiple d, multiple n, and SHD/FDR metrics.
    """
    out_dir = ensure_dir(out_dir)
    matrices_dir = ensure_dir(out_dir / "matrices")
    edge_multipliers = edge_multipliers or {"ER": 2, "SF": 4}
    rows: List[Dict] = []

    total = len(d_values) * len(n_values) * len(graph_types) * len(sem_types) * len(seeds)
    pbar = tqdm(total=total, desc="synthetic datasets")

    for d in d_values:
        for n in n_values:
            for graph_type in graph_types:
                expected_edges = int(edge_multipliers.get(graph_type.upper(), 2) * d)
                for sem_type in sem_types:
                    for seed in seeds:
                        dataset_seed = 100000 * d + 1000 * n + 100 * seed + (0 if graph_type.upper() == "ER" else 50)
                        B_true, W_true, X = simulate_dataset(
                            d=d,
                            expected_edges=expected_edges,
                            graph_type=graph_type,
                            n=n,
                            sem_type=sem_type,
                            seed=dataset_seed,
                        )
                        if standardize_data:
                            X = standardize(X)

                        if save_matrices:
                            prefix = f"d{d}_n{n}_{graph_type}_{sem_type}_seed{seed}"
                            np.save(matrices_dir / f"{prefix}_B_true.npy", B_true)
                            np.save(matrices_dir / f"{prefix}_W_true.npy", W_true)

                        for method in methods:
                            metrics, W_est, B_est = _evaluate_one(
                                B_true, W_true, X, method=method, lambda1=lambda1, w_threshold=w_threshold, seed=seed
                            )
                            metrics.update(
                                {
                                    "d": d,
                                    "n": n,
                                    "graph_type": graph_type.upper(),
                                    "sem_type": sem_type.lower(),
                                    "expected_edges": expected_edges,
                                    "lambda1": lambda1 if method.lower() != "notears" else 0.0,
                                    "w_threshold": w_threshold,
                                }
                            )
                            rows.append(metrics)
                            if save_matrices:
                                np.save(matrices_dir / f"{prefix}_{metrics['method']}_W_est.npy", W_est)
                                np.save(matrices_dir / f"{prefix}_{metrics['method']}_B_est.npy", B_est)
                        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synthetic_metrics.csv", index=False)
    return df


def quick_synthetic(out_dir: str | Path) -> pd.DataFrame:
    """Small run for debugging the pipeline."""
    return run_synthetic_experiment(
        out_dir=out_dir,
        d_values=(5, 10),
        n_values=(100,),
        graph_types=("ER",),
        sem_types=("gauss",),
        seeds=(0, 1),
        methods=("notears", "notears-l1", "corr"),
        lambda1=0.1,
        w_threshold=0.3,
        save_matrices=True,
    )
