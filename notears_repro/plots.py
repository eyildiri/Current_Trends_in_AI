from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def plot_weight_matrices(matrices: Dict[str, np.ndarray], out_path: str | Path, vlim: Optional[float] = None) -> None:
    """Save side-by-side heatmaps of true/estimated weight matrices."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    if vlim is None:
        vlim = max(float(np.max(np.abs(M))) for M in matrices.values())
        vlim = max(vlim, 1e-6)

    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for ax, (title, M) in zip(axes[0], matrices.items()):
        im = ax.imshow(M, vmin=-vlim, vmax=vlim, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("target node")
        ax.set_ylabel("source node")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_lines(
    df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    group_cols=("graph_type", "sem_type", "n"),
) -> None:
    """Plot metric vs d, grouped by method. Keeps a simple Matplotlib style."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    if metric not in df.columns:
        raise ValueError(f"metric {metric} not in dataframe")

    grouped = list(df.groupby(list(group_cols))) if group_cols else [((), df)]
    for key, sub in grouped:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, mdf in sub.groupby("method"):
            agg = mdf.groupby("d")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["d"], agg["mean"], yerr=agg["std"].fillna(0), marker="o", label=method)
        ax.set_xlabel("d (number of nodes)")
        ax.set_ylabel(metric.upper())
        ax.legend()
        title = key if isinstance(key, tuple) else (key,)
        ax.set_title(" | ".join(map(str, title)))
        suffix = "_".join(map(str, title)).replace(" ", "")
        fig.tight_layout()
        fig.savefig(out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}"), dpi=200)
        plt.close(fig)


def save_summary_tables(df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    metrics = [c for c in ["shd", "fdr", "tpr", "fpr", "nnz", "runtime_sec"] if c in df.columns]
    group_cols = [c for c in ["method", "graph_type", "sem_type", "n", "d"] if c in df.columns]
    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
