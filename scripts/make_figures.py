from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd

from notears_repro.plots import plot_metric_lines, save_summary_tables


def parse_args():
    p = argparse.ArgumentParser(description="Make plots from synthetic_metrics.csv")
    p.add_argument("--metrics-csv", default="results/synthetic/synthetic_metrics.csv")
    p.add_argument("--out-dir", default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)
    save_summary_tables(df, out_dir)
    for metric in ["shd", "fdr", "tpr", "nnz", "runtime_sec"]:
        if metric in df.columns:
            plot_metric_lines(df, metric, out_dir / f"{metric}.png")
    print(f"Saved figures/tables to {out_dir}")


if __name__ == "__main__":
    main()
