from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from notears_repro.realdata import run_sachs_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Run Sachs real-data experiment.")
    p.add_argument("--data-path", default="data/sachs.csv")
    p.add_argument("--out-dir", default="results/sachs")
    p.add_argument("--lambda1", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--methods", nargs="+", default=["notears", "notears-l1", "ges", "pc", "lingam"])
    return p.parse_args()


def main():
    args = parse_args()
    df = run_sachs_experiment(args.data_path, args.out_dir, args.lambda1, args.threshold, args.methods)
    print(df)


if __name__ == "__main__":
    main()
