from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from notears_repro.data import simulate_dataset
from notears_repro.exact import exhaustive_order_search, gobnilp_placeholder_instructions
from notears_repro.metrics import count_accuracy
from notears_repro.notears import linear_notears
from notears_repro.utils import standardize


def parse_args():
    p = argparse.ArgumentParser(description="Tiny exact-order sanity comparison.")
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    B_true, W_true, X = simulate_dataset(args.d, expected_edges=2 * args.d, graph_type="ER", n=args.n, sem_type="gauss", seed=args.seed)
    X = standardize(X)
    nt = linear_notears(X, lambda1=0.0, w_threshold=0.3)
    ex = exhaustive_order_search(X, max_d=8, w_threshold=0.3)
    print("NOTEARS", count_accuracy(B_true, nt.B_est))
    print("Exact-order", count_accuracy(B_true, ex.B_est), "score", ex.score, "order", ex.order)
    print("\n", gobnilp_placeholder_instructions())


if __name__ == "__main__":
    main()
