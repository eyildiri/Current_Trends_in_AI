import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path

from notears_repro.data import simulate_dataset
from notears_repro.metrics import count_accuracy, is_dag
from notears_repro.notears import linear_notears
from notears_repro.utils import standardize


def main():
    B_true, W_true, X = simulate_dataset(d=5, expected_edges=8, graph_type="ER", n=200, sem_type="gauss", seed=1)
    X = standardize(X)
    res = linear_notears(X, lambda1=0.1, w_threshold=0.3, verbose=True, scipy_maxiter=300)
    print("success:", res.success, res.message)
    print("acyclic:", is_dag(res.B_est))
    print(count_accuracy(B_true, res.B_est))


if __name__ == "__main__":
    main()
