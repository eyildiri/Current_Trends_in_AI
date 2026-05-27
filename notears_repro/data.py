from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

GraphType = Literal["ER", "SF"]
SemType = Literal["gauss", "exp", "gumbel"]


def _random_permutation_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    P = np.eye(d)
    return P[rng.permutation(d)]


def _is_dag_binary(B: np.ndarray) -> bool:
    B = (np.asarray(B) != 0).astype(int)
    d = B.shape[0]
    indeg = B.sum(axis=0).astype(int).tolist()
    queue = [i for i in range(d) if indeg[i] == 0]
    seen = 0
    while queue:
        i = queue.pop()
        seen += 1
        for j in np.where(B[i] != 0)[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    return seen == d


def simulate_dag(
    d: int,
    expected_edges: int,
    graph_type: GraphType,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate a random binary DAG.

    The generation follows the spirit of the NOTEARS experiments: random ER or
    scale-free skeletons are made acyclic by orienting edges according to a hidden
    topological order, then permuting node labels.

    Args:
        d: Number of nodes.
        expected_edges: Target number of directed edges. For ER this is used as
            an expectation. For SF it is approximate.
        graph_type: "ER" for Erdős-Rényi or "SF" for scale-free.
        seed: Random seed.

    Returns:
        B: d x d binary adjacency matrix, where B[i, j] = 1 means i -> j.
    """
    if d < 2:
        raise ValueError("d must be at least 2")
    if expected_edges < 0:
        raise ValueError("expected_edges must be non-negative")

    rng = np.random.default_rng(seed)
    graph_type = graph_type.upper()

    if graph_type == "ER":
        max_edges = d * (d - 1) / 2
        p = min(max(expected_edges / max_edges, 0.0), 1.0)
        B = (rng.random((d, d)) < p).astype(float)
        B = np.tril(B, k=-1)  # already acyclic under the natural order

    elif graph_type == "SF":
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("networkx is required for scale-free DAG simulation") from exc

        # For Barabasi-Albert, number of undirected edges is roughly m*(d-m).
        m = max(1, min(d - 1, int(round(expected_edges / max(d, 1)))))
        G = nx.barabasi_albert_graph(d, m, seed=seed)
        order = rng.permutation(d)
        rank = np.empty(d, dtype=int)
        rank[order] = np.arange(d)
        B = np.zeros((d, d), dtype=float)
        for u, v in G.edges():
            if rank[u] < rank[v]:
                B[u, v] = 1.0
            else:
                B[v, u] = 1.0

    else:
        raise ValueError("graph_type must be 'ER' or 'SF'")

    P = _random_permutation_matrix(d, rng)
    B_perm = P.T @ B @ P
    B_perm = (B_perm != 0).astype(float)
    np.fill_diagonal(B_perm, 0)
    if not _is_dag_binary(B_perm):
        raise RuntimeError("internal error: generated graph is not a DAG")
    return B_perm


def simulate_parameter(
    B: np.ndarray,
    weight_low: float = 0.5,
    weight_high: float = 2.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Assign signed random weights to a binary DAG."""
    if weight_low <= 0 or weight_high <= 0 or weight_low >= weight_high:
        raise ValueError("Require 0 < weight_low < weight_high")
    rng = np.random.default_rng(seed)
    B = (np.asarray(B) != 0).astype(float)
    signs = rng.choice([-1.0, 1.0], size=B.shape)
    magnitudes = rng.uniform(weight_low, weight_high, size=B.shape)
    W = B * signs * magnitudes
    np.fill_diagonal(W, 0.0)
    return W


def simulate_linear_sem(
    W: np.ndarray,
    n: int,
    sem_type: SemType = "gauss",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate data from the linear SEM X = XW + Z.

    With row-wise observations, the closed-form solution is
        X = Z (I - W)^{-1}.

    Args:
        W: Weighted adjacency matrix with W[i, j] = i -> j.
        n: Number of samples.
        sem_type: Noise distribution: 'gauss', 'exp', or 'gumbel'.
        noise_scale: Scale of independent noise terms.
        seed: Random seed.
    """
    W = np.asarray(W, dtype=float)
    d = W.shape[0]
    rng = np.random.default_rng(seed)

    sem_type = sem_type.lower()
    if sem_type == "gauss":
        Z = rng.normal(loc=0.0, scale=noise_scale, size=(n, d))
    elif sem_type == "exp":
        Z = rng.exponential(scale=noise_scale, size=(n, d)) - noise_scale
    elif sem_type == "gumbel":
        # Mean of Gumbel(loc=0, scale=s) is EulerGamma * s.
        euler_gamma = 0.5772156649015329
        Z = rng.gumbel(loc=0.0, scale=noise_scale, size=(n, d)) - euler_gamma * noise_scale
    else:
        raise ValueError("sem_type must be 'gauss', 'exp', or 'gumbel'")

    X = Z @ np.linalg.inv(np.eye(d) - W)
    return X.astype(float)


def simulate_dataset(
    d: int,
    expected_edges: int,
    graph_type: GraphType,
    n: int,
    sem_type: SemType,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience helper returning B_true, W_true, X."""
    B_true = simulate_dag(d, expected_edges, graph_type, seed=seed)
    W_true = simulate_parameter(B_true, seed=seed + 1)
    X = simulate_linear_sem(W_true, n=n, sem_type=sem_type, seed=seed + 2)
    return B_true, W_true, X
