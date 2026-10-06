"""ch.21  Brute force and quantum computers: the optimal trading trajectory as an integer search, at the size one
Mac Studio can finish (docs/hardware.md: 4 workers, hours not days).

The chapter discretises a non-convex problem so that every feasible answer can be enumerated: capital is cut into K
whole units, each ordered allocation of those units across N assets becomes a weight vector, each vector may be long or
short, and a trajectory is one such vector per horizon. Nothing about the objective is used except its value, so
non-continuous costs and ill-conditioned covariances are no obstacle; the price is that the search is exponential in
the number of horizons.

- partition_count      21.5.1  C(K + N - 1, N - 1), the number of ordered allocations of K units to N assets
- partitions           21.5.1  those allocations, order mattering: (1, 2, 3) and (3, 2, 1) are different
- feasible_weights     21.5.2  the set Omega: |w_i| = p_i / K for each partition, times every sign pattern in
                       {-1, 1}^N, so every vector satisfies sum |w_i| = 1
- transaction_costs    21.3    tau_1 from the initial holding, tau_h from the previous horizon, each a per-asset
                       factor times the square root of the change in allocation
- trajectory_sharpe    21.3    sum_h (mu_h' w_h - tau_h) / sqrt(sum_h w_h' V_h w_h)
- evaluate_trajectories 21.5.3 the best trajectory over Phi, the Cartesian product of Omega with H repetitions
- static_solution      21.6.2  the trajectory made of each horizon's local optimum, scored with costs
- random_matrix        21.6.1  a Gaussian matrix of a chosen rank
- problem_params       21.6.1  H sets of mu, V and cost factors

The full case is out of reach here and that is a measured fact, not an opinion: see trajectory_count and
docs/claims/ch21.md for the number of trajectories against this machine's rate.
"""
from __future__ import annotations

from math import comb

import numpy as np

from pmlab.afml.parallel import mp_pandas_obj


def partition_count(k: int, n: int) -> int:
    """21.5.1: the number of whole-number solutions of x_1 + ... + x_n = k, which is C(k + n - 1, n - 1)."""
    return comb(int(k) + int(n) - 1, int(n) - 1)


def partitions(k: int, n: int) -> np.ndarray:
    """21.5.1: every ordered allocation of k whole units to n slots, one row each, in lexicographic order.

    Order matters, so (1, 2, 3) and (3, 2, 1) are both present; the row count is partition_count(k, n).
    """
    k, n = int(k), int(n)
    if n < 1 or k < 0:
        raise ValueError(f"partitions needs n >= 1 and k >= 0, got k={k}, n={n}")
    if n == 1:
        return np.array([[k]], dtype=int)
    rows = []
    for first in range(k + 1):
        rest = partitions(k - first, n - 1)
        rows.append(np.column_stack([np.full(len(rest), first, dtype=int), rest]))
    return np.vstack(rows)


def feasible_weights(k: int, n: int, unique: bool = False) -> np.ndarray:
    """21.5.2: the set Omega of weight vectors, |w_i| = p_i / k over every partition, with every sign pattern.

    The full-investment constraint sum_i |w_i| = 1 holds by construction. The book's set has
    partition_count(k, n) * 2^n members; a partition with a zero gives the same vector under both signs of that slot,
    so unique=True drops those repeats (which is what makes the search worth running here).
    """
    if int(k) < 1:
        raise ValueError(f"feasible_weights needs k >= 1 whole units of capital, got {k}")
    p = partitions(k, n) / float(k)
    signs = np.array(list(np.ndindex(*(2,) * int(n))), dtype=int) * 2 - 1
    omega = (p[:, None, :] * signs[None, :, :]).reshape(-1, int(n))
    return np.unique(omega, axis=0) if unique else omega


def transaction_costs(traj: np.ndarray, cost: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """21.3: tau, one cost per horizon. tau_h = sum_n c_{n,h} sqrt(|w_{n,h} - w_{n,h-1}|), with w_{n,0} = w0."""
    traj, cost, w0 = np.asarray(traj, float), np.asarray(cost, float), np.asarray(w0, float)
    previous = np.vstack([w0[None, :], traj[:-1]])
    return (cost * np.sqrt(np.abs(traj - previous))).sum(axis=1)


def trajectory_sharpe(traj: np.ndarray, mu: np.ndarray, cov: np.ndarray, cost: np.ndarray, w0: np.ndarray) -> float:
    """21.3: SR[r] = sum_h (mu_h' w_h - tau_h) / sqrt(sum_h w_h' V_h w_h) for the trajectory traj (H x N)."""
    traj = np.asarray(traj, float)
    gain = float((np.asarray(mu, float) * traj).sum() - transaction_costs(traj, cost, w0).sum())
    var = float(np.einsum("hi,hij,hj->", traj, np.asarray(cov, float), traj))
    if var <= 0:
        # a riskless trajectory: with a rank-deficient V this is a real point of the feasible set (21.5.3 uses no
        # property of the objective), so it is ranked by its gain and not deleted by being called the worst.
        return float("inf") if gain > 0 else (float("-inf") if gain < 0 else 0.0)
    return gain / np.sqrt(var)


def trajectory_count(k: int, n: int, horizons: int, unique: bool = False) -> int:
    """21.5.3: |Phi| = |Omega|^H, the number of trajectories an exhaustive search has to score."""
    size = len(feasible_weights(k, n, unique=True)) if unique else partition_count(k, n) * 2 ** int(n)
    return size ** int(horizons)


def trajectory_at(index: int, omega: np.ndarray, horizons: int) -> np.ndarray:
    """The index-th trajectory of Phi in lexicographic order: index read as H digits base |Omega|, one row each.

    Phi is enumerated by its index rather than materialised. The chapter's own target case (6 units, 4 assets, 3
    horizons: 1,344 vectors and 2.43e9 trajectories, docs/claims/ch21.md) is 163 GiB of index tuples if the product
    is listed, which is more memory than this machine has; by index it costs nothing.
    """
    base, rows = len(omega), np.empty((int(horizons), omega.shape[1]), dtype=float)
    for h in range(int(horizons) - 1, -1, -1):
        index, digit = divmod(int(index), base)
        rows[h] = omega[digit]
    return rows


def _score_block(mu, cov, cost, w0, omega, molecule):
    """One molecule of the search: the best trajectory among the trajectory indices in molecule (ch.20's atoms).

    Returns (Sharpe ratio, index, trajectory). The index breaks a tie the same way whatever order the molecules come
    back in, so the pool and the serial path answer identically.
    """
    horizons = len(np.asarray(mu, float))
    best_sr, best_index, best = float("-inf"), -1, None
    for index in molecule:
        traj = trajectory_at(index, omega, horizons)
        sr = trajectory_sharpe(traj, mu, cov, cost, w0)
        if best is None or sr > best_sr:
            best_sr, best_index, best = sr, int(index), traj
    return best_sr, best_index, best


def evaluate_trajectories(mu: np.ndarray, cov: np.ndarray, cost: np.ndarray, w0: np.ndarray, omega: np.ndarray,
                          num_threads: int = 1, mp_batches: int = 1, mp_context: str = "spawn"):
    """21.5.3: the trajectory of Phi = Omega x ... x Omega (H times) with the highest Sharpe ratio, and that ratio.

    The search is exhaustive, so the answer is the global optimum of a non-convex problem; the cost is |Omega|^H
    evaluations, which is why the size that fits in an evening is small (docs/claims/ch21.md). Phi is walked by
    index, so the parent holds no list of trajectories and a molecule is a range, whatever |Phi| is.
    """
    horizons = len(np.asarray(mu, float))
    combos = range(len(omega) ** horizons)
    if num_threads <= 1:
        results = [_score_block(mu, cov, cost, w0, omega, combos)]
    else:
        results = mp_pandas_obj(_score_block, ("molecule", combos), num_threads=num_threads, mp_batches=mp_batches,
                                mp_context=mp_context, mu=mu, cov=cov, cost=cost, w0=w0, omega=omega)
    best_sr, _, best = max(results, key=lambda r: (r[0], -r[1]))
    return best_sr, best


def static_solution(mu: np.ndarray, cov: np.ndarray, cost: np.ndarray, w0: np.ndarray, omega: np.ndarray):
    """21.6.2: the trajectory built from each horizon's local optimum, then scored with the costs it incurs.

    Each horizon picks the weight vector with the highest single-horizon mu_h' w / sqrt(w' V_h w), ignoring what the
    move costs; the trajectory is those choices chained, and its Sharpe ratio is the one to beat.
    """
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    rows = []
    for h in range(len(mu)):
        gain = omega @ mu[h]
        var = np.einsum("oi,ij,oj->o", omega, cov[h], omega)
        with np.errstate(divide="ignore", invalid="ignore"):
            riskless = np.where(gain > 0, np.inf, np.where(gain < 0, -np.inf, 0.0))
            score = np.where(var > 0, gain / np.sqrt(var), riskless)   # as trajectory_sharpe ranks a riskless point
        rows.append(omega[int(np.argmax(score))])
    traj = np.vstack(rows)
    return trajectory_sharpe(traj, mu, cov, cost, w0), traj


def random_matrix(n_obs: int, size: int, sigma: float, rank: int, rng=None) -> np.ndarray:
    """21.6.1: an n_obs x size Gaussian matrix whose columns span `rank` dimensions, plus noise of scale sigma."""
    rng = np.random.default_rng() if rng is None else rng
    base = rng.normal(size=(n_obs, int(rank)))
    loadings = rng.normal(size=(int(rank), int(size)))
    return base @ loadings + sigma * rng.normal(size=(n_obs, int(size)))


def problem_params(size: int, horizons: int, n_obs: int = 500, sigma: float = 1.0, rank: int | None = None, rng=None):
    """21.6.1: mu (H x N), V (H x N x N) and the cost factors C (H x N) of a test problem, drawn per horizon."""
    rng = np.random.default_rng() if rng is None else rng
    size, horizons = int(size), int(horizons)
    rank = size if rank is None else int(rank)
    mu = np.vstack([rng.normal(size=size) / 100.0 for _ in range(horizons)])
    cov = np.stack([np.atleast_2d(np.cov(random_matrix(n_obs, size, sigma, rank, rng), rowvar=False))
                    for _ in range(horizons)])   # np.cov of one column is 0-d, which the einsum cannot use
    cost = np.vstack([rng.uniform(0.0, 0.01, size=size) for _ in range(horizons)])
    return mu, cov, cost


__all__ = ["partition_count", "partitions", "feasible_weights", "transaction_costs", "trajectory_sharpe",
           "trajectory_count", "trajectory_at", "evaluate_trajectories", "static_solution", "random_matrix",
           "problem_params"]
