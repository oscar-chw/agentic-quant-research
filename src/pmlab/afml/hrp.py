"""Hierarchical risk parity (AFML ch. 16): allocation from a covariance matrix without inverting it.

Three stages, as section 16.4 describes them:
1. Tree clustering (16.4.1): correlation distance d_ij = sqrt((1 - rho_ij) / 2); the distance between items is the
   Euclidean distance between their columns of that matrix (a distance of distances, d~); items are merged by
   single linkage (the nearest point algorithm, the book's example and Snippet 16.4) into an (N - 1) x 4 linkage.
2. Quasi-diagonalisation (16.4.2): the last merge is replaced by its two constituents, recursively, until only
   original items remain; the order keeps similar items next to each other.
3. Recursive bisection (16.4.3): every list of more than one item is cut in half (the first half int(n / 2) items),
   each half's variance is w' V w with w its inverse-variance weights, and the parent's weight is split
   alpha = 1 - V1 / (V1 + V2) to the first half and 1 - alpha to the second. Weights stay in [0, 1] and sum to 1.
For a diagonal covariance the result is the inverse-variance portfolio (Appendix 16.A.2).

The appendix experiments (16.A.3-16.A.4):
- generate_correlated: size0 independent normal series and size1 copies of randomly chosen ones plus normal noise of
  sd sigma1, the data of the book's numerical example (Snippet 16.4); hrp() is that snippet's allocation.
- min_variance_weights: the long-only minimum-variance portfolio, the solution CLA returns as its last turning point.
  The book runs the CLA package; a bounded quadratic programme gives the same portfolio.
- monte_carlo: the out-of-sample experiment of Snippet 16.5. Each iteration simulates size0 + size1 series (the copies
  correlated with their sources), adds two common shocks (-0.5 and +2 to one source and its first copy) and two
  specific shocks (-0.5 and +2 to the source of the last copy) after the first in-sample window, re-estimates the
  covariance on the trailing s_length observations every `rebal` observations, holds each method's weights for the
  next `rebal` returns and records the compounded out-of-sample return. The book reports the variance of that return
  across iterations: CLA 0.1157, IVP 0.0928, HRP 0.0671.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def correlation_distance(corr) -> np.ndarray:
    """d_ij = sqrt((1 - rho_ij) / 2), in [0, 1]."""
    c = np.asarray(corr, dtype=float)
    return np.sqrt(np.clip((1.0 - c) / 2.0, 0.0, None))


def column_distance(corr) -> np.ndarray:
    """d~_ij: Euclidean distance between columns i and j of the correlation distance matrix (square form)."""
    d = correlation_distance(corr)
    diff = d[:, :, None] - d[:, None, :]
    return np.sqrt(np.sum(diff ** 2, axis=0))


def tree_clustering(corr, method: str = "single") -> np.ndarray:
    """The (N - 1) x 4 linkage matrix: merged items, their distance d~, items in the new cluster."""
    return linkage(pdist(correlation_distance(corr), metric="euclidean"), method=method)


def quasi_diagonal(link) -> list[int]:
    """Original items in the order the tree leaves them: each cluster replaced by its two constituents, last first."""
    link = np.asarray(link)
    n = link.shape[0] + 1
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while any(i >= n for i in order):
        expanded = []
        for i in order:
            expanded += [int(link[i - n, 0]), int(link[i - n, 1])] if i >= n else [i]
        order = expanded
    return order


def inverse_variance(cov) -> np.ndarray:
    """w_n = (1 / V_nn) / sum_m (1 / V_mm)."""
    iv = 1.0 / np.diag(np.asarray(cov, dtype=float))
    return iv / iv.sum()


def cluster_variance(cov, items) -> float:
    """w' V w over `items`, w their inverse-variance weights."""
    sub = np.asarray(cov, dtype=float)[np.ix_(items, items)]
    w = inverse_variance(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov, order) -> np.ndarray:
    """Weights in the original item order."""
    w = np.ones(len(order))
    lists = [list(order)]
    while lists:
        lists = [part for items in lists if len(items) > 1
                 for part in (items[: len(items) // 2], items[len(items) // 2:])]
        for first, second in zip(lists[::2], lists[1::2]):
            v1, v2 = cluster_variance(cov, first), cluster_variance(cov, second)
            alpha = 1.0 - v1 / (v1 + v2)
            w[first] *= alpha
            w[second] *= 1.0 - alpha
    return w


def hrp(cov, corr=None, method: str = "single") -> dict:
    """{"weights": weights in the original order, "order": the quasi-diagonal order, "link": the linkage}.
    Every variance must be positive (an item that never varies has no inverse variance)."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("cov must be a non-empty square matrix")
    if not np.all(np.diag(cov) > 0):
        raise ValueError("every variance must be positive")
    if cov.shape[0] == 1:
        return {"weights": np.ones(1), "order": [0], "link": np.empty((0, 4))}
    if corr is None:
        sd = np.sqrt(np.diag(cov))
        corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    link = tree_clustering(corr, method)
    order = quasi_diagonal(link)
    return {"weights": recursive_bisection(cov, order), "order": order, "link": link}


def generate_correlated(n_obs: int, size0: int, size1: int, sigma1: float, rng=None) -> tuple[np.ndarray, list[int]]:
    """(n_obs x (size0 + size1) observations, the source column of each copy)."""
    rng = np.random.default_rng(rng)
    x = rng.normal(0.0, 1.0, (n_obs, size0))
    cols = rng.integers(0, size0, size1).tolist()
    return np.hstack([x, x[:, cols] + rng.normal(0.0, sigma1, (n_obs, size1))]), cols


def min_variance_weights(cov) -> np.ndarray:
    """argmin w' V w subject to sum w = 1 and 0 <= w <= 1."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    w0 = inverse_variance(cov)
    res = minimize(lambda w: w @ cov @ w, w0, jac=lambda w: 2 * cov @ w, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                   options={"ftol": 1e-15, "maxiter": 500})
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


METHODS = {"ivp": lambda cov: inverse_variance(cov), "hrp": lambda cov: hrp(cov)["weights"],
           "min_variance": min_variance_weights}


def monte_carlo(n_iter: int = 10_000, n_obs: int = 520, size0: int = 5, size1: int = 5, mu0: float = 0.0,
                sigma0: float = 1e-2, sigma1f: float = 0.25, s_length: int = 260, rebal: int = 22, rng=None) -> dict:
    """{method: compounded out-of-sample returns per iteration} and {"variance": {method: variance across iterations}}."""
    rng = np.random.default_rng(rng)
    out = {m: np.empty(n_iter) for m in METHODS}
    for it in range(n_iter):
        x = rng.normal(mu0, sigma0, (n_obs, size0))
        cols = rng.integers(0, size0, size1).tolist()
        x = np.hstack([x, x[:, cols] + rng.normal(0.0, sigma0 * sigma1f, (n_obs, size1))])
        point = rng.integers(s_length, n_obs - 1, 2)
        x[np.ix_(point, [cols[0], size0])] = np.array([[-0.5, -0.5], [2.0, 2.0]])
        point = rng.integers(s_length, n_obs - 1, 2)
        x[point, cols[-1]] = np.array([-0.5, 2.0])
        returns = {m: [] for m in METHODS}
        for start in range(s_length, n_obs, rebal):
            cov = np.cov(x[start - s_length:start], rowvar=False)
            held = x[start:start + rebal]
            for m, weights in METHODS.items():
                returns[m].append(held @ weights(cov))
        for m in METHODS:
            out[m][it] = np.prod(1.0 + np.concatenate(returns[m])) - 1.0
    return out | {"variance": {m: float(np.var(out[m], ddof=1)) for m in METHODS}}
