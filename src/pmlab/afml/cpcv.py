"""Combinatorial purged cross-validation (AFML ch.12) and the probability of backtest overfitting
(AFML 11.6; Bailey, Borwein, López de Prado and Zhu, 2017).

CPCV (12.4):
- The observations, in time order, are cut into N contiguous groups. Every choice of k groups is a
  test set, so there are C(N, k) splits (12.4.1).
- Each group is tested in C(N-1, k-1) splits, so the out-of-sample predictions assemble into
  phi = k/N C(N, k) complete backtest paths (12.4.1). paths() assigns each (split, group) to a path
  in split order: the j-th time a group is tested it belongs to path j, which is the assignment of
  12.4.1's second figure for N = 6, k = 2 (pinned in tests/test_afml_cpcv.py).
- Training drops every observation whose label span [t0, t1) overlaps a test group's envelope and
  every observation starting within `embargo` after it; with non-adjacent test groups, each group is
  purged and embargoed separately (12.4.2, purging as in 7.4).
- path_mean_variance is 12.5's argument for CPCV: the phi path statistics are correlated, so the
  variance of their mean is phi^-1 sigma^2 (1 + (phi - 1) rho_bar), not sigma^2 / phi. mean_correlation
  estimates rho_bar from the paths' own return series.
- walk_forward_average_sample is 12.2.1's third pitfall: how much of the sample stands behind the
  first half of a walk-forward's decisions.

PBO by CSCV (Bailey et al. 2017):
- A T x N matrix of per-period returns for N strategy configurations is cut into S equal blocks.
  For each of the C(S, S/2) ways to pick half the blocks as in-sample, the strategy with the best
  in-sample metric is found; its out-of-sample rank among the N gives w = rank / (N + 1) and the logit
  lambda = ln(w / (1 - w)). PBO is the share of combinations with lambda <= 0 (the in-sample winner is
  at or below the out-of-sample median).
- Also reported: the regression of out-of-sample on in-sample performance of the winner (performance
  degradation) and the probability that the winner loses out of sample.
- The default metric is the Sharpe ratio per period. A strategy with no variation in a half (in practice: no
  trade there, its untraded periods being 0 returns) scores 0, what it earned, not -inf, which would rank an
  idle half below every losing strategy.
"""
from itertools import combinations
from math import comb

import numpy as np


def n_splits(n_groups: int, k_test: int) -> int:
    return comb(n_groups, k_test)


def n_paths(n_groups: int, k_test: int) -> int:
    """phi = k / N C(N, k) = C(N - 1, k - 1)."""
    return comb(n_groups, k_test) * k_test // n_groups


def walk_forward_average_sample(n_obs: float, warm_up: float) -> dict:
    """12.2.1, third pitfall: a walk-forward with a warm-up of t0 observations out of T makes (T - t0) / 2 of its
    decisions - the first half - on (T + 3 t0) / 4 observations on average, a fraction 1/4 + (3/4) t0 / T of the
    sample. Raising the warm-up raises that fraction but shortens the backtest to T - t0 decisions."""
    if not 0 <= warm_up <= n_obs or n_obs <= 0:
        raise ValueError("0 <= warm_up <= n_obs is required")
    avg = (n_obs + 3.0 * warm_up) / 4.0
    return {"decisions": n_obs - warm_up, "half_decisions": (n_obs - warm_up) / 2.0,
            "average_sample": avg, "fraction": avg / n_obs}


def split_groups(n_groups: int, k_test: int) -> list[tuple[int, ...]]:
    return list(combinations(range(n_groups), k_test))


def paths(n_groups: int, k_test: int) -> np.ndarray:
    """(n_paths, n_groups) array: entry [j, g] is the split whose test set supplies group g to path j. Splits are
    taken in lexicographic order and a group joins the path numbered by how often it has been tested so far, the
    book's own assignment (12.4.1, the figure after Figure 12.1)."""
    splits = split_groups(n_groups, k_test)
    out = np.full((n_paths(n_groups, k_test), n_groups), -1, dtype=int)
    seen = np.zeros(n_groups, dtype=int)
    for s, groups in enumerate(splits):
        for g in groups:
            out[seen[g], g] = s
            seen[g] += 1
    return out


def group_index(t0, n_groups: int) -> np.ndarray:
    """Group id of each observation: N contiguous blocks of (nearly) equal size in t0 order."""
    t0 = np.asarray(t0)
    order = np.argsort(t0, kind="stable")
    gid = np.empty(len(t0), dtype=int)
    for g, block in enumerate(np.array_split(order, n_groups)):
        gid[block] = g
    return gid


def cpcv_splits(t0, t1, n_groups: int, k_test: int, embargo: int = 0) -> list[dict]:
    """One dict per split: train and test index arrays and the test groups."""
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    gid = group_index(t0, n_groups)
    env = [(t0[gid == g].min(), t1[gid == g].max()) for g in range(n_groups)]
    out = []
    for groups in split_groups(n_groups, k_test):
        test = np.isin(gid, groups)
        keep = ~test
        for g in groups:
            lo, hi = env[g]
            keep &= ~((t0 < hi) & (t1 > lo))                # purge: span overlaps the test group
            keep &= ~((t0 >= hi) & (t0 < hi + embargo))     # embargo after the test group
        out.append({"train": np.flatnonzero(keep), "test": np.flatnonzero(test), "groups": groups})
    return out


def assemble_paths(split_predictions: list[np.ndarray], t0, n_groups: int, k_test: int) -> np.ndarray:
    """(n_paths, n_obs) out-of-sample predictions. split_predictions[s] has one value per observation
    (NaN outside split s's test set)."""
    gid = group_index(t0, n_groups)
    P = paths(n_groups, k_test)
    out = np.full((P.shape[0], len(gid)), np.nan)
    for j in range(P.shape[0]):
        for g in range(n_groups):
            rows = gid == g
            out[j, rows] = np.asarray(split_predictions[P[j, g]])[rows]
    return out


def sharpe_columns(R) -> np.ndarray:
    """Mean / sd per column; 0 for a column with no variation (an idle half earns nothing)."""
    R = np.asarray(R, dtype=float)
    sd = R.std(axis=0, ddof=1)
    return np.divide(R.mean(axis=0), sd, out=np.zeros(R.shape[1]), where=sd > 0)


def mean_correlation(Y) -> float:
    """Average off-diagonal Pearson correlation among the rows of Y (one series per path). NaN with fewer than two
    usable rows; rows without variation are dropped, since their correlation is undefined."""
    Y = np.asarray(Y, dtype=float)
    Y = Y[np.isfinite(Y).all(axis=1) & (Y.std(axis=1) > 0)]
    if len(Y) < 2:
        return float("nan")
    C = np.corrcoef(Y)
    off = C[np.triu_indices_from(C, k=1)]
    off = off[np.isfinite(off)]
    return float(off.mean()) if len(off) else float("nan")


def path_mean_variance(y, rho: float) -> dict:
    """12.5: the variance of the mean of phi correlated path statistics {y_j}.

    sigma^2[mu] = phi^-2 (phi sigma^2 + phi (phi - 1) sigma^2 rho_bar) = phi^-1 sigma^2 (1 + (phi - 1) rho_bar),
    which for 0 <= rho_bar < 1 lies in [sigma^2 / phi, sigma^2): the independent bound below and the variance of a
    single path above. `effective_paths` is sigma^2 / sigma^2[mu], the number of independent paths that would give
    the same precision - the honest denominator when the paths reuse the same per-split forecasts.

    sigma^2 is the MARGINAL variance of a path statistic. The spread observed across phi equicorrelated paths
    estimates sigma^2 (1 - rho_bar) instead, so passing it makes the variance of the mean about that factor too
    small - a caveat the section does not state, pinned in
    tests/test_afml_cpcv.py::test_the_spread_across_correlated_paths_understates_their_marginal_variance."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    phi = len(y)
    nan = float("nan")
    if phi < 2 or not np.isfinite(rho):
        return {"paths": phi, "mean": float(y.mean()) if phi else nan, "variance": nan, "rho": float(rho),
                "variance_of_mean": nan, "sd_of_mean": nan, "variance_if_independent": nan,
                "variance_if_identical": nan, "effective_paths": nan}
    s2 = float(y.var(ddof=1))
    var_mean = s2 * (1.0 + (phi - 1) * float(rho)) / phi
    return {"paths": phi, "mean": float(y.mean()), "variance": s2, "rho": float(rho),
            "variance_of_mean": var_mean, "sd_of_mean": float(np.sqrt(var_mean)) if var_mean >= 0 else nan,
            "variance_if_independent": s2 / phi, "variance_if_identical": s2,
            "effective_paths": float(s2 / var_mean) if var_mean > 0 else float("inf")}


def pbo(M, n_blocks: int = 16, metric=sharpe_columns, max_combinations: int | None = None, rng=None) -> dict:
    """Probability of backtest overfitting by CSCV on a (T, N) returns matrix."""
    M = np.asarray(M, dtype=float)
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    T, N = M.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    if max_combinations and len(combos) > max_combinations:
        rng = np.random.default_rng(rng)
        combos = [combos[i] for i in rng.choice(len(combos), max_combinations, replace=False)]
    lam, is_best, oos_best, oos_median, winners = [], [], [], [], []
    for c in combos:
        ins = np.concatenate([blocks[b] for b in c])
        oos = np.concatenate([blocks[b] for b in range(n_blocks) if b not in c])
        r_is, r_oos = metric(M[ins]), metric(M[oos])
        best = int(np.argmax(r_is))
        winners.append(best)
        rank = 1 + np.sum(r_oos < r_oos[best]) + 0.5 * (np.sum(r_oos == r_oos[best]) - 1)   # 1 = worst, N = best
        w = rank / (N + 1)
        lam.append(np.log(w / (1 - w)))
        is_best.append(r_is[best])
        oos_best.append(r_oos[best])
        oos_median.append(np.median(r_oos))
    lam, is_best, oos_best = map(np.asarray, (lam, is_best, oos_best))
    ok = np.isfinite(is_best) & np.isfinite(oos_best)
    slope = intercept = np.nan
    if ok.sum() > 2 and np.ptp(is_best[ok]) > 0:
        slope, intercept = np.polyfit(is_best[ok], oos_best[ok], 1)
    return {"pbo": float(np.mean(lam <= 0)), "lambda": lam, "combinations": len(combos), "n_strategies": N,
            "n_periods": T, "n_blocks": n_blocks, "degradation_slope": float(slope),
            "degradation_intercept": float(intercept), "prob_oos_loss": float(np.mean(oos_best < 0)),
            "is_best": is_best, "oos_best": oos_best, "oos_median": np.asarray(oos_median),
            "winner": np.asarray(winners)}
