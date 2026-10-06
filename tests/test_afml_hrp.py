import math

import numpy as np
import pytest

from pmlab.afml import hrp


@pytest.mark.case
def test_generated_copies_have_the_correlation_their_noise_implies_and_hrp_groups_them():
    x, cols = hrp.generate_correlated(10_000, 5, 5, 0.25, rng=0)
    assert x.shape == (10_000, 10) and len(cols) == 5
    corr = np.corrcoef(x, rowvar=False)
    for k, src in enumerate(cols):
        assert corr[5 + k, src] == pytest.approx(1 / np.sqrt(1 + 0.25 ** 2), abs=0.005)
    res = hrp.hrp(np.cov(x, rowvar=False))
    order = res["order"]
    for k, src in enumerate(cols):                                  # each copy sits in the block of its source
        members = [src] + [5 + j for j, s in enumerate(cols) if s == src]
        where = sorted(order.index(m) for m in members)
        assert where[-1] - where[0] == len(members) - 1
    assert res["weights"].sum() == pytest.approx(1.0) and np.all(res["weights"] > 0)


@pytest.mark.property
def test_min_variance_weights_satisfy_the_long_only_optimality_conditions():
    two = np.array([[0.04, 0.006], [0.006, 0.09]])
    w1 = (two[1, 1] - two[0, 1]) / (two[0, 0] + two[1, 1] - 2 * two[0, 1])
    assert hrp.min_variance_weights(two) == pytest.approx([w1, 1 - w1], abs=1e-6)
    corner = np.array([[0.01, 0.018], [0.018, 0.04]])               # the unconstrained solution shorts asset 2
    assert hrp.min_variance_weights(corner) == pytest.approx([1.0, 0.0], abs=1e-6)
    rng = np.random.default_rng(1)
    for _ in range(20):
        A = rng.normal(size=(60, 8)) @ np.diag(rng.uniform(0.5, 2, 8)) + rng.normal(size=(60, 1)) * 2
        cov = np.cov(A, rowvar=False)
        w = hrp.min_variance_weights(cov)
        grad = cov @ w
        lam = grad[w > 1e-6].mean()
        assert np.all(np.abs(grad[w > 1e-6] - lam) < 1e-5 * abs(lam) + 1e-9)
        assert np.all(grad[w <= 1e-6] >= lam - 1e-6)


@pytest.mark.eval
def test_hrp_has_the_lowest_out_of_sample_variance_in_the_books_monte_carlo():
    """16.A.4 with the book's settings and 200 iterations (the book runs 10,000: CLA 0.1157, IVP 0.0928, HRP 0.0671).
    Threshold: var(IVP) > 1.1 var(HRP) and var(minimum variance) > 1.3 var(HRP); over seeds 1-8 at 150 iterations the
    ratios ranged 1.27-1.46 and 1.50-1.96. var(HRP) within 50% of the book's 0.0671 (0.084 here)."""
    res = hrp.monte_carlo(200, rng=0)
    v = res["variance"]
    assert v["ivp"] > 1.1 * v["hrp"]
    assert v["min_variance"] > 1.3 * v["hrp"]
    assert v["hrp"] == pytest.approx(0.0671, rel=0.5)                                  # the book's scale, not just order


@pytest.mark.property
def test_the_correlation_distance_is_a_true_metric():
    """16.A.1: d = sqrt((1 - rho) / 2) is a linear multiple of the Euclidean distance between the z-standardised
    series, so it inherits non-negativity, coincidence, symmetry and sub-additivity. Checked on random correlation
    matrices, and against the Euclidean identity d[x, y] = sqrt(4 T) d[X, Y] on standardised data."""
    rng = np.random.default_rng(5)
    for _ in range(40):
        n = int(rng.integers(3, 9))
        a = rng.normal(size=(int(rng.integers(n + 5, 200)), n))
        corr = np.corrcoef(a, rowvar=False)
        d = hrp.correlation_distance(corr)
        assert np.all(d >= 0) and np.allclose(d, d.T)                       # non-negativity and symmetry
        assert np.allclose(np.diag(d), 0, atol=1e-7)                        # coincidence
        assert np.all(d[~np.eye(n, dtype=bool)] > 1e-3)                     # distinct series are a distance apart
        assert np.all(d[:, :, None] <= d[:, None, :] + d[None, :, :] + 1e-12)   # sub-additivity, over every triple

    T = 500
    X = rng.normal(size=(T, 2)) @ np.array([[1.0, 0.8], [0.0, 0.6]])
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    euclid = np.sqrt(((z[:, 0] - z[:, 1]) ** 2).sum())
    rho = np.corrcoef(X, rowvar=False)[0, 1]
    assert euclid == pytest.approx(np.sqrt(4 * T) * np.sqrt((1 - rho) / 2))
    assert hrp.correlation_distance(np.corrcoef(X, rowvar=False))[0, 1] == pytest.approx(euclid / np.sqrt(4 * T))


@pytest.mark.case
def test_markowitzs_curse_the_condition_number_rises_with_correlation_and_the_inverse_destabilises():
    """16.3: the condition number of an equicorrelated N x N correlation matrix is (1 + (N - 1) rho) / (1 - rho), so
    it grows without bound as the investments become substitutes -- exactly when diversification is most wanted. The
    quadratic optimiser inverts that matrix: as rho rises its unconstrained minimum-variance weights ask for short
    positions, and a perturbation of the covariance of relative size 1e-3 moves them by 0.0015, 0.008, 0.041 and
    0.446 in total absolute weight at rho = 0, 0.6, 0.9 and 0.99. HRP inverts nothing (16.1)."""
    N, sd = 10, np.linspace(0.1, 0.3, 10)

    def equicorrelated(rho):
        c = np.full((N, N), float(rho))
        np.fill_diagonal(c, 1.0)
        return c

    def unconstrained(cov):
        w = np.linalg.solve(cov, np.ones(N))
        return w / w.sum()

    rng = np.random.default_rng(0)
    conds, moves, smallest = [], [], []
    for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
        corr = equicorrelated(rho)
        assert np.linalg.cond(corr) == pytest.approx((1 + (N - 1) * rho) / (1 - rho))
        cov = corr * np.outer(sd, sd)
        base = unconstrained(cov)
        shifts = []
        for _ in range(100):
            e = rng.normal(0, 1e-3, (N, N))
            shifts.append(np.abs(unconstrained(cov + (e + e.T) / 2 * np.outer(sd, sd)) - base).sum())
        conds.append(np.linalg.cond(corr))
        moves.append(float(np.mean(shifts)))
        smallest.append(float(base.min()))
    assert conds == sorted(conds) and moves == sorted(moves)                 # instability tracks the condition number
    assert moves[-1] > 100 * moves[0]
    assert smallest[0] > 0 > smallest[1] and smallest == sorted(smallest, reverse=True)

    singular = equicorrelated(0.99) * np.outer(sd, sd)
    singular[:, 0] = singular[0, :] = singular[:, 1] * sd[0] / sd[1]         # column 0 a copy of column 1
    w = hrp.hrp(singular)["weights"]
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)          # HRP needs no inverse


@pytest.mark.case
def test_table_16_1s_three_allocations_on_the_books_numerical_example():
    """16.5, Snippet 16.4's data (10,000 x 10, five series and five noisy copies, noise sd 0.25). The book's stylised
    facts and numbers: the minimum-variance portfolio holds 92.66% in its top five and gives three investments zero
    weight, HRP holds 62.57%, IVP spreads evenly, and in sample sigma_HRP = 0.4640 against sigma_CLA = 0.4486 -- HRP
    pays a little risk in sample for not discarding half the universe. Here (seed 0): CLA 97.4% and four zeros, HRP
    62.63%, sigma_HRP 0.4642, sigma_CLA 0.4498."""
    x, _ = hrp.generate_correlated(10_000, 5, 5, 0.25, rng=0)
    cov = np.cov(x, rowvar=False)
    cla, h, ivp = hrp.min_variance_weights(cov), hrp.hrp(cov)["weights"], hrp.inverse_variance(cov)
    top5 = lambda w: float(np.sort(w)[-5:].sum())
    sigma = lambda w: float(np.sqrt(w @ cov @ w))

    assert np.linalg.cond(np.corrcoef(x, rowvar=False)) == pytest.approx(150.93, rel=0.1)
    assert top5(cla) > 0.9 and (cla < 1e-6).sum() >= 3                       # concentrated, with weights at zero
    assert top5(h) == pytest.approx(0.6257, abs=0.01)
    assert np.all(h > 0) and top5(ivp) < top5(h) < top5(cla)                 # HRP between CLA and IVP
    assert sigma(h) == pytest.approx(0.4640, abs=0.001)
    assert sigma(cla) == pytest.approx(0.4486, abs=0.005)
    assert sigma(cla) < sigma(h) < sigma(ivp)                                # CLA wins in sample, and only in sample


@pytest.mark.eval
def test_a_covariance_of_size_n_needs_the_books_observations_and_hierarchical_risk_parity_does_not():
    """16.3: at least N (N + 1) / 2 independent observations are needed for a non-singular estimate of a size-N
    covariance matrix -- the book's own example, size 50, is 1275 observations, about five years of daily data.
    Thresholds (N = 10, so the bound is 55, true covariance known, 30 draws each): the sample covariance of T <= N
    observations is rank-deficient and its condition number exceeds 1e12, so the optimiser's inverse is meaningless,
    while HRP still returns weights that sum to one; and the minimum-variance weights estimated from fewer
    observations than the bound are at least twice as far from the true-covariance weights as those estimated from
    ten times the bound."""
    assert 50 * 51 // 2 == 1275 == 5 * 255                        # the book's worked case: five years of daily data

    rng = np.random.default_rng(0)
    n = 10
    bound = n * (n + 1) // 2
    a = rng.normal(size=(n, n)) * 0.3
    true = np.diag(np.linspace(0.5, 2.0, n)) + a @ a.T
    chol = np.linalg.cholesky(true)
    w_true = hrp.min_variance_weights(true)

    short = rng.normal(size=(n, n)) @ chol.T                      # T = N: one degree of freedom short already
    cov_short = np.cov(short, rowvar=False)
    assert np.linalg.matrix_rank(cov_short) < n and np.linalg.cond(cov_short) > 1e12
    assert hrp.hrp(cov_short)["weights"].sum() == pytest.approx(1.0)        # no inverse, no problem

    def error(t):
        out = []
        for _ in range(30):
            cov = np.cov(rng.normal(size=(t, n)) @ chol.T, rowvar=False)
            out.append(float(np.abs(hrp.min_variance_weights(cov) - w_true).sum()))
        return float(np.mean(out))

    under, at, over = error(bound // 2), error(bound), error(10 * bound)
    assert under > at > over                                       # estimation error falls as the bound is cleared
    assert under > 2 * over


@pytest.mark.property
def test_the_recursive_bisection_does_linear_work_at_logarithmic_depth(monkeypatch):
    """16.4.3: the algorithm converges in deterministic logarithmic time at best and linear time at worst. Each
    bisection halves a list, so a list of N items is an almost balanced binary tree: N - 1 splits in all (two
    cluster variances each, which is the whole cost), and every item is reached in floor(log2 N) splits at best and
    ceil(log2 N) at worst."""
    real = hrp.cluster_variance
    seen: list[list[int]] = []

    def spy(cov, items):
        seen.append(list(items))
        return real(cov, items)

    monkeypatch.setattr(hrp, "cluster_variance", spy)
    rng = np.random.default_rng(4)
    for n in (2, 3, 5, 8, 11, 16, 31):
        seen.clear()
        x = rng.normal(size=(120, n)) * rng.uniform(0.5, 2.0, n)
        hrp.hrp(np.cov(x, rowvar=False))
        assert len(seen) == 2 * (n - 1), n                         # one split per internal node: work is linear
        reached = [sum(i in part for part in seen) for i in range(n)]
        assert max(reached) == math.ceil(math.log2(n)), n          # worst case
        assert min(reached) == math.floor(math.log2(n)), n         # best case
