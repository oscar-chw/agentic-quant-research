import math

import numpy as np
import pytest

from pmlab import stats

N = 2000


def ar1(phi, n, seed):
    e = np.random.default_rng(seed).standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.property
def test_normal_sample_passes_every_normality_test():
    # Seed 1: all three tests pass (jb .71, shapiro .88, dagostino .69). Seed 0's second
    # draw fails one test at 5%, as ~1 in 7 normal samples will.
    r = stats.normality(np.random.default_rng(1).standard_normal(N))
    assert r["n"] == N and r["normal"] is True
    assert r["jarque_bera"][1] > 0.01
    assert r["dagostino"] is not None and "shapiro_subsampled" not in r


@pytest.mark.property
def test_fat_tails_and_flat_tails_are_rejected():
    t = stats.normality(np.random.default_rng(0).standard_t(3, N))
    assert t["jarque_bera"][1] < 1e-6 and t["normal"] is False
    u = stats.normality(np.random.default_rng(0).uniform(size=N))
    assert u["shapiro"][1] < 1e-6 and u["normal"] is False


@pytest.mark.property
def test_shapiro_subsamples_above_5000():
    r = stats.normality(np.random.default_rng(1).standard_normal(6000))
    assert r["n"] == 6000 and r["shapiro_subsampled"] is True
    assert r["shapiro"] == stats.normality(np.random.default_rng(1).standard_normal(6000))["shapiro"]


@pytest.mark.case
def test_small_samples_return_none_instead_of_raising():
    r = stats.normality([1.0, 2.0, float("nan")])
    assert r["n"] == 2 and r["normal"] is None and r["shapiro"] is None
    r = stats.normality(np.arange(10.0))
    assert r["dagostino"] is None and r["shapiro"] is not None and r["normal"] is not None


@pytest.mark.case
def test_ljung_box_matches_hand_computation():
    x = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
    n, h = len(x), 2
    mean = sum(x) / n
    denom = sum((v - mean) ** 2 for v in x)
    q = 0.0
    for k in range(1, h + 1):
        rho = sum((x[t] - mean) * (x[t - k] - mean) for t in range(k, n)) / denom
        q += rho ** 2 / (n - k)
    q *= n * (n + 2)
    got_q, got_p = stats.ljung_box(x, lags=h)
    assert got_q == pytest.approx(q, rel=1e-12)
    assert got_p == pytest.approx(math.exp(-q / 2), rel=1e-12)   # chi2(2) survival = e^(-q/2)


@pytest.mark.property
def test_ljung_box_separates_noise_from_ar1():
    assert stats.ljung_box(np.random.default_rng(0).standard_normal(N))[1] > 0.01
    assert stats.ljung_box(ar1(0.3, N, seed=0))[1] < 1e-6


@pytest.mark.property
def test_serial_correlation_pairs_x_t_with_x_t_minus_lag():
    r = stats.serial_correlation(ar1(0.3, N, seed=0), lag=1)
    assert r["lag"] == 1 and r["n"] == N - 1
    assert r["pearson"][0] == pytest.approx(0.3, abs=0.06) and r["significant"]


@pytest.mark.property
def test_rank_correlation_sees_monotone_dependence_pearson_misses():
    x = np.random.default_rng(0).uniform(size=N)
    r = stats.correlation(x, np.exp(5 * x))
    assert r["spearman"][0] == pytest.approx(1.0, abs=1e-12)
    assert r["kendall"][0] == pytest.approx(1.0, abs=1e-12)
    assert r["pearson"][0] < 0.95


@pytest.mark.property
def test_one_outlier_fools_pearson_not_spearman():
    x, y = np.random.default_rng(0).standard_normal((2, N))
    x[0] = y[0] = 1e3
    r = stats.correlation(x, y)
    assert abs(r["pearson"][0]) > 0.5
    assert abs(r["spearman"][0]) < 0.1


@pytest.mark.property
def test_recommendation_follows_normality():
    t1, t2 = np.random.default_rng(0).standard_t(3, (2, N))
    assert stats.correlation(t1, t2)["recommended"] == "spearman"
    # Seed 1: both rows pass normality (see test_normal_sample_passes_every_normality_test).
    n1, n2 = np.random.default_rng(1).standard_normal((2, N))
    assert stats.correlation(n1, n2)["recommended"] == "pearson"


@pytest.mark.case
def test_holm_step_down_with_monotone_adjustment():
    # sorted .005,.01,.03,.04 x 4,3,2,1 = .02,.03,.06,.04 -> running max .02,.03,.06,.06
    reject, adjusted = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert reject == [True, False, False, True]
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


@pytest.mark.case
def test_holm_caps_at_one_and_leaves_missing_pvalues_out_of_the_family():
    # m = 3: sorted .01,.6,.9 x 3,2,1 = .03,1.2,.9 -> running max .03,1.2,1.2 -> cap 1
    reject, adjusted = stats.holm([0.01, None, float("nan"), 0.6, 0.9])
    assert reject == [True, False, False, False, False]
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[3] == 1.0 and adjusted[4] == 1.0
    assert math.isnan(adjusted[1]) and math.isnan(adjusted[2])


@pytest.mark.property
def test_battery_reports_every_key_as_a_finite_number():
    b = stats.battery(np.random.default_rng(1).standard_normal(N))
    numeric = ["mean", "std", "skew", "excess_kurtosis", "jb_p", "shapiro_p", "dagostino_p",
               "lag1_pearson", "lag1_pearson_p", "lag1_spearman", "lag1_spearman_p",
               "lag1_kendall", "lag1_kendall_p", "ljung_box_q", "ljung_box_p"]
    assert set(b) == set(numeric) | {"n", "normal", "recommended", "lag1_significant"}
    for k in numeric:
        assert isinstance(b[k], float) and math.isfinite(b[k]), k
    assert b["n"] == N and b["normal"] is True and b["recommended"] == "pearson"
    assert isinstance(b["lag1_significant"], bool)


@pytest.mark.case
def test_constant_series_is_not_declared_normal():
    assert stats.normality(np.full(50, 0.3))["normal"] is None


@pytest.mark.case
def test_within_group_pairs_never_cross_a_boundary():
    x = np.array([1, 2, 3, 10, 20, 30.])
    g = np.array([0, 0, 0, 1, 1, 1])
    a, b = stats.within_group_pairs(x, g, 1)
    assert list(zip(a, b)) == [(2, 1), (3, 2), (20, 10), (30, 20)]
    assert len(stats.within_group_pairs(x, g, 3)[0]) == 0


@pytest.mark.property
def test_grouped_statistics_recover_within_window_dependence():
    rng = np.random.default_rng(4)
    xs, gs = [], []
    for w in range(3000):                       # 7 returns per window, AR(1) 0.3 inside, windows independent
        e = rng.normal(0, 1, 7)
        for t in range(1, 7):
            e[t] += 0.3 * e[t - 1]
        xs.extend(e); gs.extend([w] * 7)
    xs, gs = np.array(xs), np.array(gs)
    a, b = stats.within_group_pairs(xs, gs, 1)
    pooled = np.corrcoef(xs[1:], xs[:-1])[0, 1]
    within = np.corrcoef(a, b)[0, 1]
    assert abs(within - 0.29) < 0.03 and within - pooled > 0.02
    q, p = stats.ljung_box_grouped(xs, gs, lags=3)
    assert p < 1e-6
    iid = rng.normal(0, 1, len(xs))
    assert stats.ljung_box_grouped(iid, gs, lags=3)[1] > 0.01


@pytest.mark.property
def test_cluster_bootstrap_p_is_honest_under_within_window_heteroskedasticity():
    rng = np.random.default_rng(5)
    rejections = 0
    for trial in range(60):
        xs, gs = [], []
        for w in range(150):
            scale = np.linspace(1, 20, 7)             # variance explodes toward expiry, no dependence
            xs.extend(rng.standard_t(3, 7) * scale); gs.extend([w] * 7)
        stat = lambda x, g: np.corrcoef(*stats.within_group_pairs(x, g, 1))[0, 1]
        _, _, p = stats.cluster_bootstrap_p(np.array(xs), np.array(gs), stat, n_boot=150, seed=trial)
        rejections += p < 0.05
    assert rejections / 60 <= 0.12


@pytest.mark.case
def test_normal_only_when_no_available_test_rejects():
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, 400)
    assert stats.normality(normal)["normal"] is True
    skewed = np.concatenate([normal, [8.0, 9.0, 10.0]])       # JB and D'Agostino reject, sample still "looks" fine
    res = stats.normality(skewed)
    assert res["jarque_bera"][1] < 0.05 and res["normal"] is False


@pytest.mark.property
def test_ljung_box_sign_flip_keeps_its_size_under_window_heteroskedasticity_and_still_finds_ar1():
    """Audit R6 finding 8: with each window's own scale and a within-window scale profile, the χ² Ljung–Box on grouped
    bar returns rejected 30–64% of the time with no dependence at all."""
    rng = np.random.default_rng(8)
    windows, per = 300, 20
    g = np.repeat(np.arange(windows), per)
    scale = np.repeat(np.exp(rng.normal(0, 1.0, windows)), per) * np.tile(np.linspace(0.3, 3.0, per), windows)
    old = new = 0
    reps = 60
    for _ in range(reps):
        x = scale * rng.standard_normal(len(g))
        old += stats.ljung_box_grouped(x, g, lags=5)[1] < 0.05
        new += stats.ljung_box_signflip(x, g, lags=5, n_boot=199, seed=int(rng.integers(1e9)))[1] < 0.05
    assert old / reps >= 0.2 and new / reps <= 0.12
    e = rng.standard_normal(len(g))
    ar = e.copy()
    for i in range(1, len(g)):
        if g[i] == g[i - 1]:
            ar[i] = 0.3 * ar[i - 1] + e[i]
    q, p = stats.ljung_box_signflip(scale * ar, g, lags=5, n_boot=199)
    assert p < 0.02 and q == pytest.approx(stats.ljung_box_grouped(scale * ar - np.mean(scale * ar), g, lags=5)[0])
