import numpy as np
import pytest

from pmlab.afml import fracdiff as fd


@pytest.mark.case
def test_ffd_weights_hand_values_and_integer_orders():
    # (1 - B)^0.5: 1, -1/2, -1/8, -1/16, -5/128
    assert fd.ffd_weights(0.5, 0.03).tolist() == pytest.approx([1, -0.5, -0.125, -0.0625, -0.0390625])
    assert fd.ffd_weights(1.0, 1e-8).tolist() == [1.0, -1.0]
    assert fd.ffd_weights(2.0, 1e-8).tolist() == [1.0, -2.0, 1.0]
    assert fd.ffd_weights(0.0, 1e-8).tolist() == [1.0]


@pytest.mark.case
def test_ffd_at_integer_d_is_the_identity_or_the_difference():
    x = np.array([1.0, 4.0, 2.0, 7.0, 3.0])
    assert np.array_equal(fd.frac_diff_ffd(x, 0.0), x)
    y = fd.frac_diff_ffd(x, 1.0)
    assert np.isnan(y[0]) and y[1:].tolist() == [3.0, -2.0, 5.0, -4.0]


@pytest.mark.property
def test_ffd_matches_a_loop_and_the_expanding_version_once_the_window_is_full():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=400))
    d, thres = 0.35, 1e-3
    w = fd.ffd_weights(d, thres)
    y = fd.frac_diff_ffd(x, d, thres)
    for t in (len(w) - 1, 200, 399):
        assert y[t] == pytest.approx(sum(w[k] * x[t - k] for k in range(len(w))), rel=1e-12)
    assert np.all(np.isnan(y[:len(w) - 1]))
    e = fd.frac_diff_expanding(x, d, thres=0.0)
    # the expanding version has every weight; the gap is the truncated tail times the level
    assert np.nanmax(np.abs(e[-50:] - y[-50:])) < 0.2 * np.std(np.diff(x)) * len(x) ** 0.5


@pytest.mark.property
def test_ffd_undoes_fractional_integration():
    """x = (1 - B)^-d e, built with the expanding weights of -d; FFD of x at d recovers e."""
    rng = np.random.default_rng(1)
    n, d = 6000, 0.4
    e = rng.normal(size=n)
    psi = fd.expanding_weights(-d, n)
    x = np.array([np.dot(psi[:t + 1], e[t::-1]) for t in range(n)])
    y = fd.frac_diff_ffd(x, d, thres=1e-5)
    ok = np.isfinite(y)
    ok[:3000] = False
    assert np.corrcoef(y[ok], e[ok])[0, 1] > 0.999
    # differencing by a wrong order leaves long memory in (0.2) or over-differences (1.0)
    for wrong in (0.2, 1.0):
        z = fd.frac_diff_ffd(x, wrong, thres=1e-4)
        assert np.corrcoef(z[ok], e[ok])[0, 1] < 0.99


@pytest.mark.case
def test_adf_regression_matches_lstsq():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=300))
    r = fd.adf(x, lags=1)
    dx = np.diff(x)
    y, X = dx[1:], np.column_stack([np.ones(298), x[1:-1], dx[:-1]])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = res[0] / (298 - 3)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    assert r["stat"] == pytest.approx(beta[1] / se, rel=1e-9)
    assert r["nobs"] == 298
    assert r["crit5"] == pytest.approx(-2.86154 - 2.8903 / 298 - 4.234 / 298 ** 2 - 40.04 / 298 ** 3)


@pytest.mark.eval
def test_adf_size_and_power_by_monte_carlo():
    """Thresholds: under a random walk (n=250, 3000 paths) the 5% quantile of the statistic is within
    0.12 of the MacKinnon value and 5% +- 1.5% of p-values fall below 0.05; an AR(1) with phi = 0.8
    is rejected in > 95% of paths."""
    rng = np.random.default_rng(3)
    stats, ps = [], []
    for _ in range(3000):
        r = fd.adf(np.cumsum(rng.normal(size=250)))
        stats.append(r["stat"])
        ps.append(r["pvalue"])
    assert abs(np.quantile(stats, 0.05) - fd.adf_critical(248)) < 0.12
    assert abs(np.mean(np.array(ps) < 0.05) - 0.05) < 0.015
    rej = 0
    for _ in range(300):
        e = rng.normal(size=250)
        x = np.zeros(250)
        for t in range(1, 250):
            x[t] = 0.8 * x[t - 1] + e[t]
        rej += fd.adf(x)["stat"] < fd.adf_critical(248)
    assert rej / 300 > 0.95


@pytest.mark.case
def test_pvalue_is_continuous_at_the_regime_switch_and_hits_known_levels():
    assert fd.adf_pvalue(-1.61 - 1e-9) == pytest.approx(fd.adf_pvalue(-1.61 + 1e-9), abs=2e-3)
    assert fd.adf_pvalue(-2.862) == pytest.approx(0.05, abs=0.003)
    assert fd.adf_pvalue(-3.430) == pytest.approx(0.01, abs=0.002)
    assert fd.adf_pvalue(5.0) == 1.0 and fd.adf_pvalue(-30.0) == 0.0


@pytest.mark.case
def test_min_ffd_on_a_random_walk_needs_positive_d_and_keeps_full_correlation_at_zero():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=3000)) + 100
    res = fd.min_ffd(x, ds=np.linspace(0, 1, 11), thres=1e-3)
    t = {r["d"]: r for r in res["table"]}
    assert t[0.0]["corr"] == pytest.approx(1.0)
    assert t[0.0]["weight_sum"] == 1.0 and t[1.0]["weight_sum"] == 0.0
    assert t[0.5]["weight_sum"] == pytest.approx(fd.ffd_weights(0.5, 1e-3).sum()) and 0 < t[0.5]["weight_sum"] < 0.2
    assert t[0.0]["adf"] > t[0.0]["crit5"]            # a random walk does not pass
    assert t[1.0]["adf"] < t[1.0]["crit5"]            # its difference does
    assert res["min_d"] is not None and 0 < res["min_d"] < 0.5      # <= 1 would admit full differencing
    assert t[res["min_d"]]["adf"] < t[res["min_d"]]["crit5"]
    # the memory the differencing keeps is the point of the chapter, so both ends of the correlation column
    assert res["corr_at_min_d"] > 0.99 and t[res["min_d"]]["corr"] == pytest.approx(res["corr_at_min_d"])
    assert t[1.0]["corr"] < 0.06 and t[res["min_d"]]["corr"] > 20 * t[1.0]["corr"]
    below = [r for r in res["table"] if r["d"] < res["min_d"]]
    assert all(r["adf"] >= r["crit5"] for r in below)


@pytest.mark.case
def test_expanding_fracdiff_skips_exactly_the_points_whose_weight_loss_exceeds_the_threshold():
    """5.5's weight-loss rule (docs/afml_sections/ch05.md F5.1): lambda_l = 1 - (sum of the |w| already in the window)
    / (sum of every |w|); the first point kept is the first with lambda_l <= tau."""
    # d = 1 truncates after two weights, so only the first point loses more than tau: it alone is skipped.
    x = np.arange(1.0, 9.0)
    y = fd.frac_diff_expanding(x, 1.0, thres=0.01)
    assert np.isnan(y[0]) and not np.any(np.isnan(y[1:]))
    assert y[1:].tolist() == [1.0] * 7                                 # (1 - B)x on the kept points

    # d = 0.4 on ten points: weights and loss shares by hand from w_k = -w_{k-1} (d - k + 1) / k.
    w = [1.0, -0.4, -0.12, -0.064, -0.0416, -0.029952, -0.0229632, -0.01837056, -0.015155712, -0.0127981568]
    assert fd.expanding_weights(0.4, 10).tolist() == pytest.approx(w, rel=1e-12)
    cum = np.cumsum(np.abs(w))
    loss = 1 - cum / cum[-1]
    assert loss.tolist() == pytest.approx([0.420236, 0.188330, 0.118759, 0.081654, 0.057536,
                                           0.040170, 0.026857, 0.016207, 0.007420, 0.0], abs=1e-6)
    z = fd.frac_diff_expanding(np.arange(1.0, 11.0), 0.4, thres=0.01)
    assert int(np.isnan(z).sum()) == int((loss > 0.01).sum()) == 8      # only the last two points clear tau = 1%
    z5 = fd.frac_diff_expanding(np.arange(1.0, 11.0), 0.4, thres=0.05)
    assert int(np.isnan(z5).sum()) == int((loss > 0.05).sum()) == 5     # a looser tau keeps more points
    assert z[9] == pytest.approx(sum(w[k] * (10.0 - k) for k in range(10)), rel=1e-12)
