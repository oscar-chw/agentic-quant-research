import math

import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab import vol
from pmlab.fair import _prob

S = 1_800_000_000


def series(r, start=S):
    return pd.Series(np.exp(np.cumsum(np.r_[math.log(60000), r])), index=np.arange(start, start + len(r) + 1))


@pytest.mark.property
def test_trailing_features_ignore_the_future_and_the_target_ignores_the_past():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1e-4, 20_000)
    i = 15_000
    changed = r.copy()
    changed[i + 1:] *= 10
    for k in (300, 1800, 14400):
        assert np.array_equal(vol.trailing_mean_sq(r, k)[:i + 1], vol.trailing_mean_sq(changed, k)[:i + 1], equal_nan=True)
    tl_a, tl_b = vol.timeline(series(r)), vol.timeline(series(changed))
    ra, rb = vol.regressors(tl_a, 300.0), vol.regressors(tl_b, 300.0)
    # instant j carries the return from price j-1 to j; the change starts at return index i + 1
    cut = i + 1
    assert all(np.array_equal(ra[k][:cut], rb[k][:cut], equal_nan=True) for k in ra)
    past = r.copy()
    past[:i] *= 10
    assert vol.forward_mean_sq(r, 300)[i] == pytest.approx(vol.forward_mean_sq(past, 300)[i])
    assert vol.forward_mean_sq(r, 300)[i] == pytest.approx(np.mean(r[i + 1:i + 301] ** 2))
    assert vol.trailing_mean_sq(r, 300)[i] == pytest.approx(np.mean(r[i - 299:i + 1] ** 2))


@pytest.mark.property
def test_har_recovers_a_known_variance_relation_and_its_mean_correction():
    rng = np.random.default_rng(1)
    n = 50_000
    reg = {"a": rng.normal(-18, 1, n), "b": rng.normal(-18, 1, n)}
    target = -3.0 + 0.6 * reg["a"] + 0.2 * reg["b"] + rng.normal(0, 0.5, n)
    har = vol.HAR(["a", "b"]).fit(reg, target, np.arange(n))
    assert har.coef[1:] == pytest.approx([0.6, 0.2], abs=0.02) and har.coef[0] == pytest.approx(-3.0, abs=0.4)   # winsorised
    assert har.resid_var == pytest.approx(0.25, abs=0.03)
    s = har.sigma(reg, np.arange(10))
    assert s ** 2 == pytest.approx(np.exp(har._X(reg, np.arange(10)) @ har.coef + 0.125), rel=0.05)


@pytest.mark.case
def test_time_of_day_profile_uses_only_masked_instants_and_is_read_at_the_forecast_midpoint():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 1e-4, 2 * 86400)
    tl = vol.timeline(series(r, start=S - S % 86400))
    first_day = tl.instants < tl.instants[0] + 86400
    prof = vol.tod_profile(tl, first_day)
    wild = r.copy()
    wild[86400:] *= 100
    assert np.allclose(prof, vol.tod_profile(vol.timeline(series(wild, start=S - S % 86400)), first_day))
    day0 = S - S % 86400
    assert vol.seasonal_at([day0 + 3590], np.arange(288.0))[0] == 12       # (3590 + 150) // 300


@pytest.mark.case
def test_dvol_minute_is_known_only_after_it_closes():
    d = pd.DataFrame({"ts": [S * 1000, (S + 60) * 1000], "close": [40.0, 80.0]})
    got = vol.dvol_at([S + 59, S + 60, S + 119, S + 120], d)
    per_s = lambda v: v / 100 / math.sqrt(365 * 86400)
    assert np.isnan(got[0]) and got[1] == pytest.approx(per_s(40)) and got[2] == pytest.approx(per_s(40))
    assert got[3] == pytest.approx(per_s(80))


@pytest.mark.property
def test_local_scale_at_zero_is_the_baseline_q_and_fit_q_recovers_parameters():
    rng = np.random.default_rng(3)
    n = 60_000
    t = rng.integers(10, 290, n).astype(float)
    u = np.maximum(T - t, 1.0)
    sigma = rng.uniform(3e-5, 1e-4, n)
    m = rng.normal(0, 1, n) * sigma * np.sqrt(u)
    base = {"vol_mult": 1.3, "basis": 4e-5}
    assert np.array_equal(vol.q_prob(m, u, sigma, t, {**base, "a": 0.0, "b": 0.0}, 1e-6),
                          _prob(m, np.maximum(1.3 * sigma, 1e-6), u, 4e-5))
    truth = {**base, "a": 0.3, "b": -0.1}
    y = (rng.uniform(size=n) < vol.q_prob(m, u, sigma, t, truth, 1e-6)).astype(float)
    fit = vol.fit_q(m, u, sigma, t, y, 1e-6, local=True)
    assert fit["a"] == pytest.approx(0.3, abs=0.15) and fit["b"] == pytest.approx(-0.1, abs=0.08)
    plain = vol.fit_q(m, u, sigma, t, y, 1e-6)
    assert set(plain) == {"vol_mult", "basis", "train_log_loss"} and plain["train_log_loss"] >= fit["train_log_loss"] - 1e-9


@pytest.mark.case
def test_qlike_is_zero_for_a_perfect_forecast_and_penalises_under_forecasts_more():
    assert vol.qlike(2.0, 2.0) == pytest.approx(0.0)
    assert vol.qlike(2.0, 1.0) > vol.qlike(1.0, 2.0) > 0
