import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pmlab import WINDOW_SECONDS as T
from pmlab import pq
from pmlab.fair import (L, FairParams, FairPricer, _prob, drift_units, fair_window, fit, rule_lookback,
                        variance_units, window_inputs)

START = 1_800_000_000
LOOKBACKS = (1, 30, 60)                  # point price, 30 s TWAP, 60 s TWAP (fair.rule_lookback)
RULES = Path(__file__).parent / "fixtures" / "rules"


def synthetic_close(rng, start=START, sigma=5e-5, s0=75_000.0, before=180, after=360):
    """1s kline closes indexed by open second over [start-before-1, start+after-1]; price at i = close[i-1]."""
    instants = np.arange(start - before, start + after + 1)
    x = math.log(s0) + np.cumsum(rng.normal(0, sigma, len(instants)))
    return pd.Series(np.exp(x), index=instants - 1)


def settle(close, start, lag=0, basis=0.0, rng=None, L=L):
    price = lambda lo, hi: close.reindex(np.arange(lo, hi)).mean()       # instants lo+1..hi
    a_end = price(start + T - L - lag, start + T - lag)
    k = price(start - L - lag, start - lag)
    noise = rng.normal(0, basis) if basis else 0.0
    return math.log(a_end) - math.log(k) + noise >= 0


@pytest.mark.case
def test_variance_units_are_continuous_and_vanish_after_settlement():
    assert variance_units(T - L) == pytest.approx(61 * 121 / 360)
    assert variance_units(T - L - 1e-9) == pytest.approx(variance_units(T - L), abs=1e-6)
    assert variance_units(T - L + 1) == pytest.approx(59 * 60 * 119 / (6 * 3600))
    assert variance_units(T) == 0 and variance_units(T - 3, lag=3) == 0
    assert variance_units(100, lag=5) == pytest.approx(variance_units(105))


@pytest.mark.case
def test_certain_outcomes_are_exact():
    assert list(_prob(np.array([1e-4, -1e-4, 0.0]), 1e-5, 0.0, 0.0)) == [1.0, 0.0, 1.0]
    assert _prob(0.0, 1e-5, 0.0, 1e-4) == pytest.approx(0.5)            # basis alone keeps doubt


@pytest.mark.property
@pytest.mark.parametrize("t,lag,basis,L", [(100, 0, 0.0, 60), (270, 0, 0.0, 60), (270, 3, 2e-4, 60), (298, 0, 3e-4, 60),
                                           (100, 0, 0.0, 30), (280, 2, 1e-4, 30), (100, 0, 0.0, 1), (296, 3, 0.0, 1)])
def test_fair_matches_monte_carlo_settlement(t, lag, basis, L):
    rng = np.random.default_rng(t + lag)
    sigma, n = 1e-4, 40_000
    close = synthetic_close(np.random.default_rng(1), sigma=sigma)
    w = window_inputs(close, START, lag, halflife=1e9, L=L)
    q = float(_prob(w["mean_minus_k"][t], sigma, w["units"][t], basis))
    price_at = lambda i: close.loc[i - 1]                                # price AT instant i
    lo, hi = START + T - L - lag + 1, START + T - lag                     # settling instants
    known = [price_at(i) for i in range(lo, min(hi, START + t) + 1)]
    steps = max(hi - (START + t), 0)
    future = np.exp(math.log(price_at(START + t)) + np.cumsum(rng.normal(0, sigma, (n, steps)), axis=1))
    future = future[:, -min(steps, L):] if steps else np.empty((n, 0))
    a_end = (np.sum(known) + future.sum(axis=1)) / L
    k = np.mean([price_at(i) for i in range(START - L - lag + 1, START - lag + 1)])
    up = np.log(a_end) - math.log(k) + rng.normal(0, basis, n) * (basis > 0) >= 0
    assert len(known) + future.shape[1] == L
    assert up.mean() == pytest.approx(q, abs=4 * math.sqrt(q * (1 - q) / n) + 2e-3)


@pytest.mark.property
@pytest.mark.parametrize("L", LOOKBACKS)
@pytest.mark.parametrize("lag", [0, 2])
def test_streaming_pricer_equals_batch(lag, L):
    close = synthetic_close(np.random.default_rng(3))
    p = FairParams(halflife=45, vol_mult=1.3, basis=1e-5, lag=lag, sigma_floor=4.5e-5)  # floor binds sometimes
    batch = fair_window(close, START, p, L=L)
    pricer = FairPricer(p)
    pricer.watch(START, L=L)
    stream = []
    for sec, price in close.items():
        instant = sec + 1
        pricer.on_second(instant, price)
        if START <= instant < START + T:
            stream.append(pricer.fair(START))
    assert np.allclose(stream, batch, atol=1e-9)
    # z must be the backtest's z_q exactly (a P-model feature), not Φ⁻¹(fair), which saturates
    pricer2, zs = FairPricer(p), []
    pricer2.watch(START, L=L)
    for sec, price in close.items():
        pricer2.on_second(sec + 1, price)
        if START <= sec + 1 < START + T:
            zs.append(pricer2.z(START))
    w = window_inputs(close, START, p.lag, p.halflife, L)
    sigma = np.maximum(w["sigma_raw"] * p.vol_mult, p.sigma_floor)
    z_batch = w["mean_minus_k"] / np.sqrt(sigma ** 2 * np.maximum(w["units"], 1e-12) + p.basis ** 2)
    assert np.allclose(zs, z_batch, rtol=1e-9, atol=1e-9)


@pytest.mark.property
@pytest.mark.parametrize("L", LOOKBACKS)
def test_fair_at_t_ignores_every_later_price(L):
    close = synthetic_close(np.random.default_rng(4))
    p = FairParams(halflife=60)
    before = fair_window(close, START, p, L=L)
    for cut in (0, 120, 250, 299):
        shocked = close.copy()
        shocked.loc[START + cut:] *= 1.01                                # prices at instants > cut
        after = fair_window(shocked, START, p, L=L)
        assert np.array_equal(before[: cut + 1], after[: cut + 1])
        assert not np.allclose(before[cut + 1:], after[cut + 1:]) or cut == 299


@pytest.mark.eval
def test_fit_recovers_lag_basis_and_vol_on_synthetic_markets():
    """Threshold: true lag chosen; vol_mult within 15% of 1; basis within 2x of truth."""
    rng = np.random.default_rng(0)
    true_lag, true_basis = 2, 2e-5   # a basis much above the lag-induced K shift (~1e-5) hides the lag
    closes, outcomes = {}, {}
    for i in range(1200):
        start = START + 300 * i
        closes[start] = synthetic_close(rng, start=start, sigma=5e-5)
        outcomes[start] = settle(closes[start], start, true_lag, true_basis, rng)
    grid = list(range(0, 240, 20)) + list(range(240, 300, 2))
    params, table = fit(closes, outcomes, grid=grid, lags=range(0, 5), halflives=(1e9,))
    assert params.lag == true_lag, table.head()
    assert params.vol_mult == pytest.approx(1.0, rel=0.15)
    assert true_basis / 2 <= params.basis <= true_basis * 2, params


@pytest.mark.eval
def test_fit_finds_a_volatility_floor_when_quiet_spells_hide_the_settlement_noise():
    """Threshold: the fitted floor is the larger grid value, and its log loss is lower than no floor."""
    rng = np.random.default_rng(8)
    closes, outcomes = {}, {}
    for i in range(900):
        start = START + 300 * i
        quiet = i % 3 == 0
        close = synthetic_close(rng, start=start, sigma=4e-6 if quiet else 5e-5)
        settle_path = synthetic_close(rng, start=start, sigma=4e-5) if quiet else close   # settlement source keeps moving
        closes[start] = close
        outcomes[start] = settle(settle_path, start) if not quiet else bool(rng.random() < 0.5)
    params, table = fit(closes, outcomes, grid=range(0, 300, 20), lags=(0,), halflives=(60.0,),
                        sigma_floors=(1e-6, 3e-5))
    assert params.sigma_floor == 3e-5
    losses = table.set_index("sigma_floor")["log_loss"]
    assert losses[1e-6] > losses[3e-5]


# ---------------------------------------------------------------- every settlement rule (L = 1, 30, 60)

@pytest.mark.property
@pytest.mark.parametrize("L", LOOKBACKS)
@pytest.mark.parametrize("lag", [0, 3])
def test_units_telescope_to_the_slopes_for_every_lookback(L, lag):
    """u(t) − u(t+1) = a(t)² and D(t) − D(t+1) = a(t), both 0 once settled, and u, D are the sums of a², a still to come."""
    t = np.arange(-5, T + 3)
    a = pq.mean_slope(t, lag, L)
    assert np.allclose(variance_units(t, lag, L) - variance_units(t + 1, lag, L), a ** 2, rtol=0, atol=1e-10)
    assert np.allclose(drift_units(t, lag, L) - drift_units(t + 1, lag, L), a, rtol=0, atol=1e-10)
    assert variance_units(T - lag, lag, L) == 0 and drift_units(T - lag, lag, L) == 0
    brute = np.array([np.sum(np.clip((T - lag - np.arange(s, T + 5)) / L, 0, 1) ** p) for s in (0, 150, T - lag - L // 2 - 1)
                      for p in (2, 1)])
    closed = np.array([f(s, lag, L) for s in (0, 150, T - lag - L // 2 - 1) for f in (variance_units, drift_units)])
    assert np.allclose(closed, brute, rtol=0, atol=1e-9)
    assert (a[t >= T - lag] == 0).all() and (a[t <= T - lag - L] == 1).all()


@pytest.mark.property
@pytest.mark.parametrize("L", LOOKBACKS)
def test_q_is_a_martingale_for_every_lookback(L):
    """E_Q[p(t+1) | F_t] = p(t) by Gauss–Hermite quadrature over the next 1 s move, at every regime of each rule."""
    z, wts = np.polynomial.hermite_e.hermegauss(80)
    wts = wts / wts.sum()
    sigma = 1e-4
    for t, lag, basis in [(0, 0, 0.0), (T - L - 1, 0, 3e-5), (T - L, 0, 0.0), (T - max(L // 2, 1), 0, 2e-5),
                          (290, 3, 4e-5), (298, 0, 1e-5)]:
        u0, u1 = float(variance_units(t, lag, L)), float(variance_units(t + 1, lag, L))
        a = pq.mean_slope(t, lag, L)
        for d in (-1.1, 0.2, 1.7):
            m = d * math.sqrt(sigma ** 2 * u0 + basis ** 2)
            if u0 == 0 and basis == 0:
                continue                                                  # settled: p is a constant step
            sd1 = math.sqrt(sigma ** 2 * u1 + basis ** 2)
            if sd1 < a * sigma:          # p(t+1) steeper than the move: quadrature cannot resolve it, convolve exactly
                p_next = float(norm.cdf(m / math.hypot(a * sigma, sd1)))
            else:
                p_next = float(np.sum(wts * _prob(m + a * sigma * z, sigma, u1, basis)))
            assert p_next == pytest.approx(float(_prob(m, sigma, u0, basis)), abs=1e-9), (t, lag, d)


@pytest.mark.property
@pytest.mark.parametrize("lag,basis", [(0, 0.0), (3, 4e-5)])
def test_point_price_rule_is_the_textbook_digital(lag, basis):
    """L = 1: p(t) = Φ((x_t − k)/√(σ²τ + basis²)), k = log price at start − lag, τ = T − lag − t; settled at x_{T−lag}."""
    close = synthetic_close(np.random.default_rng(12), sigma=8e-5)
    p = FairParams(halflife=1e9, vol_mult=1.0, basis=basis, lag=lag, sigma_floor=1e-12)
    got = fair_window(close, START, p, L=1)
    w = window_inputs(close, START, lag, 1e9, L=1)
    price_at = lambda i: close.loc[i - 1]
    t = np.arange(T)
    x_t = np.log([price_at(START + i) for i in t])
    k = math.log(price_at(START - lag))
    tau = T - lag - t
    x_end = math.log(price_at(START + T - lag))
    sd = np.sqrt(w["sigma_raw"] ** 2 * np.clip(tau, 0, None) + basis ** 2)
    num = np.where(tau > 0, x_t - k, x_end - k)
    textbook = np.where(sd > 0, norm.cdf(num / np.where(sd > 0, sd, 1)), (num >= 0).astype(float))
    assert w["k"] == pytest.approx(k, abs=1e-12)
    assert np.allclose(got, textbook, rtol=0, atol=1e-10)
    assert np.allclose(variance_units(t, lag, 1), np.clip(tau, 0, None)) and np.allclose(drift_units(t, lag, 1), np.clip(tau, 0, None))


@pytest.mark.case
@pytest.mark.parametrize("L", (1, 30))
@pytest.mark.parametrize("t,lag", [(100, 0), (T - 20, 0), (T - 2, 0), (T - 12, 3)])
def test_mean_slope_is_the_batch_response_to_the_next_move_for_other_rules(L, t, lag):
    close = synthetic_close(np.random.default_rng(t))
    h = 1e-3
    bumped = close.copy()
    bumped[bumped.index >= START + t] *= math.exp(h)                      # every price from instant start+t+1 on
    w, wb = window_inputs(close, START, lag, 1e9, L), window_inputs(bumped, START, lag, 1e9, L)
    assert (wb["mean_minus_k"][t + 1] - w["mean_minus_k"][t + 1]) / h == pytest.approx(pq.mean_slope(t, lag, L), abs=1e-8)
    assert wb["mean_minus_k"][t] == w["mean_minus_k"][t]


@pytest.mark.property
@pytest.mark.parametrize("L", (1, 30))
def test_greeks_follow_the_lookback(L):
    """delta, gamma and theta at L = 1, 30 match finite differences of fair._prob with u(t; L) and slope a(t; L)."""
    sigma, basis = 1e-4, 2e-5
    for t in (100, T - L // 2 - 1, T - 2):
        u, a = float(variance_units(t, 0, L)), pq.mean_slope(t, 0, L)
        sd = math.sqrt(sigma ** 2 * u + basis ** 2)
        m = 0.4 * sd
        p_x = lambda dx: float(_prob(m + a * dx, sigma, u, basis))
        hx = 1e-3 * sd / max(a, 1e-9)
        assert pq.delta_log(m, sigma, t, basis, 0, L) == pytest.approx((p_x(hx) - p_x(-hx)) / (2 * hx), rel=1e-5)
        hx = 1e-2 * sd / max(a, 1e-9)
        assert pq.gamma_log(m, sigma, t, basis, 0, L) == pytest.approx((p_x(hx) - 2 * p_x(0) + p_x(-hx)) / hx ** 2, rel=1e-3)
        assert pq.theta(m, sigma, t, basis, 0, L) == pytest.approx(
            float(_prob(m, sigma, variance_units(t + 1, 0, L), basis) - _prob(m, sigma, u, basis)), abs=1e-15)
    assert pq.delta_log(0.0, sigma, 100, 0.0, 0, L) != pq.delta_log(0.0, sigma, 100, 0.0, 0, 60)   # L reaches the Greeks


@pytest.mark.property
def test_one_pricer_prices_windows_under_different_rules_at_once():
    """Per-window L on one stream: each watched window equals its own batch price, including an L above the
    constructor's history length (the history grows on watch)."""
    close = synthetic_close(np.random.default_rng(21), before=400, after=900)
    p = FairParams(halflife=60, vol_mult=1.1, basis=2e-5, lag=2, sigma_floor=1e-6)
    rules = {START: 1, START + 300: 30, START + 600: 120}
    pricer = FairPricer(p)
    for s, lb in rules.items():
        pricer.watch(s, L=lb)
    stream = {s: [] for s in rules}
    for sec, price in close.items():
        pricer.on_second(sec + 1, price)
        for s in rules:
            if s <= sec + 1 < s + T:
                stream[s].append(pricer.fair(s))
    for s, lb in rules.items():
        assert np.allclose(stream[s], fair_window(close, s, p, L=lb), rtol=0, atol=1e-9), lb
    assert not np.allclose(fair_window(close, START + 300, p, L=30), fair_window(close, START + 300, p, L=60))


# ---------------------------------------------------------------- reading the rule from a market

@pytest.mark.case
@pytest.mark.parametrize("name,expected", [("twap60", 60), ("twap30", 30), ("point", 1)])
def test_rule_lookback_reads_real_market_descriptions(name, expected):
    """Fixtures: Gamma markets starting 2026-09-02, 2026-08-13 and 2026-07-19 (fetched 2026-09-17)."""
    market = json.loads((RULES / f"{name}.json").read_text())
    assert rule_lookback(market) == expected
    text_only = {**market, "twap_lookback": None}
    assert rule_lookback(text_only) == expected                           # the text alone settles it
    assert rule_lookback({"description": market["description"]}) == expected
    assert rule_lookback({"resolutionSource": market["resolutionSource"]}) == expected


@pytest.mark.case
def test_rule_lookback_prefers_the_normalised_field_and_refuses_unknown_rules():
    point = json.loads((RULES / "point.json").read_text())
    assert rule_lookback({**point, "twap_lookback": 30}) == 30
    assert rule_lookback({**point, "twap_lookback": np.int64(60)}) == 60
    assert rule_lookback({**point, "twap_lookback": float("nan")}) == 1   # a missing value in a parquet column
    assert rule_lookback({**point, "twap_lookback": True}) == 1           # a bool is not a lookback
    for bad in ({}, {"description": "resolves Up if the TWAP of Bitcoin is higher"}, {"twap_lookback": None}):
        with pytest.raises(ValueError):
            rule_lookback(bad)


# ----------------------------------------------------------------------------- code review (research/code_review/pricing.md)

@pytest.mark.case
def test_an_unknown_k_prices_nan_even_when_sd_is_zero():
    """fair 2: with basis 0 (the FairParams default) a window whose K is unknown priced exactly 0 after settlement."""
    got = _prob(np.array([np.nan, 1e-4, -1e-4]), 5e-5, 0.0, 0.0)
    assert np.isnan(got[0]) and list(got[1:]) == [1.0, 0.0]
    assert np.isnan(_prob(np.nan, 5e-5, 10.0, 1e-4))


@pytest.mark.case
def test_the_batch_pricer_refuses_the_partial_history_the_streaming_pricer_refuses():
    """fair 3: closes from start − 30 with L = 60 gave a batch price (0.4479 at t = 10) where FairPricer gives NaN;
    closes from start + 10 gave no error and read σ from the end of the array."""
    close = synthetic_close(np.random.default_rng(8))
    p = FairParams(halflife=45, vol_mult=1.2, basis=1e-5, lag=0)
    short = close[close.index >= START - 30]
    batch = fair_window(short, START, p, L=60)
    pricer = FairPricer(p)
    pricer.watch(START, L=60)
    stream = []
    for sec, price in short.items():
        pricer.on_second(sec + 1, price)
        if START <= sec + 1 < START + T:
            stream.append(pricer.fair(START))
    assert np.isnan(batch).all() and np.isnan(stream).all()
    gap = close.drop(START - 20)                                                   # one K close missing
    assert np.isnan(window_inputs(gap, START, 0, 45, 60)["k"])
    assert np.isfinite(window_inputs(close, START, 0, 45, 60)["k"])
    with pytest.raises(ValueError):
        fair_window(close[close.index >= START + 10], START, p, L=60)
