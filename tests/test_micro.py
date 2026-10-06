import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab.bars.base import Bar
from pmlab.micro import CS_K, amihud, bar_micro_features, corwin_schultz, ols_slope, roll_feature, roll_spread, vpin

S = 1_800_000_000


@pytest.mark.property
def test_roll_recovers_the_bid_ask_spread_from_bouncing_prints():
    rng = np.random.default_rng(0)
    mid = 0.5 + np.cumsum(rng.normal(0, 0.0005, 20_000))        # slow-moving efficient price
    prices = mid + rng.choice([-1, 1], len(mid)) * 0.02 / 2        # spread 0.02
    assert roll_spread(prices) == pytest.approx(0.02, rel=0.1)
    assert roll_spread(mid) == pytest.approx(0.0, abs=0.003)       # no bounce, no spread


@pytest.mark.case
def test_corwin_schultz_matches_hand_formula_in_price_units_and_floors_at_zero():
    assert CS_K == pytest.approx(3 - 2 * np.sqrt(2))
    h1, l1, h2, l2 = 0.52, 0.48, 0.53, 0.49
    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
    gamma = np.log(0.53 / 0.48) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = max((np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k), 0)
    relative = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    assert corwin_schultz(h1, l1, h2, l2) == pytest.approx(relative * (h1 * l1 * h2 * l2) ** 0.25)
    # one tick of pure bid-ask bounce reads the same spread at any price level (Up/Down symmetry)
    tick = lambda p: corwin_schultz(p + .005, p - .005, p + .005, p - .005)
    assert tick(0.05) == pytest.approx(tick(0.5), rel=0.02) == pytest.approx(tick(0.95), rel=0.02)
    assert corwin_schultz(0.5, 0.5, 0.5, 0.5) == 0.0
    assert corwin_schultz(0.52, 0.50, 0.60, 0.58) == 0.0          # trending range: no spread


@pytest.mark.property
def test_kyle_lambda_is_the_price_impact_slope():
    rng = np.random.default_rng(1)
    flow = rng.normal(0, 100, 5000)
    dp = 2e-4 * flow + rng.normal(0, 0.005, 5000)
    assert ols_slope(dp, flow) == pytest.approx(2e-4, rel=0.1)
    assert np.isnan(ols_slope([1, 2], [1, 2])) and np.isnan(ols_slope([1, 2, 3], [5, 5, 5]))


@pytest.mark.case
def test_amihud_and_vpin_hand_values():
    assert amihud([0.1, 0.3], [10, 20]) == pytest.approx((0.01 + 0.015) / 2)
    assert np.isnan(amihud([0.1], [0]))
    assert vpin([10, 0], [0, 10]) == 1.0 and vpin([5, 5], [5, 5]) == 0.0
    assert vpin([8, 2], [2, 8]) == pytest.approx(12 / 20)


def bar(i, close, buy, sell, hi=None, lo=None):
    return Bar("tick", S, i, S + i, S + i, close, hi or close + .01, lo or close - .01, close, close,
               10, buy + sell, (buy + sell) * .5, 0, buy, sell, 0, buy - sell, (buy - sell) * .5, 10)


@pytest.mark.case
def test_bar_micro_features_use_only_bars_already_known():
    bars = [(S + 10 * (i + 1), bar(i, 0.5 + 0.01 * i, 10 + i, 5)) for i in range(12)]
    f = bar_micro_features("tick", bars, S, n=4)
    assert np.isnan(f["tick_vpin"][9]) and not np.isnan(f["tick_vpin"][10])   # first bar known at t=10
    later = bars[:6] + [(s, bar(b.index, 0.9, 500, 0)) for s, b in bars[6:]]
    g = bar_micro_features("tick", later, S, n=4)
    known_by = bars[6][0] - S - 1
    for k in f:
        assert np.array_equal(f[k][: known_by + 1], g[k][: known_by + 1], equal_nan=True), k
    assert not np.array_equal(f["tick_vpin"], g["tick_vpin"], equal_nan=True)
    last4 = [b for _, b in bars[8:12]]
    assert f["tick_vpin"][T - 1] == pytest.approx(vpin([b.buy_shares for b in last4], [b.sell_shares for b in last4]))


@pytest.mark.case
def test_roll_feature_is_known_one_second_after_the_trade():
    ts = np.arange(S - 100, S + 50, 0.5)
    prices = 0.5 + np.where(np.arange(len(ts)) % 2, 0.01, -0.01)
    f = roll_feature(pd.DataFrame({"ts": ts, "p_up": prices}), S, n=50)
    assert f["roll"][10] == pytest.approx(0.04, rel=0.05)     # Δp alternates ±0.02: cov = -0.0004
    tp2 = pd.DataFrame({"ts": ts, "p_up": np.where(ts >= S + 20, 0.9, prices)})
    g = roll_feature(tp2, S, n=50)
    assert np.array_equal(f["roll"][:21], g["roll"][:21], equal_nan=True)
    assert not np.array_equal(f["roll"][22:40], g["roll"][22:40], equal_nan=True)


@pytest.mark.property
def test_offline_and_live_use_the_same_window_statistics():
    from pmlab.micro import window_bar_stats
    rng = np.random.default_rng(3)
    bars, p = [], 0.5
    for i in range(15):
        p = float(np.clip(p + rng.normal(0, .02), .05, .95))
        b_, s_ = float(rng.uniform(1, 50)), float(rng.uniform(1, 50))
        bars.append(bar(i, p, b_, s_, hi=p + .01, lo=p - .01))
    known = [(S + 10 * (i + 1), b) for i, b in enumerate(bars)]
    f = bar_micro_features("tick", known, S, n=10)
    last = window_bar_stats(bars[5:15], bars[4].close)             # what the engine computes at bar 15
    for m in ("vpin", "kyle", "amihud", "cs"):
        assert f[f"tick_{m}"][T - 1] == pytest.approx(last[m], nan_ok=True), m
    first = window_bar_stats(bars[:3])                                # window start: base is the first open
    assert first["kyle"] == pytest.approx(ols_slope(np.diff([bars[0].open] + [b.close for b in bars[:3]]),
                                                    [b.share_imbalance for b in bars[:3]]))
    assert f["tick_amihud"][known[2][0] - S] == pytest.approx(first["amihud"])



@pytest.mark.case
def test_kyle_lambda_book_form_has_no_intercept_and_reports_its_t_value():
    from pmlab.micro import origin_slope
    rng = np.random.default_rng(8)
    x = rng.normal(0, 100, 400)
    y = 0.002 * x + 0.5 + rng.normal(0, 0.1, 400)            # an intercept the book's model does not have
    lam, t = origin_slope(y, x)
    resid = y - lam * x
    assert lam == pytest.approx(float(x @ y) / float(x @ x))
    assert t == pytest.approx(lam / np.sqrt(resid @ resid / (len(x) - 1) / (x @ x)))
    assert all(np.isnan(v) for v in origin_slope(y[:2], x[:2]))
    free = np.polyfit(x, y, 1)[0]
    assert free == pytest.approx(0.002, rel=0.05) and lam != pytest.approx(free, rel=1e-3)
