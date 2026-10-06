import numpy as np
import pytest

from pmlab.bars import BAR_COLUMNS, build, calibrate, make_builder, logit_returns
from pmlab.bars.standard import DollarBars, TickBars, TimeBars, VolumeBars
from pmlab.events import make_trade

W = 1_000_000


def tr(t, p=0.5, sign=1, shares=1.0, window=W):
    return make_trade(window + t, window, p, sign, shares, p * shares)


def run(builder, trades, window=W):
    builder.new_window(window)
    out = [b for b in (builder.update(x) for x in trades) if b is not None]
    last = builder.finish()
    return out + ([last] if last else [])


def random_window(rng, window, n=300):
    ts = np.sort(rng.uniform(0, 300, n))
    return [tr(t, p=float(rng.uniform(0.02, 0.98)), sign=int(rng.choice([-1, 1])),
               shares=float(rng.lognormal(2, 1)), window=window) for t in ts]


@pytest.mark.case
def test_tick_bars_ohlc_vwap_flow_and_partial_dropped():
    trades = [tr(1, .50, 1, 2), tr(2, .55, -1, 1), tr(3, .45, 1, 1),
              tr(4, .60, 1, 5), tr(5, .40, -1, 5), tr(6, .52, 1, 1), tr(7, .53, 1, 1)]
    bars = run(TickBars(3), trades)
    assert len(bars) == 2                                   # trade 7 is a dropped partial
    b = bars[0]
    assert (b.open, b.high, b.low, b.close) == (.50, .55, .45, .45)
    assert b.vwap == pytest.approx((.5 * 2 + .55 + .45) / 4)
    assert (b.n_ticks, b.shares, b.buy_shares, b.sell_shares) == (3, 4, 3, 1)
    assert (b.tick_imbalance, b.share_imbalance) == (1, 2)
    assert (b.ts_open, b.ts_close, b.index) == (W + 1, W + 3, 0)
    assert bars[1].index == 1 and bars[1].n_ticks == 3


@pytest.mark.case
def test_volume_bar_closes_on_the_first_crossing_trade():
    bars = run(VolumeBars(10), [tr(i + 1, shares=q) for i, q in enumerate([4, 4, 4, 1, 9, 2])])
    assert [(b.n_ticks, b.shares) for b in bars] == [(3, 12), (2, 10)]


@pytest.mark.case
def test_dollar_bars_count_risk_not_cash():
    # 10 shares at .5 carry 5.0 risk dollars; 10 shares at .98 only ~1.4 despite $9.80 cash.
    assert tr(1, .5, shares=10).risk_usd == pytest.approx(5.0)
    assert tr(1, .98, shares=10).risk_usd == pytest.approx(10 * np.sqrt(.98 * .02))
    bars = run(DollarBars(5), [tr(1, .98, shares=10), tr(2, .98, shares=10), tr(3, .5, shares=10)])
    assert len(bars) == 1 and bars[0].n_ticks == 3


@pytest.mark.case
def test_time_bars_close_on_later_trade_skip_empty_buckets_and_finish_at_window_end():
    bars = run(TimeBars(60), [tr(5), tr(30), tr(61), tr(200), tr(290)])
    assert [(b.ts_close - W, b.n_ticks) for b in bars] == [(60, 2), (120, 1), (240, 1), (300, 1)]


@pytest.mark.case
def test_new_window_discards_partial_and_wrong_window_raises():
    b = TickBars(2)
    b.new_window(W)
    b.update(tr(1))
    b.new_window(W + 300)
    assert b.n == 0 and b.index == 0
    with pytest.raises(ValueError):
        b.update(tr(1, window=W))


@pytest.mark.property
@pytest.mark.parametrize("cls,size,total", [(TickBars, 17, lambda x: 1.0),
                                            (VolumeBars, 400.0, lambda x: x.shares),
                                            (DollarBars, 150.0, lambda x: x.risk_usd)])
def test_size_bars_close_exactly_at_first_crossing_and_conserve_flow(cls, size, total):
    rng = np.random.default_rng(7)
    for w in range(5):
        window = W + 300 * w
        trades = random_window(rng, window)
        builder = cls(size)
        bars = run(builder, trades, window)
        pos = 0
        for bar in bars:
            chunk = trades[pos:pos + bar.n_ticks]
            flows = [total(x) for x in chunk]
            assert sum(flows) >= size                     # bar is full
            assert sum(flows[:-1]) < size                 # ...and not one trade later than needed
            assert bar.shares == pytest.approx(sum(x.shares for x in chunk))
            assert bar.ts_open == chunk[0].ts and bar.ts_close == chunk[-1].ts
            pos += bar.n_ticks
        assert sum(total(x) for x in trades[pos:]) < size  # only an unfinished remainder is left


@pytest.mark.property
def test_calibrated_builders_hit_the_target_bar_count():
    rng = np.random.default_rng(11)
    prior = {W + 300 * i: random_window(rng, W + 300 * i) for i in range(20)}
    test = {W + 300 * i: random_window(rng, W + 300 * i) for i in range(20, 40)}
    cal = calibrate(prior)
    for kind in ("time", "tick", "volume", "dollar"):
        bars = build(test, make_builder(kind, cal, 30))
        per_window = len(bars) / len(test)
        assert 30 * 0.8 <= per_window <= 30 * 1.05, (kind, per_window)
        assert list(bars.columns) == BAR_COLUMNS
        assert bars.groupby("window")["index"].apply(lambda s: list(s) == list(range(len(s)))).all()


@pytest.mark.case
def test_logit_returns_never_cross_a_window():
    rng = np.random.default_rng(3)
    windows = {W: random_window(rng, W, 60), W + 300: random_window(rng, W + 300, 60)}
    bars = build(windows, TickBars(10))
    r = logit_returns(bars)
    assert len(r) == len(bars) - bars["window"].nunique()


@pytest.mark.case
def test_calibrate_outputs_match_hand_computation():
    from pmlab.bars import calibrate
    trades = {W: [tr(1, .5, 1, 4), tr(2, .5, -1, 2), tr(3, .8, 1, 10)]}
    cal = calibrate(trades)
    risk = [4 * .5, 2 * .5, 10 * .4]
    assert cal.ticks == 3 and cal.p_buy == pytest.approx(2 / 3)
    assert cal.per_window["volume"] == 16 and cal.per_window["dollar"] == pytest.approx(sum(risk))
    bv = np.array([4, -2, 10])
    assert cal.bv_mean["volume"] == pytest.approx(bv.mean()) and cal.bv_sd["volume"] == pytest.approx(bv.std())
    assert cal.v_buy["volume"] == pytest.approx(7) and cal.v_sell["volume"] == pytest.approx(2)
    assert cal.v2["dollar"] == pytest.approx(np.mean(np.square(risk)))


@pytest.mark.property
def test_fit_scale_hits_the_target_count_and_adaptive_standard_bars_follow_activity():
    from pmlab.bars import calibrate, fit_scale, make_builder, build
    rng = np.random.default_rng(21)
    busy = lambda w, n: [tr(t, float(rng.uniform(.2, .8)), int(rng.choice([-1, 1])), float(rng.lognormal(1, 1)), w)
                         for t in np.sort(rng.uniform(0, 300, n))]
    cal_windows = {W + 300 * i: busy(W + 300 * i, 400) for i in range(12)}
    cal = calibrate(cal_windows)
    for kind in ("tick_imbalance", "volume_run"):
        scale, count = fit_scale(kind, cal, 20, cal_windows)
        assert 20 * 0.9 <= count <= 20 * 1.1, (kind, count)
    # activity doubles after calibration: adaptive tick bars keep ~20 per window, fixed ones double
    later = {W + 300 * i: busy(W + 300 * i, 800) for i in range(12, 36)}
    fixed = len(build(later, make_builder("tick", cal, 20))) / len(later)
    tail = build(later, make_builder("tick", cal, 20, adaptive_size=True))
    late_windows = sorted(later)[12:]
    per_late = tail[tail["window"].isin(late_windows)].groupby("window").size().mean()
    assert fixed > 35 and 17 <= per_late <= 23, (fixed, per_late)
