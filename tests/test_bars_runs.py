import numpy as np
import pytest

from pmlab.bars import build, calibrate, make_builder
from pmlab.bars.imbalance import flow
from pmlab.bars.runs import RunBars, expected_abs_normal
from pmlab.events import make_trade

W = 1_000_000


def tr(t, sign=1, p=0.5, shares=1.0, window=W):
    return make_trade(window + t, window, p, sign, shares, p * shares)


def feed(builder, trades, window=W):
    builder.new_window(window)
    return [b for b in (builder.update(x) for x in trades) if b is not None]


@pytest.mark.case
def test_book_formula_is_expected_T_times_dominant_side_share():
    b = RunBars("tick", init_T=10, init_p_buy=0.6, init_v_buy=1, init_v_sell=1, jensen=False, anchor="bars")
    assert b.expected_threshold() == pytest.approx(6.0)
    b = RunBars("volume", init_T=10, init_p_buy=0.6, init_v_buy=2, init_v_sell=5, jensen=False, anchor="bars")
    assert b.expected_threshold() == pytest.approx(10 * max(0.6 * 2, 0.4 * 5))


@pytest.mark.case
def test_jensen_threshold_adds_the_expected_absolute_imbalance():
    # balanced tick flow: E[v]=1, mu_d=0, sd_d=1 -> T/2 + sqrt(T)*sqrt(2/pi)/2
    b = RunBars("tick", init_T=16, init_p_buy=0.5, init_v_buy=1, init_v_sell=1, init_v2=1, anchor="bars")
    assert b.expected_threshold() == pytest.approx(8 + 4 * (2 / np.pi) ** 0.5 / 2)
    assert expected_abs_normal(0, 1) == pytest.approx((2 / np.pi) ** 0.5)
    assert expected_abs_normal(50, 1) == pytest.approx(50)            # strong drift: |E D|
    assert expected_abs_normal(-3, 0) == 3
    one_sided = RunBars("tick", init_T=16, init_p_buy=1.0, init_v_buy=1, init_v_sell=1, init_v2=1, anchor="bars")
    assert one_sided.expected_threshold() == pytest.approx(16)        # Jensen gap vanishes


@pytest.mark.case
def test_opposite_trades_do_not_cancel_in_run_bars():
    # threshold 3: sells interleaved with buys still let buys reach 3
    b = RunBars("tick", init_T=6, init_p_buy=0.5, init_v_buy=1, init_v_sell=1, jensen=False, anchor="bars")
    bars = feed(b, [tr(1, 1), tr(2, -1), tr(3, 1), tr(4, -1), tr(5, 1), tr(6, -1)])
    assert [x.n_ticks for x in bars] == [5]
    assert (bars[0].buy_shares, bars[0].sell_shares) == (3, 2)


@pytest.mark.case
def test_volume_run_uses_shares_on_each_side():
    b = RunBars("volume", init_T=2, init_p_buy=0.5, init_v_buy=10, init_v_sell=10, jensen=False, anchor="bars")   # thr 10
    bars = feed(b, [tr(1, -1, shares=9), tr(2, 1, shares=9), tr(3, -1, shares=2)])
    assert [x.n_ticks for x in bars] == [3] and bars[0].sell_shares == 11


@pytest.mark.case
def test_expectations_update_after_trades_and_threshold_frozen_in_bar():
    b = RunBars("tick", init_T=10, init_p_buy=0.5, init_v_buy=1, init_v_sell=1, span_ticks=9, anchor="bars")
    b.new_window(W)
    b.update(tr(1, 1))
    frozen = b.threshold()
    for i in range(2, 30):
        bar = b.update(tr(i, 1))
        if bar:
            assert bar.threshold == frozen
            break
    else:
        pytest.fail("no bar closed: the frozen-threshold assertion never ran")
    assert b.P > 0.5


@pytest.mark.property
@pytest.mark.parametrize("measure", ["tick", "volume", "dollar"])
def test_every_run_bar_closes_at_first_crossing(measure):
    rng = np.random.default_rng(13)
    windows = {}
    for w in range(10):
        window = W + 300 * w
        pb = rng.uniform(0.2, 0.8)
        windows[window] = [tr(t, 1 if rng.random() < pb else -1, p=float(rng.uniform(.05, .95)),
                              shares=float(rng.lognormal(2, 1)), window=window)
                           for t in np.sort(rng.uniform(0, 300, 400))]
    builder = make_builder(f"{measure}_run", calibrate(windows), 30)
    for window in sorted(windows):
        trades = windows[window]
        pos = 0
        for bar in feed(builder, trades, window):
            chunk = trades[pos:pos + bar.n_ticks]
            buys = np.cumsum([flow(measure, x) if x.sign > 0 else 0 for x in chunk])
            sells = np.cumsum([flow(measure, x) if x.sign < 0 else 0 for x in chunk])
            dominant = np.maximum(buys, sells)
            assert dominant[-1] >= bar.threshold and (dominant[:-1] < bar.threshold).all()
            pos += bar.n_ticks
        assert builder.lo - 1e-9 <= builder.E_T <= builder.hi + 1e-9


@pytest.mark.property
def test_book_rules_ratchet_to_the_clip_and_the_defaults_hold_the_target():
    rng = np.random.default_rng(17)
    mk = lambda w: [tr(t, int(rng.choice([-1, 1])), shares=float(rng.lognormal(1, .5)), window=w)
                    for t in np.sort(rng.uniform(0, 300, 600))]
    prior = {W + 300 * i: mk(W + 300 * i) for i in range(10)}
    test = {W + 300 * i: mk(W + 300 * i) for i in range(10, 40)}
    cal = calibrate(prior)
    for kind in ("tick_run", "volume_run", "dollar_run"):
        book = make_builder(kind, cal, 30, jensen=False, anchor="bars")
        n_book = len(build(test, book)) / len(test)
        assert book.E_T < book.lo * 1.25 < book.T0 / 3     # ran away from T0 down to the clip
        assert n_book > 30 * 3, (kind, n_book)
        default = make_builder(kind, cal, 30)
        n = len(build(test, default)) / len(test)
        assert 30 * 0.85 <= n <= 30 * 1.15, (kind, n)
        assert default.E_T == pytest.approx(cal.ticks / 30, rel=0.1)


@pytest.mark.case
def test_activity_anchor_follows_trades_per_window():
    b = RunBars("tick", init_T=10, init_p_buy=0.5, init_v_buy=1, init_v_sell=1,
                anchor="activity", ticks_per_window=300, span_windows=1)   # 30 bars/window
    feed(b, [tr(i / 2, int((-1) ** i)) for i in range(600)])               # a 600-trade window
    b.new_window(W + 300)
    assert b.E_T == pytest.approx(20)                                       # 600 / 30
    with pytest.raises(ValueError):
        RunBars("tick", 10, 0.5, 1, 1, anchor="activity")


@pytest.mark.case
def test_expected_abs_normal_uses_the_erf_term_when_the_mean_is_not_zero():
    # E|N(mu, sd)| closed form vs numerical integration at a moderate drift
    mu, sd = 0.7, 1.3
    x = np.linspace(-12, 12, 200001)
    pdf = np.exp(-(x - mu) ** 2 / (2 * sd * sd)) / (sd * np.sqrt(2 * np.pi))
    assert expected_abs_normal(mu, sd) == pytest.approx(np.trapezoid(np.abs(x) * pdf, x), rel=1e-6)


@pytest.mark.case
def test_run_bars_default_second_moment_is_the_square_of_the_mean_flow():
    b = RunBars("volume", init_T=10, init_p_buy=0.6, init_v_buy=2, init_v_sell=5, anchor="bars")
    assert b.v2 == pytest.approx((0.6 * 2 + 0.4 * 5) ** 2)


@pytest.mark.case
def test_activity_anchor_averages_windows_with_its_span():
    b = RunBars("tick", init_T=10, init_p_buy=0.5, init_v_buy=1, init_v_sell=1,
                anchor="activity", ticks_per_window=300, span_windows=3)    # a = 0.5
    feed(b, [tr(i / 2, 1) for i in range(600)])
    b.new_window(W + 300)
    assert b.window_ticks == pytest.approx(300 + 0.5 * (600 - 300))
    assert b.E_T == pytest.approx(450 / 30)


@pytest.mark.property
def test_the_jensen_term_does_not_stop_the_bar_length_feedback():
    # with E0[T] learned from bar lengths, the E|D| threshold runs away upward (the book's runs downward,
    # test above): only the activity anchor holds the target, the E|D| term only shifts the level
    rng = np.random.default_rng(17)
    mk = lambda w: [tr(t, int(rng.choice([-1, 1])), shares=float(rng.lognormal(1, .5)), window=w)
                    for t in np.sort(rng.uniform(0, 300, 600))]
    prior = {W + 300 * i: mk(W + 300 * i) for i in range(10)}
    test = {W + 300 * i: mk(W + 300 * i) for i in range(10, 40)}
    cal = calibrate(prior)
    for kind in ("tick_run", "volume_run", "dollar_run"):
        jensen_bars = make_builder(kind, cal, 30, jensen=True, anchor="bars")
        n = len(build(test, jensen_bars)) / len(test)
        assert jensen_bars.E_T > 2.5 * jensen_bars.T0 and n < 30 * 0.6, (kind, jensen_bars.E_T, n)


@pytest.mark.case
def test_constructor_default_is_the_activity_anchor():
    b = RunBars("tick", init_T=10, init_p_buy=0.5, init_v_buy=1, init_v_sell=1, ticks_per_window=300)
    assert b.anchor == "activity" and b.bars_per_window == pytest.approx(30)
    with pytest.raises(ValueError):          # the default needs trades per window; the bar-length feedback must be asked for
        RunBars("tick", init_T=10, init_p_buy=0.5, init_v_buy=1, init_v_sell=1)
    assert RunBars("tick", 10, 0.5, 1, 1, anchor="bars").anchor == "bars"
