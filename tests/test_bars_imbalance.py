import math

import numpy as np
import pytest

from pmlab.bars import build, calibrate, make_builder
from pmlab.bars.imbalance import ImbalanceBars, flow
from pmlab.bars.tickrule import TickRule, accuracy
from pmlab.events import make_trade

W = 1_000_000


def tr(t, sign=1, p=0.5, shares=1.0, window=W):
    return make_trade(window + t, window, p, sign, shares, p * shares)


def feed(builder, trades, window=W):
    builder.new_window(window)
    return [b for b in (builder.update(x) for x in trades) if b is not None]


@pytest.mark.case
def test_tick_rule_carries_sign_through_unchanged_prices():
    rule = TickRule()
    assert [rule(p) for p in [.50, .51, .51, .50, .50, .52]] == [1, 1, 1, -1, -1, 1]
    assert accuracy([.50, .51, .50], [1, 1, 1]) == pytest.approx(2 / 3)


@pytest.mark.case
def test_tib_closes_when_imbalance_reaches_expected_T_times_expected_b():
    # E[T]=4, E[b]=0.5, no floor -> threshold 2. Signs +,-,+,+ give theta 1,0,1,2.
    b = ImbalanceBars("tick", init_T=4, init_bv=0.5, init_bv_sd=math.sqrt(0.75), floor_sd=0, anchor="bars")
    bars = feed(b, [tr(1, 1), tr(2, -1), tr(3, 1), tr(4, 1), tr(5, 1)])
    assert len(bars) == 1 and bars[0].n_ticks == 4 and bars[0].threshold == pytest.approx(2.0)
    assert bars[0].tick_imbalance == 2


@pytest.mark.case
def test_balanced_flow_uses_the_random_walk_floor_instead_of_collapsing():
    b = ImbalanceBars("tick", init_T=16, init_bv=0.0, init_bv_sd=1.0, floor_sd=1.0, anchor="bars")
    assert b.expected_threshold() == pytest.approx(4.0)       # 1 sd * sqrt(16), not 16 * 0
    no_floor = ImbalanceBars("tick", init_T=16, init_bv=0.0, init_bv_sd=1.0, floor_sd=0, anchor="bars")
    assert no_floor.expected_threshold() == 0.0


@pytest.mark.case
def test_volume_and_dollar_measures_weight_flow():
    x = tr(1, -1, p=0.9, shares=10)
    assert flow("tick", x) == 1 and flow("volume", x) == 10
    assert flow("dollar", x) == pytest.approx(10 * math.sqrt(0.09))
    vib = ImbalanceBars("volume", init_T=2, init_bv=5.0, init_bv_sd=1.0, floor_sd=0, anchor="bars")   # thr 10
    bars = feed(vib, [tr(1, 1, shares=4), tr(2, 1, shares=4), tr(3, 1, shares=4)])
    assert [b.n_ticks for b in bars] == [3]


@pytest.mark.case
def test_threshold_is_frozen_for_the_whole_bar():
    b = ImbalanceBars("tick", init_T=10, init_bv=0.2, init_bv_sd=1.0, floor_sd=0, anchor="bars")
    b.new_window(W)
    b.update(tr(1, 1))
    frozen = b.threshold()
    for i in range(2, 50):                    # all-buy flow moves E[b] toward 1 mid-bar
        bar = b.update(tr(i, 1))
        if bar:
            assert bar.threshold == frozen
            break
    else:
        pytest.fail("no bar closed: the frozen-threshold assertion never ran")
    assert b.E_bv > 0.2


@pytest.mark.case
def test_expected_T_is_clipped_both_ways_and_survives_new_window():
    # floor 100 * sd 1 * sqrt(4) = 200: the first bar takes 200 ticks, E_T would jump to 200
    long = ImbalanceBars("tick", init_T=4, init_bv=0.0, init_bv_sd=1.0, floor_sd=100, clip=2.0, span_bars=1, anchor="bars")
    bars = feed(long, [tr(i / 10, 1) for i in range(1, 201)])
    assert [b.n_ticks for b in bars] == [200] and long.E_T == 8
    long.new_window(W + 300)
    assert long.E_T == 8                                 # learned expectation is kept
    # threshold 4 * 0.1 = 0.4: every trade is a bar, E_T would fall to 1
    short = ImbalanceBars("tick", init_T=4, init_bv=0.1, init_bv_sd=0.1, floor_sd=0, clip=2.0, span_bars=1, anchor="bars")
    bars = feed(short, [tr(1, 1)])
    assert [b.n_ticks for b in bars] == [1] and short.E_T == 2


@pytest.mark.property
@pytest.mark.parametrize("measure", ["tick", "volume", "dollar"])
def test_every_imbalance_bar_closes_at_first_crossing(measure):
    rng = np.random.default_rng(5)
    windows = {}
    for w in range(10):
        window = W + 300 * w
        drift = rng.uniform(-0.6, 0.6)
        ts = np.sort(rng.uniform(0, 300, 400))
        windows[window] = [tr(t, 1 if rng.random() < (1 + drift) / 2 else -1, p=float(rng.uniform(.05, .95)),
                              shares=float(rng.lognormal(2, 1)), window=window) for t in ts]
    cal = calibrate(windows)
    builder = make_builder(f"{measure}_imbalance", cal, 30)
    for window in sorted(windows):
        trades = windows[window]
        bars = feed(builder, trades, window)
        pos = 0
        for bar in bars:
            flows = np.cumsum([x.sign * flow(measure, x) for x in trades[pos:pos + bar.n_ticks]])
            assert abs(flows[-1]) >= bar.threshold
            assert (np.abs(flows[:-1]) < bar.threshold).all()
            pos += bar.n_ticks
        assert lo_ok(builder)


def lo_ok(b):
    return b.lo - 1e-9 <= b.E_T <= b.hi + 1e-9


@pytest.mark.case
def test_tick_rule_sign_source_is_used_and_reset_per_window():
    b = ImbalanceBars("tick", init_T=2, init_bv=1.0, init_bv_sd=0.1, floor_sd=0, sign="tick_rule", anchor="bars")  # thr 2
    # true aggressor says sell, prices rise -> tick rule says buy
    bars = feed(b, [tr(1, -1, p=.50), tr(2, -1, p=.51), tr(3, -1, p=.52)])
    assert bars and bars[0].tick_imbalance == -2        # bar stats keep the true sign...
    assert b.tick_rule.last_b == 1                      # ...while the trigger used the tick rule
    b.new_window(W + 300)
    assert b.tick_rule.last_p is None


@pytest.mark.property
def test_imbalance_bars_on_calibrated_random_flow_land_near_target():
    rng = np.random.default_rng(9)
    mk = lambda w: [tr(t, int(rng.choice([-1, 1])), p=.5, shares=float(rng.lognormal(1, .5)), window=w)
                    for t in np.sort(rng.uniform(0, 300, 600))]
    prior = {W + 300 * i: mk(W + 300 * i) for i in range(10)}
    test = {W + 300 * i: mk(W + 300 * i) for i in range(10, 30)}
    cal = calibrate(prior)
    for kind in ("tick_imbalance", "volume_imbalance", "dollar_imbalance"):
        per_window = len(build(test, make_builder(kind, cal, 30))) / len(test)
        # balanced flow: the floor is the exit level of a random walk after E[T] steps,
        # whose expected exit time is exactly E[T]
        assert 30 * 0.75 <= per_window <= 30 * 1.25, (kind, per_window)


@pytest.mark.case
def test_imbalance_constructor_default_is_the_activity_anchor():
    b = ImbalanceBars("tick", init_T=10, init_bv=0.0, init_bv_sd=1.0, ticks_per_window=300)
    assert b.anchor == "activity" and b.bars_per_window == pytest.approx(30)
    with pytest.raises(ValueError):
        ImbalanceBars("tick", init_T=10, init_bv=0.0, init_bv_sd=1.0)
    assert ImbalanceBars("tick", 10, 0.0, 1.0, anchor="bars").anchor == "bars"


@pytest.mark.eval
def test_the_tick_rule_classifies_well_when_the_aggressor_moves_the_price_and_no_better_than_a_coin_when_it_does_not():
    """19.3.1: the rule is simple and yet accurate. It is accurate exactly to the extent that the aggressor's
    own trade moves the price: on a simulated tape where the taker's side moves the print with probability q,
    accuracy tracks q. Thresholds (20,000 prints): above 0.9 at q = 0.95, above 0.7 at q = 0.75, and within
    0.05 of a coin toss at q = 0.5, where the price carries no information about the side."""
    rng = np.random.default_rng(0)
    n = 20_000
    sides = rng.choice([-1, 1], n)
    for q, lo, hi in ((0.95, 0.90, 1.0), (0.75, 0.70, 0.95), (0.5, 0.45, 0.55)):
        follows = rng.random(n) < q
        step = np.where(follows, sides, -sides) * 0.01
        prices = 0.5 + np.cumsum(step)
        got = accuracy(prices.tolist(), sides.tolist())
        assert lo <= got <= hi, (q, got)
    # and it is not the trivial classifier: with every price unchanged it answers its initial sign to everything
    assert accuracy([0.5] * 100, [1] * 100) == 1.0 and accuracy([0.5] * 100, [-1] * 100) == 0.0
