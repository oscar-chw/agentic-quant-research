import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab.backtest import Config, Position, View, run_window
from pmlab.features import WindowFeatures
from pmlab.fees import fee_per_share
from pmlab.strategies import FairMaker, FlowTaker, ProbTaker

S = 1_800_000_000


def wf(**cols):
    base = {"t": np.arange(T, dtype=float), "fair": np.full(T, 0.5), "ask": np.full(T, 0.5),
            "ask_age": np.zeros(T), "bid": np.full(T, 0.48), "bid_age": np.zeros(T),
            "tick_imbalance_imb": np.full(T, np.nan), "tick_imbalance_age": np.full(T, np.nan)}
    for k, v in cols.items():
        base[k] = np.broadcast_to(np.asarray(v, dtype=float), (T,)).copy()
    return WindowFeatures(S, base, pd.DataFrame(columns=["ts", "sign", "p_up", "shares"]), 1.0)


@pytest.mark.case
def test_prob_taker_buys_up_only_when_edge_beats_fees_and_slip():
    s = ProbTaker(edge=0.03, slip=0.01, kelly=1.0, max_stake=1e9, max_window_cost=1e9)
    price = 0.50 + 0.01
    need = price + fee_per_share(price) + 0.03
    [o] = s.decide(View(wf(fair=need + 0.005, ask=0.50), 10), Position(), 1000)
    assert (o.side, o.limit) == ("up", pytest.approx(price))
    all_in = price + fee_per_share(price)
    assert o.shares * all_in == pytest.approx(((need + 0.005) - all_in) / (1 - all_in) * 1000)
    assert s.decide(View(wf(fair=need - 0.005, ask=0.50), 10), Position(), 1000) == []


@pytest.mark.case
def test_prob_taker_buys_down_when_fair_is_below_bid_and_ignores_stale_quotes():
    s = ProbTaker(edge=0.02, kelly=0.5)
    [o] = s.decide(View(wf(fair=0.30, bid=0.40), 10), Position(), 1000)
    assert o.side == "down" and o.limit == pytest.approx(0.61)
    assert s.decide(View(wf(fair=0.30, bid=0.40, bid_age=30), 10), Position(), 1000) == []


@pytest.mark.case
def test_prob_taker_candidates_are_a_superset_of_acting_seconds():
    rng = np.random.default_rng(0)
    f = wf(fair=rng.uniform(.1, .9, T), ask=rng.uniform(.1, .9, T), bid=rng.uniform(.1, .9, T),
           ask_age=rng.uniform(0, 10, T))
    s = ProbTaker(edge=0.01, kelly=1.0, max_window_cost=1e9)
    acting = {t for t in range(T) if s.decide(View(f, t), Position(), 1000)}
    assert acting and acting <= set(s.candidates(f).tolist())
    m = FairMaker(requote=5, t_min=5, t_max=240)
    acting = {t for t in range(T) if m.decide(View(f, t), Position(), 1000)}
    assert acting and acting <= set(m.candidates(f).tolist())
    fl = FlowTaker(imb=0.2)
    g = wf(tick_imbalance_imb=rng.uniform(-1, 1, T), tick_imbalance_age=rng.uniform(0, 6, T), fair=.5, ask=.5, bid=.5)
    acting = {t for t in range(T) if fl.decide(View(g, t), Position(), 1000)}
    assert acting and acting <= set(fl.candidates(g).tolist())


@pytest.mark.case
def test_flow_taker_follows_a_fresh_one_sided_bar_within_slack():
    s = FlowTaker(kind="tick_imbalance", imb=0.5, max_age=3, slack=0.02)
    fresh_buy = wf(tick_imbalance_imb=0.8, tick_imbalance_age=1, fair=0.5, ask=0.50)
    [o] = s.decide(View(fresh_buy, 10), Position(), 1000)
    assert o.side == "up"
    assert s.decide(View(wf(tick_imbalance_imb=0.8, tick_imbalance_age=5, ask=.5), 10), Position(), 1000) == []
    assert s.decide(View(wf(tick_imbalance_imb=0.3, tick_imbalance_age=1, ask=.5), 10), Position(), 1000) == []
    assert s.decide(View(wf(tick_imbalance_imb=0.8, tick_imbalance_age=1, fair=np.nan, ask=.5), 10), Position(), 1000) == []
    too_dear = wf(tick_imbalance_imb=0.8, tick_imbalance_age=1, fair=0.40, ask=0.50)   # .51 > .40 + .02
    assert s.decide(View(too_dear, 10), Position(), 1000) == []
    [d] = s.decide(View(wf(tick_imbalance_imb=-0.9, tick_imbalance_age=0, fair=.5, bid=.49), 10), Position(), 1000)
    assert d.side == "down"


@pytest.mark.case
def test_fair_maker_never_posts_a_marketable_bid():
    s = FairMaker(half_spread=0.02)
    orders = s.decide(View(wf(fair=0.60, ask=0.55, bid=0.50), 10), Position(), 1000)
    # Up bid would be .58 >= ask .55 -> skipped; Down bid .38 < Down ask (1 - .50 = .50) -> posted
    assert [(o.side, o.limit, o.maker) for o in orders] == [("down", 0.38, True)]


@pytest.mark.case
def test_window_cost_cap_limits_repeated_entries():
    tp = pd.DataFrame({"ts": np.arange(S, S + T, dtype=float), "sign": 1, "p_up": 0.5, "shares": 1e6})
    f = WindowFeatures(S, wf(fair=0.9, ask=0.5).cols, tp, 1.0)
    s = ProbTaker(edge=0.0, kelly=1.0, max_stake=40, max_window_cost=100)
    pos = run_window(s, f, 1000.0, Config(participation=1.0, print_delay=0))
    cost = sum(x.shares * x.price + x.fee for x in pos.fills)
    assert 80 <= cost <= 100 + 1e-6


@pytest.mark.case
def test_open_fair_taker_buys_the_leaning_side_once_inside_its_window():
    from pmlab.strategies import OpenFairTaker
    s = OpenFairTaker(lean=0.15, gap=0.10, t_min=2, t_max=30)
    f = wf(fair=0.70, ask=0.52, bid=0.50)                   # lean .20, gap .70 − .51 = .19
    [o] = s.decide(View(f, 5), Position(), 1000)
    assert (o.side, o.limit, o.maker) == ("up", 0.53, False)
    assert s.decide(View(f, 1), Position(), 1000) == [] and s.decide(View(f, 31), Position(), 1000) == []
    held = Position(pending=[(9, o)])
    assert s.decide(View(f, 6), held, 1000) == []           # one entry per window
    assert s.decide(View(wf(fair=0.60, ask=0.52, bid=0.50), 5), Position(), 1000) == []   # lean .10 < .15
    assert s.decide(View(wf(fair=0.70, ask=0.66, bid=0.64), 5), Position(), 1000) == []   # gap .05 < .10
    [d] = s.decide(View(wf(fair=0.25, ask=0.52, bid=0.50), 5), Position(), 1000)
    assert d.side == "down" and d.limit == pytest.approx(0.51)
    g = wf(fair=np.linspace(.2, .8, T), ask=.5, bid=.49)
    acting = {t for t in range(T) if s.decide(View(g, t), Position(), 1000)}
    assert acting <= set(s.candidates(g).tolist())


@pytest.mark.case
def test_endgame_maker_bids_on_the_decided_favourite_only_late():
    from pmlab.strategies import EndgameMaker
    s = EndgameMaker(conf=0.99, bid=0.99, t_min=270, requote=5)
    [o] = s.decide(View(wf(fair=0.995), 275), Position(), 1000)
    assert (o.side, o.limit, o.maker, o.ttl) == ("up", 0.99, True, 5)
    [d] = s.decide(View(wf(fair=0.004), 280), Position(), 1000)
    assert d.side == "down"
    assert s.decide(View(wf(fair=0.98), 275), Position(), 1000) == []      # not decided enough
    assert s.decide(View(wf(fair=0.995), 200), Position(), 1000) == []     # too early
    assert s.decide(View(wf(fair=0.995), 276), Position(), 1000) == []     # between requotes


@pytest.mark.case
def test_window_cost_cap_holds_with_the_default_print_delay():
    tp = pd.DataFrame({"ts": np.arange(S, S + T + 20, dtype=float), "sign": 1, "p_up": 0.5, "shares": 1e6})
    f = WindowFeatures(S, wf(fair=0.9, ask=0.5).cols, tp, 1.0)
    s = ProbTaker(edge=0.0, kelly=1.0, max_stake=40, max_window_cost=100, max_window_frac=1.0)
    pos = run_window(s, f, 1000.0, Config(participation=1.0))                  # print_delay = 5
    cost = sum(x.shares * x.price + x.fee for x in pos.fills)
    assert 80 <= cost <= 100 + 1e-6


@pytest.mark.case
def test_kelly_sizes_the_window_position_and_a_small_bankroll_caps_it():
    s = ProbTaker(edge=0.0, kelly=1.0, max_stake=1e9, max_window_cost=1e9, max_window_frac=0.15)
    price = 0.51
    all_in = price + fee_per_share(price)
    full = (0.7 - all_in) / (1 - all_in) * 100
    [o] = s.decide(View(wf(fair=0.7, ask=0.5), 10), Position(), 100.0)
    assert o.shares * all_in == pytest.approx(min(full, 15.0))                  # 15% of a $100 bankroll
    from pmlab.backtest import Fill
    held = Position(fills=[Fill(S, 1, S + 2, "up", 20, 0.5, 0.1, False, "")])
    [more] = s.decide(View(wf(fair=0.7, ask=0.5), 10), held, 100.0)            # $10.1 committed: $4.9 of the cap left
    assert more.shares * all_in == pytest.approx(15.0 - 10.1)
    full_held = Position(fills=[Fill(S, 1, S + 2, "up", 29, 0.5, 0.1, False, "")])
    assert s.decide(View(wf(fair=0.7, ask=0.5), 10), full_held, 100.0) == []    # $14.6 committed: < $1 left


@pytest.mark.case
def test_prob_taker_never_orders_both_sides_in_one_call():
    s = ProbTaker(edge=-1.0, kelly=1.0, slip=0.0, max_window_cost=1e9, max_window_frac=1.0)
    orders = s.decide(View(wf(fair=0.5, ask=0.30, bid=0.70), 10), Position(), 1000)   # crossed proxies
    assert len({o.side for o in orders}) == 1
