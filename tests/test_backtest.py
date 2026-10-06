import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab.backtest import Config, Fill, Order, Position, View, report, run, run_window, settle, simulate_fill
from pmlab.bars import calibrate, make_builder
from pmlab.events import trades_from_tape
from pmlab.fair import FairParams
from pmlab.features import WindowFeatures, bar_features, base_features, run_bars
from pmlab.fees import taker_fee
from pmlab.strategies import ProbTaker, kelly_fraction

S = 1_800_000_000


def tape(rows):
    return pd.DataFrame(rows, columns=["ts", "sign", "p_up", "shares"])


def features(start=S, tape_df=None, y=1.0, **cols):
    base = {"t": np.arange(T, dtype=float), "fair": np.full(T, 0.5), "ask": np.full(T, 0.5),
            "ask_age": np.zeros(T), "bid": np.full(T, 0.48), "bid_age": np.zeros(T)}
    base.update({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    return WindowFeatures(start, base, tape_df if tape_df is not None else tape([]), y)


@pytest.mark.case
def test_taker_buy_up_fills_on_later_buy_prints_within_limit_and_participation():
    tp = tape([(S + 4, -1, .50, 10), (S + 5, 1, .52, 10), (S + 6, 1, .53, 100), (S + 6, 1, .51, 4)])
    fills = simulate_fill(tp, S, 3, Order("up", 8, .52), Config(latency=1, participation=0.5, print_delay=0))
    assert [(f.ts, f.price, f.shares) for f in fills] == [(S + 5, .52, 5), (S + 6, .51, 2)]
    assert fills[0].fee == taker_fee(5, .52) and not fills[0].maker


@pytest.mark.case
def test_taker_buy_down_uses_sell_prints_at_one_minus_p():
    tp = tape([(S + 4, 1, .40, 50), (S + 4, -1, .45, 50)])
    fills = simulate_fill(tp, S, 3, Order("down", 10, .60), Config(print_delay=0))
    assert [(f.side, f.price, f.shares) for f in fills] == [("down", pytest.approx(.55), 10)]


@pytest.mark.case
def test_latency_excludes_prints_before_the_order_could_arrive_and_ttl_ends_it():
    tp = tape([(S + 3, 1, .50, 50), (S + 9, 1, .50, 50)])
    assert simulate_fill(tp, S, 3, Order("up", 1, .6, ttl=3), Config(latency=1, print_delay=0)) == []
    assert len(simulate_fill(tp, S, 3, Order("up", 1, .6, ttl=3), Config(latency=0, print_delay=0))) == 1


@pytest.mark.case
def test_print_delay_skips_prints_whose_real_match_may_precede_the_decision():
    # decision at t=3; with a 5 s stamp delay a print stamped S+8 may have matched at S+3.x
    tp = tape([(S + 5, 1, .50, 50), (S + 8, 1, .51, 50), (S + 9, 1, .52, 50)])
    [f] = simulate_fill(tp, S, 3, Order("up", 1, .6, ttl=3), Config(latency=1, print_delay=5))
    assert f.ts == S + 9


@pytest.mark.case
def test_maker_bid_fills_only_when_traded_through_and_pays_no_fee():
    cfg = Config(print_delay=0)
    at_bid = tape([(S + 5, -1, .50, 50), (S + 5, 1, .40, 50)])
    assert simulate_fill(at_bid, S, 3, Order("up", 10, .50, maker=True), cfg) == []
    through = tape([(S + 5, -1, .49, 50)])
    [f] = simulate_fill(through, S, 3, Order("up", 10, .50, maker=True), cfg)
    assert (f.price, f.shares, f.fee, f.maker) == (.50, 10, 0.0, True)
    down_through = tape([(S + 5, 1, .52, 50)])                       # Down trades at .48 < .49
    [g] = simulate_fill(down_through, S, 3, Order("down", 4, .49, maker=True), cfg)
    assert (g.side, g.price) == ("down", .49)


@pytest.mark.case
def test_settlement_pays_up_on_y_and_down_on_one_minus_y():
    pos = Position(fills=[Fill(S, 1, S + 2, "up", 10, .4, .1, False, ""),
                          Fill(S, 1, S + 2, "down", 5, .3, 0.0, True, "")])
    up = settle(pos, 1.0)
    assert (up["payoff"], up["cost"], up["fees"]) == (10, pytest.approx(5.5), .1)
    assert up["pnl"] == pytest.approx(10 - 5.5 - .1)
    assert settle(pos, 0.0)["pnl"] == pytest.approx(5 - 5.5 - .1)


@pytest.mark.case
def test_view_hides_the_future_and_the_outcome():
    f = features(fair=np.linspace(0, 1, T))
    v = View(f, 10)
    assert len(v["fair"]) == 11 and v.now("fair") == f["fair"][10]
    assert not hasattr(v, "y")


@pytest.mark.case
def test_exposure_counts_known_fills_and_live_orders_only():
    pos = Position(fills=[Fill(S, 1, S + 5, "up", 10, .5, 0, False, "")],
                   pending=[(8, Order("down", 3, .4, maker=True))])
    assert pos.exposure(5)["up"] == 0 and pos.exposure(6)["up"] == 10       # fill known after its print
    assert pos.exposure(7)["down"] == 3 and pos.exposure(8)["down"] == 0    # pending until expiry


@pytest.mark.case
def test_kelly_fraction():
    assert kelly_fraction(.6, .5) == pytest.approx(.2)
    assert kelly_fraction(.5, .6) == 0 and kelly_fraction(.9, 1.0) == 0


@pytest.mark.case
def test_run_compounds_bankroll_in_time_order():
    class BuyOnce:
        def candidates(self, f):
            return [10]

        def decide(self, v, pos, bankroll):
            return [Order("up", 100, .5)]

    w1 = features(S, tape([(S + 11, 1, .5, 1000)]), y=1.0)
    w2 = features(S + 300, tape([(S + 311, 1, .5, 1000)]), y=0.0)
    res, fills = run(BuyOnce(), [w2, w1], Config(participation=1.0, print_delay=0))
    assert list(res["start"]) == [S, S + 300]
    first = 100 - 50 - taker_fee(100, .5)
    assert res["pnl"].iloc[0] == pytest.approx(first)
    assert res["bankroll"].iloc[1] == pytest.approx(1000 + first)


def synthetic_market(rng, start=S):
    instants = np.arange(start - 200, start + T + 60)
    close = pd.Series(75_000 * np.exp(np.cumsum(rng.normal(0, 5e-5, len(instants)))), index=instants - 1)
    ts = np.sort(rng.integers(start - 30, start + T + 5, 1500)).astype(float)
    tp = pd.DataFrame({"ts": ts, "t": ts - start, "p_up": rng.uniform(.05, .95, len(ts)).round(2),
                       "sign": rng.choice([-1, 1], len(ts)), "shares": rng.lognormal(2, 1, len(ts))})
    tp["usdc"] = tp["p_up"] * tp["shares"]
    return close, tp


@pytest.mark.property
@pytest.mark.parametrize("cut", [0, 57, 150, 298])
def test_features_at_t_are_unchanged_by_anything_after_t(cut):
    rng = np.random.default_rng(cut)
    close, tp = synthetic_market(rng)
    cal = calibrate({S: trades_from_tape(tp[(tp.ts >= S) & (tp.ts < S + T)], S)})
    a = base_features(S, tp, close, FairParams(halflife=60, basis=1e-5, lag=1))
    a.update(bar_features("tick_imbalance", run_bars(make_builder("tick_imbalance", cal, 30), S, tp), S))

    close2, tp2 = close.copy(), tp.copy()
    close2.loc[S + cut:] *= np.exp(rng.normal(0, 1e-3, (close2.index >= S + cut).sum()))  # prices at instants > cut
    later = tp2.ts >= S + cut                                                             # prints known after cut
    tp2.loc[later, "p_up"] = rng.uniform(.05, .95, later.sum()).round(2)
    tp2.loc[later, "sign"] = -tp2.loc[later, "sign"]
    b = base_features(S, tp2, close2, FairParams(halflife=60, basis=1e-5, lag=1))
    b.update(bar_features("tick_imbalance", run_bars(make_builder("tick_imbalance", cal, 30), S, tp2), S))
    changed_after = False
    for name in a:
        assert np.array_equal(a[name][: cut + 1], b[name][: cut + 1], equal_nan=True), name
        changed_after |= not np.array_equal(a[name][cut + 1:], b[name][cut + 1:], equal_nan=True)
    assert changed_after or cut == T - 2


@pytest.mark.property
def test_strategy_decisions_at_t_are_unchanged_by_anything_after_t():
    rng = np.random.default_rng(1)
    close, tp = synthetic_market(rng)
    cols = base_features(S, tp, close, FairParams(halflife=60))
    cols["fair"] = rng.uniform(.05, .95, T)                                 # lots of edges to act on
    cut = 150

    class Recorder(ProbTaker):
        def decide(self, v, pos, bankroll):
            out = super().decide(v, pos, bankroll)
            if v.t <= cut:
                self.log.append((v.t, [(o.side, round(o.shares, 9), o.limit) for o in out]))
            return out

    logs = []
    for perturb in (False, True):
        c = {k: v.copy() for k, v in cols.items()}
        tp2 = tp.copy()
        if perturb:
            for k in c:
                c[k][cut + 1:] = rng.permutation(c[k][cut + 1:])
            later = tp2.ts >= S + cut
            tp2.loc[later, "shares"] = rng.lognormal(3, 1, later.sum())
        strat = Recorder(edge=0.0, kelly=1.0, max_window_cost=1e9)
        strat.log = []
        run_window(strat, WindowFeatures(S, c, tp2, y=float(perturb)), 1000.0, Config())
        logs.append(strat.log)
    assert logs[0] and logs[0] == logs[1]


@pytest.mark.case
def test_trading_stops_after_ruin():
    class AllIn:
        def candidates(self, f):
            return [10]

        def decide(self, v, pos, bankroll):
            return [Order("up", 1900, .5)]                                  # ~$950 + fee on a loser

    losers = [features(S + 300 * i, tape([(S + 300 * i + 11, 1, .5, 1e6)]), y=0.0) for i in range(3)]
    res, _ = run(AllIn(), losers, Config(participation=1.0, print_delay=0))
    assert len(res) == 1 and report(res, 3)["ruined"]


@pytest.mark.case
def test_features_survive_a_window_with_no_prints_on_one_side_or_at_all():
    rng = np.random.default_rng(2)
    close, tp = synthetic_market(rng)
    only_buys = tp[tp.sign > 0]
    cols = base_features(S, only_buys, close, FairParams())
    assert np.isnan(cols["bid"]).all() and np.isfinite(cols["ask"][-1])
    empty = base_features(S, tp.iloc[:0], close, FairParams())
    assert np.isnan(empty["last"]).all() and (empty["n_trades"] == 0).all()


@pytest.mark.case
def test_fill_costs_split_spread_slippage_and_fee_against_the_decision_mid():
    taker = Fill(S, 3, S + 9, "up", 10, 0.53, taker_fee(10, 0.53), False, "", quote=0.52, mid=0.51)
    c = taker.costs()
    assert c["half_spread"] == pytest.approx(0.1) and c["slippage"] == pytest.approx(0.1)
    assert c["total_vs_mid"] == pytest.approx(0.2 + taker_fee(10, 0.53))
    maker = Fill(S, 3, S + 9, "down", 10, 0.47, 0.0, True, "", quote=0.47, mid=0.49)
    assert maker.costs()["half_spread"] == pytest.approx(-0.2) and maker.costs()["total_vs_mid"] == pytest.approx(-0.2)


@pytest.mark.case
def test_backtest_stamps_decision_quote_and_mid_on_every_fill():
    class BuyUpAndDown:
        def candidates(self, f):
            return [10]

        def decide(self, v, pos, bankroll):
            return [Order("up", 10, 0.6), Order("down", 10, 0.6)]

    f = features(S, tape([(S + 17, 1, .55, 100), (S + 17, -1, .45, 100)]), y=1.0,
                 ask=np.full(T, 0.52), bid=np.full(T, 0.48))
    res, fills = run(BuyUpAndDown(), [f], Config(participation=1.0))
    up = next(x for x in fills if x.side == "up")
    down = next(x for x in fills if x.side == "down")
    assert (up.quote, up.mid) == (pytest.approx(0.52), pytest.approx(0.50))
    assert (down.quote, down.mid) == (pytest.approx(0.52), pytest.approx(0.50))
    assert res["slippage"].iloc[0] == pytest.approx((0.55 - 0.52) * 10 + (0.55 - 0.52) * 10)
    assert res["half_spread"].iloc[0] == pytest.approx(0.02 * 10 * 2)


@pytest.mark.case
def test_fills_carry_the_belief_and_q_at_decision_and_split_the_pnl_exactly():
    from dataclasses import asdict
    from pmlab.pq import PARTS, decompose_fills

    class Believer:
        prob = "p_model"

        def candidates(self, f):
            return [10]

        def decide(self, v, pos, bankroll):
            return [Order("up", 10, 0.6), Order("down", 5, 0.6)]

    fair, belief = np.full(T, 0.5), np.full(T, 0.5)
    fair[10], belief[10], fair[11], belief[11] = 0.53, 0.58, 0.9, 0.9      # t = 11 is after the decision
    f = features(S, tape([(S + 17, 1, .55, 100), (S + 17, -1, .45, 100)]), y=0.0, fair=fair, p_model=belief)
    res, fills = run(Believer(), [f], Config(participation=1.0))
    assert len(fills) == 2 and all((x.p_up, x.q_up) == (pytest.approx(0.58), pytest.approx(0.53)) for x in fills)
    parts = decompose_fills(pd.DataFrame([asdict(x) for x in fills]).assign(y=f.y))
    assert parts["total_usd"].sum() == pytest.approx(res["pnl"].sum())
    assert sum(parts[f"{k}_usd"].sum() for k in PARTS[:-1]) == pytest.approx(res["pnl"].sum())
    up = parts[parts["side"] == "up"].iloc[0]
    assert up["model_edge"] == pytest.approx(0.05) and up["market_mispricing"] == pytest.approx(0.53 - 0.55)


@pytest.mark.case
def test_participation_is_shared_by_all_orders_on_the_same_print():
    tp = tape([(S + 5, 1, .50, 100)])
    prints, cfg = {}, Config(participation=0.5, print_delay=0)
    a = simulate_fill(tp, S, 3, Order("up", 40, .6), cfg, prints)
    b = simulate_fill(tp, S, 3, Order("up", 40, .6), cfg, prints)
    assert sum(x.shares for x in a) == 40 and sum(x.shares for x in b) == 10    # 50 in total, not 80


@pytest.mark.case
def test_orders_stay_open_until_their_last_possible_fill_is_known():
    class Once:
        def candidates(self, f):
            return [10, 14, 16, 18, 19]

        def decide(self, v, pos, bankroll):
            self.seen.append((v.t, pos.exposure(v.t)["up"]))
            return [Order("up", 5, .6, ttl=3)] if v.t == 10 else []

    s = Once()
    s.seen = []
    run_window(s, features(S, tape([])), 1000.0, Config(latency=1, print_delay=5))
    assert [e for t, e in s.seen if t in (14, 16, 18)] == [5, 5, 5] and dict(s.seen)[19] == 0


@pytest.mark.case
def test_sizing_sees_the_previous_windows_pnl_only_once_its_outcome_is_known():
    class Sizer:
        name = "sizer"

        def candidates(self, f):
            return [0, 4, 5]

        def decide(self, v, pos, bankroll):
            self.seen.append((v.start, v.t, bankroll))
            return [Order("up", 10, .6)] if v.start == S and v.t == 0 else []

    s = Sizer()
    s.seen = []
    won = features(S, tape([(S + 2, 1, .50, 100)]), y=1.0)             # 10 shares at 0.50 pay +5 − fee
    run(s, [won, features(S + T, tape([]))], Config(participation=1.0, print_delay=0))
    after = {t: b for start, t, b in s.seen if start == S + T}
    assert after[0] == after[4] == 1000.0 and after[5] > 1000.0


@pytest.mark.property
def test_a_run_never_sees_liquidity_taken_by_an_earlier_run_on_the_same_windows():
    """Regression: the per-print participation used to persist in the window cache, so BO trials and
    strategies evaluated one after another in the same process changed each other's fills."""
    tp = tape([(S + 5 + k, 1, .50, 10) for k in range(20)])
    f = features(S, tp, y=1.0, fair=np.full(T, 0.8))
    greedy = ProbTaker(edge=-1.0, kelly=1.0, max_stake=1000, max_window_cost=1000, max_window_frac=1.0,
                       t_min=0, t_max=10, max_age=1e9, name="greedy")
    first, _ = run(greedy, [f], Config(participation=1.0, print_delay=0))
    again, _ = run(greedy, [f], Config(participation=1.0, print_delay=0))
    assert len(first) == 1 and first["n_fills"].iloc[0] > 1 and first.equals(again)


@pytest.mark.case
def test_run_resets_once_then_reports_every_settled_window_in_time_order_after_it_ran():
    class Stateful:
        def __init__(self):
            self.calls = []

        def reset(self):
            self.calls.append(("reset",))

        def candidates(self, f):
            return [10]

        def decide(self, v, pos, bankroll):
            self.calls.append(("decide", v.start, bankroll))
            return [Order("up", 100, .5)] if v.start != S + 300 else []

        def on_settle(self, start, pnl, bankroll_after):
            self.calls.append(("settle", start, pnl, bankroll_after))

    ws = [features(S + 300 * i, tape([(S + 300 * i + 11, 1, .5, 1000)]), y=float(i != 2)) for i in range(3)]
    s = Stateful()
    res, _ = run(s, [ws[2], ws[0], ws[1]], Config(participation=1.0, print_delay=0))
    win, loss = 100 - 50 - taker_fee(100, .5), -50 - taker_fee(100, .5)
    assert s.calls == [("reset",),
                       ("decide", S, 1000.0), ("settle", S, pytest.approx(win), pytest.approx(1000 + win)),
                       ("decide", S + 300, pytest.approx(1000 + win)), ("settle", S + 300, 0.0, pytest.approx(1000 + win)),
                       ("decide", S + 600, pytest.approx(1000 + win)),
                       ("settle", S + 600, pytest.approx(loss), pytest.approx(1000 + win + loss))]
    assert list(res["pnl"]) == [pytest.approx(win), pytest.approx(loss)]
    s.calls.clear()
    again, _ = run(s, ws, Config(participation=1.0, print_delay=0))
    assert s.calls[0] == ("reset",) and s.calls.count(("reset",)) == 1 and again.equals(res)


class _Script:
    """Places exactly the given orders at their seconds."""
    prob = "fair"

    def __init__(self, orders):
        self.orders = orders

    def candidates(self, f):
        return sorted(self.orders)

    def decide(self, v, pos, bankroll):
        return [Order(**o) for o in self.orders.get(v.t, [])]


@pytest.mark.case
def test_implementation_shortfall_adds_arrival_cost_opportunity_cost_and_fees_and_makers_get_spread_and_markouts():
    """Audit R6 finding 10: half-spread + slippage against the decision mid left out the unfilled shares; a maker's
    'half-spread' against the decision mid was mid drift. Proxy mids; a crossed proxy (bid > ask) is left out of
    spread statistics."""
    ask, bid = np.full(T, 0.52), np.full(T, 0.48)
    ask[40], bid[40] = 0.45, 0.55                                                    # crossed proxy at t = 40
    ask[55:80], bid[55:80] = 0.53, 0.51                                              # mid 0.52 while the maker bid rests
    ask[80:], bid[80:] = 0.62, 0.58                                                  # mid 0.60 from t = 80 (Up token)
    tp = tape([(S + 17, 1, .51, 20), (S + 47, 1, .53, 4), (S + 60, -1, .40, 40)])
    f = features(tape_df=tp, y=1.0, ask=ask, bid=bid)
    cfg = Config(latency=1, print_delay=5, participation=0.5)
    orders = {10: [dict(side="up", shares=30, limit=0.52)],                          # 10 of 30 fill at 0.51
              40: [dict(side="up", shares=5, limit=0.60)],                           # crossed at decision; 2 fill at 0.53
              50: [dict(side="up", shares=15, limit=0.45, maker=True, ttl=20)]}     # a print through at 0.40 fills 15
    pos = run_window(_Script(orders), f, 1000.0, cfg)
    s = settle(pos, 1.0)
    fills = {(x.t, x.maker): x for x in pos.fills}
    assert [(k, round(x.shares, 6)) for k, x in sorted(fills.items())] == [((10, False), 10), ((40, False), 2), ((50, True), 15)]
    mid10, mid40, mid50 = 0.50, 0.50, 0.50
    arrival = 10 * (0.51 - mid10) + 2 * (0.53 - mid40) + 15 * (0.45 - mid50)
    opportunity = 20 * (1 - mid10) + 3 * (1 - mid40) + 0 * (1 - mid50)
    assert s["is_arrival_cost"] == pytest.approx(arrival) and s["is_opportunity_cost"] == pytest.approx(opportunity)
    assert s["implementation_shortfall"] == pytest.approx(arrival + opportunity + s["fees"])
    assert s["is_paper_pnl"] - s["pnl"] == pytest.approx(s["implementation_shortfall"])            # Perold's identity
    assert s["ordered_shares"] == 50 and s["unfilled_shares"] == pytest.approx(23)
    assert s["half_spread_uncrossed"] == pytest.approx(10 * (0.52 - 0.50) + 15 * (0.45 - 0.50))  # t = 40 left out
    assert s["spread_fills_crossed"] == 1
    # the maker decided at t 50 (mid 0.50), filled at S + 60 (mid 0.52 then), the mid was 0.60 thirty seconds later; Up won
    assert s["maker_effective_spread"] == pytest.approx(15 * (0.52 - 0.45))
    assert s["maker_markout_30s"] == pytest.approx(15 * (0.60 - 0.52))
    assert s["maker_markout_settle"] == pytest.approx(15 * (1.0 - 0.52))
    assert s["maker_effective_spread"] + s["maker_markout_settle"] == pytest.approx(15 * (1.0 - 0.45))   # = its P&L
    assert s["maker_spread_shares"] == 15
    missed = []                                                                      # a window whose orders all missed
    res, _ = run(_Script({10: [dict(side="up", shares=30, limit=0.30)]}), [features(start=S + 300, tape_df=tp, y=1.0)], cfg, missed)
    assert res.empty and len(missed) == 1 and missed[0]["is_opportunity_cost"] == pytest.approx(30 * (1 - 0.49))  # default quotes
    assert missed[0]["implementation_shortfall"] == pytest.approx(missed[0]["is_paper_pnl"])


@pytest.mark.case
def test_candidates_never_see_the_outcome_or_the_tape_and_seconds_are_asked_once_in_time_order():
    """pricing review, backtest 2: candidates() received the full window (a strategy could trade only in windows it
    would win) and its order was trusted (a later second's order counted in an earlier decision's exposure)."""
    class Peek:
        def __init__(self):
            self.seen, self.asked = [], []

        def candidates(self, f):
            self.seen.append((f.y, f.tape))
            return [20, 10, 10, T, -1] if f.y is None else ([10] if f.y == 1 else [])

        def decide(self, v, pos, bankroll):
            self.asked.append(v.t)
            return []

    for y in (1.0, 0.0):
        s = Peek()
        run_window(s, features(y=y), 1000.0, Config())
        assert s.seen == [(None, None)] and s.asked == [10, 20]
