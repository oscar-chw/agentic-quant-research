import json
from datetime import datetime, timedelta, timezone
from math import comb

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from pmlab import evaluation, metrics
from pmlab.afml import backtest_pipeline as bp
from pmlab.afml import backtest_stats, betsizing, cpcv, strategy_risk

H = 3600
DAY0 = int(datetime(2026, 3, 18, tzinfo=timezone(timedelta(hours=8))).timestamp())      # Taiwan midnight


def _row(start, t0, t1, side, ask_up, bid_up, up_won, **extra):
    price = ask_up if side > 0 else 1 - bid_up
    fee = 0.07 * price * (1 - price)
    payoff = up_won if side > 0 else 1 - up_won
    cost = price + fee
    return {"start": start, "t0": t0, "t1": t1, "rule": "q_taker", "side": side, "ask_up": ask_up, "bid_up": bid_up,
            "price": price, "fee": fee, "cost": cost, "up_won": up_won, "payoff": payoff, "pnl": payoff - cost,
            "ret": (payoff - cost) / cost, "label": int(payoff > cost), "meta_label": int(payoff > cost),
            "avg_uniqueness": 0.1, "w_return_raw": 1.0, **extra}


def _events(days=12, windows_per_day=8, per_window=3, seed=0, signal=0.0):
    """Windows over each Taiwan day, always including its first (00:00) and last (23:55) window; `signal` moves
    P(side wins) with the feature f_signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        slots = np.r_[0, np.sort(rng.choice(np.arange(1, 287), windows_per_day - 2, replace=False)), 287]
        for slot in slots:
            start = DAY0 + d * 86400 + 300 * int(slot)
            x = rng.normal()
            up_won = float(rng.random() < 0.5 + signal * np.tanh(x))
            ask_up = rng.uniform(0.3, 0.7)
            for t in np.sort(rng.choice(np.arange(5, 280), per_window, replace=False)):
                side = int(rng.choice([-1, 1]))
                rows.append(_row(start, start + int(t), start + 300, side, ask_up, ask_up - 0.02, up_won,
                                 f_signal=side * x + rng.normal(scale=0.3)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- CPCV paths

@pytest.mark.property
@pytest.mark.parametrize("N,k", [(6, 2), (5, 3), (4, 1)])
def test_phi_paths_each_cover_every_group_and_window_exactly_once(N, k):
    ev = _events()
    value = {}

    def model(train, test):                        # a constant per split that identifies which split predicted
        v = (len(value) + 1) / 100
        value[frozenset(test.index)] = v
        return np.full(len(test), v)
    res = bp.run_cpcv(ev, bp.Strategy("spy", model=model), N, k, embargo=0)
    P = res["paths"]
    assert len(res["splits"]) == comb(N, k) and P.shape == (comb(N - 1, k - 1), N)     # phi[N, k] = k/N C(N, k)
    split_value = {s: value[frozenset(res["rows"]["row"].to_numpy()[sp["test"]])] for s, sp in enumerate(res["splits"])}
    for j in range(P.shape[0]):
        bets = bp.path_bets(res, j)
        assert sorted(bets["row"]) == sorted(ev.index)                    # every event exactly once on the path
        assert set(bets["group"]) == set(range(N))
        for g in range(N):
            gb = bets[bets["group"] == g]
            assert set(gb["split"]) == {P[j, g]} and g in res["splits"][P[j, g]]["groups"]
            assert np.allclose(gb["prob"], split_value[P[j, g]])
        assert res["window_pnl"][j].shape == (ev["start"].nunique(),)
        assert res["window_pnl"][j].sum() == pytest.approx(bets["pnl"].sum())
        assert res["window_pnl"][j].sum() == pytest.approx(ev["ret"].sum())   # $1 a bet earns its P&L per $
    pairs = sorted((int(P[j, g]), g) for j in range(P.shape[0]) for g in range(N))
    assert pairs == sorted((s, g) for s, sp in enumerate(res["splits"]) for g in sp["groups"])


@pytest.mark.case
def test_groups_are_whole_taiwan_days_not_utc_days():
    ev = _events(days=6)
    day = (ev["start"] - DAY0) // 86400
    ev = ev[~((day == 2) & (ev["start"] > DAY0 + 2 * 86400 + 12 * H))]       # days of unequal size: groups of rows are not days
    assert day[ev.index].value_counts().nunique() > 1
    res = bp.run_cpcv(ev, bp.Strategy("all"), n_groups=6, k_test=1, embargo=0)
    rows = res["rows"]
    tw_day = (rows["start"].to_numpy() - DAY0) // 86400
    assert (res["group"] == tw_day).all()
    # the last window of a Taiwan day (15:55 UTC) and the first of the next (16:00 UTC) share a UTC date
    last = rows["start"].to_numpy() == DAY0 + 287 * 300
    first = rows["start"].to_numpy() == DAY0 + 86400
    assert set(res["group"][last]) == {0} and set(res["group"][first]) == {1}


@pytest.mark.case
def test_purging_and_embargo_remove_exactly_the_overlapping_and_following_events():
    d = lambda day, hour: DAY0 + day * 86400 + int(hour * H)
    spans = [(d(0, 10), d(0, 11)),      # 0
             (d(0, 23), d(1, 3)),       # 1 runs into day 1
             (d(1, 2), d(1, 3)),        # 2
             (d(1, 22), d(1, 23)),      # 3 day 1's last span ends 23:00
             (d(2, 0.5), d(2, 1)),      # 4 starts 1.5 h after it
             (d(2, 12), d(2, 13)),      # 5
             (d(3, 5), d(3, 6))]        # 6
    ev = pd.DataFrame([_row(t0 - t0 % 300, t0, t1, 1, 0.5, 0.48, 1.0) for t0, t1 in spans])
    res = bp.run_cpcv(ev, bp.Strategy("all"), n_groups=4, k_test=1, embargo=2 * H)
    got = {sp["groups"]: (sorted(res["rows"]["row"].to_numpy()[sp["train"]]), sp["purged"], sp["embargoed"])
           for sp in res["splits"]}
    assert got[(0,)] == ([3, 4, 5, 6], 1, 0)          # row 2 overlaps day 0's envelope, which row 1 stretches to 03:00
    assert got[(1,)] == ([0, 5, 6], 1, 1)             # row 1 purged, row 4 embargoed
    assert got[(2,)] == ([0, 1, 2, 3, 6], 0, 0)
    assert got[(3,)] == ([0, 1, 2, 3, 4, 5], 0, 0)


@pytest.mark.property
def test_no_configuration_ever_trains_on_its_own_test_groups_or_sees_test_outcomes():
    ev = _events(days=10, windows_per_day=12, seed=3)
    days = np.unique((ev["start"] - DAY0) // 86400)
    seen = []

    def spy(train, test):
        seen.append((train, test))
        return np.full(len(test), 0.6)
    configs = [bp.Strategy("meta", model=spy, threshold=0.5),
               bp.Strategy("meta_sized", model=spy, threshold=0.55, sizing=lambda p, rows: 5 * p)]
    for cfg in configs:
        for N, k, emb in ((5, 2, 86400), (5, 1, 3 * H)):
            seen.clear()
            res = bp.run_cpcv(ev, cfg, N, k, embargo=emb)
            assert len(seen) == len(res["splits"])
            for train, test in seen:
                assert not set(bp.OUTCOMES) & set(test.columns) and "label" in train.columns
                test_days = set((test["start"] - DAY0) // 86400)
                train_days = set((train["start"] - DAY0) // 86400)
                assert not train_days & test_days and not set(train.index) & set(test.index)
                assert not np.any((train["t0"].to_numpy()[:, None] < test["t1"].to_numpy()[None, :])
                                  & (train["t1"].to_numpy()[:, None] > test["t0"].to_numpy()[None, :]))
                in_test = np.isin(days, list(test_days))
                ends = np.flatnonzero(in_test & ~np.r_[in_test[1:], False])      # last day of each run of test days
                for e in ends:
                    hi = test.loc[(test["start"] - DAY0) // 86400 == days[e], "t1"].max()
                    assert not np.any((train["t0"] >= hi) & (train["t0"] < hi + emb))
                    if emb >= 86400 and e + 1 < len(days):
                        assert days[e + 1] not in train_days


# ----------------------------------------------------------------------------------------------- orders and shortfall

@pytest.mark.case
def test_filter_sizing_and_implementation_shortfall_on_a_hand_example():
    w = DAY0 + 10 * H
    rows = [_row(w, w + 10, w + 300, 1, 0.40, 0.36, 1.0, p_model=0.9, stake=4.168, fill_ratio=0.5),
            _row(w, w + 20, w + 300, -1, 0.40, 0.36, 1.0, p_model=0.9, stake=3.28064, fill_ratio=1.0),
            _row(w + 86400, w + 86410, w + 86700, 1, 0.5, 0.48, 0.0, p_model=0.4, stake=9.0, fill_ratio=1.0),
            _row(w + 86400, w + 86420, w + 86700, 1, 0.5, 0.48, 0.0, p_model=0.9, stake=0.0, fill_ratio=1.0)]
    ev = pd.DataFrame(rows)
    strat = bp.Strategy("hand", model=lambda tr, te: te["p_model"].to_numpy(), threshold=0.5,
                        sizing=lambda p, r: r["stake"].to_numpy())
    res = bp.run_cpcv(ev, strat, n_groups=2, k_test=1, embargo=0)
    bets = bp.path_bets(res, 0).set_index("row")
    assert list(bets.index) == [0, 1]                                  # below the threshold, or no dollars: no order
    assert bets["ordered"].tolist() == pytest.approx([10.0, 5.0]) and bets["filled"].tolist() == pytest.approx([5.0, 5.0])
    # Up at 0.40 + fee 0.0168, mid 0.38, half filled; Up won:  pnl 5 x 0.5832, paper 10 x 0.62, arrival 5 x 0.02,
    # fees 5 x 0.0168, opportunity 5 x 0.62.  Down at 0.64 + fee 0.016128, mid 0.62, lost: pnl -5 x 0.656128,
    # paper -5 x 0.62, arrival 5 x 0.02, fees 5 x 0.016128
    assert bets["pnl"].tolist() == pytest.approx([2.916, -3.28064])
    assert bets["paper"].tolist() == pytest.approx([6.2, -3.1])
    assert bets["arrival_cost"].tolist() == pytest.approx([0.1, 0.1])
    assert bets["fees"].tolist() == pytest.approx([0.084, 0.08064])
    assert bets["opportunity_cost"].tolist() == pytest.approx([3.1, 0.0])
    s = bp.shortfall(bp.path_bets(res, 0))
    assert s["implementation_shortfall"] == pytest.approx(3.46464)
    assert s["is_paper"] - s["is_pnl"] == pytest.approx(s["implementation_shortfall"])
    assert s["turnover"] == pytest.approx(5.2) and s["fees_per_turnover"] == pytest.approx(0.16464 / 5.2)
    assert s["slippage_per_turnover"] == pytest.approx(0.2 / 5.2) and s["pnl_per_turnover"] == pytest.approx(-0.36464 / 5.2)
    assert s["return_on_execution_costs"] == pytest.approx(-0.36464 / 0.36464)
    assert res["window_pnl"][0].tolist() == pytest.approx([-0.36464, 0.0])


# ----------------------------------------------------------------------------------------------- ch. 14 statistics

@pytest.mark.case
def test_hhi_of_a_path_against_hand_arithmetic():
    # (Taiwan day, hour, stake, Up won): wins a, 2a, a; losses -10, -30, -10; day 1 has a window and no event; the
    # last window orders nothing
    plan = [(0, 10, 10.0, 1.0), (0, 11, 20.0, 1.0), (0, 12, 10.0, 0.0),
            (2, 10, 10.0, 1.0), (2, 11, 30.0, 0.0), (2, 12, 10.0, 0.0), (2, 13, 0.0, 1.0)]
    rows = []
    for day, hour, stake, up in plan:
        w = DAY0 + day * 86400 + hour * H
        rows.append(_row(w, w + 10, w + 300, 1, 0.5, 0.48, up, stake=stake))
    ev = pd.DataFrame(rows)
    windows = np.r_[ev["start"], DAY0 + 86400 + 10 * H]
    res = bp.run_cpcv(ev, bp.Strategy("hand", sizing=lambda p, r: r["stake"].to_numpy()), 3, 1, embargo=0,
                      windows=windows)
    row = bp.path_statistics(res, 0)
    assert row["traded_windows"] == 6
    assert row["hhi_plus"] == pytest.approx((0.375 - 1 / 3) / (1 - 1 / 3))      # w = .25, .5, .25
    assert row["hhi_minus"] == pytest.approx((0.44 - 1 / 3) / (1 - 1 / 3))      # w = .2, .6, .2
    assert row["hhi_time"] == pytest.approx((0.5 - 1 / 3) / (1 - 1 / 3))        # 3, 0, 3 traded windows a day


@pytest.mark.case
def test_drawdown_and_time_under_water_on_a_hand_series():
    dd = bp.drawdowns(np.arange(10), [0, 2, 1, 3, 3, 0, 2, 3, 4, 1])
    assert dd[["hwm_time", "hwm", "trough", "end_time", "recovered"]].values.tolist() == \
        [[1, 2, 1, 3, True], [3, 3, 0, 8, True], [8, 4, 1, 9, False]]
    assert dd["drawdown"].tolist() == [1, 3, 3] and dd["time_under_water"].tolist() == [2, 5, 1]
    rel = bp.drawdowns([0, 1, 2, 3], [10, 12, 9, 13], relative=True)
    assert rel["drawdown"].tolist() == pytest.approx([0.25]) and rel["time_under_water"].tolist() == [2]
    assert bp.drawdowns([0, 1, 2], [0, 1, 2]).empty


@pytest.mark.case
def test_twrr_bet_timing_and_correlation_on_hand_cases():
    assert bp.twrr([10, -22, 22], 100.0, 0.5) == pytest.approx(1.1 ** 2 - 1)       # 100 -> 110 -> 88 -> 110
    assert bp.twrr([10, -120, 5], 100.0, 1.0) == -1.0 and np.isnan(bp.twrr([1.0], None, 1.0))
    b = pd.DataFrame({"start": [0, 0, 0, 300, 300, 300], "t0": [10, 20, 30, 310, 320, 330],
                      "t1": [300, 300, 300, 600, 600, 600], "side": [1, 1, -1, -1, 1, 1],
                      "filled": [10.0, 5.0, 20.0, 4.0, 4.0, 1.0]})
    # window 0: +10 (bet), +15, -5 (flip: bet); window 300: -4 (bet), 0 (flat), +1 (bet)
    t = bp.bet_timing(b)
    assert (t["bets"], t["purchases"]) == (4, 6) and t["up_share"] == pytest.approx(4 / 6)
    assert t["holding_seconds"] == pytest.approx((10 * 290 + 5 * 280 + 20 * 270 + 4 * 290 + 4 * 280 + 270) / 44)


@pytest.mark.case
def test_path_statistics_read_the_path_series():
    w = DAY0 + 10 * H
    ev = pd.DataFrame([_row(w, w + 10, w + 300, 1, 0.5, 0.48, 1.0), _row(w + 300, w + 310, w + 600, 1, 0.5, 0.48, 0.0),
                       _row(w + 86400, w + 86410, w + 86700, -1, 0.52, 0.5, 0.0),
                       _row(w + 86700, w + 86710, w + 87000, 1, 0.5, 0.48, 1.0)])
    res = bp.run_cpcv(ev, bp.Strategy("s", sizing=bp.fixed_stake(10.0)), n_groups=2, k_test=1, embargo=0)
    st = bp.statistics(res, bankroll=100.0)
    row = st["paths"].iloc[0]
    win, loss = 10 * (1 - ev["cost"][0]) / ev["cost"][0], -10.0
    assert res["window_pnl"][0].tolist() == pytest.approx([win, loss, win, win])
    assert (row["bets"], row["traded_windows"], row["hit_ratio"]) == (4, 4, 0.75)
    assert row["dd_max"] == pytest.approx(-loss / (100 + win))
    assert row["tuw_max_days"] == pytest.approx((w + 87000 - (w + 300)) / 86400)    # HWM at window 0's end, exceeded at window 3's
    assert row["corr_underlying"] == pytest.approx(1 / np.sqrt(3))    # x = a, -10, a, a on y = 1, -1, -1, 1: (a + 10) / sqrt(3 (a + 10)^2)
    assert row["twrr_annual"] == pytest.approx((1 + (3 * win + loss) / 100) ** (1 / row["years"]) - 1)
    assert st["summary"].loc["pnl", "mean"] == pytest.approx(3 * win + loss)


@pytest.mark.case
def test_general_characteristics_and_performance_splits_on_a_hand_path():
    """14.3's average AUM, maximum dollar position and annualised turnover, and 14.4's long/short split and returns
    from hits and misses (docs/afml_sections/ch14.md F14.1, F14.2, F14.3). One window a day: day 0 buys $10 of Up
    290 s before settlement and wins, day 1 buys $20 of Down 200 s before settlement and loses, day 2 is untraded."""
    w = DAY0 + 10 * H
    ev = pd.DataFrame([_row(w, w + 10, w + 300, 1, 0.5, 0.48, 1.0, stake=10.0),
                       _row(w + 86400, w + 86500, w + 86700, -1, 0.52, 0.5, 1.0, stake=20.0),
                       _row(w + 172800, w + 172810, w + 173100, 1, 0.5, 0.48, 0.0, stake=0.0)])
    res = bp.run_cpcv(ev, bp.Strategy("hand", sizing=lambda p, r: r["stake"].to_numpy()), 3, 1, embargo=0)
    row = bp.path_statistics(res, 0)
    bets = bp.path_bets(res, 0)
    assert len(bets) == 2 and bets["side"].tolist() == [1, -1]

    # 14.4: the P&L split by side, and hits/misses as returns of the dollars invested
    win, loss = bets["pnl"].to_numpy()
    assert win > 0 > loss
    assert row["pnl_long"] == pytest.approx(win) and row["pnl_short"] == pytest.approx(loss)
    legs = bp.window_legs(bets)
    invested = (legs["cost"] + legs["fees"]).to_numpy()
    assert row["avg_hit"] == pytest.approx(win / invested[0]) and row["avg_miss"] == pytest.approx(loss / invested[1])
    assert row["avg_hit"] != pytest.approx(row["avg_hit_dollars"])       # returns, not dollars
    assert row["avg_hit_dollars"] == pytest.approx(win) and row["avg_miss_dollars"] == pytest.approx(loss)
    assert row["avg_miss"] == pytest.approx(-1.0)                        # a losing binary bet loses everything
    assert row["flat_windows"] == 0 and row["hit_ratio"] == pytest.approx(0.5)

    # 14.3: value open = filled shares x price, held from t0 to settlement; the two windows never overlap
    value = (bets["filled"] * bets["price"]).to_numpy()
    span = float(res["windows"][-1] + bp.T - res["windows"][0])
    assert row["avg_aum"] == pytest.approx((value[0] * 290 + value[1] * 200) / span)
    assert row["max_position_dollars"] == pytest.approx(value.max())
    assert row["turnover_annual"] == pytest.approx(value.sum() / (span / bp.YEAR) / row["avg_aum"])
    assert row["max_position_dollars"] > row["avg_aum"]                  # bursty, as a 5-minute strategy must be


@pytest.mark.case
def test_capacity_is_the_average_aum_of_the_largest_size_that_still_reaches_the_target_sharpe():
    """14.3 (docs/afml_sections/ch14.md F14.1): capacity is reported only when a market-impact coefficient is given,
    and is then the path's average AUM scaled by the largest size whose net Sharpe still reaches target_sharpe. The
    dollars a window sends are its cost plus fees, 0 for a window it passed on."""
    w = DAY0 + 10 * H
    ups = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    ev = pd.DataFrame([_row(w + (i // 3) * 86400 + 300 * (i % 3), w + (i // 3) * 86400 + 300 * (i % 3) + 10,
                            w + (i // 3) * 86400 + 300 * (i % 3) + 300, 1, 0.5, 0.48, u,
                            stake=10.0 if i != 2 else 0.0) for i, u in enumerate(ups)])
    res = bp.run_cpcv(ev, bp.Strategy("hand", sizing=lambda p, r: r["stake"].to_numpy()), 2, 1, embargo=0)
    assert "capacity" not in bp.path_statistics(res, 0)                  # no impact model, no capacity claimed
    row = bp.path_statistics(res, 0, impact=1e-3, target_sharpe=0.5)
    legs = bp.window_legs(bp.path_bets(res, 0))
    sent = pd.Series(0.0, index=pd.Index(res["windows"]))
    sent.loc[legs.index] = (legs["cost"] + legs["fees"]).to_numpy()
    assert (sent.to_numpy() == 0).sum() == 1                             # the window with a zero stake sends nothing
    want = backtest_stats.capacity(res["window_pnl"][0], sent.to_numpy(), 1e-3, 0.5,
                                   row["windows"] / row["years"], row["avg_aum"])
    assert row["capacity_multiple"] == pytest.approx(want["multiple"])
    assert row["capacity"] == pytest.approx(want["multiple"] * row["avg_aum"])
    assert row["capacity"] > row["avg_aum"] > 0                          # this book is deep enough to scale up
    assert bp.path_statistics(res, 0, impact=0.0, target_sharpe=0.5)["capacity"] == np.inf


@pytest.mark.case
def test_a_window_with_exactly_zero_pnl_is_neither_a_hit_nor_a_miss():
    """14.4 (docs/afml_sections/ch14.md F14.3): hits gained, misses lost; a flat window is counted apart."""
    traded = np.array([4.0, 0.0, -2.0])
    ret = traded / np.array([10.0, 10.0, 10.0])
    assert ret[traded > 0].mean() == pytest.approx(0.4) and ret[traded < 0].mean() == pytest.approx(-0.2)
    assert ret[traded <= 0].mean() == pytest.approx(-0.1)                # what counting the flat window as a miss gives
    w = DAY0 + 10 * H
    ev = pd.DataFrame([_row(w, w + 10, w + 300, 1, 0.5, 0.48, 1.0, stake=10.0),
                       _row(w + 86400, w + 86410, w + 86700, 1, 0.5, 0.48, 0.0, stake=10.0)])
    ev.loc[1, "payoff"] = ev.loc[1, "cost"]                              # settles exactly at cost: zero P&L
    ev.loc[1, "pnl"] = 0.0
    res = bp.run_cpcv(ev, bp.Strategy("hand", sizing=lambda p, r: r["stake"].to_numpy()), 2, 1, embargo=0)
    row = bp.path_statistics(res, 0)
    assert row["flat_windows"] == 1 and np.isnan(row["avg_miss"]) and row["avg_hit"] > 0


@pytest.mark.case
def test_the_percentile_drawdown_and_time_under_water_are_the_quantiles_of_the_episodes():
    """14.5.3 (docs/afml_sections/ch14.md F14.5): dd_p95 and tuw_p95_days are the 95th percentiles of the same
    episode series the maxima come from, not of the equity curve."""
    w = DAY0 + 10 * H
    ups = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    stakes = [10.0, 5.0, 10.0, 10.0, 20.0, 10.0, 10.0, 30.0, 10.0, 10.0]      # losses of different sizes
    ev = pd.DataFrame([_row(w + (i // 5) * 86400 + 300 * (i % 5), w + (i // 5) * 86400 + 300 * (i % 5) + 10,
                            w + (i // 5) * 86400 + 300 * (i % 5) + 300, 1, 0.5, 0.48, u, stake=st)
                       for i, (u, st) in enumerate(zip(ups, stakes))])
    res = bp.run_cpcv(ev, bp.Strategy("hand", sizing=lambda p, r: r["stake"].to_numpy()), 2, 1, embargo=0)
    row = bp.path_statistics(res, 0)
    equity = np.concatenate([[0.0], np.cumsum(res["window_pnl"][0])])
    times = np.concatenate([[res["windows"][0]], res["windows"] + bp.T])
    dd = bp.drawdowns(times, equity)
    assert len(dd) >= 2                                                   # several episodes, so a quantile can differ
    assert row["dd_max"] == pytest.approx(dd["drawdown"].max())
    assert row["dd_p95"] == pytest.approx(np.quantile(dd["drawdown"], 0.95))
    assert row["dd_p95"] < row["dd_max"]
    assert row["tuw_max_days"] == pytest.approx(dd["time_under_water"].max() / 86400)
    assert row["tuw_p95_days"] == pytest.approx(np.quantile(dd["time_under_water"], 0.95) / 86400)
    assert row["tuw_p95_days"] < row["tuw_max_days"]


@pytest.mark.case
def test_sharpe_annual_scales_the_per_window_sharpe_by_every_window_of_the_span():
    """14.7.4 (docs/afml_sections/ch14.md F14.9): the annualisation factor is sqrt(all windows per year), not sqrt of
    the TRADED windows per year, which would inflate a selective strategy."""
    w = DAY0 + 10 * H
    ups = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    ev = pd.DataFrame([_row(w + (i // 4) * 86400 + 300 * (i % 4), w + (i // 4) * 86400 + 300 * (i % 4) + 10,
                            w + (i // 4) * 86400 + 300 * (i % 4) + 300, 1, 0.5, 0.48, u,
                            stake=10.0 if i % 2 == 0 else 0.0) for i, u in enumerate(ups)])
    res = bp.run_cpcv(ev, bp.Strategy("hand", sizing=lambda p, r: r["stake"].to_numpy()), 2, 1, embargo=0)
    row = bp.path_statistics(res, 0)
    assert row["traded_windows"] == 4 < row["windows"] == 8
    assert row["sharpe"] == pytest.approx(metrics.sharpe(res["window_pnl"][0]))
    assert row["sharpe_annual"] == pytest.approx(row["sharpe"] * np.sqrt(row["windows"] / row["years"]))
    assert row["sharpe_annual"] != pytest.approx(row["sharpe"] * np.sqrt(row["traded_windows"] / row["years"]))


@pytest.mark.case
def test_concentration_is_over_traded_window_dollars_where_the_book_uses_bet_returns():
    """14.5.1 (docs/afml_sections/ch14.md F14.4): the departure is stated in the module docstring and pinned here.
    Two windows with the same RETURN but stakes of 1 and 9 concentrate almost fully in dollars and not at all in
    returns."""
    w = DAY0 + 10 * H
    ev = pd.DataFrame([_row(w, w + 10, w + 300, 1, 0.5, 0.48, 1.0, stake=1.0),
                       _row(w + 300, w + 310, w + 600, 1, 0.5, 0.48, 1.0, stake=1.0),
                       _row(w + 86400, w + 86410, w + 86700, 1, 0.5, 0.48, 1.0, stake=9.0)])
    res = bp.run_cpcv(ev, bp.Strategy("hand", sizing=lambda p, r: r["stake"].to_numpy()), 2, 1, embargo=0)
    row = bp.path_statistics(res, 0)
    legs = bp.window_legs(bp.path_bets(res, 0))
    rets = (legs["pnl"] / (legs["cost"] + legs["fees"])).to_numpy()
    assert rets[0] == pytest.approx(rets[1]) == pytest.approx(rets[2])      # same return, different stakes
    weights = np.array([1, 1, 9]) / 11                                      # dollar weights, the P&L being 1:1:9
    assert row["hhi_plus"] == pytest.approx((np.sum(weights ** 2) - 1 / 3) / (1 - 1 / 3))
    assert row["hhi_plus"] > 0.5                                            # equal returns alone would give 0
    assert "the book concentrates a bet's RETURNS" in bp.__doc__


# ----------------------------------------------------------------------------------------------- ch. 15

@pytest.mark.case
def test_precision_needed_matches_the_books_worked_examples():
    assert bp.payout_sharpe(0.7, [0.005], [-0.01], 260) == pytest.approx(1.173, abs=5e-4)     # 15.3
    assert bp.precision_needed([0.005], [-0.01], 260, 2.0) == pytest.approx(0.72, abs=5e-3)
    assert bp.precision_needed([0.005], [-0.01], 260, 2.0) == pytest.approx(strategy_risk.implied_precision(-0.01, 0.005, 260, 2.0))
    assert bp.precision_needed([1.0], [-1.0], 52, 2.0) == pytest.approx(0.6336, abs=5e-4)     # 15.2: weekly bets
    assert bp.payout_sharpe(0.55, [1.0], [-1.0], 396) == pytest.approx(2.0, abs=5e-3)
    assert np.isnan(bp.precision_needed([0.001, 0.002], [-1.0, -1.0], 10, 20.0))          # Sharpe 9.5 even at p = 1


@pytest.mark.case
def test_window_risk_reads_each_windows_two_payouts_on_a_hand_example():
    legs = pd.DataFrame({"up": [10.0, 0.0, 6.0, 3.0], "down": [0.0, 4.0, 4.0, 3.0], "cost": [5.0, 2.0, 3.0, 2.5],
                         "fees": [0.1, 0.05, 0.1, 0.05], "up_won": [1.0, 1.0, 0.0, 1.0]},
                        index=[0, 300, 600, 900])
    legs["pnl"] = np.where(legs["up_won"] == 1, legs["up"], legs["down"]) - legs["cost"] - legs["fees"]
    r = bp.window_risk(legs, years=0.01, target_sharpe=2.0, years_judged=0.1)
    # window 900 holds 3 Up and 3 Down: no outcome risk. Payouts (pi+, pi-): (4.9, -5.1) won, (1.95, -2.05) lost,
    # (2.9, 0.9) got the smaller one although it made money
    assert (r["windows"], r["bets_per_year"], r["precision"]) == (3, pytest.approx(300.0), pytest.approx(1 / 3))
    assert r["pi_plus"] == pytest.approx(9.75 / 3) and r["pi_minus"] == pytest.approx(-6.25 / 3)
    assert r["breakeven_precision"] == pytest.approx((6.25 / 3) / (16 / 3))
    assert r["precision_needed"] == pytest.approx(bp.precision_needed([4.9, 1.95, 2.9], [-5.1, -2.05, 0.9], 300, 2.0))
    assert r["prob_failure"] == pytest.approx(bp.prob_failure_binomial(1 / 3, r["precision_needed"], 30))


@pytest.mark.eval
def test_precision_needed_with_varying_payouts_reaches_the_target_in_simulation():
    """Threshold: 2,000 simulated years of 1,000 bets at the precision needed for Sharpe 2 give a pooled annual
    Sharpe within 0.1 of 2 (the two-point formula on mean payouts gives about 1.8 here)."""
    rng = np.random.default_rng(1)
    pp, pm = rng.uniform(0.1, 2.0, 200), -rng.uniform(0.3, 1.0, 200)
    n, theta = 1000, 2.0
    p = bp.precision_needed(pp, pm, n, theta)
    idx = rng.integers(0, 200, (2000, n))
    x = np.where(rng.random((2000, n)) < p, pp[idx], pm[idx])
    assert x.mean() / x.std() * np.sqrt(n) == pytest.approx(theta, abs=0.1)
    assert bp.payout_sharpe(p, pp, pm, n) == pytest.approx(theta)


@pytest.mark.eval
def test_binomial_failure_probability_is_the_books_bootstrap_of_precision():
    """Threshold: within 0.015 of 20,000 draws of 100 outcomes with replacement from 60 observed (37 wins)."""
    won = np.r_[np.ones(37), np.zeros(23)]
    rng = np.random.default_rng(2)
    for p_star in (0.55, 0.6, 0.65):
        boot = won[rng.integers(0, 60, (20000, 100))].mean(axis=1)
        assert bp.prob_failure_binomial(37 / 60, p_star, 100) == pytest.approx(np.mean(boot < p_star), abs=0.015)
    assert bp.prob_failure_binomial(0.6, np.nan, 100) == 1.0
    assert bp.prob_failure_binomial(0.6, 0.5, 100) == pytest.approx(binom.cdf(49, 100, 0.6))


# ----------------------------------------------------------------------------------------------- selection

def _fake(windows, pnl):
    return {"windows": np.asarray(windows), "window_pnl": np.atleast_2d(np.asarray(pnl, dtype=float))}


@pytest.mark.case
def test_config_matrix_is_path_mean_pnl_per_taiwan_day_with_idle_days_zero():
    w = [DAY0 + 287 * 300, DAY0 + 86400, DAY0 + 86700]                      # 23:55 day 0, 00:00 and 00:05 day 1
    a = _fake(w, [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    b = _fake(w[:1], [[7.0], [9.0]])
    M = bp.config_matrix({"a": a, "b": b})
    assert M.columns.tolist() == ["a", "b"] and M.values.tolist() == [[2.0, 8.0], [7.0, 0.0]]
    assert bp.config_matrix({"a": a}, path=1).values.ravel().tolist() == [3.0, 9.0]
    assert bp.config_matrix({"a": a}, by="window").values.ravel().tolist() == [2.0, 3.0, 4.0]


@pytest.mark.eval
def test_pbo_is_high_for_an_overfit_configuration_set_and_low_for_a_clean_one():
    """Thresholds: PBO >= 0.9 when each configuration wins on its own half of the blocks and loses on the rest (the
    in-sample winner is the out-of-sample loser by construction); PBO <= 0.1 when one configuration has real skill."""
    from itertools import combinations
    rng = np.random.default_rng(4)
    days = [DAY0 + d * 86400 + 12 * H for d in range(64)]
    block = np.arange(64) // 8
    overfit = {f"c{i}": _fake(days, np.where(np.isin(block, c), 1.0, -1.0) + rng.normal(0, 0.1, 64))
               for i, c in enumerate(combinations(range(8), 4))}
    r = bp.pbo_cscv(overfit, n_blocks=8)
    assert r["combinations"] == 70 and r["pbo"] >= 0.9 and r["prob_oos_loss"] == 1.0
    clean = {f"c{i}": _fake(days, rng.normal(0, 1, 64) + (1.0 if i == 0 else 0.0)) for i in range(20)}
    assert bp.pbo_cscv(clean, n_blocks=8)["pbo"] <= 0.1


def _logistic(train, test):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression().fit(train[["f_signal"]], train["label"])
    return m.predict_proba(test[["f_signal"]])[:, 1]


@pytest.mark.eval
def test_synthetic_end_to_end_with_dsr_from_the_registry_count_plus_the_studys_configurations(tmp_path):
    """Smoke run on synthetic events with a real signal. Thresholds: the meta-label filter's mean path P&L per $
    staked beats the unfiltered primary rule's; every per-path statistic named below is finite."""
    ev = _events(days=20, windows_per_day=24, per_window=3, seed=5, signal=0.35)
    configs = [bp.Strategy("primary"), bp.Strategy("meta", model=_logistic, threshold=0.55),
               bp.Strategy("meta_sized", model=_logistic, threshold=0.5,
                           sizing=lambda p, rows: 10 * betsizing.size_vs_price(p, rows["cost"].to_numpy()))]
    results = {c.name: bp.run_cpcv(ev, c, n_groups=6, k_test=2, embargo=86400) for c in configs}
    assert all(r["window_pnl"].shape == (5, ev["start"].nunique()) for r in results.values())
    per_dollar = {k: np.mean([bp.path_bets(r, j)["pnl"].sum() / bp.path_bets(r, j)["dollars"].sum() for j in range(5)])
                  for k, r in results.items()}
    assert per_dollar["meta"] > per_dollar["primary"]
    st = bp.statistics(results["meta_sized"], bankroll=1000.0)
    for col in ("sharpe", "hhi_plus", "hhi_minus", "hhi_time", "dd_max", "tuw_max_days", "bets", "holding_seconds",
                "corr_underlying", "implementation_shortfall", "risk_precision", "risk_precision_needed",
                "risk_prob_failure", "twrr_annual"):
        assert np.isfinite(st["paths"][col].astype(float)).all(), col
    assert st["summary"].shape[1] == 5
    pbo = bp.pbo_cscv(results, n_blocks=4)
    assert pbo["configurations"] == ["primary", "meta", "meta_sized"] and 0 <= pbo["pbo"] <= 1

    reg = tmp_path / "trial_registry.json"
    reg.write_text(json.dumps({"total_selection_trials": 500}))
    assert bp.registry_trials(reg) == 500
    d = bp.deflated_sharpe(results, bp.registry_trials(reg), n_sims=2000)
    assert d["n_trials"] == 503 and len(d["paths"]) == 5
    leg = bp.window_legs(bp.path_bets(results[d["selected"]], 0))
    h = evaluation.luck_hurdle({c: leg[c].to_numpy(float) for c in bp.LEG}, 503, 2000, 0)
    assert d["paths"]["sr_star"][0] == pytest.approx(h["sr_star"]) and d["paths"]["dsr"][0] == pytest.approx(h["dsr"])
    # docs/afml_sections/ch14.md F14.7: three configurations are too few for a V[SR], so the Gaussian column is NaN
    assert bp.GAUSSIAN_MIN_CONFIGS > 3 and d["paths"]["sr_star_gaussian"].isna().all()

    more = dict(results)
    for k, thr in enumerate((0.52, 0.58, 0.6)):
        more[f"meta_{thr}"] = bp.run_cpcv(ev, bp.Strategy(f"meta_{thr}", model=_logistic, threshold=thr),
                                          n_groups=6, k_test=2, embargo=86400)
    dm = bp.deflated_sharpe(more, bp.registry_trials(reg), selected=d["selected"], n_sims=2000)
    leg = bp.window_legs(bp.path_bets(more[dm["selected"]], 0))
    r = evaluation.leg_returns({c: leg[c].to_numpy(float) for c in bp.LEG})
    s = metrics.summary(r)
    z = []
    for res in more.values():
        rr = evaluation.leg_returns({c: bp.window_legs(bp.path_bets(res, 0))[c].to_numpy(float) for c in bp.LEG})
        z.append(metrics.sharpe(rr) * np.sqrt(len(rr) - 1))
    assert dm["paths"]["gaussian_trials"][0] == len(z) == 6              # N from the same set V is measured on
    assert dm["n_trials"] == 506                                         # the primary column still counts the registry
    star = metrics.expected_max_sharpe(len(z), np.var(z, ddof=1) / (s["n"] - 1))
    assert dm["paths"]["sr_star_gaussian"][0] == pytest.approx(star)
    assert star < metrics.expected_max_sharpe(506, np.var(z, ddof=1) / (s["n"] - 1))
    assert dm["paths"]["dsr_gaussian"][0] == pytest.approx(metrics.probabilistic_sharpe_ratio(s["sharpe"], s["n"], s["skew"], s["kurtosis"], star))


@pytest.mark.case
def test_rows_of_a_window_ending_after_the_confirmation_start_are_refused():
    """Code review afml.md, backtest_pipeline finding 1: as models.Dataset.from_rows refuses them."""
    ev = _events(days=4)
    shift = evaluation.CONFIRM_START - 300 - ev["start"].max()                  # the last window ends at the start
    at = ev.assign(start=ev["start"] + shift, t0=ev["t0"] + shift, t1=ev["t1"] + shift)
    assert bp.run_cpcv(at, bp.Strategy("all"), n_groups=2, k_test=1, embargo=0)["windows"].max() + 300 == evaluation.CONFIRM_START
    late = at.assign(start=at["start"] + 60, t0=at["t0"] + 60, t1=at["t1"] + 60)
    with pytest.raises(ValueError, match="confirmation"):
        bp.run_cpcv(late, bp.Strategy("all"), n_groups=2, k_test=1, embargo=0)


@pytest.mark.case
def test_statistics_report_the_variance_of_the_mean_over_correlated_paths():
    """12.5 (docs/afml_sections/ch12.md F12.2): the paths reuse the same C(N, k) split forecasts, so they are
    strongly correlated and the mean path Sharpe is far less precise than the spread across paths suggests."""
    # A trained model, so each split's forecasts differ. A rule that takes every event gives five identical paths,
    # a variance of ~1e-33, and inequalities below that only compare rounding noise (they flipped on Linux).
    ev = _events(days=12, windows_per_day=8, per_window=2, seed=0, signal=0.2)
    res = bp.run_cpcv(ev, bp.Strategy("m", model=_logistic, threshold=0.55, sizing=bp.fixed_stake(10.0)),
                      n_groups=6, k_test=2, embargo=0)
    st = bp.statistics(res)
    pm = st["path_mean"]
    assert pm["variance"] > 1e-4                                  # the paths really differ
    assert pm["statistic"] == "sharpe" and pm["paths"] == len(res["paths"]) == 5
    assert pm["mean"] == pytest.approx(st["paths"]["sharpe"].mean())
    assert pm["variance"] == pytest.approx(st["paths"]["sharpe"].var(ddof=1))
    assert pm["rho"] == pytest.approx(cpcv.mean_correlation(res["window_pnl"]))
    assert pm["rho"] > 0.5                                        # the paths share their per-split forecasts
    assert pm["variance_of_mean"] == pytest.approx(pm["variance"] * (1 + 4 * pm["rho"]) / 5)
    assert pm["variance_if_independent"] < pm["variance_of_mean"] < pm["variance_if_identical"]
    assert 1 <= pm["effective_paths"] < 5                         # never the 5 independent paths a reader assumes
    assert pm["sd_of_mean"] > np.sqrt(pm["variance_if_independent"])


@pytest.mark.case
def test_the_expected_maximum_sharpe_stays_under_the_books_bound():
    """12.5: E[max of I standard normals] <= sqrt(2 log I), the bound the section states beside the formula."""
    for i in (10, 100, 1000, 10_000):
        assert 0 < metrics.expected_max_sharpe(i, 1.0) < np.sqrt(2 * np.log(i))
    assert metrics.expected_max_sharpe(100, 1.0) == pytest.approx(2.53, abs=0.01)
    assert np.sqrt(2 * np.log(100)) == pytest.approx(3.03, abs=0.01)
    # E[max{y_i}] = E[max{x_i}] sigma[y_i]: the scale factors straight out
    assert metrics.expected_max_sharpe(100, 0.25) == pytest.approx(0.5 * metrics.expected_max_sharpe(100, 1.0))
    assert metrics.expected_max_sharpe(1, 1.0) == 0.0             # one trial is no selection


@pytest.mark.property
def test_the_open_position_sweep_is_the_definition_it_replaces():
    """14.3's maximum dollar position is the largest value open at one instant, [t0, t1). `aum` reads it off two
    prefix sums instead of scanning every purchase at every edge (4.4 s to 0.01 s on a real path); overlapping,
    nested, touching and zero-value spans must still give the naive answer exactly."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        t0 = rng.integers(0, 60, n).astype(float)
        t1 = t0 + rng.integers(1, 20, n)
        filled = np.where(rng.random(n) < 0.15, 0.0, rng.random(n) * 10)
        price = rng.uniform(0.01, 0.99, n)
        bets = pd.DataFrame({"t0": t0, "t1": t1, "filled": filled, "price": price})
        got = bp.aum(bets, float(t0.min()), float(t1.max() + 1))
        b = bets[bets["filled"] > 0]
        if not len(b):
            assert np.isnan(got["max_position_dollars"])
            continue
        value = (b["filled"] * b["price"]).to_numpy(float)
        a0, a1 = b["t0"].to_numpy(float), b["t1"].to_numpy(float)
        edges = np.unique(np.concatenate([a0, a1]))
        naive = max(value[(a0 <= x) & (x < a1)].sum() for x in edges[:-1]) if len(edges) > 1 else 0.0
        assert got["max_position_dollars"] == pytest.approx(naive)
