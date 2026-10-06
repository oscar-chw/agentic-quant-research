import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pmlab.afml import betsizing as bs
from pmlab.afml import sizing_pipeline as sp
from pmlab.evaluation import CONFIRM_START
from pmlab.strategies import kelly_fraction

TW_MIDNIGHT = 20_000 * 86400 - 8 * 3600          # a Taiwan (UTC+8) day starts here
S0 = TW_MIDNIGHT + 3000                           # a window start inside that day
LOOSE = dict(max_stake_frac=1.0, window_cap_frac=1.0, daily_loss_frac=None)


def rows(*events, rule="q_taker"):
    """events: (window index, second, side, price, prob, payoff) -> frame in the stage 4 interface."""
    out = []
    for k, t, side, price, prob, payoff in events:
        start = S0 + 300 * k
        out.append({"start": start, "t0": start + t, "t1": start + 300, "rule": rule, "side": side, "price": price,
                    "prob": prob, "payoff": float(payoff)})
    return pd.DataFrame(out)


def random_frame(seed, n_windows=60, start=TW_MIDNIGHT - 3000, first=0):
    """Random rows over consecutive windows across a Taiwan midnight; every window has a row at or before `first` s."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_windows):
        s = start + 300 * k
        seconds = np.concatenate([[rng.integers(0, first + 1)], rng.choice(np.arange(first + 1, 300), rng.integers(0, 7), replace=False)])
        for t in np.sort(seconds):
            out.append({"start": s, "t0": s + int(t), "t1": s + 300, "rule": "q_taker", "side": int(rng.choice([-1, 1])),
                        "price": rng.uniform(0.05, 0.95), "prob": rng.uniform(0.02, 0.98), "payoff": float(rng.random() < 0.5)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ the book's numbers

@pytest.mark.case
def test_the_books_dynamic_position_and_limit_price_example():
    # 10.6: a divergence of 10 gives size 0.95; forecast 115 against price 100 with Q = 100
    w = sp.calibrate(10, 0.95)
    assert w == pytest.approx(100 * (0.95 ** -2 - 1))
    assert sp.dynamic_target(w, 115, 100, 100) == 97
    assert sp.dynamic_target(w, 110, 100, 100) == 95
    assert sp.limit_cost(0, 97, 115, w, 100) == pytest.approx(112.3657, abs=1e-4)
    assert sp.limit_cost(97, 40, 115, w, 100) == pytest.approx(bs.limit_price(40, 97, 115, w, 100))
    assert np.isnan(sp.limit_cost(5, 5, 115, w, 100))


@pytest.mark.case
def test_the_books_power_and_sigmoid_forms():
    # figure 10.3: sgn(x)|x|^2 and x (0.1 + x^2)^-1/2
    assert sp.dynamic_size(2.0, [-0.5, 0.3, 1.0], "power") == pytest.approx([-0.25, 0.09, 1.0])
    assert sp.dynamic_size(0.1, 0.5) == pytest.approx(0.5 / math.sqrt(0.35))
    wp = sp.calibrate(0.1, 0.5, "power")
    assert wp == pytest.approx(0.30103, abs=1e-5)
    assert sp.dynamic_size(wp, 0.1, "power") == pytest.approx(0.5)
    assert sp.inverse_cost(0.7, wp, 0.5, "power") == pytest.approx(0.6)
    # limit from 0 to 3 units of Q = 10 at forecast 0.7: mean of 0.7 - (j/10)^(1/w) = 0.69214
    assert sp.limit_cost(0, 3, 0.7, wp, 10, "power") == pytest.approx(0.692145, abs=1e-6)
    for form, w in (("sigmoid", sp.calibrate(0.1, 0.95)), ("power", wp)):
        for m in (0.1, 0.5, 0.9):
            assert sp.dynamic_size(w, 0.8 - sp.inverse_cost(0.8, w, m, form), form) == pytest.approx(m)


@pytest.mark.case
def test_book_size_from_probabilities_by_the_number_of_classes():
    two = 2 * norm.cdf(0.1 / math.sqrt(0.24)) - 1
    assert two == pytest.approx(2 * norm.cdf((2 * 0.6 - 1) / (2 * math.sqrt(0.6 * 0.4))) - 1)   # the book's 2-outcome z
    assert two == pytest.approx(0.161744, abs=1e-6)
    assert sp.book_size([0.6, 0.4, 0.5], labels=(-1, 1)) == pytest.approx([two, -two, 0.0])
    # meta-labels: the predicted class 1 bets side x size; a predicted 0 bets nothing however sure
    assert sp.book_size([0.6, 0.6, 0.4, 0.1], side=[1, -1, 1, -1]) == pytest.approx([two, -two, 0.0, 0.0])
    # three classes labelled by their sizes: p~ = 0.5 against 1/3, z = 1/3
    m3 = 2 * norm.cdf(1 / 3) - 1
    assert m3 == pytest.approx(0.261117, abs=1e-6)
    P = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [1 / 3, 1 / 3, 1 / 3]]
    assert sp.book_size(P, labels=(-1, 0, 1)) == pytest.approx([-m3, m3, 0.0, 0.0])
    with pytest.raises(ValueError):
        sp.book_size(P, labels=(0, 1))


@pytest.mark.property
def test_sizes_are_monotone_in_probability_and_zero_at_one_half():
    p = np.linspace(0.01, 0.99, 99)
    m = sp.book_size(p, labels=(-1, 1))
    assert np.all(np.diff(m) > 0) and sp.book_size([0.5], labels=(-1, 1))[0] == 0.0
    meta = sp.book_size(p, side=np.ones_like(p))
    assert np.all(meta[p <= 0.5] == 0) and np.all(np.diff(meta[p >= 0.5]) > 0)
    q = np.round(np.linspace(0.30, 0.99, 70), 4)
    grid = pd.DataFrame({"start": S0, "t0": S0 + np.arange(len(q)), "t1": S0 + 300, "rule": "q_taker", "side": 1,
                         "price": 0.5, "prob": q, "payoff": 1.0})
    for spec in sp.SIZINGS:
        if spec.average:
            continue
        size, side = sp.sizes(grid, spec)
        assert np.all(np.diff(size) >= 0), spec.name
        assert size[-1] > 0, spec.name
        if spec.kind == "book" and spec.null == "classes":
            assert np.all(size[q <= 0.5] == 0) and np.all(np.diff(size[q >= 0.5]) > 0)
        if spec.kind in ("kelly", "dynamic") or spec.null == "price" or spec.edge is not None:
            assert np.all(size[q <= sp.cost_of(0.5)] == 0), spec.name     # no bet without an edge over the all-in cost


# ------------------------------------------------------------------------------------------------ averaging, discretisation

@pytest.mark.case
def test_averaging_over_the_spans_still_open_hand_example():
    t0 = [0, 10, 20, 30, 300, 310]
    t1 = [300, 25, 300, 300, 600, 600]
    m = [0.6, -0.4, 0.2, 0.0, 0.5, 0.1]
    # t=0 {0}; t=10 {0,1}; t=20 {0,1,2}; t=30 {0,2,3} (bet 1 closed at 25); t=300 {4} (spans are [t0, t1)); t=310 {4,5}
    expected = [0.6, 0.1, 0.4 / 3, 0.8 / 3, 0.5, 0.3]
    assert sp.active_average(t0, t1, m) == pytest.approx(expected)
    order = [5, 2, 0, 4, 1, 3]
    assert sp.active_average(np.take(t0, order), np.take(t1, order), np.take(m, order)) == pytest.approx(np.take(expected, order))
    assert sp.active_average([0, 5], [np.nan, 7], [1.0, 0.0]) == pytest.approx([1.0, 0.5])      # NaN end: still open
    with pytest.raises(ValueError):
        sp.active_average([0, 5], [5, 5], [1.0, 1.0])


@pytest.mark.property
def test_averaging_matches_the_book_style_average_at_every_decision():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(1, 40))
        t0 = rng.integers(0, 100, n).astype(float)
        t1 = t0 + rng.integers(1, 30, n)
        t1[rng.random(n) < 0.1] = np.nan
        m = rng.uniform(-1, 1, n)
        times, avg = bs.avg_active_sizes(t0, t1, m)
        assert sp.active_average(t0, t1, m) == pytest.approx(avg[np.searchsorted(times, t0)])


@pytest.mark.case
def test_discretisation_rounds_the_averaged_size_to_the_step():
    frame = rows((0, 10, 1, 0.5, 0.8, 1), (0, 20, 1, 0.5, 0.55, 1))
    raw = sp.book_size([0.8, 0.55], side=[1, 1])
    assert raw == pytest.approx([0.546745, 0.080056], abs=1e-6)
    size, side = sp.sizes(frame, sp.Sizing("b", "book", average=True))
    assert size == pytest.approx([0.546745, 0.313400], abs=1e-6)
    size, side = sp.sizes(frame, sp.Sizing("b", "book", average=True, step=0.2))
    # averaged first (0.5467 -> 0.6, 0.3134 -> 0.4); rounding each bet first would give (0.6, (0.6 + 0) / 2 = 0.3)
    assert size == pytest.approx([0.6, 0.4]) and side.tolist() == [1, 1]
    big = random_frame(5)
    size, _ = sp.sizes(big, sp.Sizing("b", "book", average=True, step=0.2))
    assert np.allclose(size / 0.2, np.round(size / 0.2)) and size.max() <= 1.0


# ------------------------------------------------------------------------------------------------ dollars

@pytest.mark.case
def test_dollar_pnl_of_a_binary_bet_including_the_fee():
    # $10 at ask 0.40: fee 0.07 x 0.4 x 0.6 = 0.0168 per share, all-in 0.4168, 23.99232 shares
    shares, fee, pnl = sp.bet_pnl([10.0, 10.0], [0.40, 0.40], [1, 0])
    assert shares == pytest.approx([23.992322, 23.992322], abs=1e-6)
    assert fee == pytest.approx([0.403071, 0.403071], abs=1e-6)
    assert shares * 0.40 + fee == pytest.approx([10.0, 10.0])
    assert pnl == pytest.approx([13.992322, -10.0], abs=1e-6)
    assert sp.price_from_cost(sp.cost_of([0.02, 0.4, 0.98])) == pytest.approx([0.02, 0.4, 0.98])
    # through the simulation: $50 (5% of $1,000) per window at 0.40 won, 0.40 lost, 0.90 won (all-in 0.9063)
    bets, windows = sp.simulate(rows((0, 10, 1, 0.40, 0.6, 1), (1, 10, 1, 0.40, 0.6, 0), (2, 10, -1, 0.90, 0.95, 1)),
                                sp.Sizing("fixed", "fixed"), compound=False)
    assert bets["stake"].tolist() == pytest.approx([50.0, 50.0, 50.0])
    assert windows["pnl"].tolist() == pytest.approx([69.961612, -50.0, 5.169370], abs=1e-6)
    assert windows["pnl"].tolist() == pytest.approx(bets["pnl"].tolist())
    assert windows["bankroll_after"].tolist() == pytest.approx([1069.961612, 1019.961612, 1025.130982], abs=1e-6)
    # the book's size at 0.51 is 1.6% of $50 = $0.80, under the $1 minimum order; at 0.52 it is 3.2%: $1.60
    bets, _ = sp.simulate(rows((0, 10, 1, 0.5, 0.51, 1), (1, 10, 1, 0.5, 0.52, 1)), sp.Sizing("book", "book"))
    assert bets["stake"].tolist() == pytest.approx([0.0, 1.596621], abs=1e-6)


@pytest.mark.case
def test_fixed_stake_baseline_hand_arithmetic():
    frame = rows((0, 5, 1, 0.40, 0.45, 1), (0, 10, 1, 0.40, 0.60, 1), (0, 20, 1, 0.40, 0.70, 1),
                 (0, 30, -1, 0.10, 0.90, 1), (0, 40, 1, 0.40, 0.80, 1), (0, 50, 1, 0.40, 0.80, 1))
    bets, windows = sp.simulate(frame, sp.Sizing("fixed", "fixed"), window_cap_frac=0.12)
    # 0.45: the meta-label predicts 0; $50, $50; Down against a held Up: none; $20 left under the $120 cap; nothing left
    assert bets["stake"].tolist() == pytest.approx([0, 50, 50, 0, 20, 0])
    assert bets["capped"].tolist() == [False, False, False, False, True, True]
    assert windows["stake"].tolist() == pytest.approx([120.0])
    ev = rows((0, 10, 1, 0.40, 0.45, 1), (1, 10, 1, 0.60, 0.61, 1))
    bets, _ = sp.simulate(ev, sp.Sizing("fixed_ev", "fixed", edge=0.0))
    assert bets["stake"].tolist() == pytest.approx([50.0, 0.0])            # 0.45 > 0.4168; 0.61 < 0.6168


@pytest.mark.case
def test_kelly_baseline_on_the_all_in_price_hand_arithmetic():
    frame = rows((0, 10, 1, 0.5, 0.7, 1), (0, 20, 1, 0.5, 0.7, 1), (0, 30, 1, 0.5, 0.9, 1), (1, 10, 1, 0.5, 0.7, 1))
    # all-in 0.5175; full Kelly (0.7 - 0.5175) / 0.4825 = 0.378238, (0.9 - 0.5175) / 0.4825 = 0.792746; a quarter of it
    assert kelly_fraction(0.7, 0.5175) == pytest.approx(0.378238, abs=1e-6)
    bets, windows = sp.simulate(frame, sp.Sizing("kelly", "kelly"), **LOOSE)
    # 94.56 target; the same target again buys nothing; 198.19 target tops up 103.63; window P&L 198.19 / 0.5175 - 198.19
    # = 184.78; the next window sizes on the bankroll it can see, 1184.78: 0.25 x 0.378238 x 1184.78 = 112.03
    assert bets["stake"].tolist() == pytest.approx([94.559585, 0.0, 103.626943, 112.032552], abs=1e-6)
    assert windows["pnl"].iloc[0] == pytest.approx(184.782609, abs=1e-6)
    bets, _ = sp.simulate(frame, sp.Sizing("kelly", "kelly"), compound=False, **LOOSE)
    assert bets["stake"].iloc[3] == pytest.approx(94.559585, abs=1e-6)
    # live limits, 5% of the bankroll an order and a window: $50 capped, no room left, then 5% of 1000 + 50 / 0.5175 - 50
    bets, _ = sp.simulate(frame, sp.Sizing("kelly", "kelly"))
    assert bets["stake"].tolist() == pytest.approx([50.0, 0.0, 0.0, 52.330918], abs=1e-6)
    assert bets["capped"].tolist() == [True, True, True, True]
    # a looser window cap leaves the 5% order cap: $50, the rest of the 94.56 target, $50 of the 103.63 top-up, then 5%
    # of 1000 + 144.56 / 0.5175 - 144.56
    bets, _ = sp.simulate(frame, sp.Sizing("kelly", "kelly"), window_cap_frac=0.5)
    assert bets["stake"].tolist() == pytest.approx([50.0, 44.559585, 50.0, 56.739130], abs=1e-6)


@pytest.mark.case
def test_dynamic_size_in_dollars_and_its_limit_price():
    frame = rows((0, 10, 1, 0.5, 0.8, 1))
    # x = 0.8 - 0.5175 = 0.2825; sigmoid with 10 cents -> 0.95: m = 0.99330 -> 99 units of 100 -> $49.50
    bets, _ = sp.simulate(frame, sp.Sizing("dyn", "dynamic"))
    assert bets["stake"].iloc[0] == pytest.approx(49.5)
    w = 0.1 ** 2 * (0.95 ** -2 - 1)
    all_in = np.mean([0.8 - (j / 100) * math.sqrt(w / (1 - (j / 100) ** 2)) for j in range(1, 100)])
    ask = (1.07 - math.sqrt(1.07 ** 2 - 0.28 * all_in)) / 0.14
    assert bets["limit_price"].iloc[0] == pytest.approx(ask)
    assert 0.5 < bets["limit_price"].iloc[0] < 0.8
    # power with 10 cents -> 0.95: w = ln 0.95 / ln 0.1, m = 0.2825^w = 0.97223 -> 97 units -> $48.50
    bets, _ = sp.simulate(frame, sp.Sizing("dyn", "dynamic", form="power"))
    assert bets["stake"].iloc[0] == pytest.approx(48.5)
    bets, _ = sp.simulate(rows((0, 10, 1, 0.5, 0.51, 1)), sp.Sizing("dyn", "dynamic"))
    assert bets["stake"].iloc[0] == 0.0                                                  # prob below the all-in cost


@pytest.mark.case
def test_the_day_loss_cap_uses_settlements_only_once_visible():
    day = TW_MIDNIGHT
    frame = pd.DataFrame([{"start": s, "t0": s + t, "t1": s + 300, "rule": "q_taker", "side": 1, "price": 0.5,
                           "prob": 0.9, "payoff": 0.0}
                          for s, t in ((day, 10), (day + 300, 10), (day + 600, 2), (day + 600, 10), (day + 86400, 10))])
    bets, windows = sp.simulate(frame, sp.Sizing("fixed", "fixed"), compound=False, window_cap_frac=0.2)
    # two $50 losses reach -10% of the day's $1,000 when the second settles (visible at 605 s): the bet at 602 s is
    # placed, the one at 610 s is not; the next Taiwan day trades again
    assert bets["stake"].tolist() == pytest.approx([50, 50, 50, 0, 50])
    assert windows["halted"].tolist() == [False, False, True, False]
    assert sp.summarise(bets, windows)["halted_days"] == 1


@pytest.mark.property
def test_no_look_ahead_a_bet_uses_only_what_was_known_at_its_decision():
    base = random_frame(11, n_windows=80, first=sp.SETTLE_DELAY - 1)
    kw = dict(daily_loss_frac=0.02, window_cap_frac=0.2)
    changed = halted = 0
    rng = np.random.default_rng(12)
    early = base.loc[base["t0"] - base["start"] < sp.SETTLE_DELAY, "t0"]      # before the last window's P&L is visible
    for cut in list(base["t0"].quantile([0.2, 0.5, 0.8]).round().astype(int)) + list(early.iloc[1::8]):
        later = base["t0"] > cut
        unseen = base["start"] + 300 + sp.SETTLE_DELAY > cut           # windows whose outcome is not visible at the cut
        pert = base.copy()
        pert.loc[later, "prob"] = rng.uniform(0.02, 0.98, later.sum())
        pert.loc[unseen, "payoff"] = 1.0 - pert.loc[unseen, "payoff"]
        for spec in sp.SIZINGS:
            size_a, side_a = sp.sizes(base, spec)
            size_b, side_b = sp.sizes(pert, spec)
            known = ~later.to_numpy()
            assert size_a[known] == pytest.approx(size_b[known]) and (side_a[known] == side_b[known]).all(), spec.name
            a, wa = sp.simulate(base, spec, **kw)
            b, _ = sp.simulate(pert, spec, **kw)
            k = a["t0"] <= cut
            assert a.loc[k, "stake"].tolist() == pytest.approx(b.loc[k, "stake"].tolist()), spec.name
            changed += int(not np.allclose(a.loc[~k, "stake"], b.loc[~k, "stake"]))
            halted += int(wa["halted"].any())
    assert changed > 0 and halted > 0                   # the perturbation reaches later bets and the day cap binds


# ------------------------------------------------------------------------------------------------ evaluation

@pytest.mark.case
def test_summary_statistics_hand_arithmetic():
    windows = pd.DataFrame({"start": [S0, S0 + 300, S0 + 600], "day_tw": [1, 1, 1], "stake": [50.0, 50.0, 0.0],
                            "pnl": [10.0, -20.0, 0.0], "bankroll_before": [1000.0, 1010.0, 990.0],
                            "bankroll_after": [1010.0, 990.0, 990.0], "halted": [False, False, False]})
    windows["ret"] = windows["pnl"] / windows["bankroll_before"]
    bets = pd.DataFrame({"stake": [50.0, 50.0, 0.0], "capped": [False, True, False]})
    s = sp.summarise(bets, windows)
    r = np.array([0.01, -20 / 1010, 0.0])
    assert s["pnl"] == pytest.approx(-10) and s["staked"] == pytest.approx(100) and s["return_on_stake"] == pytest.approx(-0.1)
    assert s["sharpe_window"] == pytest.approx(r.mean() / r.std(ddof=1))
    assert s["log_growth"] == pytest.approx(math.log(0.99))
    assert s["max_drawdown"] == pytest.approx(20 / 1010) and s["max_drawdown_usd"] == pytest.approx(20)
    assert (s["bets"], s["traded_windows"], s["windows"], s["capped_bets"]) == (2, 2, 3, 1)


@pytest.mark.integration
def test_synthetic_end_to_end_every_sizing_on_the_same_folds():
    rng = np.random.default_rng(7)
    out = []
    for k in range(600):
        s = TW_MIDNIGHT + 300 * k
        true = rng.uniform(0.15, 0.85)
        won = float(rng.random() < true)
        for t in np.sort(rng.choice(np.arange(5, 281), 4, replace=False)):
            price = float(np.clip(true + rng.normal(0, 0.10), 0.02, 0.98))
            for rule, prob in (("informed", np.clip(true + rng.normal(0, 0.01), 0.01, 0.99)),
                               ("no_edge", sp.cost_of(price) - 0.001)):
                out.append({"start": s, "t0": s + int(t), "t1": s + 300, "rule": rule, "side": 1, "price": price,
                            "prob": float(prob), "payoff": won, "fold": k // 200})
    res = sp.evaluate(pd.DataFrame(out))
    summary = res["summary"].set_index(["rule", "fold", "sizing"])
    assert len(summary) == 2 * 4 * len(sp.SIZINGS)
    for (rule, fold), w in res["windows"].groupby(["rule", "fold"]):
        starts = {name: tuple(g["start"]) for name, g in w.groupby("sizing")}
        assert len(set(starts.values())) == 1 and len(starts) == len(sp.SIZINGS)        # the same windows for every sizing
        assert len(next(iter(starts.values()))) == (600 if fold == "all" else 200)
    informed = summary.loc[("informed", "all")]
    for name in ("kelly", "fixed_ev", "book_price_avg", "dynamic_sigmoid", "dynamic_power"):
        assert informed.loc[name, "pnl"] > 0 and informed.loc[name, "log_growth"] > 0, name
    no_edge = summary.loc[("no_edge", "all")]
    for name in ("kelly", "fixed_ev", "book_price_avg", "dynamic_sigmoid", "dynamic_power"):
        assert no_edge.loc[name, "staked"] == 0, name                                   # prob below the all-in cost
    assert no_edge.loc["book", "staked"] > 0                                            # the book's size ignores the price
    per_fold = res["summary"][res["summary"]["fold"] != "all"].groupby(["rule", "sizing"])["windows"].sum()
    assert (per_fold == 600).all()
    dyn = res["bets"][(res["bets"]["sizing"] == "dynamic_sigmoid") & (res["bets"]["stake"] > 0)]
    assert (dyn["limit_price"] >= dyn["price"] - 1e-12).all()
    with pytest.raises(ValueError):
        sp.evaluate(pd.DataFrame(out).drop(columns="payoff"))


@pytest.mark.case
def test_rows_of_a_window_ending_after_the_confirmation_start_are_refused():
    """Code review afml.md, backtest_pipeline finding 1 (sizing_pipeline._check): as models.Dataset.from_rows."""
    s = CONFIRM_START - 300
    at = pd.DataFrame([{"start": s, "t0": s + 10, "t1": s + 300, "rule": "q_taker", "side": 1, "price": 0.5,
                        "prob": 0.6, "payoff": 1.0}])
    bets, _ = sp.simulate(at, sp.Sizing("fixed", "fixed"))
    assert bets["stake"].iloc[0] > 0
    late = at.assign(start=s + 60, t0=s + 70, t1=s + 360)
    with pytest.raises(ValueError, match="confirmation"):
        sp.simulate(late, sp.Sizing("fixed", "fixed"))
    with pytest.raises(ValueError, match="confirmation"):
        sp.evaluate(late)


# ------------------------------------------------------------------ the second reading (plan book-v2 node 16)


@pytest.mark.property
def test_the_power_form_runs_concave_to_convex_and_is_flat_at_the_inflexion_where_the_sigmoid_is_steep():
    """10.6's three advantages of the power alternative, which the form test never looked at: it reaches exactly -1
    and 1 at the ends of the range where the sigmoid only approaches them, its curvature is set by w alone, and for
    w > 1 it runs concave to convex and is almost flat around the inflexion -- the opposite of the sigmoid, which
    runs convex to concave and is at its steepest there."""
    x = np.linspace(-1.0, 1.0, 401)
    power = np.asarray(sp.dynamic_size(2.0, x, "power"), dtype=float)
    sigmoid = np.asarray(sp.dynamic_size(sp.calibrate(0.1, 0.95), x), dtype=float)
    assert power[0] == -1.0 and power[-1] == 1.0
    assert -1.0 < sigmoid[0] and sigmoid[-1] < 1.0

    left = x[1:-1] < 0
    curvature = lambda m: np.diff(m, 2)
    assert np.all(curvature(power)[left] <= 1e-12) and np.all(curvature(power)[~left] >= -1e-12)
    assert np.all(curvature(sigmoid)[left] >= -1e-12) and np.all(curvature(sigmoid)[~left] <= 1e-12)

    middle = np.abs(x) <= 0.05                        # around the inflexion
    assert np.ptp(power[middle]) < 0.01 < 0.5 < np.ptp(sigmoid[middle])

    flatter = np.asarray(sp.dynamic_size(4.0, x, "power"), dtype=float)
    assert np.ptp(flatter[middle]) < np.ptp(power[middle])         # curvature is w's alone
    assert flatter[0] == -1.0 and flatter[-1] == 1.0               # and the ends stay put
