import math

import numpy as np
import pytest
from scipy.stats import norm

from pmlab.afml import betsizing as bs
from pmlab.strategies import kelly_fraction


@pytest.mark.case
def test_size_from_probability_hand_values():
    z = 0.1 / math.sqrt(0.24)
    assert bs.size_from_prob(0.6) == pytest.approx(2 * norm.cdf(z) - 1)
    assert bs.size_from_prob(0.6) == pytest.approx(0.161744, abs=1e-6)
    assert bs.size_from_prob(0.5) == 0.0
    assert bs.size_from_prob(1 / 3, n_classes=3) == pytest.approx(0.0)
    assert bs.size_from_prob(0.6, side=-1) == pytest.approx(-0.161744, abs=1e-6)
    p = np.linspace(0.01, 0.99, 99)
    m = bs.size_from_prob(p)
    assert np.all(np.diff(m) > 0)
    assert m == pytest.approx(-bs.size_from_prob(1 - p))
    assert bs.size_from_prob(1.0) == 1.0


@pytest.mark.case
def test_size_against_a_price_uses_the_price_as_the_null():
    assert bs.size_vs_price(0.6, 0.5) == pytest.approx(bs.size_from_prob(0.6))
    assert bs.size_vs_price(0.55, 0.55) == 0.0
    assert bs.size_vs_price(0.5, 0.55) == 0.0
    assert bs.size_vs_price(0.9, 0.8) == pytest.approx(2 * norm.cdf(0.1 / math.sqrt(0.09)) - 1)


@pytest.mark.property
def test_kelly_matches_the_strategy_module():
    rng = np.random.default_rng(0)
    for p, a in rng.uniform(0.01, 0.99, (200, 2)):
        assert bs.kelly(p, a) == pytest.approx(kelly_fraction(p, a))


@pytest.mark.case
def test_average_of_active_bets():
    times, avg = bs.avg_active_sizes([0, 5, 20], [10, 15, np.nan], [1.0, -0.5, 0.2])
    assert times.tolist() == [0, 5, 10, 15, 20]
    assert avg.tolist() == pytest.approx([1.0, 0.25, -0.5, 0.0, 0.2])


@pytest.mark.case
def test_discretisation():
    assert bs.discrete_size([0.23, 0.26, -0.26, 1.04, -3.0], 0.1).tolist() == pytest.approx([0.2, 0.3, -0.3, 1.0, -1.0])


@pytest.mark.case
def test_dynamic_position_and_limit_price_on_the_book_setting():
    # divergence 10 should map to size 0.95; forecast 115, market 100, max position 100
    w = bs.calibrate_w(10, 0.95)
    assert w == pytest.approx(100 * (1 / 0.9025 - 1))
    assert bs.bet_size(w, 10) == pytest.approx(0.95)
    t_pos = bs.target_position(w, 115, 100, 100)
    assert t_pos == int(15 / math.sqrt(w + 225) * 100) == 97
    lp = bs.limit_price(t_pos, 0, 115, w, 100)
    manual = sum(115 - (j / 100) * math.sqrt(w / (1 - (j / 100) ** 2)) for j in range(1, 98)) / 97
    assert lp == pytest.approx(manual)
    assert 100 < lp < 115
    assert lp == pytest.approx(112.3657, abs=1e-4)          # the value the book's example prints
    # selling back from 97 to 40 prices each unit j = 96..40 (the book's abs() loop is empty here)
    down = bs.limit_price(40, 97, 115, w, 100)
    assert down == pytest.approx(np.mean([bs.inverse_price(115, w, j / 100) for j in range(96, 39, -1)]))
    assert bs.inverse_price(115, w, 0.96) < down < bs.inverse_price(115, w, 0.40)


@pytest.mark.property
def test_inverse_price_inverts_the_sigmoid():
    w = bs.calibrate_w(0.05, 0.8)
    for x in np.linspace(-0.2, 0.2, 21):
        m = float(bs.bet_size(w, x))
        assert bs.inverse_price(0.6, w, m) == pytest.approx(0.6 - x, abs=1e-12)


@pytest.mark.case
def test_hold_to_expiry_averaging_only_adds_to_a_position():
    # 10.4 with tokens held to expiry: the target is the mean signed size of the bets opened so far in the window
    inc, side = bs.hold_to_expiry_increments([1, 1], [1.0, 0.2], [1, 1])
    assert inc.tolist() == pytest.approx([1.0, 0.0]) and side.tolist() == [1, 1]      # target 0.6 < held 1.0: no sale
    inc, _ = bs.hold_to_expiry_increments([1, 1, 1], [0.2, 1.0, 1.0], [1, 1, 1])
    assert inc.tolist() == pytest.approx([0.2, 0.4, 2.2 / 3 - 0.6])                   # targets .2, .6, .733
    inc, side = bs.hold_to_expiry_increments([1, 1], [0.2, 1.0], [1, -1])
    assert inc.tolist() == pytest.approx([0.2, 0.0]) and side.tolist() == [1, 1]      # target -0.4 against a long
    inc, side = bs.hold_to_expiry_increments([1, 1, 2], [0.5, 0.5, 0.3], [1, -1, -1])
    assert inc.tolist() == pytest.approx([0.5, 0.0, 0.3]) and side.tolist() == [1, 1, -1]   # a new window starts flat


@pytest.mark.case
def test_stake_performance_sees_the_size_of_a_bet():
    # window a: one bet at full size earning 50% of its stake; window b: one losing bet; window c: no bet
    small = bs.stake_performance(["a", "b"], [1.0, 0.05], [0.5, -1.0], ["a", "b", "c"], fractions=(1.0,))
    big = bs.stake_performance(["a", "b"], [1.0, 1.0], [0.5, -1.0], ["a", "b", "c"], fractions=(1.0,))
    assert small["window_pnl"].tolist() == pytest.approx([0.5, -0.05, 0.0])
    assert small["pnl"] == pytest.approx(0.45) and small["staked"] == pytest.approx(1.05)
    assert small["pnl_per_dollar"] == pytest.approx(0.45 / 1.05)
    assert small["window_sharpe"] == pytest.approx(np.mean([0.5, -0.05, 0]) / np.std([0.5, -0.05, 0], ddof=1))
    assert small["log_growth"]["1"] == pytest.approx(np.log(1.5) + np.log(0.95))
    assert big["pnl"] == pytest.approx(-0.5) and big["log_growth"]["1"] == -np.inf and big["ruined"]["1"]


@pytest.mark.case
def test_the_books_two_strategy_example_pays_what_the_book_says():
    """10.2's opening example: two strategies right about the same 25% rally, sized differently. Marking each
    position to the next price, the one that kept cash for a strengthening signal makes 0.5 and the one that was
    forced to cut after the move against it loses 0.125 - the book's own two numbers."""
    prices = np.array([1.0, 0.5, 1.25])
    pnl = lambda m: float(np.sum(np.asarray(m)[:-1] * np.diff(prices)))
    assert pnl([0.5, 1.0, 0.0]) == pytest.approx(0.5)
    assert pnl([1.0, 0.5, 0.0]) == pytest.approx(-0.125)
    # both forecasts were right: the price ends 25% above where it started
    assert prices[-1] / prices[0] - 1 == pytest.approx(0.25)


@pytest.mark.case
def test_concurrency_and_the_budgeting_rule_reach_full_size_only_at_the_maximum():
    """10.2's second approach on three bets: long [0, 10), long [2, 6), short [4, 12)."""
    t0, t1, side = [0, 2, 4], [10, 6, 12], [1, 1, -1]
    times, c_l, c_s = bs.concurrency(t0, t1, side)
    assert times.tolist() == [0, 2, 4, 6, 10, 12]
    assert c_l.tolist() == [1, 2, 2, 1, 0, 0]
    assert c_s.tolist() == [0, 0, 1, 1, 1, 0]
    m = bs.budget_size(c_l, c_s)
    assert m.tolist() == pytest.approx([0.5, 1.0, 0.0, -0.5, -1.0, 0.0])
    assert np.abs(m).max() == 1.0                     # full size only when as many bets are concurrent as ever were
    # a side that never fires contributes nothing, and a bigger budget keeps the position below its maximum
    assert bs.budget_size(c_l, np.zeros(6)).tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5, 0.0, 0.0])
    assert np.abs(bs.budget_size(c_l, c_s, max_long=4, max_short=2)).max() == pytest.approx(0.5)


@pytest.mark.case
def test_a_bet_still_open_at_the_end_counts_as_concurrent_throughout():
    times, c_l, c_s = bs.concurrency([0, 5], [np.nan, 8], [1, -1])
    assert times.tolist() == [0, 5, 8]                # the open bet contributes no closing time
    assert c_l.tolist() == [1, 1, 1] and c_s.tolist() == [0, 1, 0]


@pytest.mark.property
def test_the_dynamic_target_falls_to_zero_as_the_price_reaches_the_forecast():
    """10.6: the target shrinks to nothing as the market price reaches the forecast (the algorithm takes its
    gains), and because the sigmoid is monotonic the breakeven limit always lies between price and forecast, so
    no loss is realised on the way. The limit sitting above the ask is therefore construction, not evidence."""
    w, f, Q = bs.calibrate_w(10, 0.95), 115.0, 100
    prices = np.arange(100.0, 115.5, 0.5)
    targets = np.array([bs.target_position(w, f, p, Q) for p in prices])
    assert targets[0] == 97 and targets[-1] == 0
    assert np.all(np.diff(targets) <= 0) and np.all(np.abs(targets) < Q)     # strictly inside the maximum position
    for p, t in zip(prices[:-1], targets[:-1]):
        assert p < bs.limit_price(int(t), 0, f, w, Q) < f
