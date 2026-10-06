import math

import numpy as np
import pytest
from scipy.stats import norm

from pmlab.afml import strategy_risk as sr


@pytest.mark.case
def test_symmetric_sharpe_hand_value_and_consistency():
    assert sr.sharpe_symmetric(0.55, 260) == pytest.approx(0.1 * math.sqrt(260) / (2 * math.sqrt(0.2475)))
    assert sr.sharpe_symmetric(0.5, 1000) == 0.0
    for p in (0.3, 0.52, 0.7):
        assert sr.sharpe_asymmetric(p, 52, 0.01, -0.01) == pytest.approx(sr.sharpe_symmetric(p, 52))


@pytest.mark.property
def test_implied_precision_and_frequency_invert_the_sharpe():
    rng = np.random.default_rng(0)
    for _ in range(200):
        pi_plus, pi_minus = rng.uniform(0.001, 0.1), -rng.uniform(0.001, 0.1)
        n = rng.uniform(10, 5000)
        breakeven = -pi_minus / (pi_plus - pi_minus)
        p = rng.uniform(breakeven + 0.01, 0.99) if breakeven < 0.98 else None
        if p is None:
            continue
        theta = sr.sharpe_asymmetric(p, n, pi_plus, pi_minus)
        assert sr.implied_precision(pi_minus, pi_plus, n, theta) == pytest.approx(p, rel=1e-9)
        assert sr.implied_frequency(pi_minus, pi_plus, p, theta) == pytest.approx(n, rel=1e-9)
    assert math.isnan(sr.implied_frequency(-0.02, 0.01, 0.6, 2.0))      # 0.03 * 0.6 - 0.02 < 0: no edge


@pytest.mark.eval
def test_binary_sharpe_matches_simulated_bets():
    """Threshold: the annualised Sharpe of 400,000 simulated bets is within 3% of the formula."""
    rng = np.random.default_rng(1)
    p, n, up, dn = 0.6, 1000, 0.02, -0.025
    wins = rng.random(400_000) < p
    r = np.where(wins, up, dn)
    assert r.mean() / r.std() * math.sqrt(n) == pytest.approx(sr.sharpe_asymmetric(p, n, up, dn), rel=0.03)


@pytest.mark.case
def test_probability_of_failure_snippet_and_estimation_scales():
    r = np.array([0.02] * 60 + [-0.025] * 40)          # p = 0.6
    n, target = 1000, 2.0
    p_star = sr.implied_precision(-0.025, 0.02, n, target)
    assert sr.prob_failure(r, n, target) == pytest.approx(norm.cdf(p_star, 0.6, 0.24))           # Snippet 15.5's scale
    assert sr.prob_failure(r, n, target, "snippet") == pytest.approx(norm.cdf(p_star, 0.6, 0.24))
    assert sr.prob_failure(r, n, target, "estimation_se") == pytest.approx(norm.cdf(p_star, 0.6, math.sqrt(0.24 / 100)))
    assert 0.5 < p_star < 0.6
    losing = np.array([0.02] * 50 + [-0.025] * 50)
    assert sr.prob_failure(losing, n, target, "estimation_se") > 0.5 > sr.prob_failure(r, n, target, "estimation_se")
    with pytest.raises(ValueError):
        sr.prob_failure(r, n, target, "binomial")


@pytest.mark.eval
def test_bootstrap_probability_of_failure_is_the_book_algorithm():
    """15.4.1: draw floor(n k) bets with replacement, KDE of the precision, integrate to p*. Threshold: with
    n = 250 bets a year, k = 2 years (500 draws) and 4,000 iterations, the result is within 0.02 of the normal
    N(p, p(1 - p) / 500) it approximates, and far below the snippet's scale when p* is near p."""
    r = np.array([0.02] * 60 + [-0.025] * 40)
    target = float(sr.sharpe_asymmetric(0.58, 250, 0.02, -0.025))
    p_star = sr.implied_precision(-0.025, 0.02, 250, target)
    assert p_star == pytest.approx(0.58)
    want = norm.cdf(p_star, 0.6, math.sqrt(0.24 / 500))
    got = sr.prob_failure(r, 250, target, "bootstrap", years=2.0, n_boot=4000, rng=0)
    assert got == pytest.approx(want, abs=0.02)
    assert got < sr.prob_failure(r, 250, target, "snippet") - 0.2


@pytest.mark.case
def test_two_outcome_formulas_are_reported_only_for_near_fixed_payouts():
    rng = np.random.default_rng(3)
    maker = np.array([0.007] * 990 + [-1.0] * 10)                   # a true two-point payout
    out = sr.strategy_risk(maker, 50_000, 2.0, n_boot=500, rng=0)
    assert out["two_point"] and out["payout_mismatch"] < 0.10
    assert out["sharpe_binary"] == pytest.approx(out["sharpe_sample"], rel=0.01)
    for k in ("breakeven_precision", "implied_precision", "prob_failure_snippet", "prob_failure_estimation_se",
              "prob_failure_bootstrap"):
        assert np.isfinite(out[k]), k
    price = rng.uniform(0.2, 0.8, 2000)                              # a taker buying at many prices
    taker = np.where(rng.random(2000) < price, (1 - price) / price, -1.0)
    out = sr.strategy_risk(taker, 50_000, 2.0, n_boot=500, rng=0)
    assert not out["two_point"] and out["payout_mismatch"] >= 0.10
    assert np.isfinite(out["sharpe_binary"]) and np.isfinite(out["sharpe_sample"])
    for k in ("breakeven_precision", "implied_precision", "implied_frequency", "prob_failure_snippet",
              "prob_failure_estimation_se", "prob_failure_bootstrap"):
        assert np.isnan(out[k]), k


@pytest.mark.case
def test_strategy_risk_summary():
    rng = np.random.default_rng(2)
    r = sr.mix_gaussians(0.05, -0.1, 0.05, 0.1, 0.7, 2000, rng)
    out = sr.strategy_risk(r, 250, 1.0, max_payout_mismatch=np.inf, n_boot=200, rng=0)
    pos, neg = r[r > 0], r[r <= 0]
    assert out["precision"] == pytest.approx(len(pos) / len(r))
    assert out["pi_plus"] == pytest.approx(pos.mean()) and out["pi_minus"] == pytest.approx(neg.mean())
    assert out["breakeven_precision"] == pytest.approx(-neg.mean() / (pos.mean() - neg.mean()))
    assert out["implied_precision"] > out["breakeven_precision"]
    binary_sd = (pos.mean() - neg.mean()) * math.sqrt(out["precision"] * (1 - out["precision"]))
    assert out["payout_mismatch"] == pytest.approx(abs(r.std() / binary_sd - 1))
    assert out["payout_mismatch"] > 0.10                          # a Gaussian mixture is not a two-point payout
    assert not sr.strategy_risk(r, 250, 1.0, n_boot=200, rng=0)["two_point"]


@pytest.mark.case
def test_implied_frequency_rejects_the_extraneous_root_as_snippet_15_4_does():
    # 15.4's example: +-1% payouts at p = 0.55 need 396 bets a year for an annual Sharpe of 2
    assert sr.implied_frequency(-0.01, 0.01, 0.55, 2.0) == pytest.approx(396.0)
    # the same edge cannot reach -2: squaring theta gives 396 again, which is the extraneous root
    assert math.isnan(sr.implied_frequency(-0.01, 0.01, 0.55, -2.0))
    assert math.isnan(sr.implied_frequency(-0.01, 0.01, 0.5, 2.0))                 # no edge at all
    n = sr.implied_frequency(-0.01, 0.01, 0.45, -2.0)                              # a losing rule reaches -2 at 396
    assert n == pytest.approx(396.0)
    assert sr.sharpe_asymmetric(0.45, n, 0.01, -0.01) == pytest.approx(-2.0)


@pytest.mark.case
def test_the_books_break_even_precision_of_two_thirds_and_how_little_precision_the_target_costs():
    """15.2 and 15.4's worked numbers. Symmetric payouts: (2p - 1) / (2 sqrt(p (1 - p))) is 0.1005 at p = 0.55, so an
    annual Sharpe of 2 needs 396 bets. Asymmetric (n = 260, pi- = -0.01, pi+ = 0.005): p = 0.7 gives 1.173 and
    p = 0.72 gives 2, so a 0.02 change in precision moves the Sharpe by 0.83 -- the vulnerability the chapter is
    about. The precision at which the Sharpe turns 0 is the break-even -pi- / (pi+ - pi-) = 2/3, and below it the
    Sharpe is negative, so a drop from 0.7 to 0.67 wipes out the profits."""
    assert (2 * 0.55 - 1) / (2 * math.sqrt(0.55 * 0.45)) == pytest.approx(0.1005, abs=5e-5)
    assert sr.implied_frequency(-0.01, 0.01, 0.55, 2.0) == pytest.approx(396.0)
    assert sr.sharpe_symmetric(0.55, 396) == pytest.approx(2.0, abs=5e-3)

    n, pi_minus, pi_plus = 260, -0.01, 0.005
    assert sr.sharpe_asymmetric(0.7, n, pi_plus, pi_minus) == pytest.approx(1.173, abs=5e-4)
    p_two = sr.implied_precision(pi_minus, pi_plus, n, 2.0)
    assert p_two == pytest.approx(0.72, abs=5e-3)
    assert sr.sharpe_asymmetric(p_two, n, pi_plus, pi_minus) == pytest.approx(2.0)

    breakeven = -pi_minus / (pi_plus - pi_minus)
    assert breakeven == pytest.approx(2 / 3)
    assert sr.implied_precision(pi_minus, pi_plus, n, 0.0) == pytest.approx(breakeven)
    assert sr.sharpe_asymmetric(breakeven, n, pi_plus, pi_minus) == pytest.approx(0.0, abs=1e-12)
    assert sr.sharpe_asymmetric(0.67, n, pi_plus, pi_minus) > 0 > sr.sharpe_asymmetric(0.66, n, pi_plus, pi_minus)
    assert sr.sharpe_asymmetric(0.67, n, pi_plus, pi_minus) < 0.15             # 0.7 -> 0.67 wipes out the profits


@pytest.mark.property
def test_figures_15_2_and_15_3_read_the_implied_precision_and_frequency_in_both_directions():
    """15.3's two heat-maps at their own parameters (pi+ = 0.1, theta* = 1.5). Figure 15.2: for a given frequency a
    more negative losing payout needs a higher precision, and for a given losing payout a smaller frequency needs a
    higher precision. Figure 15.3: for a given precision a more negative losing payout needs a higher frequency, and
    for a given losing payout a smaller precision needs a higher frequency. A target that cannot be reached at all
    (not a number) is the end of the same direction, so the monotonicity is asserted over the reachable cells."""
    pi_plus, theta = 0.1, 1.5
    losses = (-0.05, -0.1, -0.2, -0.4, -0.8)
    frequencies = (1000, 260, 100, 52)
    precisions = (0.9, 0.7, 0.5, 0.3)

    rows = [[sr.implied_precision(pm, pi_plus, n, theta) for pm in losses] for n in frequencies]
    for n, row in zip(frequencies, rows):
        assert row == sorted(row) and len(set(row)) == len(row), n          # deeper losses cost precision
    for col in zip(*rows):
        assert list(col) == sorted(col) and len(set(col)) == len(col)       # fewer bets cost precision
    assert rows[-1][0] < 0.5 < rows[-1][-1]                                 # the map is not flat: 0.44 to 0.94

    grid = [[sr.implied_frequency(pm, pi_plus, p, theta) for pm in losses] for p in precisions]
    for p, row in zip(precisions, grid):
        live = [x for x in row if not math.isnan(x)]
        assert live == sorted(live) and len(set(live)) == len(live), p      # deeper losses cost bets
        assert all(math.isnan(x) for x in row[len(live):])                  # past the last reachable one
    for col in zip(*grid):
        live = [x for x in col if not math.isnan(x)]
        assert live == sorted(live) and len(set(live)) == len(live)         # less precision costs bets
    assert grid[0][0] < 1.0 and math.isnan(grid[-1][0])                     # 0.63 bets a year against unreachable
