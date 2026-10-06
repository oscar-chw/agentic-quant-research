import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pmlab.afml import backtest_stats as bt


@pytest.mark.case
def test_bet_timing_counts_flattening_flips_and_the_last_time():
    pos = [0, 1, 1, 0, -1, 1, 1, 0, 0, 2]
    assert bt.bet_timing(np.arange(10), pos).tolist() == [3, 5, 7, 9]
    assert bt.bet_timing(np.arange(3), [0, 0, 0]).tolist() == [2]
    assert bt.bet_timing(np.arange(3), [1, -1, 0]).tolist() == [1, 2]


@pytest.mark.case
def test_holding_period_pairs_exits_with_the_average_entry_time():
    # enter 1 at t=1 and 1 more at t=2 (average entry 1.5), exit one unit at t=3 and one at t=4: (1.5 + 2.5) / 2
    assert bt.holding_period(np.arange(5), [0, 1, 2, 1, 0]) == pytest.approx(2.0)
    # 2 bought at t=1, flipped to -1 at t=3 (2 units held 2), the short closed at t=6 (1 unit held 3): (4 + 3) / 3
    assert bt.holding_period([0, 1, 3, 6], [0, 2, -1, 0]) == pytest.approx(7 / 3)
    assert np.isnan(bt.holding_period(np.arange(3), [0, 1, 1]))                 # nothing closed yet


@pytest.mark.property
def test_hhi_equals_the_moment_form_of_equations_50_and_51():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.exponential(0.01, rng.integers(3, 40))
        moment = (np.mean(r ** 2) / np.mean(r) ** 2 - 1) / (len(r) - 1)
        assert bt.hhi(r) == pytest.approx(moment)
        assert bt.hhi(-r) == pytest.approx(moment)
    assert bt.hhi([0.1] * 4) == pytest.approx(0.0)
    assert bt.hhi([3.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert np.isnan(bt.hhi([0.1, 0.2]))


@pytest.mark.case
def test_concentration_of_positive_negative_returns_and_of_bets_over_time():
    r = [0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1]
    c = bt.concentration(r, ["a", "a", "b", "b", "c", "c", "c"])
    assert c["h_pos"] == pytest.approx(0.0) and c["h_neg"] == pytest.approx(0.0625)
    assert c["h_time"] == pytest.approx(1 / 49)
    empty = bt.concentration(r, ["a", "a", "b", "b", "c", "c", "c"], all_periods=["a", "b", "c", "d"])
    assert empty["h_time"] == pytest.approx(19 / 147)


@pytest.mark.case
def test_drawdowns_and_time_under_water_hand_example():
    got = bt.drawdown_tuw(np.arange(7), [1, 2, 1, 3, 3, 2, 4])
    assert got["hwm_time"].tolist() == [1, 3]
    assert got["drawdown"] == pytest.approx([0.5, 1 / 3])
    assert got["tuw_time"].tolist() == [1] and got["time_under_water"].tolist() == [2]
    assert bt.drawdown_tuw(np.arange(7), [1, 2, 1, 3, 3, 2, 4], dollars=True)["drawdown"] == pytest.approx([1, 1])
    last = bt.drawdown_tuw(np.arange(5) * 10.0, [1, 2, 1, 3, 2.5])
    assert last["drawdown"] == pytest.approx([0.5, 1 / 6]) and last["time_under_water"].tolist() == [20]
    rising = bt.drawdown_tuw(np.arange(4), [1, 2, 3, 4])
    assert len(rising["drawdown"]) == 0 and len(rising["time_under_water"]) == 0


@pytest.mark.case
def test_capacity_is_the_largest_size_whose_impact_cost_still_reaches_the_target_sharpe():
    """14.3: capacity is the highest AUM that delivers a target risk-adjusted performance, and performance decays with
    size because execution costs rise. With a constant stake s and an impact of lambda per dollar sent, the net series
    at size m is m r - lambda (m s)^2, whose annualised Sharpe is sqrt(n) (mu - lambda m s^2) / sigma: linear in m, so
    the capacity multiple is (mu - theta sigma / sqrt(n)) / (lambda s^2) exactly."""
    r = np.array([0.4, -0.2, 0.9, -0.5, 0.3, 0.1])
    s = np.full(6, 10.0)
    lam, theta, n = 1e-4, 1.0, 52.0
    mu, sigma = r.mean(), r.std(ddof=1)
    want = (mu - theta * sigma / np.sqrt(n)) / (lam * s[0] ** 2)
    got = bt.capacity(r, s, lam, theta, n, avg_aum=250.0)
    assert got["sharpe_gross"] == pytest.approx(mu / sigma * np.sqrt(n))
    assert got["multiple"] == pytest.approx(want, rel=1e-9)
    assert got["capacity"] == pytest.approx(want * 250.0, rel=1e-9)

    net = got["multiple"] * r - lam * (got["multiple"] * s) ** 2          # at capacity the target is exactly met
    assert net.mean() / net.std(ddof=1) * np.sqrt(n) == pytest.approx(theta)
    bigger = 1.01 * got["multiple"] * r - lam * (1.01 * got["multiple"] * s) ** 2
    assert bigger.mean() / bigger.std(ddof=1) * np.sqrt(n) < theta        # performance decays with size

    assert bt.capacity(r, s, 0.0, theta, n)["capacity"] == np.inf         # an infinitely deep book has no limit
    assert bt.capacity(r, s, lam, 10.0, n)["capacity"] == 0.0             # a target the strategy never reaches
    assert bt.capacity(-r, s, lam, theta, n)["multiple"] == 0.0


@pytest.mark.case
def test_capacity_below_one_is_a_multiple_below_one_not_an_error():
    """A book already past its capacity (multiple < 1) used to raise brentq's 'f(a) and f(b) must have different signs',
    because the bracket was [0, 1] with excess(0) a 0/0 sentinel. The report calls capacity() unguarded, so it must
    return the shrink factor. Constant stake: the multiple is (mu - theta sigma / sqrt(n)) / (lambda s^2) exactly."""
    r = np.array([0.4, -0.2, 0.9, -0.5, 0.3, 0.1])
    s = np.full(6, 10.0)
    theta, n = 1.0, 52.0
    mu, sigma = r.mean(), r.std(ddof=1)
    lam = (mu - theta * sigma / np.sqrt(n)) / (0.3 * s[0] ** 2)            # puts the true multiple at 0.3
    got = bt.capacity(r, s, lam, theta, n, avg_aum=250.0)
    assert got["multiple"] == pytest.approx(0.3, rel=1e-9)
    assert got["capacity"] == pytest.approx(0.3 * 250.0, rel=1e-9)
    lam = lam * 1e6                                                        # far past capacity: a tiny but positive multiple
    assert 0.0 < bt.capacity(r, s, lam, theta, n)["multiple"] < 1e-5


@pytest.mark.case
def test_the_four_degenerate_cases_of_binary_classification():
    """Table 14.1, with 14.8's F1 as the harmonic mean of precision and recall. Observed all 1s: no TN and no FP, so
    precision is 1, accuracy equals recall and F1 = 2 recall / (1 + recall) >= recall. Observed all 0s: no TP and no
    FN, so recall and the harmonic mean are undefined. Predicted all 1s: recall is 1, accuracy equals precision and
    F1 >= precision. Predicted all 0s: precision and the harmonic mean are undefined. scikit-learn computes F1 as
    2 TP / (2 TP + FP + FN) instead, which is 0 rather than undefined in the latter two, and warns only when that
    denominator is 0 as well -- the behaviour the section describes."""
    def scores(y, pred):
        prec = precision_score(y, pred, zero_division=np.nan)
        rec = recall_score(y, pred, zero_division=np.nan)
        return accuracy_score(y, pred), prec, rec, 2 * prec * rec / (prec + rec)

    acc, prec, rec, f1 = scores([1, 1, 1, 1], [1, 1, 0, 1])              # observed all 1s: TN = FP = 0
    assert prec == 1.0 and acc == rec == 0.75
    assert f1 == pytest.approx(2 * rec / (1 + rec)) and f1 >= rec

    acc, prec, rec, f1 = scores([0, 0, 0, 0], [1, 0, 0, 0])              # observed all 0s: TP = FN = 0
    assert acc == 0.75 and prec == 0.0 and np.isnan(rec) and np.isnan(f1)
    assert f1_score([0, 0, 0, 0], [1, 0, 0, 0]) == 0.0                   # scikit-learn's 2 TP / (2 TP + FP + FN)

    acc, prec, rec, f1 = scores([1, 1, 0, 1], [1, 1, 1, 1])              # predicted all 1s: TN = FN = 0
    assert rec == 1.0 and acc == prec == 0.75
    assert f1 == pytest.approx(2 * prec / (1 + prec)) and f1 >= prec

    acc, prec, rec, f1 = scores([0, 0, 1, 0], [0, 0, 0, 0])              # predicted all 0s: TP = FP = 0
    assert acc == 0.75 and rec == 0.0 and np.isnan(prec) and np.isnan(f1)
    assert f1_score([0, 0, 1, 0], [0, 0, 0, 0]) == 0.0

    with pytest.warns(UndefinedMetricWarning):                           # nothing observed and nothing predicted
        assert f1_score([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
