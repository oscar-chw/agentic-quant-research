import math
from statistics import NormalDist

import numpy as np
import pytest

from pmlab import metrics


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@pytest.mark.case
def test_sharpe_is_per_period_with_ddof_1_and_nan_when_undefined():
    assert metrics.sharpe([0.01, 0.03, -0.01, np.nan]) == pytest.approx(0.5)  # mean .01, std .02
    assert math.isnan(metrics.sharpe([0.1, np.nan, np.inf]))
    assert math.isnan(metrics.sharpe([0.1, 0.1, 0.1]))


@pytest.mark.case
def test_log_growth_is_mean_log_wealth_ratio_and_ruin_is_minus_inf():
    assert metrics.log_growth([0.1, -0.1]) == pytest.approx((math.log(1.1) + math.log(0.9)) / 2,
                                                            rel=1e-12)
    assert metrics.log_growth([0.1, -1.0]) == -math.inf
    assert metrics.log_growth([0.5, -1.5]) == -math.inf


@pytest.mark.case
def test_max_drawdown_is_positive_fraction_of_peak():
    assert metrics.max_drawdown([1, 2, 1, 3]) == 0.5
    assert metrics.max_drawdown([1, 2, 3]) == 0.0


@pytest.mark.case
def test_psr_hand_value():
    sr, n = 0.1, 253
    expected = phi(sr * math.sqrt(n - 1) / math.sqrt(1 - 0 * sr + (3 - 1) / 4 * sr**2))
    assert metrics.probabilistic_sharpe_ratio(sr, n, 0.0, 3.0) == pytest.approx(expected, abs=1e-9)


@pytest.mark.case
def test_psr_rises_with_n_and_falls_with_negative_skew_and_fat_tails():
    psr = metrics.probabilistic_sharpe_ratio
    base = psr(0.1, 253, 0.0, 3.0)
    assert psr(0.1, 1000, 0.0, 3.0) > base > psr(0.1, 60, 0.0, 3.0)
    assert psr(0.1, 253, -1.0, 3.0) < base < psr(0.1, 253, 1.0, 3.0)
    assert psr(0.1, 253, 0.0, 10.0) < base


@pytest.mark.case
def test_expected_max_sharpe_hand_value():
    n, var, g, inv = 100, 0.01, 0.5772156649015329, NormalDist().inv_cdf
    expected = math.sqrt(var) * ((1 - g) * inv(1 - 1 / n) + g * inv(1 - 1 / (n * math.e)))
    assert metrics.expected_max_sharpe(n, var) == pytest.approx(expected, abs=1e-9)


@pytest.mark.eval
def test_expected_max_sharpe_within_3pct_of_monte_carlo():
    """Threshold: |MC mean of max - formula| / formula < 3% (N=100, var=0.01, 20000 reps)."""
    n_trials, var = 100, 0.01
    draws = np.random.default_rng(7).normal(0, math.sqrt(var), size=(20000, n_trials))
    mc = draws.max(axis=1).mean()
    formula = metrics.expected_max_sharpe(n_trials, var)
    assert abs(mc - formula) / formula < 0.03


@pytest.mark.property
def test_dsr_below_psr_whenever_trials_vary():
    rng, checked = np.random.default_rng(11), 0
    for _ in range(500):
        sr, n = rng.uniform(-0.3, 0.3), int(rng.integers(20, 2000))
        skew = rng.uniform(-2, 2)
        kurt = skew**2 + 1 + rng.uniform(0, 8)  # any real distribution has kurtosis >= skew^2 + 1
        trials = rng.normal(0, rng.uniform(0.01, 0.1), size=int(rng.integers(2, 200)))
        psr = metrics.probabilistic_sharpe_ratio(sr, n, skew, kurt)
        dsr = metrics.deflated_sharpe_ratio(sr, n, skew, kurt, trials)
        if 0 < psr < 1:  # both saturate to exactly 0 or 1 in float when z is extreme
            assert dsr < psr, (sr, n, skew, kurt, len(trials))
            checked += 1
    assert checked > 400  # the saturation skip must not hollow the property out


@pytest.mark.case
def test_dsr_with_one_trial_or_identical_trials_is_psr():
    psr = metrics.probabilistic_sharpe_ratio(0.1, 253, 0.0, 3.0)
    assert metrics.deflated_sharpe_ratio(0.1, 253, 0.0, 3.0, [0.1]) == psr
    assert metrics.deflated_sharpe_ratio(0.1, 253, 0.0, 3.0, [0.1, 0.1]) == psr


@pytest.mark.case
def test_summary_hand_values_and_non_excess_kurtosis():
    s = metrics.summary([1.0, 2.0, 3.0, 4.0, np.nan])
    assert s["n"] == 4 and s["mean"] == 2.5 and s["skew"] == pytest.approx(0.0)
    assert s["kurtosis"] == pytest.approx(1.64)  # m4/m2^2 = 2.5625/1.5625; excess would be -1.36
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert s["sharpe"] == metrics.sharpe([1, 2, 3, 4])
    assert s["psr"] == metrics.probabilistic_sharpe_ratio(s["sharpe"], 4, s["skew"], s["kurtosis"])
    assert s["log_growth"] == metrics.log_growth([1, 2, 3, 4])
    assert metrics.summary([0.1, -0.1, 0.2, 0.0])["hit_rate"] == 0.5  # zero is not a hit


@pytest.mark.property
def test_summary_kurtosis_of_normal_sample_is_three():
    s = metrics.summary(np.random.default_rng(3).normal(0, 1, 200_000))
    assert abs(s["kurtosis"] - 3) < 0.05



@pytest.mark.case
def test_sharpe_of_a_nearly_constant_series_is_undefined_not_huge():
    r = np.full(200, 0.0101)
    r[::7] += 1e-17
    assert np.isnan(metrics.sharpe(r))
    assert np.isfinite(metrics.sharpe(np.r_[np.full(199, 0.01), 0.02]))
