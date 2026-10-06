import numpy as np
import pandas as pd
import pytest

from pmlab.afml import labeling as lb

T = np.arange(10.0)
P = np.array([100, 101, 103, 99, 97, 104, 104, 90, 95, 100], dtype=float)


@pytest.mark.case
def test_ewm_volatility_measures_from_the_observation_before_t_minus_the_lag():
    t = np.array([0, 1, 2, 4, 7, 8], dtype=float)
    p = np.array([10, 11, 12, 10, 13, 12], dtype=float)
    got = lb.ewm_volatility(t, p, lag=2, span=3)
    # t - 2 = -2, -1, 0, 2, 5, 6 -> the observation strictly before: none, none, none, t=1, t=4, t=4
    ret = np.array([10 / 11 - 1, 13 / 10 - 1, 12 / 10 - 1])
    want = pd.Series(ret).ewm(span=3).std().to_numpy()
    assert np.isnan(got[:3]).all()
    assert np.isnan(got[3]) and np.isnan(want[0])              # one return has no standard deviation
    assert got[4:] == pytest.approx(want[1:])


@pytest.mark.eval
def test_ewm_volatility_on_a_grid_whose_step_is_the_lag_spans_two_steps():
    """Threshold: a random walk sampled every 60 s with log returns of sd 1e-3 per step, lag 60 s, span 2000; the
    mean EWMA sd over 100,000 steps is within 5% of 1e-3 sqrt(2): the observation at exactly t - 60 is skipped, so
    each return spans two steps (the snippet on daily closes with a one-day lag does the same)."""
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 100_000)))
    vol = lb.ewm_volatility(60.0 * np.arange(len(p)), p, lag=60, span=2000)
    assert np.nanmean(vol[5000:]) == pytest.approx(1e-3 * np.sqrt(2), rel=0.05)


@pytest.mark.case
def test_vertical_barrier_is_the_first_time_after_the_horizon():
    t = np.array([0, 1, 2, 5, 9], dtype=float)
    assert np.array_equal(lb.vertical_barrier(t, [0, 2, 5, 7], 3), [5, 5, 9, np.nan], equal_nan=True)


@pytest.mark.case
def test_barrier_touches_hand_path():
    # returns from t0 = 0: +1% +3% -1% -3% +4% +4% -10% -5% 0%
    long = lb.barrier_touches(T, P, [0], [np.nan], [0.02], pt=1, sl=1)
    assert long[["sl", "pt"]].iloc[0].tolist() == [4, 2]
    short = lb.barrier_touches(T, P, [0], [np.nan], [0.02], pt=1, sl=1, side=[-1])
    assert short[["sl", "pt"]].iloc[0].tolist() == [2, 4]
    off = lb.barrier_touches(T, P, [0], [np.nan], [0.02], pt=0, sl=2)
    assert np.isnan(off["pt"].iloc[0]) and off["sl"].iloc[0] == 7            # -10% < -4%; -3% is not
    capped = lb.barrier_touches(T, P, [0], [1], [0.02], pt=1, sl=1)
    assert np.isnan(capped["pt"].iloc[0]) and np.isnan(capped["sl"].iloc[0])  # nothing inside [0, 1]
    t, p = np.arange(4.0), np.array([100, 125, 50, 130])                      # +25% -50% +30%: exact in binary
    strict = lb.barrier_touches(t, p, [0], [np.nan], [0.25], pt=1, sl=2)
    assert strict["pt"].iloc[0] == 3 and np.isnan(strict["sl"].iloc[0])        # touching exactly is not crossing


@pytest.mark.case
def test_get_events_symmetric_without_a_side_and_asymmetric_with_one():
    trgt = np.full(len(T), 0.02)
    trgt[3] = 0.001
    ev = lb.get_events(T, P, [0, 3, 5], trgt, pt_sl=[1, 5], min_ret=0.005)
    assert ev["t0"].tolist() == [0, 5]                                          # the event at 3 is below min_ret
    assert ev["t1"].tolist() == [2, 7]                                          # pt_sl[0] on both sides
    assert "side" not in ev
    meta = lb.get_events(T, P, [0, 5], trgt, pt_sl=[1, 5], side=[1, 1])
    assert meta["t1"].tolist() == [2, 7]                                        # stop at -10%: event 5 falls 13.5%
    meta = lb.get_events(T, P, [0, 5], trgt, pt_sl=[0, 1], side=[1, 1], t1=[9, 6])
    assert meta["t1"].tolist() == [4, 6]                                        # no profit taking; vertical at 6
    none = lb.get_events(T, P, [8], trgt, pt_sl=[0, 0])
    assert np.isnan(none["t1"].iloc[0])
    with pytest.raises(ValueError):
        lb.get_events(T, P, [0.5], trgt, pt_sl=1)


@pytest.mark.case
def test_get_bins_for_side_and_for_meta_labels():
    bins = lb.get_bins(T, P, [0, 2, 4, 6], [4, 3, np.nan, 7.5])
    assert bins["t0"].tolist() == [0, 2, 6]                                     # no t1: dropped
    assert bins["ret"].tolist() == pytest.approx([-0.03, 99 / 103 - 1, 95 / 104 - 1])   # 7.5 back-fills to t = 8
    assert bins["bin"].tolist() == [-1, -1, -1]
    meta = lb.get_bins(T, P, [0, 2, 0], [2, 3, 1], side=[1, -1, -1])
    assert meta["ret"].tolist() == pytest.approx([0.03, 1 - 99 / 103, -0.01])
    assert meta["bin"].tolist() == [1, 1, 0]
    flat = lb.get_bins(T, P, [5], [6], side=[1])
    assert flat["bin"].tolist() == [0]                                          # a zero return is not a win


@pytest.mark.case
def test_drop_labels_removes_rare_labels_while_three_or_more_remain():
    bins = np.array([1] * 60 + [-1] * 36 + [0] * 3 + [2] * 1)
    keep = lb.drop_labels(bins, 0.05)
    assert set(bins[keep]) == {1, -1}                                           # 2 (1%), then 0 (3%)
    assert lb.drop_labels(np.array([1] * 98 + [0] * 2), 0.05).all()             # two labels: nothing dropped
    balanced = np.array([1] * 40 + [-1] * 30 + [0] * 30)
    assert lb.drop_labels(balanced, 0.05).all()
    tie = lb.drop_labels(np.array([1] * 90 + [-1] * 5 + [0] * 5), 0.05)
    assert tie.tolist() == [True] * 90 + [False] * 5 + [True] * 5              # 5% is not above 5%; tie: -1 first


@pytest.mark.case
def test_the_fast_barrier_walk_is_bit_identical_to_the_book_s_own(rng=np.random.default_rng(20260919)):
    """barrier_touches hoisted both searches and both thresholds out of the per-event loop (plan performance, node 7).

    A touch time that moved by one tick would move a label, so the fast path is asserted equal to
    `_barrier_touches_loop` exactly — same floats, same NaNs — over paths, horizons and sides that vary, not close
    to it. Equality is the reason the speed-up was allowed to land.
    """
    for trial in range(40):
        n = int(rng.integers(30, 400))
        times = np.cumsum(rng.integers(1, 4, n)).astype(float)
        price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        k = int(rng.integers(1, 40))
        t0 = rng.choice(times, size=k, replace=True)
        horizon = float(rng.choice([0.0, 5.0, 40.0, 1e9]))
        t1 = None if horizon == 0.0 else lb.vertical_barrier(times, t0, horizon)
        target = np.where(rng.random(k) < 0.1, np.nan, rng.uniform(0.001, 0.05, k))
        side = rng.choice([-1.0, 1.0], size=k) if trial % 2 else None
        pt, sl = float(rng.choice([0.0, 1.0, 2.5])), float(rng.choice([0.0, 1.0, 3.0]))
        fast = lb.barrier_touches(times, price, t0, t1, target, pt, sl, side)
        slow = lb._barrier_touches_loop(times, price, t0, t1, target, pt, sl, side)
        for column in fast.columns:
            assert np.array_equal(fast[column].to_numpy(), slow[column].to_numpy(), equal_nan=True), (trial, column)


@pytest.mark.case
def test_the_fast_barrier_walk_handles_the_empty_and_out_of_range_cases_the_loop_does():
    empty = lb.barrier_touches(T, P, [], [], [], pt=1, sl=1)
    assert len(empty) == 0 and list(empty.columns) == ["t0", "t1", "sl", "pt"]
    past = lb.barrier_touches(T, P, [99.0], [np.nan], [0.01], pt=1, sl=1)     # t0 after the last time: no path
    assert past[["sl", "pt"]].isna().all().all()
    reference = lb._barrier_touches_loop(T, P, [99.0], [np.nan], [0.01], pt=1, sl=1)
    assert past.equals(reference)
