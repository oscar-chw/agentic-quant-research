import numpy as np
import pytest

from pmlab.labels import (any_per_second, average_uniqueness, binary_pnl, cusum_filter, fixed_horizon, meta_label,
                          path_return, purged_kfold, run_length, triple_barrier)


@pytest.mark.case
def test_cusum_filter_triggers_and_resets_only_the_side_that_fired():
    r = [0.5, 0.6, -0.2, -1.2, 0.1]
    # S+: .5, 1.1 (fires, reset) 0, 0, .1   S-: 0, 0, -.2, -1.4 (fires, reset), 0
    assert cusum_filter(r, 1.0).tolist() == [0, 1, 0, -1, 0]
    # strict inequality: exactly h does not fire
    assert cusum_filter([0.5, 0.5], 1.0).tolist() == [0, 0]
    # without the reset the second rise would fire again at once
    assert cusum_filter([1.1, 0.1, 0.1], 1.0).tolist() == [1, 0, 0]


@pytest.mark.case
def test_cusum_filter_accumulates_through_missing_thresholds_but_cannot_fire_there():
    r = [0.6, 0.6, 0.1]
    assert cusum_filter(r, [1.0, np.nan, 1.0]).tolist() == [0, 0, 1]
    assert cusum_filter([0.6, np.nan, 0.6], 1.0).tolist() == [0, 0, 1]


@pytest.mark.case
def test_run_length_counts_equal_signs_and_restarts_on_zero_or_flip():
    assert run_length([1, 2, -1, -3, -0.5, 0, 4]).tolist() == [1, 2, 1, 2, 3, 0, 1]


@pytest.mark.case
def test_triple_barrier_first_touch_vertical_and_truncation():
    path = np.array([0.0, 0.0, 0.2, 0.6, -1.0, -1.0, 0.0, 0.0])
    # entry 1: ref path[0] = 0; path[1:4] = 0, .2, .6 -> +0.5 touched at second 3
    # entry 3: ref path[2] = .2; path[3:6] = .6, -1, -1 -> +0.4 >= .5? no; -1.2 <= -.5 at second 4
    # entry 6: ref path[5] = -1; path[6:8] = 0, 0 (truncated) -> +1 >= 2? no -> vertical at second 7
    label, at = triple_barrier(path, [1, 3, 6], [0.5, 0.5, 2.0], horizon=3)
    assert label.tolist() == [1.0, -1.0, 0.0]
    assert at.tolist() == [3, 4, 7]


@pytest.mark.case
def test_triple_barrier_is_nan_without_a_reference_or_a_path():
    path = np.array([np.nan, 0.0, 1.0])
    label, _ = triple_barrier(path, [1, 3], 0.5, horizon=2)
    assert np.isnan(label).tolist() == [True, True]
    with pytest.raises(ValueError):
        triple_barrier(path, [0], 0.5, horizon=2)


@pytest.mark.property
def test_triple_barrier_matches_a_brute_force_loop():
    rng = np.random.default_rng(3)
    for _ in range(50):
        path = np.cumsum(rng.normal(0, 1, 40))
        entry = rng.integers(1, 40, 25)
        width = rng.uniform(0.2, 3.0, 25)
        H = int(rng.integers(1, 15))
        label, at = triple_barrier(path, entry, width, H)
        for i in range(25):
            ref, want, when = path[entry[i] - 1], 0.0, min(entry[i] + H, len(path)) - 1
            for u in range(entry[i], min(entry[i] + H, len(path))):
                if path[u] - ref >= width[i]:
                    want, when = 1.0, u
                    break
                if path[u] - ref <= -width[i]:
                    want, when = -1.0, u
                    break
            assert (label[i], at[i]) == (want, when)


@pytest.mark.case
def test_fixed_horizon_sign_and_truncation_at_the_path_end():
    path = np.array([0.0, 1.0, 1.0, 0.5, 2.0])
    sign, end = fixed_horizon(path, [1, 2, 3, 4], horizon=2)
    # refs path[0,1,2,3] = 0, 1, 1, .5; ends path[2,3,4,4] = 1, .5, 2, 2
    assert sign.tolist() == [1.0, -1.0, 1.0, 1.0]
    assert end.tolist() == [2, 3, 4, 4]
    sign, _ = fixed_horizon(np.array([0.0, 0.0, 0.0]), [1], horizon=2)
    assert sign.tolist() == [0.0]


@pytest.mark.case
def test_average_uniqueness_of_overlapping_and_disjoint_spans():
    # [0,4) and [2,6) share seconds 2,3: 1/c = 1,1,.5,.5 and .5,.5,1,1 -> .75 each; [10,12) alone -> 1
    u = average_uniqueness([0, 2, 10], [4, 6, 12])
    assert u.tolist() == pytest.approx([0.75, 0.75, 1.0])
    with pytest.raises(ValueError):
        average_uniqueness([5], [5])


@pytest.mark.property
def test_purged_kfold_never_trains_on_labels_that_overlap_or_follow_the_test_fold():
    rng = np.random.default_rng(7)
    t0 = np.sort(rng.integers(0, 50_000, 600))
    t1 = t0 + rng.integers(1, 400, 600)
    embargo = 250
    folds = purged_kfold(t0, t1, n_splits=5, embargo=embargo)
    assert len(folds) == 5
    assert np.array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(600))
    purged_any = False
    for train, test in folds:
        lo, hi = t0[test].min(), t1[test].max()
        assert not np.intersect1d(train, test).size
        assert not np.any((t0[train] < hi) & (t1[train] > lo))
        assert not np.any((t0[train] >= hi) & (t0[train] < hi + embargo))
        purged_any |= len(train) + len(test) < 600
        assert len(train) > 0
    assert purged_any


@pytest.mark.case
def test_binary_pnl_charges_the_taker_fee_at_the_entry_price():
    assert binary_pnl(1, 0.5) == pytest.approx(1 - 0.5 - 0.07 * 0.25)
    assert binary_pnl([0, 1], [0.9, 0.2]).tolist() == pytest.approx([-0.9 - 0.07 * 0.09, 0.8 - 0.07 * 0.16])


@pytest.mark.case
def test_path_return_is_measured_from_the_reference_to_the_deciding_second():
    path = np.array([0.0, 0.0, 0.2, 0.6, -1.0, -1.0, 0.0, 0.0])
    label, at = triple_barrier(path, [1, 3, 6], [0.5, 0.5, 2.0], horizon=3)
    # touches: path[3] - path[0] = .6, path[4] - path[2] = -1.2; vertical (truncated): path[7] - path[5] = +1
    assert path_return(path, [1, 3, 6], at).tolist() == pytest.approx([0.6, -1.2, 1.0])
    assert label.tolist() == [1.0, -1.0, 0.0]
    _, end = fixed_horizon(path, [1, 2], horizon=2)
    assert path_return(path, [1, 2], end).tolist() == pytest.approx([0.2, 0.6])
    # a missing reference, a missing end, or an end before the entry has no return
    gaps = np.array([np.nan, 1.0, np.nan, 2.0])
    assert np.isnan(path_return(gaps, [1, 2, 2, 3], [3, 2, 1, 4])).tolist() == [True, True, True, True]


@pytest.mark.case
def test_meta_label_keeps_vertical_barrier_events_and_labels_them_by_the_return():
    # Snippet 3.7: bin = 1 iff side * return > 0; a zero return is a 0, nothing with a return is dropped
    ret = np.array([0.6, -1.2, 1.0, 0.0, np.nan, 0.3])
    side = np.array([1, 1, -1, 1, 1, 0])
    y, rows = meta_label(ret, side)
    assert rows.tolist() == [True, True, True, True, False, False]
    assert y[rows].tolist() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.case
def test_any_per_second_flags_the_kept_bar_when_an_earlier_bar_in_its_second_fired():
    flags = [False, True, False, False, False, True]
    seconds = [10, 11, 11, 12, 12, 13]
    assert any_per_second(flags, seconds).tolist() == [False, True, True, False, False, True]
    assert any_per_second([], []).tolist() == []
