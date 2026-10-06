from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from pmlab import labels
from pmlab.afml import sampling as sp

# the book's three-label example (4.5.3): units 0-2, 2-3 and 4-5 inclusive
T0, T1 = np.array([0, 2, 4]), np.array([3, 4, 6])


@pytest.mark.case
def test_indicator_matrix_and_uniqueness_of_the_book_example():
    ind = sp.indicator_matrix(T0, T1)
    assert ind.T.tolist() == [[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    assert sp.uniqueness_from_matrix(ind).tolist() == pytest.approx([5 / 6, 3 / 4, 1.0])


@pytest.mark.case
def test_sequential_draw_probabilities_of_the_book_example():
    assert sp.next_draw_probs(T0, T1, []).tolist() == pytest.approx([1 / 3] * 3)
    # after drawing label 2 (index 1): uniqueness 5/6, 1/2, 1 -> 5/14, 3/14, 6/14. The probabilities are
    # normalised, so they cannot tell the three uniquenesses from any common rescaling of them: pin the levels.
    ind = sp.indicator_matrix(T0, T1)
    assert [sp.uniqueness_from_matrix(ind[:, [1, j]])[-1] for j in range(3)] == pytest.approx([5 / 6, 1 / 2, 1.0])
    got = sp.next_draw_probs(T0, T1, [1])
    assert got.tolist() == pytest.approx([float(Fraction(5, 14)), float(Fraction(3, 14)), float(Fraction(6, 14))])
    # after labels 2 and 3: 5/6, 1/2, 1/2 -> 5/11, 3/11, 3/11
    assert sp.next_draw_probs(T0, T1, [1, 2]).tolist() == pytest.approx([5 / 11, 3 / 11, 3 / 11])


def _naive_probs(t0, t1, drawn):
    """The book's algorithm literally: uniqueness of the last column of indM[drawn + [i]]."""
    ind = sp.indicator_matrix(t0, t1)
    out = []
    for i in range(len(t0)):
        out.append(sp.uniqueness_from_matrix(ind[:, list(drawn) + [i]])[-1])
    out = np.array(out)
    return out / out.sum()


@pytest.mark.property
def test_sequential_bootstrap_matches_the_naive_algorithm_step_by_step():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t0, t1 = sp.random_spans(12, 40, 8, rng)
        drawn, probs = sp.seq_bootstrap(t0, t1, size=12, rng=rng, return_probs=True)
        for k in range(12):
            assert probs[k] == pytest.approx(_naive_probs(t0, t1, drawn[:k]), abs=1e-12)


@pytest.mark.property
def test_uniqueness_from_matrix_agrees_with_pmlab_labels():
    rng = np.random.default_rng(1)
    t0, t1 = sp.random_spans(50, 200, 20, rng)
    u = sp.uniqueness_from_matrix(sp.indicator_matrix(t0, t1))
    assert u == pytest.approx(labels.average_uniqueness(t0, t1))
    assert np.all(u > 0) and np.all(u <= 1.0)          # 4.4: an average of 1/c_t, so it lives in (0, 1]
    assert u.min() < 0.5 and np.any(u < 1.0)           # and overlap really does pull it below 1 here
    disjoint = sp.uniqueness_from_matrix(sp.indicator_matrix([0, 5, 11], [4, 9, 15]))
    assert disjoint.tolist() == pytest.approx([1.0, 1.0, 1.0])     # nothing concurrent, so nothing shared
    drawn = rng.integers(0, 50, 50)
    ind = sp.indicator_matrix(t0, t1)[:, drawn]
    assert sp.bootstrap_uniqueness(t0, t1, drawn) == pytest.approx(np.nanmean(sp.uniqueness_from_matrix(ind)))


@pytest.mark.eval
def test_sequential_bootstrap_raises_uniqueness_in_the_book_monte_carlo():
    """Threshold: 10 labels on 100 bars, max holding 5, 1000 trials. The book (Figure 4.2) reports medians of
    roughly 0.6 for the standard bootstrap and 0.7 for the sequential one, and a t-test p-value that rounds to
    zero; all three are asserted here, not just the direction of the difference."""
    rng = np.random.default_rng(2)
    std, seq = [], []
    for _ in range(1000):
        t0, t1 = sp.random_spans(10, 100, 5, rng)
        std.append(sp.bootstrap_uniqueness(t0, t1, sp.standard_bootstrap(10, rng=rng)))
        seq.append(sp.bootstrap_uniqueness(t0, t1, sp.seq_bootstrap(t0, t1, rng=rng)))
    std, seq = np.array(std), np.array(seq)
    assert seq.mean() - std.mean() > 0.05
    assert np.mean(seq > std) > 0.7
    assert np.median(std) == pytest.approx(0.6, abs=0.02)          # the book's two medians, in the book's units
    assert np.median(seq) == pytest.approx(0.7, abs=0.02)
    assert stats.ttest_rel(seq, std, alternative="greater").pvalue < 1e-50


@pytest.mark.case
def test_return_attribution_weights_hand_example():
    units = np.arange(6)
    r = np.array([0.1, -0.2, 0.3, 0.05, 0.0, -0.4])
    # concurrency: units 0,1 -> 1; 2 -> 2; 3 -> 1; 4,5 -> 1
    raw = np.array([abs(0.1 - 0.2 + 0.3 / 2), abs(0.3 / 2 + 0.05), abs(0.0 - 0.4)])
    w = sp.return_attribution_weights(T0, T1, units, r)
    assert w == pytest.approx(raw * 3 / raw.sum())
    assert w.sum() == pytest.approx(3.0)


@pytest.mark.case
def test_time_decay_shapes():
    u = np.array([0.5, 0.5, 1.0, 1.0, 1.0])      # cumulative 0.5 1 2 3 4
    assert sp.time_decay(u, c=1.0).tolist() == pytest.approx([1.0] * 5)
    assert sp.time_decay(u, c=0.0).tolist() == pytest.approx([0.125, 0.25, 0.5, 0.75, 1.0])
    # c = 0.5: const 0.5, slope 0.5 / 4
    assert sp.time_decay(u, c=0.5).tolist() == pytest.approx([0.5625, 0.625, 0.75, 0.875, 1.0])
    # c = -0.5: slope 1 / (0.5 * 4), const -1: the oldest half of cumulative uniqueness is zero
    assert sp.time_decay(u, c=-0.5).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.5, 1.0])
    # ageing by an explicit order: the newest label is the one with the largest key
    w = sp.time_decay(u[::-1], order=[5, 4, 3, 2, 1], c=0.0)
    assert w.tolist() == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.125])
    # c = 1 is the only value that leaves every weight at 1, and inside the book's domain the newest label
    # always carries the most weight
    for c in (-0.9, -0.5, 0.0, 0.25, 0.75):
        d = sp.time_decay(u, c=c)
        assert np.all(np.diff(d) >= 0) and d[-1] == pytest.approx(1.0) and np.all(d >= 0)
        assert (d < 1.0).any()


@pytest.mark.case
def test_docstrings_cite_the_book_snippets_and_subsections():
    doc = sp.__doc__
    assert "indicator_matrix: 1 where a label covers a unit (Snippet 4.3, 4.5.2)" in doc
    assert "(4.5.1; Snippet 4.5 in 4.5.2)" in doc and "4.5.3" not in doc.split("standard_bootstrap")[0].replace(
        "the book's worked example (4.5.3)", "")
    assert "(4.5.1)" in sp.next_draw_probs.__doc__ and "(4.5.1)" in sp.seq_bootstrap.__doc__


@pytest.mark.case
def test_concurrency_counts_the_labels_of_the_book_example_per_unit():
    """Snippet 4.1 on the 4.5.3 example: labels cover units 0-2, 2-3 and 4-5, so two labels share unit 2."""
    assert sp.concurrency(T0, T1, np.arange(6)).tolist() == [1, 1, 2, 1, 1, 1]
    ind = sp.indicator_matrix(T0, T1)
    rng = np.random.default_rng(3)
    t0, t1 = sp.random_spans(40, 150, 12, rng)
    units = np.arange(t0.min(), t1.max())
    assert sp.concurrency(t0, t1, units).tolist() == sp.indicator_matrix(t0, t1).sum(axis=1).tolist()
    assert ind.sum(axis=1).tolist() == [1, 1, 2, 1, 1, 1]
