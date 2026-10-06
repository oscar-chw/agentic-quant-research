import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, cross_val_score

from pmlab import labels
from pmlab.afml import cv
from pmlab.afml import sampling


def _sorted_spans(n, units, max_h, seed):
    t0, t1 = sampling.random_spans(n, units, max_h, np.random.default_rng(seed))
    o = np.argsort(t0, kind="stable")
    return t0[o], t1[o]


@pytest.mark.property
def test_purging_drops_exactly_the_books_three_cases():
    """Snippet 7.1 on closed spans [a, b] = [t0, t1 - 1]: drop if a in [i, j], b in [i, j] or a <= i and j <= b."""
    rng = np.random.default_rng(0)
    for seed in range(30):
        t0, t1 = sampling.random_spans(80, 300, 25, rng)
        i0 = rng.integers(0, 280, 3)
        i1 = i0 + rng.integers(1, 20, 3)
        a, b, i, j = t0, t1 - 1, i0, i1 - 1
        book = np.ones(len(t0), bool)
        for lo, hi in zip(i, j):
            book &= ~(((lo <= a) & (a <= hi)) | ((lo <= b) & (b <= hi)) | ((a <= lo) & (hi <= b)))
        assert np.array_equal(cv.train_times(t0, t1, i0, i1), book)


@pytest.mark.case
def test_embargo_times_hand_example():
    t = np.arange(10) * 10
    assert cv.embargo_times(t, 0.2).tolist() == [20, 30, 40, 50, 60, 70, 80, 90, 90, 90]
    assert cv.embargo_times(t, 0.05).tolist() == t.tolist()


@pytest.mark.property
def test_purged_kfold_without_embargo_equals_pmlab_labels():
    for seed in range(20):
        t0, t1 = _sorted_spans(120, 500, 30, seed)
        got = list(cv.PurgedKFold(5, t0, t1).split(np.zeros(len(t0))))
        want = labels.purged_kfold(t0, t1, 5, embargo=0)
        assert len(got) == len(want) == 5
        for (tr, te), (tr2, te2) in zip(got, want):
            assert np.array_equal(np.sort(tr), tr2) and np.array_equal(te, te2)


@pytest.mark.property
def test_purged_kfold_never_trains_on_an_overlap_and_embargoes_the_rows_after_the_test_block():
    for seed in range(20):
        t0, t1 = _sorted_spans(200, 800, 40, seed)
        n, pct = len(t0), 0.03
        k = int(n * pct)
        for train, test in cv.PurgedKFold(4, t0, t1, pct_embargo=pct).split(np.zeros(n)):
            lo, hi = t0[test].min(), t1[test].max()
            assert not np.any((t0[train] < hi) & (t1[train] > lo))
            before = np.flatnonzero(t1 <= lo)
            assert np.all(np.isin(before, train))                                   # nothing extra purged before
            after = np.flatnonzero(t0 >= hi)
            assert np.array_equal(np.intersect1d(train, after), after[k:])          # the first k after are embargoed


@pytest.mark.case
def test_purged_kfold_refuses_unsorted_rows_and_works_as_an_sklearn_cv():
    with pytest.raises(ValueError):
        cv.PurgedKFold(3, [5, 1, 2], [6, 2, 3])
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(int)
    t0 = np.arange(300)
    splitter = cv.PurgedKFold(3, t0, t0 + 10)
    assert len(cross_val_score(LogisticRegression(), X, y, cv=splitter)) == 3
    gs = GridSearchCV(LogisticRegression(), {"C": [0.01, 1.0]}, cv=splitter).fit(X, y)
    assert gs.best_params_["C"] in (0.01, 1.0)


@pytest.mark.case
def test_cv_score_weights_the_fit_and_the_score_where_sklearn_weights_only_the_fit():
    rng = np.random.default_rng(2)
    n = 600
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    w = rng.uniform(0.05, 3.0, n)
    t0 = np.arange(n)
    folds = cv.PurgedKFold(3, t0, t0 + 5, pct_embargo=0.01)
    got = cv.cv_score(LogisticRegression(), X, y, sample_weight=w, cv=folds)
    want = []
    for tr, te in folds.split(X):
        m = LogisticRegression().fit(X[tr], y[tr], sample_weight=w[tr])
        want.append(-log_loss(y[te], m.predict_proba(X[te]), sample_weight=w[te], labels=m.classes_))
    assert got == pytest.approx(want)
    unweighted_score = []
    for tr, te in folds.split(X):
        m = LogisticRegression().fit(X[tr], y[tr], sample_weight=w[tr])
        unweighted_score.append(-log_loss(y[te], m.predict_proba(X[te])))
    assert np.max(np.abs(np.array(unweighted_score) - got)) > 1e-3                # weighting the score matters
    acc = cv.cv_score(LogisticRegression(), X, y, sample_weight=w, scoring="accuracy", cv=list(folds.split(X)))
    assert np.all((acc > 0.6) & (acc <= 1))
    with pytest.raises(ValueError):
        cv.cv_score(LogisticRegression(), X, y, scoring="roc_auc", cv=folds)
    with pytest.raises(ValueError):
        cv.cv_score(LogisticRegression(), X, y)


@pytest.mark.case
def test_a_test_fold_missing_one_class_is_scored_with_the_models_classes_not_the_folds():
    """7.5's first work-around (scikit-learn issue 6231; docs/afml_sections/ch07.md F7.2): log loss over a test fold
    whose labels hold one class only must be told the fitted model's classes, or sklearn reads the probability columns
    against the fold's single class and returns a different number (or raises)."""
    rng = np.random.default_rng(3)
    n = 120
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)
    y[-30:] = 1                                        # the last fold holds class 1 only
    folds = [(np.arange(0, n - 30), np.arange(n - 30, n))]
    model = LogisticRegression(max_iter=500)
    got = cv.cv_score(model, X, y, cv=folds)
    fitted = LogisticRegression(max_iter=500).fit(X[folds[0][0]], y[folds[0][0]])
    prob = fitted.predict_proba(X[folds[0][1]])
    assert list(fitted.classes_) == [0, 1]
    assert got[0] == pytest.approx(-log_loss(y[folds[0][1]], prob, labels=[0, 1]))
    with pytest.raises(ValueError):                    # what the book's work-around avoids
        log_loss(y[folds[0][1]], prob, labels=np.unique(y[folds[0][1]]))
    assert np.isfinite(got[0]) and got[0] < 0


@pytest.fixture(scope="module")
def leaky():
    """7.3's set-up: a serially correlated feature that is irrelevant, and labels formed on overlapping windows.
    X is a three-column AR(1) with rho = 0.98, so X_t is close to X_{t+1}; the label of row i is whether the sum of
    60 white-noise draws starting at i is above the sample median, drawn independently of X, so Y_t is close to
    Y_{t+1}, the base rate is exactly 1/2 and nothing in X predicts Y. `signal` is the window sum itself, a feature
    that genuinely predicts, as the control. Rows are in time order with spans [i, i + 60)."""
    from scipy.signal import lfilter
    n, h, rho = 1500, 60, 0.98
    rng = np.random.default_rng(0)
    e = rng.normal(size=n + h)
    cumulative = np.concatenate([[0.0], np.cumsum(e)])
    window = cumulative[h:h + n] - cumulative[:n]
    y = (window > np.median(window)).astype(int)
    X = lfilter([1.0], [1.0, -rho], rng.normal(size=(n, 3)), axis=0)
    signal = np.column_stack([window + rng.normal(0, 0.5, n)])
    t0 = np.arange(n, dtype=float)
    return X, y, signal, t0, t0 + h


def _nearest_neighbour_accuracy(X, y, folds):
    from sklearn.neighbors import KNeighborsClassifier
    scores = [float(np.mean(KNeighborsClassifier(n_neighbors=1).fit(X[tr], y[tr]).predict(X[te]) == y[te]))
              for tr, te in folds]
    return float(np.mean(scores))


@pytest.mark.eval
def test_an_irrelevant_feature_beats_chance_on_shuffled_folds_and_stops_once_the_folds_are_purged_and_embargoed(
        leaky):
    """7.2-7.4 on the leaky set-up above: a 1-nearest-neighbour classifier can only be reading the training row next
    to each test row, because X carries nothing about Y. Threshold: shuffled 10-fold accuracy above 0.6, well over
    the book's 1/2 benchmark for a binary classifier; purged and embargoed 10-fold accuracy at or below 1/2 and
    within 0.1 of it. The control shows the folds are not simply crippled: the same splitter on a feature that does
    predict scores above 0.6."""
    from sklearn.model_selection import KFold
    X, y, signal, t0, t1 = leaky
    folds = list(cv.PurgedKFold(10, t0, t1, pct_embargo=0.01).split(X))
    shuffled = _nearest_neighbour_accuracy(X, y, KFold(10, shuffle=True, random_state=0).split(X))
    purged = _nearest_neighbour_accuracy(X, y, folds)
    assert y.mean() == 0.5
    assert shuffled > 0.6
    assert purged <= 0.5 and abs(purged - 0.5) < 0.1
    assert _nearest_neighbour_accuracy(signal, y, folds) > 0.6            # control: a real feature still scores


@pytest.mark.eval
def test_leaked_performance_keeps_improving_as_k_grows_where_purged_performance_stays_flat(leaky):
    """7.4.1-7.4.2's diagnostic: under leakage, performance improves merely by raising k towards T, because more
    testing splits put more overlapping observations in the training set; once purging and the embargo are in place
    it stops improving, which is how the book confirms that h suffices. Threshold over k = 2, 5, 10, 20, 50:
    shuffled folds gain more than 0.015 from k = 2 to k = 10 and stay above 0.6; purged folds never clear 1/2 and
    their whole range across k is under 0.03."""
    from sklearn.model_selection import KFold
    X, y, _, t0, t1 = leaky
    ks = (2, 5, 10, 20, 50)
    shuffled = {k: _nearest_neighbour_accuracy(X, y, KFold(k, shuffle=True, random_state=0).split(X)) for k in ks}
    purged = {k: _nearest_neighbour_accuracy(X, y, cv.PurgedKFold(k, t0, t1, pct_embargo=0.01).split(X)) for k in ks}
    assert shuffled[10] - shuffled[2] > 0.015
    assert min(shuffled.values()) > 0.6
    assert max(purged.values()) <= 0.5
    assert max(purged.values()) - min(purged.values()) < 0.03


@pytest.mark.property
def test_the_contiguous_test_set_short_cut_purges_exactly_what_span_by_span_purging_does():
    """7.4.1: when no training observation falls between the first and the last testing observation, testTimes may be
    a single span covering the whole test set. Checked against purging span by span on 25 random test blocks."""
    for seed in range(25):
        t0, t1 = _sorted_spans(200, 900, 35, seed)
        block = np.arange(60, 110)
        span_by_span = cv.train_times(t0, t1, t0[block], t1[block])
        envelope = cv.train_times(t0, t1, [t0[block].min()], [t1[block].max()])
        outside = np.setdiff1d(np.arange(len(t0)), block)
        assert np.array_equal(span_by_span[outside], envelope[outside])


@pytest.mark.property
def test_the_embargo_is_purging_against_a_test_span_stretched_by_h():
    """7.4.2's implementation: set Y_j = f[[t_j0, t_j1 + h]] before purging. Purging against the stretched span must
    drop exactly the rows that purging plus an explicit embargo drops, on 25 random samples."""
    for seed in range(25):
        t0, t1 = _sorted_spans(150, 700, 30, seed)
        block = np.arange(40, 80)
        h = 50.0
        lo, hi = t0[block].min(), t1[block].max()
        stretched = cv.train_times(t0, t1, [lo], [hi + h])
        purged = cv.train_times(t0, t1, [lo], [hi])
        embargoed = purged & ~((t0 >= hi) & (t0 < hi + h))
        assert np.array_equal(stretched, embargoed)
        assert np.all(stretched[t1 <= lo] == purged[t1 <= lo])                # nothing before the block is embargoed


# ------------------------------------------------------------------ the second reading (plan book-v2 node 16)


@pytest.mark.eval
def test_overlapping_features_alone_do_not_leak_and_a_model_read_on_its_own_rows_forecasts_nothing(leaky):
    """7.2's lossy compressor and 7.3's last paragraph, neither of which had a test that could fail.

    (a) Scored on the rows it was fitted on, a 1-nearest-neighbour classifier is exactly right and forecasts nothing:
    accuracy 1.0 in sample against chance on purged folds. (b) Leakage needs the pair (X, Y) to repeat, not X alone:
    the same serially correlated features with labels drawn row by row score at chance on the very folds that leak
    when the labels overlap, and so do overlapping labels with features that carry no serial correlation. Threshold:
    both controls within 0.03 of 1/2, where the leaky pair scores above 0.6 on the same shuffled folds."""
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier

    X, y, _, t0, t1 = leaky
    assert np.mean(KNeighborsClassifier(n_neighbors=1).fit(X, y).predict(X) == y) == 1.0
    purged = list(cv.PurgedKFold(10, t0, t1, pct_embargo=0.01).split(X))
    assert _nearest_neighbour_accuracy(X, y, purged) <= 0.5

    rng = np.random.default_rng(5)
    shuffled = KFold(10, shuffle=True, random_state=0)
    independent = (rng.random(len(y)) < 0.5).astype(int)              # X_i close to X_j, labels that do not overlap
    assert abs(_nearest_neighbour_accuracy(X, independent, shuffled.split(X)) - 0.5) < 0.03
    memoryless = rng.normal(size=X.shape)                            # Y_i close to Y_j, features without memory
    assert abs(_nearest_neighbour_accuracy(memoryless, y, shuffled.split(memoryless)) - 0.5) < 0.03
    assert _nearest_neighbour_accuracy(X, y, shuffled.split(X)) > 0.6          # only the pair together leaks


@pytest.mark.property
def test_the_test_folds_partition_the_rows_and_no_row_sits_in_both_sets():
    """7.2: the dataset is partitioned into k subsets and each observation belongs to one set and only one. Over 20
    random samples the k test blocks are contiguous in row order, disjoint and cover every row, and no training row
    is also a testing row -- purging only ever takes rows out of training."""
    for seed in range(20):
        t0, t1 = _sorted_spans(150, 600, 30, seed)
        folds = list(cv.PurgedKFold(5, t0, t1, pct_embargo=0.02).split(np.zeros(len(t0))))
        assert len(folds) == 5
        covered = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(covered), np.arange(len(t0)))
        for train, test in folds:
            assert np.intersect1d(train, test).size == 0
            assert np.array_equal(np.asarray(test), np.arange(test.min(), test.max() + 1))
