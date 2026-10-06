import numpy as np
import pytest
from scipy import stats
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pmlab.afml import cv, tuning


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 900
    X = rng.normal(size=(n, 3))
    y = (1.5 * X[:, 0] + rng.normal(size=n) > 0).astype(int)
    w = rng.uniform(0.1, 2.0, n)
    t0 = np.arange(n)
    return X, y, w, t0, t0 + 20


@pytest.mark.case
def test_weighted_pipeline_passes_the_weights_to_its_last_step(data):
    X, y, w, _, _ = data
    steps = lambda: [("sc", StandardScaler()), ("lr", LogisticRegression())]
    got = tuning.WeightedPipeline(steps()).fit(X, y, sample_weight=w)["lr"].coef_
    want = Pipeline(steps()).fit(X, y, lr__sample_weight=w)["lr"].coef_
    plain = Pipeline(steps()).fit(X, y)["lr"].coef_
    assert got == pytest.approx(want)
    assert np.max(np.abs(got - plain)) > 1e-3
    bag = BaggingClassifier(estimator=tuning.WeightedPipeline(steps()), n_estimators=3, random_state=0)
    assert bag.fit(X, y, sample_weight=w).score(X, y) > 0.7


@pytest.mark.case
def test_log_uniform_has_the_books_cdf_and_uniform_logs():
    a, b = 1e-3, 1e3
    dist = tuning.log_uniform(a, b)
    x = np.array([2e-3, 0.5, 1.0, 700.0])
    assert dist.cdf(x) == pytest.approx(np.log(x / a) / np.log(b / a))
    assert dist.cdf(x) == pytest.approx(np.log10(x / a) / np.log10(b / a))           # base-free
    vals = dist.rvs(size=20_000, random_state=1)
    assert vals.min() >= a and vals.max() <= b
    assert stats.kstest(np.log(vals), "uniform", args=(np.log(a), np.log(b / a))).pvalue > 0.01
    assert stats.kstest(vals, "uniform", args=(a, b - a)).pvalue < 1e-6                 # not uniform in levels
    # 9.3.1's arithmetic: a uniform draw on [0, 100] leaves 99% of the values above 1 and explores the small end
    # not at all, where the log-uniform draw splits its decades evenly - half of [1e-3, 1e3] above 1.
    assert stats.uniform(0, 100).sf(1.0) == pytest.approx(0.99)
    assert np.mean(stats.uniform(0, 100).rvs(size=20_000, random_state=2) > 1.0) == pytest.approx(0.99, abs=0.005)
    assert np.mean(vals > 1.0) == pytest.approx(0.5, abs=0.02)
    assert np.mean((vals > 0.01) & (vals < 1.0)) == pytest.approx(np.mean((vals > 1.0) & (vals < 100.0)), abs=0.02)
    with pytest.raises(ValueError):
        tuning.log_uniform(0, 1)


@pytest.mark.case
def test_grid_search_scores_every_candidate_on_purged_weighted_folds(data):
    X, y, w, t0, t1 = data
    res = tuning.purged_search(LogisticRegression(), {"C": [1e-4, 1e-2, 1.0]}, X, y, t0, t1, n_splits=4,
                               pct_embargo=0.01, sample_weight=w)
    folds = cv.PurgedKFold(4, t0, t1, 0.01)
    for params, scores in res["results"]:
        assert scores == pytest.approx(cv.cv_score(LogisticRegression(**params), X, y, w, "neg_log_loss", folds))
    means = [s.mean() for _, s in res["results"]]
    assert res["best_params"] == res["results"][int(np.argmax(means))][0]
    assert res["best_params"]["C"] == 1.0                                            # a real signal beats C = 1e-4
    assert res["model"].C == 1.0 and hasattr(res["model"], "coef_")
    assert tuning.book_scoring(y) == "f1" and tuning.book_scoring(np.sign(y - 0.5)) == "neg_log_loss"
    f1 = tuning.purged_search(LogisticRegression(), {"C": [1.0]}, X, y, t0, t1, scoring="book")
    assert f1["scoring"] == "f1" and 0.5 < f1["best_score"] <= 1


@pytest.mark.case
def test_randomised_search_draws_from_distributions_and_bagging_uses_the_weights(data):
    X, y, w, t0, t1 = data
    w = w * len(w) / w.sum()           # sklearn's bagging draws rows by weight, max_samples in units of the weight sum
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
    space = {"lr__C": tuning.log_uniform(1e-3, 1e3)}
    res = tuning.purged_search(pipe, space, X, y, t0, t1, n_iter=5, sample_weight=w, bagging=(4, 0.5, 1.0), rng=3)
    cs = [p["lr__C"] for p, _ in res["results"]]
    assert len(set(cs)) == 5 and all(1e-3 <= c <= 1e3 for c in cs)
    again = tuning.purged_search(pipe, space, X, y, t0, t1, n_iter=5, sample_weight=w, rng=3)
    assert [p["lr__C"] for p, _ in again["results"]] == cs
    bag = res["model"]
    assert isinstance(bag, BaggingClassifier) and isinstance(bag.estimator, tuning.WeightedPipeline)
    assert [len(s) for s in bag.estimators_samples_] == [int(0.5 * len(y))] * 4
    assert bag.estimators_[0]["lr"].C == res["best_params"]["lr__C"]


@pytest.mark.case
def test_bagging_draws_max_samples_of_the_rows_whatever_the_scale_of_the_weights(data):
    """9.2 / docs/afml_sections/ch09.md F9.1: a caller may pass raw average uniqueness (mean well below 1). sklearn 1.9
    reads a fractional max_samples in units of the weight sum, so purged_search turns the share into a row count and
    the draw no longer depends on the weights' scale."""
    X, y, _, t0, t1 = data
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
    raw = np.full(len(y), 0.2)                       # average uniqueness 0.2, as the event dataset gives it
    for w in (raw, raw / raw.mean(), np.linspace(0.05, 0.4, len(y))):
        res = tuning.purged_search(pipe, {"lr__C": [1.0]}, X, y, t0, t1, sample_weight=w, bagging=(4, 0.5, 1.0), rng=0)
        assert [len(s) for s in res["model"].estimators_samples_] == [int(0.5 * len(y))] * 4
    plain = BaggingClassifier(tuning.WeightedPipeline(pipe.steps), n_estimators=4, max_samples=0.5, random_state=0)
    plain.fit(X, y, sample_weight=raw)               # what sklearn does with the share left as a fraction
    assert max(len(s) for s in plain.estimators_samples_) < 0.5 * len(y) / 2


@pytest.mark.case
def test_the_grid_is_the_cartesian_product_while_the_random_budget_is_fixed_whatever_the_dimension(data):
    """9.2-9.3: a grid search visits every combination, so its cost is the product of the axes (the book's exercise
    asks how many nodes a 5 x 5 grid has); a randomised search draws a fixed number of candidates however many axes
    there are, which is the computational budget the chapter wants."""
    X, y, w, t0, t1 = data
    grid = {"C": [1e-2, 1e-1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
    res = tuning.purged_search(LogisticRegression(max_iter=500), grid, X, y, t0, t1, n_splits=3, sample_weight=w)
    assert len(res["results"]) == 5 * 2
    assert {(p["C"], p["class_weight"]) for p, _ in res["results"]} == {
        (c, cw) for c in grid["C"] for cw in grid["class_weight"]}
    one_axis = {"C": tuning.log_uniform(1e-2, 1e2)}
    four_axes = dict(one_axis, tol=tuning.log_uniform(1e-6, 1e-2), class_weight=[None, "balanced"],
                     fit_intercept=[True, False])
    for space in (one_axis, four_axes):
        drawn = tuning.purged_search(LogisticRegression(max_iter=500), space, X, y, t0, t1, n_splits=3, n_iter=7,
                                     sample_weight=w, rng=1)
        assert len(drawn["results"]) == 7
        assert all(1e-2 <= p["C"] <= 1e2 for p, _ in drawn["results"])


@pytest.mark.case
def test_log_loss_is_the_books_one_of_k_sum_and_punishes_a_confident_miss_more_than_a_timid_hit_pays():
    """9.4's formula L[Y, P] = -N^-1 sum_n sum_k y_nk log p_nk on a 1-of-K indicator matrix, and Figure 9.2's
    reading: two predictions of class 1 where the truth is 1 and 0 leave accuracy at 50% for every confidence, while
    the log loss climbs as that confidence rises."""
    from sklearn.metrics import accuracy_score, log_loss
    rng = np.random.default_rng(8)
    n, k = 60, 3
    y = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    indicator = np.zeros((n, k))
    indicator[np.arange(n), y] = 1.0
    by_hand = -np.sum(indicator * np.log(p)) / n
    assert by_hand == pytest.approx(log_loss(y, p, labels=range(k)))
    truth = np.array([1, 0])
    losses = []
    for confidence in (0.5, 0.6, 0.7, 0.8, 0.9):
        prob = np.array([[1 - confidence, confidence], [1 - confidence, confidence]])
        assert accuracy_score(truth, prob.argmax(axis=1)) == 0.5
        losses.append(log_loss(truth, prob, labels=[0, 1]))
    assert all(a < b for a, b in zip(losses, losses[1:]))
    assert losses[0] == pytest.approx(np.log(2))
    assert losses[-1] > 1.7 * losses[0]
    saved, lost = np.log(2) - -np.log(0.9), -np.log(0.1) - np.log(2)   # the hit's gain against the miss's cost at 0.9
    assert lost > 2 * saved


@pytest.mark.case
def test_f1_exposes_a_classifier_that_calls_every_case_negative_where_accuracy_and_log_loss_do_not():
    """9.2's advice to score meta-labelling with F1. With 5% positives, a model regularised into predicting the
    majority class everywhere still scores above 0.92 accuracy and a log loss under 0.3, while its F1 is exactly 0
    (zero recall) and an informative model's F1 is well above it."""
    rng = np.random.default_rng(9)
    n = 1500
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 1.645).astype(int)                       # about 5% positives
    t0 = np.arange(n)
    folds = cv.PurgedKFold(3, t0, t0 + 1)
    blind = LogisticRegression(C=1e-6, max_iter=500)
    informed = LogisticRegression(C=100.0, max_iter=1000)
    assert 0.03 < y.mean() < 0.08
    assert np.all(cv.cv_score(blind, X, y, scoring="accuracy", cv=folds) > 0.92)
    assert np.all(cv.cv_score(blind, X, y, scoring="neg_log_loss", cv=folds) > -0.3)
    assert np.all(cv.cv_score(blind, X, y, scoring="f1", cv=folds) == 0.0)
    assert cv.cv_score(informed, X, y, scoring="f1", cv=folds).mean() > 0.5


@pytest.mark.case
def test_relabelling_the_classes_leaves_accuracy_and_log_loss_alone_but_moves_f1():
    """9.2's closing remark. Swapping which class is called 1 is a relabelling: accuracy and negative log loss are
    unchanged to within floating point, F1 is not, because it scores precision and recall of the positive class."""
    rng = np.random.default_rng(10)
    n = 1500
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + rng.normal(0, 0.7, n) > 1.645).astype(int)     # about 8% positives, and not separable
    t0 = np.arange(n)
    folds = cv.PurgedKFold(3, t0, t0 + 1)
    model = LogisticRegression(C=100.0, max_iter=1000)
    for scoring in ("accuracy", "neg_log_loss"):
        assert cv.cv_score(model, X, y, scoring=scoring, cv=folds) == pytest.approx(
            cv.cv_score(model, X, 1 - y, scoring=scoring, cv=folds), abs=1e-9)
    f1_one = cv.cv_score(model, X, y, scoring="f1", cv=folds).mean()
    f1_other = cv.cv_score(model, X, 1 - y, scoring="f1", cv=folds).mean()
    assert y.mean() < 0.15
    assert abs(f1_one - f1_other) > 0.3


# ------------------------------------------------------------------ the second reading (plan book-v2 node 16)


@pytest.mark.case
def test_the_weighted_log_loss_reads_the_side_the_size_and_the_outcome_of_a_position():
    """9.4's second reason for preferring the cross-entropy: scored with chapter 4's weights, which follow the
    absolute return, it reads performance in profit-and-loss terms -- the label gives the side, the probability the
    size and the weight the outcome. Two classifiers with the same accuracy and the same confidence, one wrong on the
    big moves and one wrong on the small ones, are indistinguishable to accuracy and to the unweighted log loss; the
    weighted log loss ranks them exactly as the profit and loss of a position sized by confidence does."""
    from sklearn.metrics import accuracy_score, log_loss

    rng = np.random.default_rng(11)
    n = 400
    y = rng.integers(0, 2, n)
    absolute_return = np.where(np.arange(n) < n // 2, 0.2, 2.0)          # half small moves, half ten times larger
    big = np.arange(n) >= n // 2

    def forecast(wrong_on_big, confidence=0.9):
        right = np.where(big == wrong_on_big, 1 - confidence, confidence)
        return np.column_stack([np.where(y == 1, 1 - right, right), np.where(y == 1, right, 1 - right)])

    def pnl(prob):                                                       # side from the label, size from the p
        return float(np.sum((2 * prob[:, 1] - 1) * (2 * y - 1) * absolute_return))

    on_big, on_small = forecast(True), forecast(False)
    assert accuracy_score(y, on_big.argmax(axis=1)) == accuracy_score(y, on_small.argmax(axis=1)) == 0.5
    assert log_loss(y, on_big, labels=[0, 1]) == pytest.approx(log_loss(y, on_small, labels=[0, 1]))
    weighted = lambda prob: log_loss(y, prob, sample_weight=absolute_return, labels=[0, 1])
    assert weighted(on_big) > 5 * weighted(on_small)                     # the expensive mistakes cost more
    assert pnl(on_big) < 0 < pnl(on_small)                               # and the ranking is the profit's own
