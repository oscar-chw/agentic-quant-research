import numpy as np
import pytest
from scipy.special import comb

from pmlab.afml import ensemble


@pytest.mark.case
def test_bagging_accuracy_of_the_books_example_is_the_binomial_tail():
    # N = 100 classifiers, p = 1/3, k = 3 classes: sum the binomial terms up to int(N / k) = 33
    head = sum(comb(100, i, exact=True) * (1 / 3) ** i * (2 / 3) ** (100 - i) for i in range(34))
    assert ensemble.bagging_accuracy(100, 1 / 3, 3) == pytest.approx(1 - head)
    assert ensemble.bagging_accuracy(100, 1 / 3, 3) > 1 / 3


@pytest.mark.eval
def test_bagging_accuracy_matches_simulated_votes_and_grows_with_n_only_above_chance():
    """Threshold: 200,000 simulated committees of 25 classifiers with p = 0.6 and k = 2; the share with more than
    12 right is within 0.005 of the formula."""
    rng = np.random.default_rng(0)
    right = rng.binomial(25, 0.6, 200_000)
    assert np.mean(right > 12) == pytest.approx(ensemble.bagging_accuracy(25, 0.6, 2), abs=0.005)
    assert ensemble.bagging_accuracy(101, 0.6) > ensemble.bagging_accuracy(11, 0.6) > 0.6
    assert ensemble.bagging_accuracy(101, 0.45) < ensemble.bagging_accuracy(11, 0.45) < 0.45


@pytest.mark.eval
def test_the_binomial_formula_is_the_plurality_accuracy_at_two_classes_and_an_upper_bound_above():
    """docs/afml_sections/ch06.md F6.1. 200,000 simulated committees: with k = 2 the formula is the accuracy of a
    plurality vote; with k = 3 more than N / k right is necessary and not sufficient, so the formula sits above the
    simulated plurality accuracy by more than simulation noise."""
    rng = np.random.default_rng(7)
    n_sim, N = 200_000, 10

    def plurality(p, k):
        wrong = (1 - p) / (k - 1)
        votes = rng.multinomial(N, [p] + [wrong] * (k - 1), size=n_sim)      # column 0 is the observed class
        return float(np.mean(votes[:, 0] > votes[:, 1:].max(axis=1)))        # a tie is not a win

    assert plurality(0.6, 2) == pytest.approx(ensemble.bagging_accuracy(N, 0.6, 2), abs=0.005)
    sim3 = plurality(0.6, 3)
    assert sim3 < ensemble.bagging_accuracy(N, 0.6, 3) - 0.01               # strictly an upper bound at k = 3
    assert np.mean(rng.binomial(N, 0.6, n_sim) > N / 3) == pytest.approx(ensemble.bagging_accuracy(N, 0.6, 3), abs=0.005)


@pytest.mark.eval
def test_the_mean_squared_error_splits_into_bias_squared_variance_and_noise():
    """6.2's decomposition on a case whose three terms are known. y = 2x + N(0, 0.5) and the estimator is a constant
    (the training mean), so at x = 1 it is biased by E[mean of y over x ~ U[0, 1]] - 2 = -1. Threshold: over 20,000
    training sets the simulated E[(y - f_hat)^2] is within 1.5% of bias^2 + variance + sigma^2, and no two of the
    three terms alone come that close."""
    rng = np.random.default_rng(11)
    trials, n, sigma = 20_000, 40, 0.5
    x = rng.uniform(0, 1, size=(trials, n))
    y = 2 * x + rng.normal(0, sigma, size=(trials, n))
    f_hat = y.mean(axis=1)                                  # a constant model: the training mean
    y_new = 2 * 1.0 + rng.normal(0, sigma, trials)          # a fresh outcome at x = 1, where f[x] = 2
    mse = np.mean((y_new - f_hat) ** 2)
    bias, variance = f_hat.mean() - 2.0, f_hat.var()
    assert bias == pytest.approx(-1.0, abs=0.02)            # E[2x] = 1 against f[1] = 2
    assert mse == pytest.approx(bias**2 + variance + sigma**2, rel=0.015)
    assert mse != pytest.approx(bias**2 + variance, rel=0.015)          # the noise term is not optional
    assert mse != pytest.approx(variance + sigma**2, rel=0.015)         # nor is the bias term


@pytest.mark.eval
def test_bagged_variance_is_the_books_formula_and_bagging_stops_helping_as_the_correlation_reaches_one():
    """6.3.1. Threshold: 200,000 draws of N equicorrelated forecasts; the sample variance of their mean is within 2%
    of sigma^2 (rho + (1 - rho) / N). Figure 6.1's reading is checked too: the standard deviation falls with N at
    rho < 1 and is sigma whatever N is at rho = 1."""
    rng = np.random.default_rng(12)
    for n, rho, sigma in [(5, 0.0, 1.0), (10, 0.3, 1.0), (25, 0.7, 2.0), (30, 0.95, 1.0)]:
        cov = sigma**2 * ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))
        draws = rng.standard_normal((200_000, n)) @ np.linalg.cholesky(cov).T
        assert draws.mean(axis=1).var() == pytest.approx(ensemble.bagged_variance(sigma, rho, n), rel=0.02)
    assert ensemble.bagged_variance(1.0, 1.0, 5) == ensemble.bagged_variance(1.0, 1.0, 30) == 1.0
    falling = [ensemble.bagged_variance(1.0, 0.2, n) for n in range(5, 31)]
    assert all(a > b for a, b in zip(falling, falling[1:]))
    assert falling[-1] > 0.2 and ensemble.bagged_variance(1.0, 0.2, 10_000) == pytest.approx(0.2, abs=1e-4)
    with pytest.raises(ValueError):
        ensemble.bagged_variance(1.0, 1.2, 10)


@pytest.mark.property
def test_above_the_books_committee_size_the_vote_is_more_accurate_than_one_classifier():
    """6.3.2's N > p (p - 1 / k)^-2 implies P[X > N / k] > p. Checked over k = 2..5 and p from just above 1 / k to
    0.97, for 40 committee sizes above the threshold each; and shown to be sufficient only, not necessary."""
    for k in (2, 3, 4, 5):
        for p in np.arange(1 / k + 0.01, 0.98, 0.02):
            n0 = ensemble.estimators_for_accuracy(float(p), k)
            for n in range(int(np.ceil(n0)) + 1, int(np.ceil(n0)) + 41):
                assert ensemble.bagging_accuracy(n, float(p), k) > p
    assert ensemble.estimators_for_accuracy(0.6, 2) == pytest.approx(0.6 / 0.1**2)
    assert ensemble.bagging_accuracy(11, 0.6, 2) > 0.6                  # far below the threshold of 60, already better
    assert ensemble.bagging_accuracy(12, 0.505, 2) < 0.505              # and not always: the threshold is 20,200
    with pytest.raises(ValueError):
        ensemble.estimators_for_accuracy(0.3, 3)                        # p below 1 / k: the implication is empty


@pytest.mark.eval
def test_drawing_the_average_uniqueness_decorrelates_the_trees_where_a_full_draw_does_not():
    """6.3.1 and 6.3.3's first effect: with redundant (overlapping) observations, bootstrap samples of the full
    training size are virtually identical, so the trees' forecasts correlate and bagging cannot reduce variance.
    On 60 events repeated as 20 near-identical rows each (average uniqueness 1 / 20), a full draw of 1,200 rows holds
    every event and a draw of 5% holds about two thirds of them. Threshold: mean pairwise correlation of 30 trees'
    out-of-sample probabilities above 0.5 at max_samples = 1.0 and at least 0.2 lower at max_samples = 0.05, with the
    bagged forecast's variance falling in the same order."""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(13)
    n_events, per_event = 60, 20
    base = rng.normal(size=(n_events, 6))
    y_event = (base[:, 0] + base[:, 1] + rng.normal(0, 1.5, n_events) > 0).astype(int)
    X = np.repeat(base, per_event, axis=0) + rng.normal(0, 0.01, size=(n_events * per_event, 6))
    y = np.repeat(y_event, per_event)
    test = rng.normal(size=(300, 6))

    def trees(max_samples):
        bag = BaggingClassifier(DecisionTreeClassifier(criterion="entropy", random_state=0), n_estimators=30,
                                max_samples=max_samples, random_state=0).fit(X, y)
        p = np.column_stack([t.predict_proba(test[:, f])[:, 1]
                             for t, f in zip(bag.estimators_, bag.estimators_features_)])
        c = np.corrcoef(p, rowvar=False)
        return float(np.nanmean(c[np.triu_indices_from(c, k=1)])), float(p.mean(axis=1).var())

    full_rho, full_var = trees(1.0)
    unique_rho, unique_var = trees(1.0 / per_event)
    assert full_rho > 0.5
    assert unique_rho < full_rho - 0.2
    assert unique_var < full_var
    # the formula of 6.3.1 reads the same way: at these correlations, 30 estimators are worth far more when rho is low
    assert ensemble.bagged_variance(1.0, unique_rho, 30) < ensemble.bagged_variance(1.0, full_rho, 30)


@pytest.mark.eval
def test_out_of_bag_accuracy_is_inflated_when_observations_are_redundant():
    """6.3.3's second effect. The same 60 events repeated as 20 near-identical rows: a row left out of a bag has its
    twins in it, so the out-of-bag score reads far above what an unshuffled fold that keeps an event whole gives.
    Threshold: out-of-bag accuracy above 0.95 while the grouped, unshuffled 5-fold accuracy is at least 0.15 lower."""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(14)
    n_events, per_event = 60, 20
    base = rng.normal(size=(n_events, 6))
    y_event = (base[:, 0] + base[:, 1] + rng.normal(0, 1.5, n_events) > 0).astype(int)
    X = np.repeat(base, per_event, axis=0) + rng.normal(0, 0.01, size=(n_events * per_event, 6))
    y = np.repeat(y_event, per_event)
    tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
    oob = BaggingClassifier(tree, n_estimators=60, oob_score=True, random_state=0).fit(X, y).oob_score_
    blocks = np.array_split(np.arange(n_events * per_event), 5)         # whole events, in order, never shuffled
    scores = []
    for block in blocks:                                                # the book's StratifiedKFold(shuffle=False)
        train = np.setdiff1d(np.arange(len(y)), block)
        fitted = BaggingClassifier(tree, n_estimators=60, random_state=0).fit(X[train], y[train])
        scores.append(float(np.mean(fitted.predict(X[block]) == y[block])))
    assert oob > 0.95
    assert np.mean(scores) < oob - 0.15


@pytest.mark.eval
def test_boosting_cuts_the_bias_of_a_weak_learner_where_bagging_only_cuts_its_variance():
    """6.5-6.6: bagging addresses overfitting (variance), boosting addresses underfitting (bias). The label alternates
    over four bands of one feature, which one stump cannot express; bagging stumps averages the same underfit
    forecast, while AdaBoost's sequential reweighting of the misclassified rows builds the missing thresholds.
    Threshold: out-of-sample accuracy of one stump below 0.8, 50 bagged stumps within 0.02 of it, AdaBoost above
    0.95."""
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(15)
    x = rng.uniform(0, 4, 4000)
    X = np.column_stack([x, rng.normal(size=4000)])
    y = (np.floor(x) % 2 == 0).astype(int)
    tr, te = np.arange(2000), np.arange(2000, 4000)
    stump = lambda: DecisionTreeClassifier(max_depth=1, random_state=0)
    one = stump().fit(X[tr], y[tr]).score(X[te], y[te])
    bagged = BaggingClassifier(stump(), n_estimators=50, random_state=0).fit(X[tr], y[tr]).score(X[te], y[te])
    boosted = AdaBoostClassifier(stump(), n_estimators=50, random_state=0).fit(X[tr], y[tr]).score(X[te], y[te])
    assert one < 0.8
    assert abs(bagged - one) < 0.02
    assert boosted > 0.95


@pytest.mark.eval
def test_early_stopping_raises_one_estimators_variance_and_bagging_takes_it_back():
    """6.7: a tight early stop (the book's max_iter / tol on a slow learner) makes each base estimator noisier, and
    that extra variance is more than offset by bagging. Over 40 training sets, the forecast of one SGD logistic model
    at a fixed point: threshold: stopping after one pass gives at least 1.5x the variance of a converged fit, and
    bagging 40 early-stopped ones brings it below the converged fit's."""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import SGDClassifier

    point = np.random.default_rng(16).normal(size=(1, 5))

    def forecast_variance(model):
        out = []
        for seed in range(40):
            r = np.random.default_rng(100 + seed)
            X = r.normal(size=(400, 5))
            y = (X[:, 0] + 0.7 * X[:, 1] + r.normal(0, 1.0, 400) > 0).astype(int)
            out.append(model().fit(X, y).predict_proba(point)[0, 1])
        return float(np.var(out))

    converged = lambda: SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-6, random_state=0)
    stopped = lambda: SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=0)
    bagged = lambda: BaggingClassifier(stopped(), n_estimators=40, max_samples=0.5, random_state=0)
    v_converged, v_stopped, v_bagged = forecast_variance(converged), forecast_variance(stopped), forecast_variance(bagged)
    assert v_stopped > 1.5 * v_converged
    assert v_bagged < v_converged


# ------------------------------------------------------------------ the second reading (plan book-v2 node 16)


@pytest.mark.case
def test_the_bagged_forecast_averages_its_estimators_and_every_draw_is_with_replacement():
    """6.3's three steps, which no test asserted before: N training sets drawn with replacement, N estimators fitted
    from their own draw alone (so the fits are independent and can run in parallel), and a forecast that is the simple
    average of the N forecasts -- the mean of the probabilities when the base estimator has one, and the share of the
    estimators voting for a class when it has not."""
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(21)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 300) > 0).astype(int)
    test = rng.normal(size=(50, 4))

    bag = BaggingClassifier(DecisionTreeClassifier(max_depth=3, random_state=0), n_estimators=12,
                            max_samples=0.6, random_state=3).fit(X, y)
    members = np.column_stack([t.predict_proba(test[:, f])[:, 1]
                               for t, f in zip(bag.estimators_, bag.estimators_features_)])
    np.testing.assert_allclose(bag.predict_proba(test)[:, 1], members.mean(axis=1))
    assert members.std(axis=1).mean() > 0.05                  # the members disagree, so the mean is not any one of them

    draws = bag.estimators_samples_                            # step one: 0.6 x 300 rows, drawn with replacement
    assert all(len(d) == 180 for d in draws)
    assert all(len(np.unique(d)) < len(d) for d in draws)      # repeats: sampling is with replacement, not a subset

    for t, d, f in zip(bag.estimators_, draws, bag.estimators_features_):
        alone = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[np.ix_(d, f)], y[d])
        np.testing.assert_allclose(alone.predict_proba(test[:, f]), t.predict_proba(test[:, f]))  # step two

    votes = BaggingClassifier(LinearSVC(), n_estimators=9, max_samples=0.6, random_state=3).fit(X, y)
    share = np.mean([e.predict(test[:, f]) for e, f in zip(votes.estimators_, votes.estimators_features_)], axis=0)
    np.testing.assert_allclose(votes.predict_proba(test)[:, 1], share)      # a class probability is the vote share
    assert ((share > 0) & (share < 1)).any()                                # and the committee is not unanimous


@pytest.mark.property
def test_the_average_variance_and_correlation_are_what_turn_the_double_sum_into_the_books_formula():
    """6.3.1's algebra, not just its result: V[mean] is the double sum of the covariances over N^2, and it equals
    sigma_bar^2 (rho_bar + (1 - rho_bar) / N) only with the book's definitions -- sigma_bar^2 the mean variance and
    rho_bar the sigma-weighted mean correlation. On unequal variances the unweighted mean correlation misses the
    identity, so this would fail if the definition were relaxed to a plain average."""
    rng = np.random.default_rng(22)
    for n in (3, 7, 20):
        loading, idio = rng.uniform(0.3, 1.6, n), rng.uniform(0.05, 1.5, n)
        cov = np.outer(loading, loading) + np.diag(idio)                    # one factor: all correlations positive
        sigma, rho = ensemble.average_variance_and_correlation(cov)
        assert sigma == pytest.approx(np.sqrt(np.mean(np.diag(cov))))
        assert ensemble.bagged_variance(sigma, rho, n) == pytest.approx(cov.sum() / n**2)
        d = np.sqrt(np.diag(cov))
        plain = float(((cov / np.outer(d, d)).sum() - n) / (n * (n - 1)))   # the unweighted mean correlation
        assert plain != pytest.approx(rho, rel=1e-6)
        assert ensemble.bagged_variance(sigma, plain, n) != pytest.approx(cov.sum() / n**2, rel=1e-6)
        draws = rng.standard_normal((200_000, n)) @ np.linalg.cholesky(cov).T
        assert draws.mean(axis=1).var() == pytest.approx(ensemble.bagged_variance(sigma, rho, n), rel=0.02)
    with pytest.raises(ValueError):
        ensemble.average_variance_and_correlation([[1.0]])


@pytest.mark.eval
def test_a_forest_holds_its_draw_at_the_training_size_decorrelates_by_features_and_a_leaf_fraction_closes_the_oob_gap():
    """6.4. Four claims that had no test of their own. (1) A forest always draws as many rows as the training set,
    where bagging draws max_samples of them. (2) The second randomisation is over features: lowering max_features
    lowers the mean pairwise correlation of the trees' forecasts, and with it the bagged variance. (3) One unpruned
    tree's forecast varies far more across training sets than a forest's. (4) The book's early stop -- a minimum
    weighted fraction per leaf -- pulls the inflated out-of-bag score back towards the unshuffled k-fold score.
    Thresholds: correlation falling monotonically over max_features = 12, 6, 2, 1 and at least 0.15 in all; one tree's
    forecast variance at least 3x the forest's; and the out-of-bag-minus-fold gap falling monotonically over a leaf
    fraction of 0, 5% and 10%, by at least 0.2 in all."""
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(23)
    n, p = 400, 12
    X = rng.normal(size=(n, p))
    y = (X[:, :4].sum(axis=1) + rng.normal(0, 1.0, n) > 0).astype(int)
    test = rng.normal(size=(200, p))

    forest = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    assert {len(s) for s in forest.estimators_samples_} == {n}               # always the training size
    bagged = BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=40, max_samples=0.3,
                               random_state=0).fit(X, y)
    assert {len(s) for s in bagged.estimators_samples_} == {int(0.3 * n)}    # bagging's draw is what you ask for

    def correlation(max_features):
        f = RandomForestClassifier(n_estimators=40, max_features=max_features, random_state=0).fit(X, y)
        probs = np.column_stack([t.predict_proba(test)[:, 1] for t in f.estimators_])
        c = np.corrcoef(probs, rowvar=False)
        return float(np.nanmean(c[np.triu_indices_from(c, k=1)])), float(probs.mean(axis=1).var())

    rhos, variances = zip(*[correlation(m) for m in (p, 6, 2, 1)])
    assert all(a > b for a, b in zip(rhos, rhos[1:]))                        # fewer features, less correlated trees
    assert rhos[0] - rhos[-1] > 0.15
    assert all(a > b for a, b in zip(variances, variances[1:]))              # and a lower bagged variance, as 6.3.1 says
    assert ensemble.bagged_variance(1.0, rhos[-1], 40) < ensemble.bagged_variance(1.0, rhos[0], 40)

    point = rng.normal(size=(1, p))

    def forecast_variance(make):
        out = []
        for seed in range(25):
            r = np.random.default_rng(300 + seed)
            Xs = r.normal(size=(n, p))
            ys = (Xs[:, :4].sum(axis=1) + r.normal(0, 1.0, n) > 0).astype(int)
            out.append(make().fit(Xs, ys).predict_proba(point)[0, 1])
        return float(np.var(out))

    assert forecast_variance(lambda: DecisionTreeClassifier(random_state=0)) > 3 * forecast_variance(
        lambda: RandomForestClassifier(n_estimators=40, random_state=0))

    rows = np.random.default_rng(14)                                        # 60 events repeated as 20 near-copies
    base = rows.normal(size=(60, 6))
    y_event = (base[:, 0] + base[:, 1] + rows.normal(0, 1.5, 60) > 0).astype(int)
    Xr = np.repeat(base, 20, axis=0) + rows.normal(0, 0.01, size=(1200, 6))
    yr = np.repeat(y_event, 20)
    blocks = np.array_split(np.arange(1200), 5)                             # whole events, in order, never shuffled

    def oob_gap(leaf_fraction):
        args = dict(n_estimators=60, min_weight_fraction_leaf=leaf_fraction, random_state=0)
        oob = RandomForestClassifier(oob_score=True, **args).fit(Xr, yr).oob_score_
        folds = []
        for block in blocks:
            train = np.setdiff1d(np.arange(1200), block)
            fitted = RandomForestClassifier(**args).fit(Xr[train], yr[train])
            folds.append(float(np.mean(fitted.predict(Xr[block]) == yr[block])))
        return oob - float(np.mean(folds))

    gaps = [oob_gap(f) for f in (0.0, 0.05, 0.1)]
    assert all(a > b for a, b in zip(gaps, gaps[1:]))
    assert gaps[0] - gaps[-1] > 0.2


@pytest.mark.eval
def test_the_books_six_boosting_steps_reweight_the_misclassified_discard_the_weak_and_weight_by_accuracy():
    """6.5 and 6.6, step by step rather than by the ensemble's accuracy alone. The loop is the book's: draw under the
    current weights, fit one estimator, keep it only if it beats the acceptance threshold, raise the weight of the
    rows it got wrong, repeat, and forecast by the accuracy-weighted vote. Every fourth round is handed a classifier
    worse than chance, which the threshold must reject. Thresholds: all ten rejected; the misclassified rows of round
    one at least twice the weight of the rest; each kept weight exactly the log-odds of that round's weighted error;
    one stump below 0.8 out of sample and the boosted committee above 0.95. Then sklearn's AdaBoost on a noisy copy:
    its prediction is exactly the weight-weighted vote of its estimators, which differs from the unweighted vote and
    scores above it."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(15)
    x = rng.uniform(0, 4, 4000)
    X = np.column_stack([x, rng.normal(size=4000)])
    y = (np.floor(x) % 2 == 0).astype(int)                     # four alternating bands: one stump cannot express them
    tr, te = np.arange(2000), np.arange(2000, 4000)
    stump = lambda: DecisionTreeClassifier(max_depth=1, random_state=0)

    class Flipped:                                             # accuracy 1 - p: below any acceptance threshold
        def __init__(self, inner):
            self.inner = inner

        def predict(self, rows):
            return 1 - self.inner.predict(rows)

    n, k = len(tr), 2
    weights = np.full(n, 1.0 / n)                              # step one's weights start uniform
    kept, alphas, errors, discarded, first = [], [], [], 0, None
    draw_rng = np.random.default_rng(31)
    for round_ in range(40):
        draw = draw_rng.choice(n, size=n, replace=True, p=weights)         # step one
        estimator = stump().fit(X[tr][draw], y[tr][draw])                  # step two
        if round_ % 4 == 3:
            estimator = Flipped(estimator)
        predicted = estimator.predict(X[tr])
        error = float(np.sum(weights * (predicted != y[tr])))
        if error >= 1 - 1.0 / k:                                           # step three: discard, do not reweight
            discarded += 1
            continue
        alpha = float(np.log((1 - error) / error) + np.log(k - 1))
        if first is None:
            first = (predicted, weights.copy(), alpha)
        weights = weights * np.exp(alpha * (predicted != y[tr]))           # step four
        weights /= weights.sum()
        kept.append(estimator)                                             # step five
        alphas.append(alpha)
        errors.append(error)

    assert discarded == 10 and len(kept) == 30
    assert max(errors) < 0.5 and min(alphas) > 0
    np.testing.assert_allclose(alphas, np.log((1 - np.array(errors)) / np.array(errors)))
    wrong, before, alpha0 = first
    after = before * np.exp(alpha0 * (wrong != y[tr]))
    after /= after.sum()
    assert after[wrong != y[tr]].mean() > 2 * after[wrong == y[tr]].mean()
    assert after[wrong == y[tr]].mean() < before[wrong == y[tr]].mean()     # and the correct ones lose weight

    uniform = stump().fit(X[tr], y[tr], sample_weight=np.full(n, 1.0 / n))  # 6.6's first difference: round two
    reweighted = stump().fit(X[tr], y[tr], sample_weight=after)             # depends on round one
    assert uniform.tree_.threshold[0] != reweighted.tree_.threshold[0]
    assert not np.array_equal(uniform.predict(X[te]), reweighted.predict(X[te]))

    def vote(members, member_weights, rows):                                # step six
        tally = np.zeros((len(rows), k))
        for weight, member in zip(member_weights, members):
            tally[np.arange(len(rows)), member.predict(rows)] += weight
        return tally.argmax(axis=1)

    assert stump().fit(X[tr], y[tr]).score(X[te], y[te]) < 0.8
    assert np.mean(vote(kept, alphas, X[te]) == y[te]) > 0.95

    noisy = np.random.default_rng(16)
    x = noisy.uniform(0, 4, 4000)
    X = np.column_stack([x, noisy.normal(size=4000)])
    y = (np.floor(x) % 2 == 0).astype(int)
    y = np.where(noisy.random(4000) < 0.15, 1 - y, y)                       # 15% flipped labels: the rounds differ
    ada = AdaBoostClassifier(stump(), n_estimators=25, random_state=0).fit(X[tr], y[tr])
    fitted = ada.estimator_errors_
    np.testing.assert_allclose(ada.estimator_weights_, np.log((1 - fitted) / fitted) + np.log(k - 1))
    assert ada.estimator_weights_.max() > 10 * ada.estimator_weights_.min()
    weighted = vote(ada.estimators_, ada.estimator_weights_, X[te])
    simple = vote(ada.estimators_, np.ones(len(ada.estimators_)), X[te])
    assert np.array_equal(weighted, ada.predict(X[te]))                     # the forecast IS the weighted vote
    assert np.sum(weighted != simple) > 10
    assert np.mean(weighted == y[te]) > np.mean(simple == y[te])
