import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from pmlab.afml import importance as fi
from pmlab.labels import purged_kfold


@pytest.fixture(scope="module")
def data():
    X, y = fi.synthetic_data(n_features=15, n_informative=5, n_redundant=5, n_samples=3000, sigma=0.1, seed=0)
    t = np.arange(len(y))
    folds = purged_kfold(t, t + 1, n_splits=5, embargo=0)
    return X, y, folds


def kind(names):
    return np.array([n[0] for n in names])


@pytest.mark.eval
def test_mdi_ranks_informative_and_redundant_above_noise(data):
    """Threshold: mean MDI of I and R features > 2x that of N features, and >= 9 of the top 10 are I or R."""
    X, y, _ = data
    rf = RandomForestClassifier(n_estimators=200, max_features=1, min_samples_leaf=20, random_state=0).fit(X, y)
    imp = fi.mdi(rf, X.columns)
    assert imp["mean"].sum() == pytest.approx(1.0)
    k = kind(imp.index)
    assert imp["mean"][k != "N"].mean() > 2 * imp["mean"][k == "N"].mean()
    top = kind(imp["mean"].sort_values(ascending=False).index[:10])
    assert np.sum(top != "N") >= 9
    assert (imp["std"] > 0).all()


@pytest.mark.eval
def test_mda_is_zero_for_noise_and_positive_for_signal(data):
    """Threshold: every N feature's MDA is within 3 standard errors + 0.002 of 0; the summed MDA of the
    I features is > 0.02 (log-loss scoring, logistic model so no feature is masked by a tree)."""
    X, y, folds = data
    res = fi.mda(LogisticRegression(max_iter=1000), X.to_numpy(), y, folds, names=X.columns, rng=0)
    imp = res["imp"]
    k = kind(imp.index)
    noise = imp[k == "N"]
    assert (noise["mean"].abs() < 3 * noise["std"] + 0.002).all()
    assert imp["mean"][k == "I"].sum() > 0.02
    assert res["score"] > np.log(0.5)                  # better than a coin


@pytest.mark.eval
def test_sfi_scores_signal_features_above_noise(data):
    """Threshold: the best N feature's out-of-fold log loss is worse than the median I/R feature's."""
    X, y, folds = data
    s = fi.sfi(LogisticRegression(max_iter=1000), X.to_numpy(), y, folds, names=X.columns)
    k = kind(s.index)
    assert s["mean"][k == "N"].max() < s["mean"][k != "N"].median()
    assert s["mean"][k == "N"].max() == pytest.approx(np.log(0.5), abs=0.02)


@pytest.mark.case
def test_correlation_clusters_put_each_redundant_feature_with_its_source(data):
    X, _, _ = data
    names = list(X.columns)
    res = fi.correlation_clusters(X.to_numpy(), names)
    where = {n: c for c, members in res["members"].items() for n in members}
    for r in [n for n in names if n.startswith("R_")]:
        src = X[[n for n in names if n.startswith("I_")]].corrwith(X[r]).idxmax()
        assert where[r] == where[src]


@pytest.mark.case
def test_correlation_clusters_with_no_scorable_cut_still_returns_clusters():
    """Two features leave range(2, max_k + 1) empty, and perfectly correlated features give one label at every cut; both
    used to end in TypeError ('NoneType' object is not iterable) because no k was ever scored."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    two = fi.correlation_clusters(np.column_stack([a, rng.normal(size=200)]), ["a", "b"])
    assert sorted(map(sorted, two["members"].values())) == [["a"], ["b"]]
    same = fi.correlation_clusters(np.column_stack([a, a]), ["a", "b"])
    assert list(same["members"].values()) == [["a", "b"]]
    X = np.column_stack([a, a, 2 * a])
    assert list(fi.correlation_clusters(X, ["a", "b", "c"])["members"].values()) == [["a", "b", "c"]]
    capped = fi.correlation_clusters(np.column_stack([a, rng.normal(size=200), rng.normal(size=200)]), list("abc"), max_k=1)
    assert list(capped["members"].values()) == [["a", "b", "c"]]
    assert np.isnan(capped["silhouette"])


@pytest.mark.eval
def test_clustered_mda_credits_the_signal_clusters(data):
    """Threshold: the top cluster by clustered MDA contains an I feature, and every noise-only cluster
    is within 3 standard errors + 0.002 of 0."""
    X, y, folds = data
    names = list(X.columns)
    cl = fi.correlation_clusters(X.to_numpy(), names)
    res = fi.clustered_mda(LogisticRegression(max_iter=1000), X.to_numpy(), y, folds, cl["clusters"], rng=0)
    imp = res["imp"].sort_values("mean", ascending=False)
    assert any(n.startswith("I_") for n in cl["members"][imp.index[0]])
    for c, members in cl["members"].items():
        if all(n.startswith("N_") for n in members):
            assert abs(imp.loc[c, "mean"]) < 3 * imp.loc[c, "std"] + 0.002


@pytest.mark.case
def test_correlation_clusters_recover_the_generating_groups(data):
    X, _, _ = data
    names = list(X.columns)
    res = fi.correlation_clusters(X.to_numpy(), names)
    groups = sorted(sorted(m) for m in res["members"].values())
    for n in names:
        if n.startswith("N_"):
            assert [n] in groups                       # noise stays alone
    I = X[[n for n in names if n.startswith("I_")]]
    expect = {}
    for r in [n for n in names if n.startswith("R_")]:
        expect.setdefault(I.corrwith(X[r]).idxmax(), []).append(r)
    for src, rs in expect.items():
        assert sorted([src] + rs) in groups


@pytest.mark.case
def test_mda_value_is_the_relative_log_loss_increase(data):
    X, y, folds = data
    from sklearn.metrics import log_loss
    tr, te = folds[0]
    Xa = X.to_numpy()
    res = fi.mda(LogisticRegression(max_iter=1000), Xa, y, [(tr, te)], names=X.columns, rng=0)
    m = LogisticRegression(max_iter=1000).fit(Xa[tr], y[tr])
    ll0 = log_loss(y[te], m.predict_proba(Xa[te]))
    rng = np.random.default_rng(0)
    for j, name in enumerate(X.columns):
        Xp = Xa[te].copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        ll1 = log_loss(y[te], m.predict_proba(Xp))
        assert res["imp"].loc[name, "mean"] == pytest.approx((ll1 - ll0) / ll1)
    assert res["score"] == pytest.approx(-ll0)


@pytest.mark.case
def test_clustered_mda_shuffles_each_column_of_a_cluster_independently(monkeypatch):
    """Code review afml.md, importance finding 1: MLAM 6.5 shuffles every column of a cluster on its own in the same
    pass, so two identical columns of one cluster stop being identical; other clusters stay as they are."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    X = np.column_stack([a, a, rng.normal(size=400)])
    y = (a > 0).astype(int)
    seen, real = [], fi._score
    monkeypatch.setattr(fi, "_score", lambda m, Xs, ys, w, s: seen.append(np.array(Xs)) or real(m, Xs, ys, w, s))
    fi.clustered_mda(LogisticRegression(), X, y, [(np.arange(200), np.arange(200, 400))], {"C0": [0, 1], "C1": [2]},
                     rng=1)
    base, cluster, other = seen
    np.testing.assert_array_equal(base, X[200:])
    assert not np.array_equal(cluster[:, 0], cluster[:, 1])
    for j in (0, 1):
        assert sorted(cluster[:, j]) == sorted(X[200:, j]) and not np.array_equal(cluster[:, j], X[200:, j])
    np.testing.assert_array_equal(cluster[:, 2], X[200:, 2])
    np.testing.assert_array_equal(other[:, :2], X[200:, :2])


@pytest.mark.case
def test_the_two_clustered_mda_definitions_agree_on_single_features_and_differ_on_a_cluster():
    """docs/afml_sections/ch08.md F8.3. 8.3.2's clustered MDA is implemented twice: importance.clustered_mda shuffles
    each column of a cluster on its own (MLAM 6.5), event_importance.permuted_losses moves a group's columns together
    under one row permutation, keeping the cluster's internal correlation. Both are read here as the book's relative
    drop (base - permuted) / -permuted: the same experiment for a one-column group, a different one for a correlated
    cluster."""
    from pmlab.afml import event_importance as ei

    rng = np.random.default_rng(0)
    n = 1200
    a = rng.normal(size=n)
    X = np.column_stack([a, a + rng.normal(0, 0.05, n), rng.normal(size=n)])
    y = (a + rng.normal(0, 0.5, n) > 0).astype(int)
    train, test = np.arange(n // 2), np.arange(n // 2, n)
    model = RandomForestClassifier(n_estimators=40, min_samples_leaf=20, random_state=0)
    fitted = model.fit(X[train], y[train])
    base = -ei.row_losses(fitted, X[test], y[test]).mean()

    def joint(cols, seed):
        losses = ei.permuted_losses(fitted, X[test], y[test], {"g": list(cols)}, seed)["g"]
        return fi._relative_drop(base, -losses.mean(), "neg_log_loss")

    def independent(cols, seed):
        return fi.clustered_mda(model, X, y, [(train, test)], {"g": list(cols)}, rng=seed)["imp"]["mean"]["g"]

    # a one-column group: both permute exactly that column across the rows, so both measure the same thing
    assert joint([0], 1) > 0.1 and independent([0], 1) > 0.1
    assert joint([0], 1) == pytest.approx(independent([0], 1), abs=0.05)
    # the correlated pair: only the independent shuffle also destroys the correlation between the two columns
    pair_joint, pair_independent = joint([0, 1], 2), independent([0, 1], 2)
    assert pair_joint > 0.1 and pair_independent > 0.1
    assert pair_joint != pytest.approx(pair_independent, rel=0.05)
    # the noise column ranks far below the signal under either definition
    assert joint([2], 3) < joint([0], 1) / 4 and independent([2], 3) < independent([0], 1) / 4


@pytest.mark.case
def test_mda_scores_a_test_fold_that_holds_one_class_with_the_models_classes():
    """7.5's scikit-learn 6231 work-around inside the importance scorer (docs/afml_sections/ch07.md F7.2): a fold whose
    test labels are all 1 must still be scored against the fitted model's two classes."""
    rng = np.random.default_rng(4)
    n = 400
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)
    y[-100:] = 1
    folds = [(np.arange(0, n - 100), np.arange(n - 100, n))]
    out = fi.mda(LogisticRegression(max_iter=500), X, y, folds, names=["signal", "noise"])
    assert len(np.unique(y[folds[0][1]])) == 1
    fitted = LogisticRegression(max_iter=500).fit(X[folds[0][0]], y[folds[0][0]])
    expected = -log_loss(y[folds[0][1]], fitted.predict_proba(X[folds[0][1]]), labels=[0, 1])
    assert out["score"] == pytest.approx(expected)
    assert np.all(np.isfinite(out["imp"]["mean"].to_numpy()))


@pytest.fixture(scope="module")
def substitutes():
    """One informative column, an exact copy of it, and two noise columns, on 2,000 rows."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    y = (a + rng.normal(0, 0.6, n) > 0).astype(int)
    frame = pd.DataFrame({"a": a, "a_copy": a.copy(), "n1": rng.normal(size=n), "n2": rng.normal(size=n)})
    return frame, y


def _forest_mdi(frame, y, columns, seed=0):
    rf = RandomForestClassifier(n_estimators=300, max_features=1, min_samples_leaf=20,
                                random_state=seed).fit(frame[columns], y)
    return fi.mdi(rf, columns)["mean"]


@pytest.mark.eval
def test_two_identical_features_split_the_importance_that_one_of_them_gets_alone(substitutes):
    """8.3.1: MDI dilutes substitutes, and the importance of two identical features is halved because a tree picks
    between them with equal probability. Threshold: with max_features = 1, the copy's importance and the original's
    are each between 0.4 and 0.6 of what the original alone scores, and together they come within 10% of it."""
    frame, y = substitutes
    alone = _forest_mdi(frame, y, ["a", "n1", "n2"])["a"]
    both = _forest_mdi(frame, y, ["a", "a_copy", "n1", "n2"])
    assert alone > 0.5
    for name in ("a", "a_copy"):
        assert 0.4 * alone < both[name] < 0.6 * alone
    assert both["a"] + both["a_copy"] == pytest.approx(alone, rel=0.1)


@pytest.mark.eval
def test_mdi_gives_every_feature_importance_even_with_random_labels_and_prefers_the_one_with_more_values():
    """8.3.1's in-sample nature and the Strobl et al. bias. With labels that no feature predicts, MDI still sums to 1
    and gives every column a share (threshold: each above 0.1 of the total over 5 columns). With a real signal
    present, a continuous noise column takes at least five times the importance of a binary noise column, which is
    the bias towards predictors with many distinct values."""
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 2000
    noise = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(5)})
    random_labels = rng.integers(0, 2, n)
    imp = _forest_mdi(noise, random_labels, list(noise.columns))
    assert imp.sum() == pytest.approx(1.0)
    assert imp.min() > 0.1 and (imp > 0).all()
    a = rng.normal(size=n)
    y = (a + rng.normal(0, 0.6, n) > 0).astype(int)
    mixed = pd.DataFrame({"signal": a, "cont": rng.normal(size=n), "binary": rng.integers(0, 2, n).astype(float)})
    biased = _forest_mdi(mixed, y, list(mixed.columns))
    assert biased["cont"] > 5 * biased["binary"]
    assert biased["signal"] > biased["cont"]


@pytest.mark.eval
def test_mda_underrates_a_feature_whose_substitute_is_left_in_place(substitutes):
    """8.3.2: shuffling one of two identical features still lets the other one contribute, so both look less
    important than they are. Threshold: with the copy present, each of the pair scores between 0.4 and 0.65 of the
    MDA the single feature scores on its own, on the same purged folds and with a model that cannot mask."""
    frame, y = substitutes
    t = np.arange(len(y))
    folds = purged_kfold(t, t + 1, n_splits=5, embargo=0)
    model = LogisticRegression(max_iter=1000)
    alone = fi.mda(model, frame[["a", "n1"]].to_numpy(), y, folds, names=["a", "n1"], rng=0)["imp"]["mean"]["a"]
    pair = fi.mda(model, frame[["a", "a_copy", "n1"]].to_numpy(), y, folds, names=["a", "a_copy", "n1"],
                  rng=0)["imp"]["mean"]
    assert alone > 0.5
    for name in ("a", "a_copy"):
        assert 0.4 * alone < pair[name] < 0.65 * alone


@pytest.mark.case
def test_a_feature_that_reverses_out_of_sample_gets_a_negative_mda():
    """8.3.2's closing note: the improvement from permuting a feature may be negative, meaning the feature is
    detrimental to the forecast. A column that carries the label with one sign in the training half and the opposite
    sign in the test half must score below zero while the honest feature scores above it."""
    rng = np.random.default_rng(5)
    n = 2000
    a = rng.normal(size=n)
    y = (a + rng.normal(0, 0.6, n) > 0).astype(int)
    sign = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    reversing = sign * np.where(y > 0, 1.0, -1.0) * np.abs(rng.normal(size=n))
    folds = [(np.arange(0, n // 2), np.arange(n // 2, n))]
    imp = fi.mda(LogisticRegression(max_iter=1000), np.column_stack([a, reversing]), y, folds,
                 names=["honest", "reversing"], rng=0)["imp"]["mean"]
    assert imp["reversing"] < -0.5
    assert imp["honest"] > 0.1


@pytest.mark.eval
def test_single_feature_importance_misses_a_feature_that_is_only_useful_beside_another():
    """8.4.1's main limitation: joint effects are lost, because SFI scores each feature on its own. The label is the
    exclusive-or of two features, so neither says anything alone. Threshold: every SFI score, signal or noise, is
    within 0.05 of log(1/2) and none of them beats it; a model on both features together scores at least 0.3 nats
    better, and MDA credits both above 0.5 while the noise column stays under 0.05."""
    rng = np.random.default_rng(6)
    n = 2400
    x0, x1, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = ((x0 > 0) ^ (x1 > 0)).astype(int)
    X = np.column_stack([x0, x1, noise])
    t = np.arange(n)
    folds = purged_kfold(t, t + 1, n_splits=5, embargo=0)
    model = RandomForestClassifier(n_estimators=60, min_samples_leaf=20, random_state=0)
    alone = fi.sfi(model, X, y, folds, names=["x0", "x1", "noise"])["mean"]
    assert (np.abs(alone - np.log(0.5)) < 0.05).all()
    assert (alone <= np.log(0.5)).all()
    together = fi.mda(model, X, y, folds, names=["x0", "x1", "noise"], rng=0)
    assert together["score"] > np.log(0.5) + 0.3
    assert together["imp"]["mean"]["x0"] > 0.5 and together["imp"]["mean"]["x1"] > 0.5
    assert together["imp"]["mean"]["noise"] < 0.05


@pytest.mark.eval
def test_mdi_on_orthogonal_features_follows_the_label_free_pca_ranking_only_when_the_labels_mean_something():
    """8.4.2's confirmatory evidence: PCA ranks the features without ever seeing a label, so agreement between that
    ranking and MDI cannot have been overfit; with random labels the two rankings should have no correspondence.
    Threshold on the 40-feature synthetic set (27 components at 95% of the variance): the hyperbolic weighted tau
    between MDI and the inverse eigenvalue rank is above 0.4 for each of three forest seeds and below 0.35 in
    absolute value for each of three label permutations."""
    from pmlab.afml import event_importance as ei
    X, y = fi.synthetic_data(n_features=40, n_informative=10, n_redundant=10, n_samples=4000, sigma=0.1, seed=0)
    fit = ei.pca_fit(X.to_numpy(float), var_share=0.95)
    P = ei.pca_transform(fit, X.to_numpy(float))
    names = [f"P{i}" for i in range(P.shape[1])]
    rank = np.arange(1, P.shape[1] + 1)                      # rank 1 is the largest eigenvalue

    def tau(labels, seed):
        rf = RandomForestClassifier(n_estimators=150, max_features=1, min_samples_leaf=20,
                                    random_state=seed).fit(P, labels)
        return ei.weighted_tau(fi.mdi(rf, names)["mean"].to_numpy(), rank)

    assert P.shape[1] > 20 and fit["explained"] >= 0.95
    assert all(tau(y, seed) > 0.4 for seed in range(3))
    assert all(abs(tau(np.random.default_rng(seed).permutation(y), 0)) < 0.35 for seed in range(3))


@pytest.mark.case
def test_the_weighted_tau_punishes_a_swap_among_the_important_features_far_more_than_one_among_the_rest():
    """8.4.2's reason for preferring a weighted Kendall's tau: concordance among the most important features is what
    matters. On a perfect ranking of ten features, swapping the top two must cost several times what swapping the
    bottom two costs, while the unweighted Kendall's tau cannot tell the two swaps apart."""
    from scipy.stats import kendalltau
    from pmlab.afml import event_importance as ei
    rank = np.arange(1, 11)
    imp = 1.0 / rank                                         # importance perfectly following the PCA rank
    swap = lambda i, j: np.concatenate([imp[:i], [imp[j]], imp[i + 1:j], [imp[i]], imp[j + 1:]])
    assert ei.weighted_tau(imp, rank) == pytest.approx(1.0)
    top, bottom = ei.weighted_tau(swap(0, 1), rank), ei.weighted_tau(swap(8, 9), rank)
    assert 1.0 - top > 5 * (1.0 - bottom)
    assert kendalltau(swap(0, 1), 1.0 / rank)[0] == pytest.approx(kendalltau(swap(8, 9), 1.0 / rank)[0])


@pytest.mark.eval
def test_stacking_the_instruments_gives_a_steadier_importance_than_averaging_one_per_instrument():
    """8.5: the parallelised approach lets important features swap ranks across instruments, which raises the
    variance of the per-instrument importance; stacking fits one classifier on the whole universe instead.
    Threshold: on ten 300-row instruments from the same generator, the mean standard deviation of a feature's MDI
    across instruments is at least three times its standard deviation across ten resamples of the 3,000-row stacked
    dataset, and the stacked ranking puts all eight signal features in its top eight where the per-instrument
    rankings do not."""
    import pandas as pd
    parts = [fi.synthetic_data(n_features=12, n_informative=4, n_redundant=4, n_samples=300, sigma=0.1, seed=s)
             for s in range(10)]
    columns = list(parts[0][0].columns)
    model = lambda seed: RandomForestClassifier(n_estimators=120, max_features=1, min_samples_leaf=10,
                                                random_state=seed)

    def importance(frame, labels, seed=0):
        return fi.mdi(model(seed).fit(frame[columns], labels), columns)["mean"]

    def signal_recall(series):
        return np.mean([not str(name).startswith("N") for name in series.sort_values(ascending=False).index[:8]])

    per_instrument = pd.DataFrame([importance(X, y).to_numpy() for X, y in parts], columns=columns)
    stacked_X = pd.concat([X for X, _ in parts], ignore_index=True)
    stacked_y = np.concatenate([y for _, y in parts])
    stacked = importance(stacked_X, stacked_y)
    resamples = []
    for seed in range(10):
        idx = np.random.default_rng(seed).choice(len(stacked_y), len(stacked_y), replace=True)
        resamples.append(importance(stacked_X.iloc[idx].reset_index(drop=True), stacked_y[idx]).to_numpy())
    across_instruments = float(per_instrument.std().mean())
    across_resamples = float(pd.DataFrame(resamples, columns=columns).std().mean())
    assert across_instruments > 3 * across_resamples
    assert signal_recall(stacked) == 1.0
    assert np.mean([signal_recall(per_instrument.loc[i]) for i in per_instrument.index]) < 1.0


# ------------------------------------------------------------------ the second reading (plan book-v2 node 16)


def _masking_frame(seed=7, n=2000):
    """One informative column and a degraded copy of it -- the same information, slightly noisier -- plus noise."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    y = (a + rng.normal(0, 0.6, n) > 0).astype(int)
    return pd.DataFrame({"a": a, "degraded": a + rng.normal(0, 0.35, n), "noise": rng.normal(size=n)}), y


@pytest.mark.eval
def test_a_feature_is_masked_unless_only_one_feature_is_considered_per_split():
    """8.3.1's masking effect, which no test measured: when every feature competes at every node, the degraded copy
    of an informative column is systematically passed over and reads as unimportant. Threshold: with max_features at
    the default it scores below 0.05 while the column it copies scores above 0.9, and more than a fifth of the trees
    never split on it at all; with max_features = 1 it scores above 0.4 and every tree uses it."""
    frame, y = _masking_frame()
    columns = list(frame.columns)

    def fit(max_features):
        rf = RandomForestClassifier(n_estimators=200, max_features=max_features, max_depth=3, min_samples_leaf=20,
                                    random_state=0).fit(frame[columns], y)
        raw = np.array([t.feature_importances_ for t in rf.estimators_])
        return fi.mdi(rf, columns)["mean"], float(np.mean(raw[:, 1] == 0))

    masked, never_used = fit(None)
    fair, never_used_fair = fit(1)
    assert masked["degraded"] < 0.05 < 0.9 < masked["a"]
    assert never_used > 0.2                                  # systematically ignored, not occasionally
    assert fair["degraded"] > 0.4 and never_used_fair < 0.1   # every feature gets its chance at some levels
    assert fair["noise"] > masked["noise"]


@pytest.mark.case
def test_mdi_reads_a_zero_as_not_sampled_scales_the_error_by_the_tree_count_and_needs_a_forest():
    """8.3.1's three mechanical claims about Snippet 8.2, none of which had a test that could fail: zeros are not
    averaged (they only mean the feature was never drawn, so they become missing values), the reported error is the
    standard deviation over the square root of the tree count, and the means are normalised to sum to one with each
    between zero and one. scikit-learn's own feature_importances_ is the plain average that includes the zeros, which
    is what makes the difference measurable. MDI needs a tree ensemble: it cannot be read off a linear model."""
    frame, y = _masking_frame()
    columns = list(frame.columns)
    rf = RandomForestClassifier(n_estimators=200, max_features=None, max_depth=3, min_samples_leaf=20,
                                random_state=0).fit(frame[columns], y)
    raw = np.array([t.feature_importances_ for t in rf.estimators_], dtype=float)
    assert (raw == 0).any()                                  # the case the rule is about

    imp = fi.mdi(rf, columns)
    skipped = np.where(raw == 0, np.nan, raw)
    mean = np.nanmean(skipped, axis=0)
    std = np.nanstd(skipped, axis=0, ddof=1) / np.sqrt(len(raw))
    np.testing.assert_allclose(imp["mean"].to_numpy(), mean / np.nansum(mean))
    np.testing.assert_allclose(imp["std"].to_numpy(), std / np.nansum(mean))
    assert imp["mean"].sum() == pytest.approx(1.0)
    assert ((imp["mean"] >= 0) & (imp["mean"] <= 1)).all()

    plain = raw.mean(axis=0)                                 # what averaging the zeros in would give
    np.testing.assert_allclose(rf.feature_importances_, plain, atol=1e-12)   # sklearn's default, computed on the fly
    degraded = columns.index("degraded")
    zero_share = float(np.mean(raw[:, degraded] == 0))
    assert np.nanmean(skipped[:, degraded]) == pytest.approx(plain[degraded] / (1 - zero_share))
    assert imp["mean"].iloc[degraded] > 1.3 * (plain / plain.sum())[degraded]   # the zeros were dragging it down

    with pytest.raises(AttributeError):                      # not a tree ensemble: no per-tree impurity decrease
        fi.mdi(LogisticRegression(max_iter=200).fit(frame[columns], y), columns)


@pytest.mark.case
def test_the_mda_improvement_is_relative_to_the_best_possible_score_under_accuracy_too():
    """8.3.2 and Snippet 8.3: the importance is the improvement from permuting to not permuting, relative to the best
    score attainable -- a negative log loss of zero, or an accuracy of one. Only the log-loss branch was ever run, so
    the accuracy branch could have divided by the wrong quantity and stayed green. Checked against the book's
    arithmetic by hand, and shown to differ from the log-loss branch's denominator."""
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(8)
    n = 2000
    a = rng.normal(size=n)
    y = (a + rng.normal(0, 0.6, n) > 0).astype(int)
    X = np.column_stack([a, rng.normal(size=n)])
    train, test = np.arange(n // 2), np.arange(n // 2, n)
    model = LogisticRegression(max_iter=1000)
    got = fi.mda(model, X, y, [(train, test)], scoring="accuracy", names=["a", "noise"], rng=0)

    fitted = LogisticRegression(max_iter=1000).fit(X[train], y[train])
    base = accuracy_score(y[test], fitted.predict(X[test]))
    assert got["score"] == pytest.approx(base)
    shuffle = np.random.default_rng(0)
    for j, name in enumerate(["a", "noise"]):
        permuted = X[test].copy()
        permuted[:, j] = shuffle.permutation(permuted[:, j])
        after = accuracy_score(y[test], fitted.predict(permuted))
        assert got["imp"].loc[name, "mean"] == pytest.approx((base - after) / (1.0 - after))
        assert got["imp"].loc[name, "mean"] != pytest.approx((base - after) / -after)     # the log-loss denominator
    assert got["imp"].loc["a", "mean"] > 0.5
    assert abs(got["imp"].loc["noise", "mean"]) < 0.05
    assert fi._relative_drop(-0.4, -0.5, "neg_log_loss") == pytest.approx(0.2)
    assert fi._relative_drop(0.9, 0.5, "accuracy") == pytest.approx(0.8)


@pytest.mark.eval
def test_dropping_the_small_eigenvalues_reduces_the_dimension_and_speeds_the_fit_up():
    """8.4.2's first additional benefit: orthogonalisation also reduces the dimensionality of the feature matrix by
    dropping the components with small eigenvalues, and that usually speeds up an algorithm's convergence. Threshold
    on the 40-feature synthetic set: 27 components carry 95% of the variance, and the same solver at the same
    tolerance converges in at least three times fewer iterations on them than on the 40 raw features."""
    from pmlab.afml import event_importance as ei

    X, y = fi.synthetic_data(n_features=40, n_informative=10, n_redundant=10, n_samples=4000, sigma=0.1, seed=0)
    raw = X.to_numpy(float)
    fit = ei.pca_fit(raw, var_share=0.95)
    components = ei.pca_transform(fit, raw).astype(float)
    assert components.shape[1] < raw.shape[1] and fit["explained"] >= 0.95

    iterations = lambda M: int(LogisticRegression(max_iter=5000, tol=1e-6).fit(M, y).n_iter_[0])
    assert iterations(raw) > 3 * iterations(components)
