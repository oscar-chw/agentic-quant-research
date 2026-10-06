"""Feature importance (AFML ch.8).

- mdi: mean decrease impurity from a fitted tree ensemble, averaged over trees with zero
  importances treated as "not sampled" (the tree never saw the feature), normalised to sum to 1,
  with the standard error of the mean (8.3.1). Use trees with max_features=1 so a feature cannot be
  masked by a correlated one.
- mda: mean decrease accuracy on purged out-of-fold data: the score drop after permuting one feature
  in the test fold, relative to the permuted score (8.3.2). Scoring is log loss by default, as in
  the book, with sample weights in fit and score.
- sfi: single feature importance, the out-of-fold score of a model fitted on one feature at a time
  (8.4.1).
- clustered_mda: MDA that permutes every feature of a cluster in the same pass, so substitutes cannot
  hide each other; each column of the cluster is shuffled independently, as in López de Prado (2020),
  Machine Learning for Asset Managers, section 6.5 (not AFML 2018), which also breaks the links between
  the cluster's own columns. The clusters come from hierarchical clustering on correlation distance with k chosen
  by silhouette, a simpler stand-in for MLAM's ONC (see correlation_clusters).
- synthetic_data: informative, redundant (informative plus noise) and noise features for the
  known-answer test (8.6).

Folds are (train, test) index arrays, e.g. pmlab.labels.purged_kfold, so purging and embargo are
decided by the caller.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss


def mdi(forest, names) -> pd.DataFrame:
    imps = np.array([t.feature_importances_ for t in forest.estimators_], dtype=float)
    imps[imps == 0] = np.nan
    mean = np.nanmean(imps, axis=0)
    std = np.nanstd(imps, axis=0, ddof=1) / np.sqrt(len(imps))   # the book scales by the tree count
    total = np.nansum(mean)
    return pd.DataFrame({"mean": mean / total, "std": std / total}, index=list(names))


def _score(model, X, y, w, scoring):
    if scoring == "neg_log_loss":
        prob = model.predict_proba(X)
        return -log_loss(y, prob, sample_weight=w, labels=model.classes_)
    return accuracy_score(y, model.predict(X), sample_weight=w)


def _fit(model, X, y, w):
    m = clone(model)
    if w is None:
        return m.fit(X, y)
    return m.fit(X, y, sample_weight=w)


def _relative_drop(base, permuted, scoring):
    """(base - permuted) / -permuted for log loss, / (1 - permuted) for accuracy (8.3.2)."""
    drop = base - permuted
    return drop / -permuted if scoring == "neg_log_loss" else drop / (1.0 - permuted)


def mda(model, X, y, folds, w=None, scoring: str = "neg_log_loss", groups=None, names=None, rng=None) -> dict:
    """Permutation importance per feature (or per group of features when `groups` maps a name to column
    indices) on each fold's test set. Returns {"imp": DataFrame(mean, std), "score": mean base score}."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(rng)
    if groups is None:
        names = list(names) if names is not None else [f"f{j}" for j in range(X.shape[1])]
        groups = {nm: [j] for j, nm in enumerate(names)}
    base, rows = [], []
    for tr, te in folds:
        wtr = None if w is None else np.asarray(w)[tr]
        wte = None if w is None else np.asarray(w)[te]
        m = _fit(model, X[tr], y[tr], wtr)
        s0 = _score(m, X[te], y[te], wte, scoring)
        base.append(s0)
        row = {}
        for nm, cols in groups.items():
            Xp = X[te].copy()
            for j in cols:
                Xp[:, j] = rng.permutation(Xp[:, j])
            row[nm] = _relative_drop(s0, _score(m, Xp, y[te], wte, scoring), scoring)
        rows.append(row)
    df = pd.DataFrame(rows)
    return {"imp": pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=1) / np.sqrt(len(df))}),
            "score": float(np.mean(base)), "fold_scores": base}


def sfi(model, X, y, folds, w=None, scoring: str = "neg_log_loss", names=None) -> pd.DataFrame:
    """Out-of-fold score of a model on each feature alone (mean and standard error over folds)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    names = list(names) if names is not None else [f"f{j}" for j in range(X.shape[1])]
    out = {}
    for j, nm in enumerate(names):
        scores = []
        for tr, te in folds:
            wtr = None if w is None else np.asarray(w)[tr]
            wte = None if w is None else np.asarray(w)[te]
            m = _fit(model, X[tr][:, [j]], y[tr], wtr)
            scores.append(_score(m, X[te][:, [j]], y[te], wte, scoring))
        out[nm] = (np.mean(scores), np.std(scores, ddof=1) / np.sqrt(len(scores)))
    return pd.DataFrame(out, index=["mean", "std"]).T


def correlation_clusters(X, names, max_k: int | None = None) -> dict:
    """Clusters of features on the distance sqrt((1 - rho) / 2): average-linkage hierarchical clustering,
    cut at the k (2..max_k) with the highest mean silhouette on the precomputed distances. When no k can be scored
    (two features, max_k = 1, or perfectly correlated features) the silhouette is nan and two features are split
    unless identical, anything else is one cluster.

    MLAM 2020 (section 4.4) uses ONC: k-means on the distance matrix, k by mean / sd of silhouettes,
    then recursive re-clustering of weak clusters. On the 8.6 synthetic set one ONC pass picks k = 2 and
    lumps noise with signal, while this cut recovers each informative feature with its redundant copies
    and leaves noise as singletons (tests/test_afml_importance.py)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_samples
    names = list(names)
    Xf = pd.DataFrame(np.asarray(X, dtype=float), columns=names)
    corr = Xf.corr().fillna(0.0).to_numpy().copy()
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), "average")
    n = len(names)
    max_k = min(max_k or n - 1, n - 1)
    best, best_q = None, -np.inf
    for k in range(2, max_k + 1):
        lab = fcluster(Z, k, "maxclust")
        if len(set(lab)) < 2:
            continue
        q = silhouette_samples(dist, lab, metric="precomputed").mean()
        if q > best_q:
            best, best_q = lab, q
    if best is None:                                 # no cut could be scored: 2 features, max_k=1, or one label at every cut
        split = n == 2 and dist[0, 1] > 1e-6          # 2 features split unless identical; otherwise one cluster
        best = np.array([1, 2]) if split else np.ones(n, dtype=int)
        best_q = np.nan
    clusters = {}
    for j, lab in enumerate(best):
        clusters.setdefault(f"C{int(lab)}", []).append(j)
    return {"clusters": clusters, "silhouette": float(best_q),
            "members": {c: [names[j] for j in v] for c, v in clusters.items()}}


def clustered_mda(model, X, y, folds, clusters: dict, w=None, scoring: str = "neg_log_loss", rng=None) -> dict:
    """MDA with every feature of a cluster permuted in the same pass, each column shuffled independently (MLAM 2020,
    section 6.5)."""
    return mda(model, X, y, folds, w=w, scoring=scoring, groups=clusters, rng=rng)


def synthetic_data(n_features: int = 20, n_informative: int = 5, n_redundant: int = 5, n_samples: int = 5000,
                   sigma: float = 0.1, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Columns I_k (informative), R_k (a random informative column plus N(0, sigma) noise) and N_k
    (noise), as in the 8.6 experiment. Rows are shuffled: make_classification without shuffling
    orders samples by class, which contiguous (purged) folds would turn into a class shift."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features - n_redundant,
                               n_informative=n_informative, n_redundant=0, random_state=seed, shuffle=False)
    names = [f"I_{i}" for i in range(n_informative)] + [f"N_{i}" for i in range(n_features - n_informative - n_redundant)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    X, y = X[order], y[order]
    df = pd.DataFrame(X, columns=names)
    src = rng.choice(n_informative, n_redundant)
    for k, j in enumerate(src):
        df[f"R_{k}"] = df[f"I_{j}"] + rng.normal(0, sigma, n_samples)
    return df, y
