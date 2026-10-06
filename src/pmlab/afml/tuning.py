"""Hyper-parameter search with purged cross-validation (AFML ch.9).

- WeightedPipeline: an sklearn Pipeline whose fit accepts sample_weight and hands it to the last step, so the pipeline
  can sit inside a BaggingClassifier or any code that passes weights by that name (9.2, Snippet 9.2).
- purged_search: every candidate scored by pmlab.afml.cv.cv_score on PurgedKFold folds, the best mean score refitted on
  all rows, optionally bagged (9.2-9.3, Snippets 9.1 and 9.3). n_iter = 0 searches the whole grid; n_iter > 0 draws that
  many candidates from lists or scipy distributions (randomised search). The book runs sklearn's GridSearchCV and
  RandomizedSearchCV, whose scorers do not see the sample weights unless metadata routing is switched on; scoring
  through cv_score weights the fit and the score in every fold, which is what 7.5 asks for.
- book_scoring: the book's default, F1 for {0, 1} meta-labels and negative log loss otherwise (9.4). purged_search takes
  it when scoring="book"; this project scores meta-labels by log loss because sizes need calibrated probabilities
  (research/book_check_part2.md, deviations table).
- log_uniform: a random variable whose log is uniform on [log a, log b], CDF log(x / a) / log(b / a) whatever the
  base of the logarithm (9.3.1, Snippet 9.4); scipy's loguniform is that distribution.
"""
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline

from pmlab.afml.cv import PurgedKFold, cv_score


class WeightedPipeline(Pipeline):
    """A Pipeline whose fit accepts sample_weight and passes it to the final estimator only (9.2)."""

    def fit(self, X, y=None, sample_weight=None, **params):
        if sample_weight is not None:
            params[f"{self.steps[-1][0]}__sample_weight"] = sample_weight
        return super().fit(X, y, **params)


def log_uniform(a: float, b: float):
    """Frozen distribution of x with log x uniform on [log a, log b] (9.3.1): CDF log(x / a) / log(b / a) and density
    1 / (x log(b / a)) on [a, b]. scipy.stats.loguniform is this distribution."""
    if not 0 < a < b:
        raise ValueError("log_uniform needs 0 < a < b")
    return stats.loguniform(a, b)


def book_scoring(y) -> str:
    return "f1" if set(np.unique(y).tolist()) == {0, 1} else "neg_log_loss"


def purged_search(model, space, X, y, t0, t1, n_splits: int = 3, pct_embargo: float = 0.0, sample_weight=None,
                  scoring: str = "neg_log_loss", n_iter: int = 0, bagging=None, rng=None) -> dict:
    """{"best_params", "best_score", "results": [(params, fold scores)], "model": the refitted best (or its bagging)}.

    `space` is a dict of lists (grid) or of lists and scipy distributions (n_iter > 0). `bagging` = (n_estimators,
    max_samples, max_features) wraps the best model in a BaggingClassifier fitted with the sample weights; a Pipeline
    is rebuilt as a WeightedPipeline so the weights reach its last step in every fold and in the refit. Rows must be
    sorted by t0. `bagging[1]` (max_samples) is a share of the ROWS here: sklearn's BaggingClassifier (1.9) draws rows
    with probability proportional to the sample weights and reads a fractional max_samples in units of their sum, so
    raw average uniqueness (mean well below 1) would fit each estimator on a u times smaller draw than the book's
    max_samples = average uniqueness. The share is therefore turned into a row count before it reaches sklearn, which
    makes the draw independent of how the weights are scaled (docs/afml_sections/ch09.md F9.1)."""
    X, y = np.asarray(X), np.asarray(y)
    if isinstance(model, Pipeline) and not isinstance(model, WeightedPipeline):
        model = WeightedPipeline(model.steps)
    scoring = book_scoring(y) if scoring == "book" else scoring
    folds = list(PurgedKFold(n_splits, t0, t1, pct_embargo).split(X))
    if n_iter:
        seed = None if rng is None else int(np.random.default_rng(rng).integers(2**31 - 1))
        candidates = list(ParameterSampler(space, n_iter, random_state=seed))
    else:
        candidates = list(ParameterGrid(space))
    results = []
    for params in candidates:
        scores = cv_score(clone(model).set_params(**params), X, y, sample_weight, scoring, folds)
        results.append((params, scores))
    best = int(np.argmax([s.mean() for _, s in results]))
    best_params = results[best][0]
    fitted = clone(model).set_params(**best_params)
    if bagging is not None and bagging[1] and bagging[1] > 0:
        n_est, max_samples, max_features = bagging
        n_rows = max(1, int(round(float(max_samples) * len(y)))) if max_samples <= 1 else int(max_samples)
        fitted = BaggingClassifier(estimator=fitted, n_estimators=int(n_est), max_samples=n_rows,
                                   max_features=float(max_features),
                                   random_state=None if rng is None else int(np.random.default_rng(rng).integers(2**31 - 1)))
    fitted = fitted.fit(X, y) if sample_weight is None else fitted.fit(X, y, sample_weight=np.asarray(sample_weight))
    return {"best_params": best_params, "best_score": float(results[best][1].mean()), "results": results,
            "scoring": scoring, "model": fitted}
