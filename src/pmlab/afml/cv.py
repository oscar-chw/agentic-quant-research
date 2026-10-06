"""Cross-validation when labels overlap in time (AFML ch.7): purging, embargo, a purged k-fold splitter usable
wherever sklearn takes a `cv`, and a CV score that applies sample weights to both the fit and the score.

Spans are half-open [t0, t1) as everywhere in pmlab (pmlab.labels, pmlab.afml.sampling, pmlab.afml.cpcv): a label
decided at time e has t1 = e + one unit. The book writes a label's end as the time of its last observation and compares
with <= and >=; for integer units that end is t1 - 1 here, and the book's comparisons become the conditions below.

- train_times: purging (7.4.1, Snippet 7.1). A training observation is dropped when its span overlaps a test span,
  which covers the book's three cases (it starts inside, ends inside, or envelops the test span).
- embargo_times: for each observation time, the time `int(n x pct)` observations later, the last ones mapped to the
  final time (7.4.2, Snippet 7.2). Extending a test span's end to the embargo time of its last observation embargoes
  the training observations that follow it.
- PurgedKFold: contiguous test blocks in row order (rows sorted by t0), training rows that end at or before the block
  starts plus rows that start at or after the block's latest end, skipping the first int(n x pct_embargo) of those
  (7.4.3, Snippet 7.3). With no embargo the folds equal pmlab.labels.purged_kfold's.
- cv_score: out-of-fold scores with the sample weights passed to fit and to the score (7.5, Snippet 7.4); sklearn's
  cross_val_score weights only the fit, which the book calls out as a bug. Log loss is scored with the fitted
  model's classes, as in the snippet; accuracy and F1 (which ch.9 uses for meta-labels) are also available.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import BaseCrossValidator

SCORINGS = ("neg_log_loss", "accuracy", "f1")


def _spans(t0, t1) -> tuple[np.ndarray, np.ndarray]:
    t0, t1 = np.asarray(t0, dtype=float), np.asarray(t1, dtype=float)
    if t0.shape != t1.shape:
        raise ValueError("t0 and t1 differ in shape")
    if np.any(t1 <= t0):
        raise ValueError("every span needs t1 > t0")
    return t0, t1


def train_times(t0, t1, test_t0, test_t1) -> np.ndarray:
    """Boolean mask of the observations kept for training: no overlap with any test span [test_t0, test_t1)."""
    t0, t1 = _spans(t0, t1)
    keep = np.ones(len(t0), dtype=bool)
    for a, b in zip(np.atleast_1d(test_t0), np.atleast_1d(test_t1)):
        keep &= ~((t0 < b) & (t1 > a))
    return keep


def embargo_times(times, pct: float) -> np.ndarray:
    t = np.asarray(times)
    step = int(len(t) * pct)
    if step == 0:
        return t.copy()
    return np.concatenate([t[step:], np.repeat(t[-1:], step)])


class PurgedKFold(BaseCrossValidator):
    """sklearn splitter over rows sorted by t0 (module docstring)."""

    def __init__(self, n_splits: int = 3, t0=None, t1=None, pct_embargo: float = 0.0):
        if t0 is None or t1 is None:
            raise ValueError("PurgedKFold needs the label spans t0 and t1")
        self.t0, self.t1 = _spans(t0, t1)
        if np.any(np.diff(self.t0) < 0):
            raise ValueError("rows must be sorted by t0")
        self.n_splits = int(n_splits)
        self.pct_embargo = float(pct_embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(self.t0)
        if X is not None and len(X) != n:
            raise ValueError("X and the label spans must have the same rows")
        rows = np.arange(n)
        embargo = int(n * self.pct_embargo)
        for block in np.array_split(rows, self.n_splits):
            if not len(block):
                continue
            start, end = self.t0[block[0]], self.t1[block].max()
            left = rows[self.t1 <= start]
            first_right = int(np.searchsorted(self.t0, end, side="left"))
            right = rows[first_right + embargo:]
            yield np.concatenate([left, right]), block

    def _iter_test_indices(self, X=None, y=None, groups=None):          # required by BaseCrossValidator
        for _, test in self.split(X):
            yield test


def _score(model, X, y, w, scoring: str) -> float:
    if scoring == "neg_log_loss":
        return -log_loss(y, model.predict_proba(X), sample_weight=w, labels=model.classes_)
    if scoring == "accuracy":
        return accuracy_score(y, model.predict(X), sample_weight=w)
    return f1_score(y, model.predict(X), sample_weight=w)


def cv_score(model, X, y, sample_weight=None, scoring: str = "neg_log_loss", cv=None) -> np.ndarray:
    """One score per fold. `cv` is a splitter with split(X) (e.g. PurgedKFold) or a list of (train, test) arrays."""
    if scoring not in SCORINGS:
        raise ValueError(f"scoring must be one of {SCORINGS}")
    if cv is None:
        raise ValueError("cv_score needs purged folds: pass a PurgedKFold or a list of (train, test)")
    X, y = np.asarray(X), np.asarray(y)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    folds = cv.split(X) if hasattr(cv, "split") else cv
    out = []
    for train, test in folds:
        m = clone(model)
        m = m.fit(X[train], y[train]) if w is None else m.fit(X[train], y[train], sample_weight=w[train])
        out.append(_score(m, X[test], y[test], None if w is None else w[test], scoring))
    return np.asarray(out)
