"""Meta-label models by the book (plan afml-pipeline node 11; AFML ch. 6, 7, 9): four model families behind one
interface, a purged hyper-parameter search that treats the refit schedule as one more dimension, and out-of-fold
probabilities aligned to event rows for bet sizing (stage 5) and backtests (stage 6).

Inputs. One primary rule at a time from the event dataset (pmlab.afml.events, research/afml_events.md). The design
matrix is `f_` columns only (feature_columns); labels, spans, weights and bet columns never enter it. No row of a
window ending after pmlab.evaluation.CONFIRM_START is loaded (load_events) or accepted (Dataset.from_rows).

Families (make with MetaModel(family, params)).
- bagged_trees, the book's default (6.3.3, 6.4): a BaggingClassifier over entropy trees (`base="tree"`) or over
  one-tree random forests without their own bootstrap (`base="forest"`). max_samples is the mean average uniqueness of
  the training rows, so each tree draws about as many rows as the training set holds independent labels. Fit weights
  set the draw probabilities (sklearn draws max_samples x sum of weights rows with probability proportional to weight,
  so weights are scaled to mean 1). Class weights are off by default, a deviation from the book's
  balanced_subsample (4.8 / 6.2): the output is used as P(the side wins) by stage 5's bet sizing (10.3 needs a
  calibrated probability), and balancing multiplies the odds by (1 - pi) / pi for a base rate pi, which pulls every
  probability toward 1/2 (code review afml.md, models finding 2). `balanced=True` balances each tree on its own draw
  (_SubsampleBalance: scikit-learn 1.9 passes a bagged tree its draw as sample weights over every training row, so a
  tree's own class_weight would balance the whole training set); its probabilities are not calibrated, must not feed
  bet sizing unrecalibrated, and lose to calibrated ones under the search's uniqueness-weighted log loss.
- logistic, the floor: median imputation plus missing-value indicators, a standard scaler and L2 logistic regression
  in a WeightedPipeline (9.2: fit takes sample_weight and hands it to the final step only). The imputer and scaler are
  part of the model, so they are fitted on each training set and nowhere else.
- hgb, the boosting challenger (6.5-6.6; scikit-learn's HistGradientBoosting, no download). Its own early stopping
  validates on a random share of rows, which leaks across overlapping labels, so it is off: the number of boosting
  iterations is chosen on the latest block of training days (events.purged_kfold inside the training rows: purged,
  embargoed, validation weights from the validation rows alone), then the model is refitted on all training rows.
  Each HGB fit leaves out the columns with fewer than two distinct observed values among its own rows (scikit-learn
  1.9 cannot bin them; the book features are empty before 2026-09-16 and f_rule_L is constant within a rule period).
- xgb, the second boosting challenger (XGBoost, installed 2026-09-19 with the user's approval; needs Homebrew's
  libomp on this machine). Same discipline as hgb: no random early stopping, the number of boosting rounds comes
  from the same purged inner validation (XGBoost's own weighted `logloss` curve on that block, whose minimum is the
  round count refitted on every training row), no class balancing (scale_pos_weight stays 1). XGBoost reads NaN as
  missing and bins constant columns without complaint, so no column is dropped. 6.4 warns that boosting cannot
  reduce the variance that dominates financial labels and overfits noise; whether it does here is measured, not
  assumed.

Weights (4.4-4.7, 7.5). Every weight comes from the rows it weights alone: concurrency, return attribution and time
decay among the fitted rows (or among the scored rows), with events.sample_weights' formulas (weights()). `fit_weight`
picks the fit weight, scaled to mean 1; the default "uniqueness_decay" is average uniqueness (4.4) x time decay (4.7).
`score_weight` weights the log loss of the scored rows; the default "avg_uniqueness" counts every independent label
once (time decay ages training rows relative to the prediction; the scored rows are the out-of-sample period itself).
Return attribution (4.6: "w_return", "weight" = w_return x w_decay) stays an explicit option for a bet/no-bet
classifier only: it is about |label - price|, so a log loss weighted by it is minimised by p* = P (1 - q) /
(P (1 - q) + (1 - P) q) rather than by P (p* = 1/2 when the price q is right), and the output is not P(win) (code
review afml.md, models finding 1).

Refit schedules (a search dimension; plan_steps lists every fit). Days are Taiwan days of the window start.
- `static`: one model per fold, fitted on the fold's training rows, predicts every test day.
- `rolling_days=N`: each test day d gets a model fitted on the rows of Taiwan days d - N .. d - 1 whose spans ended
  before d's first test event (the purge). A day with fewer than N days of data before it is not predicted.
- `per_window` (or `rolling_days=N+per_window`): the day's base model (static, or rolling) is updated before each
  test window with every window of that test day that settled before the window's first event; updates restart from
  the base model each Taiwan day. Updates by family: bagged_trees warm-starts `update_trees` new trees on base +
  settled rows (a supported sklearn warm start: trees are independent); logistic refits on base + settled rows from
  the previous coefficients (warm start only speeds the solver: the loss is convex, so this is the refit); hgb
  refits on base + settled rows with the base model's iteration count. HGB's warm start is not used: it re-bins the
  new rows while the old trees keep their bin thresholds (checked on scikit-learn 1.9: the old trees' raw predictions
  during the warm-started fit were off by up to 3.3 logits), so its update is the refit. xgb refits the same way,
  with the base model's round count: continuing a booster on more rows (`xgb_model=`) would keep optimising a
  gradient fitted to the base rows only, and the extra rounds are not the ones the inner validation chose.
A rolling model and every per-window update only use rows that settled before the rows they predict; the static base
does too under cv="walk_forward", not under the book's k-fold (Folds). Weights are not recomputed over every training
row at each update: a label never leaves its window, so concurrency and return attribution count the rows of one
window, and predict_fold computes them once for the base and once for each newly settled window (_SpanCache), then
only renormalises and re-decays (code review afml.md, models finding 3). Fits happen only at the schedule's refit
points: once per base, and once per window that has newly settled rows.

Folds (7.4). `cv="kfold"` is the book's purged k-fold: contiguous blocks of whole Taiwan days (events.purged_kfold),
training spans overlapping a test span dropped, EMBARGO = events.EMBARGO_S embargoed after each test block (a day's
fair parameters are fitted on the outcomes of the events.FAIR_TRAIN_DAYS days before it; the extra day covers the
8 h between Taiwan and UTC days); the static model trains on both sides of the block. `cv="walk_forward"` restricts
the static model to training rows that settled before the block's first event, so every schedule then satisfies:
changing anything after an event's second changes no prediction for that event. Comparing schedules is only like for
like under walk_forward.

Search (9.2-9.4). PurgedSearch runs a grid (ParameterGrid) or a randomised search (ParameterSampler; log_uniform for
scale parameters) over params x schedule, with a fold per task on at most MAX_WORKERS processes. Candidates are scored
on the same rows: per fold, the test rows every candidate predicted (a rolling schedule cannot predict its first N
days). Score = negative log loss weighted by `score_weight` (9.4; the book suggests F1 for meta-labels to expose a
classifier that calls everything negative, but here base rates are near one half and stage 5 sizes bets from the
probability, so log loss is the score and the constant base-rate forecast is reported beside it). Every candidate is
one line in a JSONL trials ledger under research/trials/<run>/ (scripts/trial_registry.py counts those lines; params,
value = mean fold score).
"""
import copy
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import events as ev
from pmlab.afml import sampling
from pmlab.afml.tuning import WeightedPipeline, log_uniform  # noqa: F401 (9.2, 9.4; part of this module's interface)
from pmlab.evaluation import CONFIRM_START
from pmlab.registry import MODEL_FAMILIES, ModelFamily

ROOT = Path(__file__).resolve().parents[3]
EVENTS = ROOT / "data/cache/afml/events"
TRIALS = ROOT / "research/trials"
CVS = ("kfold", "walk_forward")
LABEL = "meta_label"
TW = 8 * 3600                     # Taiwan time, UTC+8
EMBARGO = ev.EMBARGO_S            # outlasts the fair-parameter lookback (Folds)
WEIGHTS = ("uniqueness_decay", "avg_uniqueness", "w_decay", "w_return", "weight")
FIT_WEIGHT, SCORE_WEIGHT = "uniqueness_decay", "avg_uniqueness"
MAX_WORKERS = 4
EPS = 1e-15

# The families declare themselves, each beside its own defaults, into pmlab.registry.MODEL_FAMILIES, which is the
# one place membership is tested (`require`): before this, the tuple was written here and checked in two others.
# FAMILIES, BOOSTERS and DEFAULTS are read back from the registry and are exactly what they were, in the same order.
_family = lambda name, defaults, **kw: MODEL_FAMILIES.register(name, ModelFamily(name, defaults, **kw))
_family("bagged_trees", {"base": "tree", "n_estimators": 1000, "max_features": "sqrt",
                         "min_weight_fraction_leaf": 0.05, "max_depth": None, "balanced": False, "update_trees": 10},
        note="the book's default (6.3.3, 6.4): bagged entropy trees drawn by average uniqueness")
_family("logistic", {"C": 1.0, "class_weight": None, "max_iter": 1000},
        note="the linear baseline a tree ensemble has to beat")
_family("hgb", {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 100,
                "l2_regularization": 1.0, "max_features": 1.0, "class_weight": None, "val_splits": 5},
        booster=True, note="histogram gradient boosting from scikit-learn")
_family("xgb", {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 4, "min_child_weight": 100.0,
                "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0,
                "val_splits": 5},
        needs=("xgboost",), booster=True,
        note="declared whatever is installed; xgboost is imported when the family is fitted, never at import, "
             "because the registered environment must not hold it")
FAMILIES = MODEL_FAMILIES.names()
BOOSTERS = tuple(n for n in FAMILIES if MODEL_FAMILIES.get(n).booster)   # iterations from the purged inner validation
DEFAULTS = {n: MODEL_FAMILIES.get(n).defaults for n in FAMILIES}


# ---------------------------------------------------------------------------------------------------- inputs

def feature_columns(columns, features=None) -> list[str]:
    """The model inputs: every `f_` column (in order) or the requested ones, which must all be `f_` columns."""
    columns = [str(c) for c in columns]
    cols = [c for c in columns if c.startswith("f_")] if features is None else [str(c) for c in features]
    bad = [c for c in cols if not c.startswith("f_") or c not in columns]
    if bad:
        raise ValueError(f"not model inputs: {bad}")
    return cols


def _utc_day(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def load_events(root=EVENTS, days=None, rules=None, confirm_start: int = CONFIRM_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(event rows, price paths) from the day-partitioned dataset. A UTC day partition is opened only if all of its
    windows end by the confirmation start; rows and paths of any window ending after it are dropped as well.

    Read one day after another on purpose. A thread pool was tried here (pmlab.afml.threadio, which is 1.9x on 60
    kline days) and measured at 0.95x to 1.17x on these files, cold and warm alike: an event day is a few megabytes,
    so the time goes to the pandas conversion, which holds the interpreter lock, and not to waiting on the disk.
    docs/performance.md records the numbers.
    """
    parts = []
    for p in sorted(Path(root).glob("day=*")):
        day = p.name[len("day="):]
        if _utc_day(day) + 86400 > confirm_start:
            continue
        if days is not None and day not in days:
            continue
        parts.append(p)
    if not parts:
        raise ValueError("no event days to load")
    filt = [("rule", "in", list(rules))] if rules is not None else None
    rows = pd.concat([pd.read_parquet(p / "events.parquet", filters=filt) for p in parts], ignore_index=True)
    rows = rows[rows["t1"].to_numpy(np.int64) <= confirm_start].reset_index(drop=True)
    paths = pd.concat([pd.read_parquet(p / "paths.parquet") for p in parts], ignore_index=True)
    paths = paths[paths["start"].isin(pd.unique(rows["start"]))].reset_index(drop=True)
    return rows, paths


class Dataset:
    """Arrays of one rule's event rows: X (`f_` features, float32), y (the label), spans, Taiwan day of the window
    start, and the windows' price paths (for weights). Positions index the rows in their given order."""

    def __init__(self, X, y, start, t0, t1, path_starts, path_q, features, index, rule=None):
        self.X, self.y = X, y
        self.start, self.t0, self.t1 = start, t0, t1
        self.day = (start + TW) // 86400
        self.path_starts, self.path_q = path_starts, path_q
        self.features, self.index, self.rule = features, index, rule
        self._paths = None

    @classmethod
    def from_rows(cls, rows: pd.DataFrame, paths, features=None, label: str = LABEL) -> "Dataset":
        if len(rows) and (rows["t1"].to_numpy(np.int64) > CONFIRM_START).any():
            raise ValueError("a row's window ends after the confirmation start")
        rule = None
        if "rule" in rows:
            rules = pd.unique(rows["rule"])
            if len(rules) > 1:
                raise ValueError(f"one primary rule at a time, got {list(rules)}")
            rule = str(rules[0]) if len(rules) else None
        cols = feature_columns(rows.columns, features)
        y = rows[label].to_numpy()
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be 0 or 1")
        start = rows["start"].to_numpy(np.int64)
        lookup = paths if isinstance(paths, dict) else ev._paths_lookup(paths)
        starts = np.unique(start)
        missing = [int(s) for s in starts if int(s) not in lookup]
        if missing:
            raise ValueError(f"no price path for windows {missing[:5]}")
        q = np.stack([np.asarray(lookup[int(s)], dtype=float) for s in starts]) if len(starts) else np.zeros((0, T + 1))
        return cls(rows[cols].to_numpy(np.float32), y.astype(np.int8), start, rows["t0"].to_numpy(np.int64),
                   rows["t1"].to_numpy(np.int64), starts, q, cols, rows.index, rule)

    def __len__(self):
        return len(self.y)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_paths"] = None
        return state

    @property
    def paths(self) -> dict:
        if self._paths is None:
            self._paths = dict(zip(self.path_starts.tolist(), self.path_q))
        return self._paths

    def spans(self, idx) -> pd.DataFrame:
        idx = np.asarray(idx, dtype=np.int64)
        return pd.DataFrame({"start": self.start[idx], "t0": self.t0[idx], "t1": self.t1[idx]})

    def span_weights(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """(average uniqueness, unnormalised return attribution) of the rows `idx`, concurrency among them alone
        (events.span_weights)."""
        return ev.span_weights(self.spans(idx), self.paths)

    def weights(self, idx, column: str) -> np.ndarray:
        """Weight `column` of the rows `idx` computed on those rows alone (weights())."""
        idx = np.asarray(idx, dtype=np.int64)
        return weights(*self.span_weights(idx), self.t0[idx], column)


def _check_weights(*names):
    for name in names:
        if name not in WEIGHTS:
            raise ValueError(f"weight is one of {WEIGHTS}, got {name!r}")


def weights(u, raw, t0, column: str, decay_c: float = ev.DECAY_C) -> np.ndarray:
    """One weight per row from its span weights (u = average uniqueness, raw = unnormalised return attribution) and span
    start t0, with events.sample_weights' formulas on these rows: avg_uniqueness = u (4.4); w_return = raw scaled to sum
    to the row count (4.6); w_decay = time decay on cumulative uniqueness in t0 order (4.7); weight = w_return x w_decay;
    uniqueness_decay = u x w_decay."""
    _check_weights(column)
    u, raw = np.asarray(u, dtype=float), np.asarray(raw, dtype=float)
    if column == "avg_uniqueness":
        return u
    w_return = raw * len(raw) / raw.sum() if raw.sum() > 0 else np.ones(len(raw))
    if column == "w_return":
        return w_return
    w_decay = sampling.time_decay(u, order=np.asarray(t0), c=decay_c)
    return {"w_decay": w_decay, "weight": w_return * w_decay, "uniqueness_decay": u * w_decay}[column]


def log_loss(y, p, w) -> float:
    """Weighted mean negative log likelihood of P(label = 1)."""
    y, w = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


# ---------------------------------------------------------------------------------------------------- estimators

class _SubsampleBalance:
    """Class weights from the weighted rows the estimator is given (inside bagging: its own draw), so every class
    carries the same total weight in each tree (balanced_subsample)."""

    def fit(self, X, y, sample_weight=None, **kw):
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        classes, inv = np.unique(y, return_inverse=True)
        total = np.bincount(inv, weights=w, minlength=len(classes))
        present = total > 0
        cw = np.divide(w.sum(), present.sum() * total, out=np.zeros(len(classes)), where=present)
        return super().fit(X, y, sample_weight=w * cw[inv], **kw)


class BalancedTree(_SubsampleBalance, DecisionTreeClassifier):
    pass


class BalancedForest(_SubsampleBalance, RandomForestClassifier):
    pass


def _bagging(p: dict, max_samples: float, seed: int) -> BaggingClassifier:
    tree = {"criterion": "entropy", "max_features": p["max_features"], "max_depth": p["max_depth"],
            "min_weight_fraction_leaf": p["min_weight_fraction_leaf"]}
    if p["base"] == "tree":
        base = (BalancedTree if p["balanced"] else DecisionTreeClassifier)(**tree)
    else:
        base = (BalancedForest if p["balanced"] else RandomForestClassifier)(n_estimators=1, bootstrap=False, **tree)
    return BaggingClassifier(base, n_estimators=p["n_estimators"], max_samples=min(1.0, max(max_samples, 1e-6)),
                             max_features=1.0, bootstrap=True, random_state=seed, n_jobs=1)


def _logistic(p: dict) -> WeightedPipeline:
    prep = FeatureUnion([("value", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("missing", MissingIndicator(features="all"))])
    return WeightedPipeline([("impute", prep), ("scale", StandardScaler()),
                             ("logit", LogisticRegression(C=p["C"], class_weight=p["class_weight"],
                                                          max_iter=p["max_iter"]))])


def _hgb(p: dict, max_iter: int, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_iter=int(max_iter),
                                          max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
                                          l2_regularization=p["l2_regularization"], max_features=p["max_features"],
                                          class_weight=p["class_weight"], early_stopping=False, random_state=seed)


def _xgboost_class():
    """XGBoost, imported when it is asked for rather than at module load.

    It is deliberately absent from the registered environment: requirements.lock is a frozen artifact of the
    confirmation registration, and an extra installed package fails its library check. The boosted challenger runs in
    .venv-research, built from the same lock so numpy, scipy, scikit-learn, pandas and pyarrow match version for
    version (research/preregistration_clarifications.md, 2026-09-19)."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:                              # a clear instruction beats a bare ImportError
        raise ImportError("XGBoost is not in the registered environment by design; run this with "
                          ".venv-research/bin/python (see research/meta_label.md)") from exc
    return XGBClassifier


def _xgb(p: dict, n_estimators: int, seed: int):
    """XGBoost with no class balancing (scale_pos_weight 1), NaN read as missing, one thread per fit."""
    return _xgboost_class()(n_estimators=int(n_estimators), learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                         min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                         colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
                         reg_alpha=p["reg_alpha"], gamma=p["gamma"], tree_method="hist",
                         objective="binary:logistic", eval_metric="logloss", scale_pos_weight=1.0,
                         missing=np.nan, random_state=seed, n_jobs=1)


def _two_classes(y) -> bool:
    """Both labels present. A training block with one label cannot be fitted (XGBoost refuses outright, and a
    scikit-learn estimator fitted on it has no class 1 to read a probability from), so the step predicts nothing."""
    return len(np.unique(np.asarray(y))) > 1


def _varying(X: np.ndarray) -> np.ndarray:
    """Columns with at least two distinct observed (non-NaN) values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns
        lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    return np.flatnonzero(np.isfinite(lo) & (hi > lo))


def inner_split(data: Dataset, train, n_splits: int):
    """(inner training, validation) positions: the latest block of training days validates, purged and embargoed
    (events.purged_kfold on the training rows); None with fewer than two training days."""
    train = np.asarray(train, dtype=np.int64)
    frame = pd.DataFrame({"day": data.day[train], "t0": data.t0[train], "t1": data.t1[train]})
    folds = ev.purged_kfold(frame, n_splits=n_splits, embargo=EMBARGO, group="day")
    if len(folds) < 2 or not len(folds[-1][0]):
        return None
    itr, iva = folds[-1]
    return train[itr], train[iva]


class MetaModel:
    """fit(data, train) on training positions, update(data, train) for per-window schedules, predict_proba(data, rows)
    = P(label = 1). Sample weights always come from the rows being fitted; `spans` = data.span_weights(train), when
    the caller already has them (predict_fold)."""

    def __init__(self, family: str, params: dict | None = None, fit_weight: str = FIT_WEIGHT,
                 score_weight: str = SCORE_WEIGHT, seed: int = 0):
        MODEL_FAMILIES.require(family)
        params = dict(params or {})
        unknown = sorted(set(params) - set(DEFAULTS[family]))
        if unknown:
            raise ValueError(f"unknown {family} parameters {unknown}")
        self.family, self.params = family, {**DEFAULTS[family], **params}
        if family == "bagged_trees" and self.params["base"] not in ("tree", "forest"):
            raise ValueError("bagged_trees base is 'tree' or 'forest'")
        _check_weights(fit_weight, score_weight)
        self.fit_weight, self.score_weight, self.seed = fit_weight, score_weight, seed
        self.info_ = {}

    def _xyw(self, data: Dataset, idx, spans=None):
        idx = np.asarray(idx, dtype=np.int64)
        u, raw = data.span_weights(idx) if spans is None else spans
        w = weights(u, raw, data.t0[idx], self.fit_weight)
        return data.X[idx], data.y[idx], w / w.mean(), float(np.mean(u))

    def fit(self, data: Dataset, train, spans=None) -> "MetaModel":
        train = np.asarray(train, dtype=np.int64)
        X, y, w, mean_u = self._xyw(data, train, spans)
        p = self.params
        self.info_ = {"n_train": int(len(train))}
        if self.family == "bagged_trees":
            self.estimator_ = _bagging(p, mean_u, self.seed).fit(X, y, sample_weight=w)
            self.info_["max_samples"] = mean_u
        elif self.family == "logistic":
            self.estimator_ = _logistic(p).fit(X, y, sample_weight=w)
        elif self.family == "xgb":
            n_iter, self.inner_ = p["n_estimators"], inner_split(data, train, p["val_splits"])
            if self.inner_ is not None and not all(map(_two_classes, (data.y[self.inner_[0]], data.y[self.inner_[1]]))):
                self.inner_ = None
            if self.inner_ is not None:
                itr, iva = self.inner_
                Xi, yi, wi, _ = self._xyw(data, itr)
                wv = data.weights(iva, self.score_weight)
                m = _xgb(p, p["n_estimators"], self.seed).fit(
                    Xi, yi, sample_weight=wi, eval_set=[(data.X[iva], data.y[iva])],
                    sample_weight_eval_set=[wv], verbose=False)
                n_iter = int(np.argmin(m.evals_result()["validation_0"]["logloss"])) + 1
            self.estimator_ = _xgb(p, n_iter, self.seed).fit(X, y, sample_weight=w)
            self.info_["n_iter"] = n_iter
        else:
            n_iter, self.inner_ = p["max_iter"], inner_split(data, train, p["val_splits"])
            if self.inner_ is not None and not all(map(_two_classes, (data.y[self.inner_[0]], data.y[self.inner_[1]]))):
                self.inner_ = None
            if self.inner_ is not None:
                itr, iva = self.inner_
                Xi, yi, wi, _ = self._xyw(data, itr)
                keep = _varying(Xi)
                m = _hgb(p, p["max_iter"], self.seed).fit(Xi[:, keep], yi, sample_weight=wi)
                wv = data.weights(iva, self.score_weight)
                col = list(m.classes_).index(1)
                losses = [log_loss(data.y[iva], pr[:, col], wv) for pr in m.staged_predict_proba(data.X[iva][:, keep])]
                n_iter = int(np.argmin(losses)) + 1
            self.keep_ = _varying(X)
            self.estimator_ = _hgb(p, n_iter, self.seed).fit(X[:, self.keep_], y, sample_weight=w)
            self.info_["n_iter"] = n_iter
        return self

    def update(self, data: Dataset, train, spans=None) -> "MetaModel":
        """The per-window update on `train` (the base rows plus the rows settled since): see the module docstring."""
        train = np.asarray(train, dtype=np.int64)
        X, y, w, mean_u = self._xyw(data, train, spans)
        est = self.estimator_
        if self.family == "bagged_trees":
            est.set_params(warm_start=True, n_estimators=len(est.estimators_) + self.params["update_trees"],
                           max_samples=min(1.0, max(mean_u, 1e-6)))
            est.fit(X, y, sample_weight=w)
        elif self.family == "logistic":
            est.set_params(logit__warm_start=True)
            est.fit(X, y, sample_weight=w)
        elif self.family == "xgb":
            self.estimator_ = _xgb(self.params, self.info_["n_iter"], self.seed).fit(X, y, sample_weight=w)
        else:
            self.keep_ = _varying(X)
            self.estimator_ = _hgb(self.params, self.info_["n_iter"], self.seed).fit(X[:, self.keep_], y, sample_weight=w)
        self.info_["n_train"] = int(len(train))
        return self

    def predict_proba(self, data: Dataset, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        col = list(self.estimator_.classes_).index(1)
        X = data.X[rows][:, self.keep_] if self.family == "hgb" else data.X[rows]
        return self.estimator_.predict_proba(X)[:, col]


# ---------------------------------------------------------------------------------------------------- schedules

@dataclass(frozen=True)
class Schedule:
    days: int | None = None       # None: static; N: refit each Taiwan day on the trailing N days
    per_window: bool = False

    @classmethod
    def parse(cls, text) -> "Schedule":
        days, per_window, seen = None, False, set()
        for part in str(text).split("+"):
            if part == "static":
                key = "base"
            elif part.startswith("rolling_days="):
                key, days = "base", int(part[len("rolling_days="):])
                if days < 1:
                    raise ValueError("rolling_days must be >= 1")
            elif part == "per_window":
                key, per_window = "per_window", True
            else:
                raise ValueError(f"unknown schedule {text!r}")
            if key in seen:
                raise ValueError(f"unknown schedule {text!r}")
            seen.add(key)
        return cls(days, per_window)

    def __str__(self):
        if self.days is None:
            return "per_window" if self.per_window else "static"
        return f"rolling_days={self.days}" + ("+per_window" if self.per_window else "")


@dataclass
class Step:
    """One model state: fitted on `base` (+ `settled`, per window) and predicting `test` (positions)."""
    day: int
    window: int | None
    base_key: tuple
    base: np.ndarray
    settled: np.ndarray
    test: np.ndarray

    @property
    def train(self) -> np.ndarray:
        return np.concatenate([self.base, self.settled])


def day_folds(data: Dataset, n_splits: int = 5, embargo: int = EMBARGO) -> list:
    """The book's purged k-fold over contiguous blocks of whole Taiwan days (events.purged_kfold)."""
    frame = pd.DataFrame({"day": data.day, "t0": data.t0, "t1": data.t1})
    return ev.purged_kfold(frame, n_splits=n_splits, embargo=embargo, group="day")


def plan_steps(data: Dataset, train, test, schedule, cv: str = "kfold") -> list[Step]:
    """Every model state a schedule uses to predict one fold's test rows, in time order (module docstring)."""
    if cv not in CVS:
        raise ValueError(f"cv is one of {CVS}")
    sch = schedule if isinstance(schedule, Schedule) else Schedule.parse(schedule)
    train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
    if not len(test):
        return []
    empty = np.zeros(0, dtype=np.int64)
    static = train
    if cv == "walk_forward":
        static = train[data.t1[train] <= data.t0[test].min()]
    first = data.day.min()
    steps = []
    for d in np.unique(data.day[test]):
        rows = test[data.day[test] == d]
        if sch.days is None:
            base, key = static, ("static",)
        else:
            if d - sch.days < first:
                continue
            cut = data.t0[rows].min()
            base = np.flatnonzero((data.day >= d - sch.days) & (data.day < d) & (data.t1 <= cut))
            key = ("rolling", int(d))
        if not len(base):
            continue
        if not sch.per_window:
            steps.append(Step(int(d), None, key, base, empty, rows))
            continue
        for w in np.unique(data.start[rows]):
            wrows = rows[data.start[rows] == w]
            settled = rows[data.t1[rows] <= data.t0[wrows].min()]
            steps.append(Step(int(d), int(w), key, base, settled, wrows))
    return steps


class _SpanCache:
    """Span weights (Dataset.span_weights) of a step's training rows, base + settled, without recomputing every row
    at each update. Concurrency and return attribution count the rows of one window (a span never leaves its window),
    so the base's values are computed once per base, and a settled row's values once per change of its window's
    settled rows. Settled rows sharing a window with the base (folds that split a window) are computed with the whole
    training set instead. The result equals Dataset.span_weights(step.train)."""

    def __init__(self, data: Dataset):
        self.data, self.key = data, None
        self.u, self.raw = np.full(len(data), np.nan), np.full(len(data), np.nan)
        self.done = np.zeros(len(data), bool)

    def base(self, st: Step) -> tuple[np.ndarray, np.ndarray]:
        if st.base_key != self.key:
            self.key, self.base_spans = st.base_key, self.data.span_weights(st.base)
            self.base_windows = np.unique(self.data.start[st.base])
        return self.base_spans

    def train(self, st: Step) -> tuple[np.ndarray, np.ndarray]:
        u0, raw0 = self.base(st)
        start = self.data.start
        new = st.settled[~self.done[st.settled]]
        if len(new):
            touched = np.unique(start[new])
            if np.isin(touched, self.base_windows).any():
                return self.data.span_weights(st.train)
            rows = st.settled[np.isin(start[st.settled], touched)]
            self.u[rows], self.raw[rows] = self.data.span_weights(rows)
            self.done[rows] = True
        return np.concatenate([u0, self.u[st.settled]]), np.concatenate([raw0, self.raw[st.settled]])


def predict_fold(data: Dataset, family: str, params: dict, schedule, train, test, cv: str = "kfold",
                 fit_weight: str = FIT_WEIGHT, score_weight: str = SCORE_WEIGHT, seed: int = 0) -> np.ndarray:
    """P(label = 1) for the test positions (NaN where the schedule has no model) under one refit schedule."""
    test = np.asarray(test, dtype=np.int64)
    out = np.full(len(data), np.nan)
    bases, day, model, n_settled = {}, None, None, 0
    cache = _SpanCache(data)
    for st in plan_steps(data, train, test, schedule, cv):
        if not _two_classes(data.y[st.base]):
            continue                                       # nothing to fit: those test rows stay NaN and go unscored
        if st.base_key not in bases:
            bases = {st.base_key: MetaModel(family, params, fit_weight, score_weight, seed).fit(data, st.base,
                                                                                               cache.base(st))}
        if st.window is None:
            model = bases[st.base_key]
        else:
            if st.day != day:
                model, day, n_settled = copy.deepcopy(bases[st.base_key]), st.day, 0
            if len(st.settled) > n_settled:
                model.update(data, st.train, cache.train(st))
                n_settled = len(st.settled)
        out[st.test] = model.predict_proba(data, st.test)
    return out[test]


# ---------------------------------------------------------------------------------------------------- search

def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def _fold_task(data, family, params, schedule, train, test, cv, fit_weight, score_weight, seed):
    t = time.time()
    p = predict_fold(data, family, params, schedule, train, test, cv, fit_weight, score_weight, seed)
    return p.astype(np.float32), time.time() - t


class PurgedSearch:
    """Grid (n_iter None) or randomised search over `space` (hyper-parameters of `family` plus "schedule") on purged
    folds of whole Taiwan days. After fit: cv_results_ (one row per candidate: params, schedule, mean and sd of the
    fold scores, fold scores, scored rows per fold), best_index_, best_params_, best_score_, best_fold_scores_,
    baseline_fold_scores_ (the test rows' own weighted base rate as a constant forecast), folds_, and predictions()."""

    def __init__(self, family: str, space: dict, n_iter: int | None = None, n_splits: int = 5,
                 embargo: int = EMBARGO, cv: str = "kfold", fit_weight: str = FIT_WEIGHT,
                 score_weight: str = SCORE_WEIGHT, seed: int = 0, n_jobs: int = 1, ledger=None,
                 study: str | None = None):
        MODEL_FAMILIES.require(family)
        if cv not in CVS:
            raise ValueError(f"cv is one of {CVS}")
        _check_weights(fit_weight, score_weight)
        self.family, self.space, self.n_iter = family, dict(space), n_iter
        self.n_splits, self.embargo, self.cv = n_splits, embargo, cv
        self.fit_weight, self.score_weight, self.seed = fit_weight, score_weight, seed
        self.n_jobs = max(1, min(int(n_jobs), MAX_WORKERS))
        self.study = study or family
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.ledger = Path(ledger) if ledger is not None else TRIALS / f"meta_label_{stamp}" / f"{self.study}.jsonl"

    def candidates(self) -> list[dict]:
        if self.n_iter is None:
            raw = list(ParameterGrid(self.space))
        else:
            raw = list(ParameterSampler(self.space, self.n_iter, random_state=self.seed))
        out = []
        for c in raw:
            c = {k: _plain(v) for k, v in c.items()}
            c["schedule"] = str(Schedule.parse(c.get("schedule", "static")))
            MetaModel(self.family, {k: v for k, v in c.items() if k != "schedule"})     # validates the parameters
            out.append(c)
        return out

    def fit(self, data: Dataset) -> "PurgedSearch":
        cands = self.candidates()
        folds = day_folds(data, self.n_splits, self.embargo)
        tasks = [(ci, fi) for ci in range(len(cands)) for fi in range(len(folds))]

        def args(ci, fi):
            c = cands[ci]
            return (data, self.family, {k: v for k, v in c.items() if k != "schedule"}, c["schedule"], folds[fi][0],
                    folds[fi][1], self.cv, self.fit_weight, self.score_weight, self.seed)

        if self.n_jobs == 1:
            done = [_fold_task(*args(ci, fi)) for ci, fi in tasks]
        else:
            done = Parallel(n_jobs=self.n_jobs, backend="loky")(delayed(_fold_task)(*args(ci, fi)) for ci, fi in tasks)
        oof = np.full((len(cands), len(data)), np.nan, dtype=np.float32)
        seconds = np.zeros(len(cands))
        fold_of = np.full(len(data), -1, dtype=np.int64)
        for (ci, fi), (p, sec) in zip(tasks, done):
            oof[ci, folds[fi][1]] = p
            seconds[ci] += sec
            fold_of[folds[fi][1]] = fi
        scored = np.isfinite(oof).all(axis=0)
        fold_scores = [[] for _ in cands]
        baseline, n_scored, used = [], [], []
        for fi, (_, test) in enumerate(folds):
            rows = test[scored[test]]
            if not len(rows):
                continue
            w = data.weights(rows, self.score_weight)
            y = data.y[rows]
            baseline.append(-log_loss(y, np.full(len(rows), np.sum(w * y) / np.sum(w)), w))
            for ci in range(len(cands)):
                fold_scores[ci].append(-log_loss(y, oof[ci, rows], w))
            n_scored.append(int(len(rows)))
            used.append(fi)
        if not used:
            raise ValueError("no test rows that every candidate predicts")
        means = np.array([np.mean(s) for s in fold_scores])
        self.folds_, self.fold_of_, self.scored_, self.oof_, self.data_ = folds, fold_of, scored, oof, data
        self.baseline_fold_scores_, self.scored_folds_ = baseline, used
        self.cv_results_ = pd.DataFrame({
            "params": [{k: v for k, v in c.items() if k != "schedule"} for c in cands],
            "schedule": [c["schedule"] for c in cands], "mean_score": means,
            "std_score": [float(np.std(s, ddof=1)) if len(s) > 1 else np.nan for s in fold_scores],
            "fold_scores": fold_scores, "n_scored": [n_scored] * len(cands), "seconds": seconds})
        self.best_index_ = int(np.argmax(means))
        self.best_params_ = cands[self.best_index_]
        self.best_score_ = float(means[self.best_index_])
        self.best_fold_scores_ = fold_scores[self.best_index_]
        self._write_ledger(cands, fold_scores, n_scored, seconds, data)
        return self

    def _write_ledger(self, cands, fold_scores, n_scored, seconds, data):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        days = [str(pd.Timestamp(int(d) * 86400, unit="s").date()) for d in (data.day.min(), data.day.max())]
        with self.ledger.open("a") as f:
            for ci, c in enumerate(cands):
                f.write(json.dumps({
                    "ts": ts, "params": {"family": self.family, **c}, "value": float(np.mean(fold_scores[ci])),
                    "phase": "grid" if self.n_iter is None else "random", "iteration": ci, "study": self.study,
                    "score": f"neg weighted log loss ({self.score_weight})", "fold_scores": fold_scores[ci],
                    "baseline_fold_scores": self.baseline_fold_scores_, "n_scored": n_scored, "cv": self.cv,
                    "n_splits": self.n_splits, "embargo": self.embargo, "fit_weight": self.fit_weight,
                    "rule": data.rule, "rows": len(data), "features": len(data.features), "taiwan_days": days,
                    "seconds": float(seconds[ci])}, default=_plain) + "\n")

    def predictions(self, candidate: int | None = None) -> pd.DataFrame:
        """Out-of-fold P(label = 1) of one candidate (default the best) aligned to the event rows, with their spans,
        Taiwan day, fold and whether the row was scored (predicted by every candidate)."""
        c = self.best_index_ if candidate is None else int(candidate)
        d = self.data_
        return pd.DataFrame({"start": d.start, "t": d.t0 - d.start, "t0": d.t0, "t1": d.t1,
                             "day_tw": pd.to_datetime(d.day * 86400, unit="s").strftime("%Y-%m-%d"),
                             "fold": self.fold_of_, "p": self.oof_[c].astype(float), "scored": self.scored_},
                            index=d.index)
