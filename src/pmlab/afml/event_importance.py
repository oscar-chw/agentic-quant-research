"""Feature importance on the event dataset (plan afml-pipeline stage 3; AFML ch. 7-8): the design matrix, folds,
weights, forests, per-row out-of-fold losses, inner tuning, the PCA check and the label-permutation control.
scripts/feature_importance.py runs them; every leak guard here has a test in tests/test_feature_importance.py.

Inputs. One primary rule at a time (rows of different rules at one event share every feature). The model sees `f_`
columns only, never an outcome, span, weight or bet column (INPUT_FORBIDDEN), and no row of a window that ends after
pmlab.evaluation.CONFIRM_START. Book features are left out: they exist for the recorded windows only.

Side frame. A row's label is "the rule's side won", so Up-frame features are put into the side's frame, row by row and
with no fitted parameter: signed features (BTC returns, z, market - fair and its changes, flow and bar imbalances, the
CUSUM sign) are multiplied by the side; the fair price and the Up-buy share become the side's (1 - x for Down).
Unsigned features (time, volatility, volume, spreads, ages, VPIN, Kyle's lambda, Amihud, Corwin-Schultz, Roll) are the
same in either frame.

Added activity. Binance trade counts over the last 60 and 300 s, log(1 + count), from the 1 s klines that closed by the
event instant: the kline opened at second i closes at i + 1, so an event at instant e reads klines opened in
[e - k, e - 1]; NaN if one is missing.

Folds, weights, tuning (4.4-4.5, 6.3.1, 7.3-7.4). Test folds are contiguous blocks of whole UTC days
(events.purged_kfold): a window never straddles a fold, spans are purged, training days within the embargo after a test
block are dropped. The embargo is set by the day models, not by the label span: a day's fair parameters are fitted on
the settled windows of the events.FAIR_TRAIN_DAYS days before it, so the features of the FAIR_TRAIN_DAYS days after a
test block carry its outcomes; EMBARGO_S (events.EMBARGO_S) covers those days plus the window horizon. The HAR
(events.HAR_TRAIN_DAYS of BTC klines) and the bar calibration (a day of tape) reach back too but read no outcome. Every weight a fold uses comes
from events.sample_weights on that side's own rows: the training rows' average uniqueness fits the forest and sets
max_samples, the test rows' average uniqueness weights the out-of-fold log loss. Fit weights are scaled to mean 1:
scikit-learn draws max_samples x (sum of sample weights) rows per tree, so raw uniqueness (mean well below 1) would
shrink every bootstrap below max_samples. The forest's early-stopping leaf size
is chosen by an inner purged day k-fold on the training rows alone.

Label-permutation control. Settlement outcomes are permuted across whole windows, so the rows of one window keep one
(shuffled) outcome and the dependence a leak would exploit survives, while every feature-label link is broken. Each
row's control label is its side's payoff under the shuffled outcome (prices are below 1, so label = payoff).
"""
import zlib

import numpy as np
import pandas as pd
from scipy.stats import weightedtau
from sklearn.ensemble import RandomForestClassifier

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import events as ev
from pmlab.evaluation import CONFIRM_START

GROUP = {f["name"]: f["group"] for f in ev.FEATURES}
BAR_KINDS = ("tick", "volume", "dollar", "tick_imbalance", "volume_imbalance", "dollar_imbalance", "tick_run",
             "volume_run", "dollar_run")
SIGNED = ("f_btc_ret_1", "f_btc_ret_5", "f_btc_ret_10", "f_btc_ret_30", "f_btc_ret_60", "f_z_q", "f_last_minus_q",
          "f_mid_minus_q", "f_last_minus_q_chg10", "f_mid_minus_q_chg10", "f_last_minus_q_chg30", "f_mid_minus_q_chg30",
          "f_flow", "f_cusum_sign") + tuple(f"f_{k}_imb" for k in BAR_KINDS)
SHARE = ("f_fair", "f_buy_share_60")                   # probabilities / shares of Up: 1 - x in the Down frame
TRADE_SPANS = (60, 300)
ADDED = tuple(f"f_btc_trades_{k}" for k in TRADE_SPANS)
FEATURES = [n for n in ev.FEATURE_NAMES if GROUP[n] != "book"] + list(ADDED)
INPUT_FORBIDDEN = frozenset(ev.LABEL_COLUMNS) | frozenset(ev.WEIGHT_COLUMNS) | {
    "t0", "t1", "side", "price", "price_age", "fee", "cost", "fair_side", "edge", "maker_bid", "book_price", "ask_up",
    "bid_up", "weight", "w_return", "w_decay", "start", "t", "day", "day_tw", "rule"}
EMBARGO_S = ev.EMBARGO_S                               # (FAIR_TRAIN_DAYS + 1) days, defined with the day models
BASELINE = ("f_fair", "f_z_q", "f_last_minus_q", "f_mid_minus_q", "f_tau")
CLOCK = ("f_hour_sin", "f_hour_cos", "f_dow_sin", "f_dow_cos", "f_session")
ACTIVITY = ("f_btc_trades_60", "f_btc_trades_300", "f_sigma_ewma", "f_rv_60", "f_rv_300", "f_sigma_har")


# ---------------------------------------------------------------------------------------------------- rows

def subsample(rows: pd.DataFrame, n: int, key: str) -> pd.DataFrame:
    """At most n rows, uniformly without replacement, in their original order. The draw depends on `key` (e.g. day and
    rule) and the row count only, never on a row's content."""
    if len(rows) <= n:
        return rows
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return rows.iloc[np.sort(rng.choice(len(rows), n, replace=False))]


def orient(rows: pd.DataFrame) -> pd.DataFrame:
    """Up-frame features in the frame of each row's side (module docstring)."""
    out = rows.copy()
    s = out["side"].to_numpy(float)
    if not np.isin(s, (1.0, -1.0)).all():
        raise ValueError("side must be +1 or -1")
    for c in SIGNED:
        x = out[c].to_numpy()
        out[c] = x * s.astype(x.dtype)
    for c in SHARE:
        x = out[c].to_numpy()
        out[c] = np.where(s > 0, x, 1 - x).astype(x.dtype)
    return out


def btc_trades(sec: np.ndarray, n_trades: np.ndarray, instants: np.ndarray, k: int) -> np.ndarray:
    """log(1 + Binance trades) in the klines opened in [e - k, e - 1] for each event instant e; NaN if one is missing."""
    sec = np.asarray(sec, dtype=np.int64)
    instants = np.asarray(instants, dtype=np.int64)
    lo = int(min(sec.min(), instants.min() - k))
    hi = int(max(sec.max(), instants.max())) + 1
    trades, have = np.zeros(hi - lo), np.zeros(hi - lo)
    trades[sec - lo] = np.asarray(n_trades, dtype=float)
    have[sec - lo] = 1.0
    ct = np.concatenate([[0.0], np.cumsum(trades)])
    ch = np.concatenate([[0.0], np.cumsum(have)])
    a, b = instants - k - lo, instants - lo              # klines opened in [e - k, e - 1]: positions a .. b - 1
    total, present = ct[b] - ct[a], ch[b] - ch[a]
    return np.where(present == k, np.log1p(total), np.nan)


def check_inputs(rows: pd.DataFrame, columns) -> None:
    """Refuse any model input that is not a stage-3 feature, and any row of a window ending after CONFIRM_START."""
    bad = [c for c in columns if not str(c).startswith("f_") or c in INPUT_FORBIDDEN or c not in FEATURES]
    if bad:
        raise ValueError(f"not model inputs: {bad}")
    if len(rows) and (rows["start"].to_numpy(np.int64) + T > CONFIRM_START).any():
        raise ValueError("a row's window ends after the confirmation start")


def design(rows: pd.DataFrame, columns) -> np.ndarray:
    check_inputs(rows, columns)
    return rows[list(columns)].to_numpy(np.float32)


# ---------------------------------------------------------------------------------------------------- folds, weights

def day_folds(rows: pd.DataFrame, k: int, embargo: int = EMBARGO_S) -> list:
    """Purged k-fold on whole UTC days with an embargo after each test block (events.purged_kfold); the default embargo
    outlasts the fair-parameter lookback (EMBARGO_S)."""
    return ev.purged_kfold(rows, n_splits=k, embargo=embargo, group="day")


def fold_weights(rows: pd.DataFrame, paths, train, test) -> dict:
    """Weights from each side's own rows: training uniqueness (fit weights, max_samples) and test uniqueness (scoring)."""
    train, test = np.asarray(train), np.asarray(test)
    w_tr = ev.sample_weights(rows, paths, train)["avg_uniqueness"].to_numpy()[train]
    w_te = ev.sample_weights(rows, paths, test)["avg_uniqueness"].to_numpy()[test]
    return {"fit": w_tr, "score": w_te, "max_samples": float(np.mean(w_tr))}


def forest(max_samples: float, max_features, n_estimators: int, leaf: float, seed: int, n_jobs: int = 1):
    """The book's bagged trees as a random forest: entropy splits, bootstrap draws of max_samples (mean uniqueness),
    balanced_subsample class weights, early stopping by min_weight_fraction_leaf (4.5, 6.3.1, 7.3, 8.3.1)."""
    return RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_features=max_features,
                                  max_samples=min(1.0, max(float(max_samples), 1e-3)), bootstrap=True,
                                  class_weight="balanced_subsample", min_weight_fraction_leaf=leaf,
                                  random_state=seed, n_jobs=n_jobs)


def row_losses(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row log loss of P(label = 1); a weighted mean of these is sklearn's weighted log loss."""
    col = list(model.classes_).index(1)
    p = np.clip(model.predict_proba(X)[:, col], 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def permuted_losses(model, X: np.ndarray, y: np.ndarray, groups: dict, rng) -> dict:
    """{group: per-row losses after permuting the group's columns jointly across the rows of X} (8.3.2; a cluster's
    columns move together, so its internal correlation is kept)."""
    rng = np.random.default_rng(rng)
    out = {}
    for name, cols in groups.items():
        Xp = X.copy()
        Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
        out[name] = row_losses(model, Xp, y)
    return out


def fold_losses(model, X: np.ndarray, y: np.ndarray, train, test, w_fit, groups: dict | None = None, rng=None) -> dict:
    """Fit on the training rows (weights scaled to mean 1, so a bootstrap draws max_samples of the rows), per-row losses
    on the test rows, and after permuting each group within the test rows."""
    w_fit = np.asarray(w_fit, dtype=float)
    w_fit = w_fit / w_fit.mean()
    m = model.fit(X[train], y[train], sample_weight=w_fit)
    out = {"model": m, "base": row_losses(m, X[test], y[test])}
    if groups:
        out["permuted"] = permuted_losses(m, X[test], y[test], groups, rng)
    return out


def weighted_mean(losses, w) -> float:
    return float(np.sum(np.asarray(w) * np.asarray(losses)) / np.sum(w))


def inner_leaf(rows: pd.DataFrame, paths, X: np.ndarray, y: np.ndarray, train, grid, k: int, n_estimators: int,
               max_features, seed: int) -> dict:
    """min_weight_fraction_leaf chosen by a purged day k-fold inside the training rows (weights from each inner side's
    own rows), lowest mean weighted log loss."""
    train = np.asarray(train)
    sub = rows.iloc[train].reset_index(drop=True)
    Xs, ys = X[train], y[train]
    scores = {float(leaf): [] for leaf in grid}
    for itr, ite in day_folds(sub, k):
        w = fold_weights(sub, paths, itr, ite)
        for leaf in scores:
            r = fold_losses(forest(w["max_samples"], max_features, n_estimators, leaf, seed), Xs, ys, itr, ite, w["fit"])
            scores[leaf].append(weighted_mean(r["base"], w["score"]))
    mean = {leaf: float(np.mean(v)) for leaf, v in scores.items()}
    return {"leaf": min(mean, key=mean.get), "scores": mean}


# ---------------------------------------------------------------------------------------------------- PCA check

def pca_fit(X_train: np.ndarray, var_share: float = 0.95) -> dict:
    """8.4.2 on training rows: NaN imputed by the training median, standardised by the training mean and sd, Z'Z / n
    eigendecomposed, the fewest components explaining var_share of the variance kept."""
    X = np.asarray(X_train, dtype=float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isfinite(X), X, med)
    mu, sd = Xf.mean(axis=0), Xf.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (Xf - mu) / sd
    val, vec = np.linalg.eigh(Z.T @ Z / len(Z))
    order = np.argsort(val)[::-1]
    val, vec = np.clip(val[order], 0.0, None), vec[:, order]
    keep = int(min(np.searchsorted(np.cumsum(val) / val.sum(), var_share) + 1, len(val)))
    return {"median": med, "mean": mu, "sd": sd, "eigenvalues": val[:keep], "vectors": vec[:, :keep],
            "explained": float(val[:keep].sum() / val.sum())}


def pca_transform(fit: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xf = np.where(np.isfinite(X), X, fit["median"])
    return (((Xf - fit["mean"]) / fit["sd"]) @ fit["vectors"]).astype(np.float32)


def pca_fold(X: np.ndarray, train, test, var_share: float = 0.95) -> tuple[dict, np.ndarray, np.ndarray]:
    """Orthogonal features of one fold: PCA fitted on the training rows only, applied to both sides."""
    fit = pca_fit(X[np.asarray(train)], var_share)
    return fit, pca_transform(fit, X[np.asarray(train)]), pca_transform(fit, X[np.asarray(test)])


def weighted_tau(importance, eigen_rank) -> float:
    """8.4.2: hyperbolic weighted Kendall tau between importance and the inverse PCA rank (rank 1 = largest eigenvalue),
    so that concordance among the most important components counts most."""
    return float(weightedtau(np.asarray(importance, dtype=float), 1.0 / np.asarray(eigen_rank, dtype=float))[0])


# ---------------------------------------------------------------------------------------------------- control

def permute_outcomes(rows: pd.DataFrame, seed: int) -> np.ndarray:
    """Control labels: each window's settlement replaced by that of a randomly permuted window; label = the row's side's
    payoff under it."""
    starts = rows["start"].to_numpy(np.int64)
    _, first, inv = np.unique(starts, return_index=True, return_inverse=True)
    up = rows["up_won"].to_numpy(float)[first]
    shuffled = np.random.default_rng(seed).permutation(up)[inv]
    return np.where(rows["side"].to_numpy() > 0, shuffled, 1.0 - shuffled).astype(np.int8)
