#!/usr/bin/env python3
"""Three of the book's warnings, reproduced on SYNTHETIC data with the library in src/pmlab/afml.

1. Leakage (AFML ch. 7): a feature that carries nothing about the label "predicts" it under shuffled k-fold when labels
   overlap in time; the book's purged k-fold with an embargo scores at chance, and a feature that does predict still
   scores. Most of the gap comes from keeping test folds contiguous; purging removes what is left at the fold edges,
   which tests/test_afml_cv.py measures on its own (it fails when purging is switched off; this demo does not).
2. Selection bias (AFML ch. 11, 14): the best of 100 zero-edge strategies has a high in-sample Sharpe ratio; the
   probability of backtest overfitting (CSCV) and the deflated Sharpe ratio both flag it, and both pass a strategy
   with a large planted edge. A moderate edge is printed too, unchecked: there the two measures are less certain.
3. Memory (AFML ch. 5): fractional differencing finds the smallest d at which a random walk passes the ADF test.

Each experiment runs on 10 seeds. Exit 1 if any known answer is missed, so the demo is also a check.
"""
import sys

import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis, skew
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from pmlab import metrics
from pmlab.afml import cpcv, cv, fracdiff


def nn_accuracy(X, y, folds) -> float:
    return float(np.mean([np.mean(KNeighborsClassifier(n_neighbors=1).fit(X[tr], y[tr]).predict(X[te]) == y[te])
                          for tr, te in folds]))


def leakage(rng) -> dict:
    n, h = 1500, 60                                   # labels on 60-step windows that overlap by 59 steps
    e = rng.normal(size=n + h)
    c = np.concatenate([[0.0], np.cumsum(e)])
    window = c[h:h + n] - c[:n]
    y = (window > np.median(window)).astype(int)
    X = lfilter([1.0], [1.0, -0.98], rng.normal(size=(n, 3)), axis=0)   # irrelevant, serially correlated
    signal = (window + rng.normal(0, 0.5, n))[:, None]                    # control: a feature that does predict
    t0 = np.arange(n, dtype=float)
    purged = list(cv.PurgedKFold(10, t0, t0 + h, pct_embargo=0.01).split(X))
    return {"shuffled": nn_accuracy(X, y, KFold(10, shuffle=True, random_state=0).split(X)),
            "purged": nn_accuracy(X, y, purged), "control": nn_accuracy(signal, y, purged)}


def selection(rng, edge: float) -> dict:
    T, N = 1000, 100
    R = rng.normal(0, 0.01, size=(T, N))
    R[:, 0] += edge                                   # strategy 0 carries the planted edge (0 for pure noise)
    sr = R.mean(0) / R.std(0, ddof=1)
    best = int(np.argmax(sr))
    r = R[:, best]
    return {"best_sr_annual": float(sr[best] * np.sqrt(252)),
            "pbo": cpcv.pbo(R, n_blocks=10)["pbo"],
            "dsr": metrics.deflated_sharpe_ratio(sr[best], T, skew(r), kurtosis(r, fisher=False), sr)}


def spread(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} (min {v.min():.3f}, max {v.max():.3f})"


def main() -> int:
    seeds = range(10)
    leak = [leakage(np.random.default_rng(s)) for s in seeds]
    noise = [selection(np.random.default_rng(100 + s), 0.0) for s in seeds]
    moderate = [selection(np.random.default_rng(200 + s), 0.0015) for s in seeds]   # shown, not checked
    real = [selection(np.random.default_rng(400 + s), 0.003) for s in seeds]
    ffd = [fracdiff.min_ffd(np.cumsum(np.random.default_rng(300 + s).normal(size=3000)) + 100)["min_d"] for s in seeds]
    col = lambda rows, k: [r[k] for r in rows]                       # noqa: E731
    print("SYNTHETIC data only; each line is the mean over 10 seeds.\n")
    print("1. Leakage: 1-nearest-neighbour accuracy, chance = 0.5")
    print(f"   irrelevant feature, shuffled 10-fold     {spread(col(leak, 'shuffled'))}")
    print(f"   irrelevant feature, purged + embargo     {spread(col(leak, 'purged'))}")
    print(f"   predictive feature, purged + embargo     {spread(col(leak, 'control'))}  (control)")
    print("2. Best of 100 strategies (1,000 days each)")
    for name, rows in (("all zero edge          ", noise), ("one edge, 0.15% a day  ", moderate),
                       ("one edge, 0.30% a day  ", real)):
        print(f"   {name}  annual SR {np.mean(col(rows, 'best_sr_annual')):.2f}   PBO {spread(col(rows, 'pbo'))}"
              f"   DSR {spread(col(rows, 'dsr'))}")
    print(f"3. Random walk: smallest d that passes ADF at 5%   {spread(ffd)}")
    checks = {
        "shuffled folds leak: mean accuracy > 0.6": np.mean(col(leak, "shuffled")) > 0.6,
        "purged folds do not: mean accuracy within 0.05 of 0.5": abs(np.mean(col(leak, "purged")) - 0.5) < 0.05,
        "control still predicts: mean accuracy > 0.6": np.mean(col(leak, "control")) > 0.6,
        "zero edge: mean PBO > 0.3 and every DSR < 0.95": np.mean(col(noise, "pbo")) > 0.3
                                                           and max(col(noise, "dsr")) < 0.95,
        "0.30% edge: every PBO < 0.1 and every DSR > 0.95": max(col(real, "pbo")) < 0.1 and min(col(real, "dsr")) > 0.95,
        "random walk: every d strictly between 0 and 1": all(d is not None and 0 < d < 1 for d in ffd),
    }
    print()
    for name, ok in checks.items():
        print(f"   {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
