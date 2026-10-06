"""Fractionally differentiated features (AFML ch.5).

- ffd_weights: the binomial-series weights of (1 - B)^d, truncated once |w_k| falls below a
  threshold, which gives a fixed-width window (5.5, "fixed-width window fracdiff").
- expanding_weights / frac_diff_expanding: the expanding-window variant with a weight-loss
  threshold (5.4 and 5.5), kept for comparison.
- frac_diff_ffd: the FFD series itself.
- adf: augmented Dickey-Fuller t-statistic with a constant and a fixed lag count, as the book's
  minimum-d search runs it (5.6: lag 1, constant, no automatic lag choice). Critical values are
  MacKinnon (2010)'s response surface; p-values are MacKinnon (1994)'s approximation. statsmodels
  is not a dependency, so both are coded here and checked by Monte Carlo in the tests.
- min_ffd: for a grid of d, the ADF statistic of the FFD series and its correlation with the
  original series; the minimum d is the smallest d whose statistic is below the 5% critical value
  (5.6). The table also gives the sum of the truncated weights: a non-zero sum keeps that multiple of
  the level in the FFD series.

Weights are indexed by lag: w[0] multiplies the current value, w[k] the value k steps back.
"""
import numpy as np
from scipy.stats import norm

# MacKinnon (2010) response surface for one variable: crit = b0 + b1/T + b2/T^2 + b3/T^3
_CRIT = {
    "c": {0.01: (-3.43035, -6.5393, -16.786, -79.433),
          0.05: (-2.86154, -2.8903, -4.234, -40.040),
          0.10: (-2.56677, -1.5384, -2.809, 0.0)},
    "ct": {0.01: (-3.95877, -9.0531, -28.428, -134.155),
           0.05: (-3.41049, -4.3904, -9.036, -45.374),
           0.10: (-3.12705, -2.5856, -3.925, -22.380)},
}
# MacKinnon (1994) p-value approximation, one variable, constant only
_P_C = {"max": 2.74, "min": -18.83, "star": -1.61,
        "small": (2.1659, 1.4412, 0.038269), "large": (1.7339, 0.93202, -0.12745, -0.010368)}


def expanding_weights(d: float, size: int) -> np.ndarray:
    """w_0 = 1, w_k = -w_{k-1} (d - k + 1) / k for k < size (5.4)."""
    w = np.empty(int(size))
    w[0] = 1.0
    for k in range(1, int(size)):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def ffd_weights(d: float, thres: float = 1e-5, max_width: int = 1_000_000) -> np.ndarray:
    """Weights of (1 - B)^d until |w_k| < thres (5.5). An integer d ends exactly (w_k = 0)."""
    w = [1.0]
    k = 1
    while k < max_width:
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < thres:
            break
        w.append(nxt)
        k += 1
    return np.asarray(w)


def frac_diff_ffd(x, d: float, thres: float = 1e-5) -> np.ndarray:
    """X~_t = sum_k w_k X_{t-k} over the fixed window; NaN until the window is full or when it
    holds a non-finite value (5.5)."""
    x = np.asarray(x, dtype=float)
    w = ffd_weights(d, thres)
    out = np.full(len(x), np.nan)
    width = len(w)
    if len(x) < width:
        return out
    # correlate: out[t] = sum_k w[k] x[t - k]
    out[width - 1:] = np.convolve(x, w, mode="valid")
    return out


def frac_diff_expanding(x, d: float, thres: float = 0.01) -> np.ndarray:
    """Expanding-window fracdiff (5.4): skip the first observations whose relative weight loss
    (share of the total absolute weight not yet in the window) exceeds thres."""
    x = np.asarray(x, dtype=float)
    w = expanding_weights(d, len(x))
    cum = np.cumsum(np.abs(w))
    loss = 1 - cum / cum[-1]
    skip = int(np.sum(loss > thres))
    out = np.full(len(x), np.nan)
    for t in range(skip, len(x)):
        out[t] = np.dot(w[:t + 1], x[t::-1])
    return out


def adf_critical(nobs: int, level: float = 0.05, regression: str = "c") -> float:
    b = _CRIT[regression][level]
    return float(b[0] + b[1] / nobs + b[2] / nobs ** 2 + b[3] / nobs ** 3)


def adf_pvalue(stat: float) -> float:
    """MacKinnon (1994) approximate p-value, constant, one variable."""
    if not np.isfinite(stat):
        return np.nan
    if stat > _P_C["max"]:
        return 1.0
    if stat < _P_C["min"]:
        return 0.0
    c = _P_C["small"] if stat <= _P_C["star"] else _P_C["large"]
    return float(norm.cdf(sum(ci * stat ** i for i, ci in enumerate(c))))


def adf(x, lags: int = 1, regression: str = "c") -> dict:
    """ADF regression dx_t = a [+ b t] + g x_{t-1} + sum_{i<=lags} c_i dx_{t-i} + e; t-statistic of g.

    NaNs must be removed by the caller (a gap would join unrelated observations)."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("adf needs a finite series; drop NaNs first")
    dx = np.diff(x)
    n = len(dx) - lags
    if n < lags + 4:
        return {"stat": np.nan, "pvalue": np.nan, "crit5": np.nan, "nobs": max(n, 0), "lags": lags}
    y = dx[lags:]
    cols = [np.ones(n), x[lags:-1]]
    cols += [dx[lags - i:-i] for i in range(1, lags + 1)]
    if regression == "ct":
        cols.insert(1, np.arange(n, dtype=float))
    X = np.column_stack(cols)
    j = 2 if regression == "ct" else 1
    xtx = X.T @ X
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return {"stat": np.nan, "pvalue": np.nan, "crit5": np.nan, "nobs": n, "lags": lags}
    beta = inv @ (X.T @ y)
    resid = y - X @ beta
    s2 = resid @ resid / (n - X.shape[1])
    se = np.sqrt(s2 * inv[j, j])
    stat = float(beta[j] / se) if se > 0 else np.nan
    return {"stat": stat, "pvalue": adf_pvalue(stat) if regression == "c" else np.nan,
            "crit1": adf_critical(n, 0.01, regression), "crit5": adf_critical(n, 0.05, regression),
            "crit10": adf_critical(n, 0.10, regression), "nobs": n, "lags": lags}


def min_ffd(x, ds=np.linspace(0, 1, 11), thres: float = 1e-5, lags: int = 1) -> dict:
    """Table over d of the FFD series' ADF statistic and its correlation with x (5.6), and the
    smallest d whose statistic is below the 5% critical value (None if none passes)."""
    x = np.asarray(x, dtype=float)
    rows = []
    for d in ds:
        y = frac_diff_ffd(x, float(d), thres)
        ok = np.isfinite(y) & np.isfinite(x)
        r = adf(y[ok], lags) if ok.sum() > 10 else {"stat": np.nan, "crit5": np.nan, "pvalue": np.nan, "nobs": 0}
        corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and np.std(y[ok]) > 0 else np.nan
        w = ffd_weights(float(d), thres)
        rows.append({"d": float(d), "width": len(w), "weight_sum": float(w.sum()), "adf": r["stat"],
                     "pvalue": r.get("pvalue"), "crit5": r["crit5"], "nobs": r["nobs"], "corr": corr})
    passing = [r for r in rows if np.isfinite(r["adf"]) and r["adf"] < r["crit5"]]
    best = min(passing, key=lambda r: r["d"]) if passing else None
    return {"table": rows, "min_d": None if best is None else best["d"],
            "corr_at_min_d": None if best is None else best["corr"]}
