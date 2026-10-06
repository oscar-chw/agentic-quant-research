"""Normality and serial-correlation tests for bar returns.

Normality. Jarque-Bera uses sample skew and kurtosis against an asymptotic chi2(2):
cheap, but only trustworthy for large n. Shapiro-Wilk is the most powerful omnibus
test for small and medium n (scipy's p-value is accurate up to n = 5000, so larger
samples are subsampled). D'Agostino-Pearson combines separate skew and kurtosis
z-tests; it needs n >= 20. A sample is called normal only if no available test rejects.

Dependence. Pearson r measures linear dependence only, and its p-value assumes
approximate bivariate normality; fat tails or a single outlier can dominate it.
Spearman rho and Kendall tau work on ranks, so they catch any monotone dependence and
are robust to fat tails and outliers (Kendall is the more conservative, with better
small-sample behaviour). Pearson is recommended only when both series pass normality.
Ljung-Box tests the joint null of zero autocorrelation at lags 1..h, so it catches
dependence spread thinly across lags that a single lag-1 test misses.

Every p-value here assumes independent observations. Overlapping bars, or samples
taken within one 5-minute window, violate that and make p-values too small.
Testing ~30 bar configurations at once needs a multiplicity correction: see holm.
"""
import numpy as np
from scipy import stats


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def _pair(result) -> tuple[float, float]:
    return float(result[0]), float(result[1])


def normality(x, alpha: float = 0.05) -> dict:
    x = _clean(x)
    n = len(x)
    out = {"n": n, "skew": None, "excess_kurtosis": None, "jarque_bera": None,
           "shapiro": None, "dagostino": None, "normal": None}
    if n < 3:
        return out
    out["skew"] = float(stats.skew(x))
    out["excess_kurtosis"] = float(stats.kurtosis(x))
    out["jarque_bera"] = _pair(stats.jarque_bera(x))
    if n > 5000:
        out["shapiro"] = _pair(stats.shapiro(np.random.default_rng(0).choice(x, 5000, replace=False)))
        out["shapiro_subsampled"] = True
    else:
        out["shapiro"] = _pair(stats.shapiro(x))
    if n >= 20:
        out["dagostino"] = _pair(stats.normaltest(x))
    ps = [t[1] for t in (out["jarque_bera"], out["shapiro"], out["dagostino"]) if t is not None]
    # a constant series gives NaN p-values; "normal" must not default to True then
    out["normal"] = None if any(np.isnan(ps)) else not any(p < alpha for p in ps)
    return out


def correlation(x, y, alpha: float = 0.05) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3:
        return {"n": n, "pearson": None, "spearman": None, "kendall": None,
                "recommended": None, "significant": None}
    both_normal = normality(x, alpha)["normal"] and normality(y, alpha)["normal"]
    out = {"n": n, "pearson": _pair(stats.pearsonr(x, y)),
           "spearman": _pair(stats.spearmanr(x, y)), "kendall": _pair(stats.kendalltau(x, y)),
           "recommended": "pearson" if both_normal else "spearman"}
    out["significant"] = bool(out[out["recommended"]][1] < alpha)
    return out


def serial_correlation(x, lag: int = 1, alpha: float = 0.05) -> dict:
    # NaNs go first, so an interior gap joins its neighbours; pass contiguous series.
    x = _clean(x)
    return {**correlation(x[:-lag], x[lag:], alpha), "lag": lag}


def ljung_box(x, lags: int = 10) -> tuple[float | None, float | None]:
    x = _clean(x)
    n = len(x)
    if n <= lags:
        return None, None
    d = x - x.mean()
    denom = d @ d
    rho = np.array([(d[k:] @ d[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def holm(pvalues, alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down. NaN/None p-values are left out of the family:
    they never reject, get adjusted p NaN, and do not count towards m."""
    p = np.asarray(pvalues, dtype=float)
    idx = np.flatnonzero(~np.isnan(p))
    m = len(idx)
    order = idx[np.argsort(p[idx], kind="stable")]
    stepped = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    adjusted = np.full(len(p), np.nan)
    adjusted[order] = np.minimum(stepped, 1.0)
    reject = [bool(a <= alpha) for a in adjusted]
    return reject, [float(a) for a in adjusted]


def battery(returns, lags: int = 10, alpha: float = 0.05) -> dict:
    x = _clean(returns)
    norm = normality(x, alpha)
    sc = serial_correlation(x, 1, alpha)
    q, q_p = ljung_box(x, lags)
    p_of = lambda t: None if t is None else t[1]
    stat_of = lambda t: None if t is None else t[0]
    return {
        "n": len(x),
        "mean": float(x.mean()) if len(x) else None,
        "std": float(x.std(ddof=1)) if len(x) > 1 else None,
        "skew": norm["skew"], "excess_kurtosis": norm["excess_kurtosis"],
        "jb_p": p_of(norm["jarque_bera"]), "shapiro_p": p_of(norm["shapiro"]),
        "dagostino_p": p_of(norm["dagostino"]), "normal": norm["normal"],
        "lag1_pearson": stat_of(sc["pearson"]), "lag1_pearson_p": p_of(sc["pearson"]),
        "lag1_spearman": stat_of(sc["spearman"]), "lag1_spearman_p": p_of(sc["spearman"]),
        "lag1_kendall": stat_of(sc["kendall"]), "lag1_kendall_p": p_of(sc["kendall"]),
        "recommended": sc["recommended"], "lag1_significant": sc["significant"],
        "ljung_box_q": q, "ljung_box_p": q_p,
    }


def within_group_pairs(x, groups, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(x_t, x_{t−lag}) pairs taken only inside each group, never across a boundary.

    Returns from different 5-minute windows belong to different contracts; pairing them biases
    serial correlation toward zero (research/reviews.md R1.2)."""
    x, groups = np.asarray(x, dtype=float), np.asarray(groups)
    same = np.zeros(len(x), dtype=bool)
    same[lag:] = groups[lag:] == groups[:-lag]
    a, b = x[lag:][same[lag:]], x[:-lag][same[lag:]]
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def ljung_box_grouped(x, groups, lags: int = 10) -> tuple[float | None, float | None]:
    """Box–Pierce form Q = Σ_k n_k ρ_k² with ρ_k from within-group pairs only (χ²_lags)."""
    x = np.asarray(x, dtype=float)
    mu, var = np.nanmean(x), np.nanvar(x)
    if not var > 0:
        return None, None
    q = 0.0
    for k in range(1, lags + 1):
        a, b = within_group_pairs(x, groups, k)
        if len(a) < 3:
            return None, None
        rho = np.mean((a - mu) * (b - mu)) / var
        q += len(a) * rho * rho
    return float(q), float(stats.chi2.sf(q, lags))


def ljung_box_signflip(x, groups, lags: int = 10, n_boot: int = 500, seed: int = 0) -> tuple[float | None, float | None]:
    """Q of ljung_box_grouped on the demeaned values, with its p-value from random sign flips instead of χ²
    (audit R6 finding 8). Flipping every value's sign independently removes any serial correlation but keeps each
    value's size, so the scale differences between windows and the scale profile inside them, which made the χ²
    p-value reject 30–64% of the time with no dependence, stay in the null. Valid when the values are symmetric about
    their mean under H0. p = (1 + #{Q* ≥ Q}) / (n_boot + 1)."""
    x, groups = np.asarray(x, dtype=float), np.asarray(groups)
    ok = np.isfinite(x)
    x, groups = x[ok] - np.mean(x[ok]), groups[ok]
    q0, _ = ljung_box_grouped(x, groups, lags)
    if q0 is None:
        return None, None
    pairs = []
    for k in range(1, lags + 1):
        same = np.zeros(len(x), dtype=bool)
        same[k:] = groups[k:] == groups[:-k]
        i = np.flatnonzero(same)
        pairs.append((i, i - k, x[i] * x[i - k]))
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_boot:
        b = min(64, n_boot - done)
        s = rng.integers(0, 2, (b, len(x)), dtype=np.int8) * 2 - 1
        xs = s * x
        mu, var = xs.mean(axis=1), xs.var(axis=1)
        q = np.zeros(b)
        for i, j, prod in pairs:
            num = (s[:, i] * s[:, j]) @ prod - mu * (xs[:, i].sum(axis=1) + xs[:, j].sum(axis=1)) + len(i) * mu ** 2
            rho = num / len(i) / var
            q += len(i) * rho ** 2
        hits += int(np.sum(q >= q0 - 1e-12 * abs(q0)))
        done += b
    return float(q0), float((1 + hits) / (n_boot + 1))


def cluster_bootstrap_p(x, groups, stat, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Two-sided p-value of stat = 0 by resampling whole groups: (estimate, standard error, p).

    Within-window heteroskedasticity makes textbook p-values too small (R1.5); resampling whole
    windows keeps their dependence."""
    x, groups = np.asarray(x, dtype=float), np.asarray(groups)
    est = stat(x, groups)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq))
        idx = np.concatenate([index[g] for g in pick])
        relabel = np.concatenate([np.full(len(index[g]), i) for i, g in enumerate(pick)])
        draws.append(stat(x[idx], relabel))
    se = float(np.nanstd(draws, ddof=1))
    if not se > 0 or not np.isfinite(est):
        return float(est), se, np.nan
    return float(est), se, float(2 * stats.norm.sf(abs(est) / se))
