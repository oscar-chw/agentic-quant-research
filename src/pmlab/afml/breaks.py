"""Structural breaks (AFML ch.17).

CUSUM on recursive residuals, Brown-Durbin-Evans (17.3.1):
- brown_durbin_evans: 1-step-ahead residuals of y on X refitted on [0, t) for every t >= k (k regressors),
  standardised by sqrt(1 + x_t'(X'X)^-1 x_t); S_t = cumulative sum / their standard deviation, so under a
  constant beta S_t ~ N(0, t - k + 1). The book indexes from its first fitted subsample; the statistic is
  the same. Flag: |S_t| crosses the 5% boundaries +-a (sqrt(T - k) + 2 (t - k + 1) / sqrt(T - k)),
  a = 0.948 (Brown, Durbin and Evans 1975), which control the whole path rather than each t.

CUSUM on levels, Chu-Stinchcombe-White form (17.3.2, as simplified by Homm and Breitung 2012):
- S_{n,t} = (y_t - y_n) / (sigma_t sqrt(t - n)) with sigma_t^2 the mean squared first difference up to
  t; under a driftless null S_{n,t} is standard normal. The one-sided critical value is
  c_alpha[n, t] = sqrt(b_alpha + ln(t - n)), with b_0.05 = 4.6.
- csw_cusum returns, for every t, the statistic from the first observation (n = 1, the original test)
  and the supremum over backward-shifting reference points n in [1, t) that the book proposes. Upward
  breaks by default; two_sided uses |S|. Deviation: the book compares the supremum with the pointwise
  critical value, which almost every random walk exceeds; the supremum of S / c is compared here with
  a Monte Carlo quantile under the null (csw_sup_critical).
  Second deviation (docs/afml_sections/ch17.md F17.2): two_sided compares |S_{n,t}| with the SAME
  c_alpha, which is the one-sided critical value, so flag_first then rejects on either tail and has
  about twice the nominal pointwise size (measured on 288-step random walks: about 1.8% one-sided
  against 3.5% two-sided, tests/test_afml_breaks.py::test_the_two_sided_first_flag_has_about_twice_the_one_sided_size).
  daily_flags reports that flag. The supremum flags are unaffected: csw_sup_critical simulates its
  quantiles with the same two_sided setting.

Explosiveness, supremum ADF (17.4.2):
- BSADF_t = sup over start points of the ADF statistic on [start, t], every window holding at least
  min_len regression rows; SADF = sup_t BSADF_t. The regression is dy on the lagged level, `lags`
  lagged differences and a constant, as in the book's snippets. All windows ending at t are solved at
  once from prefix sums of the regression cross-products (exact, O(T^2) small solves in total).
  Deviation (docs/afml_sections/ch17.md F17.3): min_len counts REGRESSION ROWS, so every window fits
  min_len rows whatever `lags` is, while the snippet's start points leave minSL - lags rows once lags
  differences are consumed. At lags = 0 the two are the same window set; above it the shortest windows
  here are longer, so the statistics differ (by up to about 0.32 at lags 2 on the series of
  research/book_check_part3.md). Every call in this project uses lags 0.
- sadf_critical: Monte Carlo quantiles of SADF under a Gaussian random walk of the same length, since
  the statistic's distribution depends on the sample and window sizes.
- wild_bootstrap_critical: critical values from the series' own demeaned increments with random signs, a
  unit-root null that keeps time-varying variance. Deviation from the book, which uses fixed critical
  values: volatility clustering alone makes Gaussian random-walk critical values flag intraday BTC.
- daily_flags: per day, the CSW flags and the SADF flag against both kinds of 95% critical values.

Chow-type Dickey-Fuller (17.4.1):
- chow_dfc: dy_t = delta y_(t-1) D_t + e_t with D_t = 1 from the break on, no constant, for every break date
  in [tau0 n, (1 - tau0) n); DFC = delta / se(delta); SDFC = its supremum. The level is taken relative to the
  first observation (the regression has no constant, so the origin would otherwise matter).
  chow_dfc_critical simulates the supremum under a Gaussian random walk.

Cost of the double loop (17.4.2(2)):
- adf_flops, sadf_test_count, sadf_flops: Table 17.1's operation count per ADF estimate and the counts it
  implies for one SADF update and for a whole series. The book's worked case (N = 3, T = 356631, tau = 1000)
  is reproduced exactly, so the number that decides whether a series is affordable is computed, not guessed.

Robust ADF extrema (17.4.2(4)-5):
- adf_quantiles: for each end t, the ADF statistics over every admissible start s_t (the set whose supremum
  is BSADF_t); Q_(t,q) = the q quantile of s_t and Qdot = Q_(t,q+v) - Q_(t,q-v) (QADF); C_(t,q) = the mean of
  s_t at or above Q_(t,q) and Cdot = their standard deviation (CADF). The book writes Cdot as the tail's second
  moment; its square root is used so Cdot is in the statistic's units, as Qdot is. QSADF and CSADF are the
  suprema over t. Q_(t,1) = BSADF_t.

Sub- and super-martingale tests (17.4.3):
- smt: for every end t and start t0 with at least min_len points, the fit of one trend specification on the
  window with local time i = 1..n: poly1 y = a + g i + b i^2, poly2 log y = a + g i + b i^2, exp log y = a + b i,
  power log y = a + b log i; SMT_t = sup over t0 of |b| / (se(b) n^phi), phi in [0, 1] penalising long
  windows. All windows sharing a start are solved at once from cumulative cross-products.
  Deviation (docs/afml_sections/ch17.md F17.4): the clock restarts at each window (i = 1..n) and n = t - t0 + 1.
  poly1, poly2 and exp are unaffected, their regressors being affine in i, so an affine change of clock leaves
  their t-statistics alone; SM-Power is NOT, because log of local time is not log of series time plus a constant,
  so a late window gives a different statistic from a regression on the series' own time index.
  smt_critical simulates the supremum under a random walk (a geometric one for the log specifications).
"""
import numpy as np


def csw_cusum(y, b_alpha: float = 4.6, two_sided: bool = False, sup_critical: float = 1.0) -> dict:
    """Per t: S_{1,t} and its flag; the backward-shifting supremum of S_{n,t} and the supremum of the
    ratio S_{n,t} / c_alpha[n, t]. flag_sup marks a ratio above `sup_critical`: 1.0 is the book's
    pointwise critical value, which a supremum over n exceeds almost surely under the null (a Monte
    Carlo check in the tests gives ~99% false alarms on 300-step random walks), so calibrated use passes
    the q95 of csw_sup_critical instead."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    dy2 = np.concatenate([[0.0], np.diff(y) ** 2])
    cum = np.cumsum(dy2)
    s_first = np.full(T, np.nan)
    s_sup = np.full(T, np.nan)
    ratio_sup = np.full(T, np.nan)
    n_sup = np.full(T, -1)
    flag_first = np.zeros(T, bool)
    for t in range(1, T):                       # 0-based t: y[0..t] holds t first differences
        var = cum[t] / t                         # mean squared difference up to t
        if var <= 0:
            continue
        n = np.arange(t)
        gap = t - n
        s = (y[t] - y[n]) / (np.sqrt(var) * np.sqrt(gap))
        s = np.abs(s) if two_sided else s
        crit = np.sqrt(b_alpha + np.log(gap))
        s_first[t] = s[0]
        flag_first[t] = s[0] > crit[0]
        k = int(np.argmax(s))
        s_sup[t], n_sup[t] = s[k], k
        ratio_sup[t] = np.max(s / crit)
    finite = ratio_sup[np.isfinite(ratio_sup)]
    return {"s_first": s_first, "flag_first": flag_first, "s_sup": s_sup, "n_sup": n_sup, "ratio_sup": ratio_sup,
            "flag_sup": np.nan_to_num(ratio_sup, nan=-np.inf) > sup_critical,
            "max_ratio": float(finite.max()) if len(finite) else np.nan}


def csw_sup_critical(n_obs: int, n_sim: int = 300, q=(0.9, 0.95, 0.99), two_sided: bool = False, rng=None) -> dict:
    """Monte Carlo quantiles of max_t max_n S_{n,t} / c_0.05[n, t] under a Gaussian random walk."""
    rng = np.random.default_rng(rng)
    stats = np.array([csw_cusum(np.cumsum(rng.standard_normal(n_obs)), two_sided=two_sided)["max_ratio"]
                      for _ in range(n_sim)])
    return {f"q{int(round(x * 100))}": float(np.quantile(stats, x)) for x in q} | {"n_sim": n_sim, "n_obs": n_obs}


def _design(y, lags: int):
    """Rows of the ADF regression: target dy_s, regressors [level_{s-1}, dy_{s-1..s-lags}, 1]."""
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    m = len(dy) - lags
    target = dy[lags:]
    cols = [y[lags:-1]] + [dy[lags - i:len(dy) - i] for i in range(1, lags + 1)] + [np.ones(m)]
    return target, np.column_stack(cols)


def _adf_by_end(y, min_len: int, lags: int):
    """Yield (end row e, start rows a, ADF t-statistics) for every end with at least one admissible window;
    -inf marks a singular window. Also returns the row count first (a generator's first item)."""
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("bsadf needs a finite series")
    sd = np.std(np.diff(y))
    z = (y - y[0]) / (sd if sd > 0 else 1.0)          # ADF t-statistics do not change under this rescaling
    target, X = _design(z, lags)
    m, k = X.shape
    yield m
    if m < min_len or min_len <= k:
        return
    C = np.concatenate([np.zeros((1, k, k)), np.cumsum(X[:, :, None] * X[:, None, :], axis=0)])
    D = np.concatenate([np.zeros((1, k)), np.cumsum(X * target[:, None], axis=0)])
    E = np.concatenate([[0.0], np.cumsum(target ** 2)])
    for e in range(min_len - 1, m):
        a = np.arange(0, e - min_len + 2)
        xtx = C[e + 1] - C[a]
        xty = D[e + 1] - D[a]
        yy = E[e + 1] - E[a]
        n_rows = e + 1 - a
        with np.errstate(all="ignore"):
            det = np.linalg.det(xtx)
            singular = ~np.isfinite(det) | (np.abs(det) <= 1e-10 * np.prod(np.maximum(np.diagonal(xtx, axis1=1, axis2=2), 1e-300), axis=1))
            xtx[singular] = np.eye(k)                    # e.g. a flat stretch: no regression, statistic -inf
            inv = np.linalg.inv(xtx)
            beta = np.einsum("nij,nj->ni", inv, xty)
            rss = yy - np.einsum("ni,ni->n", beta, xty)
            s2 = rss / (n_rows - k)
            t = beta[:, 0] / np.sqrt(s2 * inv[:, 0, 0])
        yield e, a, np.where(np.isfinite(t) & ~singular, t, -np.inf)


def bsadf(y, min_len: int = 20, lags: int = 0) -> dict:
    """BSADF_t for each regression row t (NaN until a window of min_len rows fits), SADF, and the
    start of the best window. Row t of the regression uses dy at original index t + lags + 1."""
    gen = _adf_by_end(y, min_len, lags)
    m = next(gen)
    out = np.full(m, np.nan)
    best_start = np.full(m, -1)
    for e, a, t in gen:
        j = int(np.argmax(t))
        out[e], best_start[e] = t[j], a[j]
    finite = out[np.isfinite(out)]
    return {"bsadf": out, "sadf": float(finite.max()) if len(finite) else np.nan, "start": best_start, "rows": m}


def adf_quantiles(y, min_len: int = 20, lags: int = 0, q: float = 0.95, v: float = 0.025) -> dict:
    """QADF (Q_(t,q), Qdot_(t,q,v)) and CADF (C_(t,q), Cdot_(t,q)) per regression row, over the finite ADF
    statistics of every admissible start; QSADF and CSADF are their suprema."""
    gen = _adf_by_end(y, min_len, lags)
    m = next(gen)
    out = {name: np.full(m, np.nan) for name in ("q", "qdot", "c", "cdot", "bsadf")}
    for e, _, t in gen:
        out["bsadf"][e] = t.max()
        t = t[np.isfinite(t)]
        if not len(t):
            continue
        qv = np.quantile(t, q)
        tail = t[t >= qv]
        out["q"][e] = qv
        out["qdot"][e] = np.quantile(t, min(q + v, 1.0)) - np.quantile(t, max(q - v, 0.0))
        out["c"][e], out["cdot"][e] = tail.mean(), tail.std()
    sup = lambda x: float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
    return out | {"qsadf": sup(out["q"]), "csadf": sup(out["c"]), "sadf": sup(out["bsadf"]), "rows": m, "quantile": q, "v": v}


def sadf_critical(n_obs: int, min_len: int, lags: int = 0, n_sim: int = 500, q=(0.9, 0.95, 0.99), rng=None) -> dict:
    rng = np.random.default_rng(rng)
    stats = np.array([bsadf(np.cumsum(rng.standard_normal(n_obs)), min_len, lags)["sadf"] for _ in range(n_sim)])
    return {f"q{int(round(x * 100))}": float(np.quantile(stats, x)) for x in q} | {"n_sim": n_sim, "n_obs": n_obs,
                                                                                   "min_len": min_len, "lags": lags}


def wild_bootstrap_critical(y, statistic, n_sim: int = 199, q=(0.95,), rng=None) -> dict:
    """Quantiles of statistic(y*) with y* = y_0 + cumsum((dy_t - mean dy) w_t), w_t = +-1 with equal odds: a unit-root
    null that keeps the sample's own time-varying variance (the wild bootstrap of Harvey, Leybourne, Sollis and
    Taylor 2016). Gaussian random-walk critical values assume constant variance, which intraday prices violate."""
    rng = np.random.default_rng(rng)
    y = np.asarray(y, dtype=float)
    e = np.diff(y) - np.mean(np.diff(y))
    sims = np.array([statistic(np.concatenate([[y[0]], y[0] + np.cumsum(e * rng.choice([-1.0, 1.0], len(e)))]))
                     for _ in range(n_sim)])
    return {f"q{int(round(x * 100))}": float(np.quantile(sims, x)) for x in q} | {"n_sim": n_sim}


def daily_flags(series_by_day: dict, min_len: int, lags: int = 0, n_sim: int = 300, wild: bool = True, rng=None) -> dict:
    """{day: flags} for level series keyed by day: two-sided CSW from the day's first observation, the
    backward-shifting CSW supremum and SADF, each against a 95% value simulated under a Gaussian random walk
    (once per length) and, with wild=True, against the day's own wild-bootstrap 95% value."""
    rng = np.random.default_rng(rng)
    crit: dict = {}
    out = {}
    for day, y in series_by_day.items():
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) not in crit:
            crit[len(y)] = (sadf_critical(len(y), min_len, lags, n_sim, rng=rng)["q95"],
                            csw_sup_critical(len(y), n_sim, two_sided=True, rng=rng)["q95"])
        sadf_c, csw_c = crit[len(y)]
        c = csw_cusum(y, two_sided=True)
        s = bsadf(y, min_len, lags)
        row = {"n": int(len(y)), "csw_first_flag": bool(c["flag_first"].any()),
               "csw_sup_ratio": c["max_ratio"], "csw_sup_crit95": csw_c, "csw_sup_flag": bool(c["max_ratio"] > csw_c),
               "sadf": s["sadf"], "sadf_crit95": sadf_c, "sadf_flag": bool(s["sadf"] > sadf_c)}
        if wild:
            ws = wild_bootstrap_critical(y, lambda z: bsadf(z, min_len, lags)["sadf"], n_sim, rng=rng)["q95"]
            wc = wild_bootstrap_critical(y, lambda z: csw_cusum(z, two_sided=True)["max_ratio"], n_sim, rng=rng)["q95"]
            row.update(sadf_wild_crit95=ws, sadf_wild_flag=bool(s["sadf"] > ws),
                       csw_sup_wild_crit95=wc, csw_sup_wild_flag=bool(c["max_ratio"] > wc))
        out[day] = row
    return out


def brown_durbin_evans(y, X=None, a: float = 0.948) -> dict:
    """Recursive residuals, their CUSUM, the 5% boundaries and whether the CUSUM crosses them (module docstring)."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    X = np.ones((T, 1)) if X is None else np.asarray(X, dtype=float).reshape(T, -1)
    k = X.shape[1]
    w = np.full(T, np.nan)
    for t in range(k, T):
        xtx = X[:t].T @ X[:t]
        if np.linalg.matrix_rank(xtx) < k:
            continue
        inv = np.linalg.inv(xtx)
        beta = inv @ (X[:t].T @ y[:t])
        w[t] = (y[t] - X[t] @ beta) / np.sqrt(1.0 + X[t] @ inv @ X[t])
    ok = np.isfinite(w)
    cusum, bound = np.full(T, np.nan), np.full(T, np.nan)
    n = int(ok.sum())
    sigma = np.std(w[ok]) if n > 1 else np.nan
    if n > 1 and sigma > 0:
        j = np.arange(1, n + 1)
        cusum[ok] = np.cumsum(w[ok]) / sigma
        bound[ok] = a * (np.sqrt(n) + 2 * j / np.sqrt(n))
    ratio = np.abs(cusum) / bound
    max_ratio = float(np.nanmax(ratio)) if np.isfinite(ratio).any() else np.nan
    return {"residuals": w, "cusum": cusum, "bound": bound, "max_ratio": max_ratio,
            "flag": bool(np.nan_to_num(max_ratio, nan=0.0) > 1.0), "k": k}


def chow_dfc(y, tau0: float = 0.15) -> dict:
    """DFC at every admissible break date (index into the first differences) and SDFC (module docstring)."""
    y = np.asarray(y, dtype=float)
    z = y - y[0]
    dy, lag = np.diff(z), z[:-1]
    n = len(dy)
    taus = np.arange(int(np.floor(tau0 * n)), int(np.ceil((1 - tau0) * n)))
    sxy = np.cumsum((lag * dy)[::-1])[::-1]
    sxx = np.cumsum((lag * lag)[::-1])[::-1]
    syy = float(dy @ dy)
    with np.errstate(all="ignore"):
        d = sxy[taus] / sxx[taus]
        s2 = (syy - d * sxy[taus]) / (n - 1)
        dfc = d / np.sqrt(s2 / sxx[taus])
    dfc = np.where(np.isfinite(dfc), dfc, -np.inf)
    j = int(np.argmax(dfc)) if len(dfc) else -1
    return {"breaks": taus, "dfc": dfc, "sdfc": float(dfc[j]) if j >= 0 else np.nan, "break_at": int(taus[j]) if j >= 0 else -1}


def chow_dfc_critical(n_obs: int, tau0: float = 0.15, n_sim: int = 500, q=(0.9, 0.95, 0.99), rng=None) -> dict:
    rng = np.random.default_rng(rng)
    stats = np.array([chow_dfc(np.cumsum(rng.standard_normal(n_obs)), tau0)["sdfc"] for _ in range(n_sim)])
    return {f"q{int(round(x * 100))}": float(np.quantile(stats, x)) for x in q} | {"n_sim": n_sim, "n_obs": n_obs}


SMT_SPECS = ("poly1", "poly2", "exp", "power")


def smt(y, spec: str = "poly1", min_len: int = 20, phi: float = 0.0) -> dict:
    """SMT_t per end point, its supremum and the start of the best window (module docstring)."""
    if spec not in SMT_SPECS:
        raise ValueError(f"spec must be one of {SMT_SPECS}")
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("smt needs a finite series")
    if spec != "poly1" and np.any(y <= 0):
        raise ValueError(f"{spec} fits log y and needs a positive series")
    target = y if spec == "poly1" else np.log(y)
    T = len(y)
    out, best_start = np.full(T, -np.inf), np.full(T, -1)
    for a in range(0, T - min_len + 1):
        n_max = T - a
        i = np.arange(1, n_max + 1, dtype=float)
        u = i / n_max                                       # t-statistics do not change with the time unit
        X = {"poly1": np.column_stack([np.ones(n_max), u, u * u]), "poly2": np.column_stack([np.ones(n_max), u, u * u]),
             "exp": np.column_stack([np.ones(n_max), u]), "power": np.column_stack([np.ones(n_max), np.log(i)])}[spec]
        k = X.shape[1]
        yy = target[a:]
        C = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)[min_len - 1:]
        D = np.cumsum(X * yy[:, None], axis=0)[min_len - 1:]
        E = np.cumsum(yy * yy)[min_len - 1:]
        n = np.arange(min_len, n_max + 1, dtype=float)
        with np.errstate(all="ignore"):
            inv = np.linalg.inv(C)
            beta = np.einsum("nij,nj->ni", inv, D)
            rss = E - np.einsum("ni,ni->n", beta, D)
            se = np.sqrt(np.maximum(rss, 0.0) / (n - k) * inv[:, -1, -1])
            stat = np.abs(beta[:, -1]) / (se * n ** phi)
        stat = np.where(np.isfinite(stat), stat, -np.inf)
        ends = a + np.arange(min_len, n_max + 1) - 1
        better = stat > out[ends]
        out[ends[better]], best_start[ends[better]] = stat[better], a
    out = np.where(np.isfinite(out), out, np.nan)
    return {"smt": out, "sup": float(np.nanmax(out)) if np.isfinite(out).any() else np.nan, "start": best_start,
            "spec": spec, "phi": phi}


def smt_critical(n_obs: int, spec: str = "poly1", min_len: int = 20, phi: float = 0.0, n_sim: int = 300,
                 q=(0.9, 0.95, 0.99), rng=None) -> dict:
    rng = np.random.default_rng(rng)
    draw = lambda: np.cumsum(rng.standard_normal(n_obs))
    stats = np.array([smt(draw() if spec == "poly1" else np.exp(0.01 * draw()), spec, min_len, phi)["sup"]
                      for _ in range(n_sim)])
    return {f"q{int(round(x * 100))}": float(np.quantile(stats, x)) for x in q} | {"n_sim": n_sim, "n_obs": n_obs}


def adf_flops(n_regressors: int, n_rows: int) -> int:
    """Floating point operations of one ADF estimate, Table 17.1 summed:
    f(N, T) = N^3 + N^2 (2T + 3) + N (4T - 1) + 2T + 2."""
    n, t = int(n_regressors), int(n_rows)
    return n ** 3 + n ** 2 * (2 * t + 3) + n * (4 * t - 1) + 2 * t + 2


def sadf_test_count(n_rows: int, min_len: int) -> int:
    """ADF fits behind one SADF series, sum_(t=tau)^T (t - tau + 1) = (T - tau + 2)(T - tau + 1) / 2 (17.4.2(2))."""
    d = int(n_rows) - int(min_len)
    return (d + 2) * (d + 1) // 2 if d >= 0 else 0


def sadf_flops(n_regressors: int, n_rows: int, min_len: int) -> dict:
    """The cost model of 17.4.2(2): `update` is g(N, T, tau) = sum_(t=tau)^T f(N, t) + (T - tau) operations for one
    SADF_T (the backward-expanding fits plus the search for their maximum); `series` is sum_(t=tau)^T g(N, t, tau),
    the whole {SADF_t} series; `tests` is the number of ADF fits in it."""
    n, t_max, tau = int(n_regressors), int(n_rows), int(min_len)
    a, b = n ** 3 + 3 * n ** 2 - n + 2, 2 * n ** 2 + 4 * n + 2      # f(N, t) = a + b t
    inner = lambda t: (t - tau + 1) * a + b * (tau + t) * (t - tau + 1) // 2      # sum_(s=tau)^t f(N, s)
    update = lambda t: inner(t) + (t - tau)
    return {"per_adf": adf_flops(n, t_max), "update": update(t_max),
            "series": sum(update(t) for t in range(tau, t_max + 1)),
            "tests": sadf_test_count(t_max, tau)}
