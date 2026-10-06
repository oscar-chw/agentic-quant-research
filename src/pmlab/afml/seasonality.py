"""Clock profiles with inference over whole days (AFML ch.2: does activity or the clock carry information?).

A profile is a matrix M with one row per period (a day for hour-of-day cells, an ISO week for weekday cells) and one
column per cell, each entry the mean of a per-window value over that period's cell. Windows are never treated as
independent: within a day, 5-minute windows share volatility and participants, and neighbouring days share regimes
(research/audit_r6_findings.md finding 1). Every interval and test here resamples whole periods in circular moving
blocks of `L` consecutive periods, with L measured on the data by the variance-inflation curve of the period series.

- local_hour, local_day, weekday: clock labels at a fixed UTC offset (Taiwan: +8 h, no daylight saving).
- cell_matrix: (period, cell, value[, weight]) -> periods × cells matrix of (weighted) means.
- vif_curve, choose_block: Var(block means) × values per block ÷ Var(values), averaged over cells, for each
  candidate block length; the shortest length reaching `frac` of the largest (the plateau), with at least
  `min_blocks` blocks.
- block_draws: circular moving-block bootstrap indices of periods.
- level_ci: per cell, the mean over periods and a t interval (G − 1 df, G = periods // L) with the block-bootstrap
  standard error.
- profile_test: H0 "every cell has the same mean". Rows are centred on their own mean, so level shocks shared by a
  whole period (a volatile day) cancel; the statistic is Σ_k ē_k² / v_k with v_k the variance of column k over rows
  divided by the rows; its null distribution comes from the recentred rows (ē subtracted) resampled in blocks. Per
  cell, the contrast ē_k with its bootstrap standard error and a two-sided t(G − 1) p-value.
- within_beta, adjusted_profile: log value = β · log activity + period effect + cell effect; β from the rows centred
  on their means (pooled OLS), the residual profile and the share of the clock profile's variance that activity
  explains, with a block-bootstrap interval (β re-estimated in every draw).
- bh: Benjamini–Hochberg adjusted p-values.
"""
import numpy as np
from scipy.stats import t as student_t

TAIWAN = 8 * 3600
BLOCK_DAYS = (1, 2, 3, 5, 7, 10, 14)
BLOCK_WEEKS = (1, 2, 3, 4)


def local_hour(sec, offset: int = 0) -> np.ndarray:
    return (np.asarray(sec, dtype=np.int64) + offset) // 3600 % 24


def local_day(sec, offset: int = 0) -> np.ndarray:
    """Days since 1970-01-01 on the clock `offset` seconds ahead of UTC."""
    return (np.asarray(sec, dtype=np.int64) + offset) // 86400


def weekday(day) -> np.ndarray:
    """Monday = 0 … Sunday = 6 for a day index from local_day (1970-01-01 was a Thursday)."""
    return (np.asarray(day, dtype=np.int64) + 3) % 7


def iso_week(day) -> np.ndarray:
    """Weeks since Monday 1969-12-29, so a week runs Monday to Sunday."""
    return (np.asarray(day, dtype=np.int64) + 3) // 7


def cell_matrix(period, cell, values, n_cells: int, weights=None, min_count: int = 1):
    """(sorted period ids, periods × n_cells weighted means, counts). A cell with fewer than `min_count` finite
    values, or zero total weight, is NaN."""
    period, cell = np.asarray(period, dtype=np.int64), np.asarray(cell, dtype=np.int64)
    x = np.asarray(values, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    ids, row = np.unique(period[ok], return_inverse=True)
    flat = row * n_cells + cell[ok]
    size = len(ids) * n_cells
    sw = np.bincount(flat, w[ok], size)
    swx = np.bincount(flat, w[ok] * x[ok], size)
    n = np.bincount(flat, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where((n >= min_count) & (sw != 0), swx / np.where(sw != 0, sw, 1.0), np.nan)
    return ids, m.reshape(len(ids), n_cells), n.reshape(len(ids), n_cells)


def vif_curve(M, period_ids, lengths=BLOCK_DAYS) -> dict:
    """{L: (mean over cells of the variance inflation at block length L, blocks)}; NaN with fewer than 3 blocks."""
    M = np.asarray(M, dtype=float)
    pid = np.asarray(period_ids, dtype=np.int64)
    out = {}
    for L in lengths:
        blocks = (pid - pid.min()) // L
        vifs = []
        for k in range(M.shape[1]):
            x, b = M[:, k], blocks
            ok = np.isfinite(x)
            x, b = x[ok], b[ok]
            _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
            base = np.var(x, ddof=1) if len(x) > 1 else np.nan
            if len(cnt) >= 3 and base > 0:
                means = np.bincount(inv, x) / cnt
                vifs.append(np.var(means, ddof=1) * cnt.mean() / base)
        n_blocks = len(np.unique(blocks))
        out[L] = (float(np.mean(vifs)) if vifs else np.nan, int(n_blocks))
    return out


def choose_block(M, period_ids, lengths=BLOCK_DAYS, frac: float = 0.95, min_blocks: int = 12) -> dict:
    """The shortest block length whose variance inflation reaches `frac` × the largest over the admissible lengths
    (those leaving at least `min_blocks` blocks). Never below 1 period."""
    curve = vif_curve(M, period_ids, lengths)
    ok = {L: v for L, (v, g) in curve.items() if g >= min_blocks and np.isfinite(v)}
    if not ok:
        return {"length": 1, "vif": np.nan, "curve": {str(L): v for L, (v, _) in curve.items()}}
    top = max(ok.values())
    L = next(L for L in lengths if L in ok and ok[L] >= frac * top) if top > 1 else min(ok)
    return {"length": int(L), "vif": float(ok[L]), "curve": {str(k): v for k, (v, _) in curve.items()}}


def block_draws(n: int, L: int, B: int, rng) -> np.ndarray:
    """B × n row indices: ceil(n / L) circular blocks of L consecutive rows with uniform starts, cut to n."""
    L = max(1, min(int(L), n))
    k = -(-n // L)
    starts = rng.integers(0, n, (B, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    return idx.reshape(B, k * L)[:, :n]


def level_ci(M, L: int, B: int = 1999, seed: int = 0, level: float = 0.95, chunk: int = 256) -> dict:
    """Per column: mean over rows (NaN rows skipped), block-bootstrap standard error, t interval with G − 1 df."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    mean = np.nanmean(M, axis=0)
    rng = np.random.default_rng(seed)
    draws = block_draws(n, L, B, rng)
    boot = np.concatenate([np.nanmean(M[draws[i:i + chunk]], axis=1) for i in range(0, B, chunk)])
    se = np.nanstd(boot, axis=0, ddof=1)
    G = max(n // max(int(L), 1), 2)
    q = student_t.ppf(0.5 + level / 2, G - 1)
    return {"mean": mean, "se": se, "lo": mean - q * se, "hi": mean + q * se, "periods": int(n), "blocks": int(G),
            "block": int(L), "counts": np.isfinite(M).sum(axis=0)}


def _q_stat(E: np.ndarray) -> np.ndarray:
    """Σ_k mean_k² / (var_k / n) over the last two axes (rows, cells); cells with no spread and a zero mean add 0."""
    n = E.shape[-2]
    m = E.mean(axis=-2)
    v = E.var(axis=-2, ddof=1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.where(v > 0, m * m / np.where(v > 0, v, 1.0), np.where(np.abs(m) > 1e-12, np.inf, 0.0))
    return terms.sum(axis=-1)


def profile_test(M, L: int, B: int = 1999, seed: int = 0, chunk: int = 128) -> dict:
    """Heterogeneity of cell means within periods (module docstring). Rows with a missing cell are left out."""
    M = np.asarray(M, dtype=float)
    M = M[np.isfinite(M).all(axis=1)]
    n, K = M.shape
    if n < 4:
        return {"periods": int(n), "p": np.nan}
    E = M - M.mean(axis=1, keepdims=True)
    ebar = E.mean(axis=0)
    Q = float(_q_stat(E))
    E0 = E - ebar
    rng = np.random.default_rng(seed)
    draws = block_draws(n, L, B, rng)
    Qb, mb = [], []
    for i in range(0, B, chunk):
        Eb = E0[draws[i:i + chunk]]
        Qb.append(_q_stat(Eb))
        mb.append(Eb.mean(axis=1))
    Qb, mb = np.concatenate(Qb), np.concatenate(mb)
    se = mb.std(axis=0, ddof=1)
    G = max(n // max(int(L), 1), 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = np.where(se > 0, ebar / np.where(se > 0, se, 1.0), 0.0)
    p_cells = 2 * student_t.sf(np.abs(tt), G - 1)
    return {"periods": int(n), "block": int(L), "blocks": int(G), "effect": ebar, "se": se, "t": tt,
            "p_cells": p_cells, "stat": Q, "p": float((1 + np.sum(Qb >= Q)) / (B + 1)), "B": int(B),
            "effect_sd": float(np.std(ebar, ddof=1)), "effect_range": float(ebar.max() - ebar.min()),
            "argmax": int(np.argmax(ebar)), "argmin": int(np.argmin(ebar))}


def within_beta(Y, X) -> float:
    """Pooled OLS slope of row-centred Y on row-centred X over rows complete in both."""
    Y, X = np.asarray(Y, dtype=float), np.asarray(X, dtype=float)
    ok = np.isfinite(Y).all(axis=1) & np.isfinite(X).all(axis=1)
    ey = Y[ok] - Y[ok].mean(axis=1, keepdims=True)
    ex = X[ok] - X[ok].mean(axis=1, keepdims=True)
    den = float(np.sum(ex * ex))
    return float(np.sum(ex * ey) / den) if den > 0 else np.nan


def adjusted_profile(Y, X, L: int, B: int = 999, seed: int = 0, beta: float | None = None) -> dict:
    """Y − β·X (β = within_beta unless given; β = 1 is the ratio Y/X in logs). Returns β, the residual matrix, the
    share of the clock profile's variance across cells that X explains (1 − var(residual effects) / var(raw
    effects)) and its block-bootstrap interval (β re-estimated per draw when not given)."""
    Y, X = np.asarray(Y, dtype=float), np.asarray(X, dtype=float)
    ok = np.isfinite(Y).all(axis=1) & np.isfinite(X).all(axis=1)
    Y, X = Y[ok], X[ok]
    fixed = beta is not None

    def share(y, x):
        b = beta if fixed else within_beta(y, x)
        raw = (y - y.mean(axis=1, keepdims=True)).mean(axis=0)
        res = y - b * x
        res_e = (res - res.mean(axis=1, keepdims=True)).mean(axis=0)
        v = np.var(raw, ddof=1)
        return b, (1 - np.var(res_e, ddof=1) / v) if v > 0 else np.nan

    b, s = share(Y, X)
    rng = np.random.default_rng(seed)
    draws = block_draws(len(Y), L, B, rng)
    boot = np.array([share(Y[d], X[d])[1] for d in draws])
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return {"beta": float(b), "residual": Y - b * X, "share_explained": float(s), "share_ci": (float(lo), float(hi)),
            "periods": int(len(Y))}


def bh(pvalues) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values; NaN stays NaN and does not count."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = len(idx)
    if not m:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    adj = p[order] * m / np.arange(1, m + 1)
    out[order] = np.minimum(np.minimum.accumulate(adj[::-1])[::-1], 1.0)
    return out
