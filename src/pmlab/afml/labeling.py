"""The book's labelling in its general form (AFML ch.3): volatility targets, the triple barrier, labels for side and
size, meta-labels and the removal of rare labels.

Inputs are arrays: `times` sorted ascending (seconds, bar indices, any numbers), `price` aligned with them, events
given by their start times. Outputs are DataFrames with one row per event, in event order.

- ewm_volatility: the EWMA standard deviation (span `span`) of returns over a lag, each measured from the last
  observation strictly before t - lag, as Snippet 3.1 does for one day. On a grid whose step divides the lag the
  return therefore spans the lag plus one step (the snippet's searchsorted takes the observation before the one at
  exactly t - lag).
- barrier_touches: for each event, the first time its return (price / price at t0 - 1, times the side) is above
  pt x target or below -sl x target, up to its vertical barrier t1 (the last time when t1 is NaN); a multiplier of 0
  disables that barrier (3.4, Snippet 3.2). Comparisons are strict, as in the snippet. The paths are walked in one
  vectorised pass over every event at once; `_barrier_touches_loop` is the snippet's own row-by-row version, kept as
  the reference the fast path is asserted to match bit for bit.
- vertical_barrier: the first time at or after t0 + horizon, NaN when the data end first (Snippet 3.4).
- get_events: targets at the events, events whose target is not above min_ret dropped, the earliest of the three
  barriers as t1 (Snippet 3.3). Without a side the barriers are symmetric at pt_sl[0]; with a side (meta-labelling)
  they are pt_sl[0] and pt_sl[1] (Snippet 3.6). t1 is NaN when no barrier is set and none is touched.
- get_bins: the return from the price at t0 to the price at t1, each taken at the first observation at or after that
  time; bin = sign(return). With a side the return is multiplied by it and bin is 1 when it is positive, else 0
  (Snippets 3.5 and 3.7). Events without t1 are dropped.
- drop_labels: keep dropping the rarest label while its share is at or below min_pct and at least three labels remain
  (Snippet 3.8); returns the mask of rows kept.

How this project uses chapter 3 (docs/afml_fidelity.md): a binary token held to settlement has no exit before expiry,
so the event dataset (pmlab.afml.events) keeps only the vertical barrier at the window end and labels the side's
settlement P&L after fees. pmlab.labels.triple_barrier is the symmetric, additive (logit units) barrier used for
direction studies. These functions are the book's general form, for research on exits and on other horizons.
"""
import numpy as np
import pandas as pd


def _sorted(times) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if len(t) > 1 and np.any(np.diff(t) < 0):
        raise ValueError("times must be sorted ascending")
    return t


def ewm_volatility(times, price, lag: float, span: float = 100) -> np.ndarray:
    """EWMA sd of lagged returns at each time; NaN where no observation precedes t - lag."""
    t, p = _sorted(times), np.asarray(price, dtype=float)
    j = np.searchsorted(t, t - lag, side="left")
    ok = j > 0
    out = np.full(len(t), np.nan)
    if ok.any():
        ret = p[ok] / p[j[ok] - 1] - 1.0
        out[ok] = pd.Series(ret).ewm(span=span).std().to_numpy()
    return out


def vertical_barrier(times, t_events, horizon: float) -> np.ndarray:
    t = _sorted(times)
    j = np.searchsorted(t, np.asarray(t_events, dtype=float) + horizon, side="left")
    return np.where(j < len(t), t[np.minimum(j, len(t) - 1)], np.nan)


def _pair(pt_sl) -> tuple[float, float]:
    v = np.atleast_1d(np.asarray(pt_sl, dtype=float))
    return (float(v[0]), float(v[0])) if len(v) == 1 else (float(v[0]), float(v[1]))


def _barrier_touches_loop(times, price, t0, t1, target, pt: float, sl: float, side=None) -> pd.DataFrame:
    """Snippet 3.2 as the book writes it: one path at a time. The reference `barrier_touches` is checked against
    (tests/test_afml_labeling.py, scripts/perf.py --check --vectorised), and the path a caller can fall back to."""
    t, p = _sorted(times), np.asarray(price, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    t1 = np.broadcast_to(np.asarray(np.nan if t1 is None else t1, dtype=float), t0.shape)
    target = np.broadcast_to(np.asarray(target, dtype=float), t0.shape)
    side = np.ones(len(t0)) if side is None else np.broadcast_to(np.asarray(side, dtype=float), t0.shape)
    out_sl, out_pt = np.full(len(t0), np.nan), np.full(len(t0), np.nan)
    for i in range(len(t0)):
        a = np.searchsorted(t, t0[i], side="left")
        end = t[-1] if np.isnan(t1[i]) else t1[i]
        b = np.searchsorted(t, end, side="right")
        if a >= len(t) or b <= a:
            continue
        path = (p[a:b] / p[a] - 1.0) * side[i]
        if sl > 0:
            hit = np.flatnonzero(path < -sl * target[i])
            out_sl[i] = t[a + hit[0]] if len(hit) else np.nan
        if pt > 0:
            hit = np.flatnonzero(path > pt * target[i])
            out_pt[i] = t[a + hit[0]] if len(hit) else np.nan
    return pd.DataFrame({"t0": t0, "t1": np.array(t1, dtype=float), "sl": out_sl, "pt": out_pt})


def barrier_touches(times, price, t0, t1, target, pt: float, sl: float, side=None) -> pd.DataFrame:
    """Columns t0, t1 (vertical barrier as given), sl and pt (first touch time or NaN).

    Same result as `_barrier_touches_loop`, **bit for bit**: the path is still the contiguous slice
    `(p[a:b] / p[a] - 1) * side`, the same three operations on the same two operands, and the comparisons are still
    strict. What left the loop is the bookkeeping around it — both barrier searches and both thresholds are computed
    once for every event at a time, so a walk over N events makes 2 calls into numpy instead of 2N. Measured on a day
    of BTC seconds (scripts/perf.py label_spans): 1.65x at a 10-minute horizon, 1.07x at 30 minutes.

    The path itself stays a loop on purpose. Laying every event's path end to end into one array was tried and is
    *slower* — 0.85x at 10 minutes and 0.47x at 30 — because the gather and the three `repeat` passes it needs cost
    more than the contiguous slices they replace. It is recorded in docs/performance.md as a vectorisation that
    measurement rejected.
    """
    t, p = _sorted(times), np.asarray(price, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    t1 = np.broadcast_to(np.asarray(np.nan if t1 is None else t1, dtype=float), t0.shape)
    target = np.broadcast_to(np.asarray(target, dtype=float), t0.shape)
    side = np.ones(len(t0)) if side is None else np.broadcast_to(np.asarray(side, dtype=float), t0.shape)
    out_sl, out_pt = np.full(len(t0), np.nan), np.full(len(t0), np.nan)
    if not len(t0) or not len(t):
        return pd.DataFrame({"t0": t0, "t1": np.array(t1, dtype=float), "sl": out_sl, "pt": out_pt})
    a = np.searchsorted(t, t0, side="left")
    b = np.searchsorted(t, np.where(np.isnan(t1), t[-1], t1), side="right")
    below, above = -sl * target, pt * target
    for i in np.flatnonzero((a < len(t)) & (b > a)):
        start = a[i]
        path = (p[start:b[i]] / p[start] - 1.0) * side[i]
        if sl > 0:
            hit = np.flatnonzero(path < below[i])
            if len(hit):
                out_sl[i] = t[start + hit[0]]
        if pt > 0:
            hit = np.flatnonzero(path > above[i])
            if len(hit):
                out_pt[i] = t[start + hit[0]]
    return pd.DataFrame({"t0": t0, "t1": np.array(t1, dtype=float), "sl": out_sl, "pt": out_pt})


def get_events(times, price, t_events, target, pt_sl, min_ret: float = 0.0, t1=None, side=None) -> pd.DataFrame:
    """Columns t0, t1 (earliest barrier), trgt and, when a side is given, side. `target` is aligned with `times` and
    read at each event time, which must be one of them; `t1` and `side` are aligned with `t_events`."""
    t = _sorted(times)
    te = np.asarray(t_events, dtype=float)
    pos = np.searchsorted(t, te, side="left")
    if np.any(pos >= len(t)) or np.any(t[np.minimum(pos, len(t) - 1)] != te):
        raise ValueError("every event time must be one of `times`")
    trgt = np.asarray(target, dtype=float)[pos]
    keep = trgt > min_ret                                         # NaN targets fail this too
    v1 = np.full(len(te), np.nan) if t1 is None else np.asarray(t1, dtype=float)
    if side is None:
        pt, _ = _pair(pt_sl)
        sl, s = pt, None
    else:
        pt, sl = _pair(pt_sl)
        s = np.asarray(side, dtype=float)[keep]
    touch = barrier_touches(t, price, te[keep], v1[keep], trgt[keep], pt, sl, s)
    first = touch[["t1", "sl", "pt"]].min(axis=1, skipna=True).to_numpy()
    out = pd.DataFrame({"t0": te[keep], "t1": first, "trgt": trgt[keep]})
    if side is not None:
        out["side"] = s
    return out


def get_bins(times, price, t0, t1, side=None) -> pd.DataFrame:
    """Columns t0, t1, ret, bin for the events whose t1 is known (module docstring)."""
    t, p = _sorted(times), np.asarray(price, dtype=float)
    t0, t1 = np.asarray(t0, dtype=float), np.asarray(t1, dtype=float)
    ok = np.isfinite(t1)
    t0, t1 = t0[ok], t1[ok]

    def at_or_after(x):
        j = np.searchsorted(t, x, side="left")
        return np.where(j < len(t), p[np.minimum(j, len(t) - 1)], np.nan)

    ret = at_or_after(t1) / at_or_after(t0) - 1.0
    if side is None:
        bin_ = np.sign(ret)
    else:
        ret = ret * np.asarray(side, dtype=float)[ok]
        bin_ = np.where(ret > 0, 1.0, np.where(np.isfinite(ret), 0.0, np.nan))
    return pd.DataFrame({"t0": t0, "t1": t1, "ret": ret, "bin": bin_})


def drop_labels(bins, min_pct: float = 0.05) -> np.ndarray:
    """Mask of rows kept after repeatedly removing the rarest label (ties: the smallest label value)."""
    b = np.asarray(bins)
    keep = np.ones(len(b), dtype=bool)
    while keep.any():
        values, counts = np.unique(b[keep], return_counts=True)
        share = counts / counts.sum()
        if share.min() > min_pct or len(values) < 3:
            break
        keep &= b != values[np.argmin(share)]
    return keep
