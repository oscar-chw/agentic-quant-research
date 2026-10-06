"""Sample weights and the sequential bootstrap (AFML ch.4).

Spans follow pmlab.labels: a label covers the half-open interval [t0, t1) of integer time units
(bars or seconds), so a label decided at unit e has t1 = e + 1. The book's indicator matrix marks
the units from t0 through the touch inclusive, which is the same set.

- indicator_matrix: 1 where a label covers a unit (Snippet 4.3, 4.5.2).
- uniqueness_from_matrix: average uniqueness of each label from an indicator matrix (Snippet 4.4, 4.5.2).
- seq_bootstrap: draws with probability proportional to each candidate's average uniqueness
  given the draws so far (4.5.1; Snippet 4.5 in 4.5.2), checked against the book's worked example (4.5.3).
  Time is compressed to the elementary segments between span endpoints, which gives the book's
  numbers exactly and costs O(draws x labels).
- standard_bootstrap: uniform draws with replacement, the baseline of the Monte Carlo (4.5.4).
- bootstrap_uniqueness: average uniqueness of a drawn sample (the statistic compared in 4.5.4).
- random_spans: random t1 as in the Monte Carlo experiment (4.5.4).
- return_attribution_weights: |sum of returns over a label's span, each divided by concurrency|,
  scaled to sum to the number of labels (4.6).
- time_decay: piecewise-linear decay on cumulative uniqueness with the newest weight 1 and the
  oldest set by c (4.7).
"""
import numpy as np


def _spans(t0, t1):
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    if t0.shape != t1.shape:
        raise ValueError("t0 and t1 differ in shape")
    if np.any(t1 <= t0):
        raise ValueError("every span needs t1 > t0")
    return t0, t1


def indicator_matrix(t0, t1, units=None) -> np.ndarray:
    """Dense (units, labels) 0/1 matrix; `units` defaults to range(min t0, max t1)."""
    t0, t1 = _spans(t0, t1)
    u = np.arange(t0.min(), t1.max()) if units is None else np.asarray(units)
    return ((u[:, None] >= t0[None, :]) & (u[:, None] < t1[None, :])).astype(np.int8)


def uniqueness_from_matrix(ind) -> np.ndarray:
    """Mean over each column's covered units of 1 / concurrency (units no label covers are ignored)."""
    ind = np.asarray(ind, dtype=float)
    c = ind.sum(axis=1, keepdims=True)
    u = np.divide(ind, c, out=np.zeros_like(ind), where=c > 0)
    n = ind.sum(axis=0)
    return np.divide(u.sum(axis=0), n, out=np.full(ind.shape[1], np.nan), where=n > 0)


def _segments(t0, t1):
    """Elementary segments between sorted endpoints: (lengths, first segment, end segment) per label."""
    edges = np.unique(np.concatenate([t0, t1]))
    lo = np.searchsorted(edges, t0)
    hi = np.searchsorted(edges, t1)
    return np.diff(edges).astype(float), lo, hi


def next_draw_probs(t0, t1, drawn) -> np.ndarray:
    """Probability of each label on the next sequential-bootstrap draw given the draws so far (4.5.1):
    its average uniqueness in the matrix of the drawn columns plus itself, normalised."""
    t0, t1 = _spans(t0, t1)
    length, lo, hi = _segments(t0, t1)
    diff = np.zeros(len(length) + 1)
    drawn = np.asarray(drawn, dtype=np.int64)
    np.add.at(diff, lo[drawn], 1)
    np.add.at(diff, hi[drawn], -1)
    count = np.cumsum(diff)[:-1]
    pref = np.concatenate([[0.0], np.cumsum(length / (count + 1.0))])
    avg_u = (pref[hi] - pref[lo]) / (t1 - t0)
    return avg_u / avg_u.sum()


def seq_bootstrap(t0, t1, size: int | None = None, rng=None, return_probs: bool = False):
    """Indices drawn by the sequential bootstrap (4.5.1). size defaults to the number of labels."""
    t0, t1 = _spans(t0, t1)
    rng = np.random.default_rng(rng)
    n = len(t0)
    size = n if size is None else int(size)
    length, lo, hi = _segments(t0, t1)
    span_len = (t1 - t0).astype(float)
    count = np.zeros(len(length))
    diff = np.zeros(len(length) + 1)
    drawn, probs = [], []
    for _ in range(size):
        pref = np.concatenate([[0.0], np.cumsum(length / (count + 1.0))])
        avg_u = (pref[hi] - pref[lo]) / span_len
        p = avg_u / avg_u.sum()
        i = int(rng.choice(n, p=p))
        drawn.append(i)
        if return_probs:
            probs.append(p)
        diff[:] = 0
        diff[lo[i]] += 1
        diff[hi[i]] -= 1
        count += np.cumsum(diff)[:-1]
    out = np.asarray(drawn, dtype=np.int64)
    return (out, probs) if return_probs else out


def standard_bootstrap(n: int, size: int | None = None, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    return rng.integers(0, n, n if size is None else int(size))


def bootstrap_uniqueness(t0, t1, drawn) -> float:
    """Mean average uniqueness of the labels in a drawn sample, repeats counted as separate labels."""
    t0, t1 = _spans(t0, t1)
    drawn = np.asarray(drawn, dtype=np.int64)
    a, b = t0[drawn], t1[drawn]
    length, lo, hi = _segments(a, b)
    diff = np.zeros(len(length) + 1)
    np.add.at(diff, lo, 1)
    np.add.at(diff, hi, -1)
    conc = np.cumsum(diff)[:-1]
    inv = np.divide(length, conc, out=np.zeros(len(length)), where=conc > 0)
    pref = np.concatenate([[0.0], np.cumsum(inv)])
    return float(np.mean((pref[hi] - pref[lo]) / (b - a)))


def random_spans(n_obs: int, n_units: int, max_h: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """t0 uniform on [0, n_units), last covered unit t0 + h with h uniform on [1, max_h) as in the
    4.5.4 experiment, written as half-open spans [t0, t0 + h + 1). The book keys labels by t0, so
    equal t0 overwrite each other there; here they stay separate labels."""
    rng = np.random.default_rng(rng)
    t0 = rng.integers(0, n_units, n_obs)
    return t0, t0 + rng.integers(1, max_h, n_obs) + 1


def concurrency(t0, t1, units) -> np.ndarray:
    """Number of labels covering each unit in `units` (sorted integers)."""
    t0, t1 = _spans(t0, t1)
    units = np.asarray(units, dtype=np.int64)
    return np.searchsorted(np.sort(t0), units, side="right") - np.searchsorted(np.sort(t1), units, side="right")


def return_attribution_weights(t0, t1, units, log_returns) -> np.ndarray:
    """w_i = |sum_{u in [t0_i, t1_i)} r_u / c_u|, scaled so the weights sum to the number of labels (4.6).

    `log_returns[j]` is the log return realised at `units[j]` (sorted); units with no label or a
    non-finite return contribute nothing."""
    t0, t1 = _spans(t0, t1)
    units = np.asarray(units, dtype=np.int64)
    r = np.asarray(log_returns, dtype=float)
    c = concurrency(t0, t1, units)
    contrib = np.where((c > 0) & np.isfinite(r), r / np.maximum(c, 1), 0.0)
    pref = np.concatenate([[0.0], np.cumsum(contrib)])
    a = np.searchsorted(units, t0, side="left")
    b = np.searchsorted(units, t1, side="left")
    w = np.abs(pref[b] - pref[a])
    total = w.sum()
    return w * len(w) / total if total > 0 else np.full(len(w), np.nan)


def time_decay(avg_uniqueness, order=None, c: float = 1.0) -> np.ndarray:
    """Decay weights on cumulative uniqueness (4.7). Labels are aged by `order` (default: as given,
    oldest first). c = 1: no decay; 0 <= c < 1: linear from c-ish (oldest) to 1 (newest); -1 < c < 0:
    the oldest |c| share of cumulative uniqueness gets weight 0."""
    u = np.asarray(avg_uniqueness, dtype=float)
    idx = np.arange(len(u)) if order is None else np.argsort(order, kind="stable")
    cum = np.cumsum(u[idx])
    last = cum[-1]
    slope = (1.0 - c) / last if c >= 0 else 1.0 / ((c + 1.0) * last)
    const = 1.0 - slope * last
    w_sorted = np.maximum(const + slope * cum, 0.0)
    out = np.empty(len(u))
    out[idx] = w_sorted
    return out
