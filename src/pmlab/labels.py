"""Event sampling, labels and label weights for information-driven bars (AFML ch.2.5, 3, 4, 7).

Pure numpy functions. Times are integer seconds. A label span is [t0, t1): it opens at the
event and closes, exclusive, when the label is decided.

- cusum_filter: symmetric CUSUM event filter on a return series (AFML 2.5.2(1), snippet 2.4).
- run_length: how many consecutive same-sign values end at each element.
- any_per_second: flag every element of a second in which any element was flagged (CUSUM on bars
  of which only the last per second is kept).
- triple_barrier: first touch of +width / -width on a path within a horizon (AFML 3.4).
- fixed_horizon: sign of the path's move over a fixed horizon (AFML 3.2).
- path_return: the move a barrier or horizon label is decided on, from the reference to the
  deciding second (the return at first touch, or at the vertical barrier).
- meta_label: 1 where side x return > 0, else 0, for every event with a return, vertical-barrier
  events included (AFML 3.6, snippet 3.7).
- average_uniqueness: mean of 1 / concurrency over each label's span (AFML 4.4).
- purged_kfold: contiguous folds in time, training purged of labels that overlap the test
  fold and embargoed after it (AFML 7.4).
- binary_pnl: P&L per share of a binary held to settlement, bought as a taker.

Path convention for the barrier labels: `path[u]` is the value at second u. An event whose
first actionable second is `entry` is measured from the reference `path[entry - 1]` over
`path[entry : entry + horizon]`, truncated at the end of the path (the window's expiry).
"""
import numpy as np

from pmlab.fees import FEE_RATE


def cusum_filter(returns, h) -> np.ndarray:
    """+1 where the upward CUSUM exceeds h, -1 where the downward one falls below -h, else 0.

    S+ = max(0, S+ + r), S- = min(0, S- + r); the side that triggers resets to 0 (the book
    checks the downside first and uses strict inequalities). A non-finite return leaves both
    sums unchanged; a non-finite or non-positive h can never trigger.
    """
    r = np.asarray(returns, dtype=float)
    h = np.broadcast_to(np.asarray(h, dtype=float), r.shape)
    out = np.zeros(len(r), dtype=np.int8)
    up = dn = 0.0
    for i in range(len(r)):
        if not np.isfinite(r[i]):
            continue
        up, dn = max(0.0, up + r[i]), min(0.0, dn + r[i])
        if not (np.isfinite(h[i]) and h[i] > 0):
            continue
        if dn < -h[i]:
            dn, out[i] = 0.0, -1
        elif up > h[i]:
            up, out[i] = 0.0, 1
    return out


def any_per_second(flags, seconds) -> np.ndarray:
    """True for every element whose second holds at least one True flag."""
    flags = np.asarray(flags, dtype=bool)
    if not len(flags):
        return flags
    _, inv = np.unique(np.asarray(seconds), return_inverse=True)
    hit = np.zeros(inv.max() + 1, dtype=bool)
    np.logical_or.at(hit, inv, flags)
    return hit[inv]


def run_length(signs) -> np.ndarray:
    """Length of the run of equal non-zero signs ending at each element (0 where the sign is 0)."""
    s = np.sign(np.asarray(signs, dtype=float))
    out = np.zeros(len(s), dtype=np.int64)
    for i in range(len(s)):
        if s[i] == 0 or not np.isfinite(s[i]):
            continue
        out[i] = out[i - 1] + 1 if i and s[i] == s[i - 1] else 1
    return out


def _window(path, entry, horizon):
    path = np.asarray(path, dtype=float)
    entry = np.asarray(entry, dtype=np.int64)
    if np.any(entry < 1):
        raise ValueError("entry must be >= 1: the reference is path[entry - 1]")
    idx = entry[:, None] + np.arange(int(horizon))[None, :]
    valid = idx < len(path)
    vals = np.where(valid, path[np.minimum(idx, len(path) - 1)], np.nan)
    ref = np.where(entry - 1 < len(path), path[np.minimum(entry - 1, len(path) - 1)], np.nan)
    return ref, vals, valid


def triple_barrier(path, entry, width, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(label, decided_at) per event.

    label = +1 if path - ref reaches +width first, -1 if it reaches -width first, 0 if neither
    happens by the vertical barrier, NaN if the reference or the whole horizon is missing.
    decided_at = the second of the touch, or the last second inside the horizon.
    """
    width = np.broadcast_to(np.asarray(width, dtype=float), np.shape(entry))
    ref, vals, valid = _window(path, entry, horizon)
    d = vals - ref[:, None]
    H = vals.shape[1]
    first = lambda hit: np.where(hit.any(axis=1), hit.argmax(axis=1), H)
    up, dn = first(d >= width[:, None]), first(d <= -width[:, None])
    label = np.where(up < dn, 1.0, np.where(dn < up, -1.0, 0.0))
    n_valid = valid.sum(axis=1)
    last = np.asarray(entry) + np.maximum(n_valid - 1, 0)
    decided = np.where(np.minimum(up, dn) < H, np.asarray(entry) + np.minimum(up, dn), last)
    bad = ~np.isfinite(ref) | ~(width > 0) | (n_valid == 0)
    return np.where(bad, np.nan, label), decided


def fixed_horizon(path, entry, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(sign of path[end] - path[entry - 1], end) with end = min(entry - 1 + horizon, len - 1)."""
    path = np.asarray(path, dtype=float)
    entry = np.asarray(entry, dtype=np.int64)
    if np.any(entry < 1):
        raise ValueError("entry must be >= 1: the reference is path[entry - 1]")
    end = np.minimum(entry - 1 + int(horizon), len(path) - 1)
    ref = np.where(entry - 1 < len(path), path[np.minimum(entry - 1, len(path) - 1)], np.nan)
    move = path[end] - ref
    return np.where(np.isfinite(move) & (end >= entry), np.sign(move), np.nan), end


def path_return(path, entry, end) -> np.ndarray:
    """path[end] - path[entry - 1] per event; NaN when either value is missing or end < entry."""
    path = np.asarray(path, dtype=float)
    entry = np.asarray(entry, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    ok = (entry >= 1) & (end >= entry) & (end < len(path)) & (entry - 1 < len(path))
    ref = path[np.clip(entry - 1, 0, len(path) - 1)]
    last = path[np.clip(end, 0, len(path) - 1)]
    return np.where(ok, last - ref, np.nan)


def meta_label(ret, side) -> tuple[np.ndarray, np.ndarray]:
    """(y, rows): y = 1.0 where side * ret > 0 else 0.0; rows = events with a finite return and a side.

    As in the book's meta-labelling bins, an event whose vertical barrier came first is labelled by the
    sign of its return like any other; a zero return is a 0. Nothing with a return is dropped."""
    x = np.asarray(ret, dtype=float) * np.asarray(side, dtype=float)
    rows = np.isfinite(x) & (np.asarray(side, dtype=float) != 0)
    return np.where(rows & (x > 0), 1.0, 0.0), rows


def average_uniqueness(t0, t1) -> np.ndarray:
    """Mean over each span [t0, t1) of 1 / (number of spans covering that second)."""
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    if len(t0) == 0:
        return np.zeros(0)
    if np.any(t1 <= t0):
        raise ValueError("every span needs t1 > t0")
    lo, hi = t0.min(), t1.max()
    diff = np.zeros(hi - lo + 1, dtype=np.int64)
    np.add.at(diff, t0 - lo, 1)
    np.add.at(diff, t1 - lo, -1)
    conc = np.cumsum(diff)[:-1]
    inv = np.divide(1.0, conc, out=np.zeros(len(conc)), where=conc > 0)
    prefix = np.concatenate([[0.0], np.cumsum(inv)])
    return (prefix[t1 - lo] - prefix[t0 - lo]) / (t1 - t0)


def purged_kfold(t0, t1, n_splits: int = 5, embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) index arrays into the inputs. Test folds are contiguous blocks of events
    ordered by t0. Training drops every event whose span overlaps the test fold's envelope
    [min t0, max t1) and every event starting within `embargo` seconds after it."""
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    order = np.argsort(t0, kind="stable")
    folds = []
    for block in np.array_split(order, n_splits):
        if not len(block):
            continue
        lo, hi = t0[block].min(), t1[block].max()
        keep = ~((t0 < hi) & (t1 > lo)) & ~((t0 >= hi) & (t0 < hi + embargo))
        keep[block] = False
        folds.append((np.flatnonzero(keep), np.sort(block)))
    return folds


def binary_pnl(win, price, fee_rate: float = FEE_RATE):
    """Per share: payoff (1 if the bought side won, else 0) - price - taker fee rate·p(1-p)."""
    price = np.asarray(price, dtype=float)
    return np.asarray(win, dtype=float) - price - fee_rate * price * (1.0 - price)
