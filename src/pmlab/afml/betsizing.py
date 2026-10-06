"""Bet sizing (AFML ch.10).

From predicted probabilities (10.3):
- size_from_prob: test statistic z = (p - 1/K) / sqrt(p (1 - p)) for the predicted class's
  probability p among K classes, size m = side (2 Phi(z) - 1).
- size_vs_price: the adaptation for a binary option bought at all-in price a. The book's null is
  1/K (no information); for a token that costs a, the no-edge probability is a, so
  z = (p - a) / sqrt(p (1 - p)) and m = 2 Phi(z) - 1, floored at 0 (a bet that loses in
  expectation is not placed). This departs from the book, which sizes on p alone.
- concurrency, budget_size: 10.2's second, strategy-independent approach. The number of long and of
  short bets active at each time gives m_t = c_l / max(c_l) - c_s / max(c_s), so the position reaches
  its maximum only when as many bets are concurrent as ever were.
- avg_active_sizes: the average size of every bet still open at each time a bet opens or closes
  (10.4).
- hold_to_expiry_increments: 10.4 for positions that cannot be sold, only held to expiry (a binary
  token here): in each window the target is the mean signed size of every bet opened so far (all
  still active), and the position moves toward it only by adding in the direction already held.
- stake_performance: dollar results of stakes that are a size times a fixed maximum stake: P&L per
  window over every eligible window (untraded windows count 0), its Sharpe, and the log growth of a
  bankroll that stakes a fraction of itself as the maximum stake, compounding window by window.
- discrete_size: sizes rounded to multiples of a step and clipped to [-1, 1] (10.5).

Dynamic position size and limit price (10.6), with the sigmoid m(w, x) = x / sqrt(w + x^2) of the
divergence x = forecast - market price:
- bet_size, target_position, inverse_price, limit_price, calibrate_w.

Kelly comparison: pmlab.strategies.kelly_fraction is (p - a) / (1 - a) for the same binary.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def size_from_prob(prob, n_classes: int = 2, side=1) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (p - 1.0 / n_classes) / np.sqrt(p * (1.0 - p))
    return np.asarray(side) * (2.0 * norm.cdf(z) - 1.0)


def size_vs_price(prob, price) -> np.ndarray:
    p, a = np.asarray(prob, dtype=float), np.asarray(price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (p - a) / np.sqrt(p * (1.0 - p))
    return np.clip(2.0 * norm.cdf(z) - 1.0, 0.0, 1.0)


def kelly(prob, price) -> np.ndarray:
    """Full Kelly stake / bankroll for a $1 binary at all-in price a: (p - a) / (1 - a), floored at 0."""
    p, a = np.asarray(prob, dtype=float), np.asarray(price, dtype=float)
    return np.where(a < 1, np.maximum((p - a) / (1.0 - a), 0.0), 0.0)


def avg_active_sizes(t0, t1, size) -> tuple[np.ndarray, np.ndarray]:
    """(times, average size of the bets active at each time). A bet is active on [t0, t1); t1 = NaN
    or inf keeps it open. Times are every t0 and every finite t1; the average is 0 when none is open."""
    t0 = np.asarray(t0, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    size = np.asarray(size, dtype=float)
    t1 = np.where(np.isfinite(t1), t1, np.inf)
    times = np.unique(np.concatenate([t0, t1[np.isfinite(t1)]]))
    out = np.zeros(len(times))
    for k, t in enumerate(times):
        act = (t0 <= t) & (t < t1)
        out[k] = size[act].mean() if act.any() else 0.0
    return times, out


def concurrency(t0, t1, side) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, long count, short count) of the bets active on [t0, t1) at each time a bet opens or closes (10.2).
    A non-finite end keeps a bet open. `side` is +1 for a long bet and -1 for a short one."""
    t0 = np.asarray(t0, dtype=float)
    t1 = np.where(np.isfinite(np.asarray(t1, dtype=float)), np.asarray(t1, dtype=float), np.inf)
    side = np.sign(np.asarray(side, dtype=float))
    times = np.unique(np.concatenate([t0, t1[np.isfinite(t1)]]))
    active = (t0[None, :] <= times[:, None]) & (times[:, None] < t1[None, :])
    return times, (active & (side > 0)).sum(axis=1), (active & (side < 0)).sum(axis=1)


def budget_size(c_long, c_short, max_long: float | None = None, max_short: float | None = None) -> np.ndarray:
    """10.2's budgeting rule: m_t = c_l / max(c_l) - c_s / max(c_s), in [-1, 1]. The maxima default to the
    maxima of the series themselves (the book allows another quantile); a side that never fires contributes 0."""
    c_long, c_short = np.asarray(c_long, dtype=float), np.asarray(c_short, dtype=float)
    ml = float(max_long if max_long is not None else (c_long.max() if len(c_long) else 0.0))
    ms = float(max_short if max_short is not None else (c_short.max() if len(c_short) else 0.0))
    long = np.divide(c_long, ml, out=np.zeros_like(c_long), where=ml > 0)
    short = np.divide(c_short, ms, out=np.zeros_like(c_short), where=ms > 0)
    return long - short


def hold_to_expiry_increments(window, size, side) -> tuple[np.ndarray, np.ndarray]:
    """(stake added at each bet as a share of the maximum position, side the position is on after the bet).

    Bets are in time order; `window` groups bets held to the same expiry. The target after bet j is the mean
    of side x size over the window's bets up to j; if it points the way the position is held (or the position
    is flat) and is larger, the position grows to it, otherwise nothing is traded."""
    window, size, side = np.asarray(window), np.asarray(size, dtype=float), np.sign(np.asarray(side, dtype=float))
    inc, held_side = np.zeros(len(size)), np.zeros(len(size), dtype=int)
    pos, total, count, current = 0.0, 0.0, 0, None
    for j in range(len(size)):
        if window[j] != current:
            pos, total, count, current = 0.0, 0.0, 0, window[j]
        total += side[j] * size[j]
        count += 1
        target = total / count
        if (pos == 0 or np.sign(target) == np.sign(pos)) and abs(target) > abs(pos):
            inc[j] = abs(target) - abs(pos)
            pos = target
        held_side[j] = int(np.sign(pos))
    return inc, held_side


def stake_performance(window, stake, return_per_dollar, all_windows, fractions=(0.1,)) -> dict:
    """window: window of each bet; stake: dollars at risk per bet in units of the maximum stake (0 when not
    filled); return_per_dollar: the bet's P&L per dollar staked. Log growth for bankroll fraction f compounds
    B_w = B_(w-1) (1 + f x window P&L) over all_windows in order; a window that loses the whole bankroll is ruin."""
    df = pd.DataFrame({"window": np.asarray(window), "stake": np.asarray(stake, dtype=float),
                       "pnl": np.asarray(stake, dtype=float) * np.nan_to_num(np.asarray(return_per_dollar, dtype=float))})
    df = df[df["stake"] > 0]
    per = df.groupby("window")["pnl"].sum().reindex(list(all_windows)).fillna(0.0).to_numpy()
    sd = per.std(ddof=1) if len(per) > 1 else np.nan
    out = {"bets": int(len(df)), "traded_windows": int(df["window"].nunique()), "windows": int(len(per)),
           "staked": float(df["stake"].sum()), "pnl": float(per.sum()),
           "pnl_per_dollar": float(per.sum() / df["stake"].sum()) if len(df) else np.nan,
           "mean_window_pnl": float(per.mean()) if len(per) else np.nan,
           "window_sharpe": float(per.mean() / sd) if sd and sd > 0 else np.nan, "window_pnl": per,
           "log_growth": {}, "ruined": {}}
    for f in fractions:
        growth = 1.0 + f * per
        key = f"{f:g}"
        out["ruined"][key] = bool(np.any(growth <= 0))
        out["log_growth"][key] = float(np.sum(np.log(growth))) if not out["ruined"][key] else -np.inf
    return out


def discrete_size(m, step: float) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return np.clip(np.round(m / step) * step, -1.0, 1.0)


def bet_size(w: float, x):
    return np.asarray(x, dtype=float) / np.sqrt(w + np.asarray(x, dtype=float) ** 2)


def target_position(w: float, forecast: float, market: float, max_pos: int) -> int:
    return int(bet_size(w, forecast - market) * max_pos)


def inverse_price(forecast: float, w: float, m: float) -> float:
    """The market price at which the sigmoid gives size m."""
    return forecast - m * np.sqrt(w / (1.0 - m * m))


def limit_price(t_pos: int, pos: int, forecast: float, w: float, max_pos: int) -> float:
    """Breakeven limit price for moving from pos to t_pos: the mean inverse price over each unit
    added (10.6). The book's loop runs over abs() bounds, which is right for buys from a flat or
    long position; here each unit j = pos + sign, ..., t_pos is used, so sells work the same way."""
    if t_pos == pos:
        return np.nan
    sgn = 1 if t_pos > pos else -1
    js = np.arange(pos + sgn, t_pos + sgn, sgn)
    return float(np.mean([inverse_price(forecast, w, j / max_pos) for j in js]))


def calibrate_w(x: float, m: float) -> float:
    """w such that a divergence x maps to size m: w = x^2 (m^-2 - 1)."""
    return x * x * (1.0 / (m * m) - 1.0)
