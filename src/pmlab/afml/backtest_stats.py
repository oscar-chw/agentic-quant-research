"""General backtest statistics from a series of target positions and bet returns (AFML ch.14).

- bet_timing: the times at which a bet ends, from target positions (14.3, Snippet 14.1): a position goes flat, or flips
  sign without passing through flat, and the last time closes whatever is open. The snippet's shift leaves no previous
  position at the first time, and pandas counts that as non-zero, so a series starting flat counts a bet at its first
  time; here a bet needs a non-zero position before it.
- holding_period: the average holding period by average-entry-time pairing (14.3, Snippet 14.2): additions move the
  average entry time, reductions close the reduced size at the current time, a flip closes the whole previous position
  and restarts the entry time; the result is the size-weighted mean holding time, in the units of `times`.
- hhi: the Herfindahl-Hirschman concentration of a set of bet returns, rescaled to [0, 1] by (sum w^2 - 1/n) / (1 - 1/n)
  with w = r / sum r; NaN for two returns or fewer (14.5, Eqs. 50-51, Snippet 14.3).
- concentration: h+ over the returns >= 0, h- over the returns < 0 and h[t], the concentration of the number of bets per
  period (the book counts per month; pass the period of each bet, and every period to count empty ones as 0).
- drawdown_tuw: the drawdown after each high-water mark that is followed by a loss, and the time under water (14.6,
  Snippet 14.4). As in the snippet, the time under water of a drawdown runs from its high-water mark to the next
  high-water mark that is followed by a drawdown (not to the recovery), and the last drawdown has none. Drawdowns are
  1 - min / hwm of a wealth series, or hwm - min in dollars. Times are in the units given (the book converts to years).
- capacity: the highest size (and so the highest AUM) whose net Sharpe still reaches a target, under a linear market
  impact that costs the square of the dollars sent (14.3).

In this project a bet is a token bought in a 5-minute window and held to settlement, so every bet closes at its
window's end and the holding period is set by the entry second; these functions matter for strategies that net
positions across windows and for the lifecycle's drawdown rules.
"""
import numpy as np
from scipy.optimize import brentq


def bet_timing(times, positions) -> np.ndarray:
    t, pos = np.asarray(times), np.asarray(positions, dtype=float)
    if len(t) == 0:
        return t
    prev = np.concatenate([[0.0], pos[:-1]])
    flat = (pos == 0) & (prev != 0)
    flip = pos * prev < 0
    ends = flat | flip
    ends[-1] = True
    return t[ends]


def holding_period(times, positions) -> float:
    t, pos = np.asarray(times, dtype=float), np.asarray(positions, dtype=float)
    elapsed = t - t[0] if len(t) else t
    entry = 0.0
    dts, ws = [], []
    for i in range(1, len(pos)):
        change = pos[i] - pos[i - 1]
        if change * pos[i - 1] >= 0:                                  # added or unchanged
            if pos[i] != 0:
                entry = (entry * pos[i - 1] + elapsed[i] * change) / pos[i]
        elif pos[i] * pos[i - 1] < 0:                                 # flipped
            dts.append(elapsed[i] - entry)
            ws.append(abs(pos[i - 1]))
            entry = elapsed[i]
        else:                                                         # reduced
            dts.append(elapsed[i] - entry)
            ws.append(abs(change))
    ws = np.asarray(ws)
    return float(np.dot(dts, ws) / ws.sum()) if ws.sum() > 0 else np.nan


def hhi(bet_returns) -> float:
    r = np.asarray(bet_returns, dtype=float)
    if len(r) <= 2:
        return np.nan
    w = r / r.sum()
    return float((np.sum(w ** 2) - 1.0 / len(r)) / (1.0 - 1.0 / len(r)))


def concentration(returns, periods=None, all_periods=None) -> dict:
    r = np.asarray(returns, dtype=float)
    out = {"h_pos": hhi(r[r >= 0]), "h_neg": hhi(r[r < 0]), "h_time": np.nan}
    if periods is not None:
        keys, counts = np.unique(np.asarray(periods), return_counts=True)
        if all_periods is not None:
            full = dict.fromkeys(list(all_periods), 0) | dict(zip(keys.tolist(), counts.tolist()))
            counts = np.array(list(full.values()), dtype=float)
        out["h_time"] = hhi(counts)
    return out


def drawdown_tuw(times, pnl, dollars: bool = False) -> dict:
    """{"hwm_time", "hwm", "min", "drawdown", "tuw_time", "time_under_water"} over the drawdown episodes."""
    t, x = np.asarray(times, dtype=float), np.asarray(pnl, dtype=float)
    hwm = np.maximum.accumulate(x)
    starts = np.flatnonzero(np.concatenate([[True], hwm[1:] != hwm[:-1]]))
    ends = np.concatenate([starts[1:], [len(x)]])
    mins = np.array([x[a:b].min() for a, b in zip(starts, ends)])
    level = hwm[starts]
    keep = level > mins
    hwm_time, level, mins = t[starts][keep], level[keep], mins[keep]
    dd = level - mins if dollars else 1.0 - mins / level
    return {"hwm_time": hwm_time, "hwm": level, "min": mins, "drawdown": dd,
            "tuw_time": hwm_time[:-1], "time_under_water": np.diff(hwm_time)}


def capacity(pnl, stake, impact: float, target_sharpe: float, periods_per_year: float,
             avg_aum: float = 1.0, max_multiple: float = 1e12) -> dict:
    """14.3's capacity: the largest size at which the strategy still reaches a target risk-adjusted performance.

    The book defines capacity as the highest AUM that delivers a target performance, and explains the decay by the
    transaction costs a larger size pays. `pnl` and `stake` are one period's profit and the dollars sent at the size
    actually run; running the same strategy `m` times larger sends `m * stake` and earns `m * pnl` gross, and pays an
    impact cost `impact * (m * stake) ** 2` -- one dollar sent moves the average fill price by `impact`, so the cost
    of a fill grows with the square of its size. The annualised Sharpe of the net series therefore falls with `m`, and
    capacity is `avg_aum` times the largest `m` that still reaches `target_sharpe`: infinite when `impact` is 0 (a
    book of unlimited depth has no capacity limit) and 0 when even an infinitesimal size misses the target.
    """
    x, s = np.asarray(pnl, dtype=float), np.asarray(stake, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    gross = float(x.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan
    out = {"sharpe_gross": gross, "multiple": np.nan, "capacity": np.nan}
    if not np.isfinite(gross) or gross <= target_sharpe:
        return out | {"multiple": 0.0, "capacity": 0.0}
    if impact <= 0 or not np.any(s > 0):
        return out | {"multiple": np.inf, "capacity": np.inf}

    def excess(m: float) -> float:
        net = m * x - impact * (m * s) ** 2
        d = net.std(ddof=1)
        return (float(net.mean() / d * np.sqrt(periods_per_year)) - target_sharpe) if d > 0 else -target_sharpe

    hi = 1.0
    while excess(hi) > 0:
        hi *= 2.0
        if hi > max_multiple:
            return out | {"multiple": np.inf, "capacity": np.inf}
    m = float(brentq(excess, hi / 2 if hi > 1 else 0.0, hi, xtol=1e-12, rtol=1e-12))
    return out | {"multiple": m, "capacity": m * avg_aum}
