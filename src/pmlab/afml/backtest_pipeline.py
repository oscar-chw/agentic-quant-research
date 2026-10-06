"""Backtests by the book for binary bets held to expiry (AFML ch. 11-15; plan afml-pipeline node 13).

A Strategy is a primary rule's events, optionally filtered by a meta-label model's probability that the side wins
(bet when P >= threshold), and sized by a function from that probability to order dollars. Every bet buys the
event's side token and holds it to the window's settlement.

CPCV over whole Taiwan days (12.4)
- The Taiwan days (UTC+8) of the backtest's windows are cut, in order, into N contiguous groups of near-equal day
  counts, so no day and no window is split. Every choice of k test groups is a split: C(N, k) splits and
  phi[N, k] = k/N C(N, k) = C(N-1, k-1) paths; cpcv.paths gives the j-th split that tests a group to path j.
- A split trains on the other groups minus, for each test group, every row whose span [t0, t1) overlaps the group's
  envelope [first t0, last t1) (purging, 7.4.1) and every row starting within `embargo` seconds after the envelope
  ends (embargo, 7.4.2; audit R6 finding 1). The default EMBARGO = events.EMBARGO_S outlasts the day models: a UTC
  day's fair parameters are fitted on the outcomes of the events.FAIR_TRAIN_DAYS days before it, and the extra day
  covers the 8 h between a Taiwan-day group's end and the next UTC day (code review afml.md, events finding 1). The
  model is fitted on the training rows and predicts the test rows, which reach it and the sizing function without
  their outcome columns (OUTCOMES). No row of a window ending after the confirmation start is accepted.
- Path j takes, for every group g, the bets that split paths[j, g] made on g: it covers every group, hence every
  window, exactly once. Window P&L along a path is in dollars, 0 for windows without a filled bet.

A bet (a row with a positive order)
- ordered shares = dollars / cost (cost = price + fee per share); filled = ordered x fill_ratio (the events' column
  when present, else 1: the print proxy fills in full); pnl = filled x (payoff - cost).
- Implementation shortfall against the side's mid at the decision (14.6; audit R6 finding 10, after Perold 1988):
  paper = ordered x (payoff - mid), arrival cost = filled x (price - mid), fees = filled x fee, opportunity cost =
  (ordered - filled) x (payoff - mid); shortfall = arrival + fees + opportunity = paper - pnl. The mid is the events'
  `mid` column, else (ask_up + bid_up) / 2 for Up and 1 minus it for Down; bets without a mid are left out of the
  shortfall sums. Per turnover (filled shares x price): fees, slippage (arrival cost) and P&L; return on execution
  costs = pnl / (arrival cost + fees).

Statistics per path (ch. 14); statistics() adds their distribution over paths
- Bets (14.3): backtest_stats.bet_timing on the net Up-equivalent position (a Down share is short an Up share) after
  each purchase, flattened at every window's settlement: a bet ends when the position returns to zero, flips, or the
  window settles. Holding period: the share-weighted mean of t1 - t0, each purchase being held to settlement. Statistics
  that need a bet's outcome (hit ratio, concentration of returns, chapter 15) use the traded window, whose positions
  all settle on one result.
- Dollar characteristics (14.3, aum()): average AUM time-weights the dollar value of the open position (filled shares
  x price, a Down position counting positive) over the backtest span; maximum dollar position size is the largest
  value open at one instant; annualised turnover is dollars traded per year over average AUM. Leverage is 1 by
  construction (nothing is borrowed and a token cannot be sold short). Capacity, the highest AUM still reaching a
  target performance, needs a market-impact coefficient of the recorded book, so it is reported only when `impact` is
  given: backtest_stats.capacity charges impact x (dollars sent)^2 per window and returns the largest size, and hence
  the largest average AUM, whose net annualised Sharpe still reaches target_sharpe.
- Performance (14.4): pnl_long and pnl_short split the path's P&L between Up (long) and Down positions. hit_ratio is
  the share of traded windows with positive P&L; avg_hit and avg_miss are the mean RETURN (P&L / dollars invested,
  fees included) of the windows that gained and of those that lost, with a window of exactly zero P&L in neither
  (counted as flat_windows); avg_hit_dollars and avg_miss_dollars are the same two means in dollars, which with
  sized bets are dominated by the large stakes.
- Time-weighted rate of return (14.4.1), where it applies: with a bankroll the account (cash included) is the capital
  at the start of every window, r_t = pnl_t / K_t with K_t = bankroll + earlier P&L, linked geometrically and
  annualised by the years elapsed. Without a bankroll the only capital is what one window invests, which is lost in
  full whenever its tokens lose, so the linked return would be 0 after the first such window: NaN.
- Concentration (14.5.1, backtest_stats.concentration): h+ of traded-window P&L >= 0, h- of < 0, and h[t] of traded
  windows per Taiwan day over the backtest's days (idle days count 0). Deviation (docs/afml_sections/ch14.md F14.4):
  the book concentrates a bet's RETURNS, these concentrate a traded window's DOLLAR P&L, so once sizing varies one
  large stake reads as concentration of returns; with a fixed stake the two agree up to the window's price.
- Drawdown and time under water (14.5.2), on cumulative P&L from 0 before the first window: the episodes and their
  drawdowns are backtest_stats.drawdown_tuw's (HWM - trough in dollars, 1 - trough / HWM on bankroll equity). The
  time under water is the text's: from the HWM (when first reached) to the moment P&L exceeds it, or to the last
  window when it never does. (The snippet, which drawdown_tuw follows, times the gap to the next HWM followed by a
  drawdown, which adds time spent making new highs.) Max and 95th percentile (14.5.3).
- Correlation with the underlying (14.3): Pearson over windows between path P&L and the sign of the BTC move that
  settled the window (+1 Up won, -1 Down won), or a given series per window.

Strategy risk (ch. 15), per path over traded windows with outcome risk (Up shares != Down shares)
- A window pays pi+ = max(up, down) - cost - fees when its larger payout happens, pi- = min(...) - cost - fees
  otherwise; precision p = the share of windows paid pi+, n = such windows per year.
- Two-outcome form: strategy_risk.strategy_risk on the windows' P&L (implied precision and frequency, failure
  probabilities), NaN there when the payouts are not near two-point.
- Payout form, the book's binomial bets with each bet's own payouts: a bet drawn from the observed ones wins its pi+_i
  with probability p, so E[X] = p mean(pi+) + (1 - p) mean(pi-) and E[X^2] = p mean(pi+^2) + (1 - p) mean(pi-^2);
  theta(p) = E[X] / sd[X] sqrt(n). theta = theta* is a quadratic in p; its root in [0, 1] with E[X] > 0 is the
  precision needed (theta rises with p there), and with constant payouts it is the book's implied precision (15.4).
  Failure probability P[p_hat < p*] for the precision of floor(n k) bets drawn with replacement (15.4.1): their wins
  are exactly binomial(floor(n k), p), so the distribution needs no kernel estimate.

Selection (11.6, 12.5, 14.7.3)
- PBO by CSCV (cpcv.pbo) over a periods x configurations matrix of path-mean P&L per Taiwan day (or window); a
  configuration idle in a period earns 0 there.
- statistics()["path_mean"] is 12.5's variance of the mean over paths: the paths reuse the same C(N, k) split
  forecasts, so they are correlated, and the precision of the mean path Sharpe is phi^-1 sigma^2 (1 + (phi - 1)
  rho_bar), not sigma^2 / phi. rho_bar is the average off-diagonal correlation of the paths' own window P&L series
  (cpcv.mean_correlation), and effective_paths says how many independent paths would give the same precision
  (docs/afml_sections/ch12.md F12.2).
- Deflated Sharpe per path of the selected configuration: the primary column is evaluation.luck_hurdle on its outcome
  leg (per traded window up and down shares, cost, fees, pnl; audit R6 finding 9) with N = the trial registry's
  selection trials + this study's configurations. The Gaussian column beside it (metrics) takes BOTH its V[SR] and
  its N from this study's configurations on the same path, put on one scale by z = SR sqrt(n - 1): V and N then
  describe the same trial set, where taking N from the whole registry and V from a handful of configurations mixed
  two (docs/afml_sections/ch14.md F14.7). It is NaN below GAUSSIAN_MIN_CONFIGS configurations, too few for a variance.

Docstrings cite the book's sections; they do not reproduce its text or code.
"""
import json
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binom

from pmlab import WINDOW_SECONDS as T
from pmlab import evaluation, metrics
from pmlab.afml import backtest_stats, cpcv, strategy_risk
from pmlab.afml.events import EMBARGO_S

TW = 8 * 3600                     # Taiwan time, UTC+8
DAY = 86400
EMBARGO = EMBARGO_S
YEAR = 365.25 * DAY
OUTCOMES = ("up_won", "payoff", "pnl", "ret", "label", "meta_label", "avg_uniqueness", "w_return_raw")
REGISTRY = Path(__file__).resolve().parents[3] / "research" / "trial_registry.json"
LEG = ("up", "down", "cost", "fees", "pnl")


def fixed_stake(dollars: float = 1.0) -> Callable:
    """Sizing that orders the same dollars at every bet, whatever the probability."""
    return lambda prob, rows: np.full(len(rows), float(dollars))


@dataclass(frozen=True)
class Strategy:
    """model(training rows, test rows without OUTCOMES) -> P(side wins) per test row, or None to take every event;
    threshold: bet only when P >= threshold (needs a model); sizing(P of the bets, their rows without OUTCOMES) ->
    order dollars >= 0 per bet (NaN: no bet)."""
    name: str
    model: Callable | None = None
    threshold: float | None = None
    sizing: Callable = fixed_stake()


# ----------------------------------------------------------------------------------------------- CPCV over days

def taiwan_day(seconds) -> np.ndarray:
    """Taiwan calendar day (days since 1970-01-01 in UTC+8) of unix seconds."""
    return (np.asarray(seconds, dtype=np.int64) + TW) // DAY


def day_groups(days, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """(distinct days in order, group of each): N contiguous groups of near-equal day counts."""
    u = np.unique(np.asarray(days, dtype=np.int64))
    if len(u) < n_groups:
        raise ValueError(f"{len(u)} days cannot make {n_groups} groups")
    g = np.empty(len(u), dtype=int)
    for k, block in enumerate(np.array_split(np.arange(len(u)), n_groups)):
        g[block] = k
    return u, g


def purged_splits(t0, t1, group, n_groups: int, k_test: int, embargo: int = EMBARGO) -> list[dict]:
    """One dict per split in cpcv.split_groups order: test groups, train and test positions, and how many rows
    purging and then the embargo removed from training."""
    t0, t1, group = (np.asarray(a, dtype=np.int64) for a in (t0, t1, group))
    env = {g: (t0[group == g].min(), t1[group == g].max()) for g in range(n_groups) if np.any(group == g)}
    out = []
    for groups in cpcv.split_groups(n_groups, k_test):
        test = np.isin(group, groups)
        purged, embargoed = np.zeros(len(t0), bool), np.zeros(len(t0), bool)
        for g in groups:
            if g in env:
                lo, hi = env[g]
                purged |= (t0 < hi) & (t1 > lo)
                embargoed |= (t0 >= hi) & (t0 < hi + embargo)
        train = ~test & ~purged & ~embargoed
        out.append({"groups": groups, "train": np.flatnonzero(train), "test": np.flatnonzero(test),
                    "purged": int(np.sum(~test & purged)), "embargoed": int(np.sum(~test & ~purged & embargoed))})
    return out


def _rows(ev: pd.DataFrame) -> pd.DataFrame:
    side = ev["side"].to_numpy(float)
    if "mid" in ev.columns:
        mid = ev["mid"].to_numpy(float)
    elif {"ask_up", "bid_up"} <= set(ev.columns):
        up = (ev["ask_up"].to_numpy(float) + ev["bid_up"].to_numpy(float)) / 2
        mid = np.where(side > 0, up, 1 - up)
    else:
        mid = np.full(len(ev), np.nan)
    payoff = ev["payoff"].to_numpy(float)
    return pd.DataFrame({"row": ev.index.to_numpy(), "start": ev["start"].to_numpy(np.int64),
                         "t0": ev["t0"].to_numpy(np.int64), "t1": ev["t1"].to_numpy(np.int64), "side": side,
                         "price": ev["price"].to_numpy(float), "fee": ev["fee"].to_numpy(float),
                         "cost": ev["cost"].to_numpy(float), "payoff": payoff, "mid": mid,
                         "fill_ratio": ev["fill_ratio"].to_numpy(float) if "fill_ratio" in ev.columns else 1.0,
                         "up_won": np.where(side > 0, payoff, 1 - payoff)})


def run_cpcv(events: pd.DataFrame, strategy: Strategy, n_groups: int, k_test: int, embargo: int = EMBARGO,
             windows=None) -> dict:
    """CPCV backtest of one strategy on one primary rule's events (columns start, t0, t1, side, price, fee, cost,
    payoff; optional mid or ask_up/bid_up, fill_ratio). `windows`: every eligible window start (default: the events'
    windows); windows without a bet earn 0. Returns the splits with each test row's probability and order dollars,
    the path matrix and window_pnl (paths x windows, dollars)."""
    if "rule" in events.columns and events["rule"].nunique() > 1:
        raise ValueError("one primary rule per backtest: rows of different rules share every feature")
    if strategy.threshold is not None and strategy.model is None:
        raise ValueError("a threshold filters a model's probabilities")
    if len(events) and (events["t1"].to_numpy(np.int64) > evaluation.CONFIRM_START).any():
        raise ValueError("a row's window ends after the confirmation start")
    ev = events.sort_values("t0", kind="stable")
    rows = _rows(ev)
    start = rows["start"].to_numpy()
    windows = np.unique(np.asarray(start if windows is None else windows, dtype=np.int64))
    if not np.isin(start, windows).all():
        raise ValueError("every event's window must be one of the windows")
    days, day_group = day_groups(taiwan_day(windows), n_groups)
    group = day_group[np.searchsorted(days, taiwan_day(start))]
    splits = purged_splits(rows["t0"], rows["t1"], group, n_groups, k_test, embargo)
    hidden = ev.drop(columns=[c for c in OUTCOMES if c in ev.columns])
    for sp in splits:
        test = hidden.iloc[sp["test"]]
        prob = (np.full(len(test), np.nan) if strategy.model is None
                else np.asarray(strategy.model(ev.iloc[sp["train"]], test), dtype=float))
        take = np.ones(len(test), bool) if strategy.threshold is None else prob >= strategy.threshold
        dollars = np.zeros(len(test))
        if take.any():
            dollars[take] = np.nan_to_num(np.asarray(strategy.sizing(prob[take], test[take]), dtype=float))
        if (dollars < 0).any():
            raise ValueError("sizing returned a negative order")
        sp["prob"], sp["dollars"] = prob, dollars
    up_won = pd.Series(rows["up_won"].to_numpy()).groupby(start).first().reindex(windows).to_numpy(float)
    out = {"strategy": strategy.name, "n_groups": n_groups, "k_test": k_test, "embargo": embargo,
           "paths": cpcv.paths(n_groups, k_test), "splits": splits, "rows": rows, "group": group,
           "windows": windows, "days": days, "day_group": day_group, "window_up_won": up_won}
    out["window_pnl"] = np.vstack([window_pnl(path_bets(out, j), windows) for j in range(len(out["paths"]))])
    return out


def order_costs(bets: pd.DataFrame) -> pd.DataFrame:
    """Shares, P&L and implementation shortfall parts of each order (module docstring)."""
    ordered = bets["dollars"] / bets["cost"]
    filled = ordered * bets["fill_ratio"]
    edge = bets["payoff"] - bets["mid"]
    return bets.assign(ordered=ordered, filled=filled, pnl=filled * (bets["payoff"] - bets["cost"]),
                       paper=ordered * edge, arrival_cost=filled * (bets["price"] - bets["mid"]),
                       fees=filled * bets["fee"], opportunity_cost=(ordered - filled) * edge,
                       shortfall=lambda d: d["arrival_cost"] + d["fees"] + d["opportunity_cost"])


def path_bets(result: dict, j: int) -> pd.DataFrame:
    """The orders of path j: for each group g, the rows of g with the probability and dollars split paths[j, g] gave
    them; rows without an order are left out."""
    rows, group, P = result["rows"], result["group"], result["paths"]
    n = len(rows)
    prob, dollars, split = np.full(n, np.nan), np.zeros(n), np.full(n, -1)
    for g in range(P.shape[1]):
        s = P[j, g]
        sp = result["splits"][s]
        mine = group[sp["test"]] == g
        pos = sp["test"][mine]
        prob[pos], dollars[pos], split[pos] = sp["prob"][mine], sp["dollars"][mine], s
    bets = rows.assign(group=group, split=split, prob=prob, dollars=dollars)
    return order_costs(bets[dollars > 0])


def window_pnl(bets: pd.DataFrame, windows) -> np.ndarray:
    return np.bincount(np.searchsorted(windows, bets["start"].to_numpy()), weights=bets["pnl"].to_numpy(),
                       minlength=len(windows))


def window_legs(bets: pd.DataFrame) -> pd.DataFrame:
    """Per window with a filled order: Up and Down shares held, cost (shares x price), fees, pnl and whether Up won;
    the outcome leg of pmlab.evaluation."""
    b = bets[bets["filled"] > 0]
    up = (b["side"] > 0).to_numpy()
    f = b["filled"].to_numpy()
    legs = pd.DataFrame({"start": b["start"].to_numpy(), "up": np.where(up, f, 0.0), "down": np.where(up, 0.0, f),
                         "cost": f * b["price"].to_numpy(), "fees": b["fees"].to_numpy(), "pnl": b["pnl"].to_numpy(),
                         "up_won": b["up_won"].to_numpy()})
    return legs.groupby("start").agg(up=("up", "sum"), down=("down", "sum"), cost=("cost", "sum"),
                                     fees=("fees", "sum"), pnl=("pnl", "sum"), up_won=("up_won", "first"))


# ----------------------------------------------------------------------------------------------- ch. 14 statistics

def drawdowns(times, equity, relative: bool = False) -> pd.DataFrame:
    """One row per drawdown episode of an equity curve (14.5.2, module docstring): hwm_time, hwm, trough, drawdown
    (HWM - trough, or 1 - trough / HWM when relative), end_time (when equity first exceeds the HWM, else the last
    time), recovered and time_under_water (end_time - hwm_time)."""
    t, e = np.asarray(times), np.asarray(equity, dtype=float)
    dd = backtest_stats.drawdown_tuw(t, e, dollars=not relative)
    k = np.searchsorted(np.maximum.accumulate(e), dd["hwm"], side="right")      # the first point above the mark
    end = t[np.minimum(k, len(e) - 1)]
    return pd.DataFrame({"hwm_time": dd["hwm_time"], "hwm": dd["hwm"], "trough": dd["min"], "drawdown": dd["drawdown"],
                         "end_time": end, "recovered": k < len(e), "time_under_water": end - dd["hwm_time"]})


def twrr(pnl, bankroll: float | None, years: float) -> float:
    """Annualised time-weighted rate of return of an account starting at `bankroll` (14.4.1); -1 once it is lost,
    NaN without a bankroll (module docstring)."""
    if bankroll is None:
        return np.nan
    pnl = np.asarray(pnl, dtype=float)
    k = bankroll + np.concatenate([[0.0], np.cumsum(pnl)[:-1]])
    if np.any(k <= 0):
        return -1.0
    phi = float(np.prod(1 + pnl / k))
    return -1.0 if phi <= 0 else float(phi ** (1 / years) - 1)


def aum(bets: pd.DataFrame, t_first: float, t_last: float) -> dict:
    """14.3's dollar characteristics of a path. Every purchase is held to its window's settlement, so the dollar value
    of the open position is its filled shares x price from t0 to t1 (a Down share counts positive, as the section
    asks). avg_aum time-weights that value over the whole backtest span; max_position_dollars is the largest dollar
    value open at one instant; turnover_annual is the dollars traded per year over the average AUM. Capacity is not
    here because it needs a market-impact coefficient of the book: pass `impact` to path_statistics and it reports
    `pmlab.afml.backtest_stats.capacity` beside these. Leverage is 1 by construction, nothing being borrowed."""
    b = bets[bets["filled"] > 0]
    span = float(t_last - t_first)
    if not len(b) or span <= 0:
        return {"avg_aum": np.nan, "max_position_dollars": np.nan, "turnover_annual": np.nan}
    value = (b["filled"] * b["price"]).to_numpy(float)
    t0, t1 = b["t0"].to_numpy(float), b["t1"].to_numpy(float)
    avg = float((value * (t1 - t0)).sum() / span)
    edges = np.unique(np.concatenate([t0, t1]))                       # the open value only changes at a t0 or a t1
    # Open value at x = sum of the purchases with t0 <= x < t1. Every span ends after it starts, so t1 <= x implies
    # t0 <= x and the second sum is a subset of the first: the difference of the two prefix sums is that set exactly.
    o0, o1 = np.argsort(t0, kind="stable"), np.argsort(t1, kind="stable")
    c0 = np.concatenate([[0.0], np.cumsum(value[o0])])
    c1 = np.concatenate([[0.0], np.cumsum(value[o1])])
    x = edges[:-1]
    held = c0[np.searchsorted(t0[o0], x, side="right")] - c1[np.searchsorted(t1[o1], x, side="right")]
    years = span / YEAR
    return {"avg_aum": avg, "max_position_dollars": float(held.max()) if len(held) else np.nan,
            "turnover_annual": float(value.sum() / years / avg) if avg > 0 and years > 0 else np.nan}


def bet_timing(bets: pd.DataFrame) -> dict:
    """Bets on the net Up-equivalent position, purchases, share-weighted holding period (s), Up share (14.3)."""
    b = bets[bets["filled"] > 0].sort_values(["start", "t0"], kind="stable")
    net = (b["side"] * b["filled"]).groupby(b["start"]).cumsum().round(9)
    settle = b.groupby("start")["t1"].first().iloc[:-1]            # bet_timing's last time closes what is still open
    seq = pd.concat([pd.DataFrame({"start": b["start"], "k": 0, "time": b["t0"], "pos": net}),
                     pd.DataFrame({"start": settle.index, "k": 1, "time": settle.to_numpy(), "pos": 0.0})])
    seq = seq.sort_values(["start", "k", "time"], kind="stable")
    n = len(backtest_stats.bet_timing(seq["time"].to_numpy(), seq["pos"].to_numpy(float)))
    hold = float(np.average(b["t1"] - b["t0"], weights=b["filled"])) if len(b) else np.nan
    return {"bets": n, "purchases": int(len(b)), "holding_seconds": hold,
            "up_share": float((b["side"] > 0).mean()) if len(b) else np.nan}


def shortfall(bets: pd.DataFrame) -> dict:
    """Implementation shortfall sums and per-turnover costs (14.6) over the orders with a decision mid."""
    b = bets[np.isfinite(bets["mid"].to_numpy(float))]
    s = {k: float(b[k].sum()) for k in ("paper", "arrival_cost", "fees", "opportunity_cost", "shortfall", "pnl")}
    turnover = float((b["filled"] * b["price"]).sum())
    ratio = lambda a, c: a / c if c else np.nan
    return {"is_orders": int(len(b)), "is_paper": s["paper"], "is_arrival_cost": s["arrival_cost"], "is_fees": s["fees"],
            "is_opportunity_cost": s["opportunity_cost"], "implementation_shortfall": s["shortfall"],
            "is_pnl": s["pnl"], "turnover": turnover, "fees_per_turnover": ratio(s["fees"], turnover),
            "slippage_per_turnover": ratio(s["arrival_cost"], turnover), "pnl_per_turnover": ratio(s["pnl"], turnover),
            "return_on_execution_costs": ratio(s["pnl"], s["arrival_cost"] + s["fees"])}


# ----------------------------------------------------------------------------------------------- ch. 15 strategy risk

def _moments(pi_plus, pi_minus) -> tuple[float, float, float, float]:
    a, b = np.asarray(pi_plus, dtype=float), np.asarray(pi_minus, dtype=float)
    return float(a.mean()), float(b.mean()), float(np.mean(a ** 2)), float(np.mean(b ** 2))


def payout_sharpe(p: float, pi_plus, pi_minus, n: float) -> float:
    """Annual Sharpe of n bets a year that win their own pi+ with probability p, else their pi- (module docstring)."""
    a, b, sa, sb = _moments(pi_plus, pi_minus)
    mean = p * a + (1 - p) * b
    var = p * sa + (1 - p) * sb - mean ** 2
    return float(mean / np.sqrt(var) * np.sqrt(n)) if var > 0 else np.nan


def precision_needed(pi_plus, pi_minus, n: float, theta: float) -> float:
    """The precision at which payout_sharpe reaches theta > 0; NaN when no precision in [0, 1] does."""
    a, b, sa, sb = _moments(pi_plus, pi_minus)
    d, m = a - b, n + theta ** 2
    qa, qb, qc = m * d ** 2, 2 * m * b * d - theta ** 2 * (sa - sb), m * b ** 2 - theta ** 2 * sb
    disc = qb ** 2 - 4 * qa * qc
    if qa <= 0 or disc < 0:
        return np.nan
    roots = [(-qb + sgn * np.sqrt(disc)) / (2 * qa) for sgn in (1, -1)]
    ok = [r for r in roots if 0 <= r <= 1 and b + r * d > 0]
    return float(min(ok)) if ok else np.nan


def prob_failure_binomial(p: float, p_star: float, n_bets: int) -> float:
    """P[precision of n_bets bets < p_star] when each wins with probability p (15.4.1); 1 when p_star is out of
    reach."""
    if not np.isfinite(p_star):
        return 1.0
    return float(binom.cdf(ceil(p_star * n_bets - 1e-9) - 1, n_bets, p))


def window_risk(legs: pd.DataFrame, years: float, target_sharpe: float = 2.0, years_judged: float = 2.0,
                rng=0) -> dict:
    """Chapter 15 on traded windows with outcome risk (module docstring)."""
    pay_up = (legs["up"] - legs["cost"] - legs["fees"]).to_numpy()
    pay_down = (legs["down"] - legs["cost"] - legs["fees"]).to_numpy()
    up, down = legs["up"].to_numpy(), legs["down"].to_numpy()
    risky = np.abs(up - down) > 1e-9
    hi, lo = np.maximum(pay_up, pay_down)[risky], np.minimum(pay_up, pay_down)[risky]
    won = np.where(legs["up_won"].to_numpy() == 1, up > down, down > up)[risky]
    m = int(risky.sum())
    n = m / years
    out = {"windows": m, "bets_per_year": n, "target_sharpe": target_sharpe}
    if m < 2:
        return out
    p = float(won.mean())
    p_star = precision_needed(hi, lo, n, target_sharpe)
    out.update(precision=p, pi_plus=float(hi.mean()), pi_minus=float(lo.mean()),
               sharpe=payout_sharpe(p, hi, lo, n), breakeven_precision=float(-lo.mean() / (hi.mean() - lo.mean())),
               precision_needed=p_star, prob_failure=prob_failure_binomial(p, p_star, floor(n * years_judged)))
    two = strategy_risk.strategy_risk(legs["pnl"].to_numpy()[risky], n, target_sharpe, years_judged, rng=rng)
    out.update({f"two_outcome_{k}": v for k, v in two.items() if k not in ("bets", "bets_per_year", "target_sharpe")})
    return out


# ----------------------------------------------------------------------------------------------- per path and paths

def path_statistics(result: dict, j: int, bankroll: float | None = None, target_sharpe: float = 2.0,
                    years_judged: float = 2.0, underlying: pd.Series | None = None, rng=0,
                    impact: float | None = None) -> dict:
    """Chapter 14 and 15 statistics of path j (module docstring). `underlying`: a value per window start. `impact`:
    the dollars of average fill-price move per dollar sent, which turns 14.3's capacity on."""
    windows, pnl = result["windows"], result["window_pnl"][j]
    bets = path_bets(result, j)
    legs = window_legs(bets)
    years = (windows[-1] + T - windows[0]) / YEAR
    sr = metrics.sharpe(pnl)
    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    times = np.concatenate([[windows[0]], windows + T])
    dd = drawdowns(times, equity if bankroll is None else bankroll + equity, relative=bankroll is not None)
    und = (2 * result["window_up_won"] - 1 if underlying is None
           else pd.Series(underlying).reindex(windows).to_numpy(float))
    ok = np.isfinite(und)
    corr = (float(np.corrcoef(pnl[ok], und[ok])[0, 1]) if ok.sum() > 2 and np.ptp(pnl[ok]) > 0 and np.ptp(und[ok]) > 0
            else np.nan)
    traded = legs["pnl"].to_numpy()
    invested = legs["cost"].to_numpy() + legs["fees"].to_numpy()
    ret = np.divide(traded, invested, out=np.full(len(traded), np.nan), where=invested > 0)
    timing = bet_timing(bets)
    conc = backtest_stats.concentration(traded, taiwan_day(legs.index.to_numpy()), result["days"])
    q = lambda s, f: float(f(s)) if len(s) else 0.0
    m = lambda a: float(a.mean()) if len(a) else np.nan
    up_bet = bets["side"].to_numpy() > 0
    dollars = aum(bets, float(windows[0]), float(windows[-1] + T))
    cap = {}
    if impact is not None:                                               # 14.3's capacity, given a book impact
        sent = pd.Series(0.0, index=pd.Index(windows))
        sent.loc[legs.index] = invested
        c = backtest_stats.capacity(pnl, sent.to_numpy(), impact, target_sharpe, len(windows) / years,
                                    dollars["avg_aum"])
        cap = {"capacity": c["capacity"], "capacity_multiple": c["multiple"]}
    return {"path": j, "windows": int(len(windows)), "traded_windows": int(len(legs)), "years": years,
            "pnl": float(pnl.sum()), "sharpe": sr, "sharpe_annual": sr * np.sqrt(len(windows) / years),
            "twrr_annual": twrr(pnl, bankroll, years),
            "pnl_long": float(bets["pnl"].to_numpy()[up_bet].sum()),        # 14.4: P&L from Up (long) positions
            "pnl_short": float(bets["pnl"].to_numpy()[~up_bet].sum()),
            "hit_ratio": float(np.mean(traded > 0)) if len(traded) else np.nan,
            "avg_hit": m(ret[traded > 0]), "avg_miss": m(ret[traded < 0]),  # 14.4: RETURNS of hits and of misses,
            "flat_windows": int(np.sum(traded == 0)),                       # a zero-P&L window being neither
            "avg_hit_dollars": m(traded[traded > 0]), "avg_miss_dollars": m(traded[traded < 0]),
            **dollars, **cap,
            **timing, "bets_per_year": timing["bets"] / years,
            "hhi_plus": conc["h_pos"], "hhi_minus": conc["h_neg"], "hhi_time": conc["h_time"],
            "dd_max": q(dd["drawdown"], np.max), "dd_p95": q(dd["drawdown"], lambda s: np.quantile(s, 0.95)),
            "tuw_max_days": q(dd["time_under_water"], np.max) / DAY,
            "tuw_p95_days": q(dd["time_under_water"], lambda s: np.quantile(s, 0.95)) / DAY,
            "corr_underlying": corr, **shortfall(bets),
            **{f"risk_{k}": v for k, v in window_risk(legs, years, target_sharpe, years_judged, rng).items()}}


def path_mean(result: dict, per: pd.DataFrame, column: str = "sharpe") -> dict:
    """12.5's variance of the mean of a per-path statistic, with rho_bar measured on the paths' window P&L
    (module docstring). `per` is statistics()["paths"]."""
    return {"statistic": column, **cpcv.path_mean_variance(per[column].to_numpy(float),
                                                           cpcv.mean_correlation(result["window_pnl"]))}


def statistics(result: dict, **kw) -> dict:
    """{"paths": one row of path_statistics per path, "summary": mean, sd, min, median and max over paths,
    "path_mean": 12.5's variance of the mean path Sharpe given how correlated the paths are}."""
    per = pd.DataFrame([path_statistics(result, j, **kw) for j in range(len(result["paths"]))])
    num = per.select_dtypes("number").drop(columns="path")
    return {"paths": per, "summary": num.agg(["mean", "std", "min", "median", "max"]).T,
            "path_mean": path_mean(result, per)}


# ----------------------------------------------------------------------------------------------- selection

def config_matrix(results: dict, by: str = "day", path: int | None = None) -> pd.DataFrame:
    """Periods x configurations of P&L (the mean over paths, or one path), by Taiwan day or window; 0 where a
    configuration has no window."""
    cols = {}
    for name, r in results.items():
        pnl = r["window_pnl"].mean(axis=0) if path is None else r["window_pnl"][path]
        s = pd.Series(pnl, index=r["windows"])
        cols[name] = s.groupby(taiwan_day(s.index.to_numpy())).sum() if by == "day" else s
    return pd.DataFrame(cols).sort_index().fillna(0.0)


def pbo_cscv(results: dict, n_blocks: int = 16, by: str = "day", path: int | None = None, **kw) -> dict:
    """Probability of backtest overfitting over every configuration of a study (cpcv.pbo on config_matrix)."""
    M = config_matrix(results, by, path)
    return {**cpcv.pbo(M.to_numpy(), n_blocks, **kw), "configurations": list(M.columns)}


def registry_trials(path: Path = REGISTRY) -> int:
    return int(json.loads(Path(path).read_text())["total_selection_trials"])


GAUSSIAN_MIN_CONFIGS = 5      # fewer configurations than this give no usable V[SR] for the Gaussian column


def deflated_sharpe(results: dict, registry: int, selected: str | None = None, n_sims: int = 20000,
                    seed: int = 0) -> dict:
    """Deflated Sharpe per path of the selected configuration (by default the best mean path Sharpe of per traded
    window returns). The primary column uses N = registry + configurations; the Gaussian column beside it takes both
    its V[SR] and its N from this study's configurations, the trial set V is measured on (module docstring)."""
    names = list(results)
    n_paths = {len(r["paths"]) for r in results.values()}
    if len(n_paths) != 1:
        raise ValueError("configurations must share their CPCV paths")
    p = n_paths.pop()
    n_trials = registry + len(names)
    legs = {k: [window_legs(path_bets(r, j)) for j in range(p)] for k, r in results.items()}

    def sharpe_n(leg):
        r = evaluation.leg_returns({c: leg[c].to_numpy(float) for c in LEG})
        return metrics.sharpe(r), len(r)
    srs = {k: [sharpe_n(leg) for leg in ls] for k, ls in legs.items()}
    if selected is None:
        mean_sr = {k: np.mean(f) if (f := [s for s, _ in v if np.isfinite(s)]) else -np.inf for k, v in srs.items()}
        selected = max(names, key=mean_sr.get)
    rows = []
    for j in range(p):
        leg = {c: legs[selected][j][c].to_numpy(float) for c in LEG}
        h = evaluation.luck_hurdle(leg, n_trials, n_sims, seed)
        r = evaluation.leg_returns(leg)
        s = metrics.summary(r) if len(r) >= 3 else None
        z = [sr * np.sqrt(n - 1) for sr, n in (srs[k][j] for k in names) if np.isfinite(sr) and n >= 3]
        sr_star_g = dsr_g = np.nan
        if s is not None and np.isfinite(s["sharpe"]) and len(z) >= GAUSSIAN_MIN_CONFIGS:
            # N and V[SR] from the same trial set: this study's configurations (docs/afml_sections/ch14.md F14.7)
            sr_star_g = metrics.expected_max_sharpe(len(z), np.var(z, ddof=1) / (s["n"] - 1))
            dsr_g = metrics.probabilistic_sharpe_ratio(s["sharpe"], s["n"], s["skew"], s["kurtosis"], sr_star_g)
        rows.append({"path": j, "traded_windows": h["n"], "sharpe": h["sharpe"], "null_sd": h["null_sd"],
                     "sr_star": h["sr_star"], "dsr": h["dsr"], "sr_star_gaussian": sr_star_g, "dsr_gaussian": dsr_g,
                     "gaussian_trials": len(z)})
    per = pd.DataFrame(rows)
    return {"selected": selected, "n_trials": n_trials, "registry_trials": registry, "configurations": len(names),
            "paths": per, "dsr_median": float(per["dsr"].median()), "dsr_min": float(per["dsr"].min()),
            "share_clearing": float((per["dsr"] >= 0.95).mean())}
