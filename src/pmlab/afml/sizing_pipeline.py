"""Bet sizing for the research pipeline (AFML ch. 10) in dollars, for binary tokens held to expiry.

Plan afml-pipeline node 12. Stage 5 runs `evaluate` on stage 4's out-of-fold probabilities.

Input: one row per event and primary rule, stage 4's predictions joined to the pmlab.afml.events rows on (start, t, rule):
  start   window start (unix s); the window is the position's expiry
  t0, t1  decision second and span end (the window end), as in the event rows
  rule    primary rule; each rule is sized and scored on its own
  side    +1 buys Up, -1 buys Down
  price   the side's ask
  prob    out-of-fold P(the side wins)
  payoff  1 if the side won, else 0
  fold    optional: the purged fold whose test set held the row
No row of a window ending after pmlab.evaluation.CONFIRM_START is accepted, as in pmlab.afml.models.

Money. A share bought at ask p costs a = p + 0.07 p (1 - p) all-in (pmlab.fees.fee_per_share, unrounded). A stake of
S dollars buys C = S / a shares, of which C x 0.07 p (1 - p) dollars is fee, and pays C at settlement if the side wins:
P&L = C payoff - S. Nothing is sold before settlement.

Sizes. Each method gives every event a size in [0, 1] and a side; pmlab.afml.betsizing holds the per-bet formulas.
- book (10.3): the predicted class is the most probable of K; its probability p~ against the no-information 1/K gives
  z = (p~ - 1/K) / sqrt(p~ (1 - p~)) and size x (2 Phi(z) - 1), x the predicted class's label (book_size). For
  meta-labels the classes are (0, 1) and the primary rule's side is the sign: predicting 0 stakes nothing. The book
  sizes on the probability alone, not on what the token costs. null="price" is the alternative betsizing.size_vs_price
  (the all-in cost as the null, floored at 0).
- average (10.4): the signed size at an event is the mean of the signed sizes of every bet still open at that second,
  t0_i <= t0 < t1_i, so only bets decided at or before the event enter it (active_average).
- step (10.5): the averaged size rounded to multiples of the step (betsizing.discrete_size), after averaging.
- dynamic (10.6): the forecast is prob (the token's expected settlement value), the market price is the all-in cost,
  the size is m(w, prob - cost) with the book's sigmoid x / sqrt(w + x^2) or its power form sgn(x)|x|^w, w calibrated so
  a chosen divergence gives a chosen size (calibrate), the target is int(m Q) units of Q, floored at 0 (a token is
  bought, never sold short). The breakeven limit price for moving from the held units to the target is the mean of the
  inverse function over each unit added (limit_cost), converted back from all-in to an ask (price_from_cost).
- baselines: fixed stake (the maximum stake per bet) when the meta-label predicts 1 (prob > 1/2) or, with `edge`, when
  prob - cost > edge; Kelly on the all-in cost, kelly x (prob - a) / (1 - a) of the bankroll as the target of the
  window's whole position, as pmlab.strategies.ProbTaker sizes it.

Dollars (simulate), in decision order, one rule at a time:
- book and dynamic sizes are targets for the window's position in units of the maximum stake; the position only grows
  toward them (a held token cannot be sold), on the event's own side (its price is the only one known), never on the
  side opposite a position already held in the window (as ProbTaker). Fixed stakes are separate bets.
- limits as the live strategies (pmlab.risk.DEFAULT): per order at most max_stake_frac of the bankroll; per window
  committed cost at most max_window_cost_frac of it (an order is shrunk to the room left); per Taiwan day no bet once
  the realised P&L of that day's settled windows is <= -daily_loss_frac x the bankroll before its first settlement;
  an order under MIN_ORDER_COST dollars is not placed; never more than the cash not already committed.
- the bankroll used for sizing is the realised bankroll visible at the decision (compound=True) or the starting one.
  A window's P&L becomes visible SETTLE_DELAY seconds after the window ends, never earlier.

Evaluation (evaluate): per rule, per fold and for every fold's rows in time order on one bankroll ("all"), every sizing
on the same rows: per-window dollar P&L and stake (untraded windows 0), return on stake, Sharpe of window returns on
the bankroll before each settlement, log growth log(B_end / B_0), maximum drawdown.

Docstrings cite the book's sections; they do not reproduce its text or code.
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS
from pmlab.afml import betsizing as bs
from pmlab.evaluation import CONFIRM_START
from pmlab.fees import FEE_RATE, fee_per_share
from pmlab.metrics import max_drawdown, sharpe
from pmlab.risk import DEFAULT as LIVE, MIN_ORDER_COST, SETTLE_DELAY, hk_day

BANKROLL = 1000.0                               # pmlab.backtest.Config and the live engine's starting bankroll
MAX_STAKE_FRAC = LIVE.max_window_cost_frac      # a full-size bet uses the whole window budget
WINDOW_CAP_FRAC = LIVE.max_window_cost_frac     # 0.05
DAILY_LOSS_FRAC = LIVE.daily_loss_frac          # 0.10
KELLY = 0.25                                    # ProbTaker's default fraction of full Kelly
REQUIRED = ("start", "t0", "t1", "rule", "side", "price", "prob", "payoff")
TINY = 1e-9                                     # averaged sizes below this are float noise from cumulative sums


# ------------------------------------------------------------------------------------------------ money

def cost_of(price):
    """All-in cost per share of a token bought at ask `price`: price + taker fee per share."""
    p = np.asarray(price, dtype=float)
    return p + fee_per_share(p)


def price_from_cost(cost, rate: float = FEE_RATE):
    """The ask p in [0, 1] whose all-in cost p + rate p (1 - p) is `cost`."""
    a = np.asarray(cost, dtype=float)
    return ((1.0 + rate) - np.sqrt((1.0 + rate) ** 2 - 4.0 * rate * a)) / (2.0 * rate)


def bet_pnl(stake, price, payoff) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(shares, fee dollars, P&L dollars) of stakes (dollars, fee included) on tokens bought at `price`, held to
    settlement at `payoff` per share."""
    s, p = np.asarray(stake, dtype=float), np.asarray(price, dtype=float)
    shares = s / cost_of(p)
    return shares, shares * fee_per_share(p), shares * np.asarray(payoff, dtype=float) - s


# ------------------------------------------------------------------------------------------------ sizes

def book_size(prob, labels=(0, 1), side=None) -> np.ndarray:
    """10.3. `prob`: P(labels[1]) per row for two classes, or an (n, K) matrix of class probabilities ordered as
    `labels`, the size each class stands for. Size = label of the most probable class x (2 Phi(z) - 1) with
    z = (p~ - 1/K) / sqrt(p~ (1 - p~)), p~ that class's probability; times `side` when given (meta-labels)."""
    P = np.asarray(prob, dtype=float)
    if P.ndim == 1:
        P = np.column_stack([1.0 - P, P])
    if P.shape[1] != len(labels):
        raise ValueError(f"{P.shape[1]} class probabilities for {len(labels)} labels")
    k = P.argmax(axis=1)
    m = bs.size_from_prob(P[np.arange(len(P)), k], n_classes=P.shape[1], side=np.asarray(labels, dtype=float)[k])
    return m if side is None else m * np.asarray(side, dtype=float)


def active_average(t0, t1, size) -> np.ndarray:
    """10.4 at each bet's decision: the mean size of the bets open then, t0_i <= t0_j < t1_i (t1 NaN or inf: open).
    The same as betsizing.avg_active_sizes read at the decision times, in O(n log n)."""
    t0, size = np.asarray(t0, dtype=float), np.asarray(size, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    t1 = np.where(np.isfinite(t1), t1, np.inf)
    if np.any(t1 <= t0):
        raise ValueError("every span must end after its decision")
    o0, o1 = np.argsort(t0, kind="stable"), np.argsort(t1, kind="stable")
    c0 = np.concatenate([[0.0], np.cumsum(size[o0])])
    c1 = np.concatenate([[0.0], np.cumsum(size[o1])])
    opened = np.searchsorted(t0[o0], t0, side="right")          # bets with t0_i <= t0_j
    closed = np.searchsorted(t1[o1], t0, side="right")          # bets with t1_i <= t0_j (these opened earlier)
    return (c0[opened] - c1[closed]) / (opened - closed)


def calibrate(x: float, m: float, form: str = "sigmoid") -> float:
    """10.6: the w that maps a divergence x to size m. Sigmoid: x^2 (m^-2 - 1). Power, x and m in (0, 1):
    ln m / ln |x|."""
    if form == "sigmoid":
        return bs.calibrate_w(x, m)
    if form == "power":
        return math.log(abs(m)) / math.log(abs(x))
    raise ValueError(f"unknown form {form!r}")


def dynamic_size(w: float, x, form: str = "sigmoid") -> np.ndarray:
    """10.6: size for divergence x = forecast - price. Sigmoid x / sqrt(w + x^2); power sgn(x) |x|^w on [-1, 1]."""
    if form == "sigmoid":
        return bs.bet_size(w, x)
    if form == "power":
        x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
        return np.sign(x) * np.abs(x) ** w
    raise ValueError(f"unknown form {form!r}")


def dynamic_target(w: float, forecast, price, max_pos: int, form: str = "sigmoid") -> np.ndarray:
    """10.6: target position int(m(w, forecast - price) max_pos), truncated toward 0."""
    return np.trunc(dynamic_size(w, np.asarray(forecast, dtype=float) - np.asarray(price, dtype=float), form)
                    * max_pos).astype(int)


def inverse_cost(forecast, w: float, m, form: str = "sigmoid"):
    """10.6: the price at which the size is m. Sigmoid: betsizing.inverse_price; power: forecast - sgn(m)|m|^(1/w)."""
    if form == "sigmoid":
        return bs.inverse_price(forecast, w, np.asarray(m, dtype=float))
    if form == "power":
        m = np.asarray(m, dtype=float)
        return forecast - np.sign(m) * np.abs(m) ** (1.0 / w)
    raise ValueError(f"unknown form {form!r}")


def limit_cost(pos: int, target: int, forecast: float, w: float, max_pos: int, form: str = "sigmoid") -> float:
    """10.6 breakeven limit for moving from `pos` to `target` units: the mean inverse price over each unit added,
    j = pos + sgn, ..., target (NaN when nothing is traded). For a token, the price is the all-in cost."""
    if target == pos:
        return np.nan
    sgn = 1 if target > pos else -1
    units = np.arange(pos + sgn, target + sgn, sgn)
    return float(np.mean(inverse_cost(forecast, w, units / max_pos, form)))


@dataclass(frozen=True)
class Sizing:
    name: str
    kind: str                           # fixed | kelly | book | dynamic
    average: bool = False               # 10.4 over the spans still open (book, dynamic)
    step: float | None = None           # 10.5 after averaging (book, dynamic)
    null: str = "classes"               # book: 1/K as the book, or "price": the all-in cost
    edge: float | None = None           # fixed: bet when prob - cost > edge; None: when prob > 1/2
    kelly: float = KELLY                # kelly: fraction of full Kelly
    form: str = "sigmoid"               # dynamic: sigmoid or power
    calib: tuple = (0.10, 0.95)         # dynamic: a divergence of 10 cents gives 95% of the maximum position
    max_pos: int = 100                  # dynamic: Q


SIZINGS = (
    Sizing("fixed", "fixed"),
    Sizing("fixed_ev", "fixed", edge=0.0),
    Sizing("kelly", "kelly"),
    Sizing("book", "book"),
    Sizing("book_avg", "book", average=True),
    Sizing("book_avg_step0.1", "book", average=True, step=0.1),
    Sizing("book_price_avg", "book", average=True, null="price"),
    Sizing("dynamic_sigmoid", "dynamic"),
    Sizing("dynamic_power", "dynamic", form="power"),
)


def sizes(df: pd.DataFrame, spec: Sizing) -> tuple[np.ndarray, np.ndarray]:
    """(size >= 0, side) per row. Fixed: 1 per bet taken. Kelly: target position as a share of the bankroll. Book and
    dynamic: target position as a share of the maximum stake, on the side of the (averaged) signed size."""
    prob, side = df["prob"].to_numpy(float), df["side"].to_numpy(float)
    cost = cost_of(df["price"].to_numpy(float))
    if spec.kind == "fixed":
        take = prob > 0.5 if spec.edge is None else prob - cost > spec.edge
        return take.astype(float), side.astype(int)
    if spec.kind == "kelly":
        return spec.kelly * bs.kelly(prob, cost), side.astype(int)
    if spec.kind == "book":
        m = side * bs.size_vs_price(prob, cost) if spec.null == "price" else book_size(prob, side=side)
    elif spec.kind == "dynamic":
        w = calibrate(*spec.calib, form=spec.form)
        m = side * np.maximum(dynamic_target(w, prob, cost, spec.max_pos, spec.form), 0) / spec.max_pos
    else:
        raise ValueError(f"unknown sizing kind {spec.kind!r}")
    if spec.average:
        m = active_average(df["t0"].to_numpy(float), df["t1"].to_numpy(float), m)
    if spec.step:
        m = bs.discrete_size(m, spec.step)
    m = np.where(np.abs(m) < TINY, 0.0, m)
    return np.abs(m), np.sign(m).astype(int)


# ------------------------------------------------------------------------------------------------ dollars

def _check(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    if not np.all(np.isfinite(df["prob"].to_numpy(float))):
        raise ValueError("prob must be finite")
    if df["rule"].nunique() > 1:
        raise ValueError("size one primary rule at a time")
    if len(df) and (df["t1"].to_numpy(np.int64) > CONFIRM_START).any():
        raise ValueError("a row's window ends after the confirmation start")


def simulate(df: pd.DataFrame, spec: Sizing, bankroll: float = BANKROLL, compound: bool = True,
             max_stake_frac: float = MAX_STAKE_FRAC, window_cap_frac: float | None = WINDOW_CAP_FRAC,
             daily_loss_frac: float | None = DAILY_LOSS_FRAC, settle_delay: float = SETTLE_DELAY,
             min_order: float = MIN_ORDER_COST) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bets, windows) for one rule's rows. bets: the rows in decision order with size, target_side, stake, shares,
    fee, pnl, capped (the order, window or cash limit cut the stake the sizing wanted, possibly to nothing),
    limit_price (dynamic: the 10.6 breakeven ask for the units bought). windows: every window with a row, in order,
    with day_tw, bets, stake, pnl, bankroll_before and bankroll_after (around its settlement), ret = pnl /
    bankroll_before, halted (a row of it came after its Taiwan day was halted)."""
    _check(df)
    d = df.sort_values("t0", kind="stable").reset_index(drop=True)
    size, tside = sizes(d, spec)
    price, payoff = d["price"].to_numpy(float), d["payoff"].to_numpy(float)
    start, t0, side = d["start"].astype(int).tolist(), d["t0"].astype(float).tolist(), d["side"].astype(int).tolist()
    cost, prob, pay = cost_of(price).tolist(), d["prob"].astype(float).tolist(), payoff.tolist()
    size_l, tside_l = size.tolist(), tside.tolist()
    w_dyn = calibrate(*spec.calib, form=spec.form) if spec.kind == "dynamic" else None
    n = len(d)
    stake, capped, limit = np.zeros(n), np.zeros(n, bool), np.full(n, np.nan)

    realised, committed = float(bankroll), 0.0
    pending: deque = deque()                     # (visible at, start, pnl, cost) of closed windows
    day_start, day_pnl, halted = {}, {}, set()
    settled: dict[int, tuple[float, float]] = {}
    win_rows: dict[int, dict] = {}
    cur, held, held_side = None, 0.0, 0

    def settle_until(now: float) -> None:
        nonlocal realised, committed
        while pending and pending[0][0] <= now:
            _, s, pnl, c = pending.popleft()
            before, day = realised, hk_day(s)
            day_start.setdefault(day, before)
            realised += pnl
            committed -= c
            day_pnl[day] = day_pnl.get(day, 0.0) + pnl
            if daily_loss_frac is not None and day_pnl[day] <= -daily_loss_frac * day_start[day]:
                halted.add(day)
            settled[s] = (before, realised)

    def close(s) -> None:
        if s is not None:
            r = win_rows[s]
            pending.append((s + WINDOW_SECONDS + settle_delay, s, r["pnl"], r["stake"]))

    for j in range(n):
        settle_until(t0[j])
        s = start[j]
        if s != cur:
            close(cur)
            settle_until(t0[j])
            cur, held, held_side = s, 0.0, 0
            win_rows[s] = {"bets": 0, "stake": 0.0, "pnl": 0.0, "halted": False}
        if hk_day(s) in halted:
            win_rows[s]["halted"] = True
            continue
        if size_l[j] <= 0 or realised <= 0 or tside_l[j] != side[j] \
                or (held_side != 0 and held_side != side[j]):
            continue
        base = realised if compound else bankroll
        max_stake = max_stake_frac * base
        if spec.kind == "fixed":
            want = size_l[j] * max_stake
        elif spec.kind == "kelly":
            want = size_l[j] * base - held
        else:
            want = size_l[j] * max_stake - held
        room = realised - committed
        if window_cap_frac is not None:
            room = min(room, window_cap_frac * base - held)
        got = min(want, max_stake, room)
        capped[j] = want >= min_order and got < want
        if got < min_order:
            continue
        stake[j] = got
        if w_dyn is not None:
            q = spec.max_pos
            lc = limit_cost(int(round(held / max_stake * q)), int(round((held + got) / max_stake * q)),
                            prob[j], w_dyn, q, spec.form)
            limit[j] = price_from_cost(lc) if np.isfinite(lc) else np.nan
        held += got
        held_side = side[j]
        committed += got
        r = win_rows[s]
        r["bets"] += 1
        r["stake"] += got
        r["pnl"] += got / cost[j] * pay[j] - got
    close(cur)
    settle_until(np.inf)

    shares, fee, pnl = bet_pnl(stake, price, payoff)
    bets = d.assign(size=size, target_side=tside, stake=stake, shares=shares, fee=fee, pnl=pnl, capped=capped,
                    limit_price=limit)
    starts = sorted(win_rows)
    windows = pd.DataFrame({
        "start": np.array(starts, dtype=np.int64), "day_tw": [hk_day(s) for s in starts],
        "bets": [win_rows[s]["bets"] for s in starts], "stake": [win_rows[s]["stake"] for s in starts],
        "pnl": [win_rows[s]["pnl"] for s in starts], "bankroll_before": [settled[s][0] for s in starts],
        "bankroll_after": [settled[s][1] for s in starts], "halted": [win_rows[s]["halted"] for s in starts]})
    with np.errstate(divide="ignore", invalid="ignore"):
        windows["ret"] = np.where(windows["bankroll_before"] > 0, windows["pnl"] / windows["bankroll_before"], np.nan)
    return bets, windows


def summarise(bets: pd.DataFrame, windows: pd.DataFrame, bankroll: float = BANKROLL) -> dict:
    """Dollar results of one simulation: counts, stake, P&L, return on stake, mean and Sharpe (ddof 1) per window
    (window returns on the bankroll before each settlement), log growth log(B_end / B_0) (-inf once the bankroll
    reaches 0), maximum drawdown of the bankroll path (fraction of the running peak and dollars)."""
    path = np.concatenate([[bankroll], windows["bankroll_after"].to_numpy(float)])
    staked, pnl = float(windows["stake"].sum()), float(windows["pnl"].sum())
    ruined = bool(np.any(path <= 0))
    growth = -np.inf if ruined else float(np.log(path[-1] / bankroll))
    return {"windows": int(len(windows)), "traded_windows": int((windows["stake"] > 0).sum()),
            "bets": int((bets["stake"] > 0).sum()), "capped_bets": int(bets["capped"].sum()),
            "halted_days": int(windows.loc[windows["halted"], "day_tw"].nunique()),
            "staked": staked, "pnl": pnl, "return_on_stake": pnl / staked if staked > 0 else np.nan,
            "mean_window_stake": float(windows["stake"].mean()) if len(windows) else np.nan,
            "mean_window_pnl": float(windows["pnl"].mean()) if len(windows) else np.nan,
            "sharpe_window": sharpe(windows["ret"].to_numpy(float)),
            "log_growth": growth, "log_growth_per_window": growth / len(windows) if len(windows) else np.nan,
            "max_drawdown": max_drawdown(path) if not ruined else 1.0,
            "max_drawdown_usd": float(np.max(np.maximum.accumulate(path) - path)),
            "final_bankroll": float(path[-1])}


def evaluate(oof: pd.DataFrame, sizings=SIZINGS, **kw) -> dict[str, pd.DataFrame]:
    """Every sizing on the same rows: per rule, per fold (each on a fresh bankroll) and 'all' (every row in time order
    on one bankroll). Keyword arguments go to simulate. Returns summary (one row per rule, fold, sizing), windows and
    bets, each with rule, fold and sizing columns."""
    missing = [c for c in REQUIRED if c not in oof.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    bankroll = kw.get("bankroll", BANKROLL)
    summary, windows, bets = [], [], []
    for rule, rows in oof.groupby("rule", sort=True):
        folds = [(f, rows[rows["fold"] == f]) for f in sorted(rows["fold"].unique())] if "fold" in rows else []
        for fold, sub in folds + [("all", rows)]:
            for spec in sizings:
                b, w = simulate(sub, spec, **kw)
                tag = {"rule": rule, "fold": str(fold), "sizing": spec.name}
                summary.append({**tag, **summarise(b, w, bankroll)})
                windows.append(w.assign(**tag))
                bets.append(b.assign(**tag))
    return {"summary": pd.DataFrame(summary), "windows": pd.concat(windows, ignore_index=True),
            "bets": pd.concat(bets, ignore_index=True)}
