"""Event-time backtest of hold-to-expiry strategies on BTC 5-minute windows.

Adapted processes (L1-preliminaries slide 15): a position taken at t must be F(t)-measurable.
That is enforced structurally, not by convention:
- strategies see a View that slices every feature array to [0, t] and never the outcome; candidates() (the
  seconds worth asking) gets the feature columns without the outcome or the tape, and the seconds are asked in
  time order whatever order it returns them in;
- fills are simulated only from prints stamped at or after t + latency;
- the exposure a strategy sees counts fills already known at t plus pending orders at full size.
tests/test_backtest.py perturbs everything after t and checks no decision at t changes.

The data-api stamps trades with block time, 0.9–4.3 s (median 2.7 s) after the real match
(measured against the websocket). A print stamped just after a decision may have matched
before it, at a price the strategy could never have hit. Fills therefore only use prints
stamped at or after t + latency + print_delay, with print_delay above the worst delay seen.
Live paper trading uses real match times and sets print_delay to 0.

Historic fills (no order book history): a taker buy of Up fills against buy-aggressor prints,
a taker buy of Down against sell-aggressor prints, at the print price, capped at the limit and
at `participation` × the print's size. A maker bid fills only when a print trades strictly
through its price. Takers pay pmlab.fees; makers pay nothing (rebates ignored).
Payoff per share at expiry: Up pays y, Down pays 1 − y.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS as T
from pmlab.features import WindowFeatures
from pmlab.fees import fee_per_share, maker_rebate, taker_fee
from pmlab.metrics import max_drawdown, summary


@dataclass
class Config:
    latency: int = 1          # seconds from decision to earliest fill
    print_delay: int = 5      # historic print timestamps trail the real match by up to ~4.3 s
    participation: float = 0.5
    bankroll: float = 1000.0
    compound: bool = True     # size from the running bankroll (Kelly) or the starting one
    ruin: float = 0.1         # stop trading once the bankroll falls below this share of the start
    maker_rebate: float = 0.0  # share of the generated taker fee paid back to makers (0.2 if eligible)


@dataclass
class Order:
    side: str                 # "up" or "down" (the token bought)
    shares: float
    limit: float              # worst token price accepted
    maker: bool = False
    ttl: int = 3              # seconds a taker may wait for prints / a maker bid rests
    tag: str = ""
    quote: float = float("nan")   # token quote at decision (ask for a taker; own limit for a maker)
    mid: float = float("nan")     # token mid at decision — set by the engine, not by strategies
    p_up: float = float("nan")    # the strategy's belief P(Up) at decision (Q when it has none)
    q_up: float = float("nan")    # the fair price Q(Up) at decision
    crossed: bool = False         # the quotes behind `mid` were crossed (bid > ask): no spread statistic from them


@dataclass
class Fill:
    start: int
    t: int                    # decision second
    ts: float                 # print time that filled it
    side: str
    shares: float
    price: float              # token price
    fee: float
    maker: bool
    tag: str
    quote: float = float("nan")
    mid: float = float("nan")
    p_up: float = float("nan")
    q_up: float = float("nan")
    mid_fill: float = float("nan")    # token mid at the fill second (backtest: print proxies), for maker spread and markouts
    mid_later: float = float("nan")   # token mid MARKOUT_SECONDS after the fill second (last second of the window at most)
    crossed_fill: bool = False        # the quotes behind mid_fill were crossed
    crossed: bool = False             # the decision quotes behind quote and mid were crossed

    def costs(self) -> dict:
        """Execution cost against the mid at decision, split into its parts (per fill, in $).

        half_spread: quote − mid (a taker pays it; a maker's is negative: it earns it);
        slippage: fill price − quote (delay and walking the book); fee: taker fee.
        Holding to settlement means there is no exit trade: redemption at $1 is free."""
        q = self.shares
        half = (self.quote - self.mid) * q
        return {"half_spread": half, "slippage": (self.price - self.quote) * q, "fee": self.fee,
                "total_vs_mid": (self.price - self.mid) * q + self.fee}


def decision_prices(f, t: int, order: "Order", belief: str = "fair") -> None:
    """Stamp the order with the token quote and mid the strategy saw at t (for cost attribution)
    and with Q and the strategy's belief P at t (for the P/Q split of its P&L, pmlab.pq)."""
    order.q_up = float(f.cols["fair"][t])
    order.p_up = float(f.cols[belief][t]) if belief in f.cols else order.q_up
    ask, bid = f.cols["ask"][t], f.cols["bid"][t]
    mid_up = (ask + bid) / 2
    order.mid = mid_up if order.side == "up" else 1 - mid_up
    order.quote = order.limit if order.maker else (ask if order.side == "up" else 1 - bid)
    order.crossed = bool(bid > ask)


class View:
    """What a strategy may know at second t: every column up to and including t."""

    def __init__(self, f: WindowFeatures, t: int):
        self._f, self.t, self.start = f, t, f.start

    def __getitem__(self, name: str) -> np.ndarray:
        return self._f.cols[name][: self.t + 1]

    def now(self, name: str) -> float:
        return float(self._f.cols[name][self.t])

    def has(self, name: str) -> bool:
        return name in self._f.cols


@dataclass
class Position:
    fills: list = field(default_factory=list)
    pending: list = field(default_factory=list)       # (expires_at_second, Order)
    orders: list = field(default_factory=list)        # (decision second, Order, shares filled): implementation shortfall

    def exposure(self, t: int) -> dict:
        """Shares and cost per side known at second t, counting live orders as filled."""
        out = {"up": 0.0, "down": 0.0, "cost": 0.0}
        for f in self.fills:
            if f.ts < f.start + t:
                out[f.side] += f.shares
                out["cost"] += f.shares * f.price + f.fee
        for expires, o in self.pending:
            if expires > t:
                out[o.side] += o.shares
                out["cost"] += o.shares * (o.limit + (0.0 if o.maker else fee_per_share(o.limit)))
        return out


def token_prints(tape: pd.DataFrame, side: str, maker: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ts, token price, shares) of prints that could fill an order buying `side`.

    Taker buying Up lifts asks -> buy-aggressor prints. A maker bid for Up is hit by
    sell-aggressor prints. Down mirrors both, priced at 1 - p_up.
    """
    buy_aggr = tape["sign"].to_numpy() > 0
    want = buy_aggr if (side == "up") != maker else ~buy_aggr
    ts = tape["ts"].to_numpy(dtype=float)[want]
    p = tape["p_up"].to_numpy(dtype=float)[want]
    return ts, (p if side == "up" else 1.0 - p), tape["shares"].to_numpy(dtype=float)[want]


def simulate_fill(tape: pd.DataFrame, start: int, t: int, order: Order, cfg: Config,
                  prints: dict | None = None) -> list[Fill]:
    """`prints` caches token_prints per (side, maker) across calls on the same tape, together with
    how much of each print this window's earlier orders already took: participation is a cap per
    print, shared by all orders, not per order (research/reviews.md R2.2)."""
    key = (order.side, order.maker)
    if prints is None:
        prints = {}
    if key not in prints:
        ts_, price_, size_ = token_prints(tape, order.side, order.maker)
        prints[key] = (ts_, price_, size_, np.zeros(len(ts_)))
    ts, price, size, used = prints[key]
    lo = start + t + cfg.latency + cfg.print_delay
    hi = lo + order.ttl
    i = np.searchsorted(ts, lo, side="left")
    fills, left = [], order.shares
    while left > 1e-9 and i < len(ts) and ts[i] < hi:
        through = price[i] < order.limit if order.maker else price[i] <= order.limit
        available = cfg.participation * size[i] - used[i]
        if through and available > 1e-12:
            q = min(left, available)
            used[i] += q
            px = order.limit if order.maker else price[i]
            fee = -maker_rebate(q, px, cfg.maker_rebate) if order.maker else taker_fee(q, px)
            fills.append(Fill(start, t, ts[i], order.side, q, px, fee, order.maker, order.tag, order.quote, order.mid,
                              order.p_up, order.q_up, crossed=order.crossed))
            left -= q
        i += 1
    return fills


SETTLE_VISIBLE = 5      # seconds after a window ends before its P&L may size the next orders (= risk.SETTLE_DELAY)


def run_window(strategy, f: WindowFeatures, bankroll: float, cfg: Config, early_bankroll: float | None = None) -> Position:
    """`early_bankroll` sizes decisions before second SETTLE_VISIBLE: the bankroll without the P&L of the window that
    ended at this one's start, whose outcome is not yet known then (R6)."""
    pos = Position()
    prints = f.meta.setdefault("prints", {})
    for entry in prints.values():      # the print cache outlives this run; the liquidity this run took does not
        entry[3][:] = 0.0
    columns_only = WindowFeatures(f.start, f.cols, None, None, {})   # candidates() never sees the outcome or the tape
    for t in sorted({int(t) for t in strategy.candidates(columns_only) if 0 <= int(t) < T}):
        view = View(f, t)
        for order in strategy.decide(view, pos, bankroll if early_bankroll is None or t >= SETTLE_VISIBLE else early_bankroll):
            if order.shares <= 0:
                continue
            decision_prices(f, t, order, getattr(strategy, "prob", "fair"))
            # an order counts as open until its last possible fill is known (R2.1)
            pos.pending.append((t + cfg.latency + cfg.print_delay + order.ttl, order))
            fills = simulate_fill(f.tape, f.start, t, order, cfg, prints)
            for x in fills:                                  # ex-post marks for cost analysis only (never seen by strategies)
                x.mid_fill, x.crossed_fill = token_mid(f, x.side, int(x.ts - f.start))
                x.mid_later = token_mid(f, x.side, int(x.ts - f.start) + MARKOUT_SECONDS)[0]
            pos.fills.extend(fills)
            pos.orders.append((t, order, sum(x.shares for x in fills)))
    return pos


MARKOUT_SECONDS = 30
SHORTFALL_COLS = ("ordered_shares", "unfilled_shares", "is_paper_pnl", "is_arrival_cost", "is_opportunity_cost",
                  "implementation_shortfall", "half_spread_uncrossed", "spread_fills_crossed", "maker_effective_spread",
                  "maker_markout_30s", "maker_markout_settle", "maker_spread_shares")


def token_mid(f, side: str, second: int) -> tuple[float, bool]:
    """(token mid, quotes crossed) at a window second clipped to [0, T − 1], from the window's ask/bid columns."""
    k = min(max(second, 0), T - 1)
    ask, bid = float(f.cols["ask"][k]), float(f.cols["bid"][k])
    mid = (ask + bid) / 2
    return (mid if side == "up" else 1 - mid), bool(bid > ask)


def shortfall(pos: Position, y: float) -> dict:
    """Transaction-cost analysis of one window (audit R6 finding 10), in $, all against the token mid (the backtest's
    print proxies; the live book in pmlab.performance):
    is_paper_pnl = Σ orders ordered shares × (y_side − mid at decision): the orders filled in full at the arrival mid;
    is_arrival_cost = Σ fills shares × (fill price − mid at decision); is_opportunity_cost = Σ orders unfilled shares ×
    (y_side − mid at decision): what the shares that did not fill would have made; implementation_shortfall =
    is_arrival_cost + fees + is_opportunity_cost = is_paper_pnl − pnl (Perold 1988). half_spread_uncrossed =
    Σ fills shares × (quote − mid) over fills whose decision quotes were not crossed (spread_fills_crossed counts the
    rest); makers, over fills with uncrossed quotes at the fill second: maker_effective_spread = Σ shares × (mid at fill
    − limit), maker_markout_30s = Σ shares × (mid 30 s later − mid at fill), maker_markout_settle = Σ shares ×
    (y_side − mid at fill), maker_spread_shares their shares; effective spread + settlement markout = their P&L."""
    ys = lambda side: y if side == "up" else 1 - y
    ok = lambda v: v == v
    orders = [(o, filled) for _, o, filled in pos.orders if ok(o.mid)]
    out = {"ordered_shares": float(sum(o.shares for _, o, _ in pos.orders)),
           "unfilled_shares": float(sum(max(o.shares - filled, 0.0) for _, o, filled in pos.orders)),
           "is_paper_pnl": float(sum(o.shares * (ys(o.side) - o.mid) for o, _ in orders)),
           "is_arrival_cost": float(sum(x.shares * (x.price - x.mid) for x in pos.fills if ok(x.mid))),
           "is_opportunity_cost": float(sum(max(o.shares - filled, 0.0) * (ys(o.side) - o.mid) for o, filled in orders))}
    out["implementation_shortfall"] = out["is_arrival_cost"] + float(sum(x.fee for x in pos.fills)) + out["is_opportunity_cost"]
    clean = [x for x in pos.fills if ok(x.mid) and ok(x.quote) and not x.crossed]
    out["half_spread_uncrossed"] = float(sum(x.shares * (x.quote - x.mid) for x in clean))
    out["spread_fills_crossed"] = int(sum(1 for x in pos.fills if x.crossed))
    makers = [x for x in pos.fills if x.maker and ok(x.mid_fill) and not x.crossed_fill]
    out["maker_effective_spread"] = float(sum(x.shares * (x.mid_fill - x.price) for x in makers))
    out["maker_markout_30s"] = float(sum(x.shares * (x.mid_later - x.mid_fill) for x in makers if ok(x.mid_later)))
    out["maker_markout_settle"] = float(sum(x.shares * (ys(x.side) - x.mid_fill) for x in makers))
    out["maker_spread_shares"] = float(sum(x.shares for x in makers))
    return out


def settle(pos: Position, y: float) -> dict:
    up = sum(x.shares for x in pos.fills if x.side == "up")
    down = sum(x.shares for x in pos.fills if x.side == "down")
    cost = sum(x.shares * x.price for x in pos.fills)
    fees = sum(x.fee for x in pos.fills)
    payoff = up * y + down * (1 - y)
    parts = [f.costs() for f in pos.fills]
    add = lambda k: float(np.nansum([c[k] for c in parts])) if parts else 0.0
    return {"up": up, "down": down, "cost": cost, "fees": fees, "payoff": payoff,
            "pnl": payoff - cost - fees, "n_fills": len(pos.fills),
            "half_spread": add("half_spread"), "slippage": add("slippage"),
            "maker_fills": sum(f.maker for f in pos.fills), **shortfall(pos, y)}


def run(strategy, windows: list[WindowFeatures], cfg: Config = Config(),
        untraded: list | None = None) -> tuple[pd.DataFrame, list[Fill]]:
    """Windows in time order; one row per window that traded. `untraded`, when given, receives {"start", **shortfall}
    for every window with orders but no fill: their opportunity cost is part of implementation shortfall (audit R6).

    Optional hooks, for strategies that keep state across windows (pmlab.risk): `reset()` is called once
    before the first window, so nothing carries over from an earlier run; `on_settle(start, pnl, bankroll_after)`
    after each window has been run and settled (pnl 0.0 when it did not trade), in time order, before the
    next window. The call carries no clock: the strategy itself must not use a settlement before the window
    has ended (start + 300 s)."""
    bankroll, rows, all_fills = cfg.bankroll, [], []
    reset, on_settle = getattr(strategy, "reset", None), getattr(strategy, "on_settle", None)
    if reset is not None:
        reset()
    last_end, before_last = None, bankroll       # the previous traded window's end and the bankroll before its P&L
    for f in sorted(windows, key=lambda w: w.start):
        if bankroll < cfg.ruin * cfg.bankroll:
            break
        early = before_last if cfg.compound and last_end == f.start else None
        pos = run_window(strategy, f, bankroll if cfg.compound else cfg.bankroll, cfg, early)
        if not pos.fills:
            if untraded is not None and pos.orders:
                untraded.append({"start": f.start, **shortfall(pos, f.y)})
            if on_settle is not None:
                on_settle(f.start, 0.0, bankroll)
            continue
        s = settle(pos, f.y)
        rows.append({"start": f.start, "bankroll": bankroll, **s, "ret": s["pnl"] / bankroll})
        last_end, before_last = f.start + T, bankroll
        bankroll += s["pnl"]
        all_fills.extend(pos.fills)
        if on_settle is not None:
            on_settle(f.start, s["pnl"], bankroll)
    return pd.DataFrame(rows), all_fills


def report(results: pd.DataFrame, n_windows: int, cfg: Config = Config()) -> dict:
    """Per-traded-window statistics plus totals. Sharpe is per traded window, not annualised."""
    if results.empty:
        return {"traded_windows": 0, "windows": n_windows, "pnl": 0.0, "fees": 0.0, "sharpe": np.nan,
                "log_growth": 0.0, "max_drawdown": 0.0, "psr": np.nan, "hit_rate": np.nan}
    s = summary(results["ret"].to_numpy())
    equity = np.concatenate([[cfg.bankroll], cfg.bankroll + results["pnl"].cumsum().to_numpy()])
    return {"traded_windows": len(results), "windows": n_windows, "pnl": float(results["pnl"].sum()),
            "fees": float(results["fees"].sum()), "turnover": float(results["cost"].sum()),
            "fills": int(results["n_fills"].sum()), "sharpe": s["sharpe"], "psr": s["psr"],
            "skew": s["skew"], "kurtosis": s["kurtosis"], "log_growth": s["log_growth"],
            "hit_rate": s["hit_rate"], "max_drawdown": max_drawdown(equity),
            "ruined": bool(equity[-1] < cfg.ruin * cfg.bankroll),
            "return": float(equity[-1] / cfg.bankroll - 1)}
