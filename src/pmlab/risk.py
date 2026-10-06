"""Risk overlay: RiskManaged wraps any strategy and vetoes or shrinks the orders it proposes.

The wrapper has the interface the backtester and the live engine use (candidates, decide, and every
other attribute passed through to the wrapped strategy), so a managed variant runs wherever the raw
one runs and the two can be compared on the same windows. It never adds or enlarges an order.

Per decision second (checked only when the wrapped strategy proposes at least one order; the first
failing check names the veto, in this order):
  daily_loss   the Taiwan day (UTC+8) of this window is halted: realised P&L of that day's settled
               windows ≤ −daily_loss_frac × the bankroll before that day's first settled window
  drawdown     paused: equity after the latest visible settlement fell to ≤ (1 − drawdown_frac) × its
               running peak; the pause lasts drawdown_pause_s from the moment that settlement became
               visible, then the peak is reset to the equity at that moment (a new fall is needed)
  cooldown     paused: cooldown_losses consecutive settled windows with P&L < 0 (P&L = 0, e.g. a window
               not traded, neither extends nor breaks the run); pause cooldown_s, then the count restarts
  time_left    300 − t outside [min_time_left_s, max_time_left_s]
  non_finite   fair (Q, "fair" column) or σ ("sigma" column) missing or not finite
  sigma_band   σ (per √s, EWMA of 1 s Binance log returns × vol_mult, floored) outside [sigma_lo, sigma_hi]
  stale_quote  max(ask_age, bid_age) > max_quote_age_s, or either age not finite
  spread       (ask − bid) × 100 in the Up frame > max_spread_cents (rounded to 1e-4 ¢), or not finite
Microstructure gates (adverse selection / toxicity), per second, a missing value never vetoes (no measure yet):
  vpin         volume_vpin > max_vpin        VPIN over the last 10 closed volume bars of this window
  kyle         volume_kyle > max_kyle        Kyle's λ, OLS slope of Δp on share imbalance, last 10 volume bars
  amihud       dollar_amihud > max_amihud    Amihud's λ, mean |Δ logit p| per risk dollar, last 10 dollar bars
  roll         roll > max_roll               Roll spread estimate (price units) from recent trade prices
  cs           tick_cs > max_cs              Corwin–Schultz spread (price units), last 10 tick bars
  micro_spread (ask − bid) × 100 > max_micro_spread_cents (a second, quantile-based spread gate)
  The four bar-window gates (BAR_GATES) apply only once the window has MICRO_BARS closed bars of their kind, the bars
  their estimate is built over (counted from the <kind>_n30 column: Σ n30 at t, t − 30, t − 60, …). Before that the
  measure comes from fewer bars and exceeds its q95 threshold for sample size, not toxicity (docs/afml_sections/ch19.md
  F19.6, research/book_check_part3.md finding 7): the gate neither vetoes nor passes, the other limits decide, and the
  skip is counted (orders_warming, windows_warming) and recorded as `<gate>_warming` with the vetoes. Without the count
  column the gate applies as before.
Per order (after all of the above pass):
  depth_imbalance  live only (historic runs have no book: depth_imb is absent and the gate is skipped):
               an Up buy is vetoed when depth_imb < −max_depth_imb, a Down buy when depth_imb > max_depth_imb,
               depth_imb = (bid depth − ask depth) / (bid depth + ask depth), top 5 levels of the Up book
  window_cost_cap  committed window cost (known fills + open orders, as Position.exposure) + this order's
               cost (shares × (limit + taker fee per share; makers pay no fee)) may not exceed
               max_window_cost_frac × the bankroll passed to decide(); an order that does not fit is
               shrunk to the room left, or dropped when less than MIN_ORDER_COST dollars of room is left.

Causality. Realised P&L reaches the wrapper through on_settle(start, pnl, bankroll_after), which
pmlab.backtest.run calls after every window. A settlement becomes visible to decisions at second
start + 300 + settle_delay (default 5 s, the live engine's wait for the Polymarket clock), never
earlier, whenever the call itself happens; all loss, drawdown and cool-down state is advanced
lazily to the decision second from those visible settlements only. reset() (called by run) clears
all state and counters so nothing leaks between runs (research/reviews.md R5.5).

Corrections. The live engine calls on_settle again for a window when its official outcome replaces the
provisional one. From the moment that call is visible, the stop state is the one the corrected P&L gives
as if it had been known when the window's first result became visible (the frozen backtest's view): a stop
the provisional result set is lifted, one the corrected result sets is in force, and every later
bankroll moves by the difference (research/code_review/live.md engine 1). Decisions already made stand.
The last RETRO to 2 × RETRO settlements are kept for that replay; a correction of an older window only
adjusts that day's P&L and the equity.
"""
import copy
import heapq
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from pmlab import WINDOW_SECONDS as T
from pmlab.fees import fee_per_share

ROOT = Path(__file__).resolve().parents[2]
HK = 8 * 3600
SETTLE_DELAY = 5            # seconds after the window end before its P&L may be used
MIN_ORDER_COST = 1.0        # dollars: a shrunk order smaller than this is dropped instead
RETRO = 144                 # settlements kept for replaying a correction (12 h of windows; the live engine corrects within 3 h)
STATE = ("_day_pnl", "_day_start", "_halted", "_equity", "_peak", "_paused_until", "_cooldown_until", "_streak", "triggers")
# microstructure gate → (limit field, feature column); the bar kinds are fixed here
MICRO = {"vpin": ("max_vpin", "volume_vpin"), "kyle": ("max_kyle", "volume_kyle"), "amihud": ("max_amihud", "dollar_amihud"),
         "roll": ("max_roll", "roll"), "cs": ("max_cs", "tick_cs")}
MICRO_REASONS = (*MICRO, "micro_spread", "depth_imbalance")
MICRO_BARS = 10             # bars in each bar-window estimate (pmlab.micro.bar_micro_features n, pmlab.live.engine.MICRO_BARS)
BAR_GATES = {"vpin": "volume_n30", "kyle": "volume_n30", "amihud": "dollar_n30", "cs": "tick_n30"}   # gate -> its bars' count
WARMING = tuple(f"{g}_warming" for g in BAR_GATES)
REASONS = ("daily_loss", "drawdown", "cooldown", "time_left", "non_finite", "sigma_band", "stale_quote", "spread",
           *MICRO, "micro_spread", "depth_imbalance", "window_cost_cap")


def hk_day(start: int) -> int:
    """Taiwan (UTC+8) calendar day index of a unix second."""
    return (int(start) + HK) // 86400


@dataclass(frozen=True)
class RiskLimits:
    """Every limit is off when None. Units are in the names."""
    max_spread_cents: float | None = None       # Up-frame ask − bid, cents
    max_quote_age_s: float | None = None        # max(ask_age, bid_age), seconds
    require_finite: bool = False                # fair and sigma must be finite
    sigma_lo: float | None = None               # σ per √s, inclusive band
    sigma_hi: float | None = None
    sigma_q_lo: float | None = None             # quantile of training σ → sigma_lo (resolve() before use)
    sigma_q_hi: float | None = None
    min_time_left_s: float | None = None        # 300 − t, seconds, inclusive
    max_time_left_s: float | None = None
    max_window_cost_frac: float | None = None   # share of the bankroll passed to decide()
    daily_loss_frac: float | None = None        # share of the bankroll at the Taiwan day's start
    drawdown_frac: float | None = None          # share of the running equity peak
    drawdown_pause_s: float = 4 * 3600
    cooldown_losses: int | None = None          # consecutive losing windows
    cooldown_s: float = 1800
    max_vpin: float | None = None               # volume_vpin, share imbalance fraction [0, 1]
    max_kyle: float | None = None               # volume_kyle, probability per share
    max_amihud: float | None = None             # dollar_amihud, |Δ logit p| per risk dollar
    max_roll: float | None = None               # roll, probability units
    max_cs: float | None = None                 # tick_cs, probability units
    max_micro_spread_cents: float | None = None
    max_depth_imb: float | None = None          # |depth_imb| against the order side (live book only)
    micro_q: float | None = None                # quantile of training values → every microstructure gate (resolve())


OFF = RiskLimits()

# Fixed before any managed result was computed; nothing here was tuned on a test day. The σ upper
# quantile is resolved per walk-forward day on that day's training windows only. Numbers seen before
# fixing them (training windows of the first test day, HK 2026-09-05…09-10 only): proxy spread median
# 1¢, 97.5% 5¢; max quote age median 2 s, 90% 21 s, 7.6% > 30 s; 27% of σ values sit on the fitted floor.
DEFAULT = RiskLimits(
    max_spread_cents=5.0,       # half-spread 2.5¢ + taker fee up to 1.75¢ exceeds most edge thresholds
    max_quote_age_s=30.0,       # no print (history) or book event (live) on a side for 10% of a window
    require_finite=True,
    sigma_q_hi=0.975,           # top 2.5% σ of the training seconds: jumps, where fills are most adverse;
                                # no lower bound: the fitted σ floor already handles quiet spells (R2.5)
    min_time_left_s=5,          # last 5 s: an order cannot be filled and known before the end
    max_time_left_s=295,        # first 5 s: K not yet known (~2 s) and the last window's P&L not yet visible
    max_window_cost_frac=0.05,  # one binary outcome may cost at most 5% of the bankroll
    daily_loss_frac=0.10,
    drawdown_frac=0.15,
    drawdown_pause_s=4 * 3600,
    cooldown_losses=5,
    cooldown_s=1800,
)
# DEFAULT plus every microstructure gate at the 95% quantile of the training values: each gate alone would veto
# 5% of training seconds (23% for their union on the first test day's training windows). Fixed, not tuned.
MICRO_DEFAULT = replace(DEFAULT, micro_q=0.95)


def sigma_values(windows) -> np.ndarray:
    """Every finite σ (all 300 seconds) of the given windows, float64."""
    vals = np.concatenate([np.asarray(w.cols["sigma"], dtype=float) for w in windows]) if windows else np.array([])
    return vals[np.isfinite(vals)]


def micro_values(windows) -> dict[str, np.ndarray]:
    """Every finite value (all 300 seconds) of each microstructure gate's measure in the given windows, float64;
    depth_imb as |depth_imb|, only where the windows have a book."""
    cols = {**{g: c for g, (_, c) in MICRO.items()}, "micro_spread": None, "depth_imbalance": "depth_imb"}
    out = {}
    for gate, col in cols.items():
        parts = [(np.asarray(w.cols["ask"], dtype=float) - np.asarray(w.cols["bid"], dtype=float)) * 100 if col is None
                 else np.abs(np.asarray(w.cols[col], dtype=float)) if gate == "depth_imbalance" else np.asarray(w.cols[col], dtype=float)
                 for w in windows if col is None or col in w.cols]
        vals = np.concatenate(parts) if parts else np.array([])
        out[gate] = vals[np.isfinite(vals)]
    return out


def resolve(limits: RiskLimits, training_windows) -> RiskLimits:
    """Replace quantile thresholds by numbers computed on training windows only (numpy linear quantile):
    the σ band from sigma_q_lo / sigma_q_hi, and with micro_q every microstructure gate from micro_q of its
    measure (the depth gate stays off when the windows have no depth_imb, as in every historic run)."""
    if limits.sigma_q_lo is not None or limits.sigma_q_hi is not None:
        vals = sigma_values(training_windows)
        if not len(vals):
            raise ValueError("no finite sigma in the training windows")
        q = lambda x, fixed: fixed if x is None else float(np.quantile(vals, x))
        limits = replace(limits, sigma_lo=q(limits.sigma_q_lo, limits.sigma_lo), sigma_hi=q(limits.sigma_q_hi, limits.sigma_hi),
                         sigma_q_lo=None, sigma_q_hi=None)
    if limits.micro_q is not None:
        vals = micro_values(training_windows)
        fields = {**{g: f for g, (f, _) in MICRO.items()}, "micro_spread": "max_micro_spread_cents", "depth_imbalance": "max_depth_imb"}
        limits = replace(limits, micro_q=None, **{f: float(np.quantile(vals[g], limits.micro_q)) if len(vals[g]) else None
                                                  for g, f in fields.items()})
    return limits


def cvar(returns, level: float = 0.05) -> float:
    """Mean of the worst ceil(level × n) finite values (NaN when there are none)."""
    r = np.sort(np.asarray(returns, dtype=float)[np.isfinite(np.asarray(returns, dtype=float))])
    if not len(r):
        return float("nan")
    return float(r[: max(1, math.ceil(round(level * len(r), 9)))].mean())


def max_drawdown_usd(pnl) -> float:
    """Largest fall of the cumulative P&L path (starting at 0) below its running peak, in dollars."""
    path = np.concatenate([[0.0], np.cumsum(np.asarray(pnl, dtype=float))])
    return float(np.max(np.maximum.accumulate(path) - path))


def _now(view, name: str) -> float:
    return view.now(name) if view.has(name) else float("nan")


def bars_known(n30, t: int) -> int:
    """Bars of a window known by second t from its <kind>_n30 column (bars that became known in the last 30 s)."""
    return int(np.asarray(n30[: t + 1], dtype=float)[t::-30].sum())


class RiskManaged:
    """`strategy` with `limits` applied; named `<strategy.name>_managed` unless `name` is given."""

    def __init__(self, strategy, limits: RiskLimits, name: str | None = None, settle_delay: float = SETTLE_DELAY,
                 record_vetoes: bool = False):
        if limits.sigma_q_lo is not None or limits.sigma_q_hi is not None or limits.micro_q is not None:
            raise ValueError("thresholds given as quantiles: resolve them on training windows first (pmlab.risk.resolve)")
        self.inner, self.limits, self.settle_delay = strategy, limits, settle_delay
        self.record_vetoes = record_vetoes          # keep every vetoed order (start, t, reason, order) in self.vetoes
        self.name = name or f"{strategy.name}_managed"
        self.reset()

    def __getattr__(self, attr):                # prob, t_min, t_max, kind, edge ... come from the wrapped strategy
        if attr.startswith("_") or attr in ("inner", "limits", "settle_delay", "record_vetoes"):
            raise AttributeError(attr)
        return getattr(self.inner, attr)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._queue: list = []                  # (visible_at, seq, start, pnl, bankroll_after)
        self._seq = 0
        self._window_pnl: dict[int, float] = {}
        self._day_pnl: dict[int, float] = {}
        self._day_start: dict[int, float] = {}
        self._halted: set[int] = set()
        self._equity = self._peak = None
        self._paused_until = self._cooldown_until = None
        self._streak = 0
        self.triggers: list[dict] = []
        self.last_veto: dict | None = None
        self.vetoes: list[tuple] = []
        self._proposed = self._passed = self._shrunk = 0
        self._vetoed = dict.fromkeys(REASONS, 0)
        self._vetoed_windows: dict[str, set] = {r: set() for r in REASONS}
        self._warming = dict.fromkeys(WARMING, 0)
        self._warming_windows: dict[str, set] = {r: set() for r in WARMING}
        self._skipped: list[str] = []           # bar-window gates skipped at the current decision (_blocked)
        self._log: list[list] = []              # [visible, start, pnl, after, prev pnl or None] of recent settlements applied
        self._first: dict[int, list] = {}       # start -> its first entry in _log
        self._base = self._snapshot()           # the state before _log[0]

    def on_settle(self, start: int, pnl: float, bankroll_after: float, known_at: float | None = None) -> None:
        """Window `start` realised `pnl` (0.0 if it did not trade), leaving `bankroll_after`. A repeated call
        for the same window replaces its P&L (a corrected outcome) without counting it again in the loss run.
        Visible from max(start + 300 + settle_delay, known_at)."""
        visible = max(start + T + self.settle_delay, known_at if known_at is not None else -math.inf)
        heapq.heappush(self._queue, (visible, self._seq, int(start), float(pnl), float(bankroll_after)))
        self._seq += 1

    def _snapshot(self) -> dict:
        return copy.deepcopy({k: getattr(self, k) for k in STATE})

    def _replay(self, entries) -> None:
        """Apply kept settlements again from the state set before the first of them, resuming a pause where _advance did."""
        for visible, start, pnl, after, prev in entries:
            if self._paused_until is not None and self._paused_until < visible:
                self._paused_until, self._peak = None, self._equity
            self._update(visible, start, pnl, after, prev)

    def _apply(self, visible, start, pnl, after) -> None:
        """A settlement becomes visible. A repeated one (a correction) of a kept window replaces its P&L, moves the bankroll
        after it and after every later kept settlement by the difference, and replays them (module docstring)."""
        entry = self._first.get(start)
        if entry is not None:
            delta = pnl - entry[2]
            if delta == 0:
                return
            entry[2] = pnl
            for e in self._log[next(i for i, e in enumerate(self._log) if e is entry):]:
                e[3] += delta
            for k, v in copy.deepcopy(self._base).items():
                setattr(self, k, v)
            self._replay(self._log)
            return
        prev = self._window_pnl.get(start)
        self._log.append([visible, start, pnl, after, prev])
        if prev is None:
            self._first[start] = self._log[-1]
        self._update(visible, start, pnl, after, prev)
        if len(self._log) > 2 * RETRO:                  # keep the newest RETRO: move the base past the oldest
            old, self._log = self._log[:RETRO], self._log[RETRO:]
            for k, v in self._base.items():
                setattr(self, k, v)
            self._replay(old)
            self._base = self._snapshot()
            for e in old:
                if self._first.get(e[1]) is e:
                    del self._first[e[1]]
            self._replay(self._log)

    def _update(self, visible, start, pnl, after, prev) -> None:
        """Advance day P&L, equity peak and loss run by one settlement; prev: the window's P&L already applied (None: first)."""
        L, d = self.limits, hk_day(start)
        first = prev is None
        if first:
            self._day_start.setdefault(d, after - pnl)
        self._day_pnl[d] = self._day_pnl.get(d, 0.0) + pnl - (0.0 if first else prev)
        self._window_pnl[start] = pnl
        self._peak = max(after - pnl if self._peak is None else self._peak, after)
        self._equity = after
        if L.daily_loss_frac is not None and d not in self._halted and \
                self._day_pnl[d] <= -L.daily_loss_frac * self._day_start[d]:
            self._halted.add(d)
            self.triggers.append({"kind": "daily_loss", "visible_at": visible, "window": start, "hk_day": d})
        if L.drawdown_frac is not None and self._paused_until is None and after <= (1 - L.drawdown_frac) * self._peak:
            self._paused_until = visible + L.drawdown_pause_s
            self.triggers.append({"kind": "drawdown", "visible_at": visible, "window": start, "until": self._paused_until})
        if L.cooldown_losses is not None and first:
            self._streak = self._streak + 1 if pnl < 0 else 0 if pnl > 0 else self._streak
            if self._streak >= L.cooldown_losses:
                self._cooldown_until, self._streak = visible + L.cooldown_s, 0
                self.triggers.append({"kind": "cooldown", "visible_at": visible, "window": start,
                                      "until": self._cooldown_until})

    def _advance(self, now: float) -> None:
        """Apply every settlement visible at `now` and every drawdown resume due by `now`, in time order
        (a settlement visible at the resume second is applied before the peak is reset)."""
        while True:
            due = bool(self._queue) and self._queue[0][0] <= now
            if self._paused_until is not None and self._paused_until <= now and \
                    (not due or self._paused_until < self._queue[0][0]):
                self._paused_until, self._peak = None, self._equity
                continue
            if not due:
                return
            visible, _, start, pnl, after = heapq.heappop(self._queue)
            self._apply(visible, start, pnl, after)

    def _blocked(self, view, now: float) -> str | None:
        L = self.limits
        if hk_day(view.start) in self._halted:
            return "daily_loss"
        if self._paused_until is not None and now < self._paused_until:
            return "drawdown"
        if self._cooldown_until is not None and now < self._cooldown_until:
            return "cooldown"
        left = T - view.t
        if (L.min_time_left_s is not None and left < L.min_time_left_s) or \
                (L.max_time_left_s is not None and left > L.max_time_left_s):
            return "time_left"
        sigma = _now(view, "sigma")
        if L.require_finite and not (np.isfinite(_now(view, "fair")) and np.isfinite(sigma)):
            return "non_finite"
        if (L.sigma_lo is not None or L.sigma_hi is not None) and not (
                np.isfinite(sigma) and (L.sigma_lo is None or sigma >= L.sigma_lo) and (L.sigma_hi is None or sigma <= L.sigma_hi)):
            return "sigma_band"
        if L.max_quote_age_s is not None:
            ages = (_now(view, "ask_age"), _now(view, "bid_age"))
            if not all(np.isfinite(ages)) or max(ages) > L.max_quote_age_s:
                return "stale_quote"
        if L.max_spread_cents is not None:
            spread = round((_now(view, "ask") - _now(view, "bid")) * 100, 4)
            if not np.isfinite(spread) or spread > L.max_spread_cents:
                return "spread"
        for gate, (field, col) in MICRO.items():            # a measure not yet available never vetoes
            if (limit := getattr(L, field)) is None:
                continue
            if gate in BAR_GATES and view.has(BAR_GATES[gate]) and bars_known(view[BAR_GATES[gate]], view.t) < MICRO_BARS:
                self._skipped.append(gate)                  # built over fewer bars than the estimate: not a verdict
                continue
            if _now(view, col) > limit:
                return gate
        if L.max_micro_spread_cents is not None and round((_now(view, "ask") - _now(view, "bid")) * 100, 4) > L.max_micro_spread_cents:
            return "micro_spread"
        return None

    # ------------------------------------------------------------------ strategy interface
    def candidates(self, f):
        return self.inner.candidates(f)

    def decide(self, view, pos, bankroll: float) -> list:
        now = view.start + view.t
        self._advance(now)
        orders = self.inner.decide(view, pos, bankroll)
        if not orders:
            return []
        self._proposed += len(orders)
        self._skipped = []
        reason = self._blocked(view, now)
        for gate in self._skipped:
            self._warming[f"{gate}_warming"] += len(orders)
            self._warming_windows[f"{gate}_warming"].add(view.start)
            if self.record_vetoes:
                self.vetoes.extend((view.start, view.t, f"{gate}_warming", o) for o in orders)
        if reason:
            self._veto(reason, orders, view)
            return []
        if self.limits.max_depth_imb is not None and np.isfinite(depth := _now(view, "depth_imb")):
            against = lambda o: depth < -self.limits.max_depth_imb if o.side == "up" else depth > self.limits.max_depth_imb
            if blocked := [o for o in orders if against(o)]:
                self._veto("depth_imbalance", blocked, view)
                orders = [o for o in orders if not against(o)]
        kept = orders
        if self.limits.max_window_cost_frac is not None:
            room, kept = self.limits.max_window_cost_frac * bankroll - pos.exposure(view.t)["cost"], []
            for o in orders:
                unit = o.limit + (0.0 if o.maker else fee_per_share(o.limit))
                if o.shares * unit <= room + 1e-9:
                    kept.append(o)
                    room -= o.shares * unit
                elif room >= MIN_ORDER_COST and unit > 0:
                    kept.append(replace(o, shares=room / unit))
                    room = 0.0
                    self._shrunk += 1
                else:
                    self._veto("window_cost_cap", [o], view)
        self._passed += sum(1 for o in kept if any(o is x for x in orders))
        return [self._retag(o) for o in kept]

    def _retag(self, order):
        """Fills are booked to the strategy named before ':' in the tag (live engine), so tag with our name."""
        _, sep, rest = order.tag.partition(":")
        return replace(order, tag=self.name + sep + rest)

    def _veto(self, reason: str, orders: list, view) -> None:
        self._vetoed[reason] += len(orders)
        self._vetoed_windows[reason].add(view.start)
        self.last_veto = {"reason": reason, "window": view.start, "t": view.t}
        if self.record_vetoes:
            self.vetoes.extend((view.start, view.t, reason, o) for o in orders)

    def summary(self) -> dict:
        """Counts since the last reset: orders proposed = passed + shrunk + vetoed; windows with a veto per reason; orders and
        windows for which a bar-window gate was skipped for too few bars (not vetoes: the orders went on to the other checks)."""
        return {"name": self.name, "limits": asdict(self.limits), "settle_delay_s": self.settle_delay,
                "orders_proposed": self._proposed, "orders_passed": self._passed, "orders_shrunk": self._shrunk,
                "orders_vetoed": dict(self._vetoed),
                "windows_vetoed": {r: len(s) for r, s in self._vetoed_windows.items()},
                "orders_warming": dict(self._warming), "windows_warming": {r: len(s) for r, s in self._warming_windows.items()},
                "triggers": list(self.triggers)}


def build_managed(name: str, params: dict, limits: RiskLimits, suffix: str = "_managed") -> RiskManaged:
    """The managed variant `<name><suffix>` of pmlab.strategies.build(name, params); `limits` must have its
    quantile thresholds resolved to numbers (resolve(), or live_limits() for the live engine)."""
    from pmlab.strategies import build
    return RiskManaged(build(name, params), limits, name=f"{name}{suffix}")


def live_variants(strategies: dict, live_fair: dict, micro: bool = True, limits: dict | None = None, **where) -> dict:
    """Recommended live set: for every strategy the engine runs, `<name>_managed` (live_limits) and, with micro,
    `<name>_managed_micro` (live_limits(micro=True)); each wraps its own copy of the raw strategy. `limits`: frozen
    {"managed": fields, "managed_micro": fields} (research/strategy_params.json live_limits, written by
    scripts/fit_live_models.py), used as they are so later walk-forward runs or recordings cannot change them."""
    import copy
    out = {}
    get = lambda kind, **kw: RiskLimits(**limits[kind]) if limits else live_limits(live_fair, **kw, **where)
    for suffix, lim in (("_managed", get("managed")),) + ((("_managed_micro", get("managed_micro", micro=True)),) if micro else ()):
        out.update({f"{name}{suffix}": RiskManaged(copy.deepcopy(s), lim, name=f"{name}{suffix}") for name, s in strategies.items()})
    return out


def live_feature_rows(live: Path, columns: list[str], start_from: float, start_before: float):
    """Recorded live feature rows (both storage layouts, duplicates dropped) of windows starting in [start_from, start_before)."""
    import pandas as pd
    import datetime as dt
    import logging
    def overlaps(p) -> bool:                      # hour files named hour=YYYY-MM-DDTHH; an unparseable name is read anyway
        try:
            h = dt.datetime.strptime(p.parent.name, "hour=%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            return True
        return start_from - 3600 <= h < start_before + 3600
    parts = [p for p in sorted(live.glob("*/features/part-*.parquet")) if int(p.stem.split("-")[1]) / 1000 >= start_from]
    hours = [p for p in sorted(live.glob("features/hour=*/*.parquet")) if overlaps(p)]
    frames = []
    for f in parts + hours:                          # only files that can hold the range; unreadable ones skipped and logged
        try:
            d = pd.read_parquet(f, columns=None)
        except Exception as e:
            logging.getLogger(__name__).warning("live_feature_rows skipped unreadable %s: %r", f, e)
            continue
        frames.append(d[[c for c in ["window", "second", *columns] if c in d.columns]])
    rows = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(columns=["window"])
    return rows[(rows["window"] >= start_from) & (rows["window"] < start_before)]


def live_limits(live_fair: dict, walkforward: Path = ROOT / "research" / "walkforward.json", micro: bool = False,
                live: Path = ROOT / "data" / "live", sigma_band: dict | None = None) -> RiskLimits:
    """Limits for the live engine. The limits of the latest walk-forward test day. The σ band: with `sigma_band`
    (research/strategy_params.json live_risk, written by scripts/fit_live_models.py: the same quantiles of σ computed
    with the live pricing layer on the windows it was fitted on) its numbers; otherwise the walk-forward day's band
    rescaled to the live pricing layer: above the floor σ = raw EWMA σ × vol_mult, so with the same EWMA half-life a
    quantile scales by live vol_mult / walk-forward vol_mult (different half-lives raise). With micro=True the microstructure gates are added, each at the walk-forward micro quantile of
    the same measure in the live engine's own recorded feature rows (live measures come from the real book and
    live bar sizes, so historic thresholds do not transfer), using only windows that start after the stale-book
    fix (pmlab.performance.LIVE_VALID_FROM) and before the confirmation data (pmlab.evaluation.CONFIRM_START).
    The published tree has no pmlab.performance (so no LIVE_VALID_FROM), hence micro=True raises NotImplementedError."""
    if micro:                                        # fail before reading anything, not with ModuleNotFoundError halfway in
        raise NotImplementedError("live_limits(micro=True) needs pmlab.performance.LIVE_VALID_FROM, which is not in "
                                  "this published tree; use micro=False or the frozen live_managed_micro policy")
    out = json.loads(walkforward.read_text())
    limits = RiskLimits(**out["risk"]["limits_by_day"][out["test_days"][-1]])
    wf = out["fair_params"]
    if sigma_band is not None:
        limits = replace(limits, sigma_lo=sigma_band.get("sigma_lo"), sigma_hi=sigma_band.get("sigma_hi"))
    else:
        if wf["halflife"] != live_fair["halflife"]:
            raise ValueError("σ EWMA half-lives differ: the walk-forward σ band does not transfer")
        k = live_fair["vol_mult"] / wf["vol_mult"]
        scale = lambda x: None if x is None else max(x * k, live_fair["sigma_floor"])
        limits = replace(limits, sigma_lo=scale(limits.sigma_lo), sigma_hi=scale(limits.sigma_hi))
    return limits
