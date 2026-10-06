"""Streaming bar builders: one Trade in, at most one finished Bar out, O(1) per trade.

Batch, replay and live all drive a builder the same way:

    b.new_window(start)
    for trade in window_trades:          # only trades with 0 <= t < 300
        bar = b.update(trade)            # a Bar when this trade completes one
    bar = b.finish()                     # time bars can complete at the window end

new_window() discards an unfinished bar: a partial bar is not comparable with full ones.
Adaptive builders (imbalance, run) keep their learned expectations across windows; only
the accumulation restarts, because each window is a fresh token whose price path does not
continue from the last one.
"""
from dataclasses import dataclass, fields

from pmlab.events import Trade


@dataclass(slots=True)
class Bar:
    kind: str
    window: int
    index: int
    ts_open: float
    ts_close: float
    open: float
    high: float
    low: float
    close: float
    vwap: float
    n_ticks: int
    shares: float
    risk_usd: float
    usdc: float
    buy_shares: float
    sell_shares: float
    tick_imbalance: int
    share_imbalance: float
    risk_imbalance: float
    threshold: float


BAR_COLUMNS = [f.name for f in fields(Bar)]


class BarBuilder:
    kind = ""

    def __init__(self):
        self.window: int | None = None
        self.index = 0
        self._reset()

    def _reset(self):
        self.n = 0
        self.ts_open = self.o = self.h = self.l = self.c = 0.0
        self.pq = self.q = self.risk = self.usdc = self.buy = self.sell = 0.0
        self.ti = 0
        self.qi = self.ri = 0.0

    def new_window(self, window: int):
        self.window = window
        self.index = 0
        self._reset()

    def _add(self, tr: Trade):
        if tr.window != self.window:
            raise ValueError(f"trade for window {tr.window} fed to builder on window {self.window}")
        p = tr.p_up
        if self.n == 0:
            self.ts_open, self.o, self.h, self.l = tr.ts, p, p, p
            self._open()
        elif p > self.h:
            self.h = p
        elif p < self.l:
            self.l = p
        self.c = p
        self.n += 1
        self.pq += p * tr.shares
        self.q += tr.shares
        self.risk += tr.risk_usd
        self.usdc += tr.usdc
        if tr.sign > 0:
            self.buy += tr.shares
        else:
            self.sell += tr.shares
        self.ti += tr.sign
        self.qi += tr.sign * tr.shares
        self.ri += tr.sign * tr.risk_usd

    def _emit(self, ts_close: float) -> Bar:
        bar = Bar(self.kind, self.window, self.index, self.ts_open, ts_close,
                  self.o, self.h, self.l, self.c, self.pq / self.q if self.q else self.c,
                  self.n, self.q, self.risk, self.usdc, self.buy, self.sell,
                  self.ti, self.qi, self.ri, self.threshold())
        self.index += 1
        self._close(bar)
        self._reset()
        return bar

    # hooks
    def _open(self):
        """Called when a bar receives its first trade."""

    def _close(self, bar: Bar):
        """Called after a bar is emitted, before the accumulators reset."""

    def threshold(self) -> float:
        raise NotImplementedError

    def update(self, tr: Trade) -> Bar | None:
        raise NotImplementedError

    def finish(self) -> Bar | None:
        return None
