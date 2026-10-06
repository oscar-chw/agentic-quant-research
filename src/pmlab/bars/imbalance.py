"""Imbalance bars (AFML 2.3.2(1)–2.3.2(2)): sample when order flow is more one-sided than expected.

Within a bar θ_T = Σ b_t v_t, with v_t = 1 (tick), shares (volume) or risk dollars
(dollar). The bar closes when |θ_T| ≥ E0[T] · |E0[b v]|, where E0[T] is an EWMA of past
bar lengths in ticks and E0[b v] an EWMA of per-trade signed flow.

Two guards the book's formula needs in practice:
- Balanced flow drives |E0[b v]| to 0, so the threshold collapses and every trade becomes
  a bar. The threshold is floored at floor_sd · sd(b v) · √E0[T], the size a random walk
  of E0[T] trades reaches in one standard deviation.
- E0[T] learned from the bars' own lengths (anchor="bars", the book's rule) is a feedback
  loop with no restoring force: any bias in how early bars close compounds bar after bar
  (see the run-bar tests: a 3% bias walks E0[T] into its clip within a few hundred bars).
  anchor="activity", the default, instead sets E0[T] = EWMA(trades per window) / bars_per_window,
  so the threshold scale follows market activity while the flow expectations stay adaptive; it
  needs ticks_per_window, and the book's rule has to be asked for with anchor="bars".
  Either way E0[T] is clipped to [T0 / clip, T0 · clip].
The threshold is frozen when a bar opens, so one bar is judged against one target.
"""
import math

from pmlab.bars.base import Bar, BarBuilder
from pmlab.bars.tickrule import TickRule
from pmlab.events import Trade

MEASURES = ("tick", "volume", "dollar")


def flow(measure: str, tr: Trade) -> float:
    return 1.0 if measure == "tick" else tr.shares if measure == "volume" else tr.risk_usd


class _Adaptive(BarBuilder):
    """Shared EWMA bookkeeping for imbalance and run bars."""

    def __init__(self, measure: str, init_T: float, span_bars: float, span_ticks: float | None,
                 clip: float, sign: str, anchor: str, ticks_per_window: float | None,
                 span_windows: float):
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}")
        if sign not in ("aggressor", "tick_rule"):
            raise ValueError("sign must be 'aggressor' or 'tick_rule'")
        if anchor not in ("bars", "activity") or (anchor == "activity" and not ticks_per_window):
            raise ValueError("anchor must be 'bars', or 'activity' with ticks_per_window")
        super().__init__()
        self.measure = measure
        self.T0 = self.E_T = float(init_T)
        self.lo, self.hi = init_T / clip, init_T * clip
        self.a_bar = 2.0 / (span_bars + 1.0)
        self.a_tick = 2.0 / ((span_ticks or 10.0 * init_T) + 1.0)
        self.a_window = 2.0 / (span_windows + 1.0)
        self.anchor = anchor
        self.bars_per_window = ticks_per_window / init_T if ticks_per_window else None
        self.window_ticks = float(ticks_per_window or 0.0)
        self.seen = 0
        self.tick_rule = TickRule() if sign == "tick_rule" else None
        self.thr = 0.0

    def new_window(self, window: int):
        if self.anchor == "activity" and self.window is not None:
            self.window_ticks += self.a_window * (self.seen - self.window_ticks)
            self.E_T = self._clip(self.window_ticks / self.bars_per_window)
        super().new_window(window)
        self.seen = 0
        if self.tick_rule:
            self.tick_rule.reset()

    def _clip(self, T: float) -> float:
        return min(max(T, self.lo), self.hi)

    def _sign(self, tr: Trade) -> int:
        return self.tick_rule(tr.p_up) if self.tick_rule else tr.sign

    def threshold(self) -> float:
        return self.thr

    def _close(self, bar: Bar):
        if self.anchor == "bars":
            self.E_T = self._clip(self.E_T + self.a_bar * (bar.n_ticks - self.E_T))


class ImbalanceBars(_Adaptive):
    def __init__(self, measure: str, init_T: float, init_bv: float, init_bv_sd: float,
                 span_bars: float = 20, span_ticks: float | None = None, clip: float = 4.0,
                 floor_sd: float = 1.0, sign: str = "aggressor", anchor: str = "activity",
                 ticks_per_window: float | None = None, span_windows: float = 12, scale: float = 1.0):
        super().__init__(measure, init_T, span_bars, span_ticks, clip, sign, anchor,
                         ticks_per_window, span_windows)
        self.kind = f"{measure}_imbalance"
        self.E_bv = float(init_bv)
        self.E_bv2 = float(init_bv_sd) ** 2 + float(init_bv) ** 2
        self.floor_sd = floor_sd
        self.scale = scale             # fitted so realised bars per window hit the target (fit_scale)
        self.theta = 0.0

    def expected_threshold(self) -> float:
        sd = math.sqrt(max(self.E_bv2 - self.E_bv ** 2, 0.0))
        return self.scale * max(self.E_T * abs(self.E_bv), self.floor_sd * sd * math.sqrt(self.E_T))

    def _open(self):
        self.thr = self.expected_threshold()
        self.theta = 0.0

    def update(self, tr: Trade) -> Bar | None:
        b = self._sign(tr)
        self._add(tr)
        self.seen += 1
        bv = b * flow(self.measure, tr)
        self.theta += bv
        self.E_bv += self.a_tick * (bv - self.E_bv)
        self.E_bv2 += self.a_tick * (bv * bv - self.E_bv2)
        return self._emit(tr.ts) if abs(self.theta) >= self.thr else None
