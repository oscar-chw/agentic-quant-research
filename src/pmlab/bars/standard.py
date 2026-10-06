"""Standard bars (AFML 2.3.1): time, tick, volume and dollar.

A tick/volume/dollar bar closes on the first trade that takes its running total to the
size or beyond; that trade belongs to the bar. "Dollar" here is risk dollars,
shares * sqrt(p(1-p)), not as-traded USDC: buying Up at 0.98 and selling Down at 0.02 are
the same trade and must count the same.
"""
from pmlab import WINDOW_SECONDS
from pmlab.bars.base import Bar, BarBuilder
from pmlab.events import Trade


class TimeBars(BarBuilder):
    kind = "time"

    def __init__(self, seconds: float):
        super().__init__()
        self.seconds = seconds
        self.bucket = 0

    def threshold(self) -> float:
        return self.seconds

    def _bucket_end(self) -> float:
        return self.window + (self.bucket + 1) * self.seconds

    def update(self, tr: Trade) -> Bar | None:
        # A time bar is complete once time passes its end; in trade time that is known
        # when the first trade of a later bucket arrives. Empty buckets produce no bar.
        bucket = int(tr.t // self.seconds)
        bar = self._emit(self._bucket_end()) if self.n and bucket != self.bucket else None
        if self.n == 0:
            self.bucket = bucket
        self._add(tr)
        return bar

    def finish(self) -> Bar | None:
        if self.n and self._bucket_end() <= self.window + WINDOW_SECONDS:
            return self._emit(self._bucket_end())
        return None


class _SizeBars(BarBuilder):
    """Fixed size, or with bars_per_window set, size = EWMA(flow per window) / bars_per_window,
    updated at each new window. A size fixed on one calibration day misses the target count on
    busier or quieter days (research/reviews.md R1.1)."""

    def __init__(self, size: float, bars_per_window: float | None = None, span_windows: float = 12):
        super().__init__()
        if size <= 0:
            raise ValueError("bar size must be positive")
        self.size = size
        self.bars_per_window = bars_per_window
        self.window_flow = size * bars_per_window if bars_per_window else 0.0
        self.a_window = 2.0 / (span_windows + 1.0)
        self.seen = 0.0

    def new_window(self, window: int):
        if self.bars_per_window and self.window is not None:
            self.window_flow += self.a_window * (self.seen - self.window_flow)
            self.size = max(self.window_flow / self.bars_per_window, 1e-9)
        super().new_window(window)
        self.seen = 0.0

    def threshold(self) -> float:
        return self.size

    def _total(self) -> float:
        raise NotImplementedError

    def _flow(self, tr: Trade) -> float:
        raise NotImplementedError

    def update(self, tr: Trade) -> Bar | None:
        self._add(tr)
        self.seen += self._flow(tr)
        return self._emit(tr.ts) if self._total() >= self.size else None


class TickBars(_SizeBars):
    kind = "tick"

    def _total(self):
        return self.n

    def _flow(self, tr):
        return 1.0


class VolumeBars(_SizeBars):
    kind = "volume"

    def _total(self):
        return self.q

    def _flow(self, tr):
        return tr.shares


class DollarBars(_SizeBars):
    kind = "dollar"

    def _total(self):
        return self.risk

    def _flow(self, tr):
        return tr.risk_usd
