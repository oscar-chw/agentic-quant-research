"""Run bars (AFML 2.3.2(3)–2.3.2(4)): sample when one side trades more than expected.

Within a bar θ_T = max(Σ_{b=+1} v_t, Σ_{b=-1} v_t): the larger of buy flow and sell flow
(the book's cumulative definition, not the longest streak). The book closes the bar when
θ_T ≥ E0[T] · max(P[b=1] · E0[v | b=1], (1 − P[b=1]) · E0[v | b=−1]).

Writing B + S = Σ v and D = B − S = Σ b v, max(B, S) = (B + S)/2 + |D|/2, and the book's
formula equals T·E[v]/2 + |E[D]|/2. By Jensen, E|D| ≥ |E[D]|, so the default (`jensen=True`)
uses E|D| for a normal D with mean T·μ_d and variance T·σ_d²; `jensen=False` restores the book's
formula. The E|D| term is only a level change of the threshold: neither threshold makes the
expected first-passage time equal E0[T], so with E0[T] learned from bar lengths (anchor="bars")
the book's rule ratchets E0[T] down to its lower clip and the E|D| rule drifts up to its upper
clip (tests/test_bars_runs.py). What holds the target is the activity anchor (anchor="activity",
the default, as in make_builder), which sets E0[T] from trades per window; the remaining level
difference is absorbed by the fitted scale (pmlab.bars.fit_scale) in the bar study, and shows as
a higher threshold at scale 1.0 live.
Opposite trades do not cancel, so the threshold cannot collapse the way imbalance bars'
can. E0[T] is clipped exactly as for imbalance bars and the threshold is frozen at bar open.
"""
import math

from pmlab.bars.base import Bar
from pmlab.bars.imbalance import _Adaptive, flow
from pmlab.events import Trade


def expected_abs_normal(mu: float, sd: float) -> float:
    if sd <= 0:
        return abs(mu)
    z = mu / sd
    return sd * math.sqrt(2 / math.pi) * math.exp(-z * z / 2) + mu * math.erf(z / math.sqrt(2))


class RunBars(_Adaptive):
    def __init__(self, measure: str, init_T: float, init_p_buy: float, init_v_buy: float,
                 init_v_sell: float, init_v2: float | None = None, span_bars: float = 20,
                 span_ticks: float | None = None, clip: float = 4.0, sign: str = "aggressor",
                 jensen: bool = True, anchor: str = "activity", ticks_per_window: float | None = None,
                 span_windows: float = 12, scale: float = 1.0):
        super().__init__(measure, init_T, span_bars, span_ticks, clip, sign, anchor,
                         ticks_per_window, span_windows)
        self.kind = f"{measure}_run"
        self.P = float(init_p_buy)
        self.v_buy = float(init_v_buy)
        self.v_sell = float(init_v_sell)
        mean_v = self.P * self.v_buy + (1 - self.P) * self.v_sell
        self.v2 = float(init_v2) if init_v2 is not None else mean_v ** 2
        self.jensen = jensen
        self.scale = scale             # fitted so realised bars per window hit the target (fit_scale)
        self.buy_flow = self.sell_flow = 0.0

    def expected_threshold(self) -> float:
        buy, sell = self.P * self.v_buy, (1.0 - self.P) * self.v_sell
        if not self.jensen:
            return self.scale * self.E_T * max(buy, sell)
        mu_d = buy - sell
        sd_d = math.sqrt(max(self.v2 - mu_d * mu_d, 0.0))
        T = self.E_T
        return self.scale * (T * (buy + sell) / 2 + expected_abs_normal(T * mu_d, math.sqrt(T) * sd_d) / 2)

    def _open(self):
        self.thr = self.expected_threshold()
        self.buy_flow = self.sell_flow = 0.0

    def update(self, tr: Trade) -> Bar | None:
        b = self._sign(tr)
        self._add(tr)
        self.seen += 1
        v = flow(self.measure, tr)
        a = self.a_tick
        if b > 0:
            self.buy_flow += v
            self.v_buy += a * (v - self.v_buy)
        else:
            self.sell_flow += v
            self.v_sell += a * (v - self.v_sell)
        self.P += a * ((b > 0) - self.P)
        self.v2 += a * (v * v - self.v2)
        return self._emit(tr.ts) if max(self.buy_flow, self.sell_flow) >= self.thr else None
