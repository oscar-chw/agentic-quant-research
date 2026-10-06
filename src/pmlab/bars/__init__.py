"""The nine AFML bars plus time bars, sized so every kind aims at the same bars per window.

Bar statistics depend strongly on bar frequency, so kinds are only compared at equal sample
counts (research/reviews.md R1.1; the book's ch.2 exercises do not ask for this). Every builder
is made from a Calibration (flow statistics measured on a *prior* sample, never on the data
being barred) and a target number of bars per window.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS
from pmlab.bars.base import BAR_COLUMNS, Bar, BarBuilder
from pmlab.bars.imbalance import ImbalanceBars, flow
from pmlab.bars.runs import RunBars
from pmlab.bars.standard import DollarBars, TickBars, TimeBars, VolumeBars
from pmlab.events import Trade

STANDARD = ("time", "tick", "volume", "dollar")
NINE = ("tick", "volume", "dollar",
        "tick_imbalance", "volume_imbalance", "dollar_imbalance",
        "tick_run", "volume_run", "dollar_run")
KINDS = ("time",) + NINE


@dataclass(frozen=True)
class Calibration:
    windows: int
    ticks: float                 # mean trades per window
    per_window: dict             # measure -> mean flow per window
    bv_mean: dict                # measure -> E[b v]
    bv_sd: dict                  # measure -> sd(b v)
    p_buy: float                 # P[b = +1]
    v_buy: dict                  # measure -> E[v | b = +1]
    v_sell: dict                 # measure -> E[v | b = -1]
    v2: dict                     # measure -> E[v^2]


def calibrate(windows: dict[int, list[Trade]]) -> Calibration:
    trades = [tr for w in windows.values() for tr in w]
    if not trades:
        raise ValueError("cannot calibrate on zero trades")
    b = np.array([tr.sign for tr in trades], dtype=float)
    per_window, bv_mean, bv_sd, v_buy, v_sell, v2 = {}, {}, {}, {}, {}, {}
    for m in ("tick", "volume", "dollar"):
        v = np.array([flow(m, tr) for tr in trades])
        per_window[m] = v.sum() / len(windows)
        bv_mean[m], bv_sd[m] = float((b * v).mean()), float((b * v).std())
        v_buy[m] = float(v[b > 0].mean()) if (b > 0).any() else 0.0
        v_sell[m] = float(v[b < 0].mean()) if (b < 0).any() else 0.0
        v2[m] = float((v * v).mean())
    return Calibration(len(windows), len(trades) / len(windows), per_window, bv_mean, bv_sd,
                       float((b > 0).mean()), v_buy, v_sell, v2)


def make_builder(kind: str, cal: Calibration, bars_per_window: float, scale: float = 1.0,
                 adaptive_size: bool = False, **kw) -> BarBuilder:
    """Adaptive kinds default to anchor="activity" (see pmlab.bars.imbalance); pass
    anchor="bars" for the book's own-bar-length EWMA. `scale` multiplies the bar size or
    threshold (see fit_scale); `adaptive_size` lets standard bars follow activity per window."""
    T = cal.ticks / bars_per_window
    if kind == "time":
        return TimeBars(WINDOW_SECONDS / bars_per_window)
    if kind in ("tick", "volume", "dollar"):
        return {"tick": TickBars, "volume": VolumeBars, "dollar": DollarBars}[kind](
            scale * cal.per_window[kind] / bars_per_window, bars_per_window / scale if adaptive_size else None)
    kw = {"scale": scale} | kw
    measure, family = kind.split("_")
    kw = {"anchor": "activity", "ticks_per_window": cal.ticks} | kw
    if family == "imbalance":
        return ImbalanceBars(measure, T, cal.bv_mean[measure], cal.bv_sd[measure], **kw)
    if family == "run":
        return RunBars(measure, T, cal.p_buy, cal.v_buy[measure], cal.v_sell[measure], cal.v2[measure], **kw)
    raise ValueError(f"unknown bar kind {kind!r}; expected one of {KINDS}")


def build(windows: dict[int, list[Trade]], builder: BarBuilder) -> pd.DataFrame:
    """Run a builder over windows in time order. Columns are BAR_COLUMNS."""
    rows = []
    for w in sorted(windows):
        builder.new_window(w)
        for tr in windows[w]:
            bar = builder.update(tr)
            if bar is not None:
                rows.append(bar)
        bar = builder.finish()
        if bar is not None:
            rows.append(bar)
    return pd.DataFrame([[getattr(b, c) for c in BAR_COLUMNS] for b in rows], columns=BAR_COLUMNS)


def fit_scale(kind: str, cal: Calibration, bars_per_window: float, windows: dict[int, list[Trade]],
              adaptive_size: bool = True, iters: int = 14, lo: float = 0.05, hi: float = 20.0) -> tuple[float, float]:
    """Bisection on log(scale) until realised bars per window on `windows` equals the target.

    Bar statistics depend strongly on bar frequency, so kinds are only comparable at equal
    realised counts (research/reviews.md R1.1). Fit on calibration windows, never on study data.
    Returns (scale, realised bars per window)."""
    if kind == "time":
        return 1.0, float(len(build(windows, make_builder(kind, cal, bars_per_window))) / len(windows))

    def count(scale):
        b = make_builder(kind, cal, bars_per_window, scale=scale, adaptive_size=adaptive_size)
        return len(build(windows, b)) / len(windows)

    a, z = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = (a + z) / 2
        if count(np.exp(mid)) > bars_per_window:      # too many bars -> larger threshold
            a = mid
        else:
            z = mid
    scale = float(np.exp((a + z) / 2))
    return scale, count(scale)


def logit_returns(bars: pd.DataFrame, clip: float = 0.005) -> pd.Series:
    """Bar-to-bar change in logit(close) within each window (never across windows)."""
    p = bars["close"].clip(clip, 1 - clip)
    lg = np.log(p / (1 - p))
    return lg.groupby(bars["window"]).diff().dropna()


__all__ = ["Bar", "BarBuilder", "BAR_COLUMNS", "Calibration", "KINDS", "NINE", "STANDARD",
           "build", "calibrate", "fit_scale", "make_builder", "logit_returns"]
