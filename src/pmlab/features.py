"""Per-second, strictly causal feature arrays for one window: what a trader knows at second t.

Availability convention (the adapted-process rule, L1-preliminaries slide 15):
- the BTC price AT instant i is known at i;
- a trade stamped second s happened somewhere in [s, s+1), so it is known at s+1;
- a bar closed by a trade stamped s is known at s+1.
Every array has length T and entry [t] uses only information known at start + t.

Backtest, replay and live all fill the same columns. Historic runs cannot see the order
book, so bid/ask are proxied by the last sell/buy-aggressor print; live runs overwrite
them with the real book. The forward test measures that gap.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS as T
from pmlab.bars import Bar, BarBuilder
from pmlab.events import trades_from_tape
from pmlab.fair import L, FairParams, window_inputs, _prob
from pmlab.polymarket import in_window

LOGIT_CLIP = 0.005


def logit(p):
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


@dataclass
class WindowFeatures:
    start: int
    cols: dict[str, np.ndarray]
    tape: pd.DataFrame                       # full tape, for fills (never shown to strategies)
    y: float | None = None                   # outcome (never shown to strategies)
    meta: dict = field(default_factory=dict)

    def __getitem__(self, name) -> np.ndarray:
        return self.cols[name]


def _last_print(ts: np.ndarray, values: np.ndarray, known_until: np.ndarray):
    """Value and age of the last print stamped <= known_until[t] - 1, per t."""
    if not len(ts):
        return np.full(len(known_until), np.nan), np.full(len(known_until), np.nan)
    idx = np.searchsorted(ts, known_until - 1, side="right") - 1
    ok = idx >= 0
    val = np.where(ok, values[np.maximum(idx, 0)], np.nan)
    age = np.where(ok, known_until - ts[np.maximum(idx, 0)], np.nan)
    return val, age


def q_features(start: int, close: pd.Series, fair: FairParams, L: int = L) -> dict:
    """The columns that depend on the fair-price parameters and the rule's lookback L: fair, sigma,
    z_q, btc_z10, btc_z30. `dev` also depends on fair: dev_feature(last, fair)."""
    t = np.arange(T)
    w = window_inputs(close, start, fair.lag, fair.halflife, L)
    sigma = np.maximum(w["sigma_raw"] * fair.vol_mult, fair.sigma_floor)
    cols = {"fair": _prob(w["mean_minus_k"], sigma, w["units"], fair.basis), "sigma": sigma,
            "z_q": w["mean_minus_k"] / np.sqrt(sigma ** 2 * np.maximum(w["units"], 1e-12) + fair.basis ** 2)}
    x = np.log(close.reindex(np.arange(start - 61, start + T)).ffill().to_numpy())  # x[j]: instant start-60+j
    for lag in (10, 30):
        ret = x[60 + t] - x[60 + t - lag]
        cols[f"btc_z{lag}"] = ret / (sigma * np.sqrt(lag))
    return cols


def dev_feature(last, fair):
    """logit(last print) − logit(Q), 0 before the first print."""
    return logit(np.where(np.isnan(last), fair, last)) - logit(fair)


def base_features(start: int, tape: pd.DataFrame, close: pd.Series, fair: FairParams, L: int = L) -> dict:
    t = np.arange(T)
    now = start + t
    cols = {"t": t.astype(float)}
    cols.update(q_features(start, close, fair, L))

    ts = tape["ts"].to_numpy(dtype=float)
    p = tape["p_up"].to_numpy(dtype=float)
    sign = tape["sign"].to_numpy()
    for name, mask in (("last", np.ones(len(ts), bool)), ("ask", sign > 0), ("bid", sign < 0)):
        cols[name], cols[f"{name}_age"] = _last_print(ts[mask], p[mask], now)

    inw = (ts >= start) & (ts < start + T)
    risk = tape["shares"].to_numpy() * np.sqrt(np.clip(p * (1 - p), 0, None))
    cum_signed = np.concatenate([[0.0], np.cumsum(np.where(inw, sign * risk, 0.0))])
    cum_abs = np.concatenate([[0.0], np.cumsum(np.where(inw, risk, 0.0))])
    cum_n = np.concatenate([[0], np.cumsum(inw)])
    k = np.searchsorted(ts, now - 1, side="right")
    cols["flow"] = np.divide(cum_signed[k], cum_abs[k], out=np.zeros(T), where=cum_abs[k] > 0)
    cols["n_trades"] = cum_n[k].astype(float)
    cols["dev"] = dev_feature(cols["last"], cols["fair"])
    return cols


def run_bars(builder: BarBuilder, start: int, tape: pd.DataFrame) -> list[tuple[int, Bar]]:
    """Feed the window's in-window trades; return (second the bar became known, bar)."""
    builder.new_window(start)
    out = []
    for tr in trades_from_tape(in_window(tape), start):
        bar = builder.update(tr)
        if bar is not None:
            out.append((int(tr.ts) + 1, bar))
    return out


def bar_features(kind: str, known: list[tuple[int, Bar]], start: int) -> dict:
    """Last completed bar's flow and return, its age, and how many bars closed in 30 s."""
    now = start + np.arange(T)
    when = np.array([s for s, _ in known], dtype=float)
    if not len(when):
        nan = np.full(T, np.nan)
        return {f"{kind}_imb": nan, f"{kind}_ret": nan.copy(), f"{kind}_age": nan.copy(),
                f"{kind}_n30": np.zeros(T)}
    bars = [b for _, b in known]
    imb = np.array([b.risk_imbalance / b.risk_usd if b.risk_usd else 0.0 for b in bars])
    lg = logit(np.array([b.close for b in bars]))
    ret = np.diff(lg, prepend=logit(np.array([bars[0].open]))[0])
    idx = np.searchsorted(when, now, side="right") - 1
    ok = idx >= 0
    j = np.maximum(idx, 0)
    n30 = (np.searchsorted(when, now, side="right") - np.searchsorted(when, now - 30, side="right"))
    return {f"{kind}_imb": np.where(ok, imb[j], np.nan), f"{kind}_ret": np.where(ok, ret[j], np.nan),
            f"{kind}_age": np.where(ok, now - when[j], np.nan), f"{kind}_n30": n30.astype(float)}
