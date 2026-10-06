"""Synthetic event rows shaped like the event dataset (pmlab.afml.events) for the meta-label model tests.

Each Taiwan day holds `windows` 5-minute windows and each window `events` rows of one rule. A window has a latent
z; Up wins with probability sigmoid(beta z) (beta flips sign from `flip_day` on). Every row has a random side, so its
label (the side won) is predictable from f_signal = side x (z + noise). f_window is a per-window constant (a handle a
model could memorise if test rows leaked into training), f_side the side, f_noise_* noise (f_noise_1 has gaps).
Spans run from the event second to the window end; paths are a random walk that settles at the outcome.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from pmlab import WINDOW_SECONDS as T

TW = 8 * 3600
DAY0 = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()) - TW          # a Taiwan midnight, in UTC seconds


def synthetic(days: int = 10, windows: int = 20, events: int = 5, beta: float = 3.0, flip_day: int | None = None,
              seed: int = 0, rule: str = "q_taker", day0: int = DAY0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts, paths = [], []
    for d in range(days):
        b = beta if flip_day is None or d < flip_day else -beta
        for slot in np.sort(rng.choice(288, windows, replace=False)):
            start = day0 + d * 86400 + 300 * int(slot)
            z = rng.normal()
            up = float(rng.random() < 1.0 / (1.0 + np.exp(-b * z)))
            t = np.sort(rng.choice(np.arange(5, 281), events, replace=False))
            side = rng.choice([-1, 1], events).astype(np.int8)
            label = (side == (1 if up else -1)).astype(np.int8)
            noise1 = rng.normal(size=events)
            noise1[rng.random(events) < 0.1] = np.nan
            parts.append(pd.DataFrame({
                "start": np.full(events, start, np.int64), "t": t.astype(np.int32),
                "day": pd.Timestamp(start, unit="s").strftime("%Y-%m-%d"),
                "day_tw": pd.Timestamp(start + TW, unit="s").strftime("%Y-%m-%d"), "rule": rule, "side": side,
                "price": 0.5, "cost": 0.5035, "up_won": up, "payoff": label.astype(float),
                "pnl": label - 0.5035, "ret": (label - 0.5035) / 0.5035, "label": label, "meta_label": label,
                "t0": start + t.astype(np.int64), "t1": np.full(events, start + T, np.int64),
                "avg_uniqueness": np.nan, "w_return_raw": np.nan,
                "f_tau": (T - t).astype(np.float32), "f_signal": (side * (z + rng.normal(0, 0.5, events))).astype(np.float32),
                "f_side": side.astype(np.float32), "f_window": np.full(events, rng.random(), np.float32),
                "f_noise_0": rng.normal(size=events).astype(np.float32), "f_noise_1": noise1.astype(np.float32)}))
            q = np.clip(0.5 + np.cumsum(rng.normal(0, 0.01, T)), 0.01, 0.99)
            paths.append(pd.DataFrame({"start": np.full(T + 1, start, np.int64), "t": np.arange(T + 1),
                                       "q": np.concatenate([q, [up]])}))
    return pd.concat(parts, ignore_index=True), pd.concat(paths, ignore_index=True)


def priced(days: int = 10, windows: int = 100, events: int = 3, seed: int = 0, rule: str = "q_taker",
           day0: int = DAY0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windows whose Up price is the true probability of Up. BTC moves as a Gaussian random walk x (unit variance a
    second, x_0 = 0) against the strike 0, so q_t = Phi(x_t / sqrt(T - t)) = P(x_T > 0 | x_t) and the path settles at
    q_T = 1{x_T > 0}. Each row takes a random side at a random second in [5, 200]; its label is whether that side won,
    so P(label = 1 | row) = q_side, and its one feature is f_logit_q = logit(q_side). A calibrated model fitted on it
    has slope 1 on f_logit_q. The return attribution |sum of dq_u / c_u| of such a row is close to |label - q_side|,
    so it carries the outcome."""
    rng = np.random.default_rng(seed)
    parts, paths = [], []
    for d in range(days):
        for slot in np.sort(rng.choice(288, windows, replace=False)):
            start = day0 + d * 86400 + 300 * int(slot)
            x = np.concatenate([[0.0], np.cumsum(rng.normal(size=T))])
            up = float(x[T] > 0)
            q = np.concatenate([stats.norm.cdf(x[:T] / np.sqrt(T - np.arange(T))), [up]])
            t = np.sort(rng.choice(np.arange(5, 201), events, replace=False))
            side = rng.choice([-1, 1], events).astype(np.int8)
            q_side = np.clip(np.where(side > 0, q[t], 1.0 - q[t]), 1e-9, 1 - 1e-9)
            label = (side == (1 if up else -1)).astype(np.int8)
            parts.append(pd.DataFrame({
                "start": np.full(events, start, np.int64), "t": t.astype(np.int32),
                "day_tw": pd.Timestamp(start + TW, unit="s").strftime("%Y-%m-%d"), "rule": rule, "side": side,
                "up_won": up, "label": label, "meta_label": label, "t0": start + t.astype(np.int64),
                "t1": np.full(events, start + T, np.int64), "q_side": q_side,
                "f_logit_q": np.log(q_side / (1 - q_side)).astype(np.float32)}))
            paths.append(pd.DataFrame({"start": np.full(T + 1, start, np.int64), "t": np.arange(T + 1), "q": q}))
    return pd.concat(parts, ignore_index=True), pd.concat(paths, ignore_index=True)


def permute_outcomes(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Window outcomes shuffled across windows: rows of a window keep one (shuffled) outcome, labels follow."""
    out = rows.copy()
    starts = out["start"].to_numpy()
    _, first, inv = np.unique(starts, return_index=True, return_inverse=True)
    up = np.random.default_rng(seed).permutation(out["up_won"].to_numpy(float)[first])[inv]
    label = (out["side"].to_numpy() == np.where(up > 0, 1, -1)).astype(np.int8)
    out["up_won"], out["label"], out["meta_label"] = up, label, label
    return out
