"""Volatility forecasts for the Q price, each testable out of sample (node 52; docs/pq_framework.md).

Q prices the settling average with a per-second σ. The baseline is an EWMA of squared 1 s log
returns (pmlab.fair). Candidates, all causal (σ at instant i uses only what is known at i):
- har: log σ² = c0 + c1·log EWMA² + c2·log RV(30 min) + c3·log RV(4 h), fitted by OLS on the
  realised variance of the next 5 minutes (Corsi's HAR in logs, with the lognormal mean correction);
- seasonal: HAR + log of the time-of-day profile of squared returns for the next 5 minutes. The
  clock is known in advance, so using it is not look-ahead; the profile comes from training days;
- dvol: seasonal + log DVOL² (Deribit's 30-day implied volatility index: the options market's
  forward-looking σ), each minute's close known from the end of that minute;
- local: the baseline σ with a state-dependent scale sd·exp(a·τ/T + b·|z₀|), the five-minute
  digital's analogue of local volatility (a smile in z and a term structure in τ), fitted by maximum
  likelihood on outcomes. a = b = 0 is the baseline exactly.
Every candidate's vol_mult and basis are refitted on the same training windows as the baseline's.

Instants follow pmlab.fair: the price AT instant i is the close of the 1 s kline that opened at
i − 1, and the return r_i = x_i − x_{i−1} is known at i.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from pmlab import WINDOW_SECONDS as T
from pmlab.fair import ewma_var

EPS = 1e-14                   # variance floor inside logs (a flat minute has zero realised variance)
SECONDS_PER_YEAR = 365 * 86400


@dataclass
class Timeline:
    instants: np.ndarray      # consecutive seconds
    x: np.ndarray             # log price at each instant
    r: np.ndarray             # 1 s log return known at each instant

    def at(self, instants) -> np.ndarray:
        return np.asarray(instants) - self.instants[0]


def timeline(close: pd.Series) -> Timeline:
    """From 1 s kline closes indexed by open second; missing seconds carry the last price (fair does too)."""
    close = close[~close.index.duplicated(keep="last")].sort_index()
    opens = np.arange(int(close.index.min()), int(close.index.max()) + 1)
    x = np.log(close.reindex(opens).ffill().to_numpy())
    return Timeline(opens + 1, x, np.diff(x, prepend=x[0]))


def trailing_mean_sq(r: np.ndarray, k: int) -> np.ndarray:
    """Mean of r² over the k returns up to and including each instant (NaN until k are known)."""
    c = np.concatenate([[0.0], np.cumsum(r * r)])
    out = np.full(len(r), np.nan)
    out[k - 1:] = (c[k:] - c[:-k]) / k
    return out


def forward_mean_sq(r: np.ndarray, k: int) -> np.ndarray:
    """Mean of r² over the k returns strictly after each instant: the regression target, never a feature."""
    c = np.concatenate([[0.0], np.cumsum(r * r)])
    out = np.full(len(r), np.nan)
    out[:len(r) - k] = (c[k + 1:] - c[1:len(r) - k + 1]) / k
    return out


def tod_profile(tl: Timeline, mask: np.ndarray, bucket: int = 300) -> np.ndarray:
    """Mean r² per time-of-day bucket (UTC) over the instants in `mask`, circularly smoothed."""
    b = (tl.instants[mask] % 86400) // bucket
    n = 86400 // bucket
    sums = np.bincount(b, weights=tl.r[mask] ** 2, minlength=n)
    counts = np.bincount(b, minlength=n)
    prof = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    prof = pd.Series(prof).fillna(np.nanmean(prof)).to_numpy()
    return 0.25 * np.roll(prof, 1) + 0.5 * prof + 0.25 * np.roll(prof, -1)


def seasonal_at(instants, profile: np.ndarray, horizon: int = 300, bucket: int = 300) -> np.ndarray:
    """The profile for the middle of the next `horizon` seconds (the clock is known in advance)."""
    return profile[((np.asarray(instants) + horizon // 2) % 86400) // bucket]


def dvol_at(instants, dvol: pd.DataFrame) -> np.ndarray:
    """Per-second σ implied by DVOL (annualised %, 1-minute closes) as last known at each instant.

    A minute bar stamped with its open time `ts` (ms) is known from ts/1000 + 60."""
    known = (dvol["ts"].to_numpy() // 1000 + 60).astype(np.int64)
    order = np.argsort(known)
    known, close = known[order], dvol["close"].to_numpy(dtype=float)[order]
    idx = np.searchsorted(known, np.asarray(instants), side="right") - 1
    out = np.where(idx >= 0, close[np.maximum(idx, 0)], np.nan)
    return out / 100 / math.sqrt(SECONDS_PER_YEAR)


HAR_SETS = {"har": ("ewma", "rv1800", "rv14400"),
            "seasonal": ("ewma", "rv1800", "rv14400", "season"),
            "dvol": ("ewma", "rv1800", "rv14400", "season", "dvol")}


def regressors(tl: Timeline, halflife: float, profile: np.ndarray | None = None,
               dvol: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """log variance regressors at every instant of the timeline."""
    out = {"ewma": np.log(ewma_var(tl.r, halflife) + EPS),
           "rv1800": np.log(trailing_mean_sq(tl.r, 1800) + EPS),
           "rv14400": np.log(trailing_mean_sq(tl.r, 14400) + EPS)}
    if profile is not None:
        out["season"] = np.log(seasonal_at(tl.instants, profile) + EPS)
    if dvol is not None:
        out["dvol"] = np.log(dvol_at(tl.instants, dvol) ** 2 + EPS)
    return out


class HAR:
    """OLS of log forward realised variance on log regressors; predicts σ per √second.

    The target is winsorised at its training 1st/99th percentiles: in logs, the quietest five
    minutes (BTC barely ticking, log RV down to −32 against a median of −21) would otherwise dominate
    the fit and the residual variance."""

    def __init__(self, names):
        self.names = list(names)

    def _X(self, reg: dict, idx) -> np.ndarray:
        return np.column_stack([np.ones(len(idx))] + [reg[n][idx] for n in self.names])

    def fit(self, reg: dict, target_logvar: np.ndarray, idx) -> "HAR":
        X, y = self._X(reg, idx), target_logvar[idx]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        y = np.clip(y, *np.quantile(y[ok], [0.01, 0.99]))
        self.coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        self.resid_var = float(np.var(y[ok] - X[ok] @ self.coef))
        self.n = int(ok.sum())
        return self

    def sigma(self, reg: dict, idx) -> np.ndarray:
        """√E[variance]: exp(mean + ½·residual variance) of the lognormal forecast."""
        return np.sqrt(np.exp(self._X(reg, idx) @ self.coef + 0.5 * self.resid_var))


def q_prob(m, u, sigma, t, params: dict, sigma_floor: float) -> np.ndarray:
    """Q(Up) with σ → max(vol_mult·σ, floor), basis noise, and the optional local scale exp(a·τ/T + b·|z₀|)."""
    s = np.maximum(params["vol_mult"] * np.asarray(sigma, dtype=float), sigma_floor)
    sd = np.sqrt(s ** 2 * u + params["basis"] ** 2)
    a, b = params.get("a", 0.0), params.get("b", 0.0)
    if a or b:
        z0 = np.abs(m / np.where(sd > 0, sd, 1.0))
        sd = sd * np.exp(a * (T - np.asarray(t)) / T + b * np.minimum(z0, 5.0))
    return np.where(sd > 0, norm.cdf(m / np.where(sd > 0, sd, 1.0)), (np.asarray(m) >= 0).astype(float))


def _nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_q(m, u, sigma, t, y, sigma_floor: float, local: bool = False) -> dict:
    """Maximum likelihood of (vol_mult, basis[, a, b]) on sampled seconds of training windows."""
    ok = np.isfinite(sigma) & np.isfinite(m)
    m, u, sigma, t, y = m[ok], u[ok], sigma[ok], t[ok], y[ok]

    def unpack(th):
        d = {"vol_mult": math.exp(th[0]), "basis": math.exp(th[1])}
        if local:
            d.update(a=th[2], b=th[3])
        return d

    x0 = [0.0, math.log(3e-5)] + ([0.0, 0.0] if local else [])
    res = minimize(lambda th: _nll(q_prob(m, u, sigma, t, unpack(th), sigma_floor), y), x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
    return {**unpack(res.x), "train_log_loss": float(res.fun)}


def qlike(realised_var, forecast_var) -> np.ndarray:
    """Patton's QLIKE loss per forecast: robust to noise in the realised variance proxy (0 when equal)."""
    ratio = np.asarray(realised_var) / np.asarray(forecast_var)
    return ratio - np.log(ratio) - 1
