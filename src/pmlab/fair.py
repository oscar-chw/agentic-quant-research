"""Q-measure fair price of the Up token.

Pricing equation P = E_Q[X / β]: over five minutes β ≈ 1, and under Q BTC's log price is a
driftless random walk sampled once a second, so the Up token is worth Q(A_end ≥ K), where
A_end and K are the L-second price averages that settle the market (research/
resolution_rule.json). Derivation: lessons/L1_data_and_fair_price.ipynb §3. The rule averages prices: K is
formed exactly, as the log of the mean price, while the settling average is modelled as the mean of log
prices, the Gaussian approximation of the log of its mean. The Jensen gap between the two moves Q towards
Up by 1e-8 in log at σ = 5e-5/s and 2e-7 at 2e-4/s, at most 0.003 of a basis sd (research/code_review/
pricing.md, fair 1).

L is a property of the market's rule, not a fitted parameter: 60 for today's Chainlink 60 s TWAP
stream, 30 for the earlier 30 s stream, 1 for the original point-price rule (end price ≥ start
price), where the price is the textbook digital Φ((x_t − k)/√(σ²τ + basis²)). rule_lookback(market)
reads it; every function below takes it and defaults to 60, today's rule.

Two corrections, both fitted by maximum likelihood on real outcomes (fit()):
- lag: Chainlink's clock may trail Binance by a few seconds, so the settling averages are
  taken over Binance instants shifted back by `lag`;
- basis: Binance and Chainlink averages differ by noise with sd `basis` (log units),
  which caps how certain the price can get near expiry (L1 §4, last 30 seconds).
σ is an EWMA of squared 1-second log returns (causal: σ at instant t uses returns up to t),
scaled by vol_mult to absorb estimator bias and fat tails, and floored at sigma_floor: in quiet
spells Binance can sit in a sub-basis-point range for minutes (σ ≈ 2e-6) while Chainlink, which
aggregates several venues, still moves, and an unfloored fair price becomes wrongly certain
(research/reviews.md R2.5). The floor is fitted with the other parameters.

FairPricer is the streaming form used live; fair_window() is the vectorised batch form.
They must agree to rounding error (tests/test_fair.py).
"""
import math
import numbers
import re
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from pmlab import WINDOW_SECONDS as T

L = 60                    # default lookback: today's rule
SIGMA_FLOOR = 1e-6        # numerical floor; the fitted FairParams.sigma_floor is the model's


@dataclass(frozen=True)
class FairParams:
    halflife: float = 120.0   # seconds, EWMA of squared 1s log returns
    vol_mult: float = 1.0
    basis: float = 0.0        # sd of (Chainlink − Binance) settlement log difference
    lag: int = 0              # seconds Chainlink trails Binance
    sigma_floor: float = SIGMA_FLOOR

    def to_dict(self):
        return asdict(self)


def rule_lookback(market: dict) -> int:
    """Seconds averaged at each end of a market's settlement: 60 or 30 (Chainlink TWAP streams) or 1
    (point price: end price ≥ start price).

    Uses the normalised `twap_lookback` field when it holds an integer; otherwise reads the rule from
    the market's resolutionSource and description (real strings: tests/fixtures/rules). Text it cannot
    read raises, so an unknown rule is never silently priced as the default."""
    v = market.get("twap_lookback")
    if isinstance(v, numbers.Real) and not isinstance(v, bool) and float(v).is_integer() and v > 0:
        return int(v)
    text = " ".join(str(market.get(k) or "") for k in ("resolutionSource", "description"))
    if m := re.search(r"twap-(\d+)s", text, re.IGNORECASE):
        return int(m.group(1))
    if "twap" not in text.lower() and ("price at the end of the time range" in text
                                       or re.search(r"streams/btc-usd/?(\s|$)", text)):
        return 1
    raise ValueError(f"cannot read the settlement rule of market {market.get('slug', market.get('start'))!r}")


def variance_units(t, lag: int = 0, L: int = L):
    """Var(A_end | info at second t) / σ², prices sampled each second, A_end the mean of the last L.

    = Σ_{s ≥ t} a(s)², a(s) = min(τ(s)/L, 1)⁺ (pq.mean_slope). L = 1 gives τ⁺. Vectorised over t."""
    tau = np.asarray(T - lag - t, dtype=float)
    before = (tau - L) + (L + 1) * (2 * L + 1) / (6 * L)
    inside = tau * (tau + 1) * (2 * tau + 1) / (6 * L ** 2)
    return np.where(tau >= L, before, np.where(tau > 0, inside, 0.0))


def drift_units(t, lag: int = 0, L: int = L):
    """E(A_end − m_t | info at t) / μ for a drift μ per second: Σ of the slopes a(s) still to come.

    The Girsanov counterpart of variance_units (which sums a(s)²): a P drift moves the settling
    average by μ·drift_units(t), a shift that fades to 0 at settlement."""
    tau = np.asarray(T - lag - t, dtype=float)
    return np.where(tau >= L, (tau - L) + (L + 1) / 2, np.where(tau > 0, tau * (tau + 1) / (2 * L), 0.0))


def _prob(mean_minus_k, sigma, units, basis):
    """Φ((mean − k)/sd), the settled 0 or 1 when sd = 0, and NaN wherever mean − k is NaN (K not formed)."""
    sd = np.sqrt(sigma ** 2 * units + basis ** 2)
    m = np.asarray(mean_minus_k, dtype=float)
    z = m / np.where(sd > 0, sd, 1.0)
    return np.where(np.isnan(m), np.nan, np.where(sd > 0, norm.cdf(z), (m >= 0).astype(float)))


def ewma_var(r: np.ndarray, halflife: float) -> np.ndarray:
    """Causal EWMA of r² with pandas adjust=True weights (identical to FairPricer)."""
    return pd.Series(r * r).ewm(halflife=halflife, adjust=True).mean().to_numpy()


def window_inputs(close: pd.Series, start: int, lag: int, halflife: float, L: int = L) -> dict:
    """Per-second inputs for t = 0..T-1 from a Series of 1s kline closes indexed by open second.

    The price AT instant i is the close of the kline that opened at i-1. K needs every one of its L closes, as
    FairPricer does (else NaN, and so is every price); closes that start after the window start raise ValueError.
    """
    first, last = int(close.index.min()) + 1, int(close.index.max()) + 1
    if first > start:
        raise ValueError(f"closes start at instant {first}, after the window start {start}")
    instants = np.arange(first, last + 1)
    x = np.log(close.reindex(instants - 1).ffill().to_numpy())
    var = ewma_var(np.diff(x, prepend=x[0]), halflife)
    at = lambda i: i - first
    kr = close.reindex(np.arange(start - L - lag, start - lag))
    k = math.log(kr.mean()) if kr.notna().all() else float("nan")
    t = np.arange(T)
    x_t = x[at(start + t)]
    te = T - lag
    in_avg = (instants > start + te - L) & (instants <= start + te)
    cum_inside = np.cumsum(np.where(in_avg, x, 0.0))
    inside_sum = cum_inside[at(start + np.minimum(t, te))]
    tau = te - t
    mean = np.where(tau >= L, x_t, np.where(tau > 0, (inside_sum + tau * x_t) / L, inside_sum / L))
    return {"t": t, "mean_minus_k": mean - k, "sigma_raw": np.sqrt(var[at(start + t)]),
            "units": variance_units(t, lag, L), "x_t": x_t, "k": k}


def fair_window(close: pd.Series, start: int, p: FairParams, L: int = L) -> np.ndarray:
    w = window_inputs(close, start, p.lag, p.halflife, L)
    sigma = np.maximum(w["sigma_raw"] * p.vol_mult, p.sigma_floor)
    return _prob(w["mean_minus_k"], sigma, w["units"], p.basis)


class FairPricer:
    """Streaming fair price. Feed one price per second; ask for any watched window.

    Each watched window carries its own lookback L (watch(start, L)); watch it at least L + lag
    seconds before its start, or its K cannot be formed and the price stays NaN."""

    def __init__(self, params: FairParams):
        self.p = params
        self.alpha = 1.0 - 0.5 ** (1.0 / params.halflife)
        self.S = self.W = 0.0
        self.last_instant: int | None = None
        self.x = float("nan")
        self.hist: deque[tuple[int, float, float]] = deque(maxlen=L + params.lag + 4)  # (instant, price, x)
        self.windows: dict[int, dict] = {}

    def on_second(self, instant: int, price: float):
        """Price at `instant` (the last trade before it). Call once per second, in order."""
        x = math.log(price)
        r = 0.0 if self.last_instant is None else x - self.x
        self.S = (1 - self.alpha) * self.S + r * r
        self.W = (1 - self.alpha) * self.W + 1.0
        self.last_instant, self.x = instant, x
        self.hist.append((instant, price, x))
        te = T - self.p.lag
        for start, w in self.windows.items():
            n = w["L"]
            if w["k"] is None and instant >= start - self.p.lag:
                prices = [pr for i, pr, _ in self.hist if start - n - self.p.lag < i <= start - self.p.lag]
                w["k"] = math.log(sum(prices) / len(prices)) if len(prices) == n else float("nan")
            if start + te - n < instant <= start + te:
                w["inside"] += x

    def watch(self, start: int, L: int = L):
        if L + self.p.lag + 4 > self.hist.maxlen:
            self.hist = deque(self.hist, maxlen=L + self.p.lag + 4)
        self.windows.setdefault(start, {"k": None, "inside": 0.0, "L": L})

    def forget(self, start: int):
        self.windows.pop(start, None)

    def sigma(self) -> float:
        return max(math.sqrt(self.S / self.W) * self.p.vol_mult, self.p.sigma_floor) if self.W else self.p.sigma_floor

    def state(self, start: int) -> tuple[float, int] | None:
        """(mean − k, t) for a window at the last second fed: the inputs of the price and its Greeks."""
        w = self.windows[start]
        if w["k"] is None or self.last_instant is None or math.isnan(w["k"]):
            return None
        t, n = self.last_instant - start, w["L"]
        tau = T - self.p.lag - t
        mean = self.x if tau >= n else (w["inside"] + tau * self.x) / n if tau > 0 else w["inside"] / n
        return mean - w["k"], t

    def z(self, start: int) -> float:
        """(mean − k) / sd, the same quantity as features.base_features' z_q."""
        if (st := self.state(start)) is None:
            return float("nan")
        units = max(float(variance_units(st[1], self.p.lag, self.windows[start]["L"])), 1e-12)
        return float(st[0] / math.sqrt(self.sigma() ** 2 * units + self.p.basis ** 2))

    def fair(self, start: int) -> float:
        if (st := self.state(start)) is None:
            return float("nan")
        return float(_prob(st[0], self.sigma(), variance_units(st[1], self.p.lag, self.windows[start]["L"]), self.p.basis))


# ---------------------------------------------------------------- fitting

def _neg_loglik(theta, samples, sigma_floor=SIGMA_FLOOR):
    vol_mult, basis = math.exp(theta[0]), math.exp(theta[1])
    q = _prob(samples["mean_minus_k"], np.maximum(samples["sigma_raw"] * vol_mult, sigma_floor),
              samples["units"], basis)
    q = np.clip(q, 1e-9, 1 - 1e-9)
    y = samples["y"]
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


def fit(closes: dict[int, pd.Series], outcomes: dict[int, bool], grid=range(0, 300, 5),
        lags=range(0, 7), halflives=(30.0, 60.0, 120.0, 300.0),
        sigma_floors=(SIGMA_FLOOR,), lookbacks: dict[int, int] | None = None) -> tuple[FairParams, pd.DataFrame]:
    """Maximum likelihood over (vol_mult, basis) for each (lag, halflife, sigma_floor) on the grid.

    `lookbacks` maps a window start to its rule's L (rule_lookback); missing starts use L = 60.
    Samples within a window are dependent, so the likelihood is for point estimates only.
    Returns the best params and the full table of candidates with their mean log loss.
    """
    grid = np.asarray(list(grid))
    rows = []
    for lag in lags:
        for hl in halflives:
            parts = []
            for start, close in closes.items():
                w = window_inputs(close, start, lag, hl, (lookbacks or {}).get(start, L))
                parts.append(pd.DataFrame({k: w[k][grid] for k in ("mean_minus_k", "sigma_raw", "units")})
                             .assign(y=float(outcomes[start])))
            s = pd.concat(parts, ignore_index=True)
            s = {c: s[c].to_numpy() for c in s.columns}
            for floor in sigma_floors:
                res = minimize(_neg_loglik, x0=[0.0, math.log(2e-5)], args=(s, floor), method="Nelder-Mead",
                               options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 400})
                rows.append({"lag": lag, "halflife": hl, "sigma_floor": floor, "vol_mult": math.exp(res.x[0]),
                             "basis": math.exp(res.x[1]), "log_loss": res.fun})
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    b = table.iloc[0]
    return FairParams(float(b.halflife), float(b.vol_mult), float(b.basis), int(b.lag), float(b.sigma_floor)), table
