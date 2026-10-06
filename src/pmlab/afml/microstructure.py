"""The rest of AFML ch.19 next to pmlab.micro (Roll, Corwin-Schultz, Kyle, Amihud).

- beckers_parkinson: volatility from bar highs and lows (19.3.3). For a driftless Brownian log price,
  E[ln(H/L)^2] = 4 ln 2 sigma^2 and E[ln(H/L)] = sqrt(8 / pi) sigma per bar, so each moment gives an
  estimate of sigma per bar.
- hasbrouck_lambda: the slope of bar log returns on the bar's sum of b sqrt(p V) over its trades
  (19.4.3), with its t-statistic; no intercept, as the book writes the regression.
- bulk_volume_buy: bulk volume classification, V_buy = V Z(dP / sigma_dP) for a bar with price change
  dP (19.5.2, Easley, López de Prado and O'Hara 2012), Z the standard normal CDF or a Student t CDF.
- volume_buckets: equal-volume buckets filled bar by bar, a bar's buy and sell volume split pro rata
  when it straddles a bucket boundary (19.5.2).
- vpin: sum over the last n buckets of |V_sell - V_buy| / (n V) (19.5.2).
- corwin_schultz_beta, corwin_schultz_gamma, corwin_schultz_alpha, corwin_schultz_spread: the Corwin-Schultz (2012)
  spread as a series (19.3.2, Eqs. 86-89, Snippet 19.1): beta_t = the mean over the last sl pairs of bars of the sum of
  the two bars' squared log high/low ranges, gamma_t = the squared log range of the two bars together, alpha from both
  with negative values set to 0, and the relative spread S = 2 (e^alpha - 1) / (1 + e^alpha). Entry t uses bars t - sl
  .. t; earlier entries are NaN. pmlab.micro.corwin_schultz is the one-pair form the live features use, scaled to price
  units.
- high_low_sigma: the volatility per bar implied by the same beta and gamma once the spread is taken out (19.3.3,
  Snippet 19.2), sigma = (2^-1/2 - 1) sqrt(beta) / (k2 (3 - 2 sqrt 2)) + sqrt(gamma / (k2^2 (3 - 2 sqrt 2))), negative
  values set to 0.
- bar_order_imbalance: pmlab.micro.vpin under an accurate name. That function is the share-weighted mean
  absolute order imbalance of the last n bars with true aggressor sides: no equal-volume buckets and no
  classification, so it is not VPIN. pmlab.micro keeps its old name because other code imports it.
"""
import numpy as np
from scipy import stats

from pmlab.micro import vpin as _bar_imbalance

K1 = 4.0 * np.log(2.0)
K2 = np.sqrt(8.0 / np.pi)
CS_DEN = 3.0 - 2.0 * np.sqrt(2.0)


def beckers_parkinson(high, low) -> dict:
    hl = np.log(np.asarray(high, dtype=float) / np.asarray(low, dtype=float))
    hl = hl[np.isfinite(hl)]
    if not len(hl):
        return {"sigma_sq": np.nan, "sigma_abs": np.nan, "n": 0}
    return {"sigma_sq": float(np.sqrt(np.mean(hl ** 2) / K1)), "sigma_abs": float(np.mean(hl) / K2), "n": int(len(hl))}


def hasbrouck_lambda(returns, signed_root_dollars) -> dict:
    r = np.asarray(returns, dtype=float)
    x = np.asarray(signed_root_dollars, dtype=float)
    ok = np.isfinite(r) & np.isfinite(x)
    r, x = r[ok], x[ok]
    if len(r) < 3 or not np.any(x):
        return {"lambda": np.nan, "t": np.nan, "n": int(len(r))}
    lam = float(x @ r / (x @ x))
    resid = r - lam * x
    se = np.sqrt(resid @ resid / (len(r) - 1) / (x @ x))
    return {"lambda": lam, "t": float(lam / se) if se > 0 else np.nan, "n": int(len(r))}


def bulk_volume_buy(dp, volume, sigma_dp: float, df: float | None = None) -> np.ndarray:
    z = np.asarray(dp, dtype=float) / sigma_dp
    cdf = stats.norm.cdf(z) if df is None else stats.t.cdf(z, df)
    return np.asarray(volume, dtype=float) * cdf


def volume_buckets(volume, buy, bucket_volume: float) -> dict:
    """Fill buckets of bucket_volume in bar order. Returns per completed bucket its buy and sell volume and
    the index of the bar that completed it; the unfilled remainder is dropped."""
    volume = np.asarray(volume, dtype=float)
    buy = np.asarray(buy, dtype=float)
    out_buy, out_sell, out_end = [], [], []
    fill = b_acc = s_acc = 0.0
    for i, (v, bv) in enumerate(zip(volume, buy)):
        if v <= 0:
            continue
        share = bv / v
        while v > 0:
            take = min(v, bucket_volume - fill)
            fill += take
            b_acc += take * share
            s_acc += take * (1 - share)
            v -= take
            if fill >= bucket_volume * (1 - 1e-12):
                out_buy.append(b_acc)
                out_sell.append(s_acc)
                out_end.append(i)
                fill = b_acc = s_acc = 0.0
    return {"buy": np.asarray(out_buy), "sell": np.asarray(out_sell), "end_bar": np.asarray(out_end, dtype=int)}


def vpin(bucket_buy, bucket_sell, n: int = 50) -> np.ndarray:
    """VPIN after each bucket (NaN until n buckets exist)."""
    b, s = np.asarray(bucket_buy, dtype=float), np.asarray(bucket_sell, dtype=float)
    imb = np.abs(s - b)
    tot = b + s
    out = np.full(len(b), np.nan)
    if len(b) < n:
        return out
    ci = np.concatenate([[0.0], np.cumsum(imb)])
    ct = np.concatenate([[0.0], np.cumsum(tot)])
    out[n - 1:] = (ci[n:] - ci[:-n]) / (ct[n:] - ct[:-n])
    return out


def corwin_schultz_beta(high, low, sl: int = 1) -> np.ndarray:
    hl = np.log(np.asarray(high, dtype=float) / np.asarray(low, dtype=float)) ** 2
    pair = np.full(len(hl), np.nan)
    pair[1:] = hl[1:] + hl[:-1]
    out = np.full(len(hl), np.nan)
    if len(hl) >= sl + 1:
        c = np.concatenate([[0.0], np.cumsum(pair[1:])])
        out[sl:] = (c[sl:] - c[:-sl]) / sl
    return out


def corwin_schultz_gamma(high, low) -> np.ndarray:
    h, lo = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    out = np.full(len(h), np.nan)
    out[1:] = np.log(np.maximum(h[1:], h[:-1]) / np.minimum(lo[1:], lo[:-1])) ** 2
    return out


def corwin_schultz_alpha(beta, gamma) -> np.ndarray:
    beta, gamma = np.asarray(beta, dtype=float), np.asarray(gamma, dtype=float)
    alpha = (np.sqrt(2.0) - 1.0) * np.sqrt(beta) / CS_DEN - np.sqrt(gamma / CS_DEN)
    return np.where(np.isfinite(alpha), np.maximum(alpha, 0.0), np.nan)


def corwin_schultz_spread(high, low, sl: int = 1) -> np.ndarray:
    alpha = corwin_schultz_alpha(corwin_schultz_beta(high, low, sl), corwin_schultz_gamma(high, low))
    return 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))


def high_low_sigma(beta, gamma) -> np.ndarray:
    beta, gamma = np.asarray(beta, dtype=float), np.asarray(gamma, dtype=float)
    sigma = (2.0 ** -0.5 - 1.0) * np.sqrt(beta) / (K2 * CS_DEN) + np.sqrt(gamma / (K2 ** 2 * CS_DEN))
    return np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), np.nan)


def bar_order_imbalance(buy, sell) -> float:
    """Sum |buy - sell| / sum (buy + sell) over the given bars (true aggressor volumes); see module notes."""
    return _bar_imbalance(buy, sell)
