"""Strategy risk (AFML ch.15): how precision, bet frequency and payouts set a strategy's Sharpe ratio.

A strategy makes n independent bets per year; each wins pi_plus with probability p and loses
pi_minus (a negative number) otherwise.
- sharpe_symmetric: pi_plus = -pi_minus, theta = (2p - 1) sqrt(n) / (2 sqrt(p (1 - p))) (15.3).
- sharpe_asymmetric: theta = ((pi_plus - pi_minus) p + pi_minus) sqrt(n) /
  ((pi_plus - pi_minus) sqrt(p (1 - p))) (15.4).
- implied_precision: the p that gives a target theta, the root of the quadratic in 15.4.
- implied_frequency: the n that gives a target theta, 15.4, extraneous roots rejected as in Snippet 15.4.
- prob_failure: P[p < p*], p* the implied precision for the target Sharpe (15.4). Three distributions of
  the precision p:
  - scale="snippet": a normal with p (1 - p) as its standard deviation, as Snippet 15.5 is written;
  - scale="estimation_se": the standard error sqrt(p (1 - p) / T) of a precision estimated from the
    T observed bets;
  - scale="bootstrap": the 15.4.1 algorithm itself: draw floor(n k) bets with replacement (n bets a
    year, k the years investors judge over), take the precision of each draw, fit a Gaussian KDE and
    integrate it up to p*. The number of wins in floor(n k) draws with replacement is binomial, so the
    draws are sampled from that distribution directly.
  Neither of the first two is the 15.4.1 bootstrap.
- strategy_risk: everything above for one strategy's per-bet returns. The two-outcome model describes
  bets that pay a (near) fixed pi_plus or pi_minus. It has the sample mean by construction, so the
  payout mismatch |sd(returns) / ((pi_plus - pi_minus) sqrt(p (1 - p))) - 1| is also |Sharpe_binary /
  Sharpe_sample - 1|; above max_payout_mismatch (10%) the payouts are not two-point and every quantity
  that rests on the two-outcome variance (break-even and implied precision, implied frequency, the
  failure probabilities) is NaN.
- mix_gaussians: returns from a two-Gaussian mixture, the book's generator for its example (15.4.2).
"""
import numpy as np
from scipy.stats import norm


def sharpe_symmetric(p, n):
    p = np.asarray(p, dtype=float)
    return (2 * p - 1) * np.sqrt(n) / (2 * np.sqrt(p * (1 - p)))


def sharpe_asymmetric(p, n, pi_plus, pi_minus):
    p = np.asarray(p, dtype=float)
    spread = pi_plus - pi_minus
    return (spread * p + pi_minus) * np.sqrt(n) / (spread * np.sqrt(p * (1 - p)))


def implied_precision(pi_minus: float, pi_plus: float, n: float, theta: float) -> float:
    """Precision p at which n bets a year with payouts (pi_plus, pi_minus) reach annual Sharpe theta.
    The larger root; NaN when the target is out of reach."""
    spread = pi_plus - pi_minus
    a = (n + theta ** 2) * spread ** 2
    b = (2 * n * pi_minus - theta ** 2 * spread) * spread
    c = n * pi_minus ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.nan
    return float((-b + np.sqrt(disc)) / (2 * a))


def implied_frequency(pi_minus: float, pi_plus: float, p: float, theta: float) -> float:
    """Bets per year needed for annual Sharpe theta. Squaring theta admits a root with the wrong sign (a positive edge
    for a negative target, or the reverse); as Snippet 15.4 does, that root is rejected (NaN), as is a zero edge."""
    spread = pi_plus - pi_minus
    edge = spread * p + pi_minus
    if edge == 0:
        return np.nan
    freq = float((theta * spread) ** 2 * p * (1 - p) / edge ** 2)
    return freq if np.isclose(sharpe_asymmetric(p, freq, pi_plus, pi_minus), theta) else np.nan


SCALES = ("snippet", "estimation_se", "bootstrap")


def prob_failure(returns, bets_per_year: float, target_sharpe: float, scale: str = "snippet", years: float = 2.0,
                 n_boot: int = 10_000, rng=None) -> float:
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    pos, neg = r[r > 0], r[r <= 0]
    if not len(pos) or not len(neg):
        return np.nan
    p = len(pos) / len(r)
    p_star = implied_precision(neg.mean(), pos.mean(), bets_per_year, target_sharpe)
    if not np.isfinite(p_star):
        return 1.0
    if scale == "bootstrap":
        from scipy.stats import gaussian_kde
        draws = max(int(np.floor(bets_per_year * years)), 1)
        precision = np.random.default_rng(rng).binomial(draws, p, n_boot) / draws
        if np.ptp(precision) == 0:
            return float(precision[0] < p_star)
        return float(np.clip(gaussian_kde(precision).integrate_box_1d(-np.inf, p_star), 0.0, 1.0))
    sd = p * (1 - p) if scale == "snippet" else np.sqrt(p * (1 - p) / len(r))
    return float(norm.cdf(p_star, p, sd))


def strategy_risk(returns, bets_per_year: float, target_sharpe: float = 2.0, years: float = 2.0,
                  max_payout_mismatch: float = 0.10, n_boot: int = 10_000, rng=None) -> dict:
    """Everything chapter 15 reports for one strategy's per-bet returns; the two-outcome quantities only when
    the payouts are (near) two-point (module docstring)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    pos, neg = r[r > 0], r[r <= 0]
    out = {"bets": int(len(r)), "bets_per_year": float(bets_per_year), "target_sharpe": target_sharpe}
    if not len(pos) or not len(neg):
        return out | {"precision": float(len(pos) / len(r)) if len(r) else np.nan}
    p = len(pos) / len(r)
    pi_plus, pi_minus = float(pos.mean()), float(neg.mean())
    binary_sd = (pi_plus - pi_minus) * np.sqrt(p * (1 - p))
    mismatch = float(abs(r.std() / binary_sd - 1))
    two_point = bool(mismatch < max_payout_mismatch)
    out.update(precision=p, pi_plus=pi_plus, pi_minus=pi_minus,
               sharpe_binary=float(sharpe_asymmetric(p, bets_per_year, pi_plus, pi_minus)),
               sharpe_sample=float(r.mean() / r.std(ddof=1) * np.sqrt(bets_per_year)) if r.std(ddof=1) > 0 else np.nan,
               payout_mismatch=mismatch, two_point=two_point, max_payout_mismatch=max_payout_mismatch, years=years)
    nan = float("nan")
    if not two_point:
        return out | dict.fromkeys(("breakeven_precision", "implied_precision", "implied_frequency", "prob_failure_snippet",
                                    "prob_failure_estimation_se", "prob_failure_bootstrap"), nan)
    out.update(breakeven_precision=float(-pi_minus / (pi_plus - pi_minus)),
               implied_precision=implied_precision(pi_minus, pi_plus, bets_per_year, target_sharpe),
               implied_frequency=implied_frequency(pi_minus, pi_plus, p, target_sharpe),
               prob_failure_snippet=prob_failure(r, bets_per_year, target_sharpe, "snippet"),
               prob_failure_estimation_se=prob_failure(r, bets_per_year, target_sharpe, "estimation_se"),
               prob_failure_bootstrap=prob_failure(r, bets_per_year, target_sharpe, "bootstrap", years, n_boot, rng))
    return out


def mix_gaussians(mu1, mu2, sigma1, sigma2, prob1, n_obs, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    n1 = int(n_obs * prob1)
    r = np.concatenate([rng.normal(mu1, sigma1, n1), rng.normal(mu2, sigma2, n_obs - n1)])
    rng.shuffle(r)
    return r
