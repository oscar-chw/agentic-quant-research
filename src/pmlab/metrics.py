"""Performance metrics for backtest returns, including Sharpe deflated for the number of trials.

Sharpe ratios here are per period, never annualised: PSR and DSR need the SR in the same
units as the number of observations n. Kurtosis is non-excess throughout (normal = 3).
Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
"""
import numpy as np
from scipy import stats
from scipy.stats import norm


def _finite(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[np.isfinite(r)]


def sharpe(returns) -> float:
    """Per-period mean / std (ddof=1) over the finite values."""
    r = _finite(returns)
    # a (nearly) constant series has no defined Sharpe: float noise in its std would give ~1e14
    # (a maker earning the same 1¢ every window; research/book_check_part3.md finding 2)
    if len(r) < 2 or np.allclose(r, r.mean(), rtol=1e-9, atol=1e-12):
        return np.nan
    return float(r.mean() / r.std(ddof=1))


def log_growth(bankroll_returns) -> float:
    """Mean log(1 + r): expected log utility per bet. Losing the whole bankroll is -inf."""
    r = np.asarray(bankroll_returns, dtype=float)
    return -np.inf if (r <= -1).any() else float(np.mean(np.log1p(r)))


def max_drawdown(equity_curve) -> float:
    """Largest peak-to-trough fall, as a positive fraction of the peak."""
    eq = np.asarray(equity_curve, dtype=float)
    return float(np.max(1 - eq / np.maximum.accumulate(eq)))


def probabilistic_sharpe_ratio(sr, n, skew, kurtosis, sr_benchmark=0.0) -> float:
    """P(true SR > sr_benchmark) for an SR estimated from n returns with these moments."""
    denom = np.sqrt(1 - skew * sr + (kurtosis - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials, var_sr) -> float:
    """E[max SR] across n_trials zero-skill trials whose SR estimates have variance var_sr."""
    if n_trials < 2:
        return 0.0  # a single trial involves no selection, so there is nothing to deflate
    g = np.euler_gamma
    return float(np.sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / n_trials)
                                    + g * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe_ratio(sr, n, skew, kurtosis, trial_srs) -> float:
    """PSR against the best SR that len(trial_srs) zero-skill trials would produce by luck."""
    var = np.var(trial_srs, ddof=1) if len(trial_srs) > 1 else 0.0
    sr_star = expected_max_sharpe(len(trial_srs), var)
    return probabilistic_sharpe_ratio(sr, n, skew, kurtosis, sr_star)


def summary(returns) -> dict:
    """Moments, Sharpe, PSR (benchmark 0), log growth and hit rate over the finite values."""
    r = _finite(returns)
    sr = sharpe(r)
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    return {"n": len(r), "mean": float(r.mean()), "std": float(r.std(ddof=1)), "sharpe": sr,
            "skew": skew, "kurtosis": kurt,
            "psr": probabilistic_sharpe_ratio(sr, len(r), skew, kurt),
            "log_growth": log_growth(r), "hit_rate": float((r > 0).mean())}
