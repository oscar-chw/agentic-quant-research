"""Ensemble arithmetic (AFML ch.6).

- bagging_accuracy: 1 - P[X <= N / k] with X ~ Binomial(N, p), the book's Eq. 35 and Snippet 6.1 for N independent
  classifiers each right with probability p among k classes (6.3.1). More than N / k of them being right is
  *necessary* for the correct class to win a plurality vote, not sufficient: for k >= 3 the wrong votes can
  concentrate on one class and beat it (the book's own example, votes [4, 1, 5] with N = 10, k = 3 and class A
  observed, has X = 4 > 10 / 3 and still loses). So this is an upper bound on the accuracy of a plurality vote for
  k >= 3, and exactly the accuracy for k = 2, where the condition is also sufficient (docs/afml_sections/ch06.md
  F6.1). With p clearly above 1 / k it rises towards 1 as N grows; below it, it falls towards 0, so bagging cannot
  rescue classifiers that are worse than chance. The estimators are only independent when their bootstrap samples
  are, which overlapping labels prevent (4.5, 6.3.1).

- bagged_variance: the variance of the mean of N forecasts with average variance sigma^2 and average pairwise
  correlation rho, sigma^2 (rho + (1 - rho) / N) (6.3.1). It falls towards sigma^2 rho as N grows, so bagging buys a
  variance reduction only while rho < 1; at rho = 1 it is sigma^2 whatever N is. Figure 6.1 is its square root over
  N in [5, 30] and rho in [0, 1] at sigma = 1.
- estimators_for_accuracy: the book's N > p (p - 1 / k)^-2, the committee size above which P[X > N / k] exceeds the
  accuracy p of one classifier (6.3.2). It is sufficient, not necessary: smaller committees often already beat p.

The three random-forest set-ups of Snippet 6.2 are models, built by afml-pipeline node 11 (docs/afml_fidelity.md).
"""
import numpy as np
from scipy.stats import binom


def bagging_accuracy(n_estimators: int, p: float, k: int = 2) -> float:
    """P[X > N / k], X ~ Binomial(N, p): the plurality-vote accuracy for k = 2 and an upper bound on it above."""
    return float(1.0 - binom.cdf(np.floor(n_estimators / k), n_estimators, p))


def bagged_variance(sigma: float, rho: float, n_estimators: int) -> float:
    """sigma^2 (rho + (1 - rho) / N), the variance of the bagged forecast (6.3.1)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho is an average correlation in [0, 1]")
    if n_estimators < 1:
        raise ValueError("bagging needs at least one estimator")
    return float(sigma**2 * (rho + (1.0 - rho) / n_estimators))


def estimators_for_accuracy(p: float, k: int = 2) -> float:
    """p (p - 1 / k)^-2: above this many independent classifiers the vote is more accurate than one of them (6.3.2)."""
    if not 1.0 / k < p < 1.0:
        raise ValueError("the book's threshold needs 1 / k < p < 1")
    return float(p * (p - 1.0 / k) ** -2)


def average_variance_and_correlation(cov) -> tuple[float, float]:
    """The book's sigma_bar and rho_bar for a set of forecasts with covariance matrix `cov` (6.3.1).

    sigma_bar^2 is the mean of the estimators' variances, and rho_bar the sigma-weighted mean of their pairwise
    correlations, sum over i != j of sigma_i sigma_j rho_i,j divided by sigma_bar^2 N (N - 1). With these two,
    bagged_variance(sigma_bar, rho_bar, N) is exactly the double sum of the covariances over N^2 -- the identity the
    book's derivation rests on, which the plain (unweighted) mean correlation does not satisfy when the estimators'
    variances differ.
    """
    cov = np.asarray(cov, dtype=float)
    n = len(cov)
    if cov.ndim != 2 or cov.shape != (n, n) or n < 2:
        raise ValueError("cov is a square covariance matrix of at least two estimators")
    sigma_bar = float(np.sqrt(np.mean(np.diag(cov))))
    off_diagonal = float(cov.sum() - np.trace(cov))
    return sigma_bar, float(off_diagonal / (sigma_bar**2 * n * (n - 1)))
