"""Backtesting on synthetic data: optimal trading rules under an Ornstein-Uhlenbeck process (AFML ch.13).

- Discrete O-U (13.5.1): P_t = (1 - phi) E + phi P_{t-1} + sigma e_t, half-life tau = -ln 2 / ln phi,
  so phi = 2^(-1/tau).
- ou_fit: phi from the regression of P_t - E on P_{t-1} - E through the origin, sigma from the residuals
  (13.5.1). With E unknown and constant, ou_fit_constant regresses P_t on P_{t-1} with an intercept:
  phi = slope, E = intercept / (1 - slope).
- otr_mesh: the optimal trading rule search (13.5.2, 13.6): a position opened at seed 0 is closed when
  its gain exceeds the profit-taking barrier, falls below minus the stop-loss barrier, or the holding
  period passes max_hp steps; the Sharpe ratio of the closing gains is recorded for every
  (profit-taking, stop-loss) pair of the mesh. Barriers are in units of sigma.
- otr_mesh_loop: the same rule path by path, for testing.

Deviation (docs/afml_sections/ch13.md F13.1): Eq. 13.7 writes phi as cov(Y, X) / cov(X, X), and the
covariance operator subtracts the sample means of X and Y, which is a regression WITH an intercept.
ou_fit regresses through the origin instead (sum x y / sum x x), taking the forecast E at face value
rather than re-centring on the sample. On a short path entered away from E the two differ, and the
through-origin form is the less biased one there, since the sample mean of a short O-U path is a poor
estimate of E; ou_fit_constant is the with-intercept form for the case where E is unknown.

Deviation: otr_mesh uses common random numbers (one set of simulated paths for every mesh node), where
the book draws fresh paths per node. Every node's estimate is unbiased either way; common numbers make
differences between nodes less noisy.
"""
import numpy as np


def phi_from_half_life(tau: float) -> float:
    return float(2.0 ** (-1.0 / tau))


def half_life(phi: float) -> float:
    return float(-np.log(2.0) / np.log(phi)) if 0 < phi < 1 else np.inf


def ou_fit(prev, nxt, forecast=0.0) -> dict:
    prev, nxt = np.asarray(prev, dtype=float), np.asarray(nxt, dtype=float)
    x = prev - forecast
    y = nxt - forecast
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    phi = float(x @ y / (x @ x))
    resid = y - phi * x
    sigma = float(np.sqrt(resid @ resid / (len(x) - 1)))
    se = float(sigma / np.sqrt(x @ x))
    return {"phi": phi, "phi_se": se, "sigma": sigma, "half_life": half_life(phi), "n": int(len(x))}


def ou_fit_constant(prev, nxt) -> dict:
    prev, nxt = np.asarray(prev, dtype=float), np.asarray(nxt, dtype=float)
    ok = np.isfinite(prev) & np.isfinite(nxt)
    x, y = prev[ok], nxt[ok]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = float(np.sqrt(resid @ resid / (len(x) - 2)))
    se = float(sigma * np.sqrt(np.linalg.inv(X.T @ X)[1, 1]))
    phi = float(beta[1])
    return {"phi": phi, "phi_se": se, "mean": float(beta[0] / (1 - phi)) if phi != 1 else np.nan,
            "sigma": sigma, "half_life": half_life(phi), "n": int(len(x))}


def simulate_gains(forecast: float, phi: float, sigma: float = 1.0, n_iter: int = 10_000, max_hp: int = 100,
                   seed: float = 0.0, rng=None, eps: np.ndarray | None = None) -> np.ndarray:
    """(n_iter, max_hp + 1) gains P_h - seed for h = 1 .. max_hp + 1 (shocks drawn unless given)."""
    if eps is None:
        eps = np.random.default_rng(rng).standard_normal((n_iter, max_hp + 1))
    n_iter, max_hp = eps.shape[0], eps.shape[1] - 1
    g = np.empty_like(eps)
    p = np.full(n_iter, float(seed))
    for h in range(max_hp + 1):
        p = (1 - phi) * forecast + phi * p + sigma * eps[:, h]
        g[:, h] = p - seed
    return g


def _first(hit: np.ndarray) -> np.ndarray:
    H = hit.shape[1]
    return np.where(hit.any(axis=1), hit.argmax(axis=1), H)


def otr_from_gains(gains: np.ndarray, pt_grid, sl_grid) -> dict:
    """Closing gain statistics for every (pt, sl): exit at the first step with gain > pt or gain < -sl,
    else at the last step (holding period max_hp + 1, where the book's counter passes max_hp)."""
    gains = np.asarray(gains, dtype=float)
    n, H = gains.shape
    pt_grid, sl_grid = np.asarray(pt_grid, dtype=float), np.asarray(sl_grid, dtype=float)
    first_pt = np.stack([_first(gains > pt) for pt in pt_grid])        # (len pt, n)
    first_sl = np.stack([_first(gains < -sl) for sl in sl_grid])
    mean = np.empty((len(pt_grid), len(sl_grid)))
    std = np.empty_like(mean)
    hold = np.empty_like(mean)
    rows = np.arange(n)
    for i in range(len(pt_grid)):
        for j in range(len(sl_grid)):
            exit_ = np.minimum(np.minimum(first_pt[i], first_sl[j]), H - 1)
            out = gains[rows, exit_]
            mean[i, j], std[i, j], hold[i, j] = out.mean(), out.std(ddof=1), exit_.mean() + 1
    sharpe = np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 0)
    return {"pt": pt_grid, "sl": sl_grid, "mean": mean, "std": std, "sharpe": sharpe, "holding": hold}


def otr_mesh(forecast: float, phi: float, sigma: float = 1.0, pt_grid=np.linspace(0, 10, 21),
             sl_grid=np.linspace(0, 10, 21), n_iter: int = 10_000, max_hp: int = 100, rng=None) -> dict:
    gains = simulate_gains(forecast, phi, sigma, n_iter, max_hp, 0.0, rng)
    res = otr_from_gains(gains, np.asarray(pt_grid) * sigma, np.asarray(sl_grid) * sigma)
    res["pt"], res["sl"] = np.asarray(pt_grid, dtype=float), np.asarray(sl_grid, dtype=float)
    i, j = np.unravel_index(np.nanargmax(res["sharpe"]), res["sharpe"].shape)
    res["best"] = {"pt": float(res["pt"][i]), "sl": float(res["sl"][j]), "sharpe": float(res["sharpe"][i, j]),
                   "mean": float(res["mean"][i, j]), "holding": float(res["holding"][i, j])}
    return res


def otr_mesh_loop(eps: np.ndarray, forecast: float, phi: float, sigma: float, pt: float, sl: float,
                  max_hp: int) -> np.ndarray:
    """Closing gains path by path from given shocks eps (n_iter, max_hp + 1), following 13.5.2's loop."""
    out = []
    for row in eps:
        p, hp = 0.0, 0
        while True:
            p = (1 - phi) * forecast + phi * p + sigma * row[hp]
            gain = p - 0.0
            hp += 1
            if gain > pt or gain < -sl or hp > max_hp:
                out.append(gain)
                break
    return np.asarray(out)
