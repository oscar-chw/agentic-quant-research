"""P and Q side by side: Q sensitivities and calibration, no-arbitrage, and P&L attribution.

Q is the pricing measure of pmlab.fair: under Q log BTC is a driftless random walk sampled each
second, and the Up token is worth p = Φ(m / sd), with m = E_Q[settling average] − k and
sd² = σ²·u(t) + basis². P is the real-world measure of pmlab.measure (TiltModel, GPResidual):
beliefs about the outcome that Kelly sizes on. docs/pq_framework.md maps the project onto both.

Conventions
- State. At second t the Q price depends on the Markov state (I_t, x_t): I_t is the sum of log
  prices already seen inside the settling average (x_t included), x_t is the current log price.
  τ = T − lag − t.
- Delta holds I_t fixed. a(t) = ∂m/∂x_t|I = 1 for τ ≥ L, τ/L for 0 < τ < L, 0 once τ ≤ 0
  (mean_slope). That is exactly how m_{t+1} responds to the BTC move between t and t+1, so it is
  the hedge ratio for the next second. (Bumping only the print x_t inside the sum would give
  (τ + 1)/L; that is not a hedgeable sensitivity and is not used.)
- Theta is a one-second forward difference p(t+1) − p(t) with the price path held flat
  (x_{t+1} = x_t), σ, basis and k fixed. The flat print enters the inside sum while τ drops by
  one, so m does not change in any regime: theta is pure variance decay, u(t) → u(t+1).
- Q-martingale: u(t) − u(t+1) = a(t)² exactly, so E_Q[p_{t+1} | F_t] = p_t exactly, and
  theta ≈ −½·σ²·gamma_log (tests/test_pq.py).
- sd = 0 (no σ, no basis, or settled with basis 0): p is the step 1{m ≥ 0}. Every derivative
  Greek is 0 there (the outcome is known; nothing to hedge), implied σ and the implied mean shift
  are NaN. Theta uses fair._prob and so is the step change when sd reaches 0 at t + 1.
- σ everywhere is the model σ fed to fair._prob (EWMA × vol_mult, floored), per second, log units.
- L is the settlement lookback of the market's rule (fair.rule_lookback: 60, 30, or 1 for the
  point price) and defaults to 60, today's rule; every function that depends on it takes it last.
Scalar in, float out; arrays broadcast.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from pmlab import WINDOW_SECONDS as T
from pmlab import fees
from pmlab.fair import L, _prob, variance_units


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def mean_slope(t, lag: int = 0, L: int = L):
    """a(t) = ∂m/∂x_t with the inside sum fixed: 1 for τ ≥ L, τ/L inside the average, 0 after."""
    return _out(np.clip(np.asarray(T - lag - t, dtype=float) / L, 0.0, 1.0))


def _greek_state(mean_minus_k, sigma, t, basis, lag, L):
    m, sigma, t, basis = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (mean_minus_k, sigma, t, basis)))
    u = variance_units(t, lag, L)
    sd = np.sqrt(sigma ** 2 * u + basis ** 2)
    ok = sd > 0
    sd_safe = np.where(ok, sd, 1.0)
    d = m / sd_safe
    pdf = np.where(ok, norm.pdf(d), 0.0)          # every Greek below carries this factor -> 0 when sd = 0
    return {"m": m, "sigma": sigma, "basis": basis, "u": u, "sd": sd_safe, "d": d, "pdf": pdf,
            "a": np.clip((T - lag - t) / L, 0.0, 1.0)}


def delta_log(mean_minus_k, sigma, t, basis=0.0, lag: int = 0, L: int = L):
    """∂p/∂x_t per unit of log price: φ(d)·a(t)/sd."""
    g = _greek_state(mean_minus_k, sigma, t, basis, lag, L)
    return _out(g["pdf"] * g["a"] / g["sd"])


def delta_usd(mean_minus_k, sigma, t, S, basis=0.0, lag: int = 0, L: int = L):
    """∂p/∂S per $1 of BTC: delta_log / S. Hedge n Up shares with −n·delta_usd BTC."""
    return _out(np.asarray(delta_log(mean_minus_k, sigma, t, basis, lag, L)) / np.asarray(S, dtype=float))


def gamma_log(mean_minus_k, sigma, t, basis=0.0, lag: int = 0, L: int = L):
    """∂²p/∂x_t²: −d·φ(d)·a²/sd²."""
    g = _greek_state(mean_minus_k, sigma, t, basis, lag, L)
    return _out(-g["d"] * g["pdf"] * g["a"] ** 2 / g["sd"] ** 2)


def gamma_usd(mean_minus_k, sigma, t, S, basis=0.0, lag: int = 0, L: int = L):
    """∂²p/∂S²: (gamma_log − delta_log) / S²."""
    S = np.asarray(S, dtype=float)
    return _out((np.asarray(gamma_log(mean_minus_k, sigma, t, basis, lag, L))
                 - np.asarray(delta_log(mean_minus_k, sigma, t, basis, lag, L))) / S ** 2)


def vega(mean_minus_k, sigma, t, basis=0.0, lag: int = 0, L: int = L):
    """∂p/∂σ (σ per second, log units): −d·φ(d)·σ·u/sd²."""
    g = _greek_state(mean_minus_k, sigma, t, basis, lag, L)
    return _out(-g["d"] * g["pdf"] * g["sigma"] * g["u"] / g["sd"] ** 2)


def basis_sens(mean_minus_k, sigma, t, basis=0.0, lag: int = 0, L: int = L):
    """∂p/∂basis: −d·φ(d)·basis/sd²."""
    g = _greek_state(mean_minus_k, sigma, t, basis, lag, L)
    return _out(-g["d"] * g["pdf"] * g["basis"] / g["sd"] ** 2)


def theta(mean_minus_k, sigma, t, basis=0.0, lag: int = 0, L: int = L):
    """p(t+1) − p(t) per second with the price path flat (m unchanged; see module conventions)."""
    t = np.asarray(t, dtype=float)
    return _out(_prob(mean_minus_k, sigma, variance_units(t + 1, lag, L), basis)
                - _prob(mean_minus_k, sigma, variance_units(t, lag, L), basis))


# ---------------------------------------------------------------- Q calibrated to the market

def implied_sigma(price, mean_minus_k, units, basis=0.0):
    """σ such that fair._prob(m, σ, units, basis) = price: √((m/Φ⁻¹(p))² − basis²)/√units.

    NaN when price ∉ (0, 1); when sign(m) ≠ sign(Φ⁻¹(p)) (the market leans against the spot,
    which no σ explains: that is drift, see implied_mean_shift); when p = 0.5 (m ≠ 0 needs
    σ = ∞, m = 0 fits every σ); when units = 0; or when (m/Φ⁻¹(p))² < basis² (the market is more
    certain than the basis alone allows)."""
    p, m, u, b = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, mean_minus_k, units, basis)))
    inside = (p > 0) & (p < 1)
    z = norm.ppf(np.where(inside, p, 0.5))
    ok = inside & (z != 0) & (np.sign(m) == np.sign(z)) & (u > 0)
    var = (m / np.where(ok, z, 1.0)) ** 2 - b ** 2
    ok &= var >= 0
    return _out(np.where(ok, np.sqrt(np.where(ok, var, 0.0) / np.where(ok, u, 1.0)), np.nan))


def implied_mean_shift(price, sigma, units, basis, mean_minus_k):
    """Girsanov shift Δm = Φ⁻¹(p)·sd − m that makes Q's price equal the market at the model σ.

    Under the market-implied measure the settling average is N(m + Δm, sd²) instead of
    N(m, sd²): Δm > 0 means the market prices in upward drift (or a premium for Up).
    NaN when price ∉ (0, 1) or sd = 0."""
    p, s, u, b, m = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, sigma, units, basis, mean_minus_k)))
    sd = np.sqrt(s ** 2 * u + b ** 2)
    ok = (p > 0) & (p < 1) & (sd > 0)
    return _out(np.where(ok, norm.ppf(np.where(ok, p, 0.5)) * sd - m, np.nan))


# ---------------------------------------------------------------- no-arbitrage on the two books

def arbitrage(up_book, down_book, fee_rate: float = fees.FEE_RATE) -> dict:
    """Pair trades against the top of both books (pmlab.live.feeds.OrderBook or anything with
    best_bid()/best_ask() -> (price, size) | None).

    buy_edge  = 1 − ask_up − ask_down − fees: buy one of each, merge the pair for $1.
    sell_edge = bid_up + bid_down − 1 − fees: mint a pair for $1, sell one of each.
    Fees are the unrounded taker fee fee_rate·p·(1 − p) per share on each leg; minting and
    merging are free. Sizes are the smaller top-of-book size of the two legs. NaN edge and 0
    size when a leg is missing.

    Polymarket shows every order in both books (an Up bid at p is the Down ask at 1 − p), so on
    consistent books buy_edge = sell_edge = −(Up spread) − fees < 0. A positive edge only comes
    from a crossed book or one side being stale (feeds.OrderBook documents late price changes);
    treat it as a data problem before treating it as money."""
    fee = lambda p: fees.fee_per_share(p, fee_rate)
    out = {"buy_edge": float("nan"), "buy_size": 0.0, "sell_edge": float("nan"), "sell_size": 0.0}
    au, ad = up_book.best_ask(), down_book.best_ask()
    if au and ad:
        out["buy_edge"] = 1.0 - au[0] - ad[0] - fee(au[0]) - fee(ad[0])
        out["buy_size"] = float(min(au[1], ad[1]))
    bu, bd = up_book.best_bid(), down_book.best_bid()
    if bu and bd:
        out["sell_edge"] = bu[0] + bd[0] - 1.0 - fee(bu[0]) - fee(bd[0])
        out["sell_size"] = float(min(bu[1], bd[1]))
    return out


# ---------------------------------------------------------------- P vs Q vs market attribution

def pq_decomposition(outcome_y, price_paid, fee_per_share, p_belief, q_fair) -> dict:
    """Per-share P&L of a long token position, split by measure. All inputs for the token held.

    total = y − price − fee = luck + model_edge + market_mispricing + cost, exactly, where
    luck = y − P (zero mean under P if P is right), model_edge = P − Q (what the P model adds to
    the pricing model), market_mispricing = Q − price (what Q alone says the fill was worth),
    cost = −fee."""
    y, price, fee, p, q = (np.asarray(v, dtype=float) for v in (outcome_y, price_paid, fee_per_share, p_belief, q_fair))
    return {"luck": _out(y - p), "model_edge": _out(p - q), "market_mispricing": _out(q - price),
            "cost": _out(-fee), "total": _out(y - price - fee)}


PARTS = ("luck", "model_edge", "market_mispricing", "cost", "total")


def decompose_fills(fills: pd.DataFrame, y: str = "y", p_up: str = "p_up", q_up: str = "q_up") -> pd.DataFrame:
    """pq_decomposition for every fill; columns as pmlab.backtest.Fill (side, shares, price, fee).

    `y`, `p_up`, `q_up` name columns in the Up frame (outcome, P belief, Q fair); Down fills are
    flipped: y_down = 1 − y, belief 1 − p, fair 1 − q. Adds the five per-share parts and the same
    in dollars (`<part>_usd` = part × shares); total_usd sums to backtest.settle's pnl."""
    side = fills["side"].to_numpy()
    if not np.isin(side, ("up", "down")).all():
        raise ValueError(f"side must be 'up' or 'down', got {sorted(set(side) - {'up', 'down'})}")
    up = side == "up"
    flip = lambda col: np.where(up, fills[col].to_numpy(dtype=float), 1.0 - fills[col].to_numpy(dtype=float))
    shares = fills["shares"].to_numpy(dtype=float)
    parts = pq_decomposition(flip(y), fills["price"].to_numpy(dtype=float), fills["fee"].to_numpy(dtype=float) / shares,
                             flip(p_up), flip(q_up))
    out = fills.copy()
    for k in PARTS:
        out[k] = parts[k]
        out[f"{k}_usd"] = parts[k] * shares
    return out


# ---------------------------------------------------------------- delta-hedged P&L

def hedged_pnl(shares_up: float, fair_path, delta_usd_path, S_path, payoff: float, rebalance_every: int = 1) -> dict:
    """Hold `shares_up` Up-frame shares (negative = long Down) from the first second of the paths
    to settlement, hedged with −shares_up·delta_usd BTC reset every `rebalance_every` seconds.

    Paths are aligned per second from entry to the last hedge second: fair p_i, delta_usd_i
    (from the same m, σ, t as p_i) and the BTC price S_i the hedge trades at. The hedge set at
    second j is held over [i, i+1] for j ≤ i < j + k and closed at the last S; the option goes
    from p_0 to `payoff` (the settled y, 1 or 0).

    option = shares·(payoff − p_0)         unhedged P&L marked from the Q fair at entry
    hedge  = −Σ shares·Δ_held·(S_{i+1} − S_i)
    residual = Σ per-second book P&L (steps.total), which must equal option + hedge: what is left
    once BTC direction is hedged out (discrete gamma/theta, σ error, basis, jumps, the unhedged
    step from the last mark to settlement). Under Q with the right σ it has mean 0."""
    p = np.asarray(fair_path, dtype=float)
    delta = np.asarray(delta_usd_path, dtype=float)
    S = np.asarray(S_path, dtype=float)
    k = max(int(rebalance_every), 1)
    held = np.zeros(len(p))
    held[: len(p) - 1] = -shares_up * delta[(np.arange(len(p) - 1) // k) * k]   # BTC held over [i, i+1]
    step_hedge = np.append(held[:-1] * np.diff(S), 0.0)
    step_option = shares_up * np.diff(np.append(p, payoff))                     # last step: p_n−1 → payoff
    steps = pd.DataFrame({"fair": p, "S": S, "held_btc": held, "option": step_option, "hedge": step_hedge,
                          "total": step_option + step_hedge})
    return {"option": float(shares_up * (payoff - p[0])),
            "hedge": float(-shares_up * np.sum(delta[(np.arange(len(p) - 1) // k) * k] * np.diff(S))),
            "residual": float(steps["total"].sum()), "steps": steps}
