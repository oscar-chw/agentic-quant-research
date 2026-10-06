"""Microstructural features (AFML ch.19), causal, on the Up-frame tape and the nine bars.

First generation, price sequences (19.3):
- Roll (1984): trade-price bounce between bid and ask makes consecutive price changes
  negatively correlated; spread = 2·√max(0, −cov(Δp_t, Δp_{t−1})).
- Corwin–Schultz (2012): the high–low range of two bars contains variance ∝ time plus a
  spread term that does not scale with time; comparing one two-bar range with two one-bar
  ranges separates them.
Second generation, strategic trade (19.4):
- Kyle's λ: price impact per share of signed flow, the OLS slope of Δp on Σ b·v.
- Amihud's λ: mean |return| per dollar traded (here |Δlogit p| per risk dollar).
Third generation, sequential trade (19.5):
- VPIN: Σ|V_buy − V_sell| / Σ V over the last n bars. The book estimates buy volume from
  price changes; the Polymarket tape gives the true aggressor, so it is exact here.

Bar-based features are computed over the last `n` bars of the same window (a new window is
a new token) and are known when the bar that completes the sample is known.
"""
import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS as T
from pmlab.bars import Bar
from pmlab.features import logit

CS_K = 3 - 2 * np.sqrt(2)


def roll_spread(prices) -> float:
    dp = np.diff(np.asarray(prices, dtype=float))
    if len(dp) < 3:
        return np.nan
    c = np.cov(dp[1:], dp[:-1])[0, 1]
    return float(2 * np.sqrt(max(-c, 0.0)))


def corwin_schultz(h1, l1, h2, l2) -> float:
    """Spread in price units from two consecutive bars' highs and lows. Negative α is set to 0.

    The paper's S is relative to the price level; on a probability that makes one tick look
    20× wider at p = 0.05 than at 0.95 (research/reviews.md R1.4), so S is scaled back to price
    units by the geometric mean price (h1·l1·h2·l2)^¼."""
    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
    gamma = np.log(max(h1, h2) / min(l1, l2)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / CS_K - np.sqrt(gamma / CS_K)
    alpha = max(alpha, 0.0)
    relative = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(relative * (h1 * l1 * h2 * l2) ** 0.25)


def ols_slope(y, x) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.var(x) == 0:
        return np.nan
    return float(np.cov(y, x)[0, 1] / np.var(x, ddof=1))


def origin_slope(y, x) -> tuple[float, float]:
    """Slope and t-value of y = λ·x + ε without an intercept, as AFML 19.4.1 specifies Kyle's λ
    (ΔP = λ·signed volume). The t-value is the book's recommended companion (reliability of λ)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sxx = float(x @ x)
    if len(x) < 3 or sxx == 0:
        return np.nan, np.nan
    lam = float(x @ y) / sxx
    s2 = float(((y - lam * x) ** 2).sum()) / (len(x) - 1)
    return lam, (lam / np.sqrt(s2 / sxx) if s2 > 0 else np.nan)


def amihud(abs_ret, dollars) -> float:
    abs_ret, dollars = np.asarray(abs_ret, dtype=float), np.asarray(dollars, dtype=float)
    ok = dollars > 0
    return float(np.mean(abs_ret[ok] / dollars[ok])) if ok.any() else np.nan


def vpin(buy, sell) -> float:
    buy, sell = np.asarray(buy, dtype=float), np.asarray(sell, dtype=float)
    total = (buy + sell).sum()
    return float(np.abs(buy - sell).sum() / total) if total > 0 else np.nan


def _per_second(when: np.ndarray, values: dict[str, np.ndarray], start: int) -> dict[str, np.ndarray]:
    now = start + np.arange(T)
    if not len(when):
        return {k: np.full(T, np.nan) for k in values}
    idx = np.searchsorted(when, now, side="right") - 1
    ok, j = idx >= 0, np.maximum(idx, 0)
    return {k: np.where(ok, v[j], np.nan) if len(v) else np.full(T, np.nan) for k, v in values.items()}


def window_bar_stats(recent: list[Bar], prev_close: float | None = None) -> dict:
    """VPIN, Kyle's λ, Amihud's λ and Corwin–Schultz over `recent` bars of one window.

    The single definition used offline (bar_micro_features) and live (pmlab.live.engine), so the
    P-models see the same numbers in both (research/reviews.md R1.3). Price changes start from the
    previous bar's close, or from the first bar's open when `recent` starts the window.
    """
    base = recent[0].open if prev_close is None else prev_close
    closes = np.array([base] + [b.close for b in recent])
    hi = np.clip([b.high for b in recent], 1e-3, 1)
    lo = np.clip([b.low for b in recent], 1e-3, 1)
    cs = [corwin_schultz(hi[j - 1], lo[j - 1], hi[j], lo[j]) for j in range(1, len(recent))]
    kyle_book, kyle_t = origin_slope(np.diff(closes), [b.share_imbalance for b in recent])
    return {"vpin": vpin([b.buy_shares for b in recent], [b.sell_shares for b in recent]),
            # `kyle` keeps its original with-intercept definition: the frozen P models (confirmation) were fitted on it.
            # `kyle_book` is AFML 19.4.1's no-intercept λ with its t-value (research/book_check_part3.md finding 5).
            "kyle": ols_slope(np.diff(closes), [b.share_imbalance for b in recent]),
            "kyle_book": kyle_book, "kyle_t": kyle_t, "n_bars": len(recent),
            "amihud": amihud(np.abs(np.diff(logit(closes))), [b.risk_usd for b in recent]),
            "cs": float(np.mean(cs)) if cs else np.nan}


def bar_micro_features(kind: str, known: list[tuple[int, Bar]], start: int, n: int = 10) -> dict:
    """VPIN, Kyle's λ, Amihud's λ and Corwin–Schultz over the last n bars known at each second."""
    names = [f"{kind}_{m}" for m in ("vpin", "kyle", "amihud", "cs")]
    if not known:
        return {k: np.full(T, np.nan) for k in names}
    when = np.array([s for s, _ in known], dtype=float)
    bars = [b for _, b in known]
    out = {k: np.full(len(bars), np.nan) for k in names}
    for i in range(len(bars)):
        a = max(0, i - n + 1)
        st = window_bar_stats(bars[a:i + 1], bars[a - 1].close if a >= 1 else None)
        for m, k in zip(("vpin", "kyle", "amihud", "cs"), names):
            out[k][i] = st[m]
    return _per_second(when, out, start)


def roll_feature(tape: pd.DataFrame, start: int, n: int = 200) -> dict:
    """Roll spread over the last n trades known at each second (pre-window trades count)."""
    ts = tape["ts"].to_numpy(dtype=float)
    dp = pd.Series(np.diff(tape["p_up"].to_numpy(dtype=float), prepend=np.nan))
    cov = dp.rolling(n, min_periods=20).cov(dp.shift(1))
    spread = (2 * np.sqrt(np.clip(-cov, 0, None))).to_numpy()
    return _per_second(ts + 1, {"roll": spread}, start)
