"""The event dataset of the book's pipeline (plan afml-pipeline, stage 2): CUSUM-sampled decision events in Polymarket
BTC 5-minute Up/Down windows, one row per event and primary rule, with causal features, settlement labels and
meta-labels, label spans and sample weights (AFML 2.5.2(1), 3.2-3.6, 4.4-4.7, 7.4).

Availability, the rule every column obeys (pmlab.features, pmlab.backtest.View): the row of an event at second t of
window `start` uses only what is known at start + t.
- The BTC price AT instant i is the close of the 1 s kline that opened at i - 1; it is known at i.
- A Polymarket print stamped s happened somewhere in [s, s + 1), so it is known at s + 1: prints stamped <= start + t - 1.
- A bar closed by a print stamped s is known at s + 1.
- Day-level models are fitted on data that ended before the UTC day began: the fair-price parameters (the three days
  before, windows starting >= FAIR_PURGE s before the day, the day's own settlement rule when half a day of it
  exists), the bar calibration (the CAL_WINDOWS windows before the day)
  and the HAR + time-of-day volatility forecast (the HAR_TRAIN_DAYS days before, targets inside them).

Events (2.5.2(1)). A symmetric CUSUM filter per window on standardised BTC log returns z_i = r_i / sigma_{i-1}, where
sigma is the EWMA of squared 1 s returns (half-life CUSUM_HALFLIFE, floored at CUSUM_SIGMA_FLOOR) known before the
return, E_{i-1}[r_i] = 0. The sums restart at the window open (the first return inside the window is at t = 1) and an
event fires when either sum passes CUSUM_H sigma units; one return per second means at most one event per second.
Events are kept for T_MIN <= t <= T_MAX.

Primary rules at an event (one row per event x rule that has a side; +1 buys Up, -1 buys Down). Quotes are print
proxies (no historic book): the Up ask is the last buy-aggressor print, the Up bid the last sell-aggressor print, and
the Down ask is 1 - Up bid. A side's price is usable when finite, at most MAX_QUOTE_AGE s old and in
[PRICE_MIN, PRICE_MAX].
- q_taker: the side whose usable ask is below its fair price (fair - ask > 0), the larger gap when both are.
- fair_maker: the side pmlab.strategies.FairMaker (its class defaults: prob = fair, half-spread 0.03, no tuned
  parameter) would quote: bid_s = round(fair_s - 0.03, 2) within the price bounds and below the side's ask (a
  marketable bid is not posted). When both sides are quotable, the side whose bid sits closer to its ask
  (the larger fair_s - ask_s, i.e. sign(fair - mid)).
- flow: the sign of the last completed tick-imbalance bar's risk imbalance.
The price paid is the side's ask proxy plus the taker fee (pmlab.fees): cost = price + fee_per_share(price).

Labels (3.2-3.6). The only barrier is the vertical one at the window end: the token is held to settlement.
payoff = 1 if the side won; pnl = payoff - cost per share; label = 1 iff pnl > 0. The row already carries the
primary rule's side, so the meta-label (3.6: act on this primary side or not) is the same 0/1 and is kept under its own
name.

Spans and weights (4.4-4.7). Span [t0, t1) = [start + t, start + T): every label runs to the window end, so spans never
cross windows. Concurrency at a second counts every row (all rules) whose span covers it. Average uniqueness is the
mean of 1 / concurrency over the span; the return-attribution weight is |sum over the span of r_u / c_u| with r_u the
change of the Up fair price from second u to u + 1 and the settlement (0 or 1) as the price at T (a binary's log
return is not defined at 0, so price changes stand in for the book's log returns); time decay is the book's piecewise
linear decay on cumulative uniqueness in t0 order. sample_weights() recomputes all three on training rows only when a
split is given, and purged_kfold() purges and embargoes by span.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS as T
from pmlab import fair as fairmod
from pmlab import labels, pq, vol
from pmlab.afml import sampling
from pmlab.bars import NINE, Calibration, make_builder
from pmlab.evaluation import CONFIRM_START
from pmlab.events import trades_from_tape
from pmlab.features import base_features, bar_features
from pmlab.fees import fee_per_share
from pmlab.micro import roll_feature, window_bar_stats
from pmlab.polymarket import in_window
from pmlab.strategies import PRICE_MAX, PRICE_MIN, FairMaker

VERSION = 1
RULES = ("q_taker", "fair_maker", "flow")
T_MIN, T_MAX = 5, 280
CUSUM_H = 5.0                 # sigma units of 1 s BTC returns
CUSUM_HALFLIFE = 120.0
CUSUM_SIGMA_FLOOR = 1e-5
MAX_QUOTE_AGE = 5.0
MAKER_HALF_SPREAD = FairMaker.half_spread
FLOW_KIND = "tick_imbalance"
BAR_SIZE = 30                 # bars per window, pmlab.pipeline.build_features' default
MICRO_BARS = 10               # pmlab.micro.bar_micro_features' default
TW = 8 * 3600                 # Taiwan time, UTC+8
HISTORY = 14400 + 900         # BTC seconds read before a window start (HAR's 4 h realised variance)
AFTER = 60
FAIR_TRAIN_DAYS, FAIR_PURGE = 3, 600
EMBARGO_S = (FAIR_TRAIN_DAYS + 1) * 86400   # a day's features carry the outcomes of the FAIR_TRAIN_DAYS days before it
FAIR_MIN_SAME_RULE = 144      # windows of the day's own rule (half a day) needed to fit on that rule alone
FAIR_GRID = {"lags": range(0, 5), "halflives": (60.0, 120.0, 300.0),          # scripts/pricing_scoreboard.py
             "sigma_floors": (1e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5)}
HAR_TRAIN_DAYS, HAR_STEP = 7, 60
CAL_WINDOWS, WARMUP_WINDOWS = 288, 36
DECAY_C = 0.5
SESSIONS = ((0, "asia"), (8 * 3600, "europe"), (13 * 3600 + 1800, "us"), (20 * 3600, "off_hours"))   # UTC starts
RETURN_LAGS = (1, 5, 10, 30, 60)
VOLUME_SPANS = (10, 30, 60)
MICRO = ("imb", "vpin", "kyle", "amihud", "cs")
BOOK = ("book_spread", "book_depth_imb", "book_ask_size", "book_bid_size", "book_micro_minus_q")


def _feature_list() -> list[dict]:
    f = []

    def add(name, group, unit, definition):
        f.append({"name": f"f_{name}", "group": group, "unit": unit, "definition": definition})

    add("tau", "time", "s", "seconds left in the window, 300 - t")
    add("hour_sin", "time", "", "sin(2 pi h / 24), h = Taiwan (UTC+8) hour of day of the event second, fractional")
    add("hour_cos", "time", "", "cos(2 pi h / 24), same h")
    add("dow_sin", "time", "", "sin(2 pi d / 7), d = Taiwan weekday of the event second, Monday = 0")
    add("dow_cos", "time", "", "cos(2 pi d / 7), same d")
    add("session", "time", "category", "UTC session of the event second: 0 Asia 00:00-08:00 (Taiwan 08:00-16:00), "
        "1 Europe 08:00-13:30 (Taiwan 16:00-21:30), 2 US 13:30-20:00 (NYSE hours under US daylight time, the whole "
        "sample; Taiwan 21:30-04:00), 3 off hours 20:00-24:00 (Taiwan 04:00-08:00)")
    for k in RETURN_LAGS:
        add(f"btc_ret_{k}", "movement", "log", f"log BTC price at the event instant minus {k} s earlier (Binance 1 s closes)")
    add("z_q", "movement", "sd", "(settling-average estimate - strike) / its sd under Q: pmlab.features z_q with the "
        "day's walk-forward fair parameters and the window's rule lookback L")
    add("fair", "movement", "prob", "Q fair price of Up (pmlab.fair) with the day's walk-forward parameters")
    add("last_minus_q", "movement", "prob", "last print (Up frame, stamped <= t - 1) minus fair; NaN before the first print")
    add("mid_minus_q", "movement", "prob", "(Up ask proxy + Up bid proxy) / 2 minus fair; NaN when either proxy is missing")
    for k in (10, 30):
        add(f"last_minus_q_chg{k}", "movement", "prob", f"last_minus_q at t minus its value at t - {k}")
        add(f"mid_minus_q_chg{k}", "movement", "prob", f"mid_minus_q at t minus its value at t - {k}")
    add("spread_proxy", "movement", "prob", "Up ask proxy minus Up bid proxy (print proxies; can be negative)")
    add("last_age", "movement", "s", "seconds since the last print's stamp, >= 1")
    add("quote_age", "movement", "s", "max(ask proxy age, bid proxy age)")
    add("sigma_ewma", "volatility", "log per sqrt(s)", "fair model sigma: EWMA of squared 1 s BTC log returns (day's "
        "half-life) x vol_mult, floored at sigma_floor")
    for k in (60, 300):
        add(f"rv_{k}", "volatility", "log per sqrt(s)", f"sqrt(mean squared 1 s BTC log return over the last {k} s)")
    add("sigma_har", "volatility", "log per sqrt(s)", "pmlab.vol HAR + time-of-day forecast of the next 300 s "
        "(regressors log EWMA, log RV 30 min, log RV 4 h, log time-of-day profile), OLS fitted on the 7 days before the day")
    add("iv_ratio", "volatility", "ratio", "sigma implied by the print mid proxy (pmlab.pq.implied_sigma with the fair "
        "model's m, variance units and basis) / sigma_ewma, as the live engine computes it; NaN when undefined")
    for k in VOLUME_SPANS:
        add(f"shares_{k}", "volume", "shares", f"shares of prints stamped in [t - {k}, t - 1] (the market's whole tape)")
        add(f"usd_{k}", "volume", "USDC", "USDC of the same prints (price x size in the token traded)")
    add("buy_share_60", "volume", "share", "Up-buy-aggressor shares / all shares over the prints of the last 60 s; NaN if none")
    add("trade_rate_60", "volume", "prints per s", "prints stamped in [t - 60, t - 1] / 60")
    add("flow", "volume", "share", "window-to-date signed risk flow / risk (pmlab.features flow)")
    add("n_trades", "volume", "prints", "in-window prints known at t")
    for kind in NINE:
        add(f"{kind}_imb", "microstructure", "share", f"last {kind} bar's risk imbalance / risk (pmlab.features, 30 bars "
            "per window, calibrated on the 288 windows before the day)")
        add(f"{kind}_vpin", "microstructure", "share", f"sum |buy - sell| / sum shares over the last 10 {kind} bars (pmlab.micro)")
        add(f"{kind}_kyle", "microstructure", "prob per share", f"OLS slope of close change on share imbalance, last 10 {kind} bars (pmlab.micro kyle)")
        add(f"{kind}_amihud", "microstructure", "logit per risk USD", f"mean |logit close change| / risk USD, last 10 {kind} bars")
        add(f"{kind}_cs", "microstructure", "prob", f"Corwin-Schultz spread in price units, last 10 {kind} bars")
    add("roll", "microstructure", "prob", "Roll spread over the last 200 prints known at t (pmlab.micro roll_feature)")
    add("cusum_count", "regime", "events", "CUSUM events in the window at seconds 1..t, this one included")
    add("cusum_sign", "regime", "sign", "+1 if this event's upward sum fired, -1 if the downward one")
    add("rule_L", "regime", "s", "seconds averaged at each end by the window's settlement rule: 1 point price "
        "(to 2026-08-06), 30 (2026-08-07..13), 60 (from 2026-08-14)")
    add("book_spread", "book", "prob", "recorded best Up ask - best Up bid (live recorder); NaN without a recorded book")
    add("book_depth_imb", "book", "share", "(bid - ask) shares over the top 5 levels / their sum; NaN without a book")
    add("book_ask_size", "book", "shares", "shares at the best Up ask; NaN without a book")
    add("book_bid_size", "book", "shares", "shares at the best Up bid; NaN without a book")
    add("book_micro_minus_q", "book", "prob", "size-weighted microprice minus fair; NaN without a book")
    add("book", "book", "flag", "1 when the live recorder's book row exists for the event second, else 0")
    return f


FEATURES = _feature_list()
FEATURE_NAMES = [f["name"] for f in FEATURES]
LABEL_COLUMNS = ("up_won", "payoff", "pnl", "ret", "label", "meta_label")
WEIGHT_COLUMNS = ("avg_uniqueness", "w_return_raw")
COLUMNS = {
    "start": "window start, unix s (UTC)", "t": "event second in the window", "day": "UTC day of the window start",
    "day_tw": "Taiwan day of the window start", "rule": "primary rule", "side": "+1 buys Up, -1 buys Down",
    "ask_up": "Up ask print proxy", "bid_up": "Up bid print proxy", "price": "side's ask proxy",
    "price_age": "age of the side's ask proxy, s", "price_source": "print_proxy (no historic book)",
    "fee": "taker fee per share at price", "cost": "price + fee", "fair_side": "fair price of the side",
    "edge": "fair_side - cost", "maker_bid": "fair_maker's bid for the side, round(fair_side - 0.03, 2)",
    "book_price": "recorded best ask of the side where a book was recorded (not used by the label)",
    "up_won": "settlement: 1 if Up won", "payoff": "1 if the side won", "pnl": "payoff - cost per share",
    "ret": "pnl / cost", "label": "1 iff pnl > 0", "meta_label": "the same, for the primary rule's side",
    "t0": "span start = start + t", "t1": "span end (exclusive) = start + 300",
    "avg_uniqueness": "mean 1/concurrency over the span, all rows of the window", "w_return_raw":
    "|sum over the span of dq_u / concurrency_u|, dq the Up fair price change, settlement at 300 (unnormalised)"}


# ---------------------------------------------------------------------------------------------------- universe

def eligible_starts(starts, confirm_start: int = CONFIRM_START) -> list[int]:
    """Windows that ended by the confirmation start: nothing at or after it is ever read."""
    return sorted(int(s) for s in starts if int(s) + T <= confirm_start)


def fair_training_starts(inventory: pd.DataFrame, day0: int, rule_L: int) -> list[int]:
    """Windows the day's fair parameters are fitted on: the FAIR_TRAIN_DAYS days before `day0`, starting at least
    FAIR_PURGE s before it (their outcomes are settled before the day begins); the day's own rule only when that
    leaves at least FAIR_MIN_SAME_RULE windows (the basis differs by rule), else every rule. `inventory` has columns
    start, L."""
    s = inventory["start"].to_numpy()
    cand = inventory[(s >= day0 - FAIR_TRAIN_DAYS * 86400) & (s < day0 - FAIR_PURGE)]
    same = cand[cand["L"] == rule_L]
    use = same if len(same) >= FAIR_MIN_SAME_RULE else cand
    return sorted(int(x) for x in use["start"])


def calibration_starts(starts, day0: int) -> list[int]:
    """The CAL_WINDOWS latest windows that ended by `day0` (bar calibration; the last WARMUP_WINDOWS warm the builders)."""
    return sorted(int(s) for s in starts if int(s) + T <= day0)[-CAL_WINDOWS:]


# ---------------------------------------------------------------------------------------------------- day models

@dataclass
class DayContext:
    fair: fairmod.FairParams
    har: vol.HAR
    profile: np.ndarray
    meta: dict = field(default_factory=dict)


def fit_fair(closes: dict, outcomes: dict, lookbacks: dict) -> tuple[fairmod.FairParams, pd.DataFrame]:
    return fairmod.fit(closes, outcomes, grid=range(0, T, 5), lookbacks=lookbacks, **FAIR_GRID)


def fit_har(close: pd.Series, day0: int, halflife: float) -> tuple[vol.HAR, np.ndarray, dict]:
    """HAR + time-of-day on the HAR_TRAIN_DAYS days before `day0`: only klines opened before day0 - 1 (instants up to
    day0 - 1) are read, the profile uses those days' instants and every target lies inside them."""
    lo = day0 - HAR_TRAIN_DAYS * 86400
    c = close[(close.index >= lo - HISTORY) & (close.index < day0 - 1)]
    tl = vol.timeline(c)
    profile = vol.tod_profile(tl, tl.instants >= lo)
    reg = vol.regressors(tl, halflife, profile)
    target = np.log(vol.forward_mean_sq(tl.r, T) + vol.EPS)
    idx = np.flatnonzero((tl.instants >= lo) & (tl.instants % HAR_STEP == 0))
    har = vol.HAR(vol.HAR_SETS["seasonal"]).fit(reg, target, idx)
    meta = {"coef": dict(zip(["const"] + har.names, har.coef.tolist())), "resid_var": har.resid_var, "n": har.n,
            "first_instant": int(tl.instants[0]), "last_instant": int(tl.instants[-1])}
    return har, profile, meta


def make_builders(cal: Calibration) -> dict:
    return {k: make_builder(k, cal, BAR_SIZE) for k in NINE}


def warm_builders(builders: dict, windows: dict) -> None:
    """Run earlier windows' in-window trades {start: trades} through the builders, discarding their bars."""
    for s in sorted(windows):
        for b in builders.values():
            b.new_window(s)
            for tr in windows[s]:
                b.update(tr)


# ---------------------------------------------------------------------------------------------------- one window

def _close_slice(close: pd.Series, start: int) -> pd.Series:
    """1 s closes by open second over [start - HISTORY, start + T + AFTER), carried forward over missing seconds
    (as pmlab.fair and pmlab.vol do); leading seconds before the first known close are left out."""
    lo, hi = start - HISTORY, start + T + AFTER
    c = close[(close.index >= lo) & (close.index < hi)]
    c = c[~c.index.duplicated(keep="last")].sort_index().dropna()
    if c.empty or c.index.min() > start - 2:
        raise ValueError(f"window {start}: no BTC closes before its start")
    return c.reindex(np.arange(int(c.index.min()), hi)).ffill()


def _back(a: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    j = idx - k
    return np.where(j >= 0, a[np.maximum(j, 0)], np.nan)


def cusum_z(tl: vol.Timeline, start: int) -> np.ndarray:
    """z[t] = r / sigma before r for the returns at instants start + t, t = 0..T-1; z[0] = NaN (the return into the
    open belongs to the previous window)."""
    idx = tl.at(start + np.arange(T))
    var = fairmod.ewma_var(tl.r, CUSUM_HALFLIFE)
    sig_prev = np.sqrt(_back(var, idx, 1))
    z = tl.r[idx] / np.maximum(sig_prev, CUSUM_SIGMA_FLOOR)
    z[0] = np.nan
    return z


def cusum_events(z) -> tuple[np.ndarray, np.ndarray]:
    """(event seconds, their signs) of the symmetric CUSUM filter with threshold CUSUM_H on z (pmlab.labels)."""
    s = labels.cusum_filter(z, CUSUM_H)
    t = np.flatnonzero(s)
    return t, s[t].astype(np.int8)


def run_window_bars(builders: dict, start: int, trades: list) -> dict:
    """{kind: [(second the bar became known, bar)]}, as pmlab.features.run_bars for every kind from one trade list."""
    out = {}
    for kind, b in builders.items():
        b.new_window(start)
        known = []
        for tr in trades:
            bar = b.update(tr)
            if bar is not None:
                known.append((int(tr.ts) + 1, bar))
        out[kind] = known
    return out


def micro_at(kind: str, known: list, start: int, seconds, n: int = MICRO_BARS) -> dict:
    """pmlab.micro.bar_micro_features' four statistics evaluated only at `seconds` (in-window t)."""
    seconds = np.asarray(seconds, dtype=np.int64)
    out = {f"{kind}_{m}": np.full(len(seconds), np.nan) for m in ("vpin", "kyle", "amihud", "cs")}
    if not known or not len(seconds):
        return out
    when = np.array([s for s, _ in known], dtype=float)
    bars = [b for _, b in known]
    idx = np.searchsorted(when, start + seconds, side="right") - 1
    cache = {}
    for j, i in enumerate(idx):
        if i < 0:
            continue
        if i not in cache:
            a = max(0, i - n + 1)
            cache[i] = window_bar_stats(bars[a:i + 1], bars[a - 1].close if a >= 1 else None)
        for m in ("vpin", "kyle", "amihud", "cs"):
            out[f"{kind}_{m}"][j] = cache[i][m]
    return out


def _tape_windows(tape: pd.DataFrame, now: np.ndarray) -> dict:
    ts = tape["ts"].to_numpy(np.int64)
    sh = tape["shares"].to_numpy(float)
    usd = tape["usdc"].to_numpy(float)
    buy = np.where(tape["sign"].to_numpy() > 0, sh, 0.0)
    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in (("sh", sh), ("usd", usd), ("buy", buy))}
    hi = np.searchsorted(ts, now - 1, side="right")
    out = {}
    for k in VOLUME_SPANS:
        lo = np.searchsorted(ts, now - k, side="left")
        out[f"shares_{k}"] = cum["sh"][hi] - cum["sh"][lo]
        out[f"usd_{k}"] = cum["usd"][hi] - cum["usd"][lo]
        if k == 60:
            out["buy_share_60"] = np.divide(cum["buy"][hi] - cum["buy"][lo], out["shares_60"],
                                            out=np.full(len(now), np.nan), where=out["shares_60"] > 0)
            out["trade_rate_60"] = (hi - lo) / 60.0
    return out


def _session(ts: np.ndarray) -> np.ndarray:
    sod = ts % 86400
    out = np.zeros(len(ts))
    for i, (lo, _) in enumerate(SESSIONS):
        out[sod >= lo] = i
    return out


def _usable(price, age) -> np.ndarray:
    return np.isfinite(price) & (np.nan_to_num(age, nan=np.inf) <= MAX_QUOTE_AGE) & (price >= PRICE_MIN) & (price <= PRICE_MAX)


def rule_sides(fair, ask_up, ask_age, bid_up, bid_age, flow_imb) -> dict:
    """Side per rule (+1 Up, -1 Down, 0 none) at each event from the quantities known at it."""
    ask_dn = 1.0 - bid_up
    ok_up, ok_dn = _usable(ask_up, ask_age), _usable(ask_dn, bid_age)
    e_up = np.where(ok_up, fair - ask_up, np.nan)
    e_dn = np.where(ok_dn, (1.0 - fair) - ask_dn, np.nan)
    eu, ed = np.nan_to_num(e_up, nan=-np.inf), np.nan_to_num(e_dn, nan=-np.inf)
    q = np.where((eu > 0) & (eu >= ed), 1, np.where(ed > 0, -1, 0))

    with np.errstate(invalid="ignore"):
        bid_s_up = np.round(fair - MAKER_HALF_SPREAD, 2)
        bid_s_dn = np.round((1.0 - fair) - MAKER_HALF_SPREAD, 2)
    quote_up = np.isfinite(bid_s_up) & (bid_s_up >= PRICE_MIN) & (bid_s_up <= PRICE_MAX) & ~(np.isfinite(ask_up) & (bid_s_up >= ask_up))
    quote_dn = np.isfinite(bid_s_dn) & (bid_s_dn >= PRICE_MIN) & (bid_s_dn <= PRICE_MAX) & ~(np.isfinite(ask_dn) & (bid_s_dn >= ask_dn))
    gu = np.nan_to_num(fair - ask_up, nan=-np.inf)
    gd = np.nan_to_num((1.0 - fair) - ask_dn, nan=-np.inf)
    m = np.where(quote_up & quote_dn, np.where(gu >= gd, 1, -1), np.where(quote_up, 1, np.where(quote_dn, -1, 0)))

    f = np.sign(np.nan_to_num(flow_imb, nan=0.0)).astype(int)
    return {"q_taker": q.astype(np.int8), "fair_maker": m.astype(np.int8), "flow": f.astype(np.int8),
            "_maker_bid": (bid_s_up, bid_s_dn)}


def window_span_weights(t0, t1, q) -> tuple[np.ndarray, np.ndarray]:
    """(average uniqueness, |sum of r_u / c_u| over the span) for spans [t0, t1) inside one window (seconds 0..T),
    q[0..T] the Up price path, r_u = q[u + 1] - q[u]. Concurrency counts only the spans given."""
    t0 = np.asarray(t0, dtype=np.int64)
    t1 = np.asarray(t1, dtype=np.int64)
    if not len(t0):
        return np.zeros(0), np.zeros(0)
    if np.any(t1 <= t0) or t0.min() < 0 or t1.max() > T:
        raise ValueError("spans must satisfy 0 <= t0 < t1 <= T")
    edges = np.unique(np.concatenate([t0, t1]))
    lo, hi = np.searchsorted(edges, t0), np.searchsorted(edges, t1)
    d = np.zeros(len(edges))
    np.add.at(d, lo, 1)
    np.add.at(d, hi, -1)
    conc = np.cumsum(d)[:-1]
    inv = np.divide(1.0, conc, out=np.zeros(len(conc)), where=conc > 0)
    q = np.asarray(q, dtype=float)
    pu = np.concatenate([[0.0], np.cumsum(np.diff(edges) * inv)])
    pr = np.concatenate([[0.0], np.cumsum((q[edges[1:]] - q[edges[:-1]]) * inv)])
    return (pu[hi] - pu[lo]) / (t1 - t0), np.abs(pr[hi] - pr[lo])


def price_path(fair: np.ndarray, up_won: float) -> np.ndarray:
    """q[0..T]: the fair price per second (carried forward over gaps, 0.5 before the first) and the settlement at T."""
    q = pd.Series(np.asarray(fair, dtype=float)).ffill().fillna(0.5).to_numpy()
    return np.concatenate([q, [float(up_won)]])


def window_rows(start: int, tape: pd.DataFrame, close: pd.Series, L: int, up_won: float, ctx: DayContext,
                builders: dict, book: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Rows (event x rule) of one window, its price path q[0..T] and counts. `builders` carry the bar state from the
    windows before and are advanced through this one. `book`: the live recorder's rows of this window (column t)."""
    start = int(start)
    if start + T > CONFIRM_START:
        raise ValueError(f"window {start} ends after the confirmation start")
    tape = tape.sort_values("ts", kind="stable").reset_index(drop=True)
    now = start + np.arange(T)
    c = _close_slice(close, start)
    tl = vol.timeline(c)
    idx = tl.at(now)

    z = cusum_z(tl, start)
    ev_all, sign_all = cusum_events(z)
    count = np.cumsum(np.isin(np.arange(T), ev_all))
    keep = (ev_all >= T_MIN) & (ev_all <= T_MAX)
    ev, ev_sign = ev_all[keep], sign_all[keep]

    trades = trades_from_tape(in_window(tape), start)
    known = run_window_bars(builders, start, trades)
    base = base_features(start, tape, c, ctx.fair, L)
    q = price_path(base["fair"], up_won)
    stats = {"events": int(len(ev)), "events_all_t": int(len(ev_all)), **{f"rows_{r}": 0 for r in RULES},
             **{f"no_side_{r}": 0 for r in RULES}, **{f"stale_{r}": 0 for r in RULES}}
    if not len(ev):
        return pd.DataFrame(), q, stats

    e = {}
    x = tl.x
    for k in RETURN_LAGS:
        e[f"btc_ret_{k}"] = (x[idx] - _back(x, idx, k))[ev]
    for k in (60, 300):
        e[f"rv_{k}"] = np.sqrt(vol.trailing_mean_sq(tl.r, k)[idx])[ev]
    e["sigma_har"] = ctx.har.sigma(vol.regressors(tl, ctx.fair.halflife, ctx.profile), idx[ev])
    fair, sigma = base["fair"], base["sigma"]
    mid = (base["ask"] + base["bid"]) / 2
    last_q, mid_q = base["last"] - fair, mid - fair
    for k in (10, 30):
        e[f"last_minus_q_chg{k}"] = (last_q - _back(last_q, np.arange(T), k))[ev]
        e[f"mid_minus_q_chg{k}"] = (mid_q - _back(mid_q, np.arange(T), k))[ev]
    e.update(tau=(T - ev).astype(float), fair=fair[ev], z_q=base["z_q"][ev], last_minus_q=last_q[ev],
             mid_minus_q=mid_q[ev], spread_proxy=(base["ask"] - base["bid"])[ev], last_age=base["last_age"][ev],
             quote_age=np.maximum(base["ask_age"], base["bid_age"])[ev], sigma_ewma=sigma[ev],
             flow=base["flow"][ev], n_trades=base["n_trades"][ev], roll=roll_feature(tape, start)["roll"][ev])
    lag, basis = ctx.fair.lag, ctx.fair.basis
    units = fairmod.variance_units(ev, lag, L)
    m = base["z_q"][ev] * np.sqrt(sigma[ev] ** 2 * np.maximum(units, 1e-12) + basis ** 2)
    e["iv_ratio"] = pq.implied_sigma(mid[ev], m, units, basis) / sigma[ev]
    e.update({k: v[ev] for k, v in _tape_windows(tape, now).items()})
    for kind in NINE:
        e[f"{kind}_imb"] = bar_features(kind, known[kind], start)[f"{kind}_imb"][ev]
        e.update(micro_at(kind, known[kind], start, ev))
    ts = start + ev
    tw = ts + TW
    hour = (tw % 86400) / 3600.0
    dow = ((tw // 86400) + 3) % 7
    e.update(hour_sin=np.sin(2 * np.pi * hour / 24), hour_cos=np.cos(2 * np.pi * hour / 24),
             dow_sin=np.sin(2 * np.pi * dow / 7), dow_cos=np.cos(2 * np.pi * dow / 7), session=_session(ts),
             cusum_count=count[ev].astype(float), cusum_sign=ev_sign.astype(float), rule_L=np.full(len(ev), float(L)))
    b = None
    if book is not None and len(book):
        b = book.drop_duplicates("t", keep="last").set_index("t").reindex(ev)
    has_book = np.zeros(len(ev), bool) if b is None else np.isfinite(b["spread"].to_numpy(float))
    for col, src in (("book_spread", "spread"), ("book_depth_imb", "depth_imb"), ("book_ask_size", "ask_size"),
                     ("book_bid_size", "bid_size"), ("book_micro_minus_q", "microprice")):
        v = np.full(len(ev), np.nan) if b is None else b[src].to_numpy(float)
        e[col] = np.where(has_book, v - (fair[ev] if col == "book_micro_minus_q" else 0.0), np.nan)
    e["book"] = has_book.astype(float)
    feats = pd.DataFrame({n: np.asarray(e[n[2:]], dtype=np.float32) for n in FEATURE_NAMES})

    ask_up, bid_up = base["ask"][ev], base["bid"][ev]
    ask_age, bid_age = base["ask_age"][ev], base["bid_age"][ev]
    sides = rule_sides(fair[ev], ask_up, ask_age, bid_up, bid_age, e[f"{FLOW_KIND}_imb"])
    book_up = np.full(len(ev), np.nan) if b is None else b["ask"].to_numpy(float)
    book_dn = np.full(len(ev), np.nan) if b is None else 1.0 - b["bid"].to_numpy(float)
    parts = []
    for rule in RULES:
        side = sides[rule]
        price = np.where(side > 0, ask_up, 1.0 - bid_up)
        age = np.where(side > 0, ask_age, bid_age)
        ok = (side != 0) & _usable(price, age)
        stats[f"no_side_{rule}"] = int((side == 0).sum())
        stats[f"stale_{rule}"] = int(((side != 0) & ~ok).sum())
        stats[f"rows_{rule}"] = int(ok.sum())
        if not ok.any():
            continue
        s = side[ok].astype(np.int8)
        p = price[ok]
        fee = fee_per_share(p)
        fair_side = np.where(s > 0, fair[ev][ok], 1.0 - fair[ev][ok])
        payoff = np.where(s > 0, float(up_won), 1.0 - float(up_won))
        pnl = payoff - p - fee
        maker = np.where(s > 0, sides["_maker_bid"][0][ok], sides["_maker_bid"][1][ok]) if rule == "fair_maker" \
            else np.full(len(s), np.nan)
        rows = pd.DataFrame({
            "start": np.full(len(s), start, np.int64), "t": ev[ok].astype(np.int32), "rule": rule, "side": s,
            "ask_up": ask_up[ok], "bid_up": bid_up[ok], "price": p, "price_age": age[ok], "price_source": "print_proxy",
            "fee": fee, "cost": p + fee, "fair_side": fair_side, "edge": fair_side - p - fee, "maker_bid": maker,
            "book_price": np.where(s > 0, book_up[ok], book_dn[ok]),
            "up_won": np.full(len(s), float(up_won)), "payoff": payoff, "pnl": pnl, "ret": pnl / (p + fee),
            "label": (pnl > 0).astype(np.int8), "meta_label": (pnl > 0).astype(np.int8),
            "t0": start + ev[ok].astype(np.int64), "t1": np.full(len(s), start + T, np.int64)})
        parts.append(pd.concat([rows, feats[ok].reset_index(drop=True)], axis=1))
    if not parts:
        return pd.DataFrame(), q, stats
    out = pd.concat(parts, ignore_index=True)
    out["_r"] = out["rule"].map({r: i for i, r in enumerate(RULES)})
    out = out.sort_values(["t", "_r"], kind="stable").drop(columns="_r").reset_index(drop=True)
    u, w = window_span_weights(out["t0"] - start, out["t1"] - start, q)
    out.insert(out.columns.get_loc("t1") + 1, "avg_uniqueness", u)
    out.insert(out.columns.get_loc("avg_uniqueness") + 1, "w_return_raw", w)
    return out, q, stats


# ---------------------------------------------------------------------------------------------------- weights, CV

def _paths_lookup(paths) -> dict:
    if isinstance(paths, dict):
        return paths
    p = paths.sort_values(["start", "t"], kind="stable")
    starts = p["start"].to_numpy()[:: T + 1]
    return dict(zip(starts.tolist(), p["q"].to_numpy(float).reshape(-1, T + 1)))


def span_weights(events: pd.DataFrame, paths) -> tuple[np.ndarray, np.ndarray]:
    """Average uniqueness and unnormalised return attribution of every row, concurrency among these rows only."""
    paths = _paths_lookup(paths)
    u, w = np.full(len(events), np.nan), np.full(len(events), np.nan)
    st = events["start"].to_numpy(np.int64)
    t0, t1 = events["t0"].to_numpy(np.int64), events["t1"].to_numpy(np.int64)
    order = np.argsort(st, kind="stable")
    bounds = np.flatnonzero(np.diff(st[order])) + 1
    for grp in np.split(order, bounds):
        if not len(grp):
            continue
        s = int(st[grp[0]])
        u[grp], w[grp] = window_span_weights(t0[grp] - s, t1[grp] - s, paths[s])
    return u, w


def sample_weights(events: pd.DataFrame, paths, train=None, decay_c: float = DECAY_C) -> pd.DataFrame:
    """avg_uniqueness, w_return (return attribution scaled to sum to the rows used), w_decay (time decay on cumulative
    uniqueness in t0 order) and weight = w_return x w_decay (4.6-4.7). With `train` (boolean mask or positions) every
    quantity is computed from the training rows alone; the other rows get NaN."""
    mask = np.ones(len(events), bool)
    if train is not None:
        train = np.asarray(train)
        mask = train.astype(bool) if train.dtype == bool else np.isin(np.arange(len(events)), train)
    sub = events[mask]
    out = pd.DataFrame(np.nan, index=events.index, columns=["avg_uniqueness", "w_return", "w_decay", "weight"])
    if not len(sub):
        return out
    u, raw = span_weights(sub, paths)
    w_ret = raw * len(raw) / raw.sum() if raw.sum() > 0 else np.ones(len(raw))
    w_dec = sampling.time_decay(u, order=sub["t0"].to_numpy(), c=decay_c)
    out.loc[mask, "avg_uniqueness"] = u
    out.loc[mask, "w_return"] = w_ret
    out.loc[mask, "w_decay"] = w_dec
    out.loc[mask, "weight"] = w_ret * w_dec
    return out


def _merge_loop(t0, t1) -> tuple[np.ndarray, np.ndarray]:
    """The plain sweep over sorted spans, kept as the reference `_merge` is checked against."""
    o = np.argsort(t0, kind="stable")
    a, b = np.asarray(t0)[o], np.asarray(t1)[o]
    lo, hi = [], []
    for x, y in zip(a, b):
        if lo and x <= hi[-1]:
            hi[-1] = max(hi[-1], y)
        else:
            lo.append(x)
            hi.append(y)
    return np.array(lo, dtype=np.int64), np.array(hi, dtype=np.int64)


def _merge(t0, t1) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping spans merged into maximal stretches, in start order (purged_kfold's test blocks).

    Vectorised: with the spans sorted by start, a stretch ends wherever the next start is past the running maximum of
    the ends so far, and each stretch's end is the maximum of the ends inside it. That running maximum is the same
    number the sweep carries in `hi[-1]`: starts are non-decreasing, so a start that clears the *current* stretch's
    maximum has already cleared every earlier one. Integers, and the same two comparisons, so the result is identical
    to `_merge_loop` rather than merely close.
    """
    o = np.argsort(t0, kind="stable")
    a, b = np.asarray(t0, dtype=np.int64)[o], np.asarray(t1, dtype=np.int64)[o]
    if not len(a):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    reach = np.maximum.accumulate(b)
    opens = np.flatnonzero(np.concatenate([[True], a[1:] > reach[:-1]]))
    return a[opens], np.maximum.reduceat(b, opens)


def purged_kfold(events: pd.DataFrame, n_splits: int = 5, embargo: int = EMBARGO_S, group: str = "day") -> list:
    """[(train positions, test positions)]: test folds are contiguous blocks of `group` values (days by default, so a
    window is never split); training drops every row whose span [t0, t1) overlaps any test row's span, and every row
    starting within `embargo` seconds after the end of a stretch of test spans (7.4.1-7.4.2)."""
    g = events[group].to_numpy()
    keys = np.array(sorted(pd.unique(g)))
    t0, t1 = events["t0"].to_numpy(np.int64), events["t1"].to_numpy(np.int64)
    folds = []
    for block in np.array_split(keys, n_splits):
        if not len(block):
            continue
        test = np.isin(g, block)
        if not test.any():
            continue
        lo, hi = _merge(t0[test], t1[test])
        j = np.searchsorted(lo, t1, side="left") - 1           # the last test stretch starting before each span ends
        overlap = (j >= 0) & (hi[np.maximum(j, 0)] > t0)
        k = np.searchsorted(hi, t0, side="right") - 1          # the last test stretch ending at or before t0
        embargoed = (k >= 0) & (t0 < hi[np.maximum(k, 0)] + embargo)
        train = ~test & ~overlap & ~embargoed
        folds.append((np.flatnonzero(train), np.flatnonzero(test)))
    return folds
