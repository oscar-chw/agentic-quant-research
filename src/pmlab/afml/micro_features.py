"""AFML ch. 19.6, the additional microstructural features, per event of the event dataset (plan afml-pipeline node 15).

The measures of 19.3-19.5 already exist (pmlab.micro: Roll, Corwin-Schultz, Kyle, Amihud, bar order imbalance;
pmlab.afml.microstructure: Beckers-Parkinson, Hasbrouck, BVC VPIN) and the event dataset carries the bar versions as
f_ features. This module adds what 19.6 proposes beyond theory, for the events of pmlab.afml.events (one row per
(start, t); the three primary rules of an event share it):

1. Distribution of order sizes (19.6.1). Human "GUI" traders click round sizes, algorithms randomise theirs. A taker
   trade is round in shares when its size is a whole multiple of ROUND_STEP (5) shares: Polymarket's minimum limit
   order is 5 shares and the book's round sizes 5, 10, 20, 25, 50, 100, ... are all multiples of 5; it is round in
   dollars when it is not round in shares and its notional (token price x size) is a whole dollar >= $1 to the cent
   (the $1 minimum and the preset amounts of a market buy, which the UI sends in dollars). The book suggests watching
   deviations from the normal frequency, so the share among the window's last trades is also taken net of the share
   among every BTC 5-minute trade of the last hour.
2. Serial correlation of signed order flow (19.6.5, and 19.3.1's run tests on trade signs): lag-1 autocorrelation of
   aggressor signs and of signed shares, the Wald-Wolfowitz runs z, the mean and the current run of same-side trades.
3. Execution-algorithm footprints (19.6.3): a slicer sends a fixed size at near-constant intervals. Regularity is the
   coefficient of variation (sd / mean, ddof 0) of the gaps between consecutive distinct trade seconds (Poisson
   arrivals near 1, a clock-driven slicer near 0); repeated non-round sizes and the regularity of the largest such
   group; the book's own test, volume in the first seconds of each minute and its imbalance; per wallet where the live
   tape carries one (from 2026-09-17 08:40 UTC): the top wallet's share of volume, the Herfindahl index, the top
   wallet's flow and the regularity of its trades.
4. Cancellations, limit orders and market orders (19.6.2), from the recorded book (from 2026-09-16 16:54 UTC). In the
   Up frame (the Up and Down books are exact mirrors), each price_change sets a level's aggregate size; its delta
   against the replayed level (the last change at the level, or the last snapshot when it is later) is an addition
   (> 0) or a decrease (< 0). A decrease is either a trade against the level or a cancellation, so over a lookback
   cancelled shares at a level = max(0, decreased shares - shares traded at that level on that side); a decrease
   event is a cancellation event when no trade at its level and side is within MATCH_S seconds. Near the touch: the
   level is within NEAR_TOUCH_E4 (2 cents) of the side's best price before or after the change.
5. Options-implied information (19.6.4): Deribit DVOL, the only options source cached (data/cache/dvol.parquet,
   1-minute closes 2026-09-02 12:00 to 2026-09-16 19:41 UTC; no download). Level, one-hour log change and its sigma
   per sqrt(s) over the event's 300 s realised BTC sigma (f_rv_300).

Availability (the event dataset's rule, pmlab.afml.events): the row of an event at second e = start + t uses only
what is known before e.
- A tape print stamped s happened in [s, s + 1): prints stamped <= e - 1. Within a second, the data-api order.
- A book change, snapshot or websocket trade is used when its exchange time (ms) is < e * 1000. The local receive
  clock is ~2.96 s ahead of Binance's (research/microstructure.md), so exchange time is the clock of every feed here.
  Deltas are computed in (exchange time, receive time, snapshots first) order, so a change before e never depends on
  anything at or after e, and a decrease before e is never excused by a trade at or after e.
- A DVOL minute stamped with its open ts is known from ts + 60 (pmlab.vol.dvol_at): the last minute known <= e - 1.
- Nothing at or after pmlab.evaluation.CONFIRM_START is read: windows must end by it, live hour files are listed only
  when the whole hour ends by it, tape days only when they start before it, DVOL minutes only when known before it.

Book features are NaN (m19_bk_ok = 0) when the lookback [e - BOOK_LOOKBACK, e) is not fully recorded: before the
window's first snapshot, or overlapping a gap of more than GAP_S seconds without any price change in the whole
market (a socket stall, a missing or not yet compacted hour, the start of the recording), counted only once the
silence has lasted GAP_S seconds by e (a stall that starts in the last GAP_S seconds before e is not yet known at e).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pmlab import WINDOW_SECONDS as T
from pmlab import store, vol
from pmlab.evaluation import CONFIRM_START

VERSION = 1
LIVE_DIR = store.DATA_DIR / "live"
DVOL_PATH = store.DATA_DIR / "cache/dvol.parquet"

N_TRADES = 100                  # trades of the window's tokens in the size, sign and footprint statistics
MIN_TRADES = 20                 # fewer known trades: NaN
ROUND_STEP = 5                  # shares
BASELINE_S = 3600               # the normal frequency of round sizes: every BTC 5-minute trade of the last hour
BASELINE_MIN_TRADES = 500
ENTROPY_LO, ENTROPY_HI = -3, 12  # log2 size bins: <= 1/8 share .. >= 4096 shares
N_BINS = ENTROPY_HI - ENTROPY_LO + 1
MINUTE_LOOKBACK, MINUTE_OPEN_S = 300, 5
SLICER_MIN = 4                  # trades of one non-round (side, size) to count as a repeated size
WALLET_MIN_KNOWN = 0.9          # share of the trades with a wallet needed for the wallet features
BOOK_LOOKBACK = 30              # seconds
NEAR_TOUCH_E4 = 200             # 2 cents
MATCH_S = 1.0
GAP_S = 5.0
DVOL_MAX_AGE = 600
DVOL_CHANGE_S = 3600
E4 = 10_000
KEY_BIG = 10 ** 13              # key * KEY_BIG + ts_ms stays below 2**63


def _spec() -> list[dict]:
    f = []

    def add(name, group, unit, definition, source):
        f.append({"name": f"m19_{name}", "group": group, "unit": unit, "definition": definition, "source": source})

    n, tape = N_TRADES, "tape"
    add(f"round_shares_{n}", "order_size", "share", f"share of the window's last {n} trades (stamped <= e - 1) whose size "
        f"is a whole multiple of {ROUND_STEP} shares", tape)
    add(f"round_usd_{n}", "order_size", "share", "share of the same trades not round in shares whose notional (token "
        "price x size) is a whole dollar >= $1 to the cent", tape)
    add("round_dev_1h", "order_size", "share", f"share round in shares or dollars among the last {n} trades minus that "
        f"share among every BTC 5-minute trade stamped in [e - {BASELINE_S}, e - 1] (NaN under {BASELINE_MIN_TRADES})", tape)
    add(f"size_entropy_{n}", "order_size", "0..1", f"Shannon entropy of floor(log2 size) clipped to [{ENTROPY_LO}, "
        f"{ENTROPY_HI}] over the last {n} trades / ln {N_BINS}", tape)
    add(f"size_q50_{n}", "order_size", "shares", f"median size of the last {n} trades", tape)
    add(f"size_q90_{n}", "order_size", "shares", f"90% quantile of the size of the last {n} trades", tape)
    add(f"sign_ac1_{n}", "order_flow", "corr", f"lag-1 autocorrelation (Pearson, ddof 0) of the Up-frame aggressor signs "
        f"of the last {n} trades", tape)
    add(f"signed_vol_ac1_{n}", "order_flow", "corr", f"lag-1 autocorrelation of sign x shares over the last {n} trades", tape)
    add(f"runs_z_{n}", "order_flow", "sd", f"Wald-Wolfowitz runs z of the last {n} signs (< 0: fewer runs than chance, "
        "persistent flow)", tape)
    add(f"run_mean_{n}", "order_flow", "trades", f"mean length of same-sign runs among the last {n} trades", tape)
    add("run_last", "order_flow", "signed trades", f"length of the current same-sign run (within the last {n} trades) "
        "times its sign", tape)
    add(f"gap_cv_{n}", "footprint", "cv", f"sd / mean (ddof 0) of the gaps between consecutive distinct trade seconds "
        f"of the last {n} trades (Poisson ~ 1, a clock-driven slicer -> 0)", tape)
    add(f"slicer_share_{n}", "footprint", "share", f"share of the last {n} trades in a non-round (sign, size to the cent) "
        f"group with >= {SLICER_MIN} trades", tape)
    add(f"slicer_gap_cv_{n}", "footprint", "cv", "gap_cv of the largest such group (NaN without one)", tape)
    add(f"minute_open_ratio_{MINUTE_LOOKBACK}", "footprint", "ratio", f"shares stamped in the first {MINUTE_OPEN_S} "
        f"seconds of a UTC minute per second / shares per second, over the window's trades stamped in "
        f"[e - {MINUTE_LOOKBACK}, e - 1] (1 = no minute clustering; 19.6.3)", tape)
    add(f"minute_open_imb_{MINUTE_LOOKBACK}", "footprint", "share", "signed shares / shares of those minute-open trades", tape)
    w = "tape x live tape wallet (2026-09-17 08:40 UTC on)"
    add(f"wallet_known_{n}", "wallet", "share", f"share of the last {n} trades whose taker wallet is known (live tape "
        "joined on tx, outcome, side, size); 0 before the wallet was recorded", w)
    add(f"wallet_top_share_{n}", "wallet", "share", f"largest wallet's share of the shares of the last {n} trades with a "
        f"wallet (NaN when fewer than {WALLET_MIN_KNOWN:.0%} have one)", w)
    add(f"wallet_hhi_{n}", "wallet", "0..1", "Herfindahl index of the wallets' shares of that volume", w)
    add(f"wallet_top_flow_{n}", "wallet", "share", "the largest wallet's signed shares / its shares", w)
    add(f"wallet_top_gap_cv_{n}", "wallet", "cv", f"gap_cv of the trades of the wallet with the most trades (>= {SLICER_MIN})", w)
    b, lb = "recorded book (price_change, book, trade; 2026-09-16 16:54 UTC on)", BOOK_LOOKBACK
    for side in ("bid", "ask"):
        add(f"bk_cancel_{side}_{lb}", "book", "shares per s", f"cancelled shares near the {side} touch over "
            f"[e - {lb}, e): per level max(0, decreased - traded shares), summed, / {lb}", b)
        add(f"bk_add_{side}_{lb}", "book", "shares per s", f"added shares near the {side} touch over the lookback / {lb}", b)
        add(f"bk_cancel_add_{side}_{lb}", "book", "ratio", f"bk_cancel_{side} / bk_add_{side} (NaN without additions)", b)
        add(f"bk_cancel_n_{side}_{lb}", "book", "events per s", f"decreases near the {side} touch with no trade at their "
            f"level and side within {MATCH_S:g} s (trades before e only) / {lb}", b)
    add(f"bk_cancel_trade_{lb}", "book", "ratio", "cancelled shares near both touches / shares traded (websocket trades) "
        "over the lookback (NaN without trades)", b)
    add("bk_ok", "book", "flag", f"1 when [e - {lb}, e) lies after the window's first snapshot and overlaps no market-wide "
        f"price_change silence > {GAP_S:g} s; else 0 and the book features are NaN", b)
    d = "data/cache/dvol.parquet (Deribit DVOL 1-min, 2026-09-02 12:00 to 09-16 19:41 UTC)"
    add("dvol", "options", "annualised %", f"DVOL close of the last minute known <= e - 1 (known at open + 60 s; NaN when "
        f"older than {DVOL_MAX_AGE} s)", d)
    add("dvol_chg_1h", "options", "log", f"log DVOL now minus log DVOL known {DVOL_CHANGE_S} s earlier (both fresh)", d)
    add("dvol_over_rv300", "options", "ratio", "DVOL sigma per sqrt(s) (pmlab.vol.dvol_at) / the event's f_rv_300", d)
    return f


FEATURES = _spec()
FEATURE_NAMES = [f["name"] for f in FEATURES]
TAPE_NAMES = [f["name"] for f in FEATURES if f["group"] in ("order_size", "order_flow", "footprint", "wallet")]
BOOK_NAMES = [f["name"] for f in FEATURES if f["group"] == "book"]
DVOL_NAMES = [f["name"] for f in FEATURES if f["group"] == "options"]
N, LB = N_TRADES, BOOK_LOOKBACK


# ---------------------------------------------------------------------------------------------------- statistics

def round_flags(size, price) -> tuple[np.ndarray, np.ndarray]:
    """(round in shares, round in dollars and not in shares) per trade."""
    size, price = np.asarray(size, dtype=float), np.asarray(price, dtype=float)
    whole = np.round(size)
    shares = (np.abs(size - whole) < 1e-6) & (whole >= ROUND_STEP) & (np.mod(whole, ROUND_STEP) == 0)
    cents = price * size * 100.0
    cent = np.round(cents)
    usd = (np.abs(cents - cent) < 1e-4) & (np.mod(cent, 100) == 0) & (cent >= 100) & ~shares
    return shares, usd


def size_entropy(size) -> float:
    s = np.asarray(size, dtype=float)
    s = s[s > 0]
    if len(s) < 2:
        return np.nan
    b = np.clip(np.floor(np.log2(s)), ENTROPY_LO, ENTROPY_HI).astype(np.int64) - ENTROPY_LO
    p = np.bincount(b, minlength=N_BINS) / len(b)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(N_BINS))


def lag1_autocorr(x) -> float:
    """Pearson correlation of x[i] with x[i - 1] (ddof 0); NaN under 3 values or with a constant side."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.nan
    a, b = x[:-1], x[1:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def runs(sign) -> tuple[float, float, float]:
    """(Wald-Wolfowitz z, mean run length, signed length of the last run) of a +-1 sequence."""
    s = np.sign(np.asarray(sign, dtype=float))
    s = s[s != 0]
    n = len(s)
    if n < 2:
        return np.nan, np.nan, np.nan
    change = np.flatnonzero(s[1:] != s[:-1])
    r = len(change) + 1
    last = n - (int(change[-1]) + 1) if len(change) else n
    n1 = float((s > 0).sum())
    n2 = n - n1
    z = np.nan
    if n1 and n2:
        mu = 2 * n1 * n2 / n + 1
        var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
        z = (r - mu) / math.sqrt(var) if var > 0 else np.nan
    return float(z), n / r, float(last * s[-1])


def gap_cv(ts) -> float:
    """sd / mean (ddof 0) of the gaps between consecutive distinct stamps; NaN under two gaps."""
    u = np.unique(np.asarray(ts, dtype=float))
    if len(u) < 3:
        return np.nan
    g = np.diff(u)
    return float(g.std() / g.mean())


def wallet_stats(wallet, shares, sign, ts) -> dict:
    codes, uniq = pd.factorize(np.asarray(wallet, dtype=object))
    shares = np.asarray(shares, dtype=float)
    vol_w = np.bincount(codes, weights=shares, minlength=len(uniq))
    cnt = np.bincount(codes, minlength=len(uniq))
    total = vol_w.sum()
    out = {"top_share": np.nan, "hhi": np.nan, "top_flow": np.nan, "top_gap_cv": np.nan}
    if total > 0:
        share = vol_w / total
        top = int(np.argmax(vol_w))
        out.update(top_share=float(share.max()), hhi=float((share ** 2).sum()),
                   top_flow=float((np.asarray(sign, float) * shares)[codes == top].sum() / vol_w[top]))
    busy = int(np.argmax(cnt))
    if cnt[busy] >= SLICER_MIN:
        out["top_gap_cv"] = gap_cv(np.asarray(ts)[codes == busy])
    return out


# ---------------------------------------------------------------------------------------------------- tape features

@dataclass
class Tape:
    """One window's taker tape (both tokens, Up frame) in stamp order."""
    ts: np.ndarray
    sign: np.ndarray
    shares: np.ndarray
    price: np.ndarray               # the token's own price (notional = price x shares)
    wallet: np.ndarray | None = None


def wallet_key(tx, outcome, side, size) -> np.ndarray:
    size = np.round(np.asarray(size, dtype=float), 4).astype(str)
    return (pd.Series(np.asarray(tx, dtype=object)).astype(str) + "|" + pd.Series(np.asarray(outcome, dtype=object)).astype(str)
            + "|" + pd.Series(np.asarray(side, dtype=object)).astype(str) + "|" + pd.Series(size)).to_numpy(dtype=object)


def tape_arrays(tape: pd.DataFrame, wallets: dict | None = None) -> Tape:
    """From a pmlab.store tape (ts, sign, shares, price; tx, outcome, side for wallets)."""
    t = tape.sort_values("ts", kind="stable")
    w = None
    if wallets:
        keys = wallet_key(t["tx"], t["outcome"], t["side"], t["size"])
        w = np.array([wallets.get(k, "") for k in keys], dtype=object)
    return Tape(t["ts"].to_numpy(np.int64), t["sign"].to_numpy(np.int8), t["shares"].to_numpy(float),
                t["price"].to_numpy(float), w)


@dataclass
class Pool:
    """Every BTC 5-minute trade of a span, for the normal frequency of round sizes."""
    ts: np.ndarray
    cum_round: np.ndarray

    def share(self, now) -> np.ndarray:
        now = np.asarray(now, dtype=np.int64)
        hi = np.searchsorted(self.ts, now - 1, side="right")
        lo = np.searchsorted(self.ts, now - BASELINE_S, side="left")
        n = hi - lo
        return np.where(n >= BASELINE_MIN_TRADES, (self.cum_round[hi] - self.cum_round[lo]) / np.maximum(n, 1), np.nan)


def make_pool(ts, size, price) -> Pool:
    ts = np.asarray(ts, dtype=np.int64)
    o = np.argsort(ts, kind="stable")
    shares, usd = round_flags(np.asarray(size)[o], np.asarray(price)[o])
    return Pool(ts[o], np.concatenate([[0.0], np.cumsum(shares | usd)]))


def tape_features(tp: Tape, now, pool: Pool | None = None) -> dict[str, np.ndarray]:
    """The order-size, order-flow, footprint and wallet features at event seconds `now` (unix) of one window."""
    now = np.asarray(now, dtype=np.int64)
    out = {k: np.full(len(now), np.nan) for k in TAPE_NAMES}
    if not len(now):
        return out
    r_sh, r_usd = round_flags(tp.shares, tp.price)
    r_any = r_sh | r_usd
    base = pool.share(now) if pool is not None else np.full(len(now), np.nan)
    sec = np.mod(tp.ts, 60) < MINUTE_OPEN_S
    cum = lambda x: np.concatenate([[0.0], np.cumsum(x)])
    c_sh, c_open, c_open_signed = cum(tp.shares), cum(tp.shares * sec), cum(tp.shares * sec * tp.sign)
    size_key = np.round(tp.shares, 2)
    his = np.searchsorted(tp.ts, now - 1, side="right")
    for j, (e, hi) in enumerate(zip(now, his)):
        lo2 = np.searchsorted(tp.ts, e - MINUTE_LOOKBACK, side="left")
        v_all = c_sh[hi] - c_sh[lo2]
        if v_all > 0:
            v_open = c_open[hi] - c_open[lo2]
            out[f"m19_minute_open_ratio_{MINUTE_LOOKBACK}"][j] = (v_open / (MINUTE_LOOKBACK // 60 * MINUTE_OPEN_S)) / (v_all / MINUTE_LOOKBACK)
            if v_open > 0:
                out[f"m19_minute_open_imb_{MINUTE_LOOKBACK}"][j] = (c_open_signed[hi] - c_open_signed[lo2]) / v_open
        lo = max(0, hi - N_TRADES)
        if hi - lo < MIN_TRADES:
            continue
        sl = slice(lo, hi)
        sz, sg, ts = tp.shares[sl], tp.sign[sl].astype(float), tp.ts[sl]
        out[f"m19_round_shares_{N}"][j] = r_sh[sl].mean()
        out[f"m19_round_usd_{N}"][j] = r_usd[sl].mean()
        out["m19_round_dev_1h"][j] = r_any[sl].mean() - base[j]
        out[f"m19_size_entropy_{N}"][j] = size_entropy(sz)
        out[f"m19_size_q50_{N}"][j], out[f"m19_size_q90_{N}"][j] = np.quantile(sz, [0.5, 0.9])
        out[f"m19_sign_ac1_{N}"][j] = lag1_autocorr(sg)
        out[f"m19_signed_vol_ac1_{N}"][j] = lag1_autocorr(sg * sz)
        z, mean_run, last = runs(sg)
        out[f"m19_runs_z_{N}"][j], out[f"m19_run_mean_{N}"][j], out["m19_run_last"][j] = z, mean_run, last
        out[f"m19_gap_cv_{N}"][j] = gap_cv(ts)
        nr = ~r_any[sl]
        if nr.any():
            keys = sg[nr] * size_key[sl][nr]                          # sizes are > 0: one key per (sign, size)
            u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            big = cnt >= SLICER_MIN
            out[f"m19_slicer_share_{N}"][j] = cnt[big].sum() / (hi - lo)
            if big.any():
                out[f"m19_slicer_gap_cv_{N}"][j] = gap_cv(ts[nr][inv == int(np.argmax(cnt))])
        else:
            out[f"m19_slicer_share_{N}"][j] = 0.0
        if tp.wallet is not None:
            wl = tp.wallet[sl]
            known = wl != ""
            out[f"m19_wallet_known_{N}"][j] = known.mean()
            if known.mean() >= WALLET_MIN_KNOWN:
                ws = wallet_stats(wl[known], sz[known], sg[known], ts[known])
                for k in ("top_share", "hhi", "top_flow", "top_gap_cv"):
                    out[f"m19_wallet_{k}_{N}"][j] = ws[k]
        else:
            out[f"m19_wallet_known_{N}"][j] = 0.0
    return out


# ---------------------------------------------------------------------------------------------------- book features

@dataclass
class Book:
    """One window's Up-frame book changes with their deltas, in (exchange time, receive time) order."""
    ts_ms: np.ndarray
    side: np.ndarray                # 0 bid, 1 ask
    price: np.ndarray               # e4
    delta: np.ndarray               # NaN when the level's previous size is unknown
    near: np.ndarray
    first_snapshot_ms: float


@dataclass
class Trades:
    """Websocket trades of one window in the Up frame, by exchange time: the side of the book they consumed."""
    ts_ms: np.ndarray
    side: np.ndarray                # 1: a buy of Up took asks, 0: a sell of Up took bids
    price: np.ndarray               # e4
    size: np.ndarray


def book_deltas(ch: dict, snaps: dict) -> Book:
    """ch: ts_ms, recv_q, side, price (e4), size, best_bid, best_ask (e4, NaN when none) of one window's Up-frame price
    changes; snaps: ts_ms, recv_q per snapshot and levels (snap, side, price, size) with snap indexing those arrays.
    A change's previous size is the last change at its level, unless a snapshot came later (then the snapshot's size
    there, 0 if absent); before any of them it is unknown."""
    n_c, n_s = len(ch["ts_ms"]), len(snaps["ts_ms"])
    ts = np.concatenate([np.asarray(snaps["ts_ms"], np.int64), np.asarray(ch["ts_ms"], np.int64)])
    recv = np.concatenate([np.asarray(snaps["recv_q"], np.int64), np.asarray(ch["recv_q"], np.int64)])
    kind = np.r_[np.zeros(n_s, np.int8), np.ones(n_c, np.int8)]
    order = np.lexsort((kind, recv, ts))
    rank = np.empty(n_s + n_c, np.int64)
    rank[order] = np.arange(n_s + n_c)
    s_rank, c_rank = rank[:n_s], rank[n_s:]
    c_order = np.argsort(c_rank, kind="stable")                          # changes in replay order
    side = np.asarray(ch["side"], np.int64)[c_order]
    price = np.asarray(ch["price"], np.int64)[c_order]
    size = np.asarray(ch["size"], float)[c_order]
    cr = c_rank[c_order]
    key = side * 20_000 + price
    by_key = np.lexsort((cr, key))
    prev = np.full(n_c, -1, np.int64)                                    # replay position of the previous same-level change
    same = np.r_[False, key[by_key][1:] == key[by_key][:-1]]
    prev[by_key[same]] = by_key[np.flatnonzero(same) - 1]
    prev_rank = np.where(prev >= 0, cr[np.maximum(prev, 0)], -1)
    s_sorted = np.sort(s_rank)
    s_index = np.argsort(s_rank, kind="stable")
    k = np.searchsorted(s_sorted, cr) - 1                                # last snapshot before each change
    use_snap = (k >= 0) & (np.where(k >= 0, s_sorted[np.maximum(k, 0)], -1) > prev_rank)
    snap_size = np.zeros(n_c)
    if n_s and len(snaps["snap"]):
        pos_of = np.empty(n_s, np.int64)
        pos_of[s_index] = np.arange(n_s)                                  # snapshot -> its position in replay order
        lv = pos_of[np.asarray(snaps["snap"], np.int64)] * 40_000 + np.asarray(snaps["side"], np.int64) * 20_000 \
            + np.asarray(snaps["price"], np.int64)
        o = np.argsort(lv, kind="stable")
        lv_sorted, lv_size = lv[o], np.asarray(snaps["size"], float)[o]
        want = np.maximum(k, 0) * 40_000 + key
        i = np.searchsorted(lv_sorted, want)
        hit = (i < len(lv_sorted)) & (lv_sorted[np.minimum(i, len(lv_sorted) - 1)] == want)
        snap_size = np.where(hit, lv_size[np.minimum(i, len(lv_sorted) - 1)], 0.0)
    before = np.where(use_snap, snap_size, np.where(prev >= 0, size[np.maximum(prev, 0)], np.nan))
    delta = size - before
    bb = np.asarray(ch["best_bid"], float)[c_order]
    ba = np.asarray(ch["best_ask"], float)[c_order]
    best = np.where(side == 0, bb, ba)
    best_before = np.r_[np.nan, np.where(side[1:] == 0, bb[:-1], ba[:-1])]
    with np.errstate(invalid="ignore"):
        near = (np.abs(price - best) <= NEAR_TOUCH_E4) | (np.abs(price - best_before) <= NEAR_TOUCH_E4)
    first = float(np.min(snaps["ts_ms"])) if n_s else np.inf
    return Book(np.asarray(ch["ts_ms"], np.int64)[c_order], side, price, delta, near, first)


def _gap_overlaps(gaps: np.ndarray, lo: float, hi: float) -> bool:
    """Does a silence (a, b) (seconds) known by `hi` overlap [lo, hi)? A silence is known to be a gap once it has lasted
    GAP_S seconds: min(b, hi) - a > GAP_S, so a stall that begins just before hi and ends after it is not used."""
    if not len(gaps):
        return False
    a, b = gaps[:, 0], gaps[:, 1]
    return bool(np.any((a < hi) & (b > lo) & (np.minimum(b, hi) - a > GAP_S)))


def book_features(book: Book | None, trades: Trades | None, gaps, now) -> dict[str, np.ndarray]:
    """Cancellation, addition and cancel/trade features over [e - BOOK_LOOKBACK, e) at event seconds `now`."""
    now = np.asarray(now, dtype=np.int64)
    out = {k: np.full(len(now), np.nan) for k in BOOK_NAMES}
    out["m19_bk_ok"] = np.zeros(len(now))
    if book is None or not len(now):
        return out
    gaps = np.asarray(gaps, dtype=float).reshape(-1, 2)
    if trades is None:
        trades = Trades(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    t_order = np.argsort(trades.ts_ms, kind="stable")
    t_ts, t_side = np.asarray(trades.ts_ms, np.int64)[t_order], np.asarray(trades.side, np.int64)[t_order]
    t_price, t_size = np.asarray(trades.price, np.int64)[t_order], np.asarray(trades.size, float)[t_order]
    for j, e in enumerate(now):
        lo_s, hi_s = e - BOOK_LOOKBACK, e
        if book.first_snapshot_ms > lo_s * 1000 or _gap_overlaps(gaps, lo_s, hi_s):
            continue
        out["m19_bk_ok"][j] = 1.0
        a, b = np.searchsorted(book.ts_ms, [lo_s * 1000, hi_s * 1000], side="left")
        d, nr, sd, pr, ts = book.delta[a:b], book.near[a:b], book.side[a:b], book.price[a:b], book.ts_ms[a:b]
        ta, tb = np.searchsorted(t_ts, [lo_s * 1000, hi_s * 1000], side="left")
        tv_side, tv_price, tv_size = t_side[ta:tb], t_price[ta:tb], t_size[ta:tb]
        ma, mb = np.searchsorted(t_ts, [(lo_s - MATCH_S) * 1000, hi_s * 1000], side="left")
        m_key = t_side[ma:mb] * 20_000 + t_price[ma:mb]
        m_val = np.sort(m_key * KEY_BIG + t_ts[ma:mb])
        with np.errstate(invalid="ignore"):
            dec, add = nr & (d < 0), nr & (d > 0)
        cancel_total = 0.0
        for s, name in ((0, "bid"), (1, "ask")):
            ds = dec & (sd == s)
            add_vol = float(d[add & (sd == s)].sum())
            keys = pr[ds]
            cancel = 0.0
            if len(keys):
                u, inv = np.unique(keys, return_inverse=True)
                dec_vol = np.bincount(inv, weights=-d[ds], minlength=len(u))
                ts_side = tv_side == s
                traded = np.zeros(len(u))
                if ts_side.any():
                    tu = np.searchsorted(u, tv_price[ts_side])
                    ok = (tu < len(u)) & (u[np.minimum(tu, len(u) - 1)] == tv_price[ts_side])
                    np.add.at(traded, tu[ok], tv_size[ts_side][ok])
                cancel = float(np.maximum(dec_vol - traded, 0.0).sum())
                q = (s * 20_000 + keys) * KEY_BIG + ts[ds]
                lo_i = np.searchsorted(m_val, q - int(MATCH_S * 1000), side="left")
                hi_i = np.searchsorted(m_val, q + int(MATCH_S * 1000), side="right")
                out[f"m19_bk_cancel_n_{name}_{LB}"][j] = float((hi_i == lo_i).sum()) / BOOK_LOOKBACK
            else:
                out[f"m19_bk_cancel_n_{name}_{LB}"][j] = 0.0
            cancel_total += cancel
            out[f"m19_bk_cancel_{name}_{LB}"][j] = cancel / BOOK_LOOKBACK
            out[f"m19_bk_add_{name}_{LB}"][j] = add_vol / BOOK_LOOKBACK
            out[f"m19_bk_cancel_add_{name}_{LB}"][j] = cancel / add_vol if add_vol > 0 else np.nan
        traded_total = float(tv_size.sum())
        out[f"m19_bk_cancel_trade_{LB}"][j] = cancel_total / traded_total if traded_total > 0 else np.nan
    return out


def book_window_totals(book: Book | None, trades: Trades | None, start: int) -> dict:
    """Whole-window counts for the data-quality section: rows, unknown deltas, near-touch decreases and additions,
    traded shares and cancelled shares (per level max(0, decreased - traded)) over changes before start + T."""
    z = {"changes": 0, "unknown_delta": 0, "near_decrease_shares": 0.0, "near_add_shares": 0.0, "decrease_shares": 0.0,
         "traded_shares": 0.0, "near_cancel_shares": 0.0, "near_decrease_events": 0}
    if book is None:
        return z
    m = book.ts_ms < (start + T) * 1000
    d, nr, sd, pr = book.delta[m], book.near[m], book.side[m], book.price[m]
    with np.errstate(invalid="ignore"):
        dec = nr & (d < 0)
        z.update(changes=int(m.sum()), unknown_delta=int(np.isnan(d).sum()), near_decrease_shares=float(-d[dec].sum()),
                 near_add_shares=float(d[nr & (d > 0)].sum()), decrease_shares=float(-d[d < 0].sum()),
                 near_decrease_events=int(dec.sum()))
    if trades is not None and len(trades.ts_ms):
        tm = np.asarray(trades.ts_ms) < (start + T) * 1000
        z["traded_shares"] = float(np.asarray(trades.size)[tm].sum())
        dk = sd[dec] * 20_000 + pr[dec]
        tk = np.asarray(trades.side)[tm] * 20_000 + np.asarray(trades.price)[tm]
        u, inv = np.unique(dk, return_inverse=True)
        dv = np.bincount(inv, weights=-d[dec], minlength=len(u))
        tv = np.zeros(len(u))
        i = np.searchsorted(u, tk)
        ok = (i < len(u)) & (u[np.minimum(i, max(len(u) - 1, 0))] == tk) if len(u) else np.zeros(len(tk), bool)
        np.add.at(tv, i[ok], np.asarray(trades.size)[tm][ok])
        z["near_cancel_shares"] = float(np.maximum(dv - tv, 0).sum())
    else:
        z["near_cancel_shares"] = z["near_decrease_shares"]
    return z


# ---------------------------------------------------------------------------------------------------- options

def dvol_features(now, dvol: pd.DataFrame | None, rv300=None, confirm_start: int = CONFIRM_START) -> dict[str, np.ndarray]:
    now = np.asarray(now, dtype=np.int64)
    out = {k: np.full(len(now), np.nan) for k in DVOL_NAMES}
    if dvol is None or not len(dvol) or not len(now):
        return out
    known = dvol["ts"].to_numpy(np.int64) // 1000 + 60
    d = dvol[known < confirm_start]
    if not len(d):
        return out
    known = np.sort(d["ts"].to_numpy(np.int64) // 1000 + 60)

    def level(at):
        i = np.searchsorted(known, at, side="right") - 1
        fresh = (i >= 0) & (at - known[np.maximum(i, 0)] <= DVOL_MAX_AGE)
        sigma = vol.dvol_at(at, d)
        return np.where(fresh, sigma, np.nan)

    sig = level(now - 1)
    sig_before = level(now - 1 - DVOL_CHANGE_S)
    out["m19_dvol"] = sig * 100 * math.sqrt(vol.SECONDS_PER_YEAR)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["m19_dvol_chg_1h"] = np.log(sig / sig_before)
        if rv300 is not None:
            rv = np.asarray(rv300, dtype=float)
            out["m19_dvol_over_rv300"] = np.where(rv > 0, sig / rv, np.nan)
    return out


def load_dvol(path: Path = DVOL_PATH, confirm_start: int = CONFIRM_START) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    d = pd.read_parquet(path)
    return d[d["ts"].to_numpy(np.int64) // 1000 + 60 < confirm_start].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------- one window

def window_features(start: int, t, tape: Tape, pool: Pool | None = None, book: Book | None = None,
                    trades: Trades | None = None, gaps=(), dvol: pd.DataFrame | None = None, rv300=None,
                    confirm_start: int = CONFIRM_START) -> pd.DataFrame:
    """Every feature at the events (start, t) of one window: one row per event second, keyed by start and t."""
    start = int(start)
    if start + T > confirm_start:
        raise ValueError(f"window {start} ends after the confirmation start")
    t = np.asarray(t, dtype=np.int64)
    now = start + t
    feats = {**tape_features(tape, now, pool), **book_features(book, trades, gaps, now),
             **dvol_features(now, dvol, rv300, confirm_start)}
    df = pd.DataFrame({"start": np.full(len(t), start, np.int64), "t": t.astype(np.int32)})
    for k in FEATURE_NAMES:
        df[k] = np.asarray(feats[k], dtype=np.float32)
    return df


# ---------------------------------------------------------------------------------------------------- readers

def hour_name(hour: int) -> str:
    return pd.Timestamp(hour, unit="s", tz="UTC").strftime("%Y-%m-%dT%H")


def live_hours(root: Path, kind: str, lo: int, hi: int, confirm_start: int = CONFIRM_START) -> list[tuple[int, Path]]:
    """(hour, compacted file) for the hours starting in [lo, hi) that end by the confirmation start."""
    out = []
    for hour in range(lo - lo % 3600, hi, 3600):
        if hour + 3600 > confirm_start:
            break
        folder = Path(root) / kind / f"hour={hour_name(hour)}"
        for name in ("v2.parquet", "data.parquet"):
            if (folder / name).exists():
                out.append((hour, folder / name))
                break
    return out


def _is(col, value) -> np.ndarray:
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if pa.types.is_dictionary(col.type):
        col = col.dictionary_decode()
    return pc.fill_null(pc.equal(col, value), False).to_numpy(zero_copy_only=False)


def _num(col, dtype=float) -> np.ndarray:
    return np.asarray(col.to_numpy(zero_copy_only=False) if hasattr(col, "to_numpy") else col, dtype=dtype)


def read_changes(path: Path, windows, cutoff_ms: float = np.inf) -> tuple[dict[int, dict], np.ndarray]:
    """Up-frame price changes per window of `windows` (rows stamped before the window end and before `cutoff_ms`) and
    every distinct exchange stamp of the file before the cutoff (seconds, sorted) for the market-wide silence test."""
    windows = np.asarray(sorted(windows), dtype=np.int64)
    if path.name == "v2.parquet":
        t = pq.read_table(path, columns=["recv_q", "ts_ms", "window", "outcome", "side", "price_e4", "size",
                                         "best_bid_e4", "best_ask_e4"])
        all_ts = np.unique(_num(t["ts_ms"], np.int64)) / 1000.0
        t = t.filter(pc.is_in(t["window"], value_set=pa.array(windows)))
        ts_ms, recv_q = _num(t["ts_ms"], np.int64), _num(t["recv_q"], np.int64)
        price, bb, ba = _num(t["price_e4"]), _num(t["best_bid_e4"]), _num(t["best_ask_e4"])
    else:
        t = pq.read_table(path, columns=["recv", "ts", "window", "outcome", "side", "price", "size", "best_bid", "best_ask"])
        all_ts = np.unique(np.round(_num(t["ts"]) * 1000).astype(np.int64)) / 1000.0
        t = t.filter(pc.is_in(t["window"], value_set=pa.array(windows)))
        ts_ms = np.round(_num(t["ts"]) * 1000).astype(np.int64)
        recv_q = np.round(_num(t["recv"]) * (1 << 22)).astype(np.int64)
        price, bb, ba = (np.round(_num(t[c]) * E4) for c in ("price", "best_bid", "best_ask"))
    window = _num(t["window"], np.int64)
    is_up, is_sell = _is(t["outcome"], "Up"), _is(t["side"], "SELL")
    size = _num(t["size"])
    up_price = np.where(is_up, price, E4 - price)
    up_bb, up_ba = np.where(is_up, bb, E4 - ba), np.where(is_up, ba, E4 - bb)
    side = (is_sell == is_up).astype(np.int64)                         # 1: an Up ask level
    keep = (ts_ms < (window + T) * 1000) & (ts_ms < cutoff_ms)
    all_ts = all_ts[all_ts * 1000 < cutoff_ms]
    out = {}
    order = np.argsort(window[keep], kind="stable")
    w = window[keep][order]
    cols = {"ts_ms": ts_ms, "recv_q": recv_q, "side": side, "price": up_price.astype(np.int64), "size": size,
            "best_bid": up_bb, "best_ask": up_ba}
    cols = {k: v[keep][order] for k, v in cols.items()}
    bounds = np.flatnonzero(np.diff(w)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(w)]):
        if b > a:
            out[int(w[a])] = {k: v[a:b] for k, v in cols.items()}
    return out, all_ts


def read_snapshots(path: Path, windows, cutoff_ms: float = np.inf) -> dict[int, dict]:
    """Up-token book snapshots per window (stamped before the window end): ts_ms, recv_q and flattened levels."""
    windows = np.asarray(sorted(windows), dtype=np.int64)
    if path.name == "v2.parquet":
        t = pq.read_table(path, columns=["recv_q", "ts_ms", "window", "outcome", "bid_price_e4", "bid_size",
                                         "ask_price_e4", "ask_size"])
        t = t.filter(pc.is_in(t["window"], value_set=pa.array(windows)))
        t = t.filter(pa.array(_is(t["outcome"], "Up")))
        ts_ms, recv_q, window = _num(t["ts_ms"], np.int64), _num(t["recv_q"], np.int64), _num(t["window"], np.int64)

        def levels(pcol, scol):
            p, s = t[pcol].combine_chunks(), t[scol].combine_chunks()
            lens = pc.fill_null(pc.list_value_length(p), 0).to_numpy(zero_copy_only=False).astype(np.int64)
            return (np.repeat(np.arange(len(t)), lens), _num(pc.list_flatten(p), np.int64), _num(pc.list_flatten(s)))
        bids, asks = levels("bid_price_e4", "bid_size"), levels("ask_price_e4", "ask_size")
    else:
        t = pq.read_table(path, columns=["recv", "ts", "window", "outcome", "bids", "asks"]).to_pandas()
        t = t[t["window"].isin(windows) & (t["outcome"] == "Up")].reset_index(drop=True)
        ts_ms = np.round(t["ts"].to_numpy(float) * 1000).astype(np.int64)
        recv_q = np.round(t["recv"].to_numpy(float) * (1 << 22)).astype(np.int64)
        window = t["window"].to_numpy(np.int64)

        def levels(col):
            rows, prices, sizes = [], [], []
            for i, v in enumerate(t[col]):
                for p, s in json.loads(v):
                    rows.append(i)
                    prices.append(int(round(float(p) * E4)))
                    sizes.append(float(s))
            return np.asarray(rows, np.int64), np.asarray(prices, np.int64), np.asarray(sizes, float)
        bids, asks = levels("bids"), levels("asks")
    out = {}
    keep = (ts_ms < (window + T) * 1000) & (ts_ms < cutoff_ms)
    for w in np.unique(window[keep]):
        rows = np.flatnonzero(keep & (window == w))
        remap = np.full(len(window), -1, np.int64)
        remap[rows] = np.arange(len(rows))
        parts = []
        for side, (r, p, s) in ((0, bids), (1, asks)):
            m = remap[r] >= 0
            parts.append((remap[r][m], np.full(int(m.sum()), side, np.int64), p[m], s[m]))
        out[int(w)] = {"ts_ms": ts_ms[rows], "recv_q": recv_q[rows], "snap": np.concatenate([x[0] for x in parts]),
                       "side": np.concatenate([x[1] for x in parts]), "price": np.concatenate([x[2] for x in parts]),
                       "size": np.concatenate([x[3] for x in parts])}
    return out


def read_trades(path: Path, windows, cutoff_ms: float = np.inf) -> dict[int, dict]:
    """Websocket trades per window in the Up frame (stamped before the window end)."""
    t = pq.read_table(path, columns=["ts", "window", "outcome", "side", "price", "size"]).to_pandas()
    t = t[t["window"].isin(list(windows))]
    ts_ms = np.round(t["ts"].to_numpy(float) * 1000).astype(np.int64)
    window = t["window"].to_numpy(np.int64)
    is_up = (t["outcome"] == "Up").to_numpy()
    buy_up = (t["side"] == "BUY").to_numpy() == is_up
    p_up = np.where(is_up, t["price"].to_numpy(float), 1.0 - t["price"].to_numpy(float))
    keep = (ts_ms < (window + T) * 1000) & (ts_ms < cutoff_ms)
    out = {}
    for w in np.unique(window[keep]):
        m = keep & (window == w)
        out[int(w)] = {"ts_ms": ts_ms[m], "side": buy_up[m].astype(np.int64),
                       "price": np.round(p_up[m] * E4).astype(np.int64), "size": t["size"].to_numpy(float)[m]}
    return out


def _cat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


def book_windows(windows, root: Path = LIVE_DIR, confirm_start: int = CONFIRM_START, cutoff_ms: float = np.inf):
    """Stream the recorded book hour by hour. Yields (window, Book | None, Trades | None, gaps) for each window of
    `windows` once every hour holding its changes before the window end has been read, then ("hours", info, None,
    None) last. gaps: market-wide price_change silences > GAP_S seconds so far, as (from, to) seconds, including the
    time before the first recorded change and the open silence after the last one read. `cutoff_ms` drops every row
    at or after that exchange time (a truncation check)."""
    windows = sorted(int(w) for w in windows)
    if not windows:
        return
    lo, hi = windows[0] - 2 * T, windows[-1] + T
    pcs = dict(live_hours(root, "price_change", lo, hi, confirm_start))
    snaps = dict(live_hours(root, "book", lo, hi, confirm_start))
    trs = dict(live_hours(root, "trade", lo, hi, confirm_start))
    pending = {w: {"ch": [], "sn": [], "tr": []} for w in windows}
    gaps, last_ts, read = [], None, []
    todo = list(windows)
    for hour in range(lo - lo % 3600, hi, 3600):
        if hour + 3600 > confirm_start:
            break
        live = [w for w in todo if w + T > hour and w - 2 * T < hour + 3600]
        if hour in pcs:
            ch, all_ts = read_changes(pcs[hour], live if live else [0], cutoff_ms)
            for w, v in ch.items():
                pending[w]["ch"].append(v)
            if len(all_ts):
                if last_ts is None:
                    gaps.append((-np.inf, all_ts[0]))
                elif all_ts[0] - last_ts > GAP_S:
                    gaps.append((last_ts, all_ts[0]))
                inner = np.flatnonzero(np.diff(all_ts) > GAP_S)
                gaps.extend(zip(all_ts[inner], all_ts[inner + 1]))
                last_ts = all_ts[-1]
            read.append(hour)
            if live and hour in snaps:
                for w, v in read_snapshots(snaps[hour], live, cutoff_ms).items():
                    pending[w]["sn"].append(v)
            if live and hour in trs:
                for w, v in read_trades(trs[hour], live, cutoff_ms).items():
                    pending[w]["tr"].append(v)
        done = [w for w in todo if w + T <= hour + 3600]
        tail = [(last_ts if last_ts is not None else -np.inf, np.inf)]
        for w in done:
            p = pending.pop(w)
            book = trades = None
            if p["ch"]:
                sn = _cat(p["sn"]) if p["sn"] else None
                if sn is not None:
                    offsets = np.cumsum([0] + [len(x["ts_ms"]) for x in p["sn"][:-1]])
                    sn["snap"] = np.concatenate([x["snap"] + o for x, o in zip(p["sn"], offsets)])
                else:
                    sn = {"ts_ms": np.zeros(0, np.int64), "recv_q": np.zeros(0, np.int64), "snap": np.zeros(0, np.int64),
                          "side": np.zeros(0, np.int64), "price": np.zeros(0, np.int64), "size": np.zeros(0)}
                book = book_deltas(_cat(p["ch"]), sn)
            if p["tr"]:
                tr = _cat(p["tr"])
                trades = Trades(tr["ts_ms"], tr["side"], tr["price"], tr["size"])
            yield w, book, trades, np.asarray(gaps + tail, dtype=float).reshape(-1, 2)
        done_set = set(done)
        todo = [w for w in todo if w not in done_set]
    for w in todo:                                                     # hours past the confirmation start or the loop
        pending.pop(w, None)
        yield w, None, None, np.asarray([(-np.inf, np.inf)])
    yield "hours", {"price_change": sorted(pcs), "book": sorted(snaps), "trade": sorted(trs), "read": read,
                    "gaps": [(float(a), float(b)) for a, b in gaps]}, None, None


def read_wallets(lo: int, hi: int, root: Path = LIVE_DIR, confirm_start: int = CONFIRM_START) -> tuple[dict, list[int]]:
    """{wallet_key: wallet} from the compacted live tape hours starting in [lo, hi) that end by the confirmation start."""
    out, hours = {}, []
    for hour, path in live_hours(root, "tape", lo, hi, confirm_start):
        names = pq.read_schema(path).names
        if "wallet" not in names:
            continue
        t = pq.read_table(path, columns=["ts", "outcome", "side", "size", "tx", "wallet"]).to_pandas()
        t = t[(t["wallet"].fillna("") != "") & (t["ts"] < confirm_start)]
        if not len(t):
            continue
        hours.append(hour)
        out.update(zip(wallet_key(t["tx"], t["outcome"], t["side"], t["size"]), t["wallet"].astype(str)))
    return out, hours


def pool_for_day(day: str, confirm_start: int = CONFIRM_START, read_day=None) -> Pool:
    """Every cached trade of the windows starting on `day` and the day before (ending by the confirmation start)."""
    read_day = read_day or store.read_day
    d0 = store.day_start(day)
    parts = []
    for d in (store.day_name(d0 - 86400), day):
        if store.day_start(d) >= confirm_start:
            continue
        rows = read_day("tape", d)
        rows = rows[(rows["start"].to_numpy(np.int64) + T <= confirm_start) & (rows["ts"].to_numpy(np.int64) < confirm_start)]
        parts.append(rows[["ts", "size", "price"]])
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"ts": [], "size": [], "price": []})
    return make_pool(rows["ts"].to_numpy(np.int64), rows["size"].to_numpy(float), rows["price"].to_numpy(float))
