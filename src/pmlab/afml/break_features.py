"""AFML ch. 17's structural-break statistics as per-event features (plan afml-pipeline node 23, finding
docs/afml_sections/ch17.md F17.1).

Chapter 17 builds features on break tests, but until now every one of them was a description: `daily_flags` per UTC
day and, in research/afml_extensions.md section 8, one statistic per whole window computed after the window closed.
Neither can be an input to a model that decides at second e of a window. This module computes the same statistics
causally, for the events of pmlab.afml.events (one row per (start, t); the three primary rules of an event share it),
and scripts/break_features.py stores them day by day beside the event dataset, keyed by (start, t).

Series. Statistics need a level series, so each window gets a fixed grid of marks every MARK_S seconds from
LOOKBACK seconds before its open to its close, and the statistic of a mark is read at the events at or after it
(a feature is therefore at most MARK_S - 1 seconds stale, never early).
- BTC: the close of the 1 s kline that opened at m - 1, which is known at m (pmlab.afml.events), carried forward over
  missing seconds; log price for the ADF and CUSUM tests (17.4.2(1)), price for the exponential trend.
- Token: the price of the last Polymarket print stamped <= m - 1 in the Up frame (pmlab.features._last_print, the
  rule the event dataset's `last` column obeys), clipped to [PRICE_EPS, 1 - PRICE_EPS]. A token price is bounded, so
  the ADF and CUSUM tests run on its logit and the exponential trend on the price itself, as 17.4.2(1) and
  research/afml_extensions.md section 8 do inside windows. The window's own tape carries its pre-window prints, so
  the lookback is that market's own trading, never the previous window's token.

Availability. Marks before the first kline or the first print carry no value; the series starts at the first finite
mark and a statistic is NaN until MIN_LEN + 2 marks of it are known. `b17_tok_ok` is that flag for the token
(the BTC grid is complete wherever the event dataset exists). Nothing at or after pmlab.evaluation.CONFIRM_START is
read: a window must end by it (start + T > CONFIRM_START is refused, so a window ending exactly at it is
allowed, as pmlab.afml.event_importance.check_inputs has it), and no mark lies at or after the event second.

Statistics (pmlab.afml.breaks, unchanged; this module only chooses the series, the grid and the reading point):
- SADF (17.4.2): BSADF_t, the supremum over start points of the ADF statistic of the windows ending at mark t, every
  window holding at least MIN_LEN regression rows, lags 0.
- Chu-Stinchcombe-White CUSUM on levels (17.3.2): S_{1,t} from the series' first mark, the backward-shifting
  suprema sup_n S_{n,t} and sup_n -S_{n,t}, and max_n |S_{n,t}| / c_0.05[n, t] with c_0.05 = sqrt(4.6 + ln(t - n)).
  The two directions are kept apart because the Up frame is not the frame a row is judged in: a Down row's break in
  its favour is the downward one, so the two swap when the features enter the side frame (pmlab.afml.family_importance).
- The exponential trend of the sub- and super-martingale tests (17.4.3): SMT_t with spec "exp" (log y = a + b i) and
  phi = 0, the statistic that beat both nulls inside 13-15% of windows in research/afml_extensions.md section 8. On
  the token it is computed on each token's own price and `b17_tok_smt_exp` is the larger of the two, which is the
  "either token's price" statistic of that section.

Causality. Every statistic above is a function of the marks up to its own end point, so one pass over a window's grid
gives the value at every mark at once and each event reads its own. `pmlab.afml.breaks._adf_by_end` rescales the whole
series by the standard deviation of its first differences and `pmlab.afml.breaks.smt` scales local time by the window's
length; both leave every t-statistic exactly invariant (a scale change multiplies a coefficient and its standard error
alike), so a mark's statistic does not depend on the marks after it. tests/test_afml_break_features.py pins that: the
features of an event do not move when every mark after it is replaced or dropped.
"""
import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import breaks as bk
from pmlab.features import _last_print

VERSION = 1
LOOKBACK = 900                  # seconds of marks before the window open
MARK_S = 5                      # seconds between marks
MIN_LEN = 20                    # regression rows (SADF) / window length (SMT), in marks
PRICE_EPS = 1e-4                # token price clipped to [eps, 1 - eps] before log and logit
B_ALPHA = 4.6                   # the 5% constant of c_alpha (17.3.2)
SMT_SPEC, SMT_PHI = "exp", 0.0
LAGS = 0
N_MARKS = (LOOKBACK + T) // MARK_S


def _spec() -> list[dict]:
    f = []

    def add(name, series, family, unit, definition):
        f.append({"name": f"b17_{name}", "series": series, "family": family, "group": "breaks", "unit": unit,
                  "definition": definition})

    for s, label, level in (("btc", "BTC", "log price"), ("tok", "the Up token", "logit price")):
        add(f"{s}_sadf", s, "sadf", "t stat", f"BSADF at the event's mark on {label}'s {level}: the supremum over start "
            f"points of the ADF statistic of the windows ending there, at least {MIN_LEN} regression rows, lags {LAGS} (17.4.2)")
        add(f"{s}_csw", s, "csw", "sd", f"Chu-Stinchcombe-White S_(1,t) on {label}'s {level}: ({level} now - at the first "
            "mark) / (sigma sqrt(marks between)), sigma the root mean squared mark-to-mark change (17.3.2)")
        add(f"{s}_csw_pos", s, "csw", "sd", f"sup over backward-shifting reference marks n of S_(n,t) on {label}'s {level} "
            "(the largest upward break in the Up frame)")
        add(f"{s}_csw_neg", s, "csw", "sd", f"sup over n of -S_(n,t) on {label}'s {level} (the largest downward break)")
        add(f"{s}_csw_ratio", s, "csw", "ratio", f"max over n of |S_(n,t)| / sqrt({B_ALPHA} + ln(t - n)) on {label}'s "
            f"{level}: how far the largest break of either sign is past its pointwise 5% value")
    add("btc_smt_exp", "btc", "smt", "t stat", "SMT_t with the exponential trend (log price = a + b i) and phi = 0 on "
        "BTC price: the supremum over start points of |b| / se(b) of the windows ending at the event's mark (17.4.3)")
    add("tok_smt_up", "tok", "smt", "t stat", "the same statistic on the Up token's price")
    add("tok_smt_dn", "tok", "smt", "t stat", "the same statistic on the Down token's price (1 - Up)")
    add("tok_smt_exp", "tok", "smt", "t stat", "max(tok_smt_up, tok_smt_dn): the 'either token's price' exponential "
        "trend of research/afml_extensions.md section 8, which beat both nulls inside 13-15% of windows")
    add("tok_age", "tok", "staleness", "log s", "log(1 + seconds since the last Polymarket print) at the event's "
        "mark. A quiet market contributes zero mark-to-mark changes, which deflates the sigma the CUSUM divides by "
        "and inflates it, so a model reading the CUSUM without this cannot tell a break from nobody printing")
    add("tok_ok", "tok", "flag", "flag", f"1 when at least {MIN_LEN + 2} token marks were known at the event second, "
        "else 0 and every token statistic is NaN")
    return f


FEATURES = _spec()
FEATURE_NAMES = [f["name"] for f in FEATURES]
FLAG_NAMES = [f["name"] for f in FEATURES if f["family"] == "flag"]
STAT_NAMES = [f["name"] for f in FEATURES if f["family"] != "flag"]
FAMILIES = {k: [f["name"] for f in FEATURES if f["family"] == k] for k in ("sadf", "csw", "smt", "staleness")}
SERIES = {k: [f["name"] for f in FEATURES if f["series"] == k and f["family"] != "flag"] for k in ("btc", "tok")}

# Side frame (pmlab.afml.family_importance): the features above are in the Up frame, a row's label is "the rule's side
# won". S_(1,t) is signed, so it changes sign for a Down row; the two directions of the CUSUM supremum and the two
# tokens' exponential trends swap; the rest (a supremum of |.|, an explosiveness statistic) are the same either way.
SIGNED = ("b17_btc_csw", "b17_tok_csw")
PAIRS = (("b17_btc_csw_pos", "b17_btc_csw_neg"), ("b17_tok_csw_pos", "b17_tok_csw_neg"),
         ("b17_tok_smt_up", "b17_tok_smt_dn"))


# ---------------------------------------------------------------------------------------------------- series

def marks(start: int) -> np.ndarray:
    """The window's fixed grid: seconds start - LOOKBACK, ... , start + T - MARK_S."""
    return np.arange(int(start) - LOOKBACK, int(start) + T, MARK_S, dtype=np.int64)


def mark_index(start: int, t) -> np.ndarray:
    """Index of the last mark at or before the event second start + t (the reading point of each event)."""
    return (LOOKBACK + np.asarray(t, dtype=np.int64)) // MARK_S


def btc_marks(grid: np.ndarray, sec: np.ndarray, close: np.ndarray) -> np.ndarray:
    """BTC price at each mark m: the close of the kline that opened at m - 1, known at m, carried forward over missing
    seconds; NaN before the first kline that closed by the first mark."""
    sec = np.asarray(sec, dtype=np.int64)
    close = np.asarray(close, dtype=float)
    if not len(sec):
        return np.full(len(grid), np.nan)
    i = np.searchsorted(sec, grid - 1, side="right") - 1        # the last kline opened at or before m - 1
    out = np.where(i >= 0, close[np.maximum(i, 0)], np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def token_marks(grid: np.ndarray, ts: np.ndarray, p_up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Up-frame price of the last print stamped <= m - 1 at each mark m, clipped to [PRICE_EPS, 1 - PRICE_EPS];
    log(1 + its age in seconds)); both NaN before the first print."""
    ts = np.asarray(ts, dtype=np.int64)
    p = np.asarray(p_up, dtype=float)
    ok = np.isfinite(p)
    val, age = _last_print(ts[ok], p[ok], grid.astype(float))
    return np.clip(val, PRICE_EPS, 1.0 - PRICE_EPS), np.log1p(age)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), PRICE_EPS, 1.0 - PRICE_EPS)
    return np.log(p) - np.log1p(-p)


def _trim(y: np.ndarray) -> tuple[np.ndarray, int]:
    """The series from its first finite mark on, and that mark's index; a later NaN (never produced by the readers
    above, which carry forward) ends it."""
    fin = np.isfinite(np.asarray(y, dtype=float))
    if not fin.any():
        return np.zeros(0), len(y)
    first = int(np.argmax(fin))
    stop = first + int(np.argmin(fin[first:])) if not fin[first:].all() else len(y)
    return np.asarray(y, dtype=float)[first:stop], first


def _place(values: np.ndarray, first: int, n: int, shift: int = 0) -> np.ndarray:
    """A statistic computed on the trimmed series back on the full grid; `shift` is how many marks its index lags the
    grid (BSADF's regression row t ends at series index t + LAGS + 1). A non-finite value becomes NaN: `breaks.bsadf`
    marks a mark whose every window was singular (a perfectly flat stretch) with -inf, which means "no statistic",
    and an infinity in a stored column is not a value a model may read."""
    out = np.full(n, np.nan)
    v = np.asarray(values, dtype=float)
    j = first + shift + np.arange(len(v))
    keep = j < n
    out[j[keep]] = np.where(np.isfinite(v[keep]), v[keep], np.nan)
    return out


def sadf_path(level: np.ndarray) -> np.ndarray:
    """BSADF at every mark of the grid (NaN where no admissible window ends there)."""
    y, first = _trim(level)
    n = len(level)
    if len(y) < MIN_LEN + 2:
        return np.full(n, np.nan)
    return _place(bk.bsadf(y, min_len=MIN_LEN, lags=LAGS)["bsadf"], first, n, shift=LAGS + 1)


def csw_path(level: np.ndarray) -> dict:
    """S_(1,t), the two directions of the backward-shifting supremum and the two-sided ratio, at every mark."""
    y, first = _trim(level)
    n = len(level)
    nan = np.full(n, np.nan)
    if len(y) < MIN_LEN + 2:
        return {"csw": nan, "csw_pos": nan.copy(), "csw_neg": nan.copy(), "csw_ratio": nan.copy()}
    up = bk.csw_cusum(y, b_alpha=B_ALPHA)
    dn = bk.csw_cusum(-y, b_alpha=B_ALPHA)
    # max_n |S| / c = max(max_n S / c, max_n (-S) / c): the two-sided ratio without a third pass.
    with np.errstate(invalid="ignore"):
        ratio = np.fmax(up["ratio_sup"], dn["ratio_sup"])
    return {"csw": _place(up["s_first"], first, n), "csw_pos": _place(up["s_sup"], first, n),
            "csw_neg": _place(dn["s_sup"], first, n), "csw_ratio": _place(ratio, first, n)}


def smt_path(price: np.ndarray) -> np.ndarray:
    """SMT_t of the exponential trend at every mark of the grid, on the series divided by its first value.

    The t-statistic of the exponential trend does not change when the level is scaled (a factor c adds ln c to the
    intercept and leaves b and se(b) alone), but the conditioning of the solve does: `breaks.smt` regresses on local
    time scaled by the *series* length, so a short window near the end of a long grid is fitted over u in [0.92, 1]
    while the same window on a series that stops there is fitted over u in [0.05, 1]. On log BTC (about 11.5 with a
    signal of 1e-3) the two differ in the last digits, and that difference is a dependence on the marks after the
    event: measured on 273 real events, up to 4.3e-5 relative before this line and 1.1e-12 after it."""
    y, first = _trim(price)
    n = len(price)
    if len(y) < MIN_LEN + 2 or np.any(y <= 0):        # positivity of the whole trimmed series: prices are clipped
        return np.full(n, np.nan)                     # positive here, so this is a guard for the next series, not a
    y = y / y[0]                                      # look-ahead (a later mark cannot take a value away from an
    return _place(bk.smt(y, spec=SMT_SPEC, min_len=MIN_LEN, phi=SMT_PHI)["smt"], first, n)   # earlier one here)


# ---------------------------------------------------------------------------------------------------- one window

def window_features(start: int, t: np.ndarray, btc_price: np.ndarray, tok_price: np.ndarray,
                    tok_age: np.ndarray) -> pd.DataFrame:
    """One row per event second `t` of the window, with the statistics of that event's mark. `btc_price`, `tok_price`
    and `tok_age` are the window's grid readings (marks(start))."""
    start = int(start)
    t = np.asarray(t, dtype=np.int64)
    n = len(marks(start))
    if len(btc_price) != n or len(tok_price) != n or len(tok_age) != n:
        raise ValueError(f"the grid of window {start} has {n} marks")
    paths = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        log_btc = np.where(np.asarray(btc_price, float) > 0, np.log(np.maximum(np.asarray(btc_price, float), 1e-300)), np.nan)
    paths["b17_btc_sadf"] = sadf_path(log_btc)
    paths.update({f"b17_btc_{k}": v for k, v in csw_path(log_btc).items()})
    paths["b17_btc_smt_exp"] = smt_path(np.asarray(btc_price, float))
    lg = np.where(np.isfinite(tok_price), logit(tok_price), np.nan)
    paths["b17_tok_sadf"] = sadf_path(lg)
    paths.update({f"b17_tok_{k}": v for k, v in csw_path(lg).items()})
    up = np.asarray(tok_price, float)
    paths["b17_tok_smt_up"] = smt_path(up)
    paths["b17_tok_smt_dn"] = smt_path(1.0 - up)
    with np.errstate(invalid="ignore"):
        paths["b17_tok_smt_exp"] = np.fmax(paths["b17_tok_smt_up"], paths["b17_tok_smt_dn"])
    paths["b17_tok_age"] = np.asarray(tok_age, dtype=float)
    _, tok_first = _trim(up)
    j = mark_index(start, t)
    ok = (j - tok_first + 1) >= MIN_LEN + 2
    out = {"start": np.full(len(t), start, dtype=np.int64), "t": t}
    for name in STAT_NAMES:
        v = paths[name][j]
        out[name] = np.where(ok, v, np.nan).astype(np.float32) if name.startswith("b17_tok_") else v.astype(np.float32)
    out["b17_tok_ok"] = ok.astype(np.float32)
    return pd.DataFrame(out)[["start", "t"] + FEATURE_NAMES]
