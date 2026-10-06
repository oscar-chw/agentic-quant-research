"""The research universe: cached, resolved windows with a complete tape (TWAP-60 unless asked otherwise)."""
import pandas as pd

from pmlab import WINDOW_SECONDS as T
from pmlab import polymarket, store
from pmlab.evaluation import CONFIRM_START
from pmlab.events import Trade, trades_from_tape


def starts(twap_lookback: int | None = 60, confirmation: bool = False) -> list[int]:
    """Cached, resolved windows with a tape and klines, settled by `twap_lookback` (60, 30, or 1 = point price;
    None: every rule). Windows ending after evaluation.CONFIRM_START are the pre-registered confirmation data
    (research/preregistration.json): only scripts/confirm.py asks for them."""
    m = store.markets_frame()
    m = m[m["resolved"].astype(bool)]
    if twap_lookback is not None:
        m = m[m["twap_lookback"] == twap_lookback]
    if not confirmation:
        m = m[m["start"] + T <= CONFIRM_START]
    candidates = [int(s) for s in m["start"]]
    tapes, klines = store.tape_starts(), store.underlying_starts(candidates)
    return [s for s in candidates if s in tapes and s in klines]


def split_by_day(all_starts: list[int], train_frac: float = 0.7) -> tuple[list[int], list[int]]:
    days = sorted({s // 86400 for s in all_starts})
    cut = days[int(round(len(days) * train_frac))] if len(days) > 1 else days[0] + 1
    return [s for s in all_starts if s // 86400 < cut], [s for s in all_starts if s // 86400 >= cut]


def window_trades(start: int) -> list[Trade]:
    return trades_from_tape(polymarket.in_window(store.tape(store.market(start))), start)


def tapes(all_starts: list[int]) -> dict[int, pd.DataFrame]:
    return {s: store.tape(store.market(s)) for s in all_starts}


def closes(all_starts: list[int], history: int = 1500) -> dict[int, pd.Series]:
    """1 s closes per window, extended back over contiguous earlier cached windows by up to
    `history` seconds. The live engine's σ EWMA has the whole session behind it; a backtest σ
    warmed up on 180 s alone differs by up to 100% and moves the fair price by 2–3¢
    (research/parity.md). Extension stops at the first gap so no seconds are invented."""
    raw: dict[int, pd.Series] = {}

    def load(s):
        if s not in raw:
            k = store.cached_underlying(s)
            raw[s] = k.set_index("sec")["close"] if k is not None else None
        return raw[s]

    out = {}
    for s in all_starts:
        parts, w = [load(s)], s - T
        while s - w <= history + 300 and load(w) is not None:
            parts.append(raw[w])
            w -= T
        series = pd.concat(parts[::-1])
        series = series[~series.index.duplicated(keep="last")].sort_index()
        out[s] = series[series.index >= s - history - 180]
    return out


def outcomes(all_starts: list[int]) -> dict[int, bool]:
    return {s: bool(store.market(s)["up_won"]) for s in all_starts}
