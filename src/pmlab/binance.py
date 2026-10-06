"""Binance BTCUSDT 1-second klines: the proxy for the Chainlink underlying."""
import numpy as np
import pandas as pd

from pmlab.http import get_json

# Binance's public market-data host: the same klines, byte for byte, as api.binance.com, which answers HTTP 451
# to US addresses (GitHub's CI runners among them).
KLINES = "https://data-api.binance.vision/api/v3/klines"


def parse_klines(rows: list[list]) -> pd.DataFrame:
    """Kline [open_ms, o, h, l, c, v, close_ms, quote_v, n, ...] -> one row per second.

    `sec` is the kline's open second, so `close` is the last price at instant sec + 1 (the price AT instant i is the
    close of the kline that opened at i − 1). No rows give an empty frame with the same columns.
    """
    if not rows:
        return pd.DataFrame({"sec": pd.Series(dtype="int64"),
                             **{c: pd.Series(dtype=float) for c in ("open", "high", "low", "close", "volume")},
                             "n_trades": pd.Series(dtype="int64")})
    a = np.array([r[:9] for r in rows], dtype=object)
    df = pd.DataFrame({
        "sec": (a[:, 0].astype("int64") // 1000),
        "open": a[:, 1].astype(float), "high": a[:, 2].astype(float),
        "low": a[:, 3].astype(float), "close": a[:, 4].astype(float),
        "volume": a[:, 5].astype(float), "n_trades": a[:, 8].astype("int64"),
    })
    return df.drop_duplicates("sec").sort_values("sec").reset_index(drop=True)


def fetch_seconds(start_sec: int, end_sec: int, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Klines with open second in [start_sec, end_sec), gap-filled to a full 1s grid."""
    rows, cursor = [], start_sec
    while cursor < end_sec:
        page = get_json(KLINES, {"symbol": symbol, "interval": "1s", "limit": 1000,
                                 "startTime": cursor * 1000, "endTime": end_sec * 1000 - 1})
        if not page:
            break
        rows.extend(page)
        cursor = page[-1][0] // 1000 + 1
    df = parse_klines(rows)
    grid = pd.DataFrame({"sec": np.arange(start_sec, end_sec, dtype="int64")})
    df = grid.merge(df, on="sec", how="left")
    df["close"] = df["close"].ffill()   # never bfill: that would copy a later price backwards
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    df[["volume", "n_trades"]] = df[["volume", "n_trades"]].fillna(0)
    df["n_trades"] = df["n_trades"].astype("int64")
    return df
