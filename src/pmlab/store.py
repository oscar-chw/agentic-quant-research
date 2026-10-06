"""Parquet cache of closed windows under data/history, one directory per UTC day. Closed, resolved windows never
change, so they are cached forever.

    data/history/markets/day=YYYY-MM-DD/     one row per window, by the UTC day of its start
    data/history/tape/day=YYYY-MM-DD/        the taker tape: start, ts, outcome, side, price, size, tx (32 bytes);
                                             t, seq, p_up, sign, shares and usdc are rebuilt on read, bit for bit
    data/history/binance_1s/day=YYYY-MM-DD/  Binance BTCUSDT 1 s klines, one row per second of the day

A day holds data.parquet (consolidated) and pieces p-<ns>-<pid>.parquet written since; readers take both, a later
file winning per window (per second for klines). consolidate() folds the pieces into data.parquet. Windows cached in
the layout before 2026-09-17 (data/{markets,tape,binance}/<start>.parquet) are still read when the history lacks them;
`python -m pmlab.store --migrate` copies them in and verifies every window through this API.
"""
import argparse
import contextlib
import fcntl
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pmlab import WINDOW_SECONDS, binance, polymarket

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PAD_BEFORE, PAD_AFTER = 180, 60  # seconds of underlying kept around each window
TAPE_SETTLE = 900                # a tape is cached only this long after its window closed (the data-api lags)
KINDS = ("markets", "tape", "binance_1s")
LEGACY = {"markets": "markets", "tape": "tape", "binance_1s": "binance"}
DAY = "%Y-%m-%d"
TX = re.compile(r"0x[0-9a-f]{64}")

SCHEMAS = {
    "markets": pa.schema([("start", pa.int64()), ("condition_id", pa.string()), ("up_token", pa.string()),
                          ("down_token", pa.string()), ("resolved", pa.bool_()), ("up_won", pa.bool_()),
                          ("twap_lookback", pa.int16()), ("rule_source", pa.string()), ("fee_schedule", pa.string()),
                          ("description", pa.string()), ("resolutionSource", pa.string())]),
    "tape": pa.schema([("start", pa.int64()), ("ts", pa.int64()), ("outcome", pa.string()), ("side", pa.string()),
                       ("price", pa.float64()), ("size", pa.float64()), ("tx", pa.binary(32))]),
    "binance_1s": pa.schema([("sec", pa.int64()), ("open", pa.float64()), ("high", pa.float64()),
                             ("low", pa.float64()), ("close", pa.float64()), ("volume", pa.float64()),
                             ("n_trades", pa.int32())]),
}
KEY = {"markets": "start", "tape": "start", "binance_1s": "sec"}
WRITE = {  # zstd; delta-coded time columns; one row group per ~2 h of windows so a window read skips the rest
    "markets": dict(row_group_size=1 << 20),
    "tape": dict(row_group_size=64_000, column_encoding={"start": "DELTA_BINARY_PACKED", "ts": "DELTA_BINARY_PACKED"},
                 use_dictionary=["outcome", "side", "price", "size"]),
    "binance_1s": dict(row_group_size=1 << 20, column_encoding={"sec": "DELTA_BINARY_PACKED"},
                       use_dictionary=["open", "high", "low", "close", "volume", "n_trades"]),
}
TAPE_COLUMNS = ["ts", "t", "seq", "outcome", "side", "price", "size", "p_up", "sign", "shares", "usdc", "tx"]
MARKET_KEYS = ["start", "slug", "condition_id", "up_token", "down_token", "resolved", "up_won", "twap_lookback",
               "rule_source", "fee_schedule", "description", "resolutionSource"]


# --- layout -------------------------------------------------------------------------------

def history(kind: str) -> Path:
    return DATA_DIR / "history" / kind


def day_name(sec: int) -> str:
    return datetime.fromtimestamp(int(sec), timezone.utc).strftime(DAY)


def day_start(day: str) -> int:
    return int(datetime.strptime(day, DAY).replace(tzinfo=timezone.utc).timestamp())


def days(kind: str) -> list[str]:
    root = history(kind)
    return sorted(p.name.removeprefix("day=") for p in root.glob("day=*") if p.is_dir()) if root.exists() else []


def _day_files(kind: str, day: str) -> list[Path]:
    """data.parquet first, then pieces in write order."""
    return sorted(history(kind).joinpath(f"day={day}").glob("*.parquet"), key=lambda p: (p.name != "data.parquet", p.name))


def _legacy(kind: str, start: int) -> Path:
    return DATA_DIR / LEGACY[kind] / f"{start}.parquet"


# --- encoding -----------------------------------------------------------------------------

def encode(kind: str, df: pd.DataFrame, start: int | None = None) -> pa.Table:
    """The stored form of a market list, one window's tape (`start` given) or klines. Raises ValueError unless
    decode() gives the input back exactly."""
    if kind == "markets":
        rows = pd.DataFrame(df)
        if (rows["slug"] != [polymarket.slug(int(s)) for s in rows["start"]]).any():
            raise ValueError("slug is not btc-updown-5m-<start>")
        rows = rows.reindex(columns=MARKET_KEYS)
        out = pa.Table.from_pandas(rows.drop(columns="slug"), schema=SCHEMAS[kind], preserve_index=False)
        back = [decode_market(r) for r in out.to_pylist()]
        if back != [_market_dict(r) for r in rows.to_dict("records")]:
            raise ValueError("markets do not survive encoding")
        return out
    if kind == "tape":
        if not df["tx"].map(lambda x: isinstance(x, str) and TX.fullmatch(x) is not None).all():
            raise ValueError(f"tape {start}: a tx is not 0x + 64 lowercase hex digits")
        out = pa.table({"start": np.full(len(df), start, dtype=np.int64), "ts": df["ts"].to_numpy(np.int64),
                        "outcome": df["outcome"].astype(str).to_numpy(), "side": df["side"].astype(str).to_numpy(),
                        "price": df["price"].to_numpy(float), "size": df["size"].to_numpy(float),
                        "tx": pa.array([bytes.fromhex(x[2:]) for x in df["tx"]], pa.binary(32))}, schema=SCHEMAS[kind])
        back = decode_tape(out.to_pandas(), start)
        if not back.equals(_tape_frame(df)):
            raise ValueError(f"tape {start} does not survive encoding")
        return out
    out = pa.Table.from_pandas(df[SCHEMAS[kind].names].astype({"n_trades": np.int32}), schema=SCHEMAS[kind],
                               preserve_index=False)
    if not decode_klines(out.to_pandas()).equals(_klines_frame(df)):
        raise ValueError("klines do not survive encoding (n_trades beyond int32?)")
    return out


def _market_dict(m: dict) -> dict:
    """A market with every key, Python scalars, missing values None (legacy rows lack the rule's provenance)."""
    out = {k: m.get(k) for k in MARKET_KEYS}
    for k, v in out.items():
        if isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif isinstance(v, np.generic):
            out[k] = v.item()
    if out["rule_source"] is None and out["twap_lookback"] is not None:
        out["rule_source"] = "config"      # the legacy cache took twap_lookback from cryptoMarketConfig only
    return out


LEGACY_MARKET_KEYS = ["start", "slug", "condition_id", "up_token", "down_token", "resolved", "up_won", "twap_lookback",
                      "fee_schedule"]    # what a legacy market file holds (the rule's provenance came later)


def same_market(legacy: dict, history: dict) -> bool:
    """A legacy market equals its history row on every field the legacy file had."""
    return all(legacy[k] == history[k] for k in LEGACY_MARKET_KEYS)


def decode_market(row: dict) -> dict:
    return _market_dict({**row, "slug": polymarket.slug(int(row["start"]))})


def _tape_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A tape with polymarket.normalise_trades' columns and the dtypes pd.read_parquet gives them."""
    return pa.Table.from_pandas(df[TAPE_COLUMNS], preserve_index=False).cast(pa.schema(
        [("ts", pa.int64()), ("t", pa.int64()), ("seq", pa.int64()), ("outcome", pa.large_string()),
         ("side", pa.large_string()), ("price", pa.float64()), ("size", pa.float64()), ("p_up", pa.float64()),
         ("sign", pa.int8()), ("shares", pa.float64()), ("usdc", pa.float64()), ("tx", pa.large_string())])).to_pandas()


def decode_tape(rows: pd.DataFrame, start: int) -> pd.DataFrame:
    """Stored rows of one window (in stored order) -> the tape polymarket.normalise_trades made."""
    price, size = rows["price"].to_numpy(float), rows["size"].to_numpy(float)
    is_up = (rows["outcome"] == "Up").to_numpy()
    ts = rows["ts"].to_numpy(np.int64)
    raw = np.frombuffer(b"".join(rows["tx"]), np.uint8).reshape(-1, 32)
    hexed = np.frombuffer(b"0123456789abcdef", np.uint8)[np.stack([raw >> 4, raw & 15], axis=2).reshape(-1, 64)]
    tx = np.char.add("0x", hexed.view("S64").ravel().astype(str)) if len(rows) else np.array([], str)
    text = lambda values: pd.array(np.asarray(values, dtype=object), dtype="str")   # the dtype pd.read_parquet gives
    return pd.DataFrame({
        "ts": ts, "t": ts - start, "seq": np.arange(len(rows), dtype=np.int64),
        "outcome": text(rows["outcome"]), "side": text(rows["side"]), "price": price, "size": size,
        "p_up": np.where(is_up, price, 1.0 - price),
        "sign": np.where((rows["side"] == "BUY").to_numpy() == is_up, 1, -1).astype(np.int8),
        "shares": size, "usdc": price * size, "tx": text(tx),
    })


def _klines_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[SCHEMAS["binance_1s"].names].astype({"sec": np.int64, "n_trades": np.int64}).reset_index(drop=True)


def decode_klines(rows: pd.DataFrame) -> pd.DataFrame:
    return _klines_frame(rows)


# --- reading ------------------------------------------------------------------------------

_cache: dict[tuple[str, str], tuple[tuple, pd.DataFrame]] = {}
CACHE_DAYS = 3


def read_day(kind: str, day: str) -> pd.DataFrame:
    """Stored rows of one day, the latest file winning per window (per second for klines), sorted by key.
    Cached per process while the day's files are unchanged."""
    for attempt in range(5):
        files = _day_files(kind, day)
        try:
            sig = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
            if (hit := _cache.get((kind, day))) and hit[0] == sig:
                return hit[1]
            tables = []
            for i, p in enumerate(files):
                t = pq.read_table(p, schema=SCHEMAS[kind])
                tables.append(t.append_column("_file", pa.array(np.full(t.num_rows, i, np.int32))))
            break
        except FileNotFoundError:          # consolidation swapped files between listing and reading
            time.sleep(0.05 * (attempt + 1))
    else:
        raise RuntimeError(f"history {kind} {day}: files keep changing")
    key = KEY[kind]
    if tables:
        df = pa.concat_tables(tables).to_pandas()
        latest = df.groupby(key)["_file"].transform("max")
        df = df[df["_file"] == latest].sort_values([key, "_file"], kind="stable").drop(columns="_file")
        df = df.reset_index(drop=True)
    else:
        df = SCHEMAS[kind].empty_table().to_pandas()
    _cache[(kind, day)] = (sig, df)
    while len(_cache) > CACHE_DAYS * len(KINDS):
        _cache.pop(next(iter(_cache)))
    return df


def _window_rows(kind: str, start: int) -> pd.DataFrame:
    df = read_day(kind, day_name(start))
    lo, hi = np.searchsorted(df["start"].to_numpy(), [start, start + 1])
    return df.iloc[lo:hi]


def cached_market(start: int) -> dict | None:
    rows = _window_rows("markets", start)
    if len(rows):
        return decode_market(rows.iloc[-1].to_dict())
    if (path := _legacy("markets", start)).exists():
        return _market_dict(pd.read_parquet(path).iloc[0].to_dict())
    return None


def cached_tape(start: int) -> pd.DataFrame | None:
    rows = _window_rows("tape", start)
    if len(rows):
        return decode_tape(rows, start)
    if (path := _legacy("tape", start)).exists():
        return pd.read_parquet(path)
    return None


def klines(lo: int, hi: int) -> pd.DataFrame | None:
    """Cached 1 s klines with open second in [lo, hi), or None unless every second is there."""
    parts = []
    for day in sorted({day_name(s) for s in (lo, hi - 1)} | {day_name(s) for s in range(lo, hi, 86400)}):
        df = read_day("binance_1s", day)
        a, b = np.searchsorted(df["sec"].to_numpy(), [lo, hi])
        parts.append(df.iloc[a:b])
    df = pd.concat(parts, ignore_index=True) if parts else None
    if df is None or len(df) != hi - lo:
        return None
    return decode_klines(df)


def cached_underlying(start: int) -> pd.DataFrame | None:
    lo, hi = start - PAD_BEFORE, start + WINDOW_SECONDS + PAD_AFTER
    if (df := klines(lo, hi)) is not None:
        return df
    if (path := _legacy("binance_1s", start)).exists():
        return pd.read_parquet(path)
    return None


# --- writing ------------------------------------------------------------------------------

def _write(table: pa.Table, target: Path, kind: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=9, **WRITE[kind])
    if pq.ParquetFile(tmp).metadata.num_rows != table.num_rows:
        raise RuntimeError(f"{tmp}: read back a different row count")
    os.replace(tmp, target)


def put(kind: str, table: pa.Table) -> list[Path]:
    """Write stored rows as one piece per UTC day they touch."""
    key = table.column(KEY[kind]).to_numpy()
    names = np.array([day_name(k) for k in (key // 86400) * 86400]) if len(key) else np.array([])
    out = []
    for day in sorted(set(names)):
        piece = table.filter(pa.array(names == day))
        target = history(kind) / f"day={day}" / f"p-{time.time_ns()}-{os.getpid()}.parquet"
        _write(piece, target, kind)
        out.append(target)
    return out


@contextlib.contextmanager
def _locked(kind: str, day: str):
    folder = history(kind) / f"day={day}"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def consolidate(kind: str, day: str) -> int:
    """Fold a day's pieces into data.parquet (verified by row count, then by content), then delete the pieces.
    Returns the rows in data.parquet."""
    with _locked(kind, day):
        files = _day_files(kind, day)
        pieces = [p for p in files if p.name != "data.parquet"]
        if not pieces:
            return pq.ParquetFile(files[0]).metadata.num_rows if files else 0
        _cache.pop((kind, day), None)
        want = read_day(kind, day)
        target = history(kind) / f"day={day}" / "data.parquet"
        _write(pa.Table.from_pandas(want, schema=SCHEMAS[kind], preserve_index=False), target, kind)
        _cache.pop((kind, day), None)
        back = pq.read_table(target, schema=SCHEMAS[kind]).to_pandas()
        if not back.equals(want):
            raise RuntimeError(f"history {kind} {day}: consolidated file differs from its sources")
        for p in pieces:
            p.unlink()
        return len(want)


def consolidate_all(min_age: float = 3600, now: float | None = None) -> dict[str, int]:
    """Consolidate every day with pieces that ended at least `min_age` seconds ago."""
    now = time.time() if now is None else now
    done = {}
    for kind in KINDS:
        for day in days(kind):
            if day_start(day) + 86400 + min_age <= now and any(p.name != "data.parquet" for p in _day_files(kind, day)):
                done[f"{kind} {day}"] = consolidate(kind, day)
    return done


# --- the cache API ------------------------------------------------------------------------

def market(start: int) -> dict | None:
    if (m := cached_market(start)) is not None:
        return m
    m = polymarket.fetch_market(start)
    if m is None or not m["resolved"]:
        return m  # unresolved markets are not cached: their outcome is still coming
    put_markets([m])
    return m


def put_markets(markets: list[dict]) -> None:
    if markets:
        put("markets", encode("markets", pd.DataFrame(markets)))


def tape(m: dict) -> pd.DataFrame:
    if (df := cached_tape(m["start"])) is not None:
        return df
    df = polymarket.normalise_trades(polymarket.fetch_raw_trades(m["condition_id"]), m)
    closed_for = time.time() - (m["start"] + WINDOW_SECONDS)
    if polymarket.coverage_ok(df) and closed_for >= TAPE_SETTLE:  # a truncated tape must never be cached as truth
        put_tape(m["start"], df)
    return df


def put_tape(start: int, df: pd.DataFrame) -> None:
    put("tape", encode("tape", df, start))


def underlying(start: int) -> pd.DataFrame:
    if (df := cached_underlying(start)) is not None:
        return df
    lo, hi = start - PAD_BEFORE, start + WINDOW_SECONDS + PAD_AFTER
    df = binance.fetch_seconds(lo, hi)
    if time.time() >= hi + 5:  # its range has ended; before that, seconds still to come are forward-filled
        put_klines(df)
    return df


def put_klines(df: pd.DataFrame) -> None:
    if df["close"].isna().any():
        raise ValueError("klines with no price yet are not cached")
    put("binance_1s", encode("binance_1s", df))


# --- inventory ----------------------------------------------------------------------------

def _keys(kind: str) -> np.ndarray:
    """Every stored key of a kind (window starts; seconds for klines), read from the key column alone.

    Keys are made unique per file before anything is kept: a tape file repeats its window start on every
    fill, and concatenating 65 M such rows before np.unique cost 2 GB (robustness follow-up, 2026-09-17)."""
    out = []
    for day in days(kind):
        for attempt in range(3):                              # a piece consolidated since listing: list the day again
            try:
                keys = [pc.unique(pq.read_table(p, columns=[KEY[kind]]).column(0)).to_numpy() for p in _day_files(kind, day)]
                break
            except FileNotFoundError:
                keys = None
        if keys is None:
            raise RuntimeError(f"{kind} {day}: files kept changing while listing")
        out.extend(keys)
    return np.unique(np.concatenate(out)) if out else np.array([], np.int64)


def markets_frame() -> pd.DataFrame:
    """Every cached market (history and legacy), one row per start, sorted."""
    frames = [read_day("markets", d) for d in days("markets")]
    rows = [decode_market(r) for f in frames for r in f.to_dict("records")]
    have = {r["start"] for r in rows}
    if (DATA_DIR / "markets").exists():
        rows += [_market_dict(pd.read_parquet(p).iloc[0].to_dict()) for p in sorted((DATA_DIR / "markets").glob("*.parquet"))
                 if int(p.stem) not in have]
    return pd.DataFrame(rows, columns=MARKET_KEYS).sort_values("start", kind="stable").reset_index(drop=True)


def tape_starts() -> set[int]:
    out = set(_keys("tape").tolist())
    if (DATA_DIR / "tape").exists():
        out |= {int(p.stem) for p in (DATA_DIR / "tape").glob("*.parquet")}
    return out


def _runs(s: np.ndarray) -> list[tuple[int, int]]:
    if not len(s):
        return []
    breaks = np.flatnonzero(np.diff(s) != 1)
    return list(zip(s[np.r_[0, breaks + 1]].tolist(), (s[np.r_[breaks, len(s) - 1]] + 1).tolist()))


def kline_intervals() -> list[tuple[int, int]]:
    """Maximal runs [lo, hi) of cached seconds, built file by file and merged: holding every cached second
    at once (15 M rows for six months) was most of dataset.starts()' memory."""
    runs: list[tuple[int, int]] = []
    for day in days("binance_1s"):
        for attempt in range(3):
            try:
                pieces = [_runs(np.unique(pq.read_table(p, columns=[KEY["binance_1s"]]).column(0).to_numpy()))
                          for p in _day_files("binance_1s", day)]
                break
            except FileNotFoundError:
                pieces = None
        if pieces is None:
            raise RuntimeError(f"binance_1s {day}: files kept changing while listing")
        for piece in pieces:
            runs.extend(piece)
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(runs):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def underlying_starts(starts, runs: list[tuple[int, int]] | None = None, legacy: bool = True) -> set[int]:
    """The windows among `starts` whose PAD_BEFORE..PAD_AFTER klines are cached (in the history, or as legacy files)."""
    runs = kline_intervals() if runs is None else runs
    los = np.array([a for a, _ in runs], np.int64)
    out = set()
    for s in starts:
        lo, hi = s - PAD_BEFORE, s + WINDOW_SECONDS + PAD_AFTER
        i = np.searchsorted(los, lo, "right") - 1
        if (i >= 0 and runs[i][1] >= hi) or (legacy and _legacy("binance_1s", s).exists()):
            out.add(int(s))
    return out


def has(kind: str, start: int) -> bool:
    """Is window `start` cached for `kind` (klines: all PAD_BEFORE..PAD_AFTER seconds)?"""
    if kind == "binance_1s":
        return klines(start - PAD_BEFORE, start + WINDOW_SECONDS + PAD_AFTER) is not None or _legacy(kind, start).exists()
    return len(_window_rows(kind, start)) > 0 or _legacy(kind, start).exists()


# --- migration ----------------------------------------------------------------------------

def migrate(verify_only: bool = False, log=print) -> dict:
    """Copy legacy per-window files into the history (days at a time) and verify each window through the API.

    Returns counts and the windows that failed. Legacy files are never deleted here."""
    starts = sorted(int(p.stem) for p in (DATA_DIR / "markets").glob("*.parquet")) if (DATA_DIR / "markets").exists() else []
    tapes = {int(p.stem) for p in (DATA_DIR / "tape").glob("*.parquet")} if (DATA_DIR / "tape").exists() else set()
    kl = {int(p.stem) for p in (DATA_DIR / "binance").glob("*.parquet")} if (DATA_DIR / "binance").exists() else set()
    report = {"markets": 0, "tape": 0, "binance_1s": 0, "failed": []}
    market_set = set(starts)
    by_day: dict[str, list[int]] = {}
    for s in sorted(set(starts) | tapes | kl):
        by_day.setdefault(day_name(s), []).append(s)
    for day, ss in by_day.items():
        if not verify_only:
            have = set(read_day("markets", day)["start"])
            ms = [pd.read_parquet(_legacy("markets", s)).iloc[0].to_dict() for s in ss if s in market_set and s not in have]
            put_markets(ms)
            have = set(read_day("tape", day)["start"])
            if todo := [s for s in ss if s in tapes and s not in have]:
                put("tape", pa.concat_tables([encode("tape", pd.read_parquet(_legacy("tape", s)), s) for s in todo]))
            frames = [pd.read_parquet(_legacy("binance_1s", s)) for s in ss if s in kl]
            if frames:
                # windows overlap by 240 s; the later window wins a second they disagree on (the verification says)
                allk = pd.concat(frames, ignore_index=True).drop_duplicates("sec", keep="last")
                allk = allk[~allk["sec"].isin(read_day("binance_1s", day)["sec"])].sort_values("sec")
                if len(allk):
                    put("binance_1s", encode("binance_1s", allk))
            for kind in KINDS:
                consolidate(kind, day)
        for s in ss:
            ok = True
            if s in market_set:
                legacy = _market_dict(pd.read_parquet(_legacy("markets", s)).iloc[0].to_dict())
                rows = _window_rows("markets", s)
                ok &= len(rows) == 1 and same_market(legacy, decode_market(rows.iloc[0].to_dict()))
                report["markets"] += ok
            if s in tapes:
                rows = _window_rows("tape", s)
                good = len(rows) > 0 and decode_tape(rows, s).equals(pd.read_parquet(_legacy("tape", s)))
                report["tape"] += good
                ok &= good
            if s in kl:
                legacy = pd.read_parquet(_legacy("binance_1s", s))
                got = klines(s - PAD_BEFORE, s + WINDOW_SECONDS + PAD_AFTER)
                good = got is not None and got.equals(legacy)
                report["binance_1s"] += good
                ok &= good
            if not ok:
                report["failed"].append(s)
        log(f"{day}: {len(ss)} windows, {len(report['failed'])} failed so far")
    if not verify_only:
        consolidate_all(min_age=-86400)   # klines of windows near midnight left pieces in the neighbouring day
    return report


def delete_legacy(log=print) -> int:
    """Delete legacy per-window files whose window verifies against the history. Returns files deleted."""
    n = 0
    for kind in KINDS:
        folder = DATA_DIR / LEGACY[kind]
        for p in sorted(folder.glob("*.parquet")) if folder.exists() else []:
            s = int(p.stem)
            if kind == "markets":
                rows = _window_rows("markets", s)
                ok = len(rows) == 1 and same_market(_market_dict(pd.read_parquet(p).iloc[0].to_dict()),
                                                    decode_market(rows.iloc[0].to_dict()))
            elif kind == "tape":
                rows = _window_rows("tape", s)
                ok = len(rows) > 0 and decode_tape(rows, s).equals(pd.read_parquet(p))
            else:
                got = klines(s - PAD_BEFORE, s + WINDOW_SECONDS + PAD_AFTER)
                ok = got is not None and got.equals(pd.read_parquet(p))
            if ok:
                p.unlink()
                n += 1
            else:
                log(f"kept {p}: not identical in the history")
    return n


def sizes() -> dict[str, dict]:
    out = {}
    for kind in KINDS:
        files = list(history(kind).glob("day=*/*.parquet")) if history(kind).exists() else []
        legacy = list((DATA_DIR / LEGACY[kind]).glob("*.parquet")) if (DATA_DIR / LEGACY[kind]).exists() else []
        out[kind] = {"days": len(days(kind)), "bytes": sum(p.stat().st_size for p in files),
                     "legacy_files": len(legacy), "legacy_bytes": sum(p.stat().st_size for p in legacy)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The day-partitioned history cache.")
    ap.add_argument("--migrate", action="store_true", help="copy legacy per-window files in and verify every window")
    ap.add_argument("--verify", action="store_true", help="verify legacy windows against the history only")
    ap.add_argument("--delete-legacy", action="store_true", help="delete legacy files identical in the history")
    ap.add_argument("--consolidate", action="store_true", help="fold pieces of finished days into data.parquet")
    ap.add_argument("--sizes", action="store_true")
    a = ap.parse_args()
    if a.migrate or a.verify:
        r = migrate(verify_only=a.verify)
        print({k: v for k, v in r.items() if k != "failed"}, "failed:", len(r["failed"]), r["failed"][:20])
    if a.consolidate:
        print(consolidate_all())
    if a.delete_legacy:
        print("deleted", delete_legacy())
    if a.sizes:
        for kind, s in sizes().items():
            print(f"{kind:>11} {s}")
