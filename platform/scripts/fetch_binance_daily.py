#!/usr/bin/env python3
"""Download Binance spot daily (1d) klines from data.binance.vision, checksum-verified.

    python scripts/fetch_binance_daily.py --symbols BTCUSDT ETHUSDT \
        --start 2023-01 --end 2024-12 --out data/binance-1d

Fetches one monthly archive per symbol and month plus its published .CHECKSUM,
refuses any archive whose SHA-256 differs, keeps the raw archives under raw/,
treats a 404 after a symbol's first fetched month as the end of its listing (the
months fetched and the first missing one go to manifest["listing_ended"]; no later
month is asked for), and still raises on a 404 for the first month,
and writes <SYMBOL>.csv with header date,open,high,low,close,volume: the input
format of `factor-research from-ohlcv`. A manifest records every URL and digest.

This makes network requests. It is not run by tests or scripts/check.sh.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/monthly/klines"
MAX_ARCHIVE = 8 * 1024 * 1024


def months(start, end):
    """Inclusive YYYY-MM range."""
    if not all(re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", v) for v in (start, end)) or start > end:
        raise ValueError("start and end must be YYYY-MM with start <= end")
    y, m = map(int, start.split("-"))
    stop = tuple(map(int, end.split("-")))
    out = []
    while (y, m) <= stop:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def archive_url(symbol, month):
    if not re.fullmatch(r"[A-Z0-9]{2,20}", symbol):
        raise ValueError("symbol must be upper-case letters and digits, e.g. BTCUSDT")
    name = f"{symbol}-1d-{month}.zip"
    return f"{BASE}/{symbol}/1d/{name}", f"{BASE}/{symbol}/1d/{name}.CHECKSUM"


def expected_digest(checksum_text, archive_name):
    """Parse '<sha256>  <file name>' and require it to name this archive."""
    parts = checksum_text.strip().split()
    if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]) or parts[1] != archive_name:
        raise ValueError(f"malformed checksum file for {archive_name}")
    return parts[0]


def verify_archive(data, checksum_text, archive_name):
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_digest(checksum_text, archive_name):
        raise ValueError(f"checksum mismatch for {archive_name}; archive refused")
    return digest


def bar_date(open_time):
    """Spot files use milliseconds, and microseconds from 2025; both are integers."""
    if not re.fullmatch(r"\d{12,17}", open_time):
        raise ValueError("open_time must be an integer epoch timestamp")
    scale = 1_000_000 if len(open_time) >= 16 else 1_000
    moment = datetime.fromtimestamp(int(open_time) / scale, tz=timezone.utc)
    if (moment.hour, moment.minute, moment.second, moment.microsecond) != (0, 0, 0, 0):
        raise ValueError("a 1d bar must open at 00:00 UTC")
    return moment.date().isoformat()


def kline_rows(data):
    """[date, open, high, low, close, volume] rows from one archive's single CSV."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("archive must contain exactly one CSV")
        text = archive.read(names[0]).decode("utf-8")
    rows = []
    for record in csv.reader(io.StringIO(text)):
        if not record or record[0] == "open_time":
            continue
        if len(record) < 6:
            raise ValueError("kline row has fewer than six fields")
        rows.append([bar_date(record[0]), *record[1:6]])
    return rows


def download(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(MAX_ARCHIVE + 1)
    if len(data) > MAX_ARCHIVE:
        raise ValueError(f"{url} exceeds the archive size cap")
    return data


def fetch(symbols, start, end, out, get=download):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    manifest = {"source": BASE, "interval": "1d", "start": start, "end": end, "archives": [], "files": {},
                "listing_ended": {}}
    for symbol in symbols:
        bars, fetched = {}, []
        for month in months(start, end):
            url, checksum_url = archive_url(symbol, month)
            name = url.rsplit("/", 1)[1]
            try:
                data = get(url)
            except urllib.error.HTTPError as exc:
                if exc.code != 404 or not fetched:
                    raise
                # No archive after months that had one: the pair stopped trading (a delisting).
                manifest["listing_ended"][symbol] = {"last_month": fetched[-1], "first_missing_month": month}
                break
            checksum = get(checksum_url).decode("utf-8")
            digest = verify_archive(data, checksum, name)
            (out / "raw" / name).write_bytes(data)
            (out / "raw" / (name + ".CHECKSUM")).write_text(checksum)
            for row in kline_rows(data):
                if row[0] in bars:
                    raise ValueError(f"{symbol}: duplicate bar {row[0]}")
                bars[row[0]] = row
            manifest["archives"].append({"url": url, "sha256": digest})
            fetched.append(month)
        text = io.StringIO(newline="")
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        writer.writerows(bars[d] for d in sorted(bars))
        (out / f"{symbol}.csv").write_text(text.getvalue())
        manifest["files"][f"{symbol}.csv"] = {"bars": len(bars),
                                              "sha256": hashlib.sha256(text.getvalue().encode()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="first month, YYYY-MM")
    parser.add_argument("--end", required=True, help="last month, YYYY-MM")
    parser.add_argument("--out", required=True, help="new directory to create")
    args = parser.parse_args(argv)
    manifest = fetch(args.symbols, args.start, args.end, args.out)
    print(json.dumps(manifest["files"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
