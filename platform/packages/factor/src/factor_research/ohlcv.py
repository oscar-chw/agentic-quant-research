"""Turn a directory of daily OHLCV CSVs into a price CSV plus import contract.

One file per asset, named ``<ASSET>.csv``, with header
``date,open,high,low,close,volume``. Bars are treated as UTC days that close at
23:59:59; the caller declares how long after the close a price became usable.
The output is ordinary input to ``prepare``, which re-checks every contract.
"""
import csv
from datetime import date, timedelta
import io
import math
from pathlib import Path

from .artifacts import json_bytes, read_bounded, write_once
from .contracts import decode_json, timestamp

HEADER = ["date", "open", "high", "low", "close", "volume"]
CLOSE_TIME = "23:59:59"
EXPERIMENT = {"available_delay_seconds", "lookback", "cost_bps", "candidates", "splits"}


def _number(value, label, where):
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{where}: {label} is not a number") from exc
    if not math.isfinite(number) or number < 0 or (label != "volume" and number == 0):
        raise ValueError(f"{where}: {label} must be finite and positive")
    return number


def read_bars(path):
    """Validated bars of one asset file, keyed by ISO date."""
    reader = csv.reader(io.StringIO(read_bounded(path).decode("utf-8"), newline=""))
    if next(reader, None) != HEADER:
        raise ValueError(f"{path.name}: header must be {','.join(HEADER)}")
    bars = {}
    for number, row in enumerate(reader, 2):
        where = f"{path.name} row {number}"
        if len(row) != len(HEADER):
            raise ValueError(f"{where}: expected {len(HEADER)} fields")
        day = date.fromisoformat(row[0]).isoformat()
        if day != row[0]:
            raise ValueError(f"{where}: date must be YYYY-MM-DD")
        if day in bars:
            raise ValueError(f"{where}: duplicate date")
        o, h, low, c = (_number(v, k, where) for k, v in zip(HEADER[1:5], row[1:5]))
        _number(row[5], "volume", where)
        if not low <= min(o, c) <= max(o, c) <= h:
            raise ValueError(f"{where}: low <= open/close <= high violated")
        bars[day] = c
    if not bars:
        raise ValueError(f"{path.name}: no bars")
    return bars


def ohlcv_import(folder, experiment):
    """Return price CSV bytes, contract bytes and the universe decision."""
    if not isinstance(experiment, dict) or set(experiment) != EXPERIMENT:
        raise ValueError("experiment needs exactly " + ", ".join(sorted(EXPERIMENT)))
    paths = sorted(Path(folder).glob("*.csv"))
    if not paths:
        raise ValueError("no <ASSET>.csv files in the OHLCV directory")
    bars = {p.stem: read_bars(p) for p in paths}
    splits = experiment["splits"]
    first = timestamp(splits["train"]["start"])
    last = timestamp(splits["test"]["end"])
    cutoff = splits["train"]["end"]
    delay = timedelta(seconds=experiment["available_delay_seconds"])
    # Universe membership must be knowable at the end of training (the same rule
    # prepare applies); an asset first usable later would leak the listing decision.
    # Only bars inside the calendar count, as in prepare: a bar before the start
    # is not imported and so cannot establish membership.
    closes = {a: [t for t in (timestamp(d + "T" + CLOSE_TIME + "Z") for d in rows) if t >= first]
              for a, rows in bars.items()}
    known = sorted(a for a, times in closes.items() if times and min(times) + delay <= timestamp(cutoff))
    excluded = sorted(set(bars) - set(known))
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["Date", "Asset", "Close"])
    rows = 0
    for asset in known:
        for day, close in sorted(bars[asset].items()):
            at = timestamp(day + "T" + CLOSE_TIME + "Z")
            if first <= at <= last:
                writer.writerow([day, asset, format(close, ".17g")])
                rows += 1
    contract = {
        "schema_version": "factor-price-import/v1",
        "columns": {"time": "Date", "asset": "Asset", "price": "Close"},
        "timestamps": {"time": {"format": "date", "timezone": "UTC", "time_of_day": CLOSE_TIME}},
        "availability": {"mode": "assumed_delay", "observed_delay_seconds": 0,
                         "available_delay_seconds": experiment["available_delay_seconds"]},
        "universe": {"policy": "known_by_cutoff", "cutoff": cutoff},
        "calendar": {"policy": "fixed_step", "start": splits["train"]["start"],
                     "end": splits["test"]["end"], "step_seconds": 86400},
        "price_semantics": {"kind": "unadjusted_close", "unit": "quote currency per unit of the asset",
                            "adjustment_notes": "Daily bar close from the supplied OHLCV files; "
                                                "no delisting, survivorship or venue qualification."},
        "experiment": {"lookback": experiment["lookback"], "horizon": 1, "cost_bps": experiment["cost_bps"],
                       "candidates": experiment["candidates"], "splits": splits},
    }
    return dict(prices=text.getvalue().encode(), contract=json_bytes(contract),
                assets=known, excluded_assets=excluded, rows=rows)


def write_ohlcv_import(folder, experiment_path, destination):
    result = ohlcv_import(folder, decode_json(read_bounded(Path(experiment_path))))
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/"prices.csv", result["prices"])
    write_once(out/"import-contract.json", result["contract"])
    return dict(directory=str(out), assets=result["assets"],
                excluded_assets=result["excluded_assets"], rows=result["rows"])
