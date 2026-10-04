"""Versioned, explicit UTC time grid and point-in-time panel contracts."""
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO-8601 strings with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("naive timestamps are forbidden")
    return parsed.astimezone(timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Config:
    assets: tuple
    calendar: tuple
    lookback: int
    cost_bps: float
    candidates: tuple
    splits: dict


@dataclass(frozen=True)
class Observation:
    observed_at: datetime
    available_at: datetime
    price: object
    group: str


def decode_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def read_config(raw):
    data = decode_json(raw)
    fields = {"schema_version", "panel_schema", "assets", "calendar", "lookback",
              "horizon", "cost_bps", "candidates", "splits"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("config fields must exactly match factor-config/v1")
    if data["schema_version"] != "factor-config/v1" or data["panel_schema"] != "factor-panel/v1":
        raise ValueError("unsupported schema version")
    assets = data["assets"]
    if (not isinstance(assets, list) or not 2 <= len(assets) <= 200
            or any(not isinstance(a, str) or not a.strip() for a in assets)
            or len(set(assets)) != len(assets)):
        raise ValueError("assets must be 2..200 unique nonempty names")
    if not isinstance(data["calendar"], list) or not 4 <= len(data["calendar"]) <= 5000:
        raise ValueError("calendar must contain 4..5000 timestamps")
    calendar = tuple(timestamp(t) for t in data["calendar"])
    if any(b <= a for a, b in zip(calendar, calendar[1:])):
        raise ValueError("calendar must be strictly increasing and unique")
    lookback = data["lookback"]
    if type(lookback) is not int or not 1 <= lookback < len(calendar) - 1:
        raise ValueError("lookback must be a positive integer shorter than calendar minus one")
    if type(data["horizon"]) is not int or data["horizon"] != 1:
        raise ValueError("v1 requires horizon=1 grid interval for non-overlapping sleeves")
    cost = data["cost_bps"]
    if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 <= cost <= 10000:
        raise ValueError("cost_bps must be finite in [0,10000]")
    candidates = data["candidates"]
    if (not isinstance(candidates, list) or not candidates
            or any(c not in ("momentum", "reversal") for c in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError("candidates must be unique momentum/reversal baselines")
    split_data = data["splits"]
    if not isinstance(split_data, dict) or set(split_data) != {"train", "validation", "test"}:
        raise ValueError("train, validation and test splits are required")
    splits = {}
    last_end = -1
    index = {t: i for i, t in enumerate(calendar)}
    for name in ("train", "validation", "test"):
        bounds = split_data[name]
        if not isinstance(bounds, dict) or set(bounds) != {"start", "end"}:
            raise ValueError("split requires exactly start/end")
        try:
            start, end = (index[timestamp(bounds[k])] for k in ("start", "end"))
        except KeyError as exc:
            raise ValueError("split endpoints must be in the calendar") from exc
        if not last_end < start < end:
            raise ValueError("splits must be chronological, nonoverlapping and at least two grid points")
        splits[name] = (start, end)
        last_end = end
    return Config(tuple(assets), calendar, lookback, float(cost), tuple(candidates), splits)


def read_panel(raw, config):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    required = {"time", "asset", "observed_at", "available_at", "price"}
    header = reader.fieldnames
    if (header is None or len(header) != len(set(header)) or not required <= set(header)
            or set(header) - required - {"group"}):
        raise ValueError("panel requires time,asset,observed_at,available_at,price and optional group")
    allowed_times, allowed_assets = set(config.calendar), set(config.assets)
    panel = {}
    for number, row in enumerate(reader, 2):
        if number > 200001:
            raise ValueError("panel exceeds 200000-row limit")
        if None in row or any(row[k] is None for k in required):
            raise ValueError("malformed CSV row " + str(number))
        at = timestamp(row["time"])
        asset = row["asset"]
        if at not in allowed_times or asset not in allowed_assets:
            raise ValueError("row outside frozen calendar/universe at row " + str(number))
        key = (at, asset)
        if key in panel:
            raise ValueError("duplicate asset-time at row " + str(number))
        observed, available = timestamp(row["observed_at"]), timestamp(row["available_at"])
        if not at <= observed <= available:
            raise ValueError("time <= observed_at <= available_at required at row " + str(number))
        value = row["price"].strip()
        price = float(value) if value else None
        if price is not None and (not math.isfinite(price) or price <= 0):
            raise ValueError("price must be positive finite or blank at row " + str(number))
        panel[key] = Observation(observed, available, price, row.get("group", ""))
    if not panel:
        raise ValueError("panel is empty")
    return panel
