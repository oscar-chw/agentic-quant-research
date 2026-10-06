"""Deterministic long-form price import under explicitly declared clocks/policies."""
import csv
from datetime import datetime, timedelta, timezone
import io
import math
from pathlib import Path
import re

from .artifacts import json_bytes, read_bounded, sha, sync_directory, write_once
from .contracts import decode_json, iso, read_config, read_panel, timestamp

MEMBERS = {"raw-prices.csv", "import-contract.json", "panel.csv", "config.json", "import-manifest.json"}
SOURCE_MEMBERS = {"raw-prices.csv", "import-contract.json", "import-manifest.json"}


def exact(obj, keys, label):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError(label+" fields do not match the declared schema")


def fixed_zone(value):
    if value == "UTC":
        return timezone.utc
    if not isinstance(value, str) or not re.fullmatch(r"[+-]\d{2}:\d{2}", value):
        raise ValueError("timezone must be UTC, fixed +/-HH:MM or offset_in_value for ISO")
    hours, minutes = int(value[1:3]), int(value[4:])
    if minutes >= 60 or hours > 14 or (hours == 14 and minutes):
        raise ValueError("invalid fixed UTC offset")
    return timezone(timedelta(minutes=(hours*60+minutes)*(1 if value[0] == "+" else -1)))


def parse_clock(value, encoding):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty timestamp")
    value = value.strip()
    kind = encoding.get("format") if isinstance(encoding, dict) else None
    if kind == "unix":
        exact(encoding, {"format", "unit", "timezone"}, "unix encoding")
        if encoding["timezone"] != "UTC" or encoding["unit"] not in ("s", "ms", "us"):
            raise ValueError("unix requires explicit s/ms/us unit and UTC")
        if not re.fullmatch(r"-?\d+", value):
            raise ValueError("unix timestamp must be an integer in the declared unit")
        try:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)+timedelta(
                microseconds=int(value)*{"s": 1000000, "ms": 1000, "us": 1}[encoding["unit"]])
        except (ValueError, OverflowError) as exc:
            raise ValueError("unix timestamp outside supported datetime range; check unit") from exc
    if kind == "date":
        exact(encoding, {"format", "timezone", "time_of_day"}, "date encoding")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) or not re.fullmatch(r"\d{2}:\d{2}:\d{2}", encoding["time_of_day"]):
            raise ValueError("date requires YYYY-MM-DD and declared HH:MM:SS time_of_day")
        parsed = datetime.strptime(value+"T"+encoding["time_of_day"], "%Y-%m-%dT%H:%M:%S")
        return parsed.replace(tzinfo=fixed_zone(encoding["timezone"])).astimezone(timezone.utc)
    if kind != "iso8601":
        raise ValueError("time format must be date, iso8601 or unix")
    exact(encoding, {"format", "timezone"}, "ISO encoding")
    if "T" not in value and " " not in value:
        raise ValueError("ISO timestamps need a time; use explicit date/time_of_day encoding")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if encoding["timezone"] == "offset_in_value":
        if parsed.tzinfo is None:
            raise ValueError("offset_in_value refuses naive timestamps")
    else:
        zone = fixed_zone(encoding["timezone"])
        if parsed.tzinfo is not None and parsed.utcoffset() != zone.utcoffset(None):
            raise ValueError("timestamp offset conflicts with declared timezone")
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(timezone.utc)


def convert_prices(prices_raw, contract_raw):
    contract = decode_json(contract_raw)
    fields = {"schema_version", "columns", "timestamps", "availability", "universe", "calendar", "experiment", "price_semantics"}
    exact(contract, fields, "import contract")
    if contract["schema_version"] != "factor-price-import/v1":
        raise ValueError("unsupported price import schema")
    columns, clocks, availability = (contract[k] for k in ("columns", "timestamps", "availability"))
    if not isinstance(columns, dict) or not {"time", "asset", "price"} <= set(columns) or set(columns)-{"time", "asset", "price", "group", "observed_at", "available_at"}:
        raise ValueError("explicit time/asset/price column mapping required")
    if any(not isinstance(v, str) or not v for v in columns.values()) or len(set(columns.values())) != len(columns):
        raise ValueError("column mappings must be unique nonempty names")
    if not isinstance(availability, dict):
        raise ValueError("explicit availability mode required")
    mode = availability.get("mode")
    if mode == "columns":
        exact(availability, {"mode"}, "availability")
        if not {"observed_at", "available_at"} <= set(columns):
            raise ValueError("provided-clock mode requires observed_at and available_at columns")
        exact(clocks, {"time", "observed_at", "available_at"}, "timestamp encodings")
        observed_delay = available_delay = None
    elif mode == "assumed_delay":
        exact(availability, {"mode", "observed_delay_seconds", "available_delay_seconds"}, "assumed availability")
        if {"observed_at", "available_at"} & set(columns):
            raise ValueError("assumed clocks cannot silently replace mapped clock columns")
        exact(clocks, {"time"}, "timestamp encodings")
        observed_delay, available_delay = availability["observed_delay_seconds"], availability["available_delay_seconds"]
        if any(type(d) is not int or not 0 <= d <= 31536000 for d in (observed_delay, available_delay)) or available_delay < observed_delay:
            raise ValueError("assumed delays must be integer seconds with 0 <= observed <= available <= one year")
    else:
        raise ValueError("availability must declare columns or assumed_delay; no inferred clocks")
    semantics = contract["price_semantics"]
    exact(semantics, {"kind", "unit", "adjustment_notes"}, "price semantics")
    if semantics["kind"] not in ("price", "unadjusted_close", "adjusted_close") or any(not isinstance(v, str) or not v.strip() for v in semantics.values()):
        raise ValueError("declare price kind, unit and adjustment_notes")
    settings = contract["experiment"]
    exact(settings, {"lookback", "horizon", "cost_bps", "candidates", "splits"}, "experiment")
    calendar_policy = contract["calendar"]
    if not isinstance(calendar_policy, dict):
        raise ValueError("calendar policy required")
    policy = calendar_policy.get("policy")
    if policy not in ("observed_union", "fixed_step"):
        raise ValueError("calendar policy must be observed_union or fixed_step")
    exact(calendar_policy, {"policy", "start", "end"} | ({"step_seconds"} if policy == "fixed_step" else set()), "calendar")
    start, end = timestamp(calendar_policy["start"]), timestamp(calendar_policy["end"])
    if end <= start:
        raise ValueError("calendar range must increase")
    times = set()
    if policy == "fixed_step":
        step = calendar_policy["step_seconds"]
        if type(step) is not int or step <= 0:
            raise ValueError("calendar step_seconds must be a positive integer")
        duration = (end-start).total_seconds()
        if duration % step or duration/step+1 > 5000:
            raise ValueError("fixed calendar must reach end exactly within 5000 grid points")
        times = {start+timedelta(seconds=i*step) for i in range(int(duration/step)+1)}
    reader = csv.DictReader(io.StringIO(prices_raw.decode("utf-8"), newline=""))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != set(columns.values()):
        raise ValueError("CSV header must exactly match mapped columns; no implicit ignored columns")
    rows = {}
    for number, raw in enumerate(reader, 2):
        if number > 200001:
            raise ValueError("price CSV exceeds 200000 rows")
        if None in raw or any(raw[v] is None for v in columns.values()):
            raise ValueError("malformed price CSV row "+str(number))
        at = parse_clock(raw[columns["time"]], clocks["time"])
        if not start <= at <= end or (policy == "fixed_step" and at not in times):
            raise ValueError("price time outside declared calendar/grid; check timezone/unit")
        asset = raw[columns["asset"]]
        if not asset.strip() or asset != asset.strip():
            raise ValueError("asset names must be nonempty without edge whitespace")
        if (at, asset) in rows:
            raise ValueError("duplicate/conflicting asset-time price row")
        if mode == "columns":
            observed, available = (parse_clock(raw[columns[k]], clocks[k]) for k in ("observed_at", "available_at"))
        else:
            observed, available = at+timedelta(seconds=observed_delay), at+timedelta(seconds=available_delay)
        if not at <= observed <= available:
            raise ValueError("time <= observed_at <= available_at required")
        raw_price = raw[columns["price"]].strip()
        price = float(raw_price) if raw_price else None
        if price is not None and (not math.isfinite(price) or price <= 0):
            raise ValueError("price must be positive finite or blank")
        rows[(at, asset)] = (observed, available, price, raw[columns["group"]] if "group" in columns else "")
        if policy == "observed_union":
            times.add(at)
    if not rows:
        raise ValueError("price CSV is empty")
    universe = contract["universe"]
    if not isinstance(universe, dict):
        raise ValueError("explicit universe policy required")
    if universe.get("policy") == "explicit":
        exact(universe, {"policy", "assets"}, "universe")
        assets = universe["assets"]
        if not isinstance(assets, list) or any(not isinstance(a, str) for a in assets) or len(assets) != len(set(assets)):
            raise ValueError("explicit universe assets must be unique names")
        assets = sorted(assets)
    elif universe.get("policy") == "known_by_cutoff":
        exact(universe, {"policy", "cutoff"}, "universe")
        cutoff = timestamp(universe["cutoff"])
        train_start, train_end = (timestamp(settings["splits"]["train"][k]) for k in ("start", "end"))
        if not max(start, train_start) <= cutoff <= train_end:
            raise ValueError("universe cutoff must be inside the training period/calendar range")
        assets = sorted({a for (t, a), row in rows.items() if t <= cutoff and row[1] <= cutoff})
    else:
        raise ValueError("universe policy must be explicit or known_by_cutoff; no full-sample inference")
    allowed_assets = set(assets)
    if any(asset not in allowed_assets for _, asset in rows):
        raise ValueError("row outside declared/training-cutoff universe; do not infer test-only assets")
    config = dict(schema_version="factor-config/v1", panel_schema="factor-panel/v1", assets=assets,
                  calendar=[iso(t) for t in sorted(times)], **settings)
    config_raw = json_bytes(config)
    parsed_config = read_config(config_raw)  # reuse, never relax the evaluator's contract
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["time", "asset", "observed_at", "available_at", "price", "group"])
    for (at, asset), (observed, available, price, group) in sorted(rows.items()):
        writer.writerow([iso(at), asset, iso(observed), iso(available), format(price, ".17g") if price is not None else "", group])
    panel_raw = text.getvalue().encode()
    read_panel(panel_raw, parsed_config)
    source = dict(availability_status="ASSUMED_DELAY" if mode == "assumed_delay" else "PROVIDED_CLOCKS_UNAUDITED",
                  historically_point_in_time_verified=False, availability=availability, timestamps=clocks,
                  universe=universe, calendar=calendar_policy, price_semantics=semantics,
                  warnings=["Availability is an explicit model, not historical point-in-time evidence." if mode == "assumed_delay"
                            else "Input clocks are supplied, but their historical provenance is not independently verified.",
                            "No corporate-action, exchange-calendar, survivorship or executable-close qualification.",
                            "Observed-union grids cannot detect dates missing for every asset." if policy == "observed_union"
                            else "Fixed-step grid is user-declared, not an inferred exchange calendar."])
    files = {"raw-prices.csv": prices_raw, "import-contract.json": contract_raw, "panel.csv": panel_raw, "config.json": config_raw}
    files["import-manifest.json"] = json_bytes(dict(schema_version="factor-import/v1", files={k: sha(v) for k, v in files.items()},
                                                  data_source=source, rows=len(rows), missing_rows=len(times)*len(assets)-len(rows),
                                                  null_prices=sum(r[2] is None for r in rows.values())))
    if any(len(raw) > 16*1024*1024 for raw in files.values()):
        raise ValueError("prepared artifact exceeds the supported 16 MiB file limit")
    return files


def validate_prepared(files):
    if set(files) != MEMBERS:
        raise ValueError("incomplete or unexpected prepared bundle members")
    manifest = decode_json(files["import-manifest.json"])
    if manifest.get("schema_version") != "factor-import/v1" or manifest.get("files") != {k: sha(v) for k, v in files.items() if k != "import-manifest.json"}:
        raise ValueError("prepared input/contract/output hash mismatch")
    expected = convert_prices(files["raw-prices.csv"], files["import-contract.json"])
    if files != expected:
        raise ValueError("prepared output or assumptions differ from deterministic import")
    return manifest


def load_prepared(folder):
    folder = Path(folder)
    if folder.is_symlink() or {p.name for p in folder.iterdir()} != MEMBERS or any(p.is_symlink() for p in folder.iterdir()):
        raise ValueError("incomplete, unexpected or symlinked prepared bundle")
    files = {name: read_bounded(folder/name) for name in MEMBERS}
    validate_prepared(files)
    return files


def prepare_prices(prices_path, contract_path, destination):
    files = convert_prices(read_bounded(Path(prices_path)), read_bounded(Path(contract_path)))
    folder = Path(destination)
    try:
        folder.mkdir(parents=True)
    except FileExistsError:
        if load_prepared(folder) != files:
            raise ValueError("conflicting prepared destination; existing files preserved")
        return dict(directory=str(folder), status="PREPARED", reused=True)
    for name in sorted(MEMBERS-{"import-manifest.json"}):
        write_once(folder/name, files[name])
    write_once(folder/"import-manifest.json", files["import-manifest.json"])
    sync_directory(folder)
    manifest = validate_prepared({name: read_bounded(folder/name) for name in MEMBERS})
    return dict(directory=str(folder), status="PREPARED", reused=False, rows=manifest["rows"],
                availability_status=manifest["data_source"]["availability_status"])
