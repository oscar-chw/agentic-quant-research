"""Validate normalized v1 inputs before any accounting or output writes."""
import json
import re
from decimal import Decimal, localcontext

SCHEMA = "imc4-analysis/v1"
MAX_BYTES = 16 * 1024 * 1024
MAX_EVENTS = 10_000


class InputError(ValueError):
    """The input cannot be interpreted under this versioned contract."""


def keys(obj, required, optional=()):
    if not isinstance(obj, dict):
        raise InputError("each record must be an object")
    missing = set(required) - obj.keys()
    extra = obj.keys() - set(required) - set(optional)
    if missing or extra:
        raise InputError(f"missing fields {sorted(missing)}; unknown fields {sorted(extra)}")


def integer(value, name, low=0, high=10**15):
    if type(value) is not int or not low <= value <= high:
        raise InputError(f"{name} must be an integer in [{low}, {high}]")
    return value


def label(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise InputError(f"{name} must be nonempty text of at most 200 characters")
    if any(ord(c) < 32 for c in value):
        raise InputError(f"{name} must not contain control characters")
    return value


def decimal(value, name, positive=False, nonnegative=False):
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or not re.fullmatch(r"-?(0|[1-9][0-9]{0,12})(\.[0-9]{1,8})?", value):
        raise InputError(f"{name} must be a decimal string (at most 8 fractional digits) or integer")
    number = Decimal(value)
    if abs(number) > 10**12 or (positive and number <= 0) or (nonnegative and number < 0):
        raise InputError(f"{name} is outside its permitted range")
    return number


def money(number):
    if number is None:
        return None
    if number == 0:
        return "0"
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_config(config):
    keys(config, ["type", "schema", "label", "data_kind", "currency", "timestamp_unit",
                  "start_timestamp", "initial_cash", "mark_max_age", "markout_horizon", "instruments"])
    if config["type"] != "config" or config["schema"] != SCHEMA:
        raise InputError(f"first record must be config with schema {SCHEMA}")
    if config["data_kind"] not in ("synthetic", "normalized_offline"):
        raise InputError("data_kind must be synthetic or normalized_offline")
    for name in ("label", "currency", "timestamp_unit"):
        label(config[name], name)
    integer(config["start_timestamp"], "start_timestamp")
    integer(config["mark_max_age"], "mark_max_age")
    integer(config["markout_horizon"], "markout_horizon", 1)
    config["initial_cash"] = decimal(config["initial_cash"], "initial_cash")
    instruments = config["instruments"]
    if not isinstance(instruments, dict) or not 1 <= len(instruments) <= 64:
        raise InputError("instruments must contain 1 to 64 instruments")
    for symbol, spec in instruments.items():
        label(symbol, "instrument")
        keys(spec, ["initial_position", "limit"], ["initial_mark"])
        integer(spec["initial_position"], "initial_position", -10**9, 10**9)
        integer(spec["limit"], "limit", 0, 10**9)
        if "initial_mark" in spec:
            spec["initial_mark"] = decimal(spec["initial_mark"], "initial_mark", positive=True)
        elif spec["initial_position"] != 0:
            raise InputError(f"opening position for {symbol} requires initial_mark")
    return config


def parse_event(event, config):
    common = ["type", "timestamp", "sequence", "instrument"]
    if not isinstance(event, dict):
        raise InputError("each event must be an object")
    kind = event.get("type")
    if kind == "quote":
        keys(event, common, ["mark", "bid", "ask", "note"])
        if "mark" in event:
            if "bid" in event or "ask" in event:
                raise InputError("quote requires either mark or bid/ask, not both")
            event["mark"] = decimal(event["mark"], "mark", positive=True)
            event["bid"] = event["ask"] = None
        else:
            if "bid" not in event or "ask" not in event:
                raise InputError("quote requires both bid and ask, or an explicit mark")
            for name in ("bid", "ask"):
                event[name] = decimal(event[name], name, positive=True)
            if event["bid"] > event["ask"]:
                raise InputError("crossed quote: bid exceeds ask")
            event["mark"] = (event["bid"] + event["ask"]) / 2
    elif kind == "fill":
        keys(event, common + ["fill_id", "side", "units", "price"], ["fee", "note"])
        label(event["fill_id"], "fill_id")
        if event["side"] not in ("buy", "sell"):
            raise InputError("side must be buy or sell")
        integer(event["units"], "units", 1, 10**6)
        event["price"] = decimal(event["price"], "price", positive=True)
        event["fee"] = decimal(event.get("fee", 0), "fee", nonnegative=True)
    else:
        raise InputError("event type must be quote or fill")
    integer(event["timestamp"], "timestamp", config["start_timestamp"])
    integer(event["sequence"], "sequence")
    if not isinstance(event["instrument"], str) or event["instrument"] not in config["instruments"]:
        raise InputError("unknown instrument")
    if "note" in event:
        label(event["note"], "note")
    return event


def load(data):
    """Return validated config/events. No imports, networks, sorting or deduplication."""
    if len(data) > MAX_BYTES:
        raise InputError(f"input exceeds {MAX_BYTES} bytes")
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise InputError("input must be UTF-8") from exc
    if not lines or len(lines) > MAX_EVENTS + 1:
        raise InputError(f"input must have a config and at most {MAX_EVENTS} events")
    config = None
    events = []
    previous = None
    fill_ids = set()
    with localcontext() as context:
        context.prec = 50
        for line_number, line in enumerate(lines, 1):
            try:
                obj = json.loads(line, object_pairs_hook=no_duplicate_keys,
                                 parse_constant=lambda s: (_ for _ in ()).throw(InputError(f"invalid JSON constant {s}")))
                if line_number == 1:
                    config = parse_config(obj)
                    previous = (config["start_timestamp"], -1)
                else:
                    event = parse_event(obj, config)
                    if event["type"] == "fill":
                        if event["fill_id"] in fill_ids:
                            raise InputError(f"duplicate fill identity {event['fill_id']}; all retries must be resolved upstream")
                        fill_ids.add(event["fill_id"])
                    key = (event["timestamp"], event["sequence"])
                    if key <= previous:
                        raise InputError("events must be strictly increasing by (timestamp, sequence); input is not sorted automatically")
                    previous = key
                    events.append(event)
            except (ValueError, TypeError, KeyError) as exc:
                raise InputError(f"line {line_number}: {exc}") from exc
    return config, events
