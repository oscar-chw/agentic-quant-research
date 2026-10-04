"""Independent parser for one pinned community log writer, not an official adapter."""
import csv
import hashlib
import io
import json
import re

from .contracts import InputError, MAX_BYTES, MAX_EVENTS, decimal, integer, keys, label, load, money, no_duplicate_keys

FORMAT = "nabayansaha/prosperity4btest-log"
REVISION = "0094c681f8cd019889761e6431a1a47ea151aaa8"
CONFIG_SCHEMA = "imc4-analysis-community-import/v1"
HEADER = ("day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;"
          "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss").split(";")


def _json(text):
    def reject(value):
        raise InputError(f"invalid JSON constant {value}")
    try:
        return json.loads(text, object_pairs_hook=no_duplicate_keys, parse_float=str, parse_constant=reject)
    except (ValueError, TypeError, RecursionError) as exc:
        raise InputError(f"invalid import JSON: {exc}") from exc


def _text(data):
    try:
        return data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise InputError("import inputs must be UTF-8") from exc


def _csv_int(text, name, low=0):
    if not re.fullmatch(r"-?(0|[1-9][0-9]*)", text):
        raise InputError(f"{name} must be an integer")
    return integer(int(text), name, low)


def _remove_object_trailing_commas(text):
    """Handle the pinned writer's final object comma; never alter quoted text."""
    result = []
    quoted = escaped = False
    for index, char in enumerate(text):
        if quoted:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        if char == ',':
            following = index + 1
            while following < len(text) and text[following].isspace():
                following += 1
            if following < len(text) and text[following] == '}':
                continue
        result.append(char)
    return ''.join(result)


def _config(raw):
    if len(raw) > 65536:
        raise InputError("import configuration exceeds 64 KiB")
    config = _json(_text(raw))
    keys(config, ["schema", "source_format", "producer_revision", "edition", "round", "day", "mode", "self_id",
                  "quantity_unit", "sequence_policy", "fill_identity_policy", "fee_policy", "analysis"])
    expected = {"schema": CONFIG_SCHEMA, "source_format": FORMAT, "producer_revision": REVISION,
                "edition": "prosperity4", "quantity_unit": "integer_units",
                "sequence_policy": "activities_then_trades_source_order",
                "fill_identity_policy": "source_index_refuse_indistinguishable"}
    for field, value in expected.items():
        if config[field] != value:
            raise InputError(f"unsupported {field}; expected {value}")
    if config["mode"] not in ("own_fills", "quotes_only"):
        raise InputError("mode must be own_fills or quotes_only")
    label(config["round"], "round")
    integer(config["day"], "day", -10**9, 10**9)
    label(config["self_id"], "self_id")
    keys(config["fee_policy"], ["per_fill", "basis"])
    decimal(config["fee_policy"]["per_fill"], "per_fill", nonnegative=True)
    label(config["fee_policy"]["basis"], "fee policy basis")
    # Delegate the opening accounting/limits contract to the existing validator.
    normalized_config = (json.dumps(config["analysis"], separators=(",", ":")) + "\n").encode()
    checked, _ = load(normalized_config)
    if checked["currency"] != "XIREC" or checked["timestamp_unit"] != "source_timestamp":
        raise InputError("this format requires XIREC prices and opaque source_timestamp units; no scaling or UTC conversion")
    if config["mode"] == "quotes_only":
        if checked["initial_cash"] != 0 or any(s["initial_position"] != 0 for s in checked["instruments"].values()):
            raise InputError("quotes_only requires explicit zero cash and zero initial positions; strategy P&L is not evaluated")
    return config


def _activities(text, config):
    try:
        rows = list(csv.reader(io.StringIO(text.strip()), delimiter=";", strict=True))
    except csv.Error as exc:
        raise InputError(f"malformed activity CSV: {exc}") from exc
    if not rows or rows[0] != HEADER:
        raise InputError("activity header does not match the pinned 17-column format")
    if not 1 <= len(rows)-1 <= MAX_EVENTS:
        raise InputError("activity log must contain 1 to 10,000 rows")
    events = []
    previous = -1
    seen = set()
    for index, cells in enumerate(rows[1:], 1):
        if len(cells) != len(HEADER):
            raise InputError(f"partial activity row {index}: expected 17 columns")
        row = dict(zip(HEADER, cells))
        if _csv_int(row["day"], "activity day", -10**9) != config["day"]:
            raise InputError("activity day differs from declaration; merged/multiple days are unsupported")
        timestamp = _csv_int(row["timestamp"], "activity timestamp")
        symbol = row["product"]
        if symbol not in config["analysis"]["instruments"]:
            raise InputError(f"undeclared activity instrument {symbol}")
        if timestamp < previous or (timestamp, symbol) in seen:
            raise InputError("activities must be timestamp-ordered with one row per instrument/timestamp")
        previous = timestamp
        seen.add((timestamp, symbol))
        for side in ("bid", "ask"):
            for level in (1, 2, 3):
                price, volume = row[f"{side}_price_{level}"], row[f"{side}_volume_{level}"]
                if bool(price) != bool(volume):
                    raise InputError(f"partial book level in activity row {index}")
                if price:
                    decimal(price, "book price", positive=True)
                    _csv_int(volume, "book volume", 1)
        decimal(row["mid_price"], "recorded mid price", positive=True)
        decimal(row["profit_and_loss"], "source profit_and_loss")
        events.append({"type": "quote", "timestamp": timestamp, "instrument": symbol,
                       "mark": row["mid_price"], "note": f"Imported activity row {index}; source mid_price"})
    return events


def _trades(text, config, raw_hash):
    rows = _json(_remove_object_trailing_commas(text))
    if not isinstance(rows, list) or len(rows) > MAX_EVENTS:
        raise InputError("trade history must be an array of at most 10,000 rows")
    events = []
    previous = -1
    identities = set()
    market_count = own_count = 0
    for index, row in enumerate(rows):
        keys(row, ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"])
        timestamp = integer(row["timestamp"], "trade timestamp")
        if timestamp < previous:
            raise InputError("trade history must be timestamp-ordered")
        previous = timestamp
        if not isinstance(row["symbol"], str) or row["symbol"] not in config["analysis"]["instruments"]:
            raise InputError("undeclared trade instrument")
        if row["currency"] != "XIREC":
            raise InputError("trade currency must be XIREC; no currency conversion")
        for side in ("buyer", "seller"):
            if not isinstance(row[side], str):
                raise InputError("buyer/seller identities must be text")
            if row[side]:
                label(row[side], side)
        price = decimal(row["price"], "trade price", positive=True)
        quantity = integer(row["quantity"], "trade quantity", 1, 10**6)
        buy = row["buyer"] == config["self_id"]
        sell = row["seller"] == config["self_id"]
        if buy and sell:
            raise InputError("both trade sides match self_id; self-trade accounting is ambiguous")
        if not buy and not sell:
            market_count += 1
            continue
        own_count += 1
        identity = (timestamp, row["symbol"], row["buyer"], row["seller"], price, quantity)
        if identity in identities:
            raise InputError("indistinguishable duplicate own trade; source lacks execution IDs; resolve upstream")
        identities.add(identity)
        if config["mode"] == "quotes_only":
            continue
        events.append({"type": "fill", "timestamp": timestamp, "instrument": row["symbol"],
                       "fill_id": f"source-{raw_hash}-trade-{index}", "side": "buy" if buy else "sell",
                       "units": quantity, "price": money(price), "fee": config["fee_policy"]["per_fill"],
                       "note": f"Imported trade row {index}; side by declared self_id; fee by declared policy"})
    if config["mode"] == "own_fills" and own_count == 0:
        raise InputError("no own fills matched self_id; verify identity or explicitly select quotes_only")
    return events, {"trade_rows": len(rows), "own_trade_rows": own_count, "market_trade_rows_excluded": market_count,
                    "own_trade_rows_excluded": own_count if config["mode"] == "quotes_only" else 0}


def normalize_community_log(raw_source, raw_config):
    """Return normalized bytes and provenance. Monetary computation stays in analyzer."""
    if len(raw_source) > MAX_BYTES:
        raise InputError("raw log exceeds the 16-MiB input cap")
    config = _config(raw_config)
    text = _text(raw_source)
    if not text.startswith("Sandbox logs:\n") or text.count("\nActivities log:\n") != 1 or text.count("\nTrade History:\n") != 1:
        raise InputError("expected exactly one Sandbox logs / Activities log / Trade History section")
    _, remaining = text.split("\nActivities log:\n")
    activity_text, trade_text = remaining.split("\nTrade History:\n")
    raw_hash = hashlib.sha256(raw_source).hexdigest()
    quotes = _activities(activity_text, config)
    fills, counts = _trades(trade_text, config, raw_hash)
    if len(quotes) + counts["trade_rows"] > MAX_EVENTS:
        raise InputError("combined raw activity/trade rows exceed the 10,000-row cap")
    # This merge is a declared producer-specific mapping, not an inferred real-time order.
    tagged = [(e["timestamp"], 0, i, e) for i, e in enumerate(quotes)]
    tagged.extend((e["timestamp"], 1, i, e) for i, e in enumerate(fills))
    tagged.sort(key=lambda item: item[:3])
    normalized_rows = [config["analysis"]]
    previous = None
    sequence = 0
    for timestamp, _, _, event in tagged:
        sequence = sequence + 1 if timestamp == previous else 0
        normalized_rows.append(dict(event, sequence=sequence))
        previous = timestamp
    normalized = ("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in normalized_rows) + "\n").encode()
    load(normalized)  # Preserve the existing event/byte/accounting contract, including caps.
    provenance = {"schema": CONFIG_SCHEMA, "source_format": FORMAT, "declared_producer_revision": REVISION,
                  "edition": config["edition"], "round": config["round"], "day": config["day"], "mode": config["mode"],
                  "timestamp_unit": "source_timestamp", "quantity_unit": config["quantity_unit"], "currency": "XIREC",
                  "self_id": config["self_id"], "sequence_policy": config["sequence_policy"],
                  "fill_identity_policy": config["fill_identity_policy"], "fee_policy": config["fee_policy"],
                  "raw_source_sha256": raw_hash, "raw_source_bytes": len(raw_source),
                  "config_sha256": hashlib.sha256(raw_config).hexdigest(), "config_bytes": len(raw_config),
                  "normalized_sha256": hashlib.sha256(normalized).hexdigest(), "normalized_bytes": len(normalized),
                  "activity_rows": len(quotes), "normalized_events": len(normalized_rows)-1, **counts,
                  "strategy_pnl_available": config["mode"] == "own_fills",
                  "limitations": "Declared community writer identity, not authenticated or official compatibility. Source mid_price is the mark; source P&L is not used. Fees are configured, not observed. Source-index IDs do not prove economic uniqueness. Sandbox text/depth remain raw only. No decisions, access fees, conversions or multi-day reconstruction."}
    return normalized, provenance
