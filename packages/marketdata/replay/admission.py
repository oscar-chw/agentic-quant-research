"""Offline raw-feed admission with recorder clocks and explicit continuity.

This is a boundary around replay.feed's book engine, not a socket collector or
an exchange sequence-gap detector. Supplied timestamps are not authenticated.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from replay.book import PRICE_SCALE, SIDE_ASK, SIDE_BID, SIZE_SCALE
from replay.feed import Feed, scale_price, scale_size

MAX_INPUT = 16 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_LEVELS = 10_000
MAX_INTEGER = 2**53
UNAVAILABLE = {
    "GAP",
    "AWAITING_SNAPSHOT",
    "OUT_OF_ORDER",
    "NO_TWO_SIDED_DEPTH",
    "CROSSED_BOOK",
}


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def adapter_identity():
    base = Path(__file__).parent
    return digest(
        canonical(
            {
                n: digest((base / n).read_bytes())
                for n in ("admission.py", "feed.py", "book.py")
            }
        )
    )


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON field")
        out[key] = value
    return out


def _constant(value):
    raise ValueError("nonfinite JSON value: " + value)


def _integer(value, label):
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        raise ValueError("bounded unsigned integer required: " + label)
    return value


def _scaled(value, scale, limit):
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{1,16}(?:\.[0-9]{1,8})?", value
    ):
        raise ValueError("bounded decimal string required")
    scaled = Fraction(value) * scale
    if scaled.denominator != 1 or not 0 <= scaled <= limit:
        raise ValueError("numeric value outside exact grid or range")
    if (scale_price(value) if scale == PRICE_SCALE else scale_size(value)) != int(
        scaled
    ):
        raise ValueError("feed conversion disagrees with exact decimal value")
    return int(scaled)


def _message(message, instrument):
    if not isinstance(message, dict):
        raise TypeError("message object required")
    stamp = message.get("timestamp")
    if not isinstance(stamp, str) or not re.fullmatch(r"[0-9]{1,16}", stamp):
        raise ValueError("explicit event timestamp string required")
    event = _integer(int(stamp), "timestamp")
    kind = message.get("event_type")
    if kind == "book":
        if message.get("asset_id") != instrument:
            raise ValueError("single declared instrument required")
        levels = []
        for name in ("bids", "asks"):
            values = message.get(name)
            if not isinstance(values, list):
                raise TypeError("explicit bid/ask level lists required")
            if len(values) > MAX_LEVELS:
                raise ValueError("snapshot exceeds level bound")
            seen = set()
            for row in values:
                if not isinstance(row, dict) or not {"price", "size"} <= row.keys():
                    raise ValueError("price/size level required")
                price = _scaled(row["price"], PRICE_SCALE, PRICE_SCALE)
                _scaled(row["size"], SIZE_SCALE, MAX_INTEGER)
                if price in seen:
                    raise ValueError("duplicate snapshot level")
                seen.add(price)
                levels.append(row)
        if len(levels) > MAX_LEVELS:
            raise ValueError("snapshot exceeds total level bound")
    elif kind == "price_change":
        entries = message.get("price_changes")
        if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_LEVELS:
            raise ValueError("nonempty bounded price_changes required")
        for row in entries:
            if not isinstance(row, dict) or row.get("asset_id") != instrument:
                raise ValueError("single declared instrument required")
            if row.get("side") not in {"BUY", "SELL"}:
                raise ValueError("exact BUY/SELL side required")
            _scaled(row.get("price"), PRICE_SCALE, PRICE_SCALE)
            _scaled(row.get("size"), SIZE_SCALE, MAX_INTEGER)
    else:
        raise ValueError("unsupported event; do not silently discard archive records")
    return event, kind


def admit(raw: bytes):
    """Return header, quote states and diagnostics in recorded processing order.

    O(records * current book depth) for top-level extraction; bounded memory.
    The underlying Book keeps aggregate sizes, including zero-size deletion.
    """
    if len(raw) > MAX_INPUT:
        raise ValueError("archive exceeds 16 MiB")
    lines = raw.decode("utf-8").splitlines()
    if not 2 <= len(lines) <= MAX_RECORDS + 1:
        raise ValueError("archive needs header and 1..100000 records")
    decoded = [
        json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
        for line in lines
    ]
    header = decoded[0]
    required = {"schema", "instrument", "data_kind", "clock_evidence"}
    if not isinstance(header, dict) or set(header) not in (
        required,
        required | {"origin"},
    ):
        raise ValueError("exact quant-feed-archive/v1 header required")
    if header["schema"] != "quant-feed-archive/v1" or header["data_kind"] not in {
        "synthetic",
        "user-provided",
    }:
        raise ValueError("unsupported archive schema/data kind")
    if header["clock_evidence"] not in {
        "PROVIDED_CLOCKS_UNAUDITED",
        "SYNTHETIC_ZERO_LATENCY",
    }:
        raise ValueError("explicit clock evidence required")
    if (
        header["clock_evidence"] == "SYNTHETIC_ZERO_LATENCY"
        and header["data_kind"] != "synthetic"
    ):
        raise ValueError("zero-latency bridge is synthetic only")
    if "origin" in header:
        origin = header["origin"]
        if (
            header["clock_evidence"] != "SYNTHETIC_ZERO_LATENCY"
            or not isinstance(origin, dict)
            or set(origin) != {"raw_sha256", "snapshot_sha256"}
            or any(
                not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                for v in origin.values()
            )
        ):
            raise ValueError("synthetic bridge origin requires raw/snapshot SHA256")
    instrument = header["instrument"]
    if not isinstance(instrument, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,80}", instrument
    ):
        raise ValueError("safe single instrument required")
    state, rows, counts = Feed(), [], Counter()
    received_before = available_before = -1
    high_water = floor = 0
    segment, based = 0, False
    previous_status = "AWAITING_SNAPSHOT"
    for index, envelope in enumerate(decoded[1:]):
        if not isinstance(envelope, dict):
            raise TypeError("archive record object required")
        kind = envelope.get("kind")
        fields = {"record_id", "kind", "received_ms", "available_ms"}
        if kind not in {"message", "gap"} or set(envelope) != fields | (
            {"message"} if kind == "message" else {"reason"}
        ):
            raise ValueError("exact message/gap envelope fields required")
        if _integer(envelope["record_id"], "record_id") != index:
            raise ValueError("contiguous collector record IDs required")
        received = _integer(envelope["received_ms"], "received_ms")
        available = _integer(envelope["available_ms"], "available_ms")
        if not received_before <= received <= available or available < available_before:
            raise ValueError(
                "receipt/availability must be nondecreasing and received <= available"
            )
        received_before, available_before = received, available
        row = {
            "instrument": instrument,
            "event_ms": None,
            "received_ms": received,
            "available_ms": available,
            "sequence": index,
            "revision": 0,
            "segment": segment,
            "status": "AWAITING_SNAPSHOT",
            "bid_tick": None,
            "ask_tick": None,
            "bid_size": None,
            "ask_size": None,
        }
        if kind == "gap":
            if envelope["reason"] not in {
                "disconnect",
                "resubscribe",
                "archive_end",
                "operator_gap",
            }:
                raise ValueError("declared gap reason required")
            state, based, floor = Feed(), False, max(floor, received)
            segment += 1
            row["status"] = "GAP"
            counts["gap:" + envelope["reason"]] += 1
        else:
            event, event_kind = _message(envelope["message"], instrument)
            if event > received:
                raise ValueError("event clock must not exceed receipt")
            if (
                header["clock_evidence"] == "SYNTHETIC_ZERO_LATENCY"
                and not event == received == available
            ):
                raise ValueError("synthetic zero-latency clocks disagree")
            row["event_ms"] = event
            if event < max(high_water, floor):
                state, based = Feed(), False
                segment += 1
                row["status"] = "OUT_OF_ORDER"
            elif event_kind == "price_change" and not based:
                row["status"] = "AWAITING_SNAPSHOT"
            else:
                if state.apply(envelope["message"]).disagreed:
                    segment += 1
                    counts["snapshot_disagreement"] += 1
                based = True
                book = state.books[instrument]
                if len(book.levels) > MAX_LEVELS:
                    raise ValueError("book exceeds level bound")
                bids = [
                    (tick, size)
                    for (side, tick), size in book.levels.items()
                    if side == SIDE_BID
                ]
                asks = [
                    (tick, size)
                    for (side, tick), size in book.levels.items()
                    if side == SIDE_ASK
                ]
                if not bids or not asks:
                    row["status"] = "NO_TWO_SIDED_DEPTH"
                else:
                    bid, qb = max(bids)
                    ask, qa = min(asks)
                    if not 0 < bid < ask:
                        row["status"] = "CROSSED_BOOK"
                    else:
                        row.update(
                            status="AVAILABLE",
                            bid_tick=bid,
                            ask_tick=ask,
                            bid_size=qb,
                            ask_size=qa,
                        )
                if row["status"] != "AVAILABLE" and previous_status == "AVAILABLE":
                    segment += 1
            high_water = max(high_water, event)
        row["segment"] = segment
        counts[row["status"]] += 1
        previous_status = row["status"]
        rows.append(row)
    return (
        header,
        rows,
        {
            "schema": "feed-admission/v1",
            "records": len(rows),
            "counts": dict(sorted(counts.items())),
            "sequence_kind": "collector-record-order",
            "exchange_completeness_verified": False,
            "clock_evidence": header["clock_evidence"],
            "source_sha256": digest(raw),
            "adapter_sha256": adapter_identity(),
        },
    )
