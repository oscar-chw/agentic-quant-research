"""Versioned collector-ordered quote states; no venue sequence is inferred."""

import csv
import io
import re

EXTRA_KEYS = {"sequence_kind", "source_sha256", "adapter_sha256"}
STATUSES = {
    "AVAILABLE",
    "GAP",
    "AWAITING_SNAPSHOT",
    "OUT_OF_ORDER",
    "NO_TWO_SIDED_DEPTH",
    "CROSSED_BOOK",
}


def validate_spec(spec):
    if spec["sequence_kind"] != "collector-record-order":
        raise ValueError("quote-spec/v2 requires explicit collector record order")
    for name in ("source_sha256", "adapter_sha256"):
        if not isinstance(spec[name], str) or not re.fullmatch(
            r"[0-9a-f]{64}", spec[name]
        ):
            raise ValueError("explicit SHA256 required: " + name)


def read_states(raw, spec, columns, max_rows, max_input):
    if len(raw) > max_input:
        raise ValueError("CSV exceeds 16 MiB")
    fields = (*columns, "status", "segment")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if reader.fieldnames != list(fields):
        raise ValueError("quote-state CSV needs exact ordered columns")
    rows, previous_available, previous_received, previous_segment = [], -1, -1, 0
    for index, row in enumerate(reader):
        if (
            index >= max_rows
            or set(row) != set(fields)
            or row["instrument"] != spec["instrument"]
        ):
            raise ValueError("invalid quote-state row count, fields or instrument")
        status = row["status"]
        if status not in STATUSES:
            raise ValueError("explicit supported quote status required")
        for name in (*columns[1:], "segment"):
            value = row[name]
            if (
                value == ""
                and status != "AVAILABLE"
                and name in {"event_ms", "bid_tick", "ask_tick", "bid_size", "ask_size"}
            ):
                row[name] = None
                continue
            if (
                not isinstance(value, str)
                or not re.fullmatch(r"[0-9]{1,16}", value)
                or int(value) > 2**53
            ):
                raise ValueError("bounded unsigned integer state required: " + name)
            row[name] = int(value)
        if row["sequence"] != index or row["revision"] != 0:
            raise ValueError("contiguous collector sequence and revision zero required")
        if (
            not previous_received <= row["received_ms"] <= row["available_ms"]
            or row["available_ms"] < previous_available
        ):
            raise ValueError("nondecreasing received/available clocks required")
        if row["event_ms"] is not None and row["event_ms"] > row["received_ms"]:
            raise ValueError("event <= received required")
        if row["segment"] < previous_segment:
            raise ValueError("continuity segment cannot decrease")
        if (
            rows
            and rows[-1]["status"] == "AVAILABLE"
            and status != "AVAILABLE"
            and row["segment"] == previous_segment
        ):
            raise ValueError("unavailability must break the continuity segment")
        if status == "AVAILABLE":
            if (
                not 0 < row["bid_tick"] < row["ask_tick"]
                or not row["bid_size"]
                or not row["ask_size"]
            ):
                raise ValueError("available state requires positive uncrossed depth")
        elif any(
            row[k] is not None for k in ("bid_tick", "ask_tick", "bid_size", "ask_size")
        ):
            raise ValueError("unavailable state must not contain price/depth values")
        previous_available, previous_received, previous_segment = (
            row["available_ms"],
            row["received_ms"],
            row["segment"],
        )
        rows.append(row)
    if not rows:
        raise ValueError("at least one quote state required")
    return rows
