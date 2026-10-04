"""Availability-aware top-of-book features and a fixed quote-forecast experiment.

All price calculations use rational ticks. This predicts a later observed quote,
not an executable price, and makes no claim to implement Stoikov's micro-price.
"""

from __future__ import annotations

import csv
import io
import json
import re
from fractions import Fraction
from itertools import pairwise

from .contracts import _reject_json_constant, _unique_object
from .quote_states import EXTRA_KEYS, read_states, validate_spec

COLUMNS = (
    "instrument",
    "event_ms",
    "received_ms",
    "available_ms",
    "sequence",
    "revision",
    "bid_tick",
    "ask_tick",
    "bid_size",
    "ask_size",
)
FEATURE_KEYS = {
    "schema",
    "instrument",
    "data_kind",
    "price_tick",
    "start_ms",
    "step_ms",
    "count",
    "max_age_ms",
}
MAX_ROWS = 100_000
MAX_INPUT = 16 * 1024 * 1024


def strict_json(raw: bytes):
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )


def feature_spec(spec: dict) -> dict:
    extras = (
        EXTRA_KEYS
        if isinstance(spec, dict) and spec.get("schema") == "quote-spec/v2"
        else set()
    )
    if not isinstance(spec, dict) or set(spec) != FEATURE_KEYS | extras | {
        "train_count",
        "validation_count",
    }:
        raise ValueError("exact versioned quote-spec fields required")
    if spec["schema"] not in {"quote-spec/v1", "quote-spec/v2"} or spec[
        "data_kind"
    ] not in {
        "synthetic",
        "user-provided",
    }:
        raise ValueError("unsupported quote schema or data kind")
    if not isinstance(spec["instrument"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,80}", spec["instrument"]
    ):
        raise ValueError("safe, explicit single instrument required")
    for key in (
        "start_ms",
        "step_ms",
        "count",
        "max_age_ms",
        "train_count",
        "validation_count",
    ):
        if type(spec[key]) is not int or not 0 <= spec[key] <= 2**53:
            raise ValueError("nonnegative integer required: " + key)
    if not 6 <= spec["count"] <= MAX_ROWS or not spec["step_ms"]:
        raise ValueError("require 6..100000 decisions and a positive step")
    if (
        min(
            spec["train_count"],
            spec["validation_count"],
            spec["count"] - spec["train_count"] - spec["validation_count"],
        )
        < 2
    ):
        raise ValueError("each chronological split needs at least two decisions")
    if spec["start_ms"] + spec["step_ms"] * spec["count"] > 2**53:
        raise ValueError("decision clock exceeds supported integer range")
    if (
        not isinstance(spec["price_tick"], str)
        or not re.fullmatch(r"[0-9]{1,12}(?:\.[0-9]{1,12})?", spec["price_tick"])
        or Fraction(spec["price_tick"]) <= 0
    ):
        raise ValueError("explicit positive decimal price tick required")
    if extras:
        validate_spec(spec)
    return {k: spec[k] for k in sorted(FEATURE_KEYS | extras)}


def read_quotes(raw: bytes, spec: dict) -> list[dict]:
    if spec["schema"] == "quote-spec/v2":
        return read_states(raw, spec, COLUMNS, MAX_ROWS, MAX_INPUT)
    if len(raw) > MAX_INPUT:
        raise ValueError("CSV exceeds 16 MiB")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if reader.fieldnames != list(COLUMNS):
        raise ValueError("quote CSV needs the exact ordered columns")
    identities = {}
    sequence_times = {}
    for i, row in enumerate(reader, 1):
        if i > MAX_ROWS:
            raise ValueError("CSV exceeds 100000 observations")
        if set(row) != set(COLUMNS) or row["instrument"] != spec["instrument"]:
            raise ValueError("wrong columns or instrument")
        for name in COLUMNS[1:]:
            text = row[name]
            if not isinstance(text, str) or not re.fullmatch(r"[0-9]{1,16}", text):
                raise ValueError("unsigned integer quote field required: " + name)
            row[name] = int(text)
            if row[name] > 2**53:
                raise ValueError("quote field exceeds supported integer range")
        if not row["event_ms"] <= row["received_ms"] <= row["available_ms"]:
            raise ValueError("clocks must satisfy event <= received <= available")
        if not 0 < row["bid_tick"] < row["ask_tick"]:
            raise ValueError("positive uncrossed two-sided quote required")
        identity = row["sequence"], row["revision"]
        if identity in identities and identities[identity] != row:
            raise ValueError("conflicting sequence/revision identity")
        identities[identity] = row
        seq = row["sequence"]
        if seq in sequence_times and sequence_times[seq] != row["event_ms"]:
            raise ValueError("a revision cannot change its original event clock")
        sequence_times[seq] = row["event_ms"]
    if not identities:
        raise ValueError("at least one observation required")
    times = [sequence_times[s] for s in sorted(sequence_times)]
    if any(a > b for a, b in pairwise(times)):
        raise ValueError("venue sequence disagrees with event-clock order")
    return sorted(
        identities.values(),
        key=lambda r: (r["available_ms"], r["sequence"], r["revision"]),
    )


def feature_row(row: dict | None, decision: int, max_age: int) -> dict:
    result = {"decision_ms": decision, "status": "NO_QUOTE"}
    if row is None:
        return result
    result.update(
        {
            k: row[k]
            for k in ("sequence", "revision", "event_ms", "received_ms", "available_ms")
        }
    )
    if "segment" in row:
        result.update(segment=row["segment"], sequence_kind="collector-record-order")
        if row["status"] != "AVAILABLE":
            return result | {"status": row["status"]}
    if decision - row["event_ms"] > max_age:
        return result | {"status": "STALE"}
    bid, ask, qb, qa = (
        row[k] for k in ("bid_tick", "ask_tick", "bid_size", "ask_size")
    )
    if not qb or not qa:
        return result | {"status": "NO_TWO_SIDED_DEPTH"}
    spread = ask - bid
    imbalance = Fraction(qb - qa, qb + qa)
    midpoint = Fraction(bid + ask, 2)
    return result | {
        "status": "AVAILABLE",
        "spread_ticks": spread,
        "imbalance": str(imbalance),
        "midpoint_ticks": str(midpoint),
        "weighted_midpoint_ticks": str(midpoint + imbalance * spread / 2),
    }


def materialize(rows: list[dict], spec: dict) -> list[dict]:
    """One availability-ordered sweep; a delayed older sequence cannot roll back state."""
    output, position, current = [], 0, None
    for i in range(spec["count"]):
        decision = spec["start_ms"] + i * spec["step_ms"]
        while position < len(rows) and rows[position]["available_ms"] <= decision:
            row = rows[position]
            if current is None or (row["sequence"], row["revision"]) > (
                current["sequence"],
                current["revision"],
            ):
                current = row
            position += 1
        output.append(feature_row(current, decision, spec["max_age_ms"]))
    return output


def evaluate(features: list[dict], spec: dict) -> tuple[dict, list[dict]]:
    """Use stored features directly. Fixed forecasts; no fitting or winner selection."""
    boundaries = (
        0,
        spec["train_count"],
        spec["train_count"] + spec["validation_count"],
        spec["count"],
    )
    metrics, pairs = {}, []
    for split, (lo, hi) in zip(
        ("train", "validation", "test"), pairwise(boundaries), strict=True
    ):
        errors = {"midpoint": [], "weighted_midpoint": []}
        excluded = {}
        for i in range(lo, hi - 1):
            a, b = features[i : i + 2]
            item = {
                "split": split,
                "decision_ms": a["decision_ms"],
                "target_decision_ms": b["decision_ms"],
            }
            if a["status"] != "AVAILABLE" or b["status"] != "AVAILABLE":
                reason = (
                    "input:" + a["status"]
                    if a["status"] != "AVAILABLE"
                    else "target:" + b["status"]
                )
                excluded[reason] = excluded.get(reason, 0) + 1
                pairs.append(item | {"status": "EXCLUDED", "reason": reason})
                continue
            if spec["schema"] == "quote-spec/v2" and a["segment"] != b["segment"]:
                reason = "continuity_break"
                excluded[reason] = excluded.get(reason, 0) + 1
                pairs.append(item | {"status": "EXCLUDED", "reason": reason})
                continue
            target = Fraction(b["midpoint_ticks"])
            result = item | {
                "status": "EVALUATED",
                "target_ticks": str(target),
                "target_available_ms": b["available_ms"],
            }
            for model in errors:
                prediction = Fraction(a[model + "_ticks"])
                error = prediction - target
                errors[model].append(error)
                result[model] = {
                    "forecast_ticks": str(prediction),
                    "error_ticks": str(error),
                }
            pairs.append(result)
        count = len(errors["midpoint"])
        metrics[split] = {
            "candidate_pairs": hi - lo - 1,
            "evaluated_pairs": count,
            "excluded": excluded,
            "coverage": str(Fraction(count, hi - lo - 1)),
            "models": {
                model: {
                    "mae_ticks": str(sum(map(abs, es), Fraction()) / count)
                    if count
                    else None,
                    "mse_ticks_squared": str(
                        sum((e * e for e in es), Fraction()) / count
                    )
                    if count
                    else None,
                }
                for model, es in errors.items()
            },
        }
    report = {
        "schema": "quote-forecast-report/v1",
        "evidence_tier": "E0",
        "data_kind": spec["data_kind"],
        "historically_point_in_time_verified": False,
        "availability": "PROVIDED_CLOCKS_UNAUDITED",
        "price_tick": spec["price_tick"],
        "target": "next decision as-of observed midpoint; stale/missing pairs excluded",
        "selection": "none; two fixed forecasts, no fitted parameters",
        "profitability_evaluated": False,
        "splits": metrics,
    }
    if spec["schema"] == "quote-spec/v2":
        report.update(
            schema="quote-forecast-report/v2",
            sequence_kind=spec["sequence_kind"],
            source_sha256=spec["source_sha256"],
            adapter_sha256=spec["adapter_sha256"],
            continuity="explicit breaks exclude pairs, including breaks between grid times",
            exchange_completeness_verified=False,
        )
    return report, pairs
