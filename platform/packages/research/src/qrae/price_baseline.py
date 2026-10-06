"""Point-in-time price ingestion and a lagged-momentum baseline."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any

getcontext().prec = 28
UTC = timezone.utc


class DataQualityError(ValueError):
    """Raised when input data cannot support an honest point-in-time run."""

    def __init__(
        self, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataQualityError(
            "DQ_TIMESTAMP_PARSE", f"invalid ISO timestamp: {value}"
        ) from exc
    if parsed.tzinfo is None:
        raise DataQualityError("DQ_TIMESTAMP_TZ", "timestamps must include a timezone")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class PriceRow:
    event_time: datetime
    available_time: datetime
    instrument: str
    price: Decimal


@dataclass(frozen=True)
class BacktestPoint:
    exit_time: datetime
    gross_return: Decimal
    cost_return: Decimal
    net_return: Decimal
    turnover: Decimal
    active_instruments: int
    flat_entry_cost_return: Decimal = Decimal(0)
    flat_entry_turnover: Decimal = Decimal(0)
    flat_exit_cost_return: Decimal = Decimal(0)
    flat_exit_turnover: Decimal = Decimal(0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "exit_time": self.exit_time.isoformat().replace("+00:00", "Z"),
            "gross_return": float(self.gross_return),
            "cost_return": float(self.cost_return),
            "net_return": float(self.net_return),
            "turnover": float(self.turnover),
            "active_instruments": self.active_instruments,
            "flat_entry_cost_return": float(self.flat_entry_cost_return),
            "flat_entry_turnover": float(self.flat_entry_turnover),
            "flat_exit_cost_return": float(self.flat_exit_cost_return),
            "flat_exit_turnover": float(self.flat_exit_turnover),
        }


def load_price_rows(
    path: Path, cutoff: datetime
) -> tuple[list[PriceRow], dict[str, Any]]:
    required = {"event_time", "available_time", "instrument", "price"}
    rows: list[PriceRow] = []
    seen: set[tuple[str, datetime]] = set()
    last_seen: dict[str, datetime] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise DataQualityError(
                "DQ_SCHEMA",
                "price CSV is missing required columns",
                {"missing": missing},
            )
        for line_number, raw in enumerate(reader, start=2):
            instrument = (raw.get("instrument") or "").strip()
            if not instrument:
                raise DataQualityError(
                    "DQ_INSTRUMENT",
                    "instrument must be nonempty",
                    {"line": line_number},
                )
            event_time = parse_utc(raw.get("event_time") or "")
            available_time = parse_utc(raw.get("available_time") or "")
            if available_time > event_time:
                raise DataQualityError(
                    "PIT_NOT_AVAILABLE",
                    "a row was not available at its decision timestamp",
                    {"line": line_number, "instrument": instrument},
                )
            if event_time > cutoff:
                raise DataQualityError(
                    "PIT_AFTER_CUTOFF",
                    "dataset contains observations after the registered cutoff",
                    {"line": line_number, "instrument": instrument},
                )
            try:
                price = Decimal(raw.get("price") or "")
            except InvalidOperation as exc:
                raise DataQualityError(
                    "DQ_PRICE_PARSE", "price must be numeric", {"line": line_number}
                ) from exc
            if not price.is_finite() or price <= 0:
                raise DataQualityError(
                    "DQ_PRICE_RANGE",
                    "price must be finite and positive",
                    {"line": line_number},
                )
            key = (instrument, event_time)
            if key in seen:
                raise DataQualityError(
                    "DQ_DUPLICATE",
                    "duplicate instrument/event_time",
                    {"line": line_number},
                )
            if instrument in last_seen and event_time <= last_seen[instrument]:
                raise DataQualityError(
                    "DQ_NON_MONOTONIC",
                    "rows for each instrument must be strictly increasing",
                    {"line": line_number, "instrument": instrument},
                )
            seen.add(key)
            last_seen[instrument] = event_time
            rows.append(PriceRow(event_time, available_time, instrument, price))
    if not rows:
        raise DataQualityError("DQ_EMPTY", "price dataset is empty")
    return rows, {
        "status": "PASS",
        "row_count": len(rows),
        "instrument_count": len({row.instrument for row in rows}),
        "minimum_event_time": min(row.event_time for row in rows)
        .isoformat()
        .replace("+00:00", "Z"),
        "maximum_event_time": max(row.event_time for row in rows)
        .isoformat()
        .replace("+00:00", "Z"),
        "checks": {
            "schema": "PASS",
            "timestamps_utc": "PASS",
            "availability": "PASS",
            "cutoff": "PASS",
            "duplicates": "PASS",
            "monotonicity": "PASS",
            "positive_prices": "PASS",
        },
    }


def _instrument_points(
    rows: list[PriceRow], lookback: int, cost_rate: Decimal
) -> list[
    tuple[datetime, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]
]:
    if len(rows) < lookback + 3:
        raise DataQualityError(
            "DQ_INSUFFICIENT_HISTORY",
            "instrument has too few rows for the registered lookback",
            {
                "instrument": rows[0].instrument,
                "rows": len(rows),
                "required": lookback + 3,
            },
        )
    for index, current in enumerate(rows):
        if current.available_time > current.event_time:
            raise DataQualityError(
                "PIT_NOT_AVAILABLE",
                "a row was not available at its decision timestamp",
                {"instrument": current.instrument, "index": index},
            )
        if not current.price.is_finite() or current.price <= 0:
            raise DataQualityError(
                "DQ_PRICE_RANGE",
                "price must be finite and positive",
                {"instrument": current.instrument, "index": index},
            )
        if index and current.event_time <= rows[index - 1].event_time:
            raise DataQualityError(
                "DQ_NON_MONOTONIC",
                "rows for each instrument must be strictly increasing",
                {"instrument": current.instrument, "index": index},
            )

    output: list[
        tuple[datetime, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]
    ] = []
    previous_signal = Decimal(0)
    for index in range(lookback, len(rows) - 2):
        signal_row = rows[index]
        fill_row = rows[index + 1]
        exit_row = rows[index + 2]
        decision_time = max(signal_row.event_time, signal_row.available_time)
        if fill_row.event_time <= decision_time:
            raise DataQualityError(
                "PIT_NON_CAUSAL_FILL",
                "fill timestamp must be strictly later than the signal decision timestamp",
                {"instrument": signal_row.instrument, "signal_index": index},
            )
        formation_return = signal_row.price / rows[index - lookback].price - Decimal(1)
        signal = (
            Decimal(1)
            if formation_return > 0
            else Decimal(-1)
            if formation_return < 0
            else Decimal(0)
        )
        next_return = exit_row.price / fill_row.price - Decimal(1)
        gross = signal * next_return
        turnover = abs(signal - previous_signal)
        cost = cost_rate * turnover
        flat_entry_turnover = abs(signal)
        flat_entry_cost = cost_rate * flat_entry_turnover
        output.append(
            (
                exit_row.event_time,
                gross,
                cost,
                turnover,
                flat_entry_cost,
                flat_entry_turnover,
                flat_entry_cost,
                flat_entry_turnover,
            )
        )
        previous_signal = signal
    return output


def run_lagged_momentum(
    rows: list[PriceRow], lookback: int, cost_bps: float
) -> list[BacktestPoint]:
    if lookback < 1:
        raise ValueError("lookback must be at least one")
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    grouped: dict[str, list[PriceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.instrument].append(row)
    grids = {
        tuple(row.event_time for row in instrument_rows)
        for instrument_rows in grouped.values()
    }
    if len(grids) != 1:
        raise DataQualityError(
            "DQ_UNALIGNED_PANEL",
            "generic equal-weight baseline requires one identical event-time grid across instruments",
        )
    by_exit: dict[
        datetime,
        list[tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]],
    ] = defaultdict(list)
    cost_rate = Decimal(str(cost_bps)) / Decimal(10000)
    for instrument_rows in grouped.values():
        for (
            exit_time,
            gross,
            cost,
            turnover,
            flat_entry_cost,
            flat_entry_turnover,
            flat_exit_cost,
            flat_exit_turnover,
        ) in _instrument_points(instrument_rows, lookback, cost_rate):
            by_exit[exit_time].append(
                (
                    gross,
                    cost,
                    turnover,
                    flat_entry_cost,
                    flat_entry_turnover,
                    flat_exit_cost,
                    flat_exit_turnover,
                )
            )
    points: list[BacktestPoint] = []
    for exit_time in sorted(by_exit):
        values = by_exit[exit_time]
        count = Decimal(len(values))
        gross = sum((value[0] for value in values), Decimal(0)) / count
        cost = sum((value[1] for value in values), Decimal(0)) / count
        turnover = sum((value[2] for value in values), Decimal(0)) / count
        flat_entry_cost = sum((value[3] for value in values), Decimal(0)) / count
        flat_entry_turnover = sum((value[4] for value in values), Decimal(0)) / count
        flat_exit_cost = sum((value[5] for value in values), Decimal(0)) / count
        flat_exit_turnover = sum((value[6] for value in values), Decimal(0)) / count
        points.append(
            BacktestPoint(
                exit_time,
                gross,
                cost,
                gross - cost,
                turnover,
                len(values),
                flat_entry_cost,
                flat_entry_turnover,
                flat_exit_cost,
                flat_exit_turnover,
            )
        )
    if len(points) < 3:
        raise DataQualityError(
            "DQ_INSUFFICIENT_OBSERVATIONS",
            "baseline requires at least three portfolio observations",
        )
    return points


def split_points(
    points: list[BacktestPoint], train_fraction: float, validation_fraction: float
) -> dict[str, list[BacktestPoint]]:
    if (
        not 0 < train_fraction < 1
        or not 0 < validation_fraction < 1
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError("invalid chronological split fractions")
    count = len(points)
    train_end = max(1, int(count * train_fraction))
    validation_end = max(
        train_end + 1, int(count * (train_fraction + validation_fraction))
    )
    validation_end = min(validation_end, count - 1)
    if train_end >= validation_end or validation_end >= count:
        raise DataQualityError(
            "DQ_SPLIT", "too few observations for three chronological splits"
        )
    raw_splits = {
        "train": points[:train_end],
        "validation": points[train_end:validation_end],
        "test": points[validation_end:],
    }
    splits: dict[str, list[BacktestPoint]] = {}
    for name, segment in raw_splits.items():
        first = segment[0]
        reset_first = replace(
            first,
            cost_return=first.flat_entry_cost_return,
            net_return=first.gross_return - first.flat_entry_cost_return,
            turnover=first.flat_entry_turnover,
        )
        adjusted = [reset_first, *segment[1:]]
        last = adjusted[-1]
        adjusted[-1] = replace(
            last,
            cost_return=last.cost_return + last.flat_exit_cost_return,
            net_return=last.gross_return
            - last.cost_return
            - last.flat_exit_cost_return,
            turnover=last.turnover + last.flat_exit_turnover,
        )
        splits[name] = adjusted
    return splits


def _equity_path(returns: Iterable[float]) -> tuple[float, float, bool]:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    bankrupt = False
    for value in returns:
        if bankrupt:
            continue
        if value <= -1.0:
            equity = 0.0
            bankrupt = True
        else:
            equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return equity, max_drawdown, bankrupt


def metrics(points: list[BacktestPoint]) -> dict[str, Any]:
    net = [float(point.net_return) for point in points]
    gross = [float(point.gross_return) for point in points]
    costs = [float(point.cost_return) for point in points]
    mean = statistics.fmean(net) if net else 0.0
    sample_std = statistics.stdev(net) if len(net) > 1 else 0.0
    ratio = mean / sample_std if sample_std > 0 else None
    gross_equity, _, _ = _equity_path(gross)
    net_equity, max_drawdown, bankrupt = _equity_path(net)
    return {
        "observations": len(points),
        "gross_compounded_return": gross_equity - 1.0,
        "net_compounded_return": net_equity - 1.0,
        "mean_net_return_per_observation": mean,
        "sample_std_net_return_per_observation": sample_std,
        "mean_over_sample_std_not_annualized": ratio,
        "maximum_drawdown": max_drawdown,
        "win_rate": sum(value > 0 for value in net) / len(net) if net else None,
        "turnover_units": sum(float(point.turnover) for point in points),
        "cost_return_sum": sum(costs),
        "bankrupt_path": bankrupt,
        "units": {
            "returns": "decimal return",
            "cost_return_sum": "decimal return summed across observations",
            "turnover_units": "absolute position change, equal-weight portfolio",
            "ratio": "per-observation, not annualized",
        },
    }


def validate_accounting(
    points: list[BacktestPoint], tolerance: Decimal = Decimal("1e-18")
) -> dict[str, Any]:
    for index, point in enumerate(points):
        residual = point.gross_return - point.cost_return - point.net_return
        if abs(residual) > tolerance:
            raise ValueError(f"accounting mismatch at observation {index}")
    timestamps = [point.exit_time for point in points]
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise ValueError("portfolio exit timestamps must be unique and increasing")
    return {
        "status": "PASS",
        "observations_recomputed": len(points),
        "tolerance": str(tolerance),
    }
