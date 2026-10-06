"""Small independent reference for the registered PRICE_BASELINE contract.

Reads frozen CSV bytes itself and uses exact rational prices. It deliberately
does not import the production loader, signal generator, split code or metrics.
This is an internal semantic oracle, not external research reproduction.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


class PriceReferenceError(ValueError):
    """Frozen inputs cannot support the registered reference calculation."""


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise PriceReferenceError("timestamp must be text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PriceReferenceError("timestamp needs a timezone")
    return parsed.astimezone(timezone.utc)


def reference_baseline(path: Path, order: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct raw and independently flat-bounded split accounting.

    At exit index e, compare prices e-2 and e-2-lookback to choose the
    position. Fill at e-1 and mark at e. Equal weights apply to each instrument's
    gross return and absolute position changes, including short reversals.
    O(rows) arithmetic operations and memory, excluding rational bit complexity.
    """
    try:
        return _reference_baseline(path, order)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        raise PriceReferenceError(f"invalid frozen reference inputs: {exc}") from exc


def _reference_baseline(path: Path, order: dict[str, Any]) -> dict[str, Any]:
    strategy, experiment = order["strategy"], order["experiment"]
    lookback, cost_bps = strategy["lookback"], strategy["cost_bps"]
    if order["kind"] != "PRICE_BASELINE" or strategy["kind"] != "lagged_momentum":
        raise PriceReferenceError("unsupported strategy")
    if type(lookback) is not int or lookback < 1:
        raise PriceReferenceError("invalid lookback")
    if (
        type(cost_bps) not in (int, float)
        or not math.isfinite(cost_bps)
        or cost_bps < 0
    ):
        raise PriceReferenceError("invalid cost")
    rate = Fraction(str(cost_bps)) / 10000
    cutoff = _time(order["cutoff"])
    panels: dict[str, dict[datetime, Fraction]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"event_time", "available_time", "instrument", "price"} <= set(
            reader.fieldnames or []
        ):
            raise PriceReferenceError("missing price columns")
        for row in reader:
            instrument = row["instrument"].strip()
            event, available = _time(row["event_time"]), _time(row["available_time"])
            price = Fraction(row["price"])
            if not instrument or price <= 0 or available > event or event > cutoff:
                raise PriceReferenceError("invalid price, identity or availability")
            panel = panels.setdefault(instrument, {})
            if panel and event <= next(reversed(panel)):
                raise PriceReferenceError("duplicate or non-monotonic instrument time")
            panel[event] = price
    if not panels:
        raise PriceReferenceError("empty panel")
    times = list(next(iter(panels.values())))
    if any(list(panel) != times for panel in panels.values()):
        raise PriceReferenceError("unaligned panel")
    assets = sorted(panels)
    weight = Fraction(1, len(assets))
    ledger = []
    for exit_index in range(lookback + 2, len(times)):
        signal_time, old_time = times[exit_index - 2], times[exit_index - 2 - lookback]
        fill_time, exit_time = times[exit_index - 1], times[exit_index]
        positions = [
            int(panels[a][signal_time] > panels[a][old_time])
            - int(panels[a][signal_time] < panels[a][old_time])
            for a in assets
        ]
        gross = (
            sum(
                (
                    position
                    * (panels[a][exit_time] - panels[a][fill_time])
                    / panels[a][fill_time]
                    for a, position in zip(assets, positions, strict=True)
                ),
                Fraction(),
            )
            * weight
        )
        ledger.append((exit_time, positions, gross))
    count = len(ledger)
    if count < 3:
        raise PriceReferenceError("insufficient observations")
    train, validation = experiment["train_fraction"], experiment["validation_fraction"]
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1
            for v in (train, validation)
        )
        or train + validation >= 1
    ):
        raise PriceReferenceError("invalid split fractions")
    cut1 = max(1, int(count * train))
    cut2 = min(count - 1, max(cut1 + 1, int(count * (train + validation))))
    minimum = experiment["min_test_observations"]
    if (
        type(minimum) is not int
        or minimum < 1
        or count - cut2 < minimum
        or not 0 < cut1 < cut2 < count
    ):
        raise PriceReferenceError("insufficient split observations")

    def account(start: int, end: int, liquidate: bool) -> list[dict[str, Any]]:
        previous = [0] * len(assets)
        output = []
        for index in range(start, end):
            exit_time, positions, gross = ledger[index]
            flat = sum(abs(position) for position in positions) * weight
            turnover = (
                sum(
                    abs(position - old)
                    for position, old in zip(positions, previous, strict=True)
                )
                * weight
            )
            if liquidate and index == end - 1:
                turnover += flat
            cost = rate * turnover
            output.append(
                {
                    "exit_time": exit_time.isoformat().replace("+00:00", "Z"),
                    "gross_return": float(gross),
                    "cost_return": float(cost),
                    "net_return": float(gross - cost),
                    "turnover": float(turnover),
                    "active_instruments": len(assets),
                    "flat_entry_cost_return": float(rate * flat),
                    "flat_entry_turnover": float(flat),
                    "flat_exit_cost_return": float(rate * flat),
                    "flat_exit_turnover": float(flat),
                }
            )
            previous = positions
        return output

    return {
        "series": account(0, count, False),
        "splits": {
            name: account(start, end, True)
            for name, start, end in [
                ("train", 0, cut1),
                ("validation", cut1, cut2),
                ("test", cut2, count),
            ]
        },
        "instruments": len(assets),
    }
