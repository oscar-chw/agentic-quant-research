from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from qrae.price_baseline import (
    BacktestPoint,
    DataQualityError,
    PriceRow,
    load_price_rows,
    metrics,
    run_lagged_momentum,
    split_points,
    validate_accounting,
)

UTC = timezone.utc


def row(day: int, price: str, *, available_day: int | None = None) -> PriceRow:
    event = datetime(2024, 1, day, tzinfo=UTC)
    available = datetime(2024, 1, available_day or day, tzinfo=UTC)
    return PriceRow(event, available, "TEST", Decimal(price))


def test_lagged_signal_uses_next_return_not_same_bar():
    rows = [
        row(1, "100"),
        row(2, "110"),
        row(3, "90"),
        row(4, "99"),
        row(5, "100"),
        row(6, "98"),
    ]
    points = run_lagged_momentum(rows, lookback=1, cost_bps=0)

    # Day 2 forms the signal, day 3 is the first eligible fill, and day 4 exits.
    assert points[0].exit_time == datetime(2024, 1, 4, tzinfo=UTC)
    assert points[0].gross_return == Decimal(1) * (
        Decimal(99) / Decimal(90) - Decimal(1)
    )
    assert points[0].gross_return > 0
    assert validate_accounting(points)["status"] == "PASS"


def test_cost_is_charged_on_position_change():
    rows = [
        row(1, "100"),
        row(2, "110"),
        row(3, "120"),
        row(4, "108"),
        row(5, "100"),
        row(6, "105"),
    ]
    points = run_lagged_momentum(rows, lookback=1, cost_bps=10)

    assert points[0].turnover == Decimal(1)
    assert points[0].cost_return == Decimal("0.001")
    assert points[0].net_return == points[0].gross_return - points[0].cost_return


def test_chronological_splits_are_disjoint_and_metrics_are_labeled():
    rows = [row(day, str(100 + day)) for day in range(1, 15)]
    points = run_lagged_momentum(rows, lookback=1, cost_bps=5)
    splits = split_points(points, train_fraction=0.6, validation_fraction=0.2)

    assert max(point.exit_time for point in splits["train"]) < min(
        point.exit_time for point in splits["validation"]
    )
    assert max(point.exit_time for point in splits["validation"]) < min(
        point.exit_time for point in splits["test"]
    )
    result = metrics(splits["test"])
    assert result["observations"] == len(splits["test"])
    assert result["units"]["ratio"] == "per-observation, not annualized"


def test_each_split_starts_flat_and_charges_its_first_position():
    rows = [row(day, str(100 + day)) for day in range(1, 15)]
    points = run_lagged_momentum(rows, lookback=1, cost_bps=5)

    assert points[1].cost_return == Decimal(0)
    splits = split_points(points, train_fraction=0.6, validation_fraction=0.2)

    for segment in splits.values():
        first = segment[0]
        assert first.turnover == Decimal(1)
        assert first.cost_return == Decimal("0.0005")
        assert first.net_return == first.gross_return - first.cost_return
        last = segment[-1]
        assert last.turnover == Decimal(1)
        assert last.cost_return == Decimal("0.0005")
        assert last.net_return == last.gross_return - last.cost_return
        assert validate_accounting(segment)["status"] == "PASS"


def test_generic_multi_instrument_baseline_rejects_staggered_event_grids():
    first = [row(day, str(100 + day)) for day in range(1, 8)]
    second = [
        PriceRow(
            item.event_time + timedelta(hours=12),
            item.available_time + timedelta(hours=12),
            "OTHER",
            item.price,
        )
        for item in first
    ]

    with pytest.raises(DataQualityError) as caught:
        run_lagged_momentum(first + second, lookback=1, cost_bps=5)
    assert caught.value.code == "DQ_UNALIGNED_PANEL"


def test_metrics_stop_equity_at_bankruptcy_without_negative_equity_revival():
    points = [
        BacktestPoint(
            exit_time=datetime(2024, 1, 2, tzinfo=UTC),
            gross_return=Decimal("-1.5"),
            cost_return=Decimal(0),
            net_return=Decimal("-1.5"),
            turnover=Decimal(1),
            active_instruments=1,
        ),
        BacktestPoint(
            exit_time=datetime(2024, 1, 3, tzinfo=UTC),
            gross_return=Decimal("-1.5"),
            cost_return=Decimal(0),
            net_return=Decimal("-1.5"),
            turnover=Decimal(0),
            active_instruments=1,
        ),
    ]

    result = metrics(points)

    assert result["gross_compounded_return"] == -1.0
    assert result["net_compounded_return"] == -1.0
    assert result["maximum_drawdown"] == 1.0
    assert result["bankrupt_path"] is True


def test_loader_rejects_data_not_available_at_decision_time(tmp_path: Path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "event_time,available_time,instrument,price\n"
        "2024-01-01T00:00:00Z,2024-01-02T00:00:00Z,TEST,100\n",
        encoding="utf-8",
    )

    with pytest.raises(DataQualityError) as caught:
        load_price_rows(path, datetime(2024, 1, 3, tzinfo=UTC))
    assert caught.value.code == "PIT_NOT_AVAILABLE"


def test_loader_rejects_observation_after_cutoff(tmp_path: Path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "event_time,available_time,instrument,price\n"
        "2024-01-04T00:00:00Z,2024-01-04T00:00:00Z,TEST,100\n",
        encoding="utf-8",
    )

    with pytest.raises(DataQualityError) as caught:
        load_price_rows(path, datetime(2024, 1, 3, tzinfo=UTC))
    assert caught.value.code == "PIT_AFTER_CUTOFF"
