"""A self-consistent generation error must not become verified research."""

import csv
import json
import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from test_kernel import NOW, write_order

from qrae import kernel
from qrae.kernel import KernelError
from qrae.price_baseline import load_price_rows, run_lagged_momentum, split_points
from qrae.price_reference import PriceReferenceError, reference_baseline


@pytest.mark.parametrize("fault", ["gross", "cost"])
def test_consistent_generation_fault_cannot_publish(tmp_path, monkeypatch, fault):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = kernel.run_lagged_momentum

    def wrong_generator(*args):
        points = original(*args)
        if fault == "gross":
            return [
                replace(
                    p,
                    gross_return=p.gross_return + Decimal("0.01"),
                    net_return=p.net_return + Decimal("0.01"),
                )
                for p in points
            ]
        return [
            replace(
                p,
                cost_return=p.cost_return * 2,
                net_return=p.gross_return - p.cost_return * 2,
                flat_entry_cost_return=p.flat_entry_cost_return * 2,
                flat_exit_cost_return=p.flat_exit_cost_return * 2,
            )
            for p in points
        ]

    monkeypatch.setattr(kernel, "run_lagged_momentum", wrong_generator)
    output = tmp_path / "research"
    with pytest.raises(KernelError, match="price-reference"):
        kernel.run_price_baseline(
            write_order(workspace, "consistent-error"),
            workspace=workspace,
            output_root=output,
            now=NOW,
        )
    assert not (output / "latest.json").exists()
    assert not list(output.glob("runs/*/manifest.json"))


def reference_fixture(
    tmp_path, panels, lookback=1, cost=100.0, train=0.2, validation=0.2
):
    path = tmp_path / "reference.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_time", "available_time", "instrument", "price"])
        for asset, prices in panels.items():
            for index, price in enumerate(prices):
                stamp = (
                    datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
                ).isoformat()
                writer.writerow([stamp, stamp, asset, price])
    order = {
        "kind": "PRICE_BASELINE",
        "cutoff": "2025-01-01T00:00:00Z",
        "strategy": {"kind": "lagged_momentum", "lookback": lookback, "cost_bps": cost},
        "experiment": {
            "train_fraction": train,
            "validation_fraction": validation,
            "min_test_observations": 1,
        },
    }
    return path, order


def test_hand_calculated_equal_weight_long_short_flat_and_single_point_splits(tmp_path):
    path, order = reference_fixture(
        tmp_path,
        {
            "A": [100, 110, 121, 121, 100, 90, 99, "108.9"],
            "B": [100, 90, 81, 81, 100, 110, 99, "89.1"],
        },
    )
    ref = reference_baseline(path, order)
    expected_loss = float((-Fraction(21, 121) - Fraction(19, 81)) / 2)
    assert [p["gross_return"] for p in ref["series"]] == pytest.approx(
        [0, expected_loss, 0, -0.1, -0.1]
    )
    assert [p["turnover"] for p in ref["series"]] == [1, 0, 1, 1, 0]
    assert [p["cost_return"] for p in ref["series"]] == [0.01, 0, 0.01, 0.01, 0]
    assert ref["splits"]["train"][0]["turnover"] == 2
    assert ref["splits"]["train"][0]["net_return"] == -0.02
    assert ref["splits"]["validation"][0]["net_return"] == pytest.approx(
        expected_loss - 0.02
    )
    assert [p["net_return"] for p in ref["splits"]["test"]] == [0, -0.11, -0.11]
    assert all(p["active_instruments"] == 2 for p in ref["series"])


@pytest.mark.parametrize("lookback", [1, 3, 7])
@pytest.mark.parametrize("cost", [0.0, 5.0, 100.0])
def test_seeded_panels_match_rational_reference_including_reversals(
    tmp_path, lookback, cost
):
    rng = random.Random(914)
    panels = {
        asset: [rng.choice([80, 90, 100, 100, 110, 120]) for _ in range(40)]
        for asset in ["A", "B"]
    }
    path, order = reference_fixture(tmp_path, panels, lookback, cost, 0.6, 0.2)
    ref = reference_baseline(path, order)
    rows, _ = load_price_rows(path, datetime(2025, 1, 1, tzinfo=timezone.utc))
    actual = run_lagged_momentum(rows, lookback, cost)
    assert len(actual) == len(ref["series"])
    for point, expected in zip(actual, ref["series"], strict=True):
        assert kernel._equivalent_metrics(point.as_dict(), expected)
    for name, points in split_points(actual, 0.6, 0.2).items():
        assert len(points) == len(ref["splits"][name])
        for point, expected in zip(points, ref["splits"][name], strict=True):
            assert kernel._equivalent_metrics(point.as_dict(), expected)


@pytest.mark.parametrize(
    "fault", ["late", "after_cutoff", "duplicate", "unaligned", "zero_price"]
)
def test_reference_refuses_unavailable_or_invalid_frozen_inputs(tmp_path, fault):
    panels = {
        "A": [100, 101, 102, 103, 100, 99, 101, 105],
        "B": [90, 89, 87, 90, 89, 87, 90, 91],
    }
    if fault == "unaligned":
        panels["B"].pop()
    if fault == "zero_price":
        panels["A"][3] = 0
    path, order = reference_fixture(tmp_path, panels)
    lines = path.read_text().splitlines()
    if fault == "late":
        fields = lines[1].split(",")
        fields[1] = "2024-01-02T00:00:00+00:00"
        lines[1] = ",".join(fields)
    elif fault == "after_cutoff":
        order["cutoff"] = "2023-01-01T00:00:00Z"
    elif fault == "duplicate":
        lines.insert(2, lines[1])
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(PriceReferenceError):
        reference_baseline(path, order)


def test_fresh_verification_rechecks_prices_and_preserves_legacy_receipt_shape(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Emulate a v1.0 producer at its receipt boundary, preserving actual hash/state bindings.
    validate = kernel._artifact_replay_validation
    monkeypatch.setattr(
        kernel,
        "_artifact_replay_validation",
        lambda *args, **kwargs: validate(*args, schema_version="1.0"),
    )
    result = kernel.run_price_baseline(
        write_order(workspace, "legacy"),
        workspace=workspace,
        output_root=tmp_path / "runs",
        now=NOW,
    )
    run_dir = Path(result["run_dir"])
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    monkeypatch.setattr(kernel, "_artifact_replay_validation", validate)
    assert kernel.verify_run(run_dir)["status"] == "HUMAN_REVIEW"
    assert (
        json.loads((run_dir / "validation.json").read_text())["schema_version"] == "1.0"
    )
    assert all(p.read_bytes() == value for p, value in before.items())
    reference = kernel.reference_baseline

    def disagree(*args):
        value = reference(*args)
        value["series"][0]["gross_return"] += 0.01
        return value

    monkeypatch.setattr(kernel, "reference_baseline", disagree)
    with pytest.raises(KernelError, match="price-reference"):
        kernel.verify_run(run_dir)


def test_new_receipt_names_the_semantic_gate_without_promoting_evidence(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = kernel.run_price_baseline(
        write_order(workspace, "reference-receipt"),
        workspace=workspace,
        output_root=tmp_path / "runs",
        now=NOW,
    )

    validation = json.loads((Path(result["run_dir"]) / "validation.json").read_text())
    assert validation["schema_version"] == "1.1"
    assert validation["checks"]["price_to_position_replay"]["status"] == "PASS"
    assert validation["validator"]["external_independent_reproduction"] is False
    assert result["evidence_tier"] == "E0"
    assert result["live_trading_authorized"] is False
