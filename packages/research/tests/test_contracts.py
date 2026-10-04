from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qrae.contracts import (
    ContractError,
    canonical_json,
    canonical_sha256,
    load_work_order,
    sha256_file,
)

NOW = datetime(2026, 7, 11, tzinfo=timezone.utc)


def _valid_payload(workspace: Path) -> dict:
    dataset = workspace / "data" / "prices.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("timestamp,close\n2026-07-10T00:00:00Z,100\n", encoding="utf-8")
    return {
        "schema_version": "1.0",
        "work_order_id": "price-baseline-001",
        "kind": "PRICE_BASELINE",
        "market_family": "crypto",
        "created_at": "2026-07-10T00:00:00Z",
        "expires_at": "2026-07-12T00:00:00Z",
        "cutoff": "2026-07-10T00:00:00Z",
        "dataset": {
            "path": "data/prices.csv",
            "sha256": sha256_file(dataset),
            "format": "csv",
        },
        "hypothesis": {
            "mechanism": "Delayed price adjustment can create short-lived persistence.",
            "prediction": "Lagged positive returns predict the next-period return sign.",
            "horizon": "one bar",
            "falsifier": "After-cost out-of-sample performance is non-positive.",
        },
        "experiment": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "min_test_observations": 1,
        },
        "strategy": {"kind": "lagged_momentum", "lookback": 3, "cost_bps": 2.5},
        "evidence_ceiling": "E0",
        "allow_live_trading": False,
    }


def _write_order(workspace: Path, payload: dict) -> Path:
    path = workspace / "work-order.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_work_order_accepts_valid_contract_and_returns_immutable_value(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)

    order = load_work_order(_write_order(workspace, payload), workspace, now=NOW)

    assert order.work_order_id == "price-baseline-001"
    assert order.dataset.sha256 == payload["dataset"]["sha256"]
    assert order.hypothesis.horizon == "one bar"
    assert order.experiment.train_fraction == 0.6
    assert order.strategy.lookback == 3
    assert order.allow_live_trading is False
    with pytest.raises(FrozenInstanceError):
        order.work_order_id = "changed"


def test_canonical_helpers_are_order_independent():
    left = {"b": [2, 1], "a": "value"}
    right = {"a": "value", "b": [2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_rejects_expired_work_order(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["expires_at"] = "2026-07-11T00:00:00Z"

    with pytest.raises(ContractError, match="ERR_EXPIRED"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


def test_rejects_dataset_hash_mismatch(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["dataset"]["sha256"] = "0" * 64

    with pytest.raises(ContractError, match="ERR_HASH_MISMATCH"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../prices.csv",
        "C:/private/prices.csv",
        "/etc/passwd",
        "data/./prices.csv",
        "data//prices.csv",
        "data\\prices.csv",
        "data/\nprices.csv",
        "data/\x7fprices.csv",
    ],
)
def test_rejects_traversal_and_absolute_dataset_paths(tmp_path: Path, unsafe_path: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["dataset"]["path"] = unsafe_path

    with pytest.raises(ContractError, match="ERR_UNSAFE_PATH"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


def test_rejects_secret_like_keys_recursively(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["hypothesis"]["api_key"] = "dummy-test-value"

    with pytest.raises(ContractError, match="ERR_SECRET_KEY"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


def test_rejects_live_trading(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["allow_live_trading"] = True

    with pytest.raises(ContractError, match="ERR_LIVE_TRADING"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


def test_v0_rejects_e1_without_authenticated_provenance_adapter(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["evidence_ceiling"] = "E1"

    with pytest.raises(ContractError, match="ERR_EVIDENCE_CEILING"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


def test_rejects_cutoff_after_expiry(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["cutoff"] = "2026-07-13T00:00:00Z"

    with pytest.raises(ContractError, match="ERR_FUTURE_CUTOFF"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


def test_rejects_future_issuance_and_future_cutoff(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["created_at"] = "2026-07-11T01:00:00Z"

    with pytest.raises(ContractError, match="ERR_FUTURE_ISSUANCE"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)

    payload = _valid_payload(workspace)
    payload["cutoff"] = "2026-07-11T01:00:00Z"
    with pytest.raises(ContractError, match="ERR_FUTURE_CUTOFF"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)


@pytest.mark.parametrize(
    ("train", "validation", "minimum"),
    [(0.0, 0.2, 1), (0.6, 1.0, 1), (0.8, 0.2, 1), (0.6, 0.2, 0)],
)
def test_rejects_invalid_experiment_splits(
    tmp_path: Path,
    train: float,
    validation: float,
    minimum: int,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_payload(workspace)
    payload["experiment"] = {
        "train_fraction": train,
        "validation_fraction": validation,
        "min_test_observations": minimum,
    }

    with pytest.raises(ContractError, match="ERR_INVALID_SPLIT"):
        load_work_order(_write_order(workspace, payload), workspace, now=NOW)
