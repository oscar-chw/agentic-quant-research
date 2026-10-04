from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.data_catalog import (
    CatalogError,
    DataCatalog,
    _snapshot_identity,
    verify_provenance_envelope,
)
from qrae.kernel import run_price_baseline, verify_run
from qrae.synthetic_import import import_synthetic_prices

OBSERVED = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path) -> tuple[DataCatalog, Path, dict]:
    prices = tmp_path / "prices.csv"
    prices.write_bytes((REPO / "examples/price_baseline/prices.csv").read_bytes())
    derivation = {
        "schema_version": "qrae.synthetic-derivation.v1",
        "synthetic": True,
        "market_family": "synthetic_prediction_prices",
        "generator_sha256": sha256_bytes(b"fixture generator source"),
        "parameters": {"seed": 7, "price_rule": "synthetic midpoint"},
        "collector_raw_sha256": sha256_bytes(b"synthetic raw collector fixture"),
        "collector_normalized_sha256": sha256_bytes(
            b"synthetic collector normalization"
        ),
        "normalized_price_sha256": sha256_bytes(prices.read_bytes()),
    }
    return DataCatalog(tmp_path / "catalog"), prices, derivation


def test_synthetic_import_binds_work_order_and_verifies_without_catalog(tmp_path: Path):
    catalog, prices, derivation = fixture(tmp_path)
    snapshot = import_synthetic_prices(catalog, prices, derivation, OBSERVED)
    assert import_synthetic_prices(catalog, prices, derivation, OBSERVED) == snapshot
    assert snapshot["source_uri"] == "urn:qrae:synthetic:" + sha256_bytes(
        canonical_json_bytes(derivation)
    )
    assert snapshot["availability_basis"] == "OBSERVED_AT_IMPORT"
    assert snapshot["evidence_ceiling"] == "E0"
    assert snapshot["source_http_date"] is None
    assert snapshot["response_headers"] == {}
    assert catalog.verify_catalog()["snapshots"] == 1
    provenance = catalog.resolve_provenance(snapshot["snapshot_id"], as_of=OBSERVED)
    assert provenance["schema_version"] == "1.1"
    assert provenance["adapter"]["source_kind"] == "LOCAL_SYNTHETIC"
    assert provenance["adapter"]["allowed_hosts"] == []
    assert provenance["synthetic_derivation"] == derivation

    order = json.loads((REPO / "examples/price_baseline/work_order.json").read_text())
    order.update(
        {
            "schema_version": "1.1",
            "market_family": derivation["market_family"],
            "cutoff": "2026-09-13T00:00:00Z",
        }
    )
    order["dataset"] = {
        "path": "prices.csv",
        "sha256": snapshot["normalized_sha256"],
        "format": "csv",
        "catalog_snapshot": {
            "catalog_root": "catalog",
            "snapshot_id": snapshot["snapshot_id"],
        },
    }
    order_path = tmp_path / "order.json"
    order_path.write_text(json.dumps(order))
    result = run_price_baseline(
        order_path, workspace=tmp_path, output_root=tmp_path / "runs", now=OBSERVED
    )
    assert result["status"] == "HUMAN_REVIEW"
    assert result["evidence_tier"] == "E0"
    assert result["live_trading_authorized"] is False
    run_dir = Path(result["run_dir"])
    assert (
        json.loads((run_dir / "snapshot/provenance.json").read_text())[
            "synthetic_derivation"
        ]
        == derivation
    )
    shutil.rmtree(tmp_path / "catalog")
    assert verify_provenance_envelope(provenance) == provenance
    assert verify_run(run_dir)["status"] == "HUMAN_REVIEW"


@pytest.mark.parametrize(
    "change",
    [
        {"synthetic": False},
        {"schema_version": "historical.v1"},
        {"source_uri": "https://invented.example/prices.csv"},
        {"normalized_price_sha256": "0" * 64},
        {"parameters": {"value": float("nan")}},
    ],
)
def test_synthetic_import_rejects_false_or_mismatched_declaration(
    tmp_path: Path, change: dict
):
    catalog, prices, derivation = fixture(tmp_path)
    derivation.update(change)
    with pytest.raises(CatalogError):
        import_synthetic_prices(catalog, prices, derivation, OBSERVED)
    assert catalog.stats()["snapshots"] == 0


def direct_arguments(snapshot: dict, prices: Path, derivation: dict) -> dict:
    return {
        "request_id": "direct-negative-case",
        "adapter_id": snapshot["adapter_id"],
        "source_uri": snapshot["source_uri"],
        "requested_at": OBSERVED,
        "retrieved_at": OBSERVED,
        "available_at": OBSERVED,
        "cutoff": OBSERVED,
        "availability_basis": "OBSERVED_AT_IMPORT",
        "raw_bytes": canonical_json_bytes(derivation),
        "normalized_bytes": prices.read_bytes(),
        "normalized_schema": "qrae.price-bars.v1",
        "item_count": snapshot["item_count"],
        "max_source_event_time": datetime.fromisoformat(
            snapshot["max_source_event_time"].replace("Z", "+00:00")
        ),
        "source_http_date": None,
        "response_headers": {},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_uri", "urn:qrae:synthetic:" + "0" * 64),
        ("source_uri", "https://invented.example/prices.csv"),
        ("raw_bytes", b"{}\n"),
        ("normalized_bytes", b"changed normalized data"),
        ("availability_basis", "OBSERVED_AT_RETRIEVAL"),
        ("source_http_date", OBSERVED),
        ("response_headers", {"content-type": "text/csv"}),
    ],
)
def test_catalog_registration_enforces_synthetic_binding(
    tmp_path: Path, field: str, value
):
    catalog, prices, derivation = fixture(tmp_path)
    snapshot = import_synthetic_prices(catalog, prices, derivation, OBSERVED)
    arguments = direct_arguments(snapshot, prices, derivation)
    arguments[field] = value
    with pytest.raises(CatalogError):
        catalog.register_snapshot(**arguments)
    assert catalog.verify_catalog()["snapshots"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "1.0"},
        {"source_kind": "HTTPS_BYTES"},
        {"allowed_hosts": ["invented.example"]},
        {"base_url": "https://invented.example"},
        {"auth_mode": "manifest-declared"},
        {"evidence_ceiling": "E1"},
    ],
)
def test_synthetic_adapter_cannot_masquerade_as_external_source(
    tmp_path: Path, change: dict
):
    catalog, prices, derivation = fixture(tmp_path)
    snapshot = import_synthetic_prices(catalog, prices, derivation, OBSERVED)
    adapter = catalog.resolve_provenance(snapshot["snapshot_id"], as_of=OBSERVED)[
        "adapter"
    ]
    contract = {
        k: v
        for k, v in adapter.items()
        if k not in {"registered_at", "contract_sha256"}
    }
    contract.update(change)
    with pytest.raises(CatalogError):
        catalog.register_adapter(contract, registered_at=OBSERVED)


@pytest.mark.parametrize(
    "field,value",
    [
        ("synthetic", False),
        ("normalized_price_sha256", "0" * 64),
        ("market_family", "different-market"),
    ],
)
def test_portable_provenance_rejects_coherently_rehashed_bad_derivation(
    tmp_path: Path, field: str, value
):
    catalog, prices, derivation = fixture(tmp_path)
    snapshot = import_synthetic_prices(catalog, prices, derivation, OBSERVED)
    envelope = copy.deepcopy(
        catalog.resolve_provenance(snapshot["snapshot_id"], as_of=OBSERVED)
    )
    envelope["synthetic_derivation"][field] = value
    raw = canonical_json_bytes(envelope["synthetic_derivation"])
    record = envelope["snapshot"]
    record["raw_sha256"] = sha256_bytes(raw)
    record["raw_size_bytes"] = len(raw)
    record["raw_object_path"] = (
        f"objects/raw/{record['raw_sha256'][:2]}/{record['raw_sha256']}.bin"
    )
    record["source_uri"] = "urn:qrae:synthetic:" + record["raw_sha256"]
    record["snapshot_id"] = sha256_bytes(
        canonical_json_bytes(_snapshot_identity(record))
    )
    record["request_fingerprint"] = record["snapshot_id"]
    with pytest.raises(CatalogError):
        verify_provenance_envelope(envelope)


def test_catalog_detects_changed_synthetic_raw_object(tmp_path: Path):
    catalog, prices, derivation = fixture(tmp_path)
    snapshot = import_synthetic_prices(catalog, prices, derivation, OBSERVED)
    raw_path = tmp_path / "catalog" / snapshot["raw_object_path"]
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(CatalogError):
        catalog.verify_catalog()
