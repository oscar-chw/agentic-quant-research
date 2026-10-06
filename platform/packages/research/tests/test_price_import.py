from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae.artifacts import sha256_bytes
from qrae.cli import main as cli_main
from qrae.contracts import load_work_order
from qrae.data_catalog import DataCatalog
from qrae.kernel import run_price_baseline, verify_run
from qrae.price_import import PriceBarImportError, import_price_bars

UTC = timezone.utc
OBSERVED = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)


def _contract() -> dict:
    return {
        "schema_version": "1.0",
        "adapter_id": "fixture-https-price-bars-v1",
        "adapter_version": "1.0.0",
        "market_family": "crypto",
        "source_kind": "HTTPS_BYTES",
        "base_url": "https://data.example.test",
        "allowed_hosts": ["data.example.test"],
        "read_only": True,
        "auth_mode": "none",
        "data_tier": "TIER_0",
        "normalized_schema": "qrae.price-bars.v1",
        "license_status": "TEST_FIXTURE_ONLY",
        "evidence_ceiling": "E0",
    }


def test_imported_price_bars_bind_and_run_offline(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "prices.csv"
    csv_path.write_bytes(
        (repository / "examples" / "price_baseline" / "prices.csv").read_bytes()
    )

    catalog = DataCatalog(workspace / "catalog")
    snapshot = import_price_bars(
        catalog,
        adapter_contract=_contract(),
        request_id="fixture-import-001",
        source_uri="https://data.example.test/prices.csv",
        csv_path=csv_path,
        clock=lambda: OBSERVED,
    )
    assert snapshot["availability_basis"] == "OBSERVED_AT_IMPORT"
    assert snapshot["normalized_sha256"] == sha256_bytes(csv_path.read_bytes())

    payload = json.loads(
        (repository / "examples" / "price_baseline" / "work_order.json").read_text(
            encoding="utf-8"
        )
    )
    payload.update(
        {
            "schema_version": "1.1",
            "work_order_id": "imported-price-bars",
            "market_family": "crypto",
            "created_at": (OBSERVED - timedelta(minutes=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "expires_at": (OBSERVED + timedelta(days=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "cutoff": OBSERVED.isoformat().replace("+00:00", "Z"),
        }
    )
    payload["dataset"] = {
        "path": "data/prices.csv",
        "sha256": snapshot["normalized_sha256"],
        "format": "csv",
        "catalog_snapshot": {
            "catalog_root": "catalog",
            "snapshot_id": snapshot["snapshot_id"],
        },
    }
    order_path = workspace / "work_order.json"
    order_path.write_text(json.dumps(payload), encoding="utf-8")

    load_work_order(order_path, workspace, now=OBSERVED)
    result = run_price_baseline(
        order_path,
        workspace=workspace,
        output_root=tmp_path / "quantos",
        now=OBSERVED,
    )
    assert result["status"] == "HUMAN_REVIEW"
    assert verify_run(result["run_dir"])["status"] == "HUMAN_REVIEW"


def test_import_rejects_noncanonical_or_wrong_contract(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "event_time,available_time,instrument,price,secret\n"
        "2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,X,1,nope\n",
        encoding="utf-8",
    )
    with pytest.raises(PriceBarImportError, match="exactly match"):
        import_price_bars(
            DataCatalog(tmp_path / "catalog"),
            adapter_contract=_contract(),
            request_id="bad-columns",
            source_uri="https://data.example.test/prices.csv",
            csv_path=csv_path,
            clock=lambda: OBSERVED,
        )

    csv_path.write_text(
        "event_time,available_time,instrument,price\n"
        "2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,X,1,SECRET\n",
        encoding="utf-8",
    )
    with pytest.raises(PriceBarImportError, match="exactly match"):
        import_price_bars(
            DataCatalog(tmp_path / "catalog-wide"),
            adapter_contract=_contract(),
            request_id="bad-width",
            source_uri="https://data.example.test/prices.csv",
            csv_path=csv_path,
            clock=lambda: OBSERVED,
        )

    contract = _contract()
    contract["source_kind"] = "HTTPS_JSON"
    csv_path.write_text(
        "event_time,available_time,instrument,price\n"
        "2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,X,1\n",
        encoding="utf-8",
    )
    with pytest.raises(PriceBarImportError, match="HTTPS_BYTES"):
        import_price_bars(
            DataCatalog(tmp_path / "catalog-2"),
            adapter_contract=contract,
            request_id="bad-contract",
            source_uri="https://data.example.test/prices.csv",
            csv_path=csv_path,
            clock=lambda: OBSERVED,
        )


def test_price_bar_import_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "event_time,available_time,instrument,price\n"
        "2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,X,1\n",
        encoding="utf-8",
    )
    contract_path = tmp_path / "adapter.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    code = cli_main(
        [
            "price-bars-import",
            "--catalog-root",
            str(tmp_path / "catalog"),
            "--adapter-contract",
            str(contract_path),
            "--request-id",
            "cli-import-001",
            "--source-uri",
            "https://data.example.test/prices.csv",
            "--csv",
            str(csv_path),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["ok"] is True
    assert result["availability_basis"] == "OBSERVED_AT_IMPORT"
