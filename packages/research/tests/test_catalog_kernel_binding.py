from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from qrae.cli import main as cli_main
from qrae.contracts import ContractError, load_work_order
from qrae.data_catalog import DataCatalog
from qrae.kernel import KernelError, run_price_baseline, verify_run

UTC = timezone.utc
NOW = datetime(2026, 7, 11, tzinfo=UTC)
CUTOFF = datetime(2024, 2, 9, tzinfo=UTC)


def _price_bytes() -> bytes:
    rows = ["event_time,available_time,instrument,price"]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for index in range(40):
        event = start + timedelta(days=index)
        stamp = event.isoformat().replace("+00:00", "Z")
        rows.append(f"{stamp},{stamp},TEST,{100 + index}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def _adapter_contract() -> dict:
    return {
        "schema_version": "1.0",
        "adapter_id": "fixture-price-bars-v1",
        "adapter_version": "1.0.0",
        "market_family": "test_prices",
        "source_kind": "HTTPS_BYTES",
        "base_url": "https://example.test",
        "allowed_hosts": ["example.test"],
        "read_only": True,
        "auth_mode": "none",
        "data_tier": "TIER_0",
        "normalized_schema": "qrae.price-bars.v1",
        "license_status": "TEST_FIXTURE_ONLY",
        "evidence_ceiling": "E0",
    }


def _catalog_order(
    workspace: Path,
    *,
    order_id: str = "catalog-bound",
    available_at: datetime = CUTOFF,
    normalized_bytes: bytes | None = None,
) -> tuple[Path, Path, dict]:
    prices = _price_bytes()
    dataset = workspace / "data" / f"{order_id}.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_bytes(prices)
    catalog_root = workspace / "catalog"
    catalog = DataCatalog(catalog_root)
    catalog.register_adapter(
        _adapter_contract(), registered_at=available_at - timedelta(seconds=1)
    )
    snapshot = catalog.register_snapshot(
        request_id=f"request-{order_id}",
        adapter_id="fixture-price-bars-v1",
        source_uri=f"https://example.test/prices?dataset={order_id}",
        requested_at=available_at - timedelta(seconds=1),
        retrieved_at=available_at,
        available_at=available_at,
        cutoff=available_at,
        availability_basis="OBSERVED_AT_RETRIEVAL",
        raw_bytes=prices,
        normalized_bytes=normalized_bytes if normalized_bytes is not None else prices,
        normalized_schema="qrae.price-bars.v1",
        item_count=40,
        max_source_event_time=CUTOFF,
        source_http_date=available_at,
        response_headers={"content-type": "text/csv", "etag": f"{order_id}-fixture"},
    )
    payload = {
        "schema_version": "1.1",
        "work_order_id": order_id,
        "kind": "PRICE_BASELINE",
        "market_family": "test_prices",
        "created_at": "2026-07-10T00:00:00Z",
        "expires_at": "2026-07-12T00:00:00Z",
        "cutoff": CUTOFF.isoformat().replace("+00:00", "Z"),
        "dataset": {
            "path": f"data/{order_id}.csv",
            "sha256": sha256_file(dataset),
            "format": "csv",
            "catalog_snapshot": {
                "catalog_root": "catalog",
                "snapshot_id": snapshot["snapshot_id"],
            },
        },
        "hypothesis": {
            "mechanism": "Delayed adjustment creates persistence.",
            "prediction": "Lagged returns predict the next return sign.",
            "horizon": "one observation",
            "falsifier": "Locked after-cost test return is non-positive.",
        },
        "experiment": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "min_test_observations": 5,
        },
        "strategy": {"kind": "lagged_momentum", "lookback": 3, "cost_bps": 5.0},
        "evidence_ceiling": "E0",
        "allow_live_trading": False,
    }
    order_path = workspace / f"{order_id}.json"
    order_path.write_text(json.dumps(payload), encoding="utf-8")
    return order_path, catalog_root, snapshot


def test_catalog_bound_contract_freezes_provenance_and_replays_without_catalog(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order_path, catalog_root, snapshot = _catalog_order(workspace)

    order = load_work_order(order_path, workspace, now=NOW)
    assert order.schema_version == "1.1"
    assert order.dataset.catalog_snapshot is not None
    assert order.dataset.catalog_snapshot.snapshot_id == snapshot["snapshot_id"]

    output = tmp_path / "quantos"
    first = run_price_baseline(
        order_path, workspace=workspace, output_root=output, now=NOW
    )
    second = run_price_baseline(
        order_path, workspace=workspace, output_root=output, now=NOW
    )
    assert first["status"] == "HUMAN_REVIEW"
    assert second["idempotent_replay"] is True
    run_dir = Path(first["run_dir"])
    provenance = json.loads(
        (run_dir / "snapshot" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["snapshot"]["snapshot_id"] == snapshot["snapshot_id"]
    assert provenance["selected_object"]["sha256"] == sha256_file(
        run_dir / "snapshot" / "prices.csv"
    )
    assert provenance["snapshot"]["evidence_ceiling"] == "E0"
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["checks"]["catalog_provenance"]["status"] == "PASS"
    claims = json.loads((run_dir / "claim_ledger.json").read_text(encoding="utf-8"))
    assert any(claim["claim_id"] == "C004" for claim in claims["claims"])
    assert "Catalog provenance frozen: `true`" in (run_dir / "report.md").read_text(
        encoding="utf-8"
    )

    shutil.rmtree(catalog_root)
    assert verify_run(run_dir)["status"] == "HUMAN_REVIEW"


def test_catalog_binding_rejects_future_snapshot_and_hash_mismatch(tmp_path: Path):
    future_workspace = tmp_path / "future"
    future_workspace.mkdir()
    future_order, _, _ = _catalog_order(
        future_workspace,
        order_id="future",
        available_at=CUTOFF + timedelta(microseconds=1),
    )
    with pytest.raises(ContractError, match="ERR_PROVENANCE_PIT"):
        load_work_order(future_order, future_workspace, now=NOW)

    mismatch_workspace = tmp_path / "mismatch"
    mismatch_workspace.mkdir()
    mismatch_order, _, _ = _catalog_order(
        mismatch_workspace,
        order_id="mismatch",
        normalized_bytes=b"different normalized bytes\n",
    )
    with pytest.raises(ContractError, match="ERR_PROVENANCE_HASH"):
        load_work_order(mismatch_order, mismatch_workspace, now=NOW)

    market_workspace = tmp_path / "market"
    market_workspace.mkdir()
    market_order, _, _ = _catalog_order(market_workspace, order_id="wrong-market")
    payload = json.loads(market_order.read_text(encoding="utf-8"))
    payload["market_family"] = "different_market"
    market_order.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="ERR_PROVENANCE_MARKET"):
        load_work_order(market_order, market_workspace, now=NOW)


def test_catalog_binding_rejects_symlinked_catalog_root(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order_path, catalog_root, _ = _catalog_order(workspace, order_id="symlink")
    real = workspace / "catalog-real"
    catalog_root.rename(real)
    try:
        os.symlink(real, catalog_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ContractError, match="ERR_UNSAFE_PATH"):
        load_work_order(order_path, workspace, now=NOW)


def test_frozen_provenance_tampering_fails_even_with_rehashed_manifest(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order_path, _, _ = _catalog_order(workspace, order_id="tamper-provenance")
    output = tmp_path / "quantos"
    result = run_price_baseline(
        order_path, workspace=workspace, output_root=output, now=NOW
    )
    run_dir = Path(result["run_dir"])
    (output / "latest.json").unlink()

    provenance_path = run_dir / "snapshot" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["snapshot"]["item_count"] += 1
    altered = canonical_json_bytes(provenance)
    provenance_path.write_bytes(altered)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        if item["path"] == "snapshot/provenance.json":
            item["sha256"] = sha256_bytes(altered)
            item["bytes"] = len(altered)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(KernelError, match=r"state commitment|provenance"):
        verify_run(run_dir)


def test_catalog_bound_data_failure_quarantines_with_frozen_provenance(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order_path, _, _ = _catalog_order(workspace, order_id="quarantine")
    payload = json.loads(order_path.read_text(encoding="utf-8"))
    payload["experiment"]["min_test_observations"] = 1_000_000
    order_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_price_baseline(
        order_path,
        workspace=workspace,
        output_root=tmp_path / "quantos",
        now=NOW,
    )
    run_dir = Path(result["run_dir"])
    assert result["status"] == "QUARANTINED"
    assert (run_dir / "snapshot" / "provenance.json").is_file()
    assert verify_run(run_dir)["status"] == "QUARANTINED"


def test_cli_runs_and_verifies_catalog_bound_work_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order_path, _, _ = _catalog_order(workspace, order_id="cli-bound")
    output = tmp_path / "quantos"

    code = cli_main(
        [
            "run",
            "--work-order",
            str(order_path),
            "--workspace",
            str(workspace),
            "--output-root",
            str(output),
        ],
        now=NOW,
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["ok"] is True
    assert result["dataset_provenance_sha256"]

    code = cli_main(["verify", "--run-dir", result["run_dir"]])
    verified = json.loads(capsys.readouterr().out)
    assert code == 0 and verified["ok"] is True
    assert verified["status"] == "HUMAN_REVIEW"


def test_catalog_mutation_before_run_cannot_replace_latest(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "quantos"
    good_order, _, _ = _catalog_order(workspace, order_id="good")
    good = run_price_baseline(
        good_order, workspace=workspace, output_root=output, now=NOW
    )
    latest_before = (output / "latest.json").read_bytes()

    bad_order, catalog_root, bad_snapshot = _catalog_order(
        workspace, order_id="mutated"
    )
    object_path = catalog_root / bad_snapshot["normalized_object_path"]
    object_path.write_bytes(b"tampered\n")
    with pytest.raises(ContractError, match=r"CATALOG_HASH_MISMATCH|ERR_PROVENANCE"):
        run_price_baseline(bad_order, workspace=workspace, output_root=output, now=NOW)
    assert (output / "latest.json").read_bytes() == latest_before
    assert json.loads((output / "latest.json").read_text())["run_id"] == good["run_id"]
