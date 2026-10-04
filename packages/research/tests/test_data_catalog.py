from __future__ import annotations

import json
import os
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae.data_catalog import CatalogError, DataCatalog, verify_provenance_envelope

UTC = timezone.utc
T0 = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)


def adapter_contract() -> dict:
    return {
        "schema_version": "1.0",
        "adapter_id": "test-public-markets-v1",
        "adapter_version": "1.0.0",
        "market_family": "prediction_markets",
        "source_kind": "HTTPS_JSON",
        "base_url": "https://example.test",
        "allowed_hosts": ["example.test"],
        "read_only": True,
        "auth_mode": "none",
        "data_tier": "TIER_0",
        "normalized_schema": "test.markets.v1",
        "license_status": "UNVERIFIED_PUBLIC_API_TERMS",
        "evidence_ceiling": "E0",
    }


def register_adapter(catalog: DataCatalog) -> dict:
    return catalog.register_adapter(adapter_contract(), registered_at=T0)


def register_snapshot(
    catalog: DataCatalog,
    *,
    request_id: str = "request-001",
    retrieved_at: datetime = T0 + timedelta(minutes=1),
    raw: bytes = b'[{"id":"1","revision":1}]',
) -> dict:
    normalized = b'{"market_id":"1","revision":1}\n'
    return catalog.register_snapshot(
        request_id=request_id,
        adapter_id="test-public-markets-v1",
        source_uri="https://example.test/markets?active=true&limit=1",
        requested_at=retrieved_at - timedelta(seconds=5),
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        cutoff=retrieved_at,
        availability_basis="OBSERVED_AT_RETRIEVAL",
        raw_bytes=raw,
        normalized_bytes=normalized,
        normalized_schema="test.markets.v1",
        item_count=1,
        max_source_event_time=retrieved_at - timedelta(hours=1),
        source_http_date=retrieved_at - timedelta(seconds=1),
        response_headers={"content-type": "application/json", "etag": "test-etag"},
    )


def test_catalog_persists_registered_adapter_and_hash_bound_snapshot(tmp_path: Path):
    root = tmp_path / "catalog"
    catalog = DataCatalog(root)
    adapter = register_adapter(catalog)
    snapshot = register_snapshot(catalog)

    reopened = DataCatalog(root)
    resolved = reopened.resolve_snapshot(
        snapshot["snapshot_id"], as_of=T0 + timedelta(hours=1)
    )
    verified = reopened.verify_catalog()

    assert adapter["adapter_id"] == "test-public-markets-v1"
    assert adapter["contract_sha256"] == snapshot["adapter_contract_sha256"]
    assert resolved == snapshot
    assert resolved["evidence_ceiling"] == "E0"
    assert resolved["availability_basis"] == "OBSERVED_AT_RETRIEVAL"
    assert not Path(resolved["raw_object_path"]).is_absolute()
    assert verified == {
        "schema_version": "1.0",
        "adapters": 1,
        "snapshots": 1,
        "latest_pointers": 1,
        "orphan_objects": 0,
    }


def test_provenance_envelope_is_portable_and_self_verifying(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    snapshot = register_snapshot(catalog)
    envelope = catalog.resolve_provenance(
        snapshot["snapshot_id"], as_of=T0 + timedelta(hours=1)
    )

    assert verify_provenance_envelope(envelope) == envelope
    assert envelope["selected_object"]["sha256"] == snapshot["normalized_sha256"]
    forged = json.loads(json.dumps(envelope))
    forged["snapshot"]["item_count"] += 1
    with pytest.raises(CatalogError, match="CATALOG_HASH_MISMATCH"):
        verify_provenance_envelope(forged)


def test_point_in_time_resolution_uses_only_snapshots_available_by_cutoff(
    tmp_path: Path,
):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    first = register_snapshot(catalog, request_id="request-first")
    second = register_snapshot(
        catalog,
        request_id="request-revision",
        retrieved_at=T0 + timedelta(hours=1),
        raw=b'[{"id":"1","revision":2}]',
    )

    between = catalog.resolve_latest(
        "test-public-markets-v1", as_of=T0 + timedelta(minutes=30)
    )
    current = catalog.resolve_latest(
        "test-public-markets-v1", as_of=T0 + timedelta(hours=2)
    )

    assert between["snapshot_id"] == first["snapshot_id"]
    assert current["snapshot_id"] == second["snapshot_id"]
    with pytest.raises(CatalogError, match="PIT_NOT_AVAILABLE"):
        catalog.resolve_latest("test-public-markets-v1", as_of=T0)
    with pytest.raises(CatalogError, match="PIT_NOT_AVAILABLE"):
        catalog.resolve_snapshot(
            second["snapshot_id"], as_of=T0 + timedelta(minutes=30)
        )


def test_mutated_availability_index_cannot_bypass_point_in_time_gate(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    future = register_snapshot(
        catalog,
        request_id="request-future",
        retrieved_at=T0 + timedelta(hours=1),
    )
    with sqlite3.connect(catalog.root / "catalog.sqlite3") as connection:
        connection.execute(
            "UPDATE snapshots SET available_at_us = 0 WHERE snapshot_id = ?",
            (future["snapshot_id"],),
        )

    with pytest.raises(CatalogError, match="CATALOG_HASH_MISMATCH"):
        catalog.resolve_snapshot(
            future["snapshot_id"], as_of=T0 + timedelta(minutes=30)
        )
    with pytest.raises(CatalogError, match="CATALOG_HASH_MISMATCH"):
        catalog.resolve_latest(
            "test-public-markets-v1", as_of=T0 + timedelta(minutes=30)
        )


def test_tampered_object_is_rehashed_and_rejected_on_resolution(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    snapshot = register_snapshot(catalog)
    raw_path = catalog.root / snapshot["raw_object_path"]
    raw_path.write_bytes(b"tampered")

    with pytest.raises(CatalogError, match="CATALOG_HASH_MISMATCH"):
        catalog.resolve_snapshot(snapshot["snapshot_id"], as_of=T0 + timedelta(hours=1))
    with pytest.raises(CatalogError, match="CATALOG_HASH_MISMATCH"):
        catalog.verify_catalog()


def test_concurrent_idempotent_registration_creates_one_snapshot(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: register_snapshot(catalog), range(16)))

    assert len({result["snapshot_id"] for result in results}) == 1
    assert catalog.stats()["snapshots"] == 1
    assert catalog.verify_catalog()["latest_pointers"] == 1


def test_conflicting_idempotency_key_is_rejected_without_mutation(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    first = register_snapshot(catalog)

    with pytest.raises(CatalogError, match="CATALOG_IDEMPOTENCY_CONFLICT"):
        register_snapshot(catalog, raw=b'[{"id":"different"}]')

    assert catalog.stats()["snapshots"] == 1
    assert (
        catalog.resolve_latest("test-public-markets-v1", as_of=T0 + timedelta(hours=1))[
            "snapshot_id"
        ]
        == first["snapshot_id"]
    )


def test_registering_older_snapshot_never_regresses_latest_pointer(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    newer = register_snapshot(
        catalog,
        request_id="request-newer",
        retrieved_at=T0 + timedelta(hours=2),
        raw=b'[{"id":"1","revision":2}]',
    )
    older = register_snapshot(
        catalog,
        request_id="request-older",
        retrieved_at=T0 + timedelta(hours=1),
        raw=b'[{"id":"1","revision":1}]',
    )

    assert catalog.latest_snapshot_id("test-public-markets-v1") == newer["snapshot_id"]
    assert (
        catalog.resolve_latest(
            "test-public-markets-v1", as_of=T0 + timedelta(hours=1, minutes=30)
        )["snapshot_id"]
        == older["snapshot_id"]
    )


def test_transaction_failure_preserves_catalog_and_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)

    def fail_insert(*args, **kwargs):
        raise RuntimeError("synthetic transaction failure")

    monkeypatch.setattr(DataCatalog, "_insert_snapshot", fail_insert)
    with pytest.raises(RuntimeError, match="synthetic transaction failure"):
        register_snapshot(catalog)

    assert catalog.stats()["snapshots"] == 0
    assert catalog.latest_snapshot_id("test-public-markets-v1") is None


def test_symlinked_object_store_is_rejected(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(catalog.root / "objects")
    try:
        os.symlink(outside, catalog.root / "objects", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(CatalogError, match="CATALOG_UNSAFE_PATH"):
        register_snapshot(catalog)
    assert not list(outside.iterdir())


def test_unknown_adapter_secret_uri_and_false_e1_fail_closed(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    with pytest.raises(CatalogError, match="CATALOG_UNKNOWN_ADAPTER"):
        register_snapshot(catalog)

    register_adapter(catalog)
    with pytest.raises(CatalogError, match="CATALOG_SECRET_URI"):
        catalog.register_snapshot(
            request_id="secret-request",
            adapter_id="test-public-markets-v1",
            source_uri="https://example.test/markets?api_key=not-a-real-value",
            requested_at=T0,
            retrieved_at=T0,
            available_at=T0,
            cutoff=T0,
            availability_basis="OBSERVED_AT_RETRIEVAL",
            raw_bytes=b"[]",
            normalized_bytes=b"{}\n",
            normalized_schema="test.markets.v1",
            item_count=1,
            max_source_event_time=None,
            source_http_date=None,
            response_headers={"content-type": "application/json"},
        )

    snapshot = register_snapshot(catalog)
    with pytest.raises(CatalogError, match="CATALOG_EVIDENCE_CEILING"):
        catalog.resolve_snapshot(
            snapshot["snapshot_id"],
            as_of=T0 + timedelta(hours=1),
            requested_evidence_tier="E1",
        )


@pytest.mark.parametrize(
    "query_key",
    [
        "x_api_key",
        "ocp_apim_subscription_key",
        "clientSecret",
        "authToken",
        "accessKey",
    ],
)
def test_catalog_rejects_sensitive_query_key_aliases(tmp_path: Path, query_key: str):
    catalog = DataCatalog(tmp_path / query_key)
    register_adapter(catalog)
    with pytest.raises(CatalogError, match="CATALOG_SECRET_URI"):
        catalog.register_snapshot(
            request_id=f"secret-{query_key}",
            adapter_id="test-public-markets-v1",
            source_uri=f"https://example.test/markets?{query_key}=not-a-secret",
            requested_at=T0,
            retrieved_at=T0,
            available_at=T0,
            cutoff=T0,
            availability_basis="OBSERVED_AT_RETRIEVAL",
            raw_bytes=b"[]",
            normalized_bytes=b"{}\n",
            normalized_schema="test.markets.v1",
            item_count=1,
            max_source_event_time=None,
            source_http_date=None,
            response_headers={"content-type": "application/json"},
        )


def test_adapter_contract_is_immutable_and_research_only(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    contract = adapter_contract()
    first = catalog.register_adapter(contract, registered_at=T0)
    second = catalog.register_adapter(contract, registered_at=T0 + timedelta(hours=1))
    assert first == second

    changed = adapter_contract()
    changed["base_url"] = "https://other.example.test"
    changed["allowed_hosts"] = ["other.example.test"]
    with pytest.raises(CatalogError, match="CATALOG_ADAPTER_CONFLICT"):
        catalog.register_adapter(changed, registered_at=T0)

    unsafe = adapter_contract()
    unsafe["read_only"] = False
    with pytest.raises(CatalogError, match="CATALOG_ADAPTER_UNSAFE"):
        DataCatalog(tmp_path / "other").register_adapter(unsafe, registered_at=T0)

    e1 = adapter_contract()
    e1["evidence_ceiling"] = "E1"
    with pytest.raises(CatalogError, match="CATALOG_EVIDENCE_CEILING"):
        DataCatalog(tmp_path / "e1").register_adapter(e1, registered_at=T0)


def test_registered_at_is_hash_bound(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter(catalog)
    snapshot = register_snapshot(catalog)

    connection = sqlite3.connect(tmp_path / "catalog" / "catalog.sqlite3")
    try:
        connection.execute(
            "UPDATE adapters SET registered_at = ? WHERE adapter_id = ?",
            ("2024-01-01T01:00:00Z", "test-public-markets-v1"),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CatalogError, match="CATALOG_HASH_MISMATCH"):
        catalog.resolve_provenance(
            snapshot["snapshot_id"], as_of=T0 + timedelta(hours=1)
        )


def test_snapshot_cannot_predate_adapter_registration(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    catalog.register_adapter(
        adapter_contract(), registered_at=T0 + timedelta(minutes=1)
    )
    with pytest.raises(CatalogError, match="registered after the request"):
        register_snapshot(catalog)


def test_v011_unbound_adapter_time_is_refused_without_rewriting_objects(tmp_path: Path):
    root = tmp_path / "catalog"
    catalog = DataCatalog(root)
    register_adapter(catalog)
    snapshot = register_snapshot(catalog)
    object_bytes = (root / snapshot["normalized_object_path"]).read_bytes()

    connection = sqlite3.connect(root / "catalog.sqlite3")
    try:
        connection.execute("DELETE FROM catalog_meta WHERE key = 'adapter_hash_scope'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CatalogError, match="CATALOG_LEGACY_HASH_SCOPE"):
        DataCatalog(root)
    assert (root / snapshot["normalized_object_path"]).read_bytes() == object_bytes
