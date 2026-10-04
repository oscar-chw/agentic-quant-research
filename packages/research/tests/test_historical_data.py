from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae import historical_data
from qrae.artifacts import canonical_json_bytes
from qrae.cli import main as cli_main
from qrae.data_catalog import DataCatalog
from qrae.historical_data import (
    MARKET_FAMILIES,
    MARKET_FAMILY_CONTROLS,
    HistoricalDataError,
    catalog_historical_source,
    import_historical_source,
    verify_historical_import_bundle,
)

UTC = timezone.utc
AS_OF = datetime(2025, 1, 3, 0, 0, tzinfo=UTC)
NOW = datetime(2026, 7, 12, 0, 0, tzinfo=UTC)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_ref(root: Path, relative: str, data: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": _sha(data), "size_bytes": len(data)}


def _fixture(root: Path, family: str = "CRYPTO") -> tuple[dict, Path]:
    prices = (
        b"event_time,available_time,instrument,price\n"
        b"2025-01-01T00:00:00Z,2024-12-31T23:59:00Z,TEST,100\n"
        b"2025-01-02T00:00:00Z,2025-01-01T23:59:00Z,TEST,101\n"
    )
    price_ref = _write_ref(root, "data/prices.csv", prices)
    license_ref = _write_ref(
        root,
        "evidence/license.txt",
        b"Provider permits this dataset for local research.\n",
    )
    calendar_ref = _write_ref(
        root,
        "evidence/calendar.txt",
        b"UTC source calendar, continuous observations.\n",
    )
    revision_ref = _write_ref(
        root, "evidence/revisions.txt", b"This fixture is immutable and not restated.\n"
    )
    controls = {
        name: _write_ref(
            root,
            f"evidence/controls/{name}.txt",
            f"Documented control: {name}.\n".encode(),
        )
        for name in MARKET_FAMILY_CONTROLS[family]
    }
    manifest = {
        "schema_version": "qrae.historical-source-manifest.v1",
        "dataset_id": f"fixture-{family.lower()}",
        "market_family": family,
        "provider": "Fixture Provider",
        "provider_dataset_id": "fixture-prices-v1",
        "source_uri": (
            "https://data.example.test/historical/prices.csv?symbol=TEST&interval=1d"
        ),
        "price_data": {**price_ref, "schema": "qrae.price-bars.v1"},
        "license": {
            "use_scope": "LOCAL_RESEARCH_ONLY",
            "evidence_type": "PROVIDER_AGREEMENT",
            "valid_from": "2026-01-01T00:00:00Z",
            "expires_at": "2027-01-01T00:00:00Z",
            "entitlement_id_sha256": _sha(b"fixture-entitlement-id"),
            "evidence": license_ref,
        },
        "bitemporal": {
            "event_time_field": "event_time",
            "available_time_field": "available_time",
            "source_published_at": "2026-07-09T00:00:00Z",
            "acquired_at": "2026-07-10T00:00:00Z",
            "ingested_at": "2026-07-11T00:00:00Z",
        },
        "calendar": {
            "calendar_id": "FIXTURE_CALENDAR",
            "timezone": "UTC",
            "session_policy": "SOURCE_DEFINED",
            "evidence": calendar_ref,
        },
        "revision_policy": {
            "mode": "IMMUTABLE",
            "vintage_id": "INITIAL",
            "supersedes_manifest_sha256": None,
            "evidence": revision_ref,
        },
        "family_controls": controls,
    }
    path = root / "historical-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest, path


def _rewrite_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize("family", sorted(MARKET_FAMILIES))
def test_all_registered_families_require_hashed_controls_and_remain_non_authoritative(
    tmp_path: Path, family: str
):
    _, manifest_path = _fixture(tmp_path, family)

    result = import_historical_source(
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        now=NOW,
    )

    assert result.validation["market_family"] == family
    assert set(result.validation["family_controls"]) == set(
        MARKET_FAMILY_CONTROLS[family]
    )
    assert result.validation["price_data"]["row_count"] == 2
    assert result.validation["price_data"]["max_available_time"] < "2025-01-03"
    assert result.e1_eligibility == {
        "data_input_eligible": False,
        "maximum_candidate_tier": "E1",
        "e1_claim_authorized": False,
        "legal_license_truth_verified": False,
        "basis": "MECHANICAL_CHECKS_PASS_CATALOG_LINEAGE_UNVERIFIED",
        "required_downstream_artifact": "COMPATIBLE_TEMPORAL_VALIDATION",
    }


@pytest.mark.parametrize("target", ["price", "license"])
def test_hash_bound_price_and_license_evidence_reject_tampering(
    tmp_path: Path, target: str
):
    manifest, manifest_path = _fixture(tmp_path)
    relative = (
        manifest["price_data"]["path"]
        if target == "price"
        else manifest["license"]["evidence"]["path"]
    )
    path = tmp_path / relative
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(HistoricalDataError, match="HISTORICAL_HASH"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )


def test_expired_or_not_yet_active_entitlement_fails_closed(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["license"]["expires_at"] = "2026-07-12T00:00:00Z"
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(HistoricalDataError, match="HISTORICAL_LICENSE_EXPIRED"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )

    manifest["license"]["valid_from"] = "2026-07-13T00:00:00Z"
    manifest["license"]["expires_at"] = "2027-01-01T00:00:00Z"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_LICENSE"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )


def test_row_availability_after_as_of_is_rejected_even_when_hashes_match(
    tmp_path: Path,
):
    manifest, manifest_path = _fixture(tmp_path)
    prices = (
        b"event_time,available_time,instrument,price\n"
        b"2025-01-01T00:00:00Z,2025-01-04T00:00:00Z,TEST,100\n"
    )
    manifest["price_data"].update(_write_ref(tmp_path, "data/prices.csv", prices))
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(HistoricalDataError, match="HISTORICAL_PIT"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )


def test_price_rows_must_be_strictly_increasing_per_instrument(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    prices = (
        b"event_time,available_time,instrument,price\n"
        b"2025-01-02T00:00:00Z,2025-01-01T23:59:00Z,TEST,101\n"
        b"2025-01-01T00:00:00Z,2024-12-31T23:59:00Z,TEST,100\n"
    )
    manifest["price_data"].update(_write_ref(tmp_path, "data/prices.csv", prices))
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(HistoricalDataError, match="HISTORICAL_MONOTONICITY"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )


def test_family_control_set_is_exact_and_unknown_families_are_rejected(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path, "EQUITIES_ETFS")
    manifest["family_controls"].pop("corporate_actions")
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_FAMILY_CONTROL"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )

    manifest, manifest_path = _fixture(tmp_path / "unknown")
    manifest["market_family"] = "UNREGISTERED"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_FAMILY_CONTROL"):
        import_historical_source(
            manifest_path,
            workspace=tmp_path / "unknown",
            as_of=AS_OF,
            now=NOW,
        )


def test_revision_mode_controls_duplicate_event_vintages(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    prices = (
        b"event_time,available_time,instrument,price\n"
        b"2025-01-01T00:00:00Z,2024-12-31T23:59:00Z,TEST,100\n"
        b"2025-01-01T00:00:00Z,2025-01-01T00:00:00Z,TEST,101\n"
    )
    manifest["price_data"].update(_write_ref(tmp_path, "data/prices.csv", prices))
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_REVISION"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )

    manifest["revision_policy"]["mode"] = "VINTAGED"
    manifest["revision_policy"]["vintage_id"] = "VINTAGE_20250103"
    manifest["revision_policy"]["supersedes_manifest_sha256"] = _sha(b"prior-manifest")
    prices = (
        b"event_time,available_time,instrument,price\n"
        b"2025-01-01T00:00:00Z,2024-12-31T23:59:00Z,TEST,100\n"
        b"2025-01-02T00:00:00Z,2025-01-02T00:00:00Z,TEST,101\n"
    )
    manifest["price_data"].update(_write_ref(tmp_path, "data/prices.csv", prices))
    _rewrite_manifest(manifest_path, manifest)
    result = import_historical_source(
        manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
    )
    assert result.validation["price_data"]["row_count"] == 2
    assert result.validation["revision_policy"]["mode"] == "VINTAGED"


def test_calendar_paths_and_secret_like_evidence_fail_closed(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["calendar"]["timezone"] = "Not/A_Real_Zone"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_CALENDAR"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )

    manifest, manifest_path = _fixture(tmp_path / "secret")
    secret = b"api_key = abcdefghijklmnopqrstuvwxyz\n"
    manifest["license"]["evidence"].update(
        _write_ref(tmp_path / "secret", "evidence/license.txt", secret)
    )
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_SECRET"):
        import_historical_source(
            manifest_path,
            workspace=tmp_path / "secret",
            as_of=AS_OF,
            now=NOW,
        )

    manifest, manifest_path = _fixture(tmp_path / "escape")
    manifest["price_data"]["path"] = "../outside.csv"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_PATH"):
        import_historical_source(
            manifest_path,
            workspace=tmp_path / "escape",
            as_of=AS_OF,
            now=NOW,
        )

    manifest, manifest_path = _fixture(tmp_path / "uri")
    manifest["source_uri"] = "https://user:password@data.example.test/prices.csv"
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_SOURCE_URI"):
        import_historical_source(
            manifest_path,
            workspace=tmp_path / "uri",
            as_of=AS_OF,
            now=NOW,
        )

    manifest, manifest_path = _fixture(tmp_path / "encoded-uri")
    manifest["source_uri"] = (
        "https://data.example.test/prices.csv?api%255Fkey=abcdefghijklmnop"
    )
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_SOURCE_URI"):
        import_historical_source(
            manifest_path,
            workspace=tmp_path / "encoded-uri",
            as_of=AS_OF,
            now=NOW,
        )


def test_non_utc_iana_calendar_is_portable_on_windows(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["calendar"]["timezone"] = "America/New_York"
    _rewrite_manifest(manifest_path, manifest)

    result = import_historical_source(
        manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
    )
    assert result.validation["calendar"]["timezone"] == "America/New_York"


@pytest.mark.parametrize(
    "query_key",
    [
        "x_api_key",
        "ocp_apim_subscription_key",
        "x%255Fapi%255Fkey",
        "clientSecret",
        "authToken",
        "accessKey",
    ],
)
def test_source_uri_rejects_sensitive_query_key_aliases(tmp_path: Path, query_key: str):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["source_uri"] = (
        f"https://data.example.test/prices.csv?{query_key}=not-a-secret"
    )
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(HistoricalDataError, match=r"HISTORICAL_(SOURCE_URI|SECRET)"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )


@pytest.mark.parametrize(
    "evidence",
    [
        b"clientSecret = abcdefghijklmnopqrstuvwxyz\n",
        b"x_api_key = abcdefghijklmnopqrstuvwxyz\n",
        b"aws_secret_access_key = abcdefghijklmnopqrstuvwxyz\n",
        b"client_secret = abcdefghijklmnopqrstuvwxyz\n",
        b'{"x_api_key": "abcdefghijklmnopqrstuvwxyz"}\n',
        b'{"aws_secret_access_key": "abcdefghijklmnopqrstuvwxyz"}\n',
        b'{"client_secret": "abcdefghijklmnopqrstuvwxyz"}\n',
    ],
)
def test_secret_assignment_alias_in_evidence_fails_closed(
    tmp_path: Path, evidence: bytes
):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["license"]["evidence"].update(
        _write_ref(tmp_path, "evidence/license.txt", evidence)
    )
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_SECRET"):
        import_historical_source(
            manifest_path, workspace=tmp_path, as_of=AS_OF, now=NOW
        )


def test_catalog_helper_freezes_validated_local_import_bundle_and_prices_at_e0(
    tmp_path: Path,
):
    _, manifest_path = _fixture(tmp_path)
    exact_manifest = manifest_path.read_bytes()
    exact_prices = (tmp_path / "data" / "prices.csv").read_bytes()
    catalog = DataCatalog(tmp_path / "catalog")

    result = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="licensed-history-fixture-001",
        now=NOW,
    )
    snapshot = catalog.resolve_snapshot(result["catalog"]["snapshot_id"], as_of=NOW)
    provenance = catalog.resolve_provenance(snapshot["snapshot_id"], as_of=NOW)

    bundle_bytes = (catalog.root / snapshot["raw_object_path"]).read_bytes()
    bundle = verify_historical_import_bundle(
        bundle_bytes, normalized_price_bytes=exact_prices, now=NOW
    )
    with pytest.raises(HistoricalDataError, match="HISTORICAL_HASH"):
        verify_historical_import_bundle(
            bundle_bytes,
            normalized_price_bytes=exact_prices + b"\n",
            now=NOW,
        )
    assert (
        catalog.root / snapshot["normalized_object_path"]
    ).read_bytes() == exact_prices
    assert bundle["kind"] == "LOCAL_MANIFEST_IMPORT"
    assert bundle["validated_as_of"] == "2025-01-03T00:00:00Z"
    assert bundle["validated_at"] == "2026-07-12T00:00:00Z"
    assert bundle["source_manifest"]["sha256"] == _sha(exact_manifest)
    assert bundle["source_manifest"]["size_bytes"] == len(exact_manifest)
    assert bundle["source_manifest"]["content_utf8"].encode() == exact_manifest
    assert bundle["validation"]["revision_policy"]["lineage_status"] == (
        "CATALOG_ROOT_VALIDATED"
    )
    assert snapshot["raw_sha256"] == _sha(bundle_bytes)
    assert snapshot["normalized_sha256"] == _sha(exact_prices)
    assert result["catalog"]["raw_object_path"] == snapshot["raw_object_path"]
    assert (
        result["catalog"]["normalized_object_path"]
        == snapshot["normalized_object_path"]
    )
    assert result["catalog"]["normalized_price_size_bytes"] == len(exact_prices)
    assert snapshot["evidence_ceiling"] == "E0"
    assert snapshot["availability_basis"] == "OBSERVED_AT_IMPORT"
    assert provenance["adapter"]["source_kind"] == "LOCAL_MANIFEST_IMPORT"
    assert provenance["adapter"]["auth_mode"] == "manifest-declared"
    assert snapshot["response_headers"] == {}
    assert result["e1_eligibility"]["data_input_eligible"] is True
    assert result["e1_eligibility"]["e1_claim_authorized"] is False
    assert result["e1_eligibility"]["legal_license_truth_verified"] is False


@pytest.mark.parametrize(
    ("derived_class", "field_path", "forged_value"),
    [
        ("identity", ("dataset_id",), "forged-dataset"),
        ("price-schema", ("price_data", "schema"), "qrae.other-bars.v1"),
        ("price-ref", ("price_data", "sha256"), "f" * 64),
        ("price-quality", ("price_data", "row_count"), 1),
        (
            "license",
            ("license_evidence", "entitlement_id_sha256"),
            "f" * 64,
        ),
        (
            "bitemporal",
            ("bitemporal", "source_published_at"),
            "2026-07-08T00:00:00Z",
        ),
        ("calendar", ("calendar", "timezone"), "Asia/Taipei"),
        ("revision", ("revision_policy", "mode"), "RESTATABLE"),
        (
            "family-controls",
            ("family_controls", "fork_delisting_policy", "sha256"),
            "f" * 64,
        ),
        ("eligibility", ("e1_eligibility", "maximum_candidate_tier"), "E5"),
        ("lineage", ("revision_policy", "lineage_status"), "ROOT_DECLARED"),
        ("times", ("validated_at",), "2026-07-12T00:00:01Z"),
    ],
)
def test_frozen_bundle_rejects_rehashed_manifest_validation_conflicts(
    tmp_path: Path,
    derived_class: str,
    field_path: tuple[str, ...],
    forged_value: str,
):
    _, manifest_path = _fixture(tmp_path)
    catalog = DataCatalog(tmp_path / "catalog")
    result = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="forged-validation-source",
        now=NOW,
    )
    snapshot = catalog.resolve_request(
        result["catalog"]["adapter_id"], "forged-validation-source"
    )
    assert snapshot is not None
    bundle = json.loads((catalog.root / snapshot["raw_object_path"]).read_bytes())
    normalized = (catalog.root / snapshot["normalized_object_path"]).read_bytes()
    target = bundle["validation"]
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = forged_value
    bundle["validation_sha256"] = _sha(canonical_json_bytes(bundle["validation"]))

    assert derived_class
    with pytest.raises(HistoricalDataError, match="HISTORICAL_BUNDLE"):
        verify_historical_import_bundle(
            canonical_json_bytes(bundle),
            normalized_price_bytes=normalized,
            now=NOW,
        )


def test_revision_lineage_must_extend_latest_matching_dataset_without_time_regression(
    tmp_path: Path,
):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["revision_policy"].update(
        {"mode": "VINTAGED", "vintage_id": "ROOT", "supersedes_manifest_sha256": None}
    )
    _rewrite_manifest(manifest_path, manifest)
    catalog = DataCatalog(tmp_path / "catalog")
    root = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="history-root",
        now=NOW,
    )

    manifest["revision_policy"].update(
        {
            "vintage_id": "REVISION_2",
            "supersedes_manifest_sha256": root["canonical_manifest_sha256"],
        }
    )
    manifest["bitemporal"].update(
        {
            "source_published_at": "2026-07-12T01:00:00Z",
            "acquired_at": "2026-07-12T02:00:00Z",
            "ingested_at": "2026-07-12T03:00:00Z",
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    revision = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="history-revision-2",
        now=NOW + timedelta(days=1),
    )
    assert revision["revision_policy"]["lineage_status"] == "VALIDATED_AGAINST_LATEST"
    assert (
        revision["revision_policy"]["parent_snapshot_id"]
        == root["catalog"]["snapshot_id"]
    )

    manifest["revision_policy"]["vintage_id"] = "FORK"
    manifest["revision_policy"]["supersedes_manifest_sha256"] = root[
        "canonical_manifest_sha256"
    ]
    manifest["bitemporal"].update(
        {
            "source_published_at": "2026-07-13T01:00:00Z",
            "acquired_at": "2026-07-13T02:00:00Z",
            "ingested_at": "2026-07-13T03:00:00Z",
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_REVISION_LINEAGE"):
        catalog_historical_source(
            catalog,
            manifest_path,
            workspace=tmp_path,
            as_of=AS_OF,
            request_id="history-fork",
            now=NOW + timedelta(days=2),
        )


def test_orphan_revision_and_expired_frozen_bundle_fail_closed(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["revision_policy"].update(
        {
            "mode": "VINTAGED",
            "vintage_id": "ORPHAN",
            "supersedes_manifest_sha256": _sha(b"missing-parent"),
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="HISTORICAL_REVISION_LINEAGE"):
        catalog_historical_source(
            DataCatalog(tmp_path / "catalog"),
            manifest_path,
            workspace=tmp_path,
            as_of=AS_OF,
            request_id="history-orphan",
            now=NOW,
        )

    _root_manifest, root_path = _fixture(tmp_path / "expiry")
    catalog = DataCatalog(tmp_path / "expiry-catalog")
    result = catalog_historical_source(
        catalog,
        root_path,
        workspace=tmp_path / "expiry",
        as_of=AS_OF,
        request_id="history-expiry",
        now=NOW,
    )
    snapshot = catalog.resolve_snapshot(result["catalog"]["snapshot_id"], as_of=NOW)
    raw = (catalog.root / snapshot["raw_object_path"]).read_bytes()
    normalized = (catalog.root / snapshot["normalized_object_path"]).read_bytes()
    forged = json.loads(raw)
    forged["validation"]["dataset_id"] = "forged"
    with pytest.raises(HistoricalDataError, match="HISTORICAL_HASH"):
        verify_historical_import_bundle(
            canonical_json_bytes(forged),
            normalized_price_bytes=normalized,
            now=NOW,
        )
    with pytest.raises(HistoricalDataError, match="HISTORICAL_LICENSE_EXPIRED"):
        verify_historical_import_bundle(
            raw,
            normalized_price_bytes=normalized,
            now=datetime(2027, 1, 1, 0, 0, tzinfo=UTC),
        )


def test_revision_lineage_rejects_ingestion_time_regression(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["revision_policy"].update(
        {"mode": "RESTATABLE", "vintage_id": "ROOT", "supersedes_manifest_sha256": None}
    )
    _rewrite_manifest(manifest_path, manifest)
    catalog = DataCatalog(tmp_path / "catalog")
    root = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="time-root",
        now=NOW,
    )
    manifest["revision_policy"].update(
        {
            "vintage_id": "REGRESSED",
            "supersedes_manifest_sha256": root["canonical_manifest_sha256"],
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="timestamps regress"):
        catalog_historical_source(
            catalog,
            manifest_path,
            workspace=tmp_path,
            as_of=AS_OF,
            request_id="time-regressed",
            now=NOW + timedelta(days=1),
        )


def test_backdated_revision_cannot_fork_behind_absolute_latest(tmp_path: Path):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["revision_policy"].update(
        {"mode": "VINTAGED", "vintage_id": "ROOT", "supersedes_manifest_sha256": None}
    )
    _rewrite_manifest(manifest_path, manifest)
    catalog = DataCatalog(tmp_path / "catalog")
    root = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="absolute-root",
        now=NOW,
    )

    manifest["revision_policy"].update(
        {
            "vintage_id": "LATEST",
            "supersedes_manifest_sha256": root["canonical_manifest_sha256"],
        }
    )
    manifest["bitemporal"].update(
        {
            "source_published_at": "2026-07-13T01:00:00Z",
            "acquired_at": "2026-07-13T02:00:00Z",
            "ingested_at": "2026-07-13T03:00:00Z",
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="absolute-latest",
        now=NOW + timedelta(days=2),
    )

    manifest["revision_policy"]["vintage_id"] = "BACKDATED_FORK"
    manifest["revision_policy"]["supersedes_manifest_sha256"] = root[
        "canonical_manifest_sha256"
    ]
    manifest["bitemporal"].update(
        {
            "source_published_at": "2026-07-12T01:00:00Z",
            "acquired_at": "2026-07-12T02:00:00Z",
            "ingested_at": "2026-07-12T03:00:00Z",
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    with pytest.raises(HistoricalDataError, match="absolute latest"):
        catalog_historical_source(
            catalog,
            manifest_path,
            workspace=tmp_path,
            as_of=AS_OF,
            request_id="absolute-backdated-fork",
            now=NOW + timedelta(days=1),
        )


def test_historical_import_exact_replay_is_idempotent_before_and_after_revision(
    tmp_path: Path,
):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["revision_policy"].update(
        {"mode": "VINTAGED", "vintage_id": "ROOT", "supersedes_manifest_sha256": None}
    )
    _rewrite_manifest(manifest_path, manifest)
    root_bytes = manifest_path.read_bytes()
    catalog = DataCatalog(tmp_path / "catalog")
    first = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="idempotent-root",
        now=NOW,
    )
    immediate = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="idempotent-root",
        now=NOW + timedelta(hours=1),
    )
    assert immediate["catalog"]["snapshot_id"] == first["catalog"]["snapshot_id"]
    assert catalog.stats()["snapshots"] == 1

    manifest["revision_policy"].update(
        {
            "vintage_id": "REVISION",
            "supersedes_manifest_sha256": first["canonical_manifest_sha256"],
        }
    )
    manifest["bitemporal"].update(
        {
            "source_published_at": "2026-07-12T01:00:00Z",
            "acquired_at": "2026-07-12T02:00:00Z",
            "ingested_at": "2026-07-12T03:00:00Z",
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="idempotent-revision",
        now=NOW + timedelta(days=1),
    )

    manifest_path.write_bytes(root_bytes)
    after_revision = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="idempotent-root",
        now=NOW + timedelta(days=2),
    )
    assert after_revision["catalog"]["snapshot_id"] == first["catalog"]["snapshot_id"]
    assert catalog.stats()["snapshots"] == 2


def test_concurrent_exact_historical_import_reuses_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, manifest_path = _fixture(tmp_path)
    catalog = DataCatalog(tmp_path / "catalog")
    register_adapter = catalog.register_adapter
    register_snapshot = catalog.register_snapshot
    later_adapter_registered = threading.Event()
    later_snapshot_registered = threading.Event()
    ready = threading.Barrier(2)

    def ordered_register_adapter(contract, *, registered_at):
        if registered_at == NOW:
            assert later_adapter_registered.wait(timeout=5)
            return register_adapter(contract, registered_at=registered_at)
        result = register_adapter(contract, registered_at=registered_at)
        later_adapter_registered.set()
        return result

    def synchronized_register(**kwargs):
        ready.wait(timeout=5)
        bundle = json.loads(kwargs["raw_bytes"])
        if bundle["validated_at"] == "2026-07-12T00:00:00Z":
            assert later_snapshot_registered.wait(timeout=5)
            return register_snapshot(**kwargs)
        try:
            return register_snapshot(**kwargs)
        finally:
            later_snapshot_registered.set()

    monkeypatch.setattr(catalog, "register_adapter", ordered_register_adapter)
    monkeypatch.setattr(catalog, "register_snapshot", synchronized_register)

    def run_import(observed: datetime):
        return catalog_historical_source(
            catalog,
            manifest_path,
            workspace=tmp_path,
            as_of=AS_OF,
            request_id="concurrent-idempotent-root",
            now=observed,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_import, (NOW, NOW + timedelta(seconds=1))))

    assert len({result["catalog"]["snapshot_id"] for result in results}) == 1
    assert catalog.stats()["snapshots"] == 1


def test_concurrent_exact_revision_with_different_times_reuses_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest, manifest_path = _fixture(tmp_path)
    manifest["revision_policy"].update(
        {"mode": "VINTAGED", "vintage_id": "ROOT", "supersedes_manifest_sha256": None}
    )
    _rewrite_manifest(manifest_path, manifest)
    catalog = DataCatalog(tmp_path / "catalog")
    root = catalog_historical_source(
        catalog,
        manifest_path,
        workspace=tmp_path,
        as_of=AS_OF,
        request_id="concurrent-revision-root",
        now=NOW,
    )
    manifest["revision_policy"].update(
        {
            "vintage_id": "REVISION",
            "supersedes_manifest_sha256": root["canonical_manifest_sha256"],
        }
    )
    manifest["bitemporal"].update(
        {
            "source_published_at": "2026-07-12T01:00:00Z",
            "acquired_at": "2026-07-12T02:00:00Z",
            "ingested_at": "2026-07-12T03:00:00Z",
        }
    )
    _rewrite_manifest(manifest_path, manifest)
    original_lineage = historical_data._catalog_lineage_validation
    register_snapshot = catalog.register_snapshot
    earlier_waiting = threading.Event()
    later_snapshot_registered = threading.Event()
    earlier = NOW + timedelta(days=1)
    later = earlier + timedelta(seconds=1)

    def ordered_lineage(*args, observed, **kwargs):
        if observed == earlier:
            earlier_waiting.set()
            assert later_snapshot_registered.wait(timeout=5)
        return original_lineage(*args, observed=observed, **kwargs)

    def signal_later_register(**kwargs):
        bundle = json.loads(kwargs["raw_bytes"])
        try:
            return register_snapshot(**kwargs)
        finally:
            if bundle["validated_at"] == later.isoformat().replace("+00:00", "Z"):
                later_snapshot_registered.set()

    monkeypatch.setattr(historical_data, "_catalog_lineage_validation", ordered_lineage)
    monkeypatch.setattr(catalog, "register_snapshot", signal_later_register)

    def run_import(observed: datetime):
        return catalog_historical_source(
            catalog,
            manifest_path,
            workspace=tmp_path,
            as_of=AS_OF,
            request_id="concurrent-idempotent-revision",
            now=observed,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        earlier_future = pool.submit(run_import, earlier)
        assert earlier_waiting.wait(timeout=5)
        later_future = pool.submit(run_import, later)
        results = [earlier_future.result(), later_future.result()]

    assert len({result["catalog"]["snapshot_id"] for result in results}) == 1
    assert catalog.stats()["snapshots"] == 2


def test_historical_import_cli_catalogs_candidate_without_e1_claim(
    tmp_path: Path, capsys
):
    _, manifest_path = _fixture(tmp_path)
    code = cli_main(
        [
            "historical-import",
            "--catalog-root",
            str(tmp_path / "catalog"),
            "--manifest",
            str(manifest_path),
            "--workspace",
            str(tmp_path),
            "--as-of",
            "2025-01-03T00:00:00Z",
            "--request-id",
            "historical-cli-001",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["catalog"]["evidence_ceiling"] == "E0"
    assert payload["e1_eligibility"]["e1_claim_authorized"] is False
