"""Import locally generated price fixtures without claiming an HTTPS origin."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes
from .data_catalog import CatalogError, DataCatalog, _synthetic_derivation
from .path_safety import has_link_or_reparse_component
from .price_baseline import load_price_rows

MAX_SYNTHETIC_PRICE_BYTES = 64 * 1024 * 1024
_COLUMNS = ["event_time", "available_time", "instrument", "price"]


def import_synthetic_prices(
    catalog: DataCatalog,
    csv_path: str | Path,
    derivation: Mapping[str, Any],
    observed: datetime,
) -> dict[str, Any]:
    """Register one declared synthetic derivation and its exact price CSV at E0.

    The source URI hashes the canonical derivation manifest. Upstream/generator
    hashes are recorded declarations, not proof that those inputs were executed.
    This helper does not fetch a source or claim historical market availability.
    """

    if not isinstance(catalog, DataCatalog) or not isinstance(derivation, Mapping):
        raise CatalogError(
            "CATALOG_SYNTHETIC_ARGUMENT: catalog or derivation is invalid"
        )
    if (
        not isinstance(observed, datetime)
        or observed.tzinfo is None
        or observed.utcoffset() != timedelta(0)
    ):
        raise CatalogError(
            "CATALOG_SYNTHETIC_TIME: observed must be timezone-aware UTC"
        )
    try:
        raw_bytes = canonical_json_bytes(dict(derivation))
    except (TypeError, ValueError) as exc:
        raise CatalogError(
            "CATALOG_SYNTHETIC_SCHEMA: derivation must be finite JSON"
        ) from exc
    manifest = _synthetic_derivation(raw_bytes)
    source = Path(csv_path)
    if has_link_or_reparse_component(source) or not source.is_file():
        raise CatalogError("CATALOG_SYNTHETIC_PATH: CSV must be a regular local file")
    size = source.stat().st_size
    if not 0 < size <= MAX_SYNTHETIC_PRICE_BYTES:
        raise CatalogError(
            "CATALOG_SYNTHETIC_SIZE: CSV must be nonempty and at most 64 MiB"
        )
    price_bytes = source.read_bytes()
    if (
        len(price_bytes) != size
        or sha256_bytes(price_bytes) != manifest["normalized_price_sha256"]
    ):
        raise CatalogError(
            "CATALOG_SYNTHETIC_BINDING: CSV differs from derivation hash"
        )
    try:
        reader = csv.reader(io.StringIO(price_bytes.decode("utf-8"), newline=""))
        header = next(reader)
        exact_widths = all(len(row) == len(_COLUMNS) for row in reader)
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise CatalogError("CATALOG_SYNTHETIC_CSV: invalid UTF-8 price CSV") from exc
    if header != _COLUMNS or not exact_widths:
        raise CatalogError(
            "CATALOG_SYNTHETIC_CSV: exact qrae.price-bars.v1 columns required"
        )
    rows, _ = load_price_rows(source, observed)
    if any(row.available_time != row.event_time for row in rows):
        raise CatalogError(
            "CATALOG_SYNTHETIC_TIME: fixture event and availability times must agree"
        )
    if source.read_bytes() != price_bytes:
        raise CatalogError("CATALOG_SYNTHETIC_BINDING: CSV changed during validation")

    raw_hash = sha256_bytes(raw_bytes)
    adapter_id = "synthetic-" + sha256_bytes(
        canonical_json_bytes(
            {
                "generator_sha256": manifest["generator_sha256"],
                "market_family": manifest["market_family"],
            }
        )
    )
    contract = {
        "schema_version": "1.1",
        "adapter_id": adapter_id,
        "adapter_version": "1.0.0",
        "market_family": manifest["market_family"],
        "source_kind": "LOCAL_SYNTHETIC",
        "base_url": "urn:qrae:synthetic",
        "allowed_hosts": [],
        "read_only": True,
        "auth_mode": "none",
        "data_tier": "TIER_0",
        "normalized_schema": "qrae.price-bars.v1",
        "license_status": "SYNTHETIC_FIXTURE_ONLY;NO_EXTERNAL_ORIGIN_CLAIM",
        "evidence_ceiling": "E0",
    }
    stamp = observed.isoformat().replace("+00:00", "Z")
    request_id = "synthetic-" + sha256_bytes(
        canonical_json_bytes({"derivation_sha256": raw_hash, "observed": stamp})
    )
    catalog.register_adapter(contract, registered_at=observed)
    return catalog.register_snapshot(
        request_id=request_id,
        adapter_id=adapter_id,
        source_uri="urn:qrae:synthetic:" + raw_hash,
        requested_at=observed,
        retrieved_at=observed,
        available_at=observed,
        cutoff=observed,
        availability_basis="OBSERVED_AT_IMPORT",
        raw_bytes=raw_bytes,
        normalized_bytes=price_bytes,
        normalized_schema="qrae.price-bars.v1",
        item_count=len(rows),
        max_source_event_time=max(row.event_time for row in rows),
        source_http_date=None,
        response_headers={},
    )


__all__ = ["import_synthetic_prices"]
