"""Offline import of observed HTTPS price-bar CSVs into the E0 catalog."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_catalog import DataCatalog
from .path_safety import has_link_or_reparse_component
from .price_baseline import load_price_rows


class PriceBarImportError(ValueError):
    """A local price-bar import violated its bounded evidence contract."""

    def __init__(self, message: str) -> None:
        self.code = (
            message.partition(":")[0] if ":" in message else "PRICE_IMPORT_ERROR"
        )
        super().__init__(message)


NORMALIZED_SCHEMA = "qrae.price-bars.v1"
MAX_IMPORT_BYTES = 64 * 1024 * 1024
MAX_CONTRACT_BYTES = 256 * 1024
_COLUMNS = ["event_time", "available_time", "instrument", "price"]

Clock = Callable[[], datetime]


def load_adapter_contract(path: str | Path) -> dict[str, Any]:
    """Load a strict JSON adapter contract without accepting duplicate keys."""

    source = Path(path)
    if has_link_or_reparse_component(source) or not source.is_file():
        raise PriceBarImportError(
            "PRICE_IMPORT_PATH: adapter contract must be a regular file"
        )
    if source.stat().st_size <= 0 or source.stat().st_size > MAX_CONTRACT_BYTES:
        raise PriceBarImportError(
            "PRICE_IMPORT_CONTRACT: adapter contract is empty or exceeds 256 KiB"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PriceBarImportError(
                    f"PRICE_IMPORT_CONTRACT: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PriceBarImportError(f"PRICE_IMPORT_CONTRACT: non-finite {constant}")
            ),
        )
    except PriceBarImportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PriceBarImportError(
            "PRICE_IMPORT_CONTRACT: adapter contract is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PriceBarImportError(
            "PRICE_IMPORT_CONTRACT: adapter contract must be an object"
        )
    return value


def import_price_bars(
    catalog: DataCatalog,
    *,
    adapter_contract: Mapping[str, Any],
    request_id: str,
    source_uri: str,
    csv_path: str | Path,
    clock: Clock = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Validate and catalog one local observation of an HTTPS price-bar CSV.

    This does not prove remote origin, historical availability, licensing, or alpha.
    The observation is therefore immutable but capped at E0.
    """

    if not isinstance(catalog, DataCatalog) or not isinstance(
        adapter_contract, Mapping
    ):
        raise PriceBarImportError(
            "PRICE_IMPORT_ARGUMENT: catalog or contract is invalid"
        )
    if adapter_contract.get("source_kind") != "HTTPS_BYTES":
        raise PriceBarImportError(
            "PRICE_IMPORT_CONTRACT: price imports require source_kind HTTPS_BYTES"
        )
    if adapter_contract.get("normalized_schema") != NORMALIZED_SCHEMA:
        raise PriceBarImportError(
            f"PRICE_IMPORT_CONTRACT: normalized_schema must be {NORMALIZED_SCHEMA}"
        )

    source = Path(csv_path)
    if has_link_or_reparse_component(source) or not source.is_file():
        raise PriceBarImportError("PRICE_IMPORT_PATH: CSV must be a regular file")
    size = source.stat().st_size
    if size <= 0 or size > MAX_IMPORT_BYTES:
        raise PriceBarImportError("PRICE_IMPORT_SIZE: CSV is empty or exceeds 64 MiB")
    before = source.read_bytes()
    if len(before) != size or b"\x00" in before:
        raise PriceBarImportError(
            "PRICE_IMPORT_CONTENT: CSV changed or contains NUL bytes"
        )
    try:
        text = before.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""))
        header = next(reader)
        widths_are_exact = all(len(row) == len(_COLUMNS) for row in reader)
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise PriceBarImportError(
            "PRICE_IMPORT_CONTENT: CSV must be non-empty strict UTF-8"
        ) from exc
    if header != _COLUMNS or not widths_are_exact:
        raise PriceBarImportError(
            "PRICE_IMPORT_CONTENT: every CSV row must exactly match qrae.price-bars.v1"
        )

    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PriceBarImportError(
            "PRICE_IMPORT_TIME: clock must return a timezone-aware value"
        )
    observed = observed.astimezone(timezone.utc)
    rows, _ = load_price_rows(source, observed)
    after = source.read_bytes()
    if after != before:
        raise PriceBarImportError("PRICE_IMPORT_CHANGED: CSV changed during validation")

    adapter = catalog.register_adapter(adapter_contract, registered_at=observed)
    return catalog.register_snapshot(
        request_id=request_id,
        adapter_id=adapter["adapter_id"],
        source_uri=source_uri,
        requested_at=observed,
        retrieved_at=observed,
        available_at=observed,
        cutoff=observed,
        availability_basis="OBSERVED_AT_IMPORT",
        raw_bytes=before,
        normalized_bytes=before,
        normalized_schema=NORMALIZED_SCHEMA,
        item_count=len(rows),
        max_source_event_time=max(row.event_time for row in rows),
        source_http_date=None,
        response_headers={"content-type": "text/csv"},
    )


__all__ = [
    "MAX_CONTRACT_BYTES",
    "MAX_IMPORT_BYTES",
    "NORMALIZED_SCHEMA",
    "PriceBarImportError",
    "import_price_bars",
    "load_adapter_contract",
]
