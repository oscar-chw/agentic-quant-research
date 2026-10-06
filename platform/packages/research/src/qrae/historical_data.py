"""Fail-closed validation for licensed historical research-data manifests.

This module verifies local files, declared entitlement evidence, and mechanical
point-in-time controls.  It deliberately cannot determine whether a licence is
legally valid and never authorizes an E1 claim by itself.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .data_catalog import CatalogError, DataCatalog
from .path_safety import has_link_or_reparse_component


class HistoricalDataError(ValueError):
    """A historical source failed its data-integrity contract."""

    def __init__(self, message: str) -> None:
        self.code = (
            message.partition(":")[0] if ":" in message else "HISTORICAL_DATA_ERROR"
        )
        super().__init__(message)


SCHEMA_VERSION = "qrae.historical-source-manifest.v1"
PRICE_SCHEMA = "qrae.price-bars.v1"
MAX_MANIFEST_BYTES = 512 * 1024
MAX_PRICE_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_IMPORT_BUNDLE_BYTES = 2 * 1024 * 1024

MARKET_FAMILY_CONTROLS: dict[str, frozenset[str]] = {
    "PREDICTION_MARKETS": frozenset(
        {"resolution_settlement_rules", "market_lifecycle"}
    ),
    "CRYPTO": frozenset({"venue_symbol_history", "fork_delisting_policy"}),
    "EQUITIES_ETFS": frozenset({"corporate_actions", "point_in_time_universe"}),
    "FUTURES_RATES_COMMODITIES_FX": frozenset(
        {"contract_roll_or_vintage", "settlement_fixing_rules"}
    ),
    "OPTIONS": frozenset(
        {"contract_terms_symbology", "exercise_assignment_adjustments"}
    ),
    "ALTERNATIVE_SOCIAL_WALLET": frozenset(
        {"collection_methodology", "entity_identity_revision_policy"}
    ),
}
MARKET_FAMILIES = frozenset(MARKET_FAMILY_CONTROLS)

_MANIFEST_FIELDS = {
    "schema_version",
    "dataset_id",
    "market_family",
    "provider",
    "provider_dataset_id",
    "source_uri",
    "price_data",
    "license",
    "bitemporal",
    "calendar",
    "revision_policy",
    "family_controls",
}
_FILE_REF_FIELDS = {"path", "sha256", "size_bytes"}
_PRICE_FIELDS = _FILE_REF_FIELDS | {"schema"}
_LICENSE_FIELDS = {
    "use_scope",
    "evidence_type",
    "valid_from",
    "expires_at",
    "entitlement_id_sha256",
    "evidence",
}
_BITEMPORAL_FIELDS = {
    "event_time_field",
    "available_time_field",
    "source_published_at",
    "acquired_at",
    "ingested_at",
}
_CALENDAR_FIELDS = {
    "calendar_id",
    "timezone",
    "session_policy",
    "evidence",
}
_REVISION_FIELDS = {
    "mode",
    "vintage_id",
    "supersedes_manifest_sha256",
    "evidence",
}
_REVISION_MODES = {"IMMUTABLE", "APPEND_ONLY", "VINTAGED", "RESTATABLE"}
_SESSION_POLICIES = {
    "CONTINUOUS_24_7",
    "EXCHANGE_SESSIONS",
    "BUSINESS_DAYS",
    "EVENT_DEFINED",
    "SOURCE_DEFINED",
}
_EVIDENCE_TYPES = {
    "PROVIDER_AGREEMENT",
    "DATASET_TERMS",
    "ENTITLEMENT_RECORD",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\b(?:sk|rk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        rb"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|secret)\b"
        rb"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}"
    ),
    re.compile(
        rb"(?i)\b(?:clientsecret|authtoken|accesskey|subscriptionkey|bearertoken)\b"
        rb"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}"
    ),
)
_SECRET_ASSIGNMENT_RE = re.compile(
    rb"(?im)(?:[\"']|\\[\"'])?"
    rb"(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,127})"
    rb"(?:[\"']|\\[\"'])?[ \t]*[:=][ \t]*(?:[\"']|\\[\"'])?"
    rb"(?P<value>[A-Za-z0-9_./+=-]{12,})"
)
_SECRET_QUERY_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "access-key",
    "access_key",
    "access-token",
    "access_token",
    "authorization",
    "auth",
    "bearer",
    "credential",
    "key",
    "password",
    "passwd",
    "secret",
    "signature",
    "sig",
    "token",
}
_SECRET_QUERY_SUFFIXES = (
    "_auth",
    "_authorization",
    "_bearer",
    "_credential",
    "_key",
    "_password",
    "_secret",
    "_sig",
    "_signature",
    "_token",
)
_SECRET_COMPACT_SUFFIXES = (
    "accesskey",
    "accesstoken",
    "apikey",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "subscriptionkey",
)
_CSV_COLUMNS = ["event_time", "available_time", "instrument", "price"]
_IMPORT_BUNDLE_FIELDS = {
    "schema_version",
    "kind",
    "source_manifest",
    "validation",
    "validation_sha256",
    "validated_as_of",
    "validated_at",
}
_SOURCE_MANIFEST_BUNDLE_FIELDS = {
    "sha256",
    "size_bytes",
    "canonical_sha256",
    "content_utf8",
}
_IMPORT_VALIDATION_FIELDS = {
    "schema_version",
    "dataset_id",
    "market_family",
    "provider",
    "provider_dataset_id",
    "source_uri",
    "manifest_sha256",
    "canonical_manifest_sha256",
    "price_data",
    "license_evidence",
    "bitemporal",
    "calendar",
    "revision_policy",
    "family_controls",
    "e1_eligibility",
    "validated_as_of",
    "validated_at",
}
_LICENSE_VALIDATION_FIELDS = _FILE_REF_FIELDS | {
    "evidence_type",
    "use_scope",
    "entitlement_id_sha256",
    "valid_from",
    "expires_at",
    "integrity_scope",
}
_REVISION_VALIDATION_FIELDS = _REVISION_FIELDS | {
    "lineage_status",
    "parent_snapshot_id",
}
_PRICE_QUALITY_FIELDS = {
    "row_count",
    "instrument_count",
    "min_event_time",
    "max_event_time",
    "min_available_time",
    "max_available_time",
}
_PRICE_VALIDATION_FIELDS = _PRICE_FIELDS | _PRICE_QUALITY_FIELDS
_BITEMPORAL_VALIDATION_FIELDS = _BITEMPORAL_FIELDS | {"as_of"}
_ELIGIBILITY_FIELDS = {
    "data_input_eligible",
    "maximum_candidate_tier",
    "e1_claim_authorized",
    "legal_license_truth_verified",
    "basis",
    "required_downstream_artifact",
}


@dataclass(frozen=True, slots=True)
class HistoricalSourceImport:
    """Verified bytes plus a serializable, non-authoritative validation record."""

    manifest: dict[str, Any]
    manifest_sha256: str
    canonical_manifest_sha256: str
    manifest_bytes: bytes
    price_bytes: bytes
    validation: dict[str, Any]

    @property
    def e1_eligibility(self) -> dict[str, Any]:
        return dict(self.validation["e1_eligibility"])


def _error(code: str, detail: str) -> HistoricalDataError:
    return HistoricalDataError(f"{code}: {detail}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error(
            "HISTORICAL_MANIFEST_JSON", "manifest is not canonical JSON data"
        ) from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contains_secret(data: bytes) -> bool:
    if any(pattern.search(data) is not None for pattern in _SECRET_PATTERNS):
        return True
    canonical_keys = {
        re.sub(rb"[^a-z0-9]+", b"_", key.encode()).strip(b"_")
        for key in _SECRET_QUERY_KEYS
    }
    suffixes = tuple(item.encode() for item in _SECRET_QUERY_SUFFIXES)
    compact_suffixes = tuple(item.encode() for item in _SECRET_COMPACT_SUFFIXES)
    for match in _SECRET_ASSIGNMENT_RE.finditer(data):
        key = match.group("key").lower()
        normalized = re.sub(rb"[^a-z0-9]+", b"_", key).strip(b"_")
        compact = re.sub(rb"[^a-z0-9]+", b"", key)
        if (
            normalized in canonical_keys
            or normalized.endswith(suffixes)
            or compact.endswith(compact_suffixes)
        ):
            return True
    return False


def _require_mapping(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _error("HISTORICAL_MANIFEST_SCHEMA", f"{name} fields are invalid")
    return dict(value)


def _require_text(value: Any, name: str, *, limit: int = 2048) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise _error("HISTORICAL_MANIFEST_SCHEMA", f"{name} must be bounded text")
    return value


def _require_safe_id(value: Any, name: str) -> str:
    value = _require_text(value, name, limit=128)
    if not _SAFE_ID_RE.fullmatch(value):
        raise _error("HISTORICAL_MANIFEST_SCHEMA", f"{name} is not a safe identifier")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not _SHA256_RE.fullmatch(value)
        or value == "0" * 64
    ):
        raise _error("HISTORICAL_MANIFEST_SCHEMA", f"{name} must be a nonzero SHA-256")
    return value


def _source_uri(value: Any) -> str:
    uri = _require_text(value, "source_uri", limit=2048)
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError as exc:
        raise _error("HISTORICAL_SOURCE_URI", "source_uri is malformed") from exc
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.hostname
        or (port is not None and port != 443)
        or parsed.hostname.casefold().rstrip(".") in {"localhost", "127.0.0.1", "::1"}
    ):
        raise _error(
            "HISTORICAL_SOURCE_URI",
            "source_uri must be a credential-free HTTPS URI",
        )
    decoded_query = parsed.query
    for _ in range(3):
        next_value = unquote_plus(decoded_query)
        if next_value == decoded_query:
            break
        decoded_query = next_value
    canonical_secret_keys = {
        re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
        for key in _SECRET_QUERY_KEYS
    }
    for query in {parsed.query, decoded_query}:
        for key, _ in parse_qsl(query.replace(";", "&"), keep_blank_values=True):
            decoded_key = key
            for _ in range(3):
                next_key = unquote_plus(decoded_key)
                if next_key == decoded_key:
                    break
                decoded_key = next_key
            normalized = re.sub(r"[^a-z0-9]+", "_", decoded_key.casefold()).strip("_")
            compact = re.sub(r"[^a-z0-9]+", "", decoded_key.casefold())
            if (
                normalized in canonical_secret_keys
                or normalized.endswith(_SECRET_QUERY_SUFFIXES)
                or compact.endswith(_SECRET_COMPACT_SUFFIXES)
            ):
                raise _error(
                    "HISTORICAL_SOURCE_URI", "source_uri contains a credential field"
                )
    if re.search(
        r"(?:^|[^A-Za-z0-9_])(?:api[_-]?key|access[_-]?token|authorization|auth|bearer|"
        r"credential|password|passwd|secret|signature|token)\s*=",
        decoded_query,
        flags=re.IGNORECASE,
    ):
        raise _error("HISTORICAL_SOURCE_URI", "source_uri contains encoded credentials")
    return uri


def _utc_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise _error("HISTORICAL_TIME", f"{name} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _error("HISTORICAL_TIME", f"{name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise _error("HISTORICAL_TIME", f"{name} must be UTC")
    parsed = parsed.astimezone(timezone.utc)
    if _timestamp(parsed) != value:
        raise _error("HISTORICAL_TIME", f"{name} is not canonical")
    return parsed


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_utc_arg(value: datetime, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise _error("HISTORICAL_TIME", f"{name} must be a timezone-aware UTC datetime")
    return value.astimezone(timezone.utc)


def _relative_path(value: Any, name: str) -> str:
    value = _require_text(value, name, limit=1024)
    if "\\" in value:
        raise _error("HISTORICAL_PATH", f"{name} must be a normalized POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise _error("HISTORICAL_PATH", f"{name} escapes the workspace")
    return value


def _workspace_root(value: str | os.PathLike[str]) -> Path:
    original = Path(value)
    if has_link_or_reparse_component(original) or not original.is_dir():
        raise _error("HISTORICAL_PATH", "workspace must be a regular local directory")
    resolved = original.resolve()
    if has_link_or_reparse_component(resolved):
        raise _error("HISTORICAL_PATH", "workspace contains a link or reparse point")
    return resolved


def _target_inside(root: Path, relative: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative).parts)
    if has_link_or_reparse_component(target):
        raise _error("HISTORICAL_PATH", f"referenced path is linked: {relative}")
    try:
        resolved = target.resolve(strict=True)
        inside = os.path.commonpath((str(root), str(resolved))) == str(root)
    except (OSError, ValueError) as exc:
        raise _error(
            "HISTORICAL_PATH", f"referenced file is unavailable: {relative}"
        ) from exc
    if not inside or resolved != target.resolve():
        raise _error(
            "HISTORICAL_PATH", f"referenced path escapes the workspace: {relative}"
        )
    return resolved


def _read_regular(root: Path, relative: str, *, limit: int, label: str) -> bytes:
    target = _target_inside(root, relative)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise _error(
            "HISTORICAL_PATH", f"{label} must be a readable regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > limit
        ):
            raise _error(
                "HISTORICAL_SIZE", f"{label} is empty, non-regular, or oversized"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise _error("HISTORICAL_SIZE", f"{label} exceeds its size limit")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            getattr(before, "st_mtime_ns", None),
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            getattr(after, "st_mtime_ns", None),
        )
        if identity_before != identity_after or total != before.st_size:
            raise _error("HISTORICAL_CHANGED", f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_ref_projection(
    value: Any,
    *,
    label: str,
    limit: int = MAX_EVIDENCE_BYTES,
) -> dict[str, Any]:
    ref = _require_mapping(value, _FILE_REF_FIELDS, label)
    relative = _relative_path(ref["path"], f"{label}.path")
    digest = _require_sha256(ref["sha256"], f"{label}.sha256")
    size = ref["size_bytes"]
    if type(size) is not int or size <= 0 or size > limit:
        raise _error("HISTORICAL_SIZE", f"{label}.size_bytes is invalid")
    return {"path": relative, "sha256": digest, "size_bytes": size}


def _verify_file_ref(
    value: Any,
    *,
    root: Path,
    label: str,
    limit: int = MAX_EVIDENCE_BYTES,
) -> tuple[dict[str, Any], bytes]:
    ref = _file_ref_projection(value, label=label, limit=limit)
    data = _read_regular(root, ref["path"], limit=limit, label=label)
    if len(data) != ref["size_bytes"]:
        raise _error("HISTORICAL_SIZE", f"{label} size differs from the manifest")
    if _sha256(data) != ref["sha256"]:
        raise _error("HISTORICAL_HASH", f"{label} hash differs from the manifest")
    if _contains_secret(data):
        raise _error("HISTORICAL_SECRET", f"{label} contains secret-like material")
    return ref, data


def _parse_json(data: bytes, label: str) -> dict[str, Any]:
    if b"\x00" in data or _contains_secret(data):
        raise _error("HISTORICAL_SECRET", f"{label} contains forbidden material")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error("HISTORICAL_MANIFEST_JSON", f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                _error("HISTORICAL_MANIFEST_JSON", f"non-finite value {value}")
            ),
        )
    except HistoricalDataError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(
            "HISTORICAL_MANIFEST_JSON", f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise _error("HISTORICAL_MANIFEST_SCHEMA", f"{label} must be an object")
    return parsed


def _validate_price_csv(
    data: bytes,
    *,
    as_of: datetime,
) -> dict[str, Any]:
    if b"\x00" in data or _contains_secret(data):
        raise _error("HISTORICAL_SECRET", "price CSV contains forbidden material")
    try:
        text = data.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise _error(
            "HISTORICAL_CSV", "price CSV is not strict non-empty UTF-8"
        ) from exc
    if header != _CSV_COLUMNS:
        raise _error(
            "HISTORICAL_CSV", "price CSV header must exactly match qrae.price-bars.v1"
        )

    seen_events: set[tuple[str, str]] = set()
    last_event_by_instrument: dict[str, datetime] = {}
    row_count = 0
    min_event: datetime | None = None
    max_event: datetime | None = None
    min_available: datetime | None = None
    max_available: datetime | None = None
    try:
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(_CSV_COLUMNS):
                raise _error(
                    "HISTORICAL_CSV", f"row {line_number} has an invalid width"
                )
            event_text, available_text, instrument, price_text = row
            event_time = _utc_datetime(event_text, f"row {line_number} event_time")
            available_time = _utc_datetime(
                available_text, f"row {line_number} available_time"
            )
            instrument = _require_text(
                instrument, f"row {line_number} instrument", limit=256
            )
            if available_time > event_time:
                raise _error(
                    "HISTORICAL_PIT",
                    f"row {line_number} was unavailable at its decision timestamp",
                )
            if event_time > as_of:
                raise _error("HISTORICAL_PIT", f"row {line_number} is after as_of")
            try:
                price = Decimal(price_text)
            except InvalidOperation as exc:
                raise _error(
                    "HISTORICAL_CSV", f"row {line_number} price is invalid"
                ) from exc
            if not price.is_finite() or price <= 0:
                raise _error(
                    "HISTORICAL_CSV",
                    f"row {line_number} price must be finite and positive",
                )
            event_key = (instrument, event_text)
            if event_key in seen_events:
                raise _error(
                    "HISTORICAL_REVISION",
                    f"row {line_number} duplicates an instrument decision timestamp",
                )
            seen_events.add(event_key)
            previous_event = last_event_by_instrument.get(instrument)
            if previous_event is not None and event_time <= previous_event:
                raise _error(
                    "HISTORICAL_MONOTONICITY",
                    f"row {line_number} is not strictly increasing for {instrument}",
                )
            last_event_by_instrument[instrument] = event_time
            row_count += 1
            min_event = event_time if min_event is None else min(min_event, event_time)
            max_event = event_time if max_event is None else max(max_event, event_time)
            min_available = (
                available_time
                if min_available is None
                else min(min_available, available_time)
            )
            max_available = (
                available_time
                if max_available is None
                else max(max_available, available_time)
            )
    except csv.Error as exc:
        raise _error("HISTORICAL_CSV", "price CSV syntax is invalid") from exc
    if (
        row_count == 0
        or min_event is None
        or max_event is None
        or min_available is None
        or max_available is None
    ):
        raise _error("HISTORICAL_CSV", "price CSV contains no observations")
    return {
        "row_count": row_count,
        "instrument_count": len({item[0] for item in seen_events}),
        "min_event_time": _timestamp(min_event),
        "max_event_time": _timestamp(max_event),
        "min_available_time": _timestamp(min_available),
        "max_available_time": _timestamp(max_available),
    }


def _validate_historical_manifest(
    manifest: Mapping[str, Any],
    *,
    workspace: str | os.PathLike[str],
    as_of: datetime,
    now: datetime | None,
    manifest_sha256: str,
    manifest_bytes: bytes,
) -> HistoricalSourceImport:
    root = _workspace_root(workspace)
    cutoff = _require_utc_arg(as_of, "as_of")
    observed = _require_utc_arg(now or datetime.now(timezone.utc), "now")
    if cutoff > observed:
        raise _error("HISTORICAL_PIT", "as_of cannot be in the future")

    value = _require_mapping(manifest, _MANIFEST_FIELDS, "manifest")
    if value["schema_version"] != SCHEMA_VERSION:
        raise _error("HISTORICAL_MANIFEST_SCHEMA", "unsupported schema_version")
    dataset_id = _require_safe_id(value["dataset_id"], "dataset_id")
    family = _require_text(value["market_family"], "market_family", limit=64)
    if family not in MARKET_FAMILIES:
        raise _error("HISTORICAL_FAMILY_CONTROL", "market_family is not registered")
    provider = _require_text(value["provider"], "provider", limit=256)
    provider_dataset_id = _require_text(
        value["provider_dataset_id"], "provider_dataset_id", limit=256
    )
    source_uri = _source_uri(value["source_uri"])

    bitemporal = _require_mapping(value["bitemporal"], _BITEMPORAL_FIELDS, "bitemporal")
    if (
        bitemporal["event_time_field"] != "event_time"
        or bitemporal["available_time_field"] != "available_time"
    ):
        raise _error("HISTORICAL_PIT", "bitemporal fields must bind qrae.price-bars.v1")
    source_published = _utc_datetime(
        bitemporal["source_published_at"], "source_published_at"
    )
    acquired = _utc_datetime(bitemporal["acquired_at"], "acquired_at")
    ingested = _utc_datetime(bitemporal["ingested_at"], "ingested_at")
    if not source_published <= acquired <= ingested <= observed:
        raise _error(
            "HISTORICAL_PIT",
            "source publication/acquisition/ingestion order is invalid",
        )

    license_value = _require_mapping(value["license"], _LICENSE_FIELDS, "license")
    if license_value["use_scope"] != "LOCAL_RESEARCH_ONLY":
        raise _error("HISTORICAL_LICENSE", "only LOCAL_RESEARCH_ONLY scope is accepted")
    if license_value["evidence_type"] not in _EVIDENCE_TYPES:
        raise _error("HISTORICAL_LICENSE", "license evidence_type is unsupported")
    entitlement_hash = _require_sha256(
        license_value["entitlement_id_sha256"], "entitlement_id_sha256"
    )
    valid_from = _utc_datetime(license_value["valid_from"], "license.valid_from")
    expires_at = _utc_datetime(license_value["expires_at"], "license.expires_at")
    if valid_from >= expires_at or not valid_from <= ingested < expires_at:
        raise _error("HISTORICAL_LICENSE", "license interval does not cover ingestion")
    if observed < valid_from or observed >= expires_at:
        raise _error(
            "HISTORICAL_LICENSE_EXPIRED",
            "declared local-research entitlement is inactive",
        )
    license_ref, _ = _verify_file_ref(
        license_value["evidence"], root=root, label="license.evidence"
    )

    calendar = _require_mapping(value["calendar"], _CALENDAR_FIELDS, "calendar")
    calendar_id = _require_safe_id(calendar["calendar_id"], "calendar.calendar_id")
    timezone_name = _require_text(calendar["timezone"], "calendar.timezone", limit=128)
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise _error(
            "HISTORICAL_CALENDAR", "calendar.timezone is not an IANA timezone"
        ) from exc
    if calendar["session_policy"] not in _SESSION_POLICIES:
        raise _error("HISTORICAL_CALENDAR", "calendar.session_policy is unsupported")
    calendar_ref, _ = _verify_file_ref(
        calendar["evidence"], root=root, label="calendar.evidence"
    )

    revision = _require_mapping(
        value["revision_policy"], _REVISION_FIELDS, "revision_policy"
    )
    mode = revision["mode"]
    if mode not in _REVISION_MODES:
        raise _error("HISTORICAL_REVISION", "revision mode is unsupported")
    vintage_id = _require_safe_id(revision["vintage_id"], "revision_policy.vintage_id")
    supersedes = revision["supersedes_manifest_sha256"]
    if supersedes is not None:
        supersedes = _require_sha256(supersedes, "supersedes_manifest_sha256")
    if mode == "IMMUTABLE" and supersedes is not None:
        raise _error(
            "HISTORICAL_REVISION", "IMMUTABLE data cannot supersede a manifest"
        )
    revision_ref, _ = _verify_file_ref(
        revision["evidence"], root=root, label="revision_policy.evidence"
    )

    controls = value["family_controls"]
    required_controls = MARKET_FAMILY_CONTROLS[family]
    if not isinstance(controls, Mapping) or set(controls) != set(required_controls):
        raise _error(
            "HISTORICAL_FAMILY_CONTROL",
            f"{family} requires exactly {sorted(required_controls)!r}",
        )
    verified_controls: dict[str, dict[str, Any]] = {}
    for name in sorted(required_controls):
        verified_controls[name], _ = _verify_file_ref(
            controls[name], root=root, label=f"family_controls.{name}"
        )

    price = _require_mapping(value["price_data"], _PRICE_FIELDS, "price_data")
    if price["schema"] != PRICE_SCHEMA:
        raise _error("HISTORICAL_CSV", "price_data.schema must be qrae.price-bars.v1")
    price_ref, price_bytes = _verify_file_ref(
        {field: price[field] for field in _FILE_REF_FIELDS},
        root=root,
        label="price_data",
        limit=MAX_PRICE_BYTES,
    )
    price_quality = _validate_price_csv(price_bytes, as_of=cutoff)

    canonical_manifest = _canonical_json_bytes(value)
    if _contains_secret(canonical_manifest):
        raise _error("HISTORICAL_SECRET", "manifest contains secret-like material")
    canonical_digest = _sha256(canonical_manifest)
    source_digest = manifest_sha256
    _require_sha256(source_digest, "manifest_sha256")
    if (
        not isinstance(manifest_bytes, bytes)
        or _sha256(manifest_bytes) != source_digest
    ):
        raise _error("HISTORICAL_HASH", "manifest bytes differ from manifest_sha256")
    validation = {
        "schema_version": "qrae.historical-import-validation.v1",
        "dataset_id": dataset_id,
        "market_family": family,
        "provider": provider,
        "provider_dataset_id": provider_dataset_id,
        "source_uri": source_uri,
        "manifest_sha256": source_digest,
        "canonical_manifest_sha256": canonical_digest,
        "price_data": {**price_ref, "schema": PRICE_SCHEMA, **price_quality},
        "license_evidence": {
            **license_ref,
            "evidence_type": license_value["evidence_type"],
            "use_scope": "LOCAL_RESEARCH_ONLY",
            "entitlement_id_sha256": entitlement_hash,
            "valid_from": _timestamp(valid_from),
            "expires_at": _timestamp(expires_at),
            "integrity_scope": "DECLARED_SCOPE_HASH_AND_EXPIRY_ONLY",
        },
        "bitemporal": {
            "event_time_field": "event_time",
            "available_time_field": "available_time",
            "as_of": _timestamp(cutoff),
            "source_published_at": _timestamp(source_published),
            "acquired_at": _timestamp(acquired),
            "ingested_at": _timestamp(ingested),
        },
        "calendar": {
            "calendar_id": calendar_id,
            "timezone": timezone_name,
            "session_policy": calendar["session_policy"],
            "evidence": calendar_ref,
        },
        "revision_policy": {
            "mode": mode,
            "vintage_id": vintage_id,
            "supersedes_manifest_sha256": supersedes,
            "evidence": revision_ref,
            "lineage_status": (
                "ROOT_DECLARED" if supersedes is None else "UNVALIDATED_PARENT"
            ),
            "parent_snapshot_id": None,
        },
        "family_controls": verified_controls,
        "e1_eligibility": {
            "data_input_eligible": False,
            "maximum_candidate_tier": "E1",
            "e1_claim_authorized": False,
            "legal_license_truth_verified": False,
            "basis": "MECHANICAL_CHECKS_PASS_CATALOG_LINEAGE_UNVERIFIED",
            "required_downstream_artifact": "COMPATIBLE_TEMPORAL_VALIDATION",
        },
        "validated_as_of": _timestamp(cutoff),
        "validated_at": _timestamp(observed),
    }
    return HistoricalSourceImport(
        manifest=json.loads(json.dumps(value)),
        manifest_sha256=source_digest,
        canonical_manifest_sha256=canonical_digest,
        manifest_bytes=manifest_bytes,
        price_bytes=price_bytes,
        validation=validation,
    )


def validate_historical_manifest(
    manifest: Mapping[str, Any],
    *,
    workspace: str | os.PathLike[str],
    as_of: datetime,
    now: datetime | None = None,
) -> HistoricalSourceImport:
    """Validate one strict in-memory manifest and its exact referenced files.

    ``as_of`` is the knowledge cutoff applied to every CSV ``available_time``.
    Passing this gate makes the data a candidate E1 input only; a compatible
    downstream temporal-validation artifact remains mandatory.  In-memory
    manifests are identified by their canonical JSON hash; use
    :func:`import_historical_source` to bind the bytes of a manifest file.
    """

    canonical = _canonical_json_bytes(dict(manifest))
    return _validate_historical_manifest(
        manifest,
        workspace=workspace,
        as_of=as_of,
        now=now,
        manifest_sha256=_sha256(canonical),
        manifest_bytes=canonical,
    )


def import_historical_source(
    manifest_path: str | os.PathLike[str],
    *,
    workspace: str | os.PathLike[str],
    as_of: datetime,
    now: datetime | None = None,
) -> HistoricalSourceImport:
    """Strictly load and validate a local historical source manifest."""

    root = _workspace_root(workspace)
    supplied = Path(manifest_path)
    if supplied.is_absolute():
        try:
            relative = supplied.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise _error(
                "HISTORICAL_PATH", "manifest must remain inside workspace"
            ) from exc
    else:
        relative = _relative_path(os.fspath(manifest_path), "manifest_path")
    raw = _read_regular(root, relative, limit=MAX_MANIFEST_BYTES, label="manifest")
    manifest = _parse_json(raw, "manifest")
    return _validate_historical_manifest(
        manifest,
        workspace=root,
        as_of=as_of,
        now=now,
        manifest_sha256=_sha256(raw),
        manifest_bytes=raw,
    )


def _verify_manifest_validation_projection(
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    source: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    normalized_price_bytes: bytes,
    now: datetime | None,
) -> None:
    """Rebuild every manifest-derived validation field and compare once.

    Catalog lineage may change only the lineage pair and its mechanically
    corresponding eligibility verdict.  Everything else is a deterministic
    projection of the frozen manifest, manifest hashes, and validation times.
    """

    manifest_value = _require_mapping(manifest, _MANIFEST_FIELDS, "bundled manifest")
    validation_value = _require_mapping(
        validation, _IMPORT_VALIDATION_FIELDS, "validation record"
    )
    if manifest_value["schema_version"] != SCHEMA_VERSION:
        raise _error("HISTORICAL_BUNDLE", "bundled manifest schema is unsupported")

    dataset_id = _require_safe_id(manifest_value["dataset_id"], "dataset_id")
    family = _require_text(manifest_value["market_family"], "market_family", limit=64)
    if family not in MARKET_FAMILIES:
        raise _error("HISTORICAL_BUNDLE", "bundled market family is unsupported")
    provider = _require_text(manifest_value["provider"], "provider", limit=256)
    provider_dataset_id = _require_text(
        manifest_value["provider_dataset_id"], "provider_dataset_id", limit=256
    )
    source_uri = _source_uri(manifest_value["source_uri"])

    validated_as_of = _utc_datetime(bundle["validated_as_of"], "validated_as_of")
    validated_at = _utc_datetime(bundle["validated_at"], "validated_at")
    if validated_as_of > validated_at:
        raise _error("HISTORICAL_BUNDLE", "validation cutoff exceeds validation time")

    manifest_price = _require_mapping(
        manifest_value["price_data"], _PRICE_FIELDS, "bundled manifest price_data"
    )
    if manifest_price["schema"] != PRICE_SCHEMA:
        raise _error("HISTORICAL_BUNDLE", "bundled price schema is unsupported")
    expected_price_ref = _file_ref_projection(
        {field: manifest_price[field] for field in _FILE_REF_FIELDS},
        label="bundled manifest price_data",
        limit=MAX_PRICE_BYTES,
    )
    if (
        not isinstance(normalized_price_bytes, bytes)
        or len(normalized_price_bytes) != expected_price_ref["size_bytes"]
        or _sha256(normalized_price_bytes) != expected_price_ref["sha256"]
    ):
        raise _error(
            "HISTORICAL_HASH", "normalized price bytes differ from bundled manifest"
        )
    expected_price = {
        **expected_price_ref,
        "schema": PRICE_SCHEMA,
        **_validate_price_csv(normalized_price_bytes, as_of=validated_as_of),
    }
    validation_price = _require_mapping(
        validation_value["price_data"],
        _PRICE_VALIDATION_FIELDS,
        "price validation",
    )
    actual_price = dict(validation_price)
    row_count = validation_price["row_count"]
    instrument_count = validation_price["instrument_count"]
    if (
        type(row_count) is not int
        or row_count <= 0
        or type(instrument_count) is not int
        or instrument_count <= 0
        or instrument_count > row_count
    ):
        raise _error("HISTORICAL_BUNDLE", "price validation counts are invalid")
    min_event = _utc_datetime(validation_price["min_event_time"], "min_event_time")
    max_event = _utc_datetime(validation_price["max_event_time"], "max_event_time")
    min_available = _utc_datetime(
        validation_price["min_available_time"], "min_available_time"
    )
    max_available = _utc_datetime(
        validation_price["max_available_time"], "max_available_time"
    )
    if (
        min_event > max_event
        or max_event > validated_as_of
        or min_available > max_available
        or min_available > min_event
        or max_available > max_event
    ):
        raise _error("HISTORICAL_BUNDLE", "price validation times are invalid")

    manifest_license = _require_mapping(
        manifest_value["license"], _LICENSE_FIELDS, "bundled manifest license"
    )
    if manifest_license["use_scope"] != "LOCAL_RESEARCH_ONLY":
        raise _error("HISTORICAL_BUNDLE", "bundled license scope is unsupported")
    if manifest_license["evidence_type"] not in _EVIDENCE_TYPES:
        raise _error(
            "HISTORICAL_BUNDLE", "bundled license evidence type is unsupported"
        )
    valid_from = _utc_datetime(manifest_license["valid_from"], "license.valid_from")
    expires_at = _utc_datetime(manifest_license["expires_at"], "license.expires_at")
    entitlement_hash = _require_sha256(
        manifest_license["entitlement_id_sha256"], "entitlement_id_sha256"
    )
    expected_license = {
        **_file_ref_projection(
            manifest_license["evidence"], label="bundled manifest license.evidence"
        ),
        "evidence_type": manifest_license["evidence_type"],
        "use_scope": "LOCAL_RESEARCH_ONLY",
        "entitlement_id_sha256": entitlement_hash,
        "valid_from": _timestamp(valid_from),
        "expires_at": _timestamp(expires_at),
        "integrity_scope": "DECLARED_SCOPE_HASH_AND_EXPIRY_ONLY",
    }
    actual_license = _require_mapping(
        validation_value["license_evidence"],
        _LICENSE_VALIDATION_FIELDS,
        "license validation",
    )

    manifest_bitemporal = _require_mapping(
        manifest_value["bitemporal"],
        _BITEMPORAL_FIELDS,
        "bundled manifest bitemporal",
    )
    if (
        manifest_bitemporal["event_time_field"] != "event_time"
        or manifest_bitemporal["available_time_field"] != "available_time"
    ):
        raise _error("HISTORICAL_BUNDLE", "bundled bitemporal fields are unsupported")
    source_published = _utc_datetime(
        manifest_bitemporal["source_published_at"], "source_published_at"
    )
    acquired = _utc_datetime(manifest_bitemporal["acquired_at"], "acquired_at")
    ingested = _utc_datetime(manifest_bitemporal["ingested_at"], "ingested_at")
    if (
        not source_published <= acquired <= ingested <= validated_at
        or valid_from >= expires_at
        or not valid_from <= ingested < expires_at
        or not valid_from <= validated_at < expires_at
    ):
        raise _error(
            "HISTORICAL_BUNDLE", "bundled source or entitlement time is invalid"
        )
    expected_bitemporal = {
        "event_time_field": "event_time",
        "available_time_field": "available_time",
        "as_of": _timestamp(validated_as_of),
        "source_published_at": _timestamp(source_published),
        "acquired_at": _timestamp(acquired),
        "ingested_at": _timestamp(ingested),
    }
    actual_bitemporal = _require_mapping(
        validation_value["bitemporal"],
        _BITEMPORAL_VALIDATION_FIELDS,
        "bitemporal validation",
    )

    manifest_calendar = _require_mapping(
        manifest_value["calendar"], _CALENDAR_FIELDS, "bundled manifest calendar"
    )
    calendar_id = _require_safe_id(
        manifest_calendar["calendar_id"], "calendar.calendar_id"
    )
    timezone_name = _require_text(
        manifest_calendar["timezone"], "calendar.timezone", limit=128
    )
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise _error(
            "HISTORICAL_BUNDLE", "bundled calendar timezone is invalid"
        ) from exc
    if manifest_calendar["session_policy"] not in _SESSION_POLICIES:
        raise _error("HISTORICAL_BUNDLE", "bundled session policy is unsupported")
    expected_calendar = {
        "calendar_id": calendar_id,
        "timezone": timezone_name,
        "session_policy": manifest_calendar["session_policy"],
        "evidence": _file_ref_projection(
            manifest_calendar["evidence"], label="bundled manifest calendar.evidence"
        ),
    }
    actual_calendar = _require_mapping(
        validation_value["calendar"], _CALENDAR_FIELDS, "calendar validation"
    )

    manifest_revision = _require_mapping(
        manifest_value["revision_policy"],
        _REVISION_FIELDS,
        "bundled manifest revision policy",
    )
    mode = manifest_revision["mode"]
    if mode not in _REVISION_MODES:
        raise _error("HISTORICAL_BUNDLE", "bundled revision mode is unsupported")
    vintage_id = _require_safe_id(
        manifest_revision["vintage_id"], "revision_policy.vintage_id"
    )
    supersedes = manifest_revision["supersedes_manifest_sha256"]
    if supersedes is not None:
        supersedes = _require_sha256(supersedes, "supersedes_manifest_sha256")
    if mode == "IMMUTABLE" and supersedes is not None:
        raise _error("HISTORICAL_BUNDLE", "immutable data cannot supersede a manifest")
    actual_revision = _require_mapping(
        validation_value["revision_policy"],
        _REVISION_VALIDATION_FIELDS,
        "revision validation",
    )
    lineage_status = actual_revision["lineage_status"]
    parent_snapshot_id = actual_revision["parent_snapshot_id"]
    unverified_basis = "MECHANICAL_CHECKS_PASS_CATALOG_LINEAGE_UNVERIFIED"
    verified_basis = "MECHANICAL_CHECKS_AND_CATALOG_LINEAGE_VALIDATED"
    if lineage_status == "ROOT_DECLARED":
        lineage_valid = supersedes is None and parent_snapshot_id is None
        eligible = False
        eligibility_basis = unverified_basis
    elif lineage_status == "CATALOG_ROOT_VALIDATED":
        lineage_valid = supersedes is None and parent_snapshot_id is None
        eligible = True
        eligibility_basis = verified_basis
    elif lineage_status == "UNVALIDATED_PARENT":
        lineage_valid = supersedes is not None and parent_snapshot_id is None
        eligible = False
        eligibility_basis = unverified_basis
    elif lineage_status == "VALIDATED_AGAINST_LATEST":
        lineage_valid = supersedes is not None
        parent_snapshot_id = _require_sha256(
            parent_snapshot_id, "revision_policy.parent_snapshot_id"
        )
        eligible = True
        eligibility_basis = verified_basis
    else:
        lineage_valid = False
        eligible = False
        eligibility_basis = unverified_basis
    if not lineage_valid:
        raise _error("HISTORICAL_BUNDLE", "revision lineage state is invalid")
    expected_revision = {
        "mode": mode,
        "vintage_id": vintage_id,
        "supersedes_manifest_sha256": supersedes,
        "evidence": _file_ref_projection(
            manifest_revision["evidence"],
            label="bundled manifest revision_policy.evidence",
        ),
        "lineage_status": lineage_status,
        "parent_snapshot_id": parent_snapshot_id,
    }
    expected_eligibility = {
        "data_input_eligible": eligible,
        "maximum_candidate_tier": "E1",
        "e1_claim_authorized": False,
        "legal_license_truth_verified": False,
        "basis": eligibility_basis,
        "required_downstream_artifact": "COMPATIBLE_TEMPORAL_VALIDATION",
    }
    actual_eligibility = _require_mapping(
        validation_value["e1_eligibility"],
        _ELIGIBILITY_FIELDS,
        "E1 eligibility",
    )

    manifest_controls = manifest_value["family_controls"]
    required_controls = MARKET_FAMILY_CONTROLS[family]
    if not isinstance(manifest_controls, Mapping) or set(manifest_controls) != set(
        required_controls
    ):
        raise _error("HISTORICAL_BUNDLE", "bundled family controls are invalid")
    expected_controls = {
        name: _file_ref_projection(
            manifest_controls[name], label=f"bundled manifest family_controls.{name}"
        )
        for name in sorted(required_controls)
    }
    validation_controls = validation_value["family_controls"]
    if not isinstance(validation_controls, Mapping) or set(validation_controls) != set(
        required_controls
    ):
        raise _error("HISTORICAL_BUNDLE", "validation family controls are invalid")
    actual_controls = {
        name: _file_ref_projection(
            validation_controls[name], label=f"validation family_controls.{name}"
        )
        for name in sorted(required_controls)
    }

    expected_projection = {
        "schema_version": "qrae.historical-import-validation.v1",
        "dataset_id": dataset_id,
        "market_family": family,
        "provider": provider,
        "provider_dataset_id": provider_dataset_id,
        "source_uri": source_uri,
        "manifest_sha256": source["sha256"],
        "canonical_manifest_sha256": source["canonical_sha256"],
        "price_data": expected_price,
        "license_evidence": expected_license,
        "bitemporal": expected_bitemporal,
        "calendar": expected_calendar,
        "revision_policy": expected_revision,
        "family_controls": expected_controls,
        "e1_eligibility": expected_eligibility,
        "validated_as_of": _timestamp(validated_as_of),
        "validated_at": _timestamp(validated_at),
    }
    actual_projection = {
        "schema_version": validation_value["schema_version"],
        "dataset_id": validation_value["dataset_id"],
        "market_family": validation_value["market_family"],
        "provider": validation_value["provider"],
        "provider_dataset_id": validation_value["provider_dataset_id"],
        "source_uri": validation_value["source_uri"],
        "manifest_sha256": validation_value["manifest_sha256"],
        "canonical_manifest_sha256": validation_value["canonical_manifest_sha256"],
        "price_data": actual_price,
        "license_evidence": dict(actual_license),
        "bitemporal": dict(actual_bitemporal),
        "calendar": dict(actual_calendar),
        "revision_policy": dict(actual_revision),
        "family_controls": actual_controls,
        "e1_eligibility": dict(actual_eligibility),
        "validated_as_of": validation_value["validated_as_of"],
        "validated_at": validation_value["validated_at"],
    }
    if actual_projection != expected_projection:
        raise _error(
            "HISTORICAL_BUNDLE",
            "manifest-derived validation semantic projection differs",
        )
    if now is not None:
        observed = _require_utc_arg(now, "now")
        if observed < valid_from or observed >= expires_at:
            raise _error(
                "HISTORICAL_LICENSE_EXPIRED",
                "frozen local-research entitlement is inactive",
            )


def verify_historical_import_bundle(
    data: bytes,
    *,
    normalized_price_bytes: bytes,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Offline-verify a frozen import bundle with its normalized-price sidecar.

    Verification needs no source workspace or live catalog, but it requires
    the paired normalized bytes so hashes and all derived price-quality fields
    can be recomputed. Passing ``now`` also enforces the declared entitlement
    interval; omitting it permits later archival verification after expiry.
    """

    if not isinstance(data, bytes) or not data or len(data) > MAX_IMPORT_BUNDLE_BYTES:
        raise _error("HISTORICAL_SIZE", "import bundle is empty or oversized")
    bundle = _parse_json(data, "historical import bundle")
    if _canonical_json_bytes(bundle) != data:
        raise _error("HISTORICAL_BUNDLE", "import bundle is not canonical JSON")
    bundle = _require_mapping(bundle, _IMPORT_BUNDLE_FIELDS, "import bundle")
    if (
        bundle["schema_version"] != "qrae.historical-import-bundle.v1"
        or bundle["kind"] != "LOCAL_MANIFEST_IMPORT"
    ):
        raise _error("HISTORICAL_BUNDLE", "import bundle schema or kind is invalid")

    source = _require_mapping(
        bundle["source_manifest"],
        _SOURCE_MANIFEST_BUNDLE_FIELDS,
        "source_manifest",
    )
    content = source["content_utf8"]
    if not isinstance(content, str):
        raise _error("HISTORICAL_BUNDLE", "source manifest content must be UTF-8 text")
    manifest_bytes = content.encode("utf-8")
    if (
        type(source["size_bytes"]) is not int
        or source["size_bytes"] <= 0
        or source["size_bytes"] > MAX_MANIFEST_BYTES
        or len(manifest_bytes) != source["size_bytes"]
        or _sha256(manifest_bytes)
        != _require_sha256(source["sha256"], "source_manifest.sha256")
    ):
        raise _error("HISTORICAL_HASH", "source manifest bytes differ from bundle")
    manifest = _parse_json(manifest_bytes, "bundled source manifest")
    canonical_manifest_sha256 = _sha256(_canonical_json_bytes(manifest))
    if canonical_manifest_sha256 != _require_sha256(
        source["canonical_sha256"], "source_manifest.canonical_sha256"
    ):
        raise _error("HISTORICAL_HASH", "canonical manifest hash differs")

    validation = _require_mapping(
        bundle["validation"], _IMPORT_VALIDATION_FIELDS, "validation record"
    )
    validation_sha256 = _require_sha256(
        bundle["validation_sha256"], "validation_sha256"
    )
    if _sha256(_canonical_json_bytes(validation)) != validation_sha256:
        raise _error("HISTORICAL_HASH", "validation record hash differs")
    _verify_manifest_validation_projection(
        manifest,
        validation,
        source,
        bundle,
        normalized_price_bytes=normalized_price_bytes,
        now=now,
    )
    return bundle


def _catalog_adapter_id(validation: Mapping[str, Any]) -> str:
    identity = {
        "dataset_id": validation["dataset_id"],
        "market_family": validation["market_family"],
        "provider": validation["provider"],
        "provider_dataset_id": validation["provider_dataset_id"],
    }
    return f"historical-local-{_sha256(_canonical_json_bytes(identity))}"


def _catalog_lineage_validation(
    catalog: DataCatalog,
    imported: HistoricalSourceImport,
    *,
    adapter_id: str,
    as_of: datetime,
    observed: datetime,
) -> tuple[dict[str, Any], str | None]:
    validation = json.loads(json.dumps(imported.validation))
    revision = validation["revision_policy"]
    supersedes = revision["supersedes_manifest_sha256"]
    absolute_latest_id = catalog.latest_snapshot_id(adapter_id)
    try:
        previous = catalog.resolve_latest(adapter_id, as_of=observed)
    except CatalogError as exc:
        if exc.code != "PIT_NOT_AVAILABLE":
            raise _error(
                "HISTORICAL_REVISION_LINEAGE", "catalog lineage lookup failed"
            ) from exc
        if absolute_latest_id is not None:
            raise _error(
                "HISTORICAL_REVISION_LINEAGE",
                "revision time regresses behind an existing snapshot",
            ) from exc
        previous = None

    if (
        previous is not None
        and absolute_latest_id is not None
        and previous["snapshot_id"] != absolute_latest_id
    ):
        raise _error(
            "HISTORICAL_REVISION_LINEAGE",
            "revision does not extend the absolute latest snapshot",
        )

    if previous is None:
        if supersedes is not None:
            raise _error(
                "HISTORICAL_REVISION_LINEAGE",
                "superseded manifest is absent from this dataset lineage",
            )
        revision["lineage_status"] = "CATALOG_ROOT_VALIDATED"
        revision["parent_snapshot_id"] = None
    else:
        raw = _read_regular(
            catalog.root,
            previous["raw_object_path"],
            limit=MAX_IMPORT_BUNDLE_BYTES,
            label="prior historical import bundle",
        )
        parent_prices = _read_regular(
            catalog.root,
            previous["normalized_object_path"],
            limit=MAX_PRICE_BYTES,
            label="prior historical normalized prices",
        )
        parent_bundle = verify_historical_import_bundle(
            raw, normalized_price_bytes=parent_prices
        )
        parent = parent_bundle["validation"]
        parent_revision = parent["revision_policy"]
        identity_fields = (
            "dataset_id",
            "market_family",
            "provider",
            "provider_dataset_id",
        )
        if any(parent.get(field) != validation.get(field) for field in identity_fields):
            raise _error(
                "HISTORICAL_REVISION_LINEAGE", "revision changes dataset identity"
            )
        if supersedes != parent.get(
            "canonical_manifest_sha256"
        ) or imported.canonical_manifest_sha256 == parent.get(
            "canonical_manifest_sha256"
        ):
            raise _error(
                "HISTORICAL_REVISION_LINEAGE",
                "revision does not extend the latest manifest",
            )
        if (
            parent_revision.get("lineage_status") == "UNVALIDATED_PARENT"
            or parent_revision.get("mode") == "IMMUTABLE"
            or revision["mode"] != parent_revision.get("mode")
            or revision["vintage_id"] == parent_revision.get("vintage_id")
        ):
            raise _error(
                "HISTORICAL_REVISION_LINEAGE", "revision policy continuity is invalid"
            )
        parent_bitemporal = parent["bitemporal"]
        current_bitemporal = validation["bitemporal"]
        ordered_fields = ("source_published_at", "acquired_at", "ingested_at")
        if any(
            _utc_datetime(current_bitemporal[field], field)
            < _utc_datetime(parent_bitemporal[field], field)
            for field in ordered_fields
        ) or _utc_datetime(
            current_bitemporal["ingested_at"], "ingested_at"
        ) <= _utc_datetime(parent_bitemporal["ingested_at"], "parent.ingested_at"):
            raise _error("HISTORICAL_REVISION_LINEAGE", "revision timestamps regress")
        if as_of < _utc_datetime(
            parent_bundle["validated_as_of"], "parent.validated_as_of"
        ) or observed <= _utc_datetime(
            parent_bundle["validated_at"], "parent.validated_at"
        ):
            raise _error(
                "HISTORICAL_REVISION_LINEAGE", "revision validation time regresses"
            )
        revision["lineage_status"] = "VALIDATED_AGAINST_LATEST"
        revision["parent_snapshot_id"] = previous["snapshot_id"]

    validation["e1_eligibility"]["data_input_eligible"] = True
    validation["e1_eligibility"]["basis"] = (
        "MECHANICAL_CHECKS_AND_CATALOG_LINEAGE_VALIDATED"
    )
    return validation, previous["snapshot_id"] if previous is not None else None


def _import_bundle_bytes(
    imported: HistoricalSourceImport,
    validation: Mapping[str, Any],
) -> bytes:
    try:
        manifest_text = imported.manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _error("HISTORICAL_MANIFEST_JSON", "manifest is not UTF-8") from exc
    bundle: dict[str, Any] = {
        "schema_version": "qrae.historical-import-bundle.v1",
        "kind": "LOCAL_MANIFEST_IMPORT",
        "source_manifest": {
            "sha256": imported.manifest_sha256,
            "size_bytes": len(imported.manifest_bytes),
            "canonical_sha256": imported.canonical_manifest_sha256,
            "content_utf8": manifest_text,
        },
        "validation": dict(validation),
        "validation_sha256": _sha256(_canonical_json_bytes(validation)),
        "validated_as_of": validation["validated_as_of"],
        "validated_at": validation["validated_at"],
    }
    data = _canonical_json_bytes(bundle)
    verify_historical_import_bundle(data, normalized_price_bytes=imported.price_bytes)
    return data


def _catalog_result(
    validation: Mapping[str, Any],
    bundle: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **json.loads(json.dumps(validation)),
        "catalog": {
            "adapter_id": snapshot["adapter_id"],
            "adapter_contract_sha256": snapshot["adapter_contract_sha256"],
            "snapshot_id": snapshot["snapshot_id"],
            "raw_import_bundle_sha256": snapshot["raw_sha256"],
            "raw_object_path": snapshot["raw_object_path"],
            "source_manifest_sha256": bundle["source_manifest"]["sha256"],
            "canonical_manifest_sha256": bundle["source_manifest"]["canonical_sha256"],
            "validation_sha256": bundle["validation_sha256"],
            "validated_as_of": bundle["validated_as_of"],
            "validated_at": bundle["validated_at"],
            "normalized_price_sha256": snapshot["normalized_sha256"],
            "normalized_price_size_bytes": snapshot["normalized_size_bytes"],
            "normalized_object_path": snapshot["normalized_object_path"],
            "availability_basis": snapshot["availability_basis"],
            "evidence_ceiling": snapshot["evidence_ceiling"],
        },
    }


def _verified_catalog_replay(
    catalog: DataCatalog,
    imported: HistoricalSourceImport,
    snapshot: Mapping[str, Any],
    *,
    observed: datetime,
    concurrent_winner: bool = False,
) -> dict[str, Any]:
    available_at = _utc_datetime(snapshot["available_at"], "snapshot.available_at")
    if available_at > observed and not concurrent_winner:
        raise _error(
            "HISTORICAL_REVISION_LINEAGE",
            "idempotent replay time precedes the existing snapshot",
        )
    raw = _read_regular(
        catalog.root,
        snapshot["raw_object_path"],
        limit=MAX_IMPORT_BUNDLE_BYTES,
        label="existing historical import bundle",
    )
    normalized = _read_regular(
        catalog.root,
        snapshot["normalized_object_path"],
        limit=MAX_PRICE_BYTES,
        label="existing historical normalized prices",
    )
    bundle = verify_historical_import_bundle(
        raw,
        normalized_price_bytes=normalized,
        now=max(observed, available_at) if concurrent_winner else observed,
    )
    source = bundle["source_manifest"]
    try:
        bundled_manifest = source["content_utf8"].encode("utf-8")
    except (AttributeError, KeyError, UnicodeError) as exc:
        raise _error(
            "HISTORICAL_IDEMPOTENCY", "existing source manifest is invalid"
        ) from exc
    if (
        source["sha256"] != imported.manifest_sha256
        or source["canonical_sha256"] != imported.canonical_manifest_sha256
        or bundled_manifest != imported.manifest_bytes
        or bundle["validated_as_of"] != imported.validation["validated_as_of"]
        or snapshot["source_uri"] != imported.validation["source_uri"]
        or snapshot["normalized_sha256"] != _sha256(imported.price_bytes)
        or snapshot["normalized_size_bytes"] != len(imported.price_bytes)
        or snapshot["evidence_ceiling"] != "E0"
    ):
        raise _error(
            "HISTORICAL_IDEMPOTENCY",
            "request_id is already bound to different historical content",
        )
    return _catalog_result(bundle["validation"], bundle, snapshot)


def catalog_historical_source(
    catalog: DataCatalog,
    manifest_path: str | os.PathLike[str],
    *,
    workspace: str | os.PathLike[str],
    as_of: datetime,
    request_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify and register one historical source in the current E0 catalog.

    A canonical local-import bundle containing the exact manifest text and
    frozen validation is the raw object; the verified price CSV is normalized.
    Registration intentionally remains E0: this helper cannot establish legal
    entitlement truth or satisfy the downstream temporal-validation gate.
    """

    if not isinstance(catalog, DataCatalog):
        raise _error("HISTORICAL_ARGUMENT", "catalog must be a DataCatalog")
    observed = _require_utc_arg(now or datetime.now(timezone.utc), "now")
    imported = import_historical_source(
        manifest_path,
        workspace=workspace,
        as_of=as_of,
        now=observed,
    )
    adapter_id = _catalog_adapter_id(imported.validation)
    existing = catalog.resolve_request(adapter_id, request_id)
    if existing is not None:
        return _verified_catalog_replay(catalog, imported, existing, observed=observed)

    try:
        validation, expected_latest_snapshot_id = _catalog_lineage_validation(
            catalog,
            imported,
            adapter_id=adapter_id,
            as_of=_require_utc_arg(as_of, "as_of"),
            observed=observed,
        )
    except HistoricalDataError:
        # Another importer may have committed this exact request after the
        # initial lookup but before lineage validation.  Only a fully verified
        # request/content replay may recover; genuine forks still fail closed.
        existing = catalog.resolve_request(adapter_id, request_id)
        if existing is None:
            raise
        return _verified_catalog_replay(
            catalog,
            imported,
            existing,
            observed=observed,
            concurrent_winner=True,
        )
    source_uri = validation["source_uri"]
    parsed = urlsplit(source_uri)
    hostname = parsed.hostname.casefold().rstrip(".")
    origin = f"https://{hostname}"
    bundle_bytes = _import_bundle_bytes(imported, validation)
    bundle = verify_historical_import_bundle(
        bundle_bytes,
        normalized_price_bytes=imported.price_bytes,
        now=observed,
    )
    contract = {
        "schema_version": "1.0",
        "adapter_id": adapter_id,
        "adapter_version": "1.0.0",
        "market_family": validation["market_family"],
        "source_kind": "LOCAL_MANIFEST_IMPORT",
        "base_url": origin,
        "allowed_hosts": [hostname],
        "read_only": True,
        "auth_mode": "manifest-declared",
        "data_tier": "TIER_0",
        "normalized_schema": PRICE_SCHEMA,
        "license_status": (
            "DECLARED_LOCAL_RESEARCH_ONLY;LEGAL_TRUTH_UNVERIFIED;"
            "VALIDATION_FROZEN_PER_SNAPSHOT"
        ),
        "evidence_ceiling": "E0",
    }
    adapter = catalog.register_adapter(contract, registered_at=observed)
    catalog_observed = max(
        observed,
        _utc_datetime(adapter["registered_at"], "adapter.registered_at"),
    )
    bundle = verify_historical_import_bundle(
        bundle_bytes,
        normalized_price_bytes=imported.price_bytes,
        now=catalog_observed,
    )
    price_record = imported.validation["price_data"]
    try:
        snapshot = catalog.register_snapshot(
            request_id=request_id,
            adapter_id=adapter_id,
            source_uri=source_uri,
            requested_at=catalog_observed,
            retrieved_at=catalog_observed,
            available_at=catalog_observed,
            cutoff=catalog_observed,
            availability_basis="OBSERVED_AT_IMPORT",
            raw_bytes=bundle_bytes,
            normalized_bytes=imported.price_bytes,
            normalized_schema=PRICE_SCHEMA,
            item_count=price_record["row_count"],
            max_source_event_time=_utc_datetime(
                price_record["max_event_time"], "price_data.max_event_time"
            ),
            source_http_date=None,
            response_headers={},
            expected_latest_snapshot_id=expected_latest_snapshot_id,
        )
    except CatalogError as exc:
        if exc.code in {"CATALOG_IDEMPOTENCY_CONFLICT", "CATALOG_LINEAGE_CONFLICT"}:
            existing = catalog.resolve_request(adapter_id, request_id)
            if existing is not None:
                return _verified_catalog_replay(
                    catalog,
                    imported,
                    existing,
                    observed=catalog_observed,
                    concurrent_winner=True,
                )
        if exc.code == "CATALOG_LINEAGE_CONFLICT":
            raise _error(
                "HISTORICAL_REVISION_LINEAGE",
                "catalog latest changed during revision import",
            ) from exc
        raise
    return _catalog_result(validation, bundle, snapshot)


__all__ = [
    "MARKET_FAMILIES",
    "MARKET_FAMILY_CONTROLS",
    "MAX_EVIDENCE_BYTES",
    "MAX_IMPORT_BUNDLE_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_PRICE_BYTES",
    "PRICE_SCHEMA",
    "SCHEMA_VERSION",
    "HistoricalDataError",
    "HistoricalSourceImport",
    "catalog_historical_source",
    "import_historical_source",
    "validate_historical_manifest",
    "verify_historical_import_bundle",
]
