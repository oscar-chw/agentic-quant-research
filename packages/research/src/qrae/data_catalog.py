"""Durable, point-in-time data catalog for research-only market snapshots."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlsplit

from .artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from .path_safety import has_link_or_reparse_component, is_link_or_reparse


class CatalogError(ValueError):
    """A catalog operation violated an integrity or safety contract."""

    def __init__(self, message: str) -> None:
        self.code = message.partition(":")[0] if ":" in message else "CATALOG_ERROR"
        super().__init__(message)


_SCHEMA_VERSION = "1.0"
_SYNTHETIC_ADAPTER_SCHEMA = "1.1"
_SYNTHETIC_PROVENANCE_SCHEMA = "1.1"
_SYNTHETIC_URI_PREFIX = "urn:qrae:synthetic:"
_SYNTHETIC_LICENSE = "SYNTHETIC_FIXTURE_ONLY;NO_EXTERNAL_ORIGIN_CLAIM"
_MAX_SYNTHETIC_MANIFEST_BYTES = 256 * 1024
_ADAPTER_HASH_SCOPE = "contract-and-registered-at-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ADAPTER_FIELDS = {
    "schema_version",
    "adapter_id",
    "adapter_version",
    "market_family",
    "source_kind",
    "base_url",
    "allowed_hosts",
    "read_only",
    "auth_mode",
    "data_tier",
    "normalized_schema",
    "license_status",
    "evidence_ceiling",
}
_ADAPTER_RECORD_FIELDS = _ADAPTER_FIELDS | {"contract_sha256", "registered_at"}
_SNAPSHOT_RECORD_FIELDS = {
    "schema_version",
    "snapshot_id",
    "request_fingerprint",
    "adapter_id",
    "request_id",
    "source_uri",
    "requested_at",
    "retrieved_at",
    "available_at",
    "cutoff",
    "availability_basis",
    "raw_sha256",
    "raw_size_bytes",
    "raw_object_path",
    "normalized_sha256",
    "normalized_size_bytes",
    "normalized_object_path",
    "normalized_schema",
    "item_count",
    "max_source_event_time",
    "source_http_date",
    "response_headers",
    "adapter_contract_sha256",
    "evidence_ceiling",
}
_SAFE_RESPONSE_HEADERS = {
    "cache-control",
    "content-length",
    "content-type",
    "date",
    "etag",
    "expires",
    "last-modified",
    "vary",
}
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
_EVIDENCE_ORDER = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}

_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_LATEST_UNSET = object()


def _root_lock(root: Path) -> threading.RLock:
    key = str(root)
    with _LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise CatalogError(f"CATALOG_INVALID_FIELD: {field} must be a non-empty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise CatalogError(
            f"CATALOG_INVALID_FIELD: {field} contains control characters"
        )
    return value


def _require_safe_id(value: Any, field: str) -> str:
    value = _require_text(value, field)
    if not _SAFE_ID_RE.fullmatch(value):
        raise CatalogError(f"CATALOG_INVALID_FIELD: {field} is not a safe identifier")
    return value


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CatalogError(f"CATALOG_INVALID_TIME: {field} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise CatalogError(f"CATALOG_INVALID_TIME: {field} must use UTC")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _epoch_microseconds(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000 + value.microsecond


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise CatalogError(
            f"CATALOG_INVALID_TIME: persisted {field} is invalid"
        ) from exc
    return _utc(parsed, field)


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise CatalogError(
            "CATALOG_INVALID_JSON: value is not finite canonical JSON"
        ) from exc


def _validate_source_uri(
    value: Any, allowed_hosts: list[str], source_kind: str | None = None
) -> str:
    uri = _require_text(value, "source_uri")
    if source_kind == "LOCAL_SYNTHETIC":
        if allowed_hosts != [] or not re.fullmatch(
            r"urn:qrae:synthetic:[0-9a-f]{64}", uri
        ):
            raise CatalogError("CATALOG_UNSAFE_URI: invalid synthetic content identity")
        return uri
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError as exc:
        raise CatalogError("CATALOG_UNSAFE_URI: source_uri is malformed") from exc
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.hostname
        or (port is not None and port != 443)
        or parsed.hostname.casefold().rstrip(".") not in allowed_hosts
    ):
        raise CatalogError(
            "CATALOG_UNSAFE_URI: source_uri violates the adapter contract"
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
    query_forms = {parsed.query, decoded_query}
    for query in query_forms:
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
                raise CatalogError(
                    "CATALOG_SECRET_URI: source_uri contains a credential field"
                )
    if re.search(
        r"(?:^|[^A-Za-z0-9_])(?:api[_-]?key|access[_-]?token|authorization|auth|bearer|"
        r"credential|password|passwd|secret|signature|token)\s*=",
        decoded_query,
        flags=re.IGNORECASE,
    ):
        raise CatalogError(
            "CATALOG_SECRET_URI: source_uri contains encoded credentials"
        )
    return uri


def _normalize_headers(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) > 64:
        raise CatalogError(
            "CATALOG_INVALID_HEADERS: response_headers must be a bounded object"
        )
    headers: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise CatalogError(
                "CATALOG_INVALID_HEADERS: header names and values must be strings"
            )
        name = raw_name.strip().casefold()
        if name not in _SAFE_RESPONSE_HEADERS:
            continue
        if (
            not name
            or len(raw_value) > 4096
            or any(char in raw_value for char in ("\r", "\n", "\x00"))
        ):
            raise CatalogError("CATALOG_INVALID_HEADERS: response header is unsafe")
        if name in headers and headers[name] != raw_value.strip():
            raise CatalogError(
                "CATALOG_INVALID_HEADERS: conflicting duplicate response header"
            )
        headers[name] = raw_value.strip()
    return dict(sorted(headers.items()))


def _normalize_adapter(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or set(contract) != _ADAPTER_FIELDS:
        raise CatalogError(
            "CATALOG_ADAPTER_SCHEMA: adapter contract fields are invalid"
        )
    result = dict(contract)
    for field in _ADAPTER_FIELDS - {"allowed_hosts", "read_only"}:
        result[field] = _require_text(result[field], field)
    result["adapter_id"] = _require_safe_id(result["adapter_id"], "adapter_id")
    synthetic = result["source_kind"] == "LOCAL_SYNTHETIC"
    expected_schema = _SYNTHETIC_ADAPTER_SCHEMA if synthetic else _SCHEMA_VERSION
    if result["schema_version"] != expected_schema:
        raise CatalogError("CATALOG_ADAPTER_SCHEMA: unsupported adapter schema")
    if type(result["read_only"]) is not bool or result["read_only"] is not True:
        raise CatalogError("CATALOG_ADAPTER_UNSAFE: adapters must be read-only")
    source_auth = (result["source_kind"], result["auth_mode"])
    if (
        source_auth
        not in {
            ("HTTPS_JSON", "none"),
            ("HTTPS_BYTES", "none"),
            ("LOCAL_MANIFEST_IMPORT", "manifest-declared"),
            ("LOCAL_SYNTHETIC", "none"),
        }
        or result["data_tier"] != "TIER_0"
    ):
        raise CatalogError(
            "CATALOG_ADAPTER_UNSAFE: source/auth mode is outside the read-only TIER_0 policy"
        )
    if result["evidence_ceiling"] != "E0":
        raise CatalogError(
            "CATALOG_EVIDENCE_CEILING: this adapter cannot claim above E0"
        )
    hosts = result["allowed_hosts"]
    if synthetic:
        if (
            hosts != []
            or result["base_url"] != "urn:qrae:synthetic"
            or result["normalized_schema"] != "qrae.price-bars.v1"
            or result["license_status"] != _SYNTHETIC_LICENSE
        ):
            raise CatalogError(
                "CATALOG_ADAPTER_UNSAFE: synthetic adapters require local identity and explicit fixture scope"
            )
        return _canonical_mapping(result)
    if not isinstance(hosts, list) or not hosts or len(hosts) > 16:
        raise CatalogError(
            "CATALOG_ADAPTER_SCHEMA: allowed_hosts must be a non-empty list"
        )
    normalized_hosts: list[str] = []
    for host in hosts:
        host = _require_text(host, "allowed_hosts").casefold().rstrip(".")
        if (
            not host
            or "/" in host
            or ":" in host
            or "@" in host
            or host == "localhost"
            or host.endswith(".localhost")
        ):
            raise CatalogError("CATALOG_ADAPTER_UNSAFE: allowed host is invalid")
        normalized_hosts.append(host)
    if len(set(normalized_hosts)) != len(normalized_hosts):
        raise CatalogError("CATALOG_ADAPTER_SCHEMA: allowed_hosts contains duplicates")
    result["allowed_hosts"] = sorted(normalized_hosts)

    try:
        base = urlsplit(result["base_url"])
        base_port = base.port
    except ValueError as exc:
        raise CatalogError("CATALOG_ADAPTER_UNSAFE: base_url is malformed") from exc
    if (
        base.scheme.casefold() != "https"
        or base.username is not None
        or base.password is not None
        or base.query
        or base.fragment
        or not base.hostname
        or (base_port is not None and base_port != 443)
        or base.hostname.casefold().rstrip(".") not in result["allowed_hosts"]
    ):
        raise CatalogError("CATALOG_ADAPTER_UNSAFE: base_url violates allowed_hosts")
    return _canonical_mapping(result)


def _synthetic_derivation(raw_bytes: bytes) -> dict[str, Any]:
    """Validate a canonical declaration; upstream hashes remain caller declarations."""

    if (
        not isinstance(raw_bytes, bytes)
        or not 0 < len(raw_bytes) <= _MAX_SYNTHETIC_MANIFEST_BYTES
    ):
        raise CatalogError("CATALOG_SYNTHETIC_SCHEMA: manifest must be bounded bytes")
    try:
        value = json.loads(raw_bytes.decode("utf-8"))
        canonical = canonical_json_bytes(value)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise CatalogError("CATALOG_SYNTHETIC_SCHEMA: invalid manifest JSON") from exc
    fields = {
        "schema_version",
        "synthetic",
        "market_family",
        "generator_sha256",
        "parameters",
        "collector_raw_sha256",
        "collector_normalized_sha256",
        "normalized_price_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or canonical != raw_bytes
        or value["schema_version"] != "qrae.synthetic-derivation.v1"
        or value["synthetic"] is not True
        or not isinstance(value["parameters"], dict)
    ):
        raise CatalogError("CATALOG_SYNTHETIC_SCHEMA: invalid synthetic declaration")
    _require_text(value["market_family"], "market_family")
    for field in (
        "generator_sha256",
        "collector_raw_sha256",
        "collector_normalized_sha256",
        "normalized_price_sha256",
    ):
        if not isinstance(value[field], str) or not _SHA256_RE.fullmatch(value[field]):
            raise CatalogError(f"CATALOG_SYNTHETIC_SCHEMA: {field} must be SHA-256")
    return value


def _validate_synthetic_binding(
    record: Mapping[str, Any], adapter: Mapping[str, Any], raw_bytes: bytes
) -> dict[str, Any]:
    derivation = _synthetic_derivation(raw_bytes)
    raw_hash = sha256_bytes(raw_bytes)
    if (
        record["source_uri"] != _SYNTHETIC_URI_PREFIX + raw_hash
        or record["raw_sha256"] != raw_hash
        or record["raw_size_bytes"] != len(raw_bytes)
        or derivation["normalized_price_sha256"] != record["normalized_sha256"]
        or derivation["market_family"] != adapter["market_family"]
        or record["normalized_schema"] != "qrae.price-bars.v1"
        or record["evidence_ceiling"] != "E0"
        or record["availability_basis"] != "OBSERVED_AT_IMPORT"
        or record["requested_at"] != record["retrieved_at"]
        or record["source_http_date"] is not None
        or record["response_headers"] != {}
    ):
        raise CatalogError(
            "CATALOG_SYNTHETIC_BINDING: derivation or observation differs"
        )
    return derivation


def _relative_object_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CatalogError("CATALOG_UNSAFE_PATH: object path is not normalized")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or not value.startswith("objects/")
    ):
        raise CatalogError("CATALOG_UNSAFE_PATH: object path escapes the catalog")
    return value


def _snapshot_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "adapter_id",
        "request_id",
        "source_uri",
        "requested_at",
        "retrieved_at",
        "available_at",
        "cutoff",
        "availability_basis",
        "raw_sha256",
        "raw_size_bytes",
        "normalized_sha256",
        "normalized_size_bytes",
        "normalized_schema",
        "item_count",
        "max_source_event_time",
        "source_http_date",
        "response_headers",
        "adapter_contract_sha256",
        "evidence_ceiling",
    )
    return {field: record[field] for field in fields}


def _adapter_record_sha256(contract: Mapping[str, Any], registered_at: str) -> str:
    """Bind the immutable adapter contract to its canonical registration time."""

    parsed = _parse_timestamp(registered_at, "registered_at")
    if _timestamp(parsed) != registered_at:
        raise CatalogError("CATALOG_HASH_MISMATCH: registered_at is not canonical")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "contract": dict(contract),
                "registered_at": registered_at,
            }
        )
    )


def verify_provenance_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a frozen catalog provenance envelope without the live catalog."""

    if not isinstance(value, Mapping):
        raise CatalogError("CATALOG_PROVENANCE_SCHEMA: envelope must be an object")
    envelope = _canonical_mapping(value)
    expected_fields = {
        "schema_version",
        "kind",
        "adapter",
        "snapshot",
        "selected_object",
    }
    synthetic_envelope = envelope.get("schema_version") == _SYNTHETIC_PROVENANCE_SCHEMA
    if synthetic_envelope:
        expected_fields.add("synthetic_derivation")
    if (
        set(envelope) != expected_fields
        or envelope["schema_version"]
        not in {_SCHEMA_VERSION, _SYNTHETIC_PROVENANCE_SCHEMA}
        or envelope["kind"] != "CATALOG_SNAPSHOT_PROVENANCE"
    ):
        raise CatalogError("CATALOG_PROVENANCE_SCHEMA: envelope fields are invalid")

    adapter = envelope["adapter"]
    if not isinstance(adapter, dict) or set(adapter) != _ADAPTER_RECORD_FIELDS:
        raise CatalogError(
            "CATALOG_PROVENANCE_SCHEMA: adapter record fields are invalid"
        )
    normalized_adapter = _normalize_adapter(
        {key: adapter[key] for key in _ADAPTER_FIELDS}
    )
    if synthetic_envelope != (adapter["source_kind"] == "LOCAL_SYNTHETIC"):
        raise CatalogError("CATALOG_PROVENANCE_SCHEMA: synthetic envelope kind differs")
    contract_sha256 = _adapter_record_sha256(
        normalized_adapter, adapter["registered_at"]
    )
    if adapter["contract_sha256"] != contract_sha256:
        raise CatalogError("CATALOG_HASH_MISMATCH: provenance adapter hash differs")

    snapshot = envelope["snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_RECORD_FIELDS:
        raise CatalogError(
            "CATALOG_PROVENANCE_SCHEMA: snapshot record fields are invalid"
        )
    if snapshot["schema_version"] != _SCHEMA_VERSION:
        raise CatalogError("CATALOG_PROVENANCE_SCHEMA: snapshot schema is unsupported")
    for field in (
        "snapshot_id",
        "request_fingerprint",
        "raw_sha256",
        "normalized_sha256",
        "adapter_contract_sha256",
    ):
        if not isinstance(snapshot[field], str) or not _SHA256_RE.fullmatch(
            snapshot[field]
        ):
            raise CatalogError(f"CATALOG_PROVENANCE_SCHEMA: {field} is not SHA-256")
    _require_safe_id(snapshot["adapter_id"], "adapter_id")
    _require_safe_id(snapshot["request_id"], "request_id")
    if (
        type(snapshot["raw_size_bytes"]) is not int
        or snapshot["raw_size_bytes"] < 0
        or type(snapshot["normalized_size_bytes"]) is not int
        or snapshot["normalized_size_bytes"] < 0
        or type(snapshot["item_count"]) is not int
        or snapshot["item_count"] < 0
    ):
        raise CatalogError("CATALOG_PROVENANCE_SCHEMA: snapshot counts are invalid")
    if (
        not isinstance(snapshot["response_headers"], dict)
        or _normalize_headers(snapshot["response_headers"])
        != snapshot["response_headers"]
    ):
        raise CatalogError("CATALOG_PROVENANCE_SCHEMA: response headers are invalid")
    if (
        snapshot["snapshot_id"]
        != sha256_bytes(canonical_json_bytes(_snapshot_identity(snapshot)))
        or snapshot["request_fingerprint"] != snapshot["snapshot_id"]
    ):
        raise CatalogError(
            "CATALOG_HASH_MISMATCH: provenance snapshot identity differs"
        )

    requested = _parse_timestamp(snapshot["requested_at"], "requested_at")
    retrieved = _parse_timestamp(snapshot["retrieved_at"], "retrieved_at")
    available = _parse_timestamp(snapshot["available_at"], "available_at")
    cutoff = _parse_timestamp(snapshot["cutoff"], "cutoff")
    registered = _parse_timestamp(adapter["registered_at"], "registered_at")
    if (
        registered > requested
        or requested > retrieved
        or snapshot["availability_basis"]
        not in {"OBSERVED_AT_RETRIEVAL", "OBSERVED_AT_IMPORT"}
        or available != retrieved
        or cutoff != available
    ):
        raise CatalogError("CATALOG_HASH_MISMATCH: provenance PIT metadata differs")
    if (
        snapshot["max_source_event_time"] is not None
        and _parse_timestamp(snapshot["max_source_event_time"], "max_source_event_time")
        > retrieved
    ):
        raise CatalogError(
            "CATALOG_HASH_MISMATCH: provenance source event is in the future"
        )
    if snapshot["source_http_date"] is not None:
        _parse_timestamp(snapshot["source_http_date"], "source_http_date")
    if (
        snapshot["adapter_id"] != adapter["adapter_id"]
        or snapshot["adapter_contract_sha256"] != adapter["contract_sha256"]
        or snapshot["normalized_schema"] != adapter["normalized_schema"]
        or snapshot["evidence_ceiling"] != adapter["evidence_ceiling"]
    ):
        raise CatalogError("CATALOG_HASH_MISMATCH: provenance adapter binding differs")
    _validate_source_uri(
        snapshot["source_uri"], adapter["allowed_hosts"], adapter["source_kind"]
    )
    if synthetic_envelope:
        _validate_synthetic_binding(
            snapshot, adapter, canonical_json_bytes(envelope["synthetic_derivation"])
        )

    raw_path = f"objects/raw/{snapshot['raw_sha256'][:2]}/{snapshot['raw_sha256']}.bin"
    normalized_path = (
        f"objects/normalized/{snapshot['normalized_sha256'][:2]}/"
        f"{snapshot['normalized_sha256']}.bin"
    )
    if (
        snapshot["raw_object_path"] != raw_path
        or snapshot["normalized_object_path"] != normalized_path
    ):
        raise CatalogError("CATALOG_HASH_MISMATCH: provenance object path differs")

    selected = envelope["selected_object"]
    if not isinstance(selected, dict) or set(selected) != {
        "role",
        "path",
        "sha256",
        "bytes",
    }:
        raise CatalogError(
            "CATALOG_PROVENANCE_SCHEMA: selected object fields are invalid"
        )
    if selected != {
        "role": "normalized",
        "path": snapshot["normalized_object_path"],
        "sha256": snapshot["normalized_sha256"],
        "bytes": snapshot["normalized_size_bytes"],
    }:
        raise CatalogError("CATALOG_HASH_MISMATCH: selected object binding differs")
    return envelope


class DataCatalog:
    """SQLite index plus immutable content-addressed raw/normalized objects."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        original = Path(root)
        if has_link_or_reparse_component(original):
            raise CatalogError(
                "CATALOG_UNSAFE_PATH: catalog root cannot contain links or reparse points"
            )
        original.mkdir(parents=True, exist_ok=True)
        if has_link_or_reparse_component(original):
            raise CatalogError(
                "CATALOG_UNSAFE_PATH: catalog root changed into a link or reparse point"
            )
        self.root = original.resolve()
        self._objects_root = self.root / "objects"
        self._db_path = self.root / "catalog.sqlite3"
        self._lock = _root_lock(self.root)
        with self._lock:
            self._ensure_object_root()
            self._initialize_database()

    def _ensure_object_root(self) -> None:
        if is_link_or_reparse(self.root):
            raise CatalogError(
                "CATALOG_UNSAFE_PATH: catalog root became a link or reparse point"
            )
        if is_link_or_reparse(self._objects_root):
            raise CatalogError(
                "CATALOG_UNSAFE_PATH: object store cannot be a link or reparse point"
            )
        self._objects_root.mkdir(parents=True, exist_ok=True)
        if self._objects_root.resolve() != self._objects_root:
            raise CatalogError("CATALOG_UNSAFE_PATH: object store escapes the catalog")

    def _ensure_database_path(self) -> None:
        if is_link_or_reparse(self._db_path) or (
            self._db_path.exists() and not self._db_path.is_file()
        ):
            raise CatalogError("CATALOG_UNSAFE_PATH: catalog database is unsafe")

    def _connect(self) -> sqlite3.Connection:
        self._ensure_database_path()
        connection = sqlite3.connect(
            self._db_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS adapters (
                    adapter_id TEXT PRIMARY KEY,
                    contract_json TEXT NOT NULL,
                    contract_sha256 TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    adapter_id TEXT NOT NULL REFERENCES adapters(adapter_id),
                    request_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    available_at_us INTEGER NOT NULL,
                    cutoff TEXT NOT NULL,
                    availability_basis TEXT NOT NULL,
                    raw_sha256 TEXT NOT NULL,
                    raw_size_bytes INTEGER NOT NULL,
                    raw_object_path TEXT NOT NULL,
                    normalized_sha256 TEXT NOT NULL,
                    normalized_size_bytes INTEGER NOT NULL,
                    normalized_object_path TEXT NOT NULL,
                    normalized_schema TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    max_source_event_time TEXT,
                    source_http_date TEXT,
                    response_headers_json TEXT NOT NULL,
                    adapter_contract_sha256 TEXT NOT NULL,
                    evidence_ceiling TEXT NOT NULL,
                    UNIQUE(adapter_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS snapshots_pit
                    ON snapshots(adapter_id, available_at_us DESC, retrieved_at DESC, snapshot_id DESC);
                CREATE TABLE IF NOT EXISTS latest_snapshots (
                    adapter_id TEXT PRIMARY KEY REFERENCES adapters(adapter_id),
                    snapshot_id TEXT NOT NULL UNIQUE REFERENCES snapshots(snapshot_id)
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO catalog_meta(key, value) VALUES('schema_version', ?)",
                (_SCHEMA_VERSION,),
            )
            row = connection.execute(
                "SELECT value FROM catalog_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None or row["value"] != _SCHEMA_VERSION:
                raise CatalogError("CATALOG_SCHEMA_VERSION: unsupported catalog schema")
            self._initialize_adapter_hash_scope(connection)
        finally:
            connection.close()

    def _initialize_adapter_hash_scope(self, connection: sqlite3.Connection) -> None:
        """Initialize the bound-time hash scope; refuse untrusted legacy metadata."""

        scope = connection.execute(
            "SELECT value FROM catalog_meta WHERE key = 'adapter_hash_scope'"
        ).fetchone()
        if scope is not None:
            if scope["value"] != _ADAPTER_HASH_SCOPE:
                raise CatalogError(
                    "CATALOG_SCHEMA_VERSION: unsupported adapter hash scope"
                )
            return

        counts = connection.execute(
            "SELECT (SELECT COUNT(*) FROM adapters) + (SELECT COUNT(*) FROM snapshots)"
        ).fetchone()[0]
        if counts:
            raise CatalogError(
                "CATALOG_LEGACY_HASH_SCOPE: V0.11 registration times were not hash-bound; "
                "create a new V0.12 catalog and refetch or reimport"
            )
        connection.execute(
            "INSERT INTO catalog_meta(key, value) VALUES('adapter_hash_scope', ?)",
            (_ADAPTER_HASH_SCOPE,),
        )

    def _object_target(self, relative: str) -> Path:
        relative = _relative_object_path(relative)
        self._ensure_object_root()
        target = self.root / PurePosixPath(relative)
        current = self.root
        for part in PurePosixPath(relative).parts[:-1]:
            current = current / part
            if current.exists() and is_link_or_reparse(current):
                raise CatalogError(
                    "CATALOG_UNSAFE_PATH: object parent is a link or reparse point"
                )
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.resolve() != parent:
            raise CatalogError("CATALOG_UNSAFE_PATH: object parent escapes the store")
        resolved = target.resolve(strict=False)
        try:
            resolved.relative_to(self._objects_root)
        except ValueError as exc:
            raise CatalogError(
                "CATALOG_UNSAFE_PATH: object path escapes the store"
            ) from exc
        if target.exists() and is_link_or_reparse(target):
            raise CatalogError(
                "CATALOG_UNSAFE_PATH: object cannot be a link or reparse point"
            )
        return target

    def _write_object(self, kind: str, digest: str, data: bytes) -> tuple[str, bool]:
        relative = f"objects/{kind}/{digest[:2]}/{digest}.bin"
        target = self._object_target(relative)
        if target.exists():
            if not target.is_file() or sha256_file(target) != digest:
                raise CatalogError(
                    "CATALOG_HASH_MISMATCH: existing object digest differs"
                )
            return relative, False
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
                created = True
            except FileExistsError:
                created = False
            if (
                not target.is_file()
                or is_link_or_reparse(target)
                or sha256_file(target) != digest
            ):
                raise CatalogError(
                    "CATALOG_HASH_MISMATCH: published object digest differs"
                )
            return relative, created
        finally:
            temporary.unlink(missing_ok=True)

    def register_adapter(
        self, contract: Mapping[str, Any], *, registered_at: datetime
    ) -> dict[str, Any]:
        normalized = _normalize_adapter(contract)
        registered = _timestamp(_utc(registered_at, "registered_at"))
        contract_bytes = canonical_json_bytes(normalized)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM adapters WHERE adapter_id = ?",
                    (normalized["adapter_id"],),
                ).fetchone()
                if row is not None:
                    existing_digest = _adapter_record_sha256(
                        normalized, row["registered_at"]
                    )
                    if (
                        row["contract_sha256"] != existing_digest
                        or row["contract_json"].encode() != contract_bytes
                    ):
                        raise CatalogError(
                            "CATALOG_ADAPTER_CONFLICT: adapter contract is immutable"
                        )
                    connection.execute("COMMIT")
                    return self._adapter_from_row(row)
                digest = _adapter_record_sha256(normalized, registered)
                connection.execute(
                    "INSERT INTO adapters(adapter_id, contract_json, contract_sha256, registered_at) "
                    "VALUES(?, ?, ?, ?)",
                    (
                        normalized["adapter_id"],
                        contract_bytes.decode(),
                        digest,
                        registered,
                    ),
                )
                connection.execute("COMMIT")
                return {
                    **normalized,
                    "contract_sha256": digest,
                    "registered_at": registered,
                }
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    @staticmethod
    def _adapter_from_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            contract = json.loads(row["contract_json"])
        except json.JSONDecodeError as exc:
            raise CatalogError(
                "CATALOG_CORRUPT: adapter contract JSON is invalid"
            ) from exc
        return {
            **contract,
            "contract_sha256": row["contract_sha256"],
            "registered_at": row["registered_at"],
        }

    def register_snapshot(
        self,
        *,
        request_id: str,
        adapter_id: str,
        source_uri: str,
        requested_at: datetime,
        retrieved_at: datetime,
        available_at: datetime,
        cutoff: datetime,
        availability_basis: str,
        raw_bytes: bytes,
        normalized_bytes: bytes,
        normalized_schema: str,
        item_count: int,
        max_source_event_time: datetime | None,
        source_http_date: datetime | None,
        response_headers: Mapping[str, Any],
        expected_latest_snapshot_id: str | object | None = _LATEST_UNSET,
    ) -> dict[str, Any]:
        request_id = _require_safe_id(request_id, "request_id")
        adapter_id = _require_safe_id(adapter_id, "adapter_id")
        requested = _utc(requested_at, "requested_at")
        retrieved = _utc(retrieved_at, "retrieved_at")
        available = _utc(available_at, "available_at")
        cutoff_value = _utc(cutoff, "cutoff")
        if requested > retrieved:
            raise CatalogError(
                "CATALOG_INVALID_TIME: requested_at is after retrieved_at"
            )
        if (
            availability_basis not in {"OBSERVED_AT_RETRIEVAL", "OBSERVED_AT_IMPORT"}
            or available != retrieved
        ):
            raise CatalogError(
                "CATALOG_INVALID_TIME: snapshots must be observed at retrieval or import"
            )
        if cutoff_value != available:
            raise CatalogError(
                "CATALOG_INVALID_TIME: cutoff must equal observed availability"
            )
        source_event = (
            _utc(max_source_event_time, "max_source_event_time")
            if max_source_event_time is not None
            else None
        )
        if source_event is not None and source_event > retrieved:
            raise CatalogError(
                "CATALOG_FUTURE_DATA: source event time exceeds retrieval time"
            )
        http_date = (
            _utc(source_http_date, "source_http_date")
            if source_http_date is not None
            else None
        )
        if type(item_count) is not int or item_count < 0:
            raise CatalogError(
                "CATALOG_INVALID_FIELD: item_count must be a non-negative integer"
            )
        if not isinstance(raw_bytes, bytes) or not isinstance(normalized_bytes, bytes):
            raise CatalogError("CATALOG_INVALID_FIELD: snapshot payloads must be bytes")
        normalized_schema = _require_safe_id(normalized_schema, "normalized_schema")
        headers = _normalize_headers(response_headers)
        if (
            expected_latest_snapshot_id is not _LATEST_UNSET
            and expected_latest_snapshot_id is not None
        ):
            expected_latest_snapshot_id = _require_safe_id(
                expected_latest_snapshot_id, "expected_latest_snapshot_id"
            )

        with self._lock:
            connection = self._connect()
            created_paths: list[str] = []
            try:
                connection.execute("BEGIN IMMEDIATE")
                adapter_row = connection.execute(
                    "SELECT * FROM adapters WHERE adapter_id = ?", (adapter_id,)
                ).fetchone()
                if adapter_row is None:
                    raise CatalogError(
                        "CATALOG_UNKNOWN_ADAPTER: adapter is not registered"
                    )
                adapter = self._verified_adapter_from_row(adapter_row)
                if (
                    _parse_timestamp(adapter["registered_at"], "registered_at")
                    > requested
                ):
                    raise CatalogError(
                        "CATALOG_INVALID_TIME: adapter was registered after the request"
                    )
                if normalized_schema != adapter["normalized_schema"]:
                    raise CatalogError(
                        "CATALOG_SCHEMA_MISMATCH: normalized schema differs"
                    )
                source_uri = _validate_source_uri(
                    source_uri, adapter["allowed_hosts"], adapter["source_kind"]
                )
                raw_digest = sha256_bytes(raw_bytes)
                normalized_digest = sha256_bytes(normalized_bytes)
                record: dict[str, Any] = {
                    "schema_version": _SCHEMA_VERSION,
                    "adapter_id": adapter_id,
                    "request_id": request_id,
                    "source_uri": source_uri,
                    "requested_at": _timestamp(requested),
                    "retrieved_at": _timestamp(retrieved),
                    "available_at": _timestamp(available),
                    "cutoff": _timestamp(cutoff_value),
                    "availability_basis": availability_basis,
                    "raw_sha256": raw_digest,
                    "raw_size_bytes": len(raw_bytes),
                    "normalized_sha256": normalized_digest,
                    "normalized_size_bytes": len(normalized_bytes),
                    "normalized_schema": normalized_schema,
                    "item_count": item_count,
                    "max_source_event_time": _timestamp(source_event)
                    if source_event
                    else None,
                    "source_http_date": _timestamp(http_date) if http_date else None,
                    "response_headers": headers,
                    "adapter_contract_sha256": adapter["contract_sha256"],
                    "evidence_ceiling": adapter["evidence_ceiling"],
                }
                if adapter["source_kind"] == "LOCAL_SYNTHETIC":
                    _validate_synthetic_binding(record, adapter, raw_bytes)
                fingerprint = sha256_bytes(
                    canonical_json_bytes(_snapshot_identity(record))
                )
                record["snapshot_id"] = fingerprint
                record["request_fingerprint"] = fingerprint

                existing = connection.execute(
                    "SELECT * FROM snapshots WHERE adapter_id = ? AND request_id = ?",
                    (adapter_id, request_id),
                ).fetchone()
                if existing is not None:
                    if existing["request_fingerprint"] != fingerprint:
                        raise CatalogError(
                            "CATALOG_IDEMPOTENCY_CONFLICT: request_id has different content"
                        )
                    result = self._snapshot_from_row(existing)
                    self._verify_snapshot_objects(result)
                    connection.execute("COMMIT")
                    return result

                if expected_latest_snapshot_id is not _LATEST_UNSET:
                    latest_row = connection.execute(
                        "SELECT snapshot_id FROM latest_snapshots WHERE adapter_id = ?",
                        (adapter_id,),
                    ).fetchone()
                    actual_latest = (
                        latest_row["snapshot_id"] if latest_row is not None else None
                    )
                    if actual_latest != expected_latest_snapshot_id:
                        raise CatalogError(
                            "CATALOG_LINEAGE_CONFLICT: latest snapshot changed"
                        )

                raw_path, raw_created = self._write_object("raw", raw_digest, raw_bytes)
                if raw_created:
                    created_paths.append(raw_path)
                normalized_path, normalized_created = self._write_object(
                    "normalized", normalized_digest, normalized_bytes
                )
                if normalized_created:
                    created_paths.append(normalized_path)
                record["raw_object_path"] = raw_path
                record["normalized_object_path"] = normalized_path
                self._insert_snapshot(
                    connection,
                    record,
                    available_at_us=_epoch_microseconds(available),
                )
                self._advance_latest(connection, record)
                connection.execute("COMMIT")
                return record
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                self._remove_unreferenced_objects(connection, created_paths)
                raise
            finally:
                connection.close()

    def _insert_snapshot(
        self,
        connection: sqlite3.Connection,
        record: Mapping[str, Any],
        *,
        available_at_us: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO snapshots(
                snapshot_id, adapter_id, request_id, request_fingerprint, source_uri,
                requested_at, retrieved_at, available_at, available_at_us, cutoff,
                availability_basis, raw_sha256, raw_size_bytes, raw_object_path,
                normalized_sha256, normalized_size_bytes, normalized_object_path,
                normalized_schema, item_count, max_source_event_time, source_http_date,
                response_headers_json, adapter_contract_sha256, evidence_ceiling
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["snapshot_id"],
                record["adapter_id"],
                record["request_id"],
                record["request_fingerprint"],
                record["source_uri"],
                record["requested_at"],
                record["retrieved_at"],
                record["available_at"],
                available_at_us,
                record["cutoff"],
                record["availability_basis"],
                record["raw_sha256"],
                record["raw_size_bytes"],
                record["raw_object_path"],
                record["normalized_sha256"],
                record["normalized_size_bytes"],
                record["normalized_object_path"],
                record["normalized_schema"],
                record["item_count"],
                record["max_source_event_time"],
                record["source_http_date"],
                canonical_json_bytes(record["response_headers"]).decode(),
                record["adapter_contract_sha256"],
                record["evidence_ceiling"],
            ),
        )

    @staticmethod
    def _advance_latest(
        connection: sqlite3.Connection, record: Mapping[str, Any]
    ) -> None:
        row = connection.execute(
            """
            SELECT s.available_at_us, s.retrieved_at, s.snapshot_id
            FROM latest_snapshots l JOIN snapshots s ON s.snapshot_id = l.snapshot_id
            WHERE l.adapter_id = ?
            """,
            (record["adapter_id"],),
        ).fetchone()
        candidate = (
            _epoch_microseconds(
                _parse_timestamp(record["available_at"], "available_at")
            ),
            record["retrieved_at"],
            record["snapshot_id"],
        )
        current = (
            (row["available_at_us"], row["retrieved_at"], row["snapshot_id"])
            if row is not None
            else None
        )
        if current is None or candidate > current:
            connection.execute(
                "INSERT INTO latest_snapshots(adapter_id, snapshot_id) VALUES(?, ?) "
                "ON CONFLICT(adapter_id) DO UPDATE SET snapshot_id = excluded.snapshot_id",
                (record["adapter_id"], record["snapshot_id"]),
            )

    def _remove_unreferenced_objects(
        self, connection: sqlite3.Connection, paths: list[str]
    ) -> None:
        for relative in paths:
            row = connection.execute(
                "SELECT 1 FROM snapshots WHERE raw_object_path = ? OR normalized_object_path = ? LIMIT 1",
                (relative, relative),
            ).fetchone()
            if row is None:
                target = self._object_target(relative)
                target.unlink(missing_ok=True)
                for parent in (target.parent, target.parent.parent):
                    try:
                        parent.rmdir()
                    except OSError:
                        break

    @staticmethod
    def _snapshot_from_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            headers = json.loads(row["response_headers_json"])
        except json.JSONDecodeError as exc:
            raise CatalogError(
                "CATALOG_CORRUPT: response header JSON is invalid"
            ) from exc
        return {
            "schema_version": _SCHEMA_VERSION,
            "snapshot_id": row["snapshot_id"],
            "request_fingerprint": row["request_fingerprint"],
            "adapter_id": row["adapter_id"],
            "request_id": row["request_id"],
            "source_uri": row["source_uri"],
            "requested_at": row["requested_at"],
            "retrieved_at": row["retrieved_at"],
            "available_at": row["available_at"],
            "cutoff": row["cutoff"],
            "availability_basis": row["availability_basis"],
            "raw_sha256": row["raw_sha256"],
            "raw_size_bytes": row["raw_size_bytes"],
            "raw_object_path": row["raw_object_path"],
            "normalized_sha256": row["normalized_sha256"],
            "normalized_size_bytes": row["normalized_size_bytes"],
            "normalized_object_path": row["normalized_object_path"],
            "normalized_schema": row["normalized_schema"],
            "item_count": row["item_count"],
            "max_source_event_time": row["max_source_event_time"],
            "source_http_date": row["source_http_date"],
            "response_headers": headers,
            "adapter_contract_sha256": row["adapter_contract_sha256"],
            "evidence_ceiling": row["evidence_ceiling"],
        }

    def _verified_adapter_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        adapter = self._adapter_from_row(row)
        normalized = _normalize_adapter({key: adapter[key] for key in _ADAPTER_FIELDS})
        canonical = canonical_json_bytes(normalized)
        if (
            canonical.decode() != row["contract_json"]
            or _adapter_record_sha256(normalized, row["registered_at"])
            != row["contract_sha256"]
        ):
            raise CatalogError(
                "CATALOG_HASH_MISMATCH: adapter contract verification failed"
            )
        return adapter

    def _verified_snapshot_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        record = self._snapshot_from_row(row)
        if (
            canonical_json_bytes(record["response_headers"]).decode()
            != row["response_headers_json"]
            or _normalize_headers(record["response_headers"])
            != record["response_headers"]
        ):
            raise CatalogError(
                "CATALOG_HASH_MISMATCH: response header record is not canonical"
            )
        if (
            record["snapshot_id"]
            != sha256_bytes(canonical_json_bytes(_snapshot_identity(record)))
            or record["request_fingerprint"] != record["snapshot_id"]
        ):
            raise CatalogError(
                "CATALOG_HASH_MISMATCH: snapshot record verification failed"
            )

        requested = _parse_timestamp(record["requested_at"], "requested_at")
        retrieved = _parse_timestamp(record["retrieved_at"], "retrieved_at")
        available = _parse_timestamp(record["available_at"], "available_at")
        cutoff = _parse_timestamp(record["cutoff"], "cutoff")
        if (
            requested > retrieved
            or record["availability_basis"]
            not in {"OBSERVED_AT_RETRIEVAL", "OBSERVED_AT_IMPORT"}
            or available != retrieved
            or cutoff != available
            or row["available_at_us"] != _epoch_microseconds(available)
        ):
            raise CatalogError("CATALOG_HASH_MISMATCH: snapshot PIT metadata differs")
        if (
            record["max_source_event_time"] is not None
            and _parse_timestamp(
                record["max_source_event_time"], "max_source_event_time"
            )
            > retrieved
        ):
            raise CatalogError(
                "CATALOG_HASH_MISMATCH: source event time is in the future"
            )
        if record["source_http_date"] is not None:
            _parse_timestamp(record["source_http_date"], "source_http_date")
        if (
            type(record["item_count"]) is not int
            or record["item_count"] < 0
            or type(record["raw_size_bytes"]) is not int
            or record["raw_size_bytes"] < 0
            or type(record["normalized_size_bytes"]) is not int
            or record["normalized_size_bytes"] < 0
        ):
            raise CatalogError("CATALOG_CORRUPT: snapshot counts are invalid")

        adapter_row = connection.execute(
            "SELECT * FROM adapters WHERE adapter_id = ?", (record["adapter_id"],)
        ).fetchone()
        if adapter_row is None:
            raise CatalogError("CATALOG_CORRUPT: snapshot adapter is missing")
        adapter = self._verified_adapter_from_row(adapter_row)
        if (
            _parse_timestamp(adapter["registered_at"], "registered_at") > requested
            or record["adapter_contract_sha256"] != adapter["contract_sha256"]
            or record["normalized_schema"] != adapter["normalized_schema"]
            or record["evidence_ceiling"] != adapter["evidence_ceiling"]
        ):
            raise CatalogError(
                "CATALOG_HASH_MISMATCH: snapshot adapter binding differs"
            )
        _validate_source_uri(
            record["source_uri"], adapter["allowed_hosts"], adapter["source_kind"]
        )
        if adapter["source_kind"] == "LOCAL_SYNTHETIC":
            self._verify_snapshot_objects(record)
            raw_path = self._object_target(record["raw_object_path"])
            if record["raw_size_bytes"] > _MAX_SYNTHETIC_MANIFEST_BYTES:
                raise CatalogError("CATALOG_SYNTHETIC_SCHEMA: manifest exceeds limit")
            _validate_synthetic_binding(record, adapter, raw_path.read_bytes())
        return record

    def _verify_snapshot_objects(self, record: Mapping[str, Any]) -> None:
        for path_field, hash_field, size_field in (
            ("raw_object_path", "raw_sha256", "raw_size_bytes"),
            ("normalized_object_path", "normalized_sha256", "normalized_size_bytes"),
        ):
            expected = record[hash_field]
            if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
                raise CatalogError("CATALOG_CORRUPT: object hash is invalid")
            expected_path = (
                f"objects/{'raw' if path_field == 'raw_object_path' else 'normalized'}/"
                f"{expected[:2]}/{expected}.bin"
            )
            if record[path_field] != expected_path:
                raise CatalogError(
                    "CATALOG_HASH_MISMATCH: object path is not hash-addressed"
                )
            target = self._object_target(record[path_field])
            if (
                is_link_or_reparse(target)
                or not target.is_file()
                or target.stat().st_size != record[size_field]
                or sha256_file(target) != expected
            ):
                raise CatalogError(
                    "CATALOG_HASH_MISMATCH: snapshot object verification failed"
                )

    @staticmethod
    def _check_requested_tier(record: Mapping[str, Any], requested: str) -> None:
        if requested not in _EVIDENCE_ORDER:
            raise CatalogError(
                "CATALOG_EVIDENCE_CEILING: unknown requested evidence tier"
            )
        ceiling = record["evidence_ceiling"]
        if (
            ceiling not in _EVIDENCE_ORDER
            or _EVIDENCE_ORDER[requested] > _EVIDENCE_ORDER[ceiling]
        ):
            raise CatalogError(
                "CATALOG_EVIDENCE_CEILING: requested tier exceeds snapshot ceiling"
            )

    def resolve_snapshot(
        self,
        snapshot_id: str,
        *,
        as_of: datetime,
        requested_evidence_tier: str = "E0",
    ) -> dict[str, Any]:
        snapshot_id = _require_safe_id(snapshot_id, "snapshot_id")
        as_of_us = _epoch_microseconds(_utc(as_of, "as_of"))
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
                ).fetchone()
                if row is None:
                    raise CatalogError(
                        "PIT_NOT_AVAILABLE: snapshot was unavailable at cutoff"
                    )
                record = self._verified_snapshot_from_row(connection, row)
            finally:
                connection.close()
            if (
                _epoch_microseconds(
                    _parse_timestamp(record["available_at"], "available_at")
                )
                > as_of_us
            ):
                raise CatalogError(
                    "PIT_NOT_AVAILABLE: snapshot was unavailable at cutoff"
                )
            self._check_requested_tier(record, requested_evidence_tier)
            self._verify_snapshot_objects(record)
            return record

    def resolve_provenance(
        self,
        snapshot_id: str,
        *,
        as_of: datetime,
    ) -> dict[str, Any]:
        """Resolve one E0 snapshot and its immutable adapter/object bindings."""

        snapshot = self.resolve_snapshot(
            snapshot_id,
            as_of=as_of,
            requested_evidence_tier="E0",
        )
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM adapters WHERE adapter_id = ?",
                    (snapshot["adapter_id"],),
                ).fetchone()
                if row is None:
                    raise CatalogError("CATALOG_CORRUPT: provenance adapter is missing")
                adapter = self._verified_adapter_from_row(row)
            finally:
                connection.close()
        envelope = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "CATALOG_SNAPSHOT_PROVENANCE",
            "adapter": adapter,
            "snapshot": snapshot,
            "selected_object": {
                "role": "normalized",
                "path": snapshot["normalized_object_path"],
                "sha256": snapshot["normalized_sha256"],
                "bytes": snapshot["normalized_size_bytes"],
            },
        }
        if adapter["source_kind"] == "LOCAL_SYNTHETIC":
            raw_path = self._object_target(snapshot["raw_object_path"])
            envelope["schema_version"] = _SYNTHETIC_PROVENANCE_SCHEMA
            envelope["synthetic_derivation"] = _validate_synthetic_binding(
                snapshot, adapter, raw_path.read_bytes()
            )
        return verify_provenance_envelope(envelope)

    def resolve_latest(
        self,
        adapter_id: str,
        *,
        as_of: datetime,
        requested_evidence_tier: str = "E0",
    ) -> dict[str, Any]:
        adapter_id = _require_safe_id(adapter_id, "adapter_id")
        as_of_us = _epoch_microseconds(_utc(as_of, "as_of"))
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT * FROM snapshots WHERE adapter_id = ?", (adapter_id,)
                ).fetchall()
                records = [
                    self._verified_snapshot_from_row(connection, row) for row in rows
                ]
            finally:
                connection.close()
            candidates = [
                record
                for record in records
                if _epoch_microseconds(
                    _parse_timestamp(record["available_at"], "available_at")
                )
                <= as_of_us
            ]
            if not candidates:
                raise CatalogError(
                    "PIT_NOT_AVAILABLE: no snapshot was available at cutoff"
                )
            record = max(
                candidates,
                key=lambda item: (
                    _epoch_microseconds(
                        _parse_timestamp(item["available_at"], "available_at")
                    ),
                    _epoch_microseconds(
                        _parse_timestamp(item["retrieved_at"], "retrieved_at")
                    ),
                    item["snapshot_id"],
                ),
            )
            self._check_requested_tier(record, requested_evidence_tier)
            self._verify_snapshot_objects(record)
            return record

    def latest_snapshot_id(self, adapter_id: str) -> str | None:
        adapter_id = _require_safe_id(adapter_id, "adapter_id")
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT snapshot_id FROM latest_snapshots WHERE adapter_id = ?",
                    (adapter_id,),
                ).fetchone()
                return row["snapshot_id"] if row is not None else None
            finally:
                connection.close()

    def resolve_request(
        self, adapter_id: str, request_id: str
    ) -> dict[str, Any] | None:
        """Return a verified snapshot for one adapter-scoped idempotency key."""

        adapter_id = _require_safe_id(adapter_id, "adapter_id")
        request_id = _require_safe_id(request_id, "request_id")
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM snapshots WHERE adapter_id = ? AND request_id = ?",
                    (adapter_id, request_id),
                ).fetchone()
                if row is None:
                    return None
                record = self._verified_snapshot_from_row(connection, row)
                self._verify_snapshot_objects(record)
                return record
            finally:
                connection.close()

    def stats(self) -> dict[str, int]:
        with self._lock:
            connection = self._connect()
            try:
                return {
                    "adapters": connection.execute(
                        "SELECT COUNT(*) FROM adapters"
                    ).fetchone()[0],
                    "snapshots": connection.execute(
                        "SELECT COUNT(*) FROM snapshots"
                    ).fetchone()[0],
                    "latest_pointers": connection.execute(
                        "SELECT COUNT(*) FROM latest_snapshots"
                    ).fetchone()[0],
                }
            finally:
                connection.close()

    def verify_catalog(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_object_root()
            connection = self._connect()
            try:
                adapter_rows = connection.execute(
                    "SELECT * FROM adapters ORDER BY adapter_id"
                ).fetchall()
                for row in adapter_rows:
                    self._verified_adapter_from_row(row)

                snapshot_rows = connection.execute(
                    "SELECT * FROM snapshots ORDER BY snapshot_id"
                ).fetchall()
                referenced: set[str] = set()
                for row in snapshot_rows:
                    record = self._verified_snapshot_from_row(connection, row)
                    self._verify_snapshot_objects(record)
                    referenced.update(
                        (record["raw_object_path"], record["normalized_object_path"])
                    )

                latest_rows = connection.execute(
                    "SELECT adapter_id, snapshot_id FROM latest_snapshots ORDER BY adapter_id"
                ).fetchall()
                for row in latest_rows:
                    expected = connection.execute(
                        """
                        SELECT snapshot_id FROM snapshots WHERE adapter_id = ?
                        ORDER BY available_at_us DESC, retrieved_at DESC, snapshot_id DESC LIMIT 1
                        """,
                        (row["adapter_id"],),
                    ).fetchone()
                    if (
                        expected is None
                        or row["snapshot_id"] != expected["snapshot_id"]
                    ):
                        raise CatalogError(
                            "CATALOG_LATEST_MISMATCH: latest pointer regressed"
                        )
                if snapshot_rows and len(latest_rows) != len(
                    {row["adapter_id"] for row in snapshot_rows}
                ):
                    raise CatalogError(
                        "CATALOG_LATEST_MISMATCH: latest pointer is missing"
                    )

                actual: set[str] = set()
                for path in self._objects_root.rglob("*"):
                    if is_link_or_reparse(path):
                        raise CatalogError(
                            "CATALOG_UNSAFE_PATH: object tree contains a link or reparse point"
                        )
                    if path.is_file():
                        actual.add(path.relative_to(self.root).as_posix())
                return {
                    "schema_version": _SCHEMA_VERSION,
                    "adapters": len(adapter_rows),
                    "snapshots": len(snapshot_rows),
                    "latest_pointers": len(latest_rows),
                    "orphan_objects": len(actual - referenced),
                }
            finally:
                connection.close()


__all__ = ["CatalogError", "DataCatalog", "verify_provenance_envelope"]
