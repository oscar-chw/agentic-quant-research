"""Strict, deterministic contracts for the local QRAE research kernel."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .data_catalog import CatalogError, DataCatalog
from .path_safety import is_link_or_reparse


class ContractError(ValueError):
    """A work-order contract failed deterministic validation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class CatalogSnapshotRef:
    catalog_root: str
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class DatasetRef:
    path: str
    sha256: str
    format: str
    catalog_snapshot: CatalogSnapshotRef | None = None


@dataclass(frozen=True, slots=True)
class HypothesisSpec:
    mechanism: str
    prediction: str
    horizon: str
    falsifier: str


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    train_fraction: float
    validation_fraction: float
    min_test_observations: int


@dataclass(frozen=True, slots=True)
class StrategySpec:
    kind: str
    lookback: int
    cost_bps: float


@dataclass(frozen=True, slots=True)
class WorkOrder:
    schema_version: str
    work_order_id: str
    kind: str
    market_family: str
    created_at: datetime
    expires_at: datetime
    cutoff: datetime
    dataset: DatasetRef
    hypothesis: HypothesisSpec
    experiment: ExperimentSpec
    strategy: StrategySpec
    evidence_ceiling: str
    allow_live_trading: bool


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SECRET_KEY_MARKERS = (
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "apikey",
    "privatekey",
    "accesskey",
    "clientsecret",
    "authorization",
)

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "work_order_id",
    "kind",
    "market_family",
    "created_at",
    "expires_at",
    "cutoff",
    "dataset",
    "hypothesis",
    "experiment",
    "strategy",
    "evidence_ceiling",
    "allow_live_trading",
}


def canonical_json(value: Any) -> str:
    """Return a stable UTF-8 JSON representation for hashing and ledgers."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("ERR_CANONICAL_JSON", str(exc)) from exc


def canonical_sha256(value: Any) -> str:
    """Hash the canonical UTF-8 JSON representation of *value*."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the full dataset into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ContractError("ERR_INVALID_JSON", f"non-finite JSON number {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("ERR_DUPLICATE_KEY", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("ERR_INVALID_JSON", f"cannot read {path}: {exc}") from exc


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.casefold())


def _reject_secret_keys(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if any(marker in normalized for marker in _SECRET_KEY_MARKERS):
                raise ContractError(
                    "ERR_SECRET_KEY",
                    f"secret-like key at {location}.{key}",
                )
            _reject_secret_keys(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secret_keys(nested, f"{location}[{index}]")


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("ERR_INVALID_FIELD", f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        raise ContractError("ERR_MISSING_FIELD", f"{field} missing {missing}")
    if unknown:
        raise ContractError("ERR_UNKNOWN_FIELD", f"{field} has unknown keys {unknown}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("ERR_INVALID_FIELD", f"{field} must be a nonempty string")
    return value


def _utc_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ContractError("ERR_INVALID_TIME", f"{field} must be a UTC ISO timestamp")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ContractError(
            "ERR_INVALID_TIME", f"{field} must be a UTC ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ContractError("ERR_INVALID_TIME", f"{field} must use UTC")
    return parsed.astimezone(timezone.utc)


def _normalized_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ContractError("ERR_INVALID_TIME", "now must be timezone-aware")
    return now.astimezone(timezone.utc)


def _safe_dataset_path(raw_path: Any, workspace: Path) -> tuple[str, Path]:
    relative = _nonempty_string(raw_path, "dataset.path")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in relative
        or str(posix) != relative
        or any(part in {"", ".", ".."} for part in posix.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in relative)
    ):
        raise ContractError(
            "ERR_UNSAFE_PATH", "dataset.path must be a normalized relative path"
        )

    root = workspace.resolve()
    unresolved = root / Path(relative)
    current = root
    for part in posix.parts:
        current = current / part
        if is_link_or_reparse(current):
            raise ContractError(
                "ERR_UNSAFE_PATH", "dataset.path cannot contain symlinks"
            )
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(
            "ERR_UNSAFE_PATH", "dataset.path escapes workspace"
        ) from exc
    if not resolved.is_file():
        raise ContractError("ERR_DATASET_MISSING", f"dataset is not a file: {relative}")
    return relative, resolved


def _safe_catalog_root(raw_path: Any, workspace: Path) -> tuple[str, Path]:
    relative = _nonempty_string(raw_path, "dataset.catalog_snapshot.catalog_root")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in relative
        or str(posix) != relative
        or any(part in {"", ".", ".."} for part in posix.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in relative)
    ):
        raise ContractError(
            "ERR_UNSAFE_PATH", "catalog_root must be a normalized relative path"
        )
    root = workspace.resolve()
    unresolved = root / Path(relative)
    current = root
    for part in posix.parts:
        current = current / part
        if is_link_or_reparse(current):
            raise ContractError(
                "ERR_UNSAFE_PATH", "catalog_root cannot contain symlinks"
            )
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(
            "ERR_UNSAFE_PATH", "catalog_root escapes workspace"
        ) from exc
    if not resolved.is_dir():
        raise ContractError("ERR_PROVENANCE_MISSING", "catalog_root is not a directory")
    return relative, resolved


def _parse_catalog_snapshot(raw: Any, workspace: Path) -> CatalogSnapshotRef:
    value = _object(raw, "dataset.catalog_snapshot")
    _exact_keys(value, {"catalog_root", "snapshot_id"}, "dataset.catalog_snapshot")
    relative, _ = _safe_catalog_root(value["catalog_root"], workspace)
    snapshot_id = value["snapshot_id"]
    if not isinstance(snapshot_id, str) or not _SHA256.fullmatch(snapshot_id):
        raise ContractError(
            "ERR_INVALID_DATASET",
            "dataset.catalog_snapshot.snapshot_id must be SHA-256",
        )
    return CatalogSnapshotRef(relative, snapshot_id.casefold())


def resolve_dataset_provenance(
    dataset: DatasetRef,
    *,
    workspace: str | Path,
    cutoff: datetime,
    market_family: str,
) -> dict[str, Any] | None:
    """Resolve and verify an optional catalog snapshot binding for one dataset."""

    reference = dataset.catalog_snapshot
    if reference is None:
        return None
    workspace_path = Path(workspace).resolve()
    _, catalog_root = _safe_catalog_root(reference.catalog_root, workspace_path)
    try:
        envelope = DataCatalog(catalog_root).resolve_provenance(
            reference.snapshot_id,
            as_of=cutoff,
        )
    except CatalogError as exc:
        if exc.code == "PIT_NOT_AVAILABLE":
            raise ContractError(
                "ERR_PROVENANCE_PIT",
                "catalog snapshot was unavailable at the work-order cutoff",
            ) from exc
        if exc.code == "CATALOG_UNSAFE_PATH":
            raise ContractError("ERR_UNSAFE_PATH", str(exc)) from exc
        raise ContractError("ERR_PROVENANCE", str(exc)) from exc
    snapshot = envelope["snapshot"]
    selected = envelope["selected_object"]
    if envelope["adapter"]["market_family"] != market_family:
        raise ContractError(
            "ERR_PROVENANCE_MARKET",
            "catalog adapter market_family differs from the work order",
        )
    if snapshot["normalized_schema"] != "qrae.price-bars.v1":
        raise ContractError(
            "ERR_PROVENANCE_SCHEMA",
            "catalog snapshot is not normalized qrae.price-bars.v1",
        )
    if snapshot["evidence_ceiling"] != "E0":
        raise ContractError("ERR_EVIDENCE_CEILING", "catalog snapshot must remain E0")
    if not hmac.compare_digest(selected["sha256"], dataset.sha256):
        raise ContractError(
            "ERR_PROVENANCE_HASH",
            "catalog normalized object does not match dataset.sha256",
        )
    source = (workspace_path / dataset.path).resolve()
    if not source.is_file() or not hmac.compare_digest(
        sha256_file(source), dataset.sha256
    ):
        raise ContractError(
            "ERR_HASH_MISMATCH", "dataset changed after contract validation"
        )
    if source.stat().st_size != selected["bytes"]:
        raise ContractError(
            "ERR_PROVENANCE_HASH", "catalog normalized object size differs"
        )
    return envelope


def _parse_hypothesis(raw: Any) -> HypothesisSpec:
    value = _object(raw, "hypothesis")
    fields = {"mechanism", "prediction", "horizon", "falsifier"}
    _exact_keys(value, fields, "hypothesis")
    return HypothesisSpec(
        mechanism=_nonempty_string(value["mechanism"], "hypothesis.mechanism"),
        prediction=_nonempty_string(value["prediction"], "hypothesis.prediction"),
        horizon=_nonempty_string(value["horizon"], "hypothesis.horizon"),
        falsifier=_nonempty_string(value["falsifier"], "hypothesis.falsifier"),
    )


def _fraction(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("ERR_INVALID_SPLIT", f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ContractError("ERR_INVALID_SPLIT", f"{field} must be in (0, 1)")
    return result


def _parse_experiment(raw: Any) -> ExperimentSpec:
    value = _object(raw, "experiment")
    fields = {"train_fraction", "validation_fraction", "min_test_observations"}
    _exact_keys(value, fields, "experiment")
    train = _fraction(value["train_fraction"], "experiment.train_fraction")
    validation = _fraction(
        value["validation_fraction"], "experiment.validation_fraction"
    )
    if train + validation >= 1.0:
        raise ContractError(
            "ERR_INVALID_SPLIT",
            "train_fraction + validation_fraction must be less than 1",
        )
    minimum = value["min_test_observations"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ContractError(
            "ERR_INVALID_SPLIT",
            "experiment.min_test_observations must be an integer >= 1",
        )
    return ExperimentSpec(train, validation, minimum)


def _parse_strategy(raw: Any) -> StrategySpec:
    value = _object(raw, "strategy")
    fields = {"kind", "lookback", "cost_bps"}
    _exact_keys(value, fields, "strategy")
    if value["kind"] != "lagged_momentum":
        raise ContractError(
            "ERR_INVALID_STRATEGY", "strategy.kind must be lagged_momentum"
        )
    lookback = value["lookback"]
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ContractError(
            "ERR_INVALID_STRATEGY", "strategy.lookback must be an integer >= 1"
        )
    cost = value["cost_bps"]
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        raise ContractError("ERR_INVALID_STRATEGY", "strategy.cost_bps must be numeric")
    cost_bps = float(cost)
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ContractError(
            "ERR_INVALID_STRATEGY", "strategy.cost_bps must be finite and >= 0"
        )
    return StrategySpec("lagged_momentum", lookback, cost_bps)


def load_work_order(
    path: str | Path,
    workspace: str | Path,
    now: datetime | None = None,
) -> WorkOrder:
    """Load and fully validate a local, non-trading research work order."""

    raw = _load_json(Path(path))
    value = _object(raw, "work order")
    _reject_secret_keys(value)
    _exact_keys(value, _TOP_LEVEL_FIELDS, "work order")

    schema_version = value["schema_version"]
    if schema_version not in {"1.0", "1.1"}:
        raise ContractError(
            "ERR_SCHEMA_VERSION", "schema_version must be '1.0' or '1.1'"
        )
    work_order_id = value["work_order_id"]
    if not isinstance(work_order_id, str) or not _SAFE_ID.fullmatch(work_order_id):
        raise ContractError("ERR_INVALID_ID", "work_order_id is not a safe identifier")
    if value["kind"] != "PRICE_BASELINE":
        raise ContractError("ERR_INVALID_KIND", "kind must be PRICE_BASELINE")
    market_family = _nonempty_string(value["market_family"], "market_family")

    created_at = _utc_datetime(value["created_at"], "created_at")
    expires_at = _utc_datetime(value["expires_at"], "expires_at")
    cutoff = _utc_datetime(value["cutoff"], "cutoff")
    current_time = _normalized_now(now)
    if created_at >= expires_at:
        raise ContractError("ERR_INVALID_TIME", "created_at must precede expires_at")
    if created_at > current_time:
        raise ContractError(
            "ERR_FUTURE_ISSUANCE", "created_at cannot be later than now"
        )
    if current_time >= expires_at:
        raise ContractError("ERR_EXPIRED", "work order has expired")
    if cutoff > current_time:
        raise ContractError("ERR_FUTURE_CUTOFF", "cutoff cannot be later than now")
    if cutoff > expires_at:
        raise ContractError(
            "ERR_FUTURE_CUTOFF", "cutoff cannot be later than expires_at"
        )
    if value["evidence_ceiling"] != "E0":
        raise ContractError(
            "ERR_EVIDENCE_CEILING",
            "the current research kernel remains capped at E0",
        )
    if value["allow_live_trading"] is not False:
        raise ContractError("ERR_LIVE_TRADING", "allow_live_trading must be false")

    dataset_value = _object(value["dataset"], "dataset")
    dataset_fields = {"path", "sha256", "format"}
    if schema_version == "1.1":
        dataset_fields.add("catalog_snapshot")
    _exact_keys(dataset_value, dataset_fields, "dataset")
    if dataset_value["format"] != "csv":
        raise ContractError("ERR_INVALID_DATASET", "dataset.format must be csv")
    expected_hash = dataset_value["sha256"]
    if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
        raise ContractError(
            "ERR_INVALID_DATASET", "dataset.sha256 must be 64 hexadecimal characters"
        )
    relative_path, resolved_path = _safe_dataset_path(
        dataset_value["path"], Path(workspace)
    )
    actual_hash = sha256_file(resolved_path)
    if not hmac.compare_digest(actual_hash, expected_hash.casefold()):
        raise ContractError(
            "ERR_HASH_MISMATCH", "dataset.sha256 does not match dataset content"
        )

    catalog_snapshot = (
        _parse_catalog_snapshot(dataset_value["catalog_snapshot"], Path(workspace))
        if schema_version == "1.1"
        else None
    )
    dataset = DatasetRef(
        relative_path,
        expected_hash.casefold(),
        "csv",
        catalog_snapshot,
    )
    if catalog_snapshot is not None:
        resolve_dataset_provenance(
            dataset,
            workspace=workspace,
            cutoff=cutoff,
            market_family=market_family,
        )

    return WorkOrder(
        schema_version=schema_version,
        work_order_id=work_order_id,
        kind="PRICE_BASELINE",
        market_family=market_family,
        created_at=created_at,
        expires_at=expires_at,
        cutoff=cutoff,
        dataset=dataset,
        hypothesis=_parse_hypothesis(value["hypothesis"]),
        experiment=_parse_experiment(value["experiment"]),
        strategy=_parse_strategy(value["strategy"]),
        evidence_ceiling=value["evidence_ceiling"],
        allow_live_trading=False,
    )


__all__ = [
    "CatalogSnapshotRef",
    "ContractError",
    "DatasetRef",
    "ExperimentSpec",
    "HypothesisSpec",
    "StrategySpec",
    "WorkOrder",
    "canonical_json",
    "canonical_sha256",
    "load_work_order",
    "resolve_dataset_provenance",
    "sha256_file",
]
