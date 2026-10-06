"""Immutable, run-linked ingestion for authenticated Codex draft reviews."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import (
    ArtifactError,
    _atomic_create_immutable,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .codex_broker import (
    BrokerValidationError,
    result_evidence_bindings,
    validate_result_envelope,
    validate_result_receipt,
    validate_work_order,
)
from .kernel import KernelError, verify_run
from .path_safety import has_link_or_reparse_component, is_link_or_reparse

BUNDLE_SCHEMA = "qrae.codex-review-bundle.v1"
BUNDLE_KIND = "CODEX_DRAFT_REVIEW"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FILES = frozenset({"work_order.json", "result.json", "receipt.json", "bundle.json"})
_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "bundle_id",
        "run",
        "task",
        "files",
        "evidence_bindings",
        "authority",
    }
)
_RUN_KEYS = frozenset(
    {"run_id", "status", "evidence_tier", "manifest_path", "manifest_sha256"}
)
_TASK_KEYS = frozenset(
    {
        "task_id",
        "task_kind",
        "work_order_sha256",
        "input_sha256",
        "result_sha256",
        "receipt_sha256",
        "evidence_sha256",
        "payload_sha256",
        "produced_at",
    }
)
_FILE_KEYS = frozenset({"path", "sha256", "bytes"})
_EVIDENCE_KEYS = frozenset({"path", "sha256", "bytes", "roles"})
_AUTHORITY = {
    "requested_transition": "NONE",
    "metric_authority": False,
    "state_transition_authorized": False,
    "capital_authorized": False,
    "live_trading_authorized": False,
}
_MAX_CANONICAL_JSON_BYTES = 2 * 1024 * 1024
_SAFE_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class ReviewBundleError(ValueError):
    """A Codex review could not be authenticated and frozen safely."""

    def __init__(self, message: str) -> None:
        self.code = (
            message.partition(":")[0] if ":" in message else "REVIEW_BUNDLE_ERROR"
        )
        super().__init__(message)


def _strict_keys(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReviewBundleError(f"REVIEW_SCHEMA: {label} fields are invalid")
    return value


def _canonical_mapping_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return canonical_json_bytes(dict(value))
    except (TypeError, ValueError) as exc:
        raise ReviewBundleError(
            "REVIEW_SCHEMA: value is not finite canonical JSON"
        ) from exc


def _read_canonical_mapping(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if has_link_or_reparse_component(path):
        raise ReviewBundleError(f"REVIEW_UNSAFE_PATH: {label} must be a regular file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ReviewBundleError(f"REVIEW_SCHEMA: duplicate {label} key {key!r}")
            value[key] = item
        return value

    try:
        descriptor = os.open(path, _SAFE_READ_FLAGS)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size > _MAX_CANONICAL_JSON_BYTES
            ):
                raise ReviewBundleError(
                    f"REVIEW_SCHEMA: {label} is not a bounded regular file"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ReviewBundleError(
                        f"REVIEW_SCHEMA: {label} changed while read"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ReviewBundleError(f"REVIEW_SCHEMA: {label} changed while read")
            after = os.fstat(descriptor)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ReviewBundleError(f"REVIEW_SCHEMA: {label} changed while read")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ReviewBundleError(f"REVIEW_SCHEMA: {label} contains {constant}")
            ),
        )
    except ReviewBundleError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewBundleError(
            f"REVIEW_SCHEMA: {label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_mapping_bytes(value):
        raise ReviewBundleError(f"REVIEW_SCHEMA: {label} must be canonical JSON")
    return value, raw


def _parse_produced_at(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ReviewBundleError("REVIEW_TIME: produced_at must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ReviewBundleError("REVIEW_TIME: produced_at is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReviewBundleError("REVIEW_TIME: produced_at must use UTC")
    canonical = (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    if canonical != value:
        raise ReviewBundleError(
            "REVIEW_TIME: produced_at must use canonical UTC seconds"
        )
    return parsed


def _validated_layout(run_dir: str | Path, task_id: str) -> tuple[Path, Path, Path]:
    if not isinstance(task_id, str) or not _ID_RE.fullmatch(task_id):
        raise ReviewBundleError("REVIEW_ID: task_id is invalid")
    lexical = Path(run_dir).expanduser()
    if has_link_or_reparse_component(lexical):
        raise ReviewBundleError(
            "REVIEW_UNSAFE_PATH: run_dir contains a link or reparse point"
        )
    path = lexical.resolve(strict=True)
    if not path.is_dir() or path.parent.name != "runs":
        raise ReviewBundleError(
            "REVIEW_LAYOUT: run_dir must be <output-root>/runs/<run-id>"
        )
    output_root = path.parent.parent.resolve(strict=True)
    if path.name == "" or not _ID_RE.fullmatch(path.name):
        raise ReviewBundleError("REVIEW_ID: run_id is invalid")
    bundle_dir = output_root / "codex-reviews" / path.name / task_id
    return path, output_root, bundle_dir


def _source_paths(
    output_root: Path, run_id: str, task_id: str
) -> tuple[Path, Path, Path]:
    return (
        output_root / "codex-work-orders" / run_id / f"{task_id}.json",
        output_root / "codex-outbox" / run_id / f"{task_id}.json",
        output_root / "codex-outbox" / run_id / f"{task_id}.receipt.json",
    )


def _manifest_bindings(run_dir: Path) -> tuple[dict[str, dict[str, Any]], bytes]:
    manifest, manifest_raw = _read_canonical_mapping(
        run_dir / "manifest.json", "run manifest"
    )
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise ReviewBundleError("REVIEW_RUN: run manifest artifacts are invalid")
    prefix = f"runs/{run_dir.name}/"
    bindings: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _FILE_KEYS:
            raise ReviewBundleError("REVIEW_RUN: run manifest artifact is invalid")
        bindings[prefix + row["path"]] = dict(row)
    bindings[prefix + "manifest.json"] = {
        "path": "manifest.json",
        "sha256": sha256_bytes(manifest_raw),
        "bytes": len(manifest_raw),
    }
    return bindings, manifest_raw


def _review_roles(
    order: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}
    roles.setdefault(order["input"]["path"], set()).add("INPUT")
    roles.setdefault(f"runs/{order['run_id']}/manifest.json", set()).add("RUN_MANIFEST")
    payload = result["payload"]
    for path in payload["artifact_refs"]:
        roles.setdefault(path, set()).add("PAYLOAD")
    for index, claim in enumerate(payload["claims"]):
        for path in claim["artifact_refs"]:
            roles.setdefault(path, set()).add(f"CLAIM:{index:03d}")
    return roles


def _evidence_bindings(
    output_root: Path,
    run_dir: Path,
    order: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    declared, _ = _manifest_bindings(run_dir)
    prefix = f"runs/{run_dir.name}/"
    try:
        authenticated = result_evidence_bindings(result, order, output_root)
    except BrokerValidationError as exc:
        raise ReviewBundleError(f"REVIEW_EVIDENCE: {exc}") from exc
    role_map = _review_roles(order, result)
    records: list[dict[str, Any]] = []
    for authenticated_record in authenticated:
        path = authenticated_record["path"]
        roles = role_map.get(path, {"AUTHENTICATED"})
        if not isinstance(path, str) or not path.startswith(prefix):
            raise ReviewBundleError(
                "REVIEW_EVIDENCE: evidence path is outside the source run"
            )
        relative = path[len(prefix) :]
        posix = PurePosixPath(relative)
        if (
            not relative
            or str(posix) != relative
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise ReviewBundleError("REVIEW_EVIDENCE: evidence path is not canonical")
        record = declared.get(path)
        if record is None:
            raise ReviewBundleError(
                "REVIEW_EVIDENCE: evidence is not run-manifest bound"
            )
        target = output_root / PurePosixPath(path)
        if has_link_or_reparse_component(target) or not target.is_file():
            raise ReviewBundleError(
                "REVIEW_EVIDENCE: evidence file is unsafe or missing"
            )
        if (
            authenticated_record["bytes"] != record["bytes"]
            or authenticated_record["sha256"] != record["sha256"]
        ):
            raise ReviewBundleError(
                "REVIEW_EVIDENCE: evidence differs from the run manifest"
            )
        records.append(
            {
                "path": path,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "roles": sorted(roles),
            }
        )
    input_record = next(
        (record for record in records if record["path"] == order["input"]["path"]),
        None,
    )
    if (
        input_record is None
        or input_record["sha256"] != order["input"]["sha256"]
        or input_record["bytes"] != order["input"]["size_bytes"]
    ):
        raise ReviewBundleError(
            "REVIEW_EVIDENCE: input binding differs from work order"
        )
    return records


def _file_record(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _assemble_bundle(
    run_dir: Path,
    output_root: Path,
    task_id: str,
    order: dict[str, Any],
    order_raw: bytes,
    result: dict[str, Any],
    result_raw: bytes,
    receipt: dict[str, Any],
    receipt_raw: bytes,
    *,
    receipt_key_root: str | Path | None,
) -> dict[str, Any]:
    produced = _parse_produced_at(result.get("produced_at"))
    try:
        order = validate_work_order(order, output_root, now=produced)
        result = validate_result_envelope(result, order, output_root)
        validate_result_receipt(
            receipt,
            result,
            order,
            output_root,
            receipt_key_root=receipt_key_root,
        )
    except BrokerValidationError as exc:
        raise ReviewBundleError(f"REVIEW_BROKER: {exc}") from exc
    if order["run_id"] != run_dir.name or order["task_id"] != task_id:
        raise ReviewBundleError("REVIEW_BINDING: task or run identity differs")
    if not isinstance(order.get("run_manifest"), Mapping):
        raise ReviewBundleError(
            "REVIEW_BINDING: work order has no prepare-time run manifest"
        )
    if result["status"] != "DRAFT_READY" or result["reason_code"] != "NONE":
        raise ReviewBundleError(
            "REVIEW_STATUS: only DRAFT_READY results can be ingested"
        )

    try:
        verified = verify_run(run_dir)
    except (ArtifactError, KernelError) as exc:
        raise ReviewBundleError(
            f"REVIEW_RUN: source run verification failed: {exc}"
        ) from exc
    if verified["status"] != "HUMAN_REVIEW":
        raise ReviewBundleError("REVIEW_RUN: source run is not ready for human review")
    evidence = _evidence_bindings(output_root, run_dir, order, result)
    _, run_manifest_raw = _read_canonical_mapping(
        run_dir / "manifest.json", "run manifest"
    )
    files = [
        _file_record("receipt.json", receipt_raw),
        _file_record("result.json", result_raw),
        _file_record("work_order.json", order_raw),
    ]
    descriptor: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "kind": BUNDLE_KIND,
        "run": {
            "run_id": run_dir.name,
            "status": verified["status"],
            "evidence_tier": verified["evidence_tier"],
            "manifest_path": f"runs/{run_dir.name}/manifest.json",
            "manifest_sha256": sha256_bytes(run_manifest_raw),
        },
        "task": {
            "task_id": task_id,
            "task_kind": order["task_kind"],
            "work_order_sha256": order["work_order_sha256"],
            "input_sha256": order["input"]["sha256"],
            "result_sha256": sha256_bytes(result_raw),
            "receipt_sha256": sha256_bytes(receipt_raw),
            "evidence_sha256": receipt["evidence_sha256"],
            "payload_sha256": result["payload_sha256"],
            "produced_at": result["produced_at"],
        },
        "files": files,
        "evidence_bindings": evidence,
        "authority": dict(_AUTHORITY),
    }
    descriptor["bundle_id"] = sha256_bytes(_canonical_mapping_bytes(descriptor))
    return descriptor


def _safe_review_parent(bundle_dir: Path, output_root: Path) -> Path:
    parent = bundle_dir.parent
    if has_link_or_reparse_component(parent):
        raise ReviewBundleError(
            "REVIEW_UNSAFE_PATH: review path contains a link or reparse point"
        )
    parent.mkdir(parents=True, exist_ok=True)
    if has_link_or_reparse_component(parent):
        raise ReviewBundleError("REVIEW_UNSAFE_PATH: review path became unsafe")
    resolved = parent.resolve(strict=True)
    expected_root = (output_root / "codex-reviews").resolve(strict=True)
    try:
        relative = resolved.relative_to(expected_root)
    except ValueError as exc:
        raise ReviewBundleError(
            "REVIEW_UNSAFE_PATH: bundle path escapes review root"
        ) from exc
    if len(relative.parts) != 1:
        raise ReviewBundleError("REVIEW_LAYOUT: review parent must identify one run")
    return resolved


def _discard_stage(stage: Path) -> None:
    """Remove only the known flat staging directory; never recurse."""

    try:
        for item in stage.iterdir():
            if item.is_file() and not is_link_or_reparse(item):
                item.unlink(missing_ok=True)
        stage.rmdir()
    except OSError:
        pass


def ingest_review_bundle(
    run_dir: str | Path,
    *,
    task_id: str,
    receipt_key_root: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate and freeze one DRAFT_READY outbox result without promotion."""

    source_run, output_root, bundle_dir = _validated_layout(run_dir, task_id)
    if bundle_dir.exists() or is_link_or_reparse(bundle_dir):
        return verify_review_bundle(bundle_dir, receipt_key_root=receipt_key_root)
    order_path, result_path, receipt_path = _source_paths(
        output_root, source_run.name, task_id
    )
    for path in (order_path, result_path, receipt_path):
        if has_link_or_reparse_component(path):
            raise ReviewBundleError(
                "REVIEW_UNSAFE_PATH: source path contains a link or reparse point"
            )
    order, order_raw = _read_canonical_mapping(order_path, "work order")
    result, result_raw = _read_canonical_mapping(result_path, "result")
    receipt, receipt_raw = _read_canonical_mapping(receipt_path, "receipt")

    state_before = sha256_file(source_run / "state.jsonl")
    manifest_before = sha256_file(source_run / "manifest.json")
    latest_path = output_root / "latest.json"
    latest_before = (
        _read_canonical_mapping(latest_path, "latest pointer")[1]
        if latest_path.exists() or is_link_or_reparse(latest_path)
        else None
    )
    descriptor = _assemble_bundle(
        source_run,
        output_root,
        task_id,
        order,
        order_raw,
        result,
        result_raw,
        receipt,
        receipt_raw,
        receipt_key_root=receipt_key_root,
    )
    parent = _safe_review_parent(bundle_dir, output_root)
    stage = Path(tempfile.mkdtemp(prefix=f".{task_id}.", suffix=".tmp", dir=parent))
    try:
        _atomic_create_immutable(stage / "work_order.json", order_raw)
        _atomic_create_immutable(stage / "result.json", result_raw)
        _atomic_create_immutable(stage / "receipt.json", receipt_raw)
        _atomic_create_immutable(
            stage / "bundle.json", _canonical_mapping_bytes(descriptor)
        )
    except ArtifactError as exc:
        _discard_stage(stage)
        raise ReviewBundleError(f"REVIEW_CONFLICT: {exc}") from exc

    if (
        sha256_file(source_run / "state.jsonl") != state_before
        or sha256_file(source_run / "manifest.json") != manifest_before
        or (
            _read_canonical_mapping(latest_path, "latest pointer")[1]
            if latest_path.exists() or is_link_or_reparse(latest_path)
            else None
        )
        != latest_before
    ):
        _discard_stage(stage)
        raise ReviewBundleError(
            "REVIEW_MUTATION: ingestion changed canonical research state"
        )
    try:
        stage.rename(bundle_dir)
    except OSError as exc:
        _discard_stage(stage)
        if bundle_dir.exists() or is_link_or_reparse(bundle_dir):
            return verify_review_bundle(bundle_dir, receipt_key_root=receipt_key_root)
        raise ReviewBundleError("REVIEW_CONFLICT: bundle publication failed") from exc
    return verify_review_bundle(bundle_dir, receipt_key_root=receipt_key_root)


def verify_review_bundle(
    bundle_dir: str | Path,
    *,
    receipt_key_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an ingested review using only local run state and receipt key."""

    lexical = Path(bundle_dir).expanduser()
    if has_link_or_reparse_component(lexical):
        raise ReviewBundleError(
            "REVIEW_UNSAFE_PATH: bundle contains a link or reparse point"
        )
    path = lexical.resolve(strict=True)
    if path.parent.parent.name != "codex-reviews":
        raise ReviewBundleError(
            "REVIEW_LAYOUT: bundle must be under codex-reviews/<run>/<task>"
        )
    run_id = path.parent.name
    task_id = path.name
    if not _ID_RE.fullmatch(run_id) or not _ID_RE.fullmatch(task_id):
        raise ReviewBundleError("REVIEW_ID: bundle run_id or task_id is invalid")
    output_root = path.parents[2]
    run_dir = output_root / "runs" / run_id
    if has_link_or_reparse_component(run_dir):
        raise ReviewBundleError(
            "REVIEW_UNSAFE_PATH: source run contains a link or reparse point"
        )
    entries = list(path.iterdir())
    actual = {item.name for item in entries}
    if actual != _FILES or any(
        has_link_or_reparse_component(item) or not item.is_file() for item in entries
    ):
        raise ReviewBundleError(
            "REVIEW_SCHEMA: bundle file set is incomplete or unexpected"
        )
    order, order_raw = _read_canonical_mapping(path / "work_order.json", "work order")
    result, result_raw = _read_canonical_mapping(path / "result.json", "result")
    receipt, receipt_raw = _read_canonical_mapping(path / "receipt.json", "receipt")
    persisted, _ = _read_canonical_mapping(path / "bundle.json", "bundle")
    expected = _assemble_bundle(
        run_dir,
        output_root,
        task_id,
        order,
        order_raw,
        result,
        result_raw,
        receipt,
        receipt_raw,
        receipt_key_root=receipt_key_root,
    )
    _strict_keys(persisted, _BUNDLE_KEYS, "bundle")
    _strict_keys(persisted.get("run"), _RUN_KEYS, "bundle.run")
    _strict_keys(persisted.get("task"), _TASK_KEYS, "bundle.task")
    if persisted.get("authority") != _AUTHORITY:
        raise ReviewBundleError("REVIEW_AUTHORITY: bundle grants forbidden authority")
    if persisted != expected:
        raise ReviewBundleError("REVIEW_HASH_MISMATCH: bundle commitment differs")
    if not isinstance(persisted["bundle_id"], str) or not _SHA256_RE.fullmatch(
        persisted["bundle_id"]
    ):
        raise ReviewBundleError("REVIEW_SCHEMA: bundle_id is invalid")
    return {
        "schema_version": BUNDLE_SCHEMA,
        "status": "DRAFT_INGESTED",
        "bundle_id": persisted["bundle_id"],
        "run_id": run_id,
        "task_id": task_id,
        "task_kind": persisted["task"]["task_kind"],
        "evidence_bindings": len(persisted["evidence_bindings"]),
        "state_transition_authorized": False,
        "bundle_dir": str(path),
    }


__all__ = [
    "BUNDLE_SCHEMA",
    "ReviewBundleError",
    "ingest_review_bundle",
    "verify_review_bundle",
]
