"""One-command deterministic research plus bounded local Codex review workflow."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
import weakref
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import canonical_json_bytes, sha256_bytes
from .codex_broker import (
    MAX_OUTPUT_BYTES,
    MAX_TIMEOUT_SECONDS,
    TASK_KINDS,
    BrokerValidationError,
    Runner,
    prepare_work_order,
    run_work_order,
    validate_result_envelope,
    validate_result_receipt,
    validate_work_order,
)
from .codex_reviews import ingest_review_bundle, verify_review_bundle
from .kernel import run_price_baseline, verify_run
from .path_safety import has_link_or_reparse_component, is_link_or_reparse

WORKFLOW_SCHEMA = "qrae.research-draft-workflow.v1"
INVOCATION_SCHEMA = "qrae.research-draft-invocation.v1"
_AUTHORITY = {
    "capital": False,
    "live_trading": False,
    "metric": False,
    "state_transition": False,
}
_RUN_ARTIFACTS = (
    "report.md",
    "result.json",
    "validation.json",
    "decision.json",
    "claim_ledger.json",
    "knowledge.json",
    "manifest.json",
)
_WORKFLOW_FIELDS = {
    "schema_version",
    "workflow_id",
    "status",
    "run",
    "task",
    "review",
    "authority",
}
_RUN_FIELDS = {
    "run_id",
    "status",
    "evidence_tier",
    "manifest_path",
    "manifest_sha256",
}
_TASK_FIELDS = {
    "task_id",
    "task_kind",
    "work_order_path",
    "work_order_sha256",
    "result_path",
    "result_sha256",
    "receipt_path",
    "receipt_sha256",
    "result_status",
    "reason_code",
    "invocation_id",
    "invocation_path",
    "invocation_sha256",
}
_REVIEW_FIELDS = {
    "bundle_id",
    "bundle_path",
    "bundle_sha256",
    "draft_path",
    "draft_sha256",
}
_MAX_WORKFLOW_ARTIFACT_BYTES = 2 * 1024 * 1024
_INVOCATION_FIELDS = {
    "schema_version",
    "invocation_id",
    "run_id",
    "task_id",
    "task_kind",
    "input_path",
    "enable_codex",
    "expires_minutes",
    "timeout_seconds",
    "max_output_bytes",
}
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXECUTION_LOCKS_GUARD = threading.Lock()
_EXECUTION_LOCKS: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()


class ResearchWorkflowError(ValueError):
    """The composed research/Codex workflow violated an integrity contract."""

    def __init__(self, message: str) -> None:
        self.code = message.partition(":")[0] if ":" in message else "WORKFLOW_ERROR"
        super().__init__(message)


def _execution_lock(path: Path) -> Any:
    """Return one process-local lock for a workflow task's broker result."""

    key = os.path.normcase(str(path.resolve(strict=False)))
    with _EXECUTION_LOCKS_GUARD:
        lock = _EXECUTION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _EXECUTION_LOCKS[key] = lock
        return lock


def _workflow_execution_lock_path(output_root: str | Path) -> Path:
    """Serialize one process's workflows sharing a mutable output store."""

    resolved_output = Path(output_root).resolve(strict=False)
    return resolved_output / ".workflow-single-flight"


def _strict_mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ResearchWorkflowError(f"WORKFLOW_SCHEMA: {label} fields are invalid")
    return dict(value)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ResearchWorkflowError(f"WORKFLOW_TIME: {label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchWorkflowError(f"WORKFLOW_TIME: {label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResearchWorkflowError(f"WORKFLOW_TIME: {label} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _read_regular_bytes(
    path: Path,
    label: str,
    *,
    limit: int = _MAX_WORKFLOW_ARTIFACT_BYTES,
) -> bytes:
    if has_link_or_reparse_component(path):
        raise ResearchWorkflowError(f"WORKFLOW_PATH: {label} is unavailable or unsafe")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResearchWorkflowError(
            f"WORKFLOW_PATH: {label} is unavailable or unsafe"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > limit
        ):
            raise ResearchWorkflowError(
                f"WORKFLOW_SIZE: {label} is empty, non-regular, or oversized"
            )
        chunks: list[bytes] = []
        total = 0
        while total <= limit:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int | None]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_size,
                getattr(item, "st_mtime_ns", None),
            )

        path_after = os.stat(path, follow_symlinks=False)
        if total > limit:
            raise ResearchWorkflowError(f"WORKFLOW_SIZE: {label} is oversized")
        if (
            identity(before) != identity(after)
            or identity(after) != identity(path_after)
            or total != before.st_size
        ):
            raise ResearchWorkflowError(f"WORKFLOW_CHANGED: {label} changed while read")
        return b"".join(chunks)
    except ResearchWorkflowError:
        raise
    except OSError as exc:
        raise ResearchWorkflowError(
            f"WORKFLOW_PATH: {label} could not be read safely"
        ) from exc
    finally:
        os.close(descriptor)


def _read_canonical_mapping(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, label)
    try:

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ResearchWorkflowError(
                        f"WORKFLOW_SCHEMA: {label} contains duplicate keys"
                    )
                value[key] = item
            return value

        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ResearchWorkflowError(
                    f"WORKFLOW_SCHEMA: {label} contains non-finite {constant}"
                )
            ),
        )
    except ResearchWorkflowError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResearchWorkflowError(
            f"WORKFLOW_SCHEMA: {label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ResearchWorkflowError(f"WORKFLOW_SCHEMA: {label} is not canonical JSON")
    return value, raw


def _invocation_record(
    *,
    run_id: str,
    task_id: str,
    task_kind: str,
    input_name: str,
    enable_codex: bool,
    expires_minutes: int,
    timeout_seconds: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    if not _SAFE_ID_RE.fullmatch(run_id) or not _SAFE_ID_RE.fullmatch(task_id):
        raise ResearchWorkflowError("WORKFLOW_INVOCATION: run_id or task_id is invalid")
    if task_kind not in TASK_KINDS or input_name not in _RUN_ARTIFACTS:
        raise ResearchWorkflowError(
            "WORKFLOW_INVOCATION: task kind or input is invalid"
        )
    if type(enable_codex) is not bool:
        raise ResearchWorkflowError("WORKFLOW_INVOCATION: enable_codex must be boolean")
    if type(expires_minutes) is not int or not 1 <= expires_minutes <= 24 * 60:
        raise ResearchWorkflowError(
            "WORKFLOW_BUDGET: expires_minutes is outside policy"
        )
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise ResearchWorkflowError(
            "WORKFLOW_BUDGET: timeout_seconds is outside policy"
        )
    if (
        type(max_output_bytes) is not int
        or not 1 <= max_output_bytes <= MAX_OUTPUT_BYTES
    ):
        raise ResearchWorkflowError(
            "WORKFLOW_BUDGET: max_output_bytes is outside policy"
        )
    value: dict[str, Any] = {
        "schema_version": INVOCATION_SCHEMA,
        "run_id": run_id,
        "task_id": task_id,
        "task_kind": task_kind,
        "input_path": f"runs/{run_id}/{input_name}",
        "enable_codex": enable_codex,
        "expires_minutes": expires_minutes,
        "timeout_seconds": timeout_seconds,
        "max_output_bytes": max_output_bytes,
    }
    value["invocation_id"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _validate_invocation(value: Any, *, run_id: str, task_id: str) -> dict[str, Any]:
    invocation = _strict_mapping(value, _INVOCATION_FIELDS, "invocation")
    declared = invocation.pop("invocation_id")
    if (
        invocation.get("schema_version") != INVOCATION_SCHEMA
        or invocation.get("run_id") != run_id
        or invocation.get("task_id") != task_id
        or not isinstance(declared, str)
        or declared != sha256_bytes(canonical_json_bytes(invocation))
    ):
        raise ResearchWorkflowError("WORKFLOW_BINDING: invocation commitment differs")
    invocation["invocation_id"] = declared
    expected = _invocation_record(
        run_id=run_id,
        task_id=task_id,
        task_kind=invocation["task_kind"],
        input_name=PurePosixPath(invocation["input_path"]).name,
        enable_codex=invocation["enable_codex"],
        expires_minutes=invocation["expires_minutes"],
        timeout_seconds=invocation["timeout_seconds"],
        max_output_bytes=invocation["max_output_bytes"],
    )
    if invocation != expected:
        raise ResearchWorkflowError("WORKFLOW_BINDING: invocation fields differ")
    return invocation


def _bind_invocation_to_order(
    invocation: Mapping[str, Any], order: Mapping[str, Any]
) -> None:
    issued = _parse_utc(order.get("issued_at"), "order.issued_at")
    expires = _parse_utc(order.get("expires_at"), "order.expires_at")
    if (
        order.get("run_id") != invocation["run_id"]
        or order.get("task_id") != invocation["task_id"]
        or order.get("task_kind") != invocation["task_kind"]
        or not isinstance(order.get("input"), Mapping)
        or order["input"].get("path") != invocation["input_path"]
        or not isinstance(order.get("budget"), Mapping)
        or order["budget"].get("timeout_seconds") != invocation["timeout_seconds"]
        or order["budget"].get("max_output_bytes") != invocation["max_output_bytes"]
        or expires - issued != timedelta(minutes=invocation["expires_minutes"])
    ):
        raise ResearchWorkflowError(
            "WORKFLOW_CONFLICT: invocation and work order differ"
        )


def _bind_invocation_to_result(
    invocation: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    disabled_result = (
        result.get("status") == "DEFERRED_NO_CODEX"
        and result.get("reason_code") == "CODEX_DISABLED"
    )
    if invocation["enable_codex"] is disabled_result:
        raise ResearchWorkflowError("WORKFLOW_BINDING: invocation result mode differs")


def _retry_task_id(task_id: str, order: Mapping[str, Any]) -> str:
    digest = str(order.get("work_order_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        digest = sha256_bytes(canonical_json_bytes(dict(order)))
    suffix = f"-retry-{digest[:12]}"
    return f"{task_id[: 128 - len(suffix)]}{suffix}"


def _write_bytes_once(path: Path, data: bytes) -> None:
    if (
        not isinstance(data, bytes)
        or not data
        or len(data) > _MAX_WORKFLOW_ARTIFACT_BYTES
    ):
        raise ResearchWorkflowError("WORKFLOW_SCHEMA: immutable bytes are invalid")
    if has_link_or_reparse_component(path.parent) or is_link_or_reparse(path):
        raise ResearchWorkflowError(
            "WORKFLOW_PATH: destination contains a link or reparse point"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if has_link_or_reparse_component(path.parent):
        raise ResearchWorkflowError("WORKFLOW_PATH: destination became unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    created = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            if has_link_or_reparse_component(path) or is_link_or_reparse(path):
                raise ResearchWorkflowError(
                    "WORKFLOW_CONFLICT: existing artifact is unsafe"
                ) from None
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                existing_descriptor = os.open(path, flags)
            except OSError as exc:
                raise ResearchWorkflowError(
                    "WORKFLOW_CONFLICT: existing artifact is unreadable"
                ) from exc
            try:
                before = os.fstat(existing_descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_size != len(data):
                    raise ResearchWorkflowError(
                        "WORKFLOW_CONFLICT: existing artifact is invalid"
                    )
                existing = b""
                while len(existing) <= len(data):
                    chunk = os.read(existing_descriptor, len(data) + 1 - len(existing))
                    if not chunk:
                        break
                    existing += chunk
                after = os.fstat(existing_descriptor)
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_size,
                    getattr(before, "st_mtime_ns", None),
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_size,
                    getattr(after, "st_mtime_ns", None),
                )
                if before_identity != after_identity or len(existing) != before.st_size:
                    raise ResearchWorkflowError(
                        "WORKFLOW_CHANGED: existing artifact changed while read"
                    )
            finally:
                os.close(existing_descriptor)
            if existing != data:
                raise ResearchWorkflowError(
                    f"WORKFLOW_CONFLICT: immutable artifact differs: {path.name}"
                ) from None
        if has_link_or_reparse_component(path) or not path.is_file():
            raise ResearchWorkflowError("WORKFLOW_PATH: published artifact is unsafe")
        if created and os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        data = canonical_json_bytes(dict(value))
    except (TypeError, ValueError) as exc:
        raise ResearchWorkflowError(
            "WORKFLOW_SCHEMA: value is not canonical JSON"
        ) from exc
    _write_bytes_once(path, data)


def _draft_bytes(result: Mapping[str, Any], task_kind: str) -> bytes:
    payload = result.get("payload")
    if result.get("status") != "DRAFT_READY" or not isinstance(payload, Mapping):
        raise ResearchWorkflowError(
            "WORKFLOW_DRAFT: only a validated ready result has a draft"
        )

    def literal(value: Any) -> str:
        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\\", "\\\\")
        for character in ("`", "[", "]", "(", ")", "<", ">", "!", "|", "*"):
            text = text.replace(character, f"\\{character}")
        text = re.sub(
            r"(?m)^([ \t]{0,3})(?=[#=~+\-])",
            r"\1\\",
            text,
        )
        text = re.sub(
            r"(?m)^([ \t]{0,3})(\d{1,9})([.)])(?=[ \t]|$)",
            r"\1\2\\\3",
            text,
        )
        return text

    def code(value: Any) -> str:
        return f"`{str(value).replace('`', '?')}`"

    lines = [
        f"# Codex {task_kind.replace('_', ' ').title()}",
        "",
        "> Untrusted, evidence-bounded local draft. It grants no metric, state, capital, or execution authority.",
        "",
        "## Summary",
        "",
        literal(payload["summary"]),
        "",
        "## Claims",
        "",
    ]
    claims = payload["claims"]
    if claims:
        for claim in claims:
            refs = ", ".join(code(ref) for ref in claim["artifact_refs"]) or "none"
            lines.append(
                f"- [{claim['classification']}] {literal(claim['text'])} (evidence: {refs})"
            )
    else:
        lines.append("- None.")
    refs = payload["artifact_refs"]
    lines.extend(
        [
            "",
            "## Cited artifacts",
            "",
            *([f"- {code(ref)}" for ref in refs] if refs else ["- None."]),
            "",
            "## Authority",
            "",
            "- Requested transition: `NONE`",
            "- Human review required: `true`",
            "- Live trading authorized: `false`",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _publish_draft(
    output_root: Path,
    run_id: str,
    task_id: str,
    task_kind: str,
    result: Mapping[str, Any],
) -> Path:
    path = output_root / "codex-drafts" / run_id / f"{task_id}.md"
    _write_bytes_once(path, _draft_bytes(result, task_kind))
    return path


def _validated_run_layout(run_dir: Path) -> tuple[Path, Path]:
    unresolved = run_dir
    if has_link_or_reparse_component(unresolved):
        raise ResearchWorkflowError("WORKFLOW_PATH: run directory is unsafe")
    run_dir = unresolved.resolve(strict=True)
    if run_dir.parent.name != "runs":
        raise ResearchWorkflowError(
            "WORKFLOW_LAYOUT: run directory must be under <output-root>/runs"
        )
    output_root = run_dir.parent.parent
    if has_link_or_reparse_component(output_root):
        raise ResearchWorkflowError("WORKFLOW_PATH: output root is unsafe")
    return run_dir, output_root


def prepare_review_task(
    run_dir: str | Path,
    *,
    task_id: str,
    task_kind: str,
    input_name: str = "report.md",
    expires_minutes: int = 60,
    timeout_seconds: int = 120,
    max_output_bytes: int = 128 * 1024,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Prepare and persist one manifest-bound Codex work order for a verified run."""

    source_run, output_root = _validated_run_layout(Path(run_dir))
    verified = verify_run(source_run)
    if verified["status"] != "HUMAN_REVIEW":
        raise ResearchWorkflowError(
            "WORKFLOW_RUN: only a verified HUMAN_REVIEW run may reach Codex"
        )
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if type(expires_minutes) is not int or not 1 <= expires_minutes <= 24 * 60:
        raise ResearchWorkflowError(
            "WORKFLOW_BUDGET: expires_minutes is outside policy"
        )
    allowed_names = [name for name in _RUN_ARTIFACTS if (source_run / name).is_file()]
    if input_name not in allowed_names:
        raise ResearchWorkflowError(
            "WORKFLOW_INPUT: requested run artifact is unavailable"
        )
    prefix = f"runs/{verified['run_id']}"
    allowed = [f"{prefix}/{name}" for name in allowed_names]
    destination = (
        output_root / "codex-work-orders" / verified["run_id"] / f"{task_id}.json"
    )
    if destination.exists():
        order, _ = _read_canonical_mapping(destination, "work order")
        try:
            validate_work_order(order, output_root, now=current)
        except BrokerValidationError as exc:
            if "expired" not in str(exc).casefold():
                raise ResearchWorkflowError(f"WORKFLOW_BROKER: {exc}") from exc
            try:
                issued = _parse_utc(order.get("issued_at"), "order.issued_at")
                validate_work_order(order, output_root, now=issued)
            except (BrokerValidationError, ResearchWorkflowError) as invalid:
                raise ResearchWorkflowError(
                    f"WORKFLOW_BROKER: persisted work order is invalid: {invalid}"
                ) from invalid
            successor = _retry_task_id(task_id, order)
            raise ResearchWorkflowError(
                "WORKFLOW_RETRY_REQUIRED: prepared work order expired; "
                f"retry with task_id={successor}"
            ) from exc
        issued = _parse_utc(order.get("issued_at"), "order.issued_at")
        expires = _parse_utc(order.get("expires_at"), "order.expires_at")
        if (
            order.get("task_id") != task_id
            or order.get("task_kind") != task_kind
            or order.get("input", {}).get("path") != f"{prefix}/{input_name}"
            or order.get("budget", {}).get("timeout_seconds") != timeout_seconds
            or order.get("budget", {}).get("max_output_bytes") != max_output_bytes
            or expires - issued != timedelta(minutes=expires_minutes)
        ):
            raise ResearchWorkflowError(
                "WORKFLOW_CONFLICT: existing work order differs"
            )
    else:
        order = prepare_work_order(
            output_root,
            task_id=task_id,
            run_id=verified["run_id"],
            task_kind=task_kind,
            input_path=f"{prefix}/{input_name}",
            allowed_paths=allowed,
            expires_at=current + timedelta(minutes=expires_minutes),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            now=current,
        )
        reverified = verify_run(source_run)
        binding = order.get("run_manifest")
        if (
            reverified["run_id"] != verified["run_id"]
            or not isinstance(binding, Mapping)
            or binding.get("sha256")
            != sha256_bytes(
                _read_regular_bytes(source_run / "manifest.json", "run manifest")
            )
        ):
            raise ResearchWorkflowError(
                "WORKFLOW_CHANGED: run changed while the work order was prepared"
            )
        _write_once(destination, order)
    return {
        "status": "PREPARED",
        "run_id": verified["run_id"],
        "task_id": task_id,
        "task_kind": task_kind,
        "work_order": str(destination),
        "workspace": str(output_root),
        "state_transition_authorized": False,
    }


def _descriptor_value(
    verified: Mapping[str, Any],
    task_id: str,
    order: Mapping[str, Any],
    result: Mapping[str, Any],
    invocation: Mapping[str, Any],
    *,
    manifest_sha256: str,
    order_sha256: str,
    result_sha256: str,
    receipt_sha256: str,
    invocation_sha256: str,
    review: Mapping[str, Any] | None,
    bundle_sha256: str | None,
    draft_sha256: str | None,
) -> dict[str, Any]:
    run_id = verified["run_id"]
    review_record: dict[str, Any] | None = None
    if review is not None:
        if bundle_sha256 is None or draft_sha256 is None:
            raise ResearchWorkflowError("WORKFLOW_BINDING: review hashes are missing")
        review_record = {
            "bundle_id": review["bundle_id"],
            "bundle_path": f"codex-reviews/{run_id}/{task_id}/bundle.json",
            "bundle_sha256": bundle_sha256,
            "draft_path": f"codex-drafts/{run_id}/{task_id}.md",
            "draft_sha256": draft_sha256,
        }
    status = (
        "REVIEW_READY"
        if review_record is not None
        else {
            "DEFERRED_NO_CODEX": "CODEX_DEFERRED",
            "QUARANTINED": "CODEX_QUARANTINED",
        }.get(result["status"], "CODEX_INCOMPLETE")
    )
    descriptor: dict[str, Any] = {
        "schema_version": WORKFLOW_SCHEMA,
        "status": status,
        "run": {
            "run_id": run_id,
            "status": verified["status"],
            "evidence_tier": verified["evidence_tier"],
            "manifest_path": f"runs/{run_id}/manifest.json",
            "manifest_sha256": manifest_sha256,
        },
        "task": {
            "task_id": task_id,
            "task_kind": order["task_kind"],
            "work_order_path": f"codex-work-orders/{run_id}/{task_id}.json",
            "work_order_sha256": order_sha256,
            "result_path": f"codex-outbox/{run_id}/{task_id}.json",
            "result_sha256": result_sha256,
            "receipt_path": f"codex-outbox/{run_id}/{task_id}.receipt.json",
            "receipt_sha256": receipt_sha256,
            "result_status": result["status"],
            "reason_code": result["reason_code"],
            "invocation_id": invocation["invocation_id"],
            "invocation_path": f"workflow-invocations/{run_id}/{task_id}.json",
            "invocation_sha256": invocation_sha256,
        },
        "review": review_record,
        "authority": dict(_AUTHORITY),
    }
    descriptor["workflow_id"] = sha256_bytes(canonical_json_bytes(descriptor))
    return descriptor


def _assemble_descriptor(
    output_root: Path,
    run_dir: Path,
    task_id: str,
    order: Mapping[str, Any],
    result: Mapping[str, Any],
    review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    verified = verify_run(run_dir)
    order_path = output_root / "codex-work-orders" / run_dir.name / f"{task_id}.json"
    result_path = output_root / "codex-outbox" / run_dir.name / f"{task_id}.json"
    receipt_path = (
        output_root / "codex-outbox" / run_dir.name / f"{task_id}.receipt.json"
    )
    invocation_path = (
        output_root / "workflow-invocations" / run_dir.name / f"{task_id}.json"
    )
    order_bytes = _read_regular_bytes(order_path, "work order")
    result_bytes = _read_regular_bytes(result_path, "result")
    receipt_bytes = _read_regular_bytes(receipt_path, "receipt")
    invocation, invocation_bytes = _read_canonical_mapping(
        invocation_path, "workflow invocation"
    )
    invocation = _validate_invocation(invocation, run_id=run_dir.name, task_id=task_id)
    _bind_invocation_to_order(invocation, order)
    _bind_invocation_to_result(invocation, result)
    bundle_sha256: str | None = None
    draft_sha256: str | None = None
    if review is not None:
        bundle_dir = output_root / "codex-reviews" / run_dir.name / task_id
        bundle_sha256 = sha256_bytes(
            _read_regular_bytes(bundle_dir / "bundle.json", "review bundle")
        )
        draft_sha256 = sha256_bytes(
            _read_regular_bytes(
                output_root / "codex-drafts" / run_dir.name / f"{task_id}.md",
                "derived draft",
            )
        )
    return _descriptor_value(
        verified,
        task_id,
        order,
        result,
        invocation,
        manifest_sha256=sha256_bytes(
            _read_regular_bytes(run_dir / "manifest.json", "run manifest")
        ),
        order_sha256=sha256_bytes(order_bytes),
        result_sha256=sha256_bytes(result_bytes),
        receipt_sha256=sha256_bytes(receipt_bytes),
        invocation_sha256=sha256_bytes(invocation_bytes),
        review=review,
        bundle_sha256=bundle_sha256,
        draft_sha256=draft_sha256,
    )


def run_research_draft(
    work_order_path: str | Path,
    *,
    workspace: str | Path,
    output_root: str | Path,
    task_id: str,
    task_kind: str = "REPORT_DRAFT",
    input_name: str = "report.md",
    enable_codex: bool = False,
    receipt_key_root: str | Path | None = None,
    expires_minutes: int = 60,
    timeout_seconds: int = 120,
    max_output_bytes: int = 128 * 1024,
    runner: Runner | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one complete research-draft workflow as a process-local single flight."""

    lock_path = _workflow_execution_lock_path(output_root)
    with _execution_lock(lock_path):
        return _run_research_draft(
            work_order_path,
            workspace=workspace,
            output_root=output_root,
            task_id=task_id,
            task_kind=task_kind,
            input_name=input_name,
            enable_codex=enable_codex,
            receipt_key_root=receipt_key_root,
            expires_minutes=expires_minutes,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            runner=runner,
            now=now,
        )


def _run_research_draft(
    work_order_path: str | Path,
    *,
    workspace: str | Path,
    output_root: str | Path,
    task_id: str,
    task_kind: str = "REPORT_DRAFT",
    input_name: str = "report.md",
    enable_codex: bool = False,
    receipt_key_root: str | Path | None = None,
    expires_minutes: int = 60,
    timeout_seconds: int = 120,
    max_output_bytes: int = 128 * 1024,
    runner: Runner | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run, verify, ask Codex, authenticate, ingest, and verify one review draft.

    Codex remains optional. A disabled, unavailable, invalid, or expired model run
    leaves the deterministic research run valid and records a non-promoting
    terminal workflow receipt.
    """

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run = run_price_baseline(
        work_order_path,
        workspace=workspace,
        output_root=output_root,
        now=current,
    )
    run_dir, resolved_output_root = _validated_run_layout(Path(run["run_dir"]))
    invocation = _invocation_record(
        run_id=run_dir.name,
        task_id=task_id,
        task_kind=task_kind,
        input_name=input_name,
        enable_codex=enable_codex,
        expires_minutes=expires_minutes,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    invocation_path = (
        resolved_output_root / "workflow-invocations" / run_dir.name / f"{task_id}.json"
    )
    _write_once(invocation_path, invocation)
    workflow_path = (
        resolved_output_root / "workflows" / run_dir.name / f"{task_id}.json"
    )
    if workflow_path.exists():
        verified = verify_research_draft_workflow(
            workflow_path, receipt_key_root=receipt_key_root
        )
        return {
            **verified,
            "workflow_path": str(workflow_path),
            "idempotent_replay": True,
        }

    existing_order_path = (
        resolved_output_root / "codex-work-orders" / run_dir.name / f"{task_id}.json"
    )
    existing_result_path = (
        resolved_output_root / "codex-outbox" / run_dir.name / f"{task_id}.json"
    )
    existing_receipt_path = existing_result_path.with_name(f"{task_id}.receipt.json")
    if all(
        path.is_file()
        for path in (existing_order_path, existing_result_path, existing_receipt_path)
    ):
        order, _ = _read_canonical_mapping(existing_order_path, "work order")
        result, _ = _read_canonical_mapping(existing_result_path, "result")
        receipt, _ = _read_canonical_mapping(existing_receipt_path, "receipt")
        produced = _parse_utc(result.get("produced_at"), "result.produced_at")
        try:
            order = validate_work_order(order, resolved_output_root, now=produced)
            result = validate_result_envelope(result, order, resolved_output_root)
            validate_result_receipt(
                receipt,
                result,
                order,
                resolved_output_root,
                receipt_key_root=receipt_key_root,
            )
        except BrokerValidationError as exc:
            raise ResearchWorkflowError(f"WORKFLOW_BROKER: {exc}") from exc
        expected_input = f"runs/{run_dir.name}/{input_name}"
        if (
            order["run_id"] != run_dir.name
            or order["task_id"] != task_id
            or order["task_kind"] != task_kind
            or order["input"]["path"] != expected_input
        ):
            raise ResearchWorkflowError(
                "WORKFLOW_CONFLICT: completed broker task differs"
            )
        _bind_invocation_to_order(invocation, order)
        recovered_review: Mapping[str, Any] | None = None
        if result["status"] == "DRAFT_READY":
            recovered_review = ingest_review_bundle(
                run_dir,
                task_id=task_id,
                receipt_key_root=receipt_key_root,
            )
            recovered_review = verify_review_bundle(
                recovered_review["bundle_dir"], receipt_key_root=receipt_key_root
            )
            _publish_draft(
                resolved_output_root,
                run_dir.name,
                task_id,
                task_kind,
                result,
            )
        recovered = _assemble_descriptor(
            resolved_output_root,
            run_dir,
            task_id,
            order,
            result,
            recovered_review,
        )
        _write_once(workflow_path, recovered)
        verified = verify_research_draft_workflow(
            workflow_path, receipt_key_root=receipt_key_root
        )
        return {
            **verified,
            "workflow_path": str(workflow_path),
            "idempotent_replay": True,
            "recovered": True,
        }

    prepared = prepare_review_task(
        run_dir,
        task_id=task_id,
        task_kind=task_kind,
        input_name=input_name,
        expires_minutes=expires_minutes,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        now=current,
    )
    order_path = Path(prepared["work_order"])
    order, _ = _read_canonical_mapping(order_path, "work order")
    _bind_invocation_to_order(invocation, order)
    with _execution_lock(existing_result_path):
        result = run_work_order(
            order,
            resolved_output_root,
            resolved_output_root / "codex-outbox",
            enabled=enable_codex,
            runner=runner,
            now=current,
            receipt_key_root=receipt_key_root,
        )
    review: Mapping[str, Any] | None = None
    if result["status"] == "DRAFT_READY":
        review = ingest_review_bundle(
            run_dir,
            task_id=task_id,
            receipt_key_root=receipt_key_root,
        )
        review = verify_review_bundle(
            review["bundle_dir"], receipt_key_root=receipt_key_root
        )
        _publish_draft(
            resolved_output_root,
            run_dir.name,
            task_id,
            task_kind,
            result,
        )
    descriptor = _assemble_descriptor(
        resolved_output_root, run_dir, task_id, order, result, review
    )
    _write_once(workflow_path, descriptor)
    verified = verify_research_draft_workflow(
        workflow_path, receipt_key_root=receipt_key_root
    )
    return {**verified, "workflow_path": str(workflow_path), "idempotent_replay": False}


def verify_research_draft_workflow(
    workflow_path: str | Path,
    *,
    receipt_key_root: str | Path | None = None,
) -> dict[str, Any]:
    """Offline-verify a composed research/Codex workflow receipt."""

    path = Path(workflow_path)
    if has_link_or_reparse_component(path):
        raise ResearchWorkflowError("WORKFLOW_PATH: workflow receipt is unsafe")
    path = path.resolve(strict=True)
    if path.parent.parent.name != "workflows":
        raise ResearchWorkflowError(
            "WORKFLOW_LAYOUT: receipt must be under workflows/<run-id>"
        )
    output_root = path.parents[2]
    run_id = path.parent.name
    task_id = path.stem
    value, _ = _read_canonical_mapping(path, "workflow receipt")
    descriptor = _strict_mapping(value, _WORKFLOW_FIELDS, "workflow")
    run_record = _strict_mapping(descriptor["run"], _RUN_FIELDS, "workflow.run")
    task_record = _strict_mapping(descriptor["task"], _TASK_FIELDS, "workflow.task")
    if (
        descriptor["schema_version"] != WORKFLOW_SCHEMA
        or descriptor["authority"] != _AUTHORITY
    ):
        raise ResearchWorkflowError("WORKFLOW_SCHEMA: schema or authority is invalid")
    if run_record["run_id"] != run_id or task_record["task_id"] != task_id:
        raise ResearchWorkflowError("WORKFLOW_BINDING: path identity differs")
    run_dir = output_root / "runs" / run_id
    manifest_bytes = _read_regular_bytes(run_dir / "manifest.json", "run manifest")
    verified_run = verify_run(run_dir)
    if (
        verified_run["status"] != run_record["status"]
        or verified_run["evidence_tier"] != run_record["evidence_tier"]
        or run_record["manifest_path"] != f"runs/{run_id}/manifest.json"
        or sha256_bytes(manifest_bytes) != run_record["manifest_sha256"]
    ):
        raise ResearchWorkflowError("WORKFLOW_BINDING: source run differs")

    expected_paths = {
        "work_order_path": f"codex-work-orders/{run_id}/{task_id}.json",
        "result_path": f"codex-outbox/{run_id}/{task_id}.json",
        "receipt_path": f"codex-outbox/{run_id}/{task_id}.receipt.json",
        "invocation_path": f"workflow-invocations/{run_id}/{task_id}.json",
    }
    for field, expected in expected_paths.items():
        if task_record[field] != expected:
            raise ResearchWorkflowError(f"WORKFLOW_BINDING: {field} differs")
    order_path = output_root / PurePosixPath(task_record["work_order_path"])
    result_path = output_root / PurePosixPath(task_record["result_path"])
    receipt_path = output_root / PurePosixPath(task_record["receipt_path"])
    invocation_path = output_root / PurePosixPath(task_record["invocation_path"])
    order, order_bytes = _read_canonical_mapping(order_path, "work order")
    result, result_bytes = _read_canonical_mapping(result_path, "result")
    receipt, receipt_bytes = _read_canonical_mapping(receipt_path, "receipt")
    invocation_value, invocation_bytes = _read_canonical_mapping(
        invocation_path, "workflow invocation"
    )
    invocation = _validate_invocation(invocation_value, run_id=run_id, task_id=task_id)
    produced = _parse_utc(result.get("produced_at"), "result.produced_at")
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
        raise ResearchWorkflowError(f"WORKFLOW_BROKER: {exc}") from exc
    _bind_invocation_to_order(invocation, order)
    _bind_invocation_to_result(invocation, result)
    if (
        order["run_id"] != run_id
        or order["task_id"] != task_id
        or order["task_kind"] != task_record["task_kind"]
        or result["status"] != task_record["result_status"]
        or result["reason_code"] != task_record["reason_code"]
        or sha256_bytes(order_bytes) != task_record["work_order_sha256"]
        or sha256_bytes(result_bytes) != task_record["result_sha256"]
        or sha256_bytes(receipt_bytes) != task_record["receipt_sha256"]
        or task_record["invocation_id"] != invocation["invocation_id"]
        or sha256_bytes(invocation_bytes) != task_record["invocation_sha256"]
    ):
        raise ResearchWorkflowError("WORKFLOW_BINDING: broker artifacts differ")

    review: Mapping[str, Any] | None = None
    bundle_sha256: str | None = None
    draft_sha256: str | None = None
    if descriptor["review"] is not None:
        review_record = _strict_mapping(
            descriptor["review"], _REVIEW_FIELDS, "workflow.review"
        )
        expected_bundle = f"codex-reviews/{run_id}/{task_id}/bundle.json"
        if review_record["bundle_path"] != expected_bundle:
            raise ResearchWorkflowError("WORKFLOW_BINDING: review path differs")
        bundle_path = output_root / PurePosixPath(expected_bundle)
        bundle_bytes = _read_regular_bytes(bundle_path, "review bundle")
        bundle_sha256 = sha256_bytes(bundle_bytes)
        if bundle_sha256 != review_record["bundle_sha256"]:
            raise ResearchWorkflowError("WORKFLOW_BINDING: review hash differs")
        review = verify_review_bundle(
            bundle_path.parent, receipt_key_root=receipt_key_root
        )
        if review["bundle_id"] != review_record["bundle_id"]:
            raise ResearchWorkflowError("WORKFLOW_BINDING: review bundle id differs")
        expected_draft = f"codex-drafts/{run_id}/{task_id}.md"
        if review_record["draft_path"] != expected_draft:
            raise ResearchWorkflowError("WORKFLOW_BINDING: draft path differs")
        draft_path = output_root / PurePosixPath(expected_draft)
        draft_bytes = _read_regular_bytes(draft_path, "derived draft")
        draft_sha256 = sha256_bytes(draft_bytes)
        if draft_sha256 != review_record["draft_sha256"] or draft_bytes != _draft_bytes(
            result, order["task_kind"]
        ):
            raise ResearchWorkflowError("WORKFLOW_BINDING: draft artifact differs")
    elif result["status"] == "DRAFT_READY":
        raise ResearchWorkflowError(
            "WORKFLOW_BINDING: ready draft has no review bundle"
        )

    expected = _descriptor_value(
        verified_run,
        task_id,
        order,
        result,
        invocation,
        manifest_sha256=sha256_bytes(manifest_bytes),
        order_sha256=sha256_bytes(order_bytes),
        result_sha256=sha256_bytes(result_bytes),
        receipt_sha256=sha256_bytes(receipt_bytes),
        invocation_sha256=sha256_bytes(invocation_bytes),
        review=review,
        bundle_sha256=bundle_sha256,
        draft_sha256=draft_sha256,
    )
    if descriptor != expected:
        raise ResearchWorkflowError(
            "WORKFLOW_HASH_MISMATCH: workflow commitment differs"
        )
    return {
        "schema_version": WORKFLOW_SCHEMA,
        "status": descriptor["status"],
        "workflow_id": descriptor["workflow_id"],
        "run_id": run_id,
        "task_id": task_id,
        "task_kind": task_record["task_kind"],
        "evidence_tier": run_record["evidence_tier"],
        "codex_status": result["status"],
        "reason_code": result["reason_code"],
        "review_bundle_id": review["bundle_id"] if review is not None else None,
        "state_transition_authorized": False,
    }


__all__ = [
    "WORKFLOW_SCHEMA",
    "ResearchWorkflowError",
    "prepare_review_task",
    "run_research_draft",
    "verify_research_draft_workflow",
]
