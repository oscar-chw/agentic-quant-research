"""Atomic, hash-addressed artifact storage for local QuantOS runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class ArtifactError(ValueError):
    """Raised when an artifact write would violate immutability or scope."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _relative_artifact_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactError("artifact path must be a non-empty normalized POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactError("artifact path escapes or is not normalized")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes suitable for hashing."""

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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_immutable(path: Path, data: bytes) -> None:
    """Publish bytes exactly once without a check/replace race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ArtifactError(
                    f"immutable artifact already exists with different content: {path.name}"
                ) from None
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    bytes: int


class ArtifactStore:
    """Write immutable run artifacts and a mutable verified latest pointer."""

    def __init__(self, root: Path, run_id: str) -> None:
        if not run_id or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in run_id
        ):
            raise ArtifactError("run_id must contain only letters, digits, '-' or '_'")
        self.root = root.resolve()
        runs_dir = (self.root / "runs").resolve()
        try:
            runs_dir.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError("runs directory escapes the artifact root") from exc
        runs_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = runs_dir.resolve()
        try:
            runs_dir.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError("runs directory escapes the artifact root") from exc
        self.run_id = run_id
        self.run_dir = (runs_dir / run_id).resolve()
        try:
            self.run_dir.relative_to(runs_dir)
        except ValueError as exc:
            raise ArtifactError("run directory escapes the artifact root") from exc
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.run_dir.resolve() != self.run_dir:
            raise ArtifactError("run directory changed while being created")

    def _target(self, relative_path: str) -> Path:
        normalized = _relative_artifact_path(relative_path)
        candidate = (self.run_dir / normalized).resolve()
        try:
            candidate.relative_to(self.run_dir)
        except ValueError as exc:
            raise ArtifactError("artifact path escapes the run directory") from exc
        return candidate

    def write_bytes(self, relative_path: str, data: bytes) -> ArtifactRecord:
        relative_path = _relative_artifact_path(relative_path)
        target = self._target(relative_path)
        try:
            _atomic_create_immutable(target, data)
        except ArtifactError as exc:
            raise ArtifactError(
                f"immutable artifact already exists with different content: {relative_path}"
            ) from exc
        return ArtifactRecord(
            relative_path.replace("\\", "/"), sha256_bytes(data), len(data)
        )

    def write_json(self, relative_path: str, value: Any) -> ArtifactRecord:
        try:
            data = canonical_json_bytes(value)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("artifact is not finite canonical JSON") from exc
        return self.write_bytes(relative_path, data)

    def write_text(self, relative_path: str, value: str) -> ArtifactRecord:
        data = value.encode("utf-8")
        if not data.endswith(b"\n"):
            data += b"\n"
        return self.write_bytes(relative_path, data)

    def commit_manifest(
        self,
        records: list[ArtifactRecord],
        metadata: dict[str, Any],
    ) -> ArtifactRecord:
        if not isinstance(metadata, dict):
            raise ArtifactError("manifest metadata must be a JSON object")
        try:
            canonical_json_bytes(metadata)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                "manifest metadata is not finite canonical JSON"
            ) from exc

        declared: dict[str, ArtifactRecord] = {}
        for record in records:
            if not isinstance(record, ArtifactRecord):
                raise ArtifactError("manifest records must be ArtifactRecord values")
            relative = _relative_artifact_path(record.path)
            if relative == "manifest.json" or relative in declared:
                raise ArtifactError("manifest artifact record is invalid or duplicated")
            if (
                not isinstance(record.sha256, str)
                or not _SHA256.fullmatch(record.sha256)
                or isinstance(record.bytes, bool)
                or not isinstance(record.bytes, int)
                or record.bytes < 0
            ):
                raise ArtifactError("manifest artifact record is invalid or duplicated")
            target = self._target(relative)
            if target.is_symlink() or not target.is_file():
                raise ArtifactError(f"missing artifact: {relative}")
            if (
                target.stat().st_size != record.bytes
                or sha256_file(target) != record.sha256
            ):
                raise ArtifactError(f"artifact verification failed: {relative}")
            declared[relative] = record

        actual = {
            path.relative_to(self.run_dir).as_posix()
            for path in self.run_dir.rglob("*")
            if path.is_file()
            and path.relative_to(self.run_dir).as_posix() != "manifest.json"
        }
        if set(declared) != actual:
            missing = sorted(actual - set(declared))
            stale = sorted(set(declared) - actual)
            name = (missing or stale)[0]
            raise ArtifactError(
                f"manifest does not cover the complete run bundle: {name}"
            )

        ordered = [declared[path].__dict__ for path in sorted(declared)]
        manifest = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "metadata": metadata,
            "artifacts": ordered,
        }
        return self.write_json("manifest.json", manifest)

    def publish_latest(self, manifest: ArtifactRecord) -> None:
        if manifest.path != "manifest.json":
            raise ArtifactError("latest pointer requires the run manifest record")
        path = self.run_dir / manifest.path
        if not path.is_file() or sha256_file(path) != manifest.sha256:
            raise ArtifactError("cannot publish an unverified manifest")
        verified = verify_manifest(self.run_dir)
        if verified["run_id"] != self.run_id:
            raise ArtifactError("cannot publish a manifest for another run")
        latest = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "manifest_path": f"runs/{self.run_id}/manifest.json",
            "manifest_sha256": manifest.sha256,
        }
        _atomic_replace(self.root / "latest.json", canonical_json_bytes(latest))


def verify_manifest(run_dir: Path) -> dict[str, Any]:
    """Verify every artifact named by a run manifest."""

    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ArtifactError(f"duplicate manifest key: {key}")
            value[key] = item
        return value

    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(
            raw_manifest.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except ArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("manifest is missing or invalid JSON") from exc
    try:
        canonical_manifest = canonical_json_bytes(manifest)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("manifest is not finite canonical JSON") from exc
    if not isinstance(manifest, dict) or raw_manifest != canonical_manifest:
        raise ArtifactError("manifest must be canonical JSON")
    if set(manifest) != {"schema_version", "run_id", "metadata", "artifacts"}:
        raise ArtifactError("manifest schema is invalid")
    if (
        manifest.get("schema_version") != "1.0"
        or not isinstance(manifest.get("metadata"), dict)
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise ArtifactError("manifest schema is invalid")
    if manifest.get("run_id") != run_dir.name:
        raise ArtifactError("manifest run_id does not match directory")
    declared: set[str] = set()
    declared_order: list[str] = []
    for item in manifest.get("artifacts", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ArtifactError("manifest artifact record is invalid or duplicated")
        relative = _relative_artifact_path(item["path"])
        if (
            relative == "manifest.json"
            or relative in declared
            or not isinstance(item["sha256"], str)
            or not _SHA256.fullmatch(item["sha256"])
            or isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 0
        ):
            raise ArtifactError("manifest artifact record is invalid or duplicated")
        declared.add(relative)
        declared_order.append(relative)
        unresolved_path = run_dir / relative
        path = unresolved_path.resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise ArtifactError("manifest artifact escapes run directory") from exc
        if unresolved_path.is_symlink() or not path.is_file():
            raise ArtifactError(f"missing artifact: {relative}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ArtifactError(f"artifact verification failed: {relative}")
    if declared_order != sorted(declared_order):
        raise ArtifactError("manifest artifact records are not sorted")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    extras = sorted(actual - declared - {"manifest.json"})
    if extras:
        raise ArtifactError(f"unmanifested artifact present: {extras[0]}")
    return manifest
