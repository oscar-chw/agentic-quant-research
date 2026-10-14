"""Read-only consumer for the canonical quant-shared-notes/v1 corpus."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

MAX_MANIFEST_BYTES = 256 * 1024
MAX_DOCUMENT_BYTES = 256 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")


class SharedManifestError(ValueError):
    """Invalid, stale, or inaccessible shared-manifest evidence."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _read_regular(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SharedManifestError("SHARED_PATH_REFUSED", "expected an unlinked regular file")
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise SharedManifestError("SHARED_READ_FAILED", "cannot read shared source") from exc
    if len(raw) > limit:
        raise SharedManifestError("SHARED_SIZE_REFUSED", "shared source exceeds its size limit")
    return raw


def load_shared_note(root: str | Path, document_id: str, *, expected_manifest_sha256: str) -> dict:
    """Admit one explicitly selected note against a pinned manifest and root."""
    if not isinstance(document_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", document_id):
        raise SharedManifestError("SHARED_ID_REFUSED", "invalid document id")
    if not isinstance(expected_manifest_sha256, str) or not _DIGEST.fullmatch(expected_manifest_sha256):
        raise SharedManifestError("SHARED_DIGEST_REFUSED", "expected manifest digest is required")
    try:
        root = Path(root).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise SharedManifestError("SHARED_PATH_REFUSED", "shared root is unavailable") from None
    if not root.is_dir():
        raise SharedManifestError("SHARED_PATH_REFUSED", "shared root must be a directory")
    manifest_path = root / "manifest.json"
    raw = _read_regular(manifest_path, MAX_MANIFEST_BYTES)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise SharedManifestError("SHARED_MANIFEST_STALE", "manifest digest differs from the pinned digest")
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SharedManifestError("SHARED_MANIFEST_INVALID", "manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "revision", "documents"} \
            or manifest["schema"] != "quant-shared-notes/v1" or not isinstance(manifest["documents"], list):
        raise SharedManifestError("SHARED_MANIFEST_INVALID", "expected quant-shared-notes/v1 manifest")
    rows = [row for row in manifest["documents"] if isinstance(row, dict) and row.get("id") == document_id]
    if not rows:
        return {"schema": "quant.shared-note/v1", "status": "NO_MATCHES", "matches": []}
    if len(rows) != 1:
        raise SharedManifestError("SHARED_MANIFEST_INVALID", "duplicate document id")
    row = rows[0]
    required = {"id", "document_type", "title", "path", "sha256", "source_refs", "rights"}
    if set(row) != required or row["document_type"] != "method_note":
        raise SharedManifestError("SHARED_DOCUMENT_INVALID", "expected exact method-note fields")
    if not isinstance(row["title"], str) or not row["title"].strip() or not isinstance(row["source_refs"], list):
        raise SharedManifestError("SHARED_DOCUMENT_INVALID", "title and source references are required")
    if not row["source_refs"] or any(
        not isinstance(ref, dict) or any(
            not isinstance(ref.get(key), str) or not ref[key].strip()
            for key in ("uri", "kind", "revision")
        ) for ref in row["source_refs"]
    ):
        raise SharedManifestError("SHARED_DOCUMENT_INVALID", "complete source provenance is required")
    if row["rights"] != {"license": "MIT", "basis": "original_summary", "redistribution": "public"}:
        raise SharedManifestError("SHARED_RIGHTS_REFUSED", "manifest does not grant the required redistribution rights")
    if not isinstance(row["sha256"], str) or not _DIGEST.fullmatch(row["sha256"]):
        raise SharedManifestError("SHARED_DOCUMENT_INVALID", "invalid document digest")
    if not isinstance(row["path"], str):
        raise SharedManifestError("SHARED_PATH_REFUSED", "document path must be text")
    rel = Path(row["path"])
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise SharedManifestError("SHARED_PATH_REFUSED", "document path must remain within the shared root")
    path = root / rel
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise SharedManifestError("SHARED_PATH_REFUSED", "document path cannot be resolved") from None
    if not resolved.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p != root):
        raise SharedManifestError("SHARED_PATH_REFUSED", "document path escapes the shared root or uses a symlink")
    data = _read_regular(path, MAX_DOCUMENT_BYTES)
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise SharedManifestError("SHARED_DOCUMENT_STALE", "document digest differs from the manifest")
    try:
        document = data.decode("utf-8")
    except UnicodeError as exc:
        raise SharedManifestError("SHARED_DOCUMENT_INVALID", "document is not UTF-8") from exc
    envelope = {key: row[key] for key in required}
    envelope.update(document_id=row["id"], source_sha256=row["sha256"], document=document,
                    manifest_sha256=manifest_sha256)
    return {"schema": "quant.shared-note/v1", "status": "MATCHES", "matches": [envelope]}
