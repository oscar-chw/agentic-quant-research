"""Eligibility and error semantics for local paper-note evidence.

Only standard-library dependencies. A readable, digest-matched note is eligible
context; this is not proof of the original paper's claims or retrieval quality.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

SCHEMA_VERSION = "vault.source-query.v1"
MAX_NOTE_BYTES = 2 * 1024 * 1024
MAX_CANDIDATES = 128


class SourceAccessError(RuntimeError):
    """An unavailable/invalid backend or incomplete query, never zero matches."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def inspect_note(metadata: dict, document: str | None = None, *, source_root=None) -> dict:
    """Return an eligibility decision and the exact checked local text snapshot."""
    def excluded(reason):
        return {"eligible": False, "reason": reason}

    if not isinstance(metadata, dict):
        return excluded("INVALID_METADATA")
    if metadata.get("document_type") != "paper_note":
        return excluded("NOT_PAPER_NOTE")
    identity = metadata.get("arxiv_id")
    if not isinstance(identity, str) or not identity or identity.startswith("guideline::"):
        return excluded("INVALID_IDENTITY")
    if metadata.get("source_available") is not True or metadata.get("source_status") != "present":
        return excluded("SOURCE_UNAVAILABLE_OR_UNKNOWN")
    expected = metadata.get("source_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        return excluded("MISSING_SOURCE_DIGEST")
    raw_path = metadata.get("vault_path")
    if not isinstance(raw_path, str) or not raw_path:
        return excluded("MISSING_SOURCE_PATH")
    path = Path(raw_path)
    if source_root is not None and not path.resolve().is_relative_to(Path(source_root).resolve()):
        return excluded("OUTSIDE_SOURCE_ROOT")
    if path.is_symlink() or not path.is_file():
        return excluded("MISSING_OR_NONREGULAR_SOURCE")
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_NOTE_BYTES + 1)
    except FileNotFoundError:
        return excluded("MISSING_OR_NONREGULAR_SOURCE")
    except OSError as exc:
        raise SourceAccessError("SOURCE_READ_FAILED", "cannot read a candidate note") from exc
    if len(data) > MAX_NOTE_BYTES:
        return excluded("SOURCE_TOO_LARGE")
    if hashlib.sha256(data).hexdigest() != expected:
        return excluded("SOURCE_CHANGED_REINDEX_REQUIRED")
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        return excluded("SOURCE_NOT_UTF8")
    if document is not None and (not isinstance(document, str) or document != text):
        return excluded("INDEX_DOCUMENT_DIFFERS")
    return {"eligible": True, "reason": "CURRENT_PAPER_NOTE", "document": text,
            "source_sha256": expected, "evidence_scope": "local paper note; original claims unverified"}


def select_note_sources(records: list[dict], *, source_root=None) -> dict:
    """Filter explicit records, retaining exclusion reasons and checked snapshots."""
    matches, excluded = [], []
    for record in records:
        metadata = record.get("metadata")
        decision = inspect_note(metadata, record.get("document"), source_root=source_root)
        if decision["eligible"]:
            matches.append({**record, "metadata": dict(metadata), "document": decision["document"],
                            "eligibility": {k: v for k, v in decision.items() if k != "document"}})
        else:
            excluded.append({"source_id": metadata.get("arxiv_id") if isinstance(metadata, dict) else None,
                             "reason": decision["reason"]})
    return {"schema_version": SCHEMA_VERSION, "status": "MATCHES" if matches else "NO_MATCHES",
            "matches": matches, "excluded": excluded, "scanned": len(records)}


def query_note_sources(collection, query: str, n_results: int = 8, *, source_root=None) -> dict:
    """Query a bounded ranked window; failure and incomplete no-match are explicit.

    Optional backends need only count/query. Overfetch preserves eligible lower
    ranks when stale results lead; no real-backend relevance claim is implied.
    """
    if not isinstance(query, str) or not query.strip() or type(n_results) is not int or not 1 <= n_results <= MAX_CANDIDATES:
        raise ValueError("query must be nonempty and n_results must be in 1..128")
    try:
        count = collection.count()
        if type(count) is not int or count < 0:
            raise ValueError("invalid count")
        requested = min(count, max(n_results, min(MAX_CANDIDATES, n_results * 4)))
        if count == 0:
            return {**select_note_sources([]), "index_count": 0, "candidate_limit": 0}
        result = collection.query(query_texts=[query], n_results=requested,
                                  include=["documents", "metadatas", "distances"])
    except Exception as exc:
        raise SourceAccessError("SOURCE_BACKEND_FAILED", "note query failed; no match claim was made") from exc
    try:
        groups = [result[key] for key in ("documents", "metadatas", "distances")]
        if any(not isinstance(group, list) or len(group) != 1 or not isinstance(group[0], list) for group in groups):
            raise ValueError("invalid result groups")
        documents, metadatas, distances = [group[0] for group in groups]
        if not len(documents) == len(metadatas) == len(distances) == requested:
            raise ValueError("truncated or inconsistent query rows")
        if any(not isinstance(doc, str) for doc in documents):
            raise ValueError("invalid documents")
        if any(type(d) not in (int, float) or not math.isfinite(d) for d in distances):
            raise ValueError("invalid distances")
        records = [{"metadata": meta, "document": doc, "distance": distance}
                   for doc, meta, distance in zip(documents, metadatas, distances, strict=True)]
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceAccessError("SOURCE_BACKEND_INVALID", "note query returned invalid rows") from exc
    filtered = select_note_sources(records, source_root=source_root)
    if len(filtered["matches"]) < n_results and requested < count:
        if not filtered["matches"]:
            raise SourceAccessError("SOURCE_QUERY_INCOMPLETE", "candidate limit reached; eligible matches may exist beyond the window")
        filtered["status"] = "PARTIAL"
    filtered["matches"] = filtered["matches"][:n_results]
    return {**filtered, "index_count": count, "candidate_limit": requested}


def current_note(metadata: dict, *, source_root=None) -> dict | None:
    """Snapshot a current SQLite-owned note (not a cached vector search result).

    SQLite processed rows have no stored source digest. Hash the current file
    and apply the same eligibility policy; no indexed-version claim is made.
    """
    path_value = metadata.get("vault_path")
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if source_root is not None and not path.resolve().is_relative_to(Path(source_root).resolve()):
        return None
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_NOTE_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SourceAccessError("SOURCE_READ_FAILED", "cannot snapshot local note") from exc
    if len(data) > MAX_NOTE_BYTES:
        return None
    meta = {"document_type": "paper_note", "source_available": True, "source_status": "present",
            **metadata, "source_sha256": hashlib.sha256(data).hexdigest()}
    selected = select_note_sources([{"metadata": meta}], source_root=source_root)
    return selected["matches"][0] if selected["matches"] else None
