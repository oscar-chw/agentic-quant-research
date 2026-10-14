"""Shared-manifest admission keeps source type, rights, root and digests explicit."""

import hashlib
import json

import pytest

from method_contract import shared_note_citation
from shared_manifest import SharedManifestError, load_shared_note

RIGHTS = {"license": "MIT", "basis": "original_summary", "redistribution": "public"}


@pytest.fixture
def corpus(tmp_path):
    note = b"id: shared-method\ntitle: Method note\n\nPoint-in-time cutoff.\n"
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods/note.md").write_bytes(note)
    row = {"id": "shared-method", "document_type": "method_note", "title": "Method note",
           "path": "methods/note.md", "sha256": hashlib.sha256(note).hexdigest(),
           "source_refs": [{"uri": "https://example.invalid/source", "kind": "public_code", "revision": "abc"}],
           "rights": RIGHTS}
    manifest = {"schema": "quant-shared-notes/v1", "revision": "test", "documents": [row]}
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    return tmp_path, row, hashlib.sha256(raw).hexdigest()


def rewrite_manifest(root, row):
    raw = json.dumps({"schema": "quant-shared-notes/v1", "revision": "test", "documents": [row]},
                     sort_keys=True, separators=(",", ":")).encode()
    (root / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_valid_admission_returns_complete_method_note_envelope(corpus):
    root, row, digest = corpus
    result = load_shared_note(root, row["id"], expected_manifest_sha256=digest)
    envelope = result["matches"][0]
    assert result["status"] == "MATCHES"
    assert envelope["document_id"] == "shared-method"
    assert envelope["document_type"] == "method_note"
    assert envelope["title"] == "Method note"
    assert envelope["document"].endswith("Point-in-time cutoff.\n")
    assert envelope["source_sha256"] == row["sha256"]
    assert envelope["manifest_sha256"] == digest
    assert envelope["source_refs"] == row["source_refs"] and envelope["rights"] == RIGHTS
    citation = shared_note_citation(envelope)
    assert citation["schema"] == "vault-method-citation/v1" and citation["document_type"] == "method_note"
    assert citation["document"] == envelope["document"]


def test_stale_manifest_and_tampered_document_refuse(corpus):
    root, row, digest = corpus
    with pytest.raises(SharedManifestError, match="SHARED_MANIFEST_STALE"):
        load_shared_note(root, row["id"], expected_manifest_sha256="0" * 64)
    (root / row["path"]).write_text("tampered")
    with pytest.raises(SharedManifestError, match="SHARED_DOCUMENT_STALE"):
        load_shared_note(root, row["id"], expected_manifest_sha256=digest)


@pytest.mark.parametrize("field,value,code", [
    ("path", "../outside.md", "SHARED_PATH_REFUSED"),
    ("path", "/tmp/outside.md", "SHARED_PATH_REFUSED"),
    ("rights", {"license": "unknown", "basis": "original_summary", "redistribution": "public"},
     "SHARED_RIGHTS_REFUSED"),
])
def test_escape_and_rights_refusal(corpus, field, value, code):
    root, row, _ = corpus
    row[field] = value
    digest = rewrite_manifest(root, row)
    with pytest.raises(SharedManifestError, match=code):
        load_shared_note(root, row["id"], expected_manifest_sha256=digest)


def test_no_match_is_distinct_from_manifest_read_failure(corpus, tmp_path):
    root, row, digest = corpus
    assert load_shared_note(root, "another-note", expected_manifest_sha256=digest)["status"] == "NO_MATCHES"
    (root / "manifest.json").unlink()
    with pytest.raises(SharedManifestError, match="SHARED_PATH_REFUSED"):
        load_shared_note(root, row["id"], expected_manifest_sha256=digest)


@pytest.mark.parametrize("refs", [[], [{}], ["not a source"],
    [{"uri": "https://example.invalid/source", "kind": "public_code", "revision": ""}]])
def test_missing_source_provenance_is_refused(corpus, refs):
    root, row, _ = corpus
    row["source_refs"] = refs
    digest = rewrite_manifest(root, row)
    with pytest.raises(SharedManifestError, match="SHARED_DOCUMENT_INVALID"):
        load_shared_note(root, row["id"], expected_manifest_sha256=digest)


def test_missing_root_is_a_coded_source_failure(tmp_path):
    with pytest.raises(SharedManifestError, match="SHARED_PATH_REFUSED"):
        load_shared_note(tmp_path / "absent", "some-note", expected_manifest_sha256="0" * 64)


def test_symlink_loop_is_a_coded_source_failure(corpus):
    root, row, _ = corpus
    loop = root / "loop.md"
    loop.symlink_to("loop.md")
    row["path"] = "loop.md"
    digest = rewrite_manifest(root, row)
    with pytest.raises(SharedManifestError, match="SHARED_PATH_REFUSED"):
        load_shared_note(root, row["id"], expected_manifest_sha256=digest)


def test_embedded_nul_path_is_a_coded_source_failure(corpus):
    root, row, _ = corpus
    row["path"] = "invalid\x00.md"
    digest = rewrite_manifest(root, row)
    with pytest.raises(SharedManifestError, match="SHARED_PATH_REFUSED"):
        load_shared_note(root, row["id"], expected_manifest_sha256=digest)
