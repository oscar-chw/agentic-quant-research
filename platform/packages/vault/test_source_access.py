"""Independent eligibility examples and optional adapters under safe API fakes."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from source_access import (
    SourceAccessError,
    current_note,
    inspect_note,
    query_note_sources,
    select_note_sources,
)


class Collection:
    def __init__(self, records=(), *, count=None, failure=None):
        self.records = list(records)
        self.total = len(self.records) if count is None else count
        self.failure = failure
        self.calls = []

    def count(self):
        if self.failure == "count":
            raise RuntimeError("private backend details")
        return self.total

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure == "query":
            raise RuntimeError("private backend details")
        rows = self.records[:kwargs["n_results"]]
        return {key: [[r[field] for r in rows]] for key, field in
                (("documents", "document"), ("metadatas", "metadata"), ("distances", "distance"))}


@pytest.fixture
def note(tmp_path):
    text = "An eligible local reading note. Crypto: 4/5\n"
    path = tmp_path / "note.md"
    path.write_text(text)
    return {"document": text, "distance": 0.125, "metadata": {
        "arxiv_id": "synthetic:paper", "title": "Eligible note", "categories": "q-fin.ST",
        "published": "2024-01-01", "vault_path": str(path), "document_type": "paper_note",
        "source_available": True, "source_status": "present",
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }}


def changed(note, **fields):
    return {**note, "metadata": {**note["metadata"], **fields}}


@pytest.mark.parametrize(("fields", "reason"), [
    ({"document_type": "derived_guideline"}, "NOT_PAPER_NOTE"),
    ({"source_available": False}, "SOURCE_UNAVAILABLE_OR_UNKNOWN"),
    ({"source_available": "true"}, "SOURCE_UNAVAILABLE_OR_UNKNOWN"),
    ({"source_status": "unlisted"}, "SOURCE_UNAVAILABLE_OR_UNKNOWN"),
    ({"source_sha256": None}, "MISSING_SOURCE_DIGEST"),
    ({"arxiv_id": "guideline::foo"}, "INVALID_IDENTITY"),
    ({"vault_path": ""}, "MISSING_SOURCE_PATH"),
])
def test_ineligible_metadata(note, fields, reason):
    result = select_note_sources([changed(note, **fields)])
    assert result["status"] == "NO_MATCHES"
    assert result["excluded"][0]["reason"] == reason


def test_current_index_snapshot_and_rank_preserved(note, tmp_path):
    bad = changed(note, document_type="derived_guideline")
    result = query_note_sources(Collection([bad, note]), "costs", 1, source_root=tmp_path)
    assert result["status"] == "MATCHES"
    assert result["matches"][0]["document"] == note["document"]
    assert result["matches"][0]["distance"] == 0.125
    assert result["scanned"] == 2
    assert result["matches"][0]["eligibility"]["source_sha256"] == note["metadata"]["source_sha256"]


def test_deleted_edited_and_wrong_index_text(note):
    path = Path(note["metadata"]["vault_path"])
    assert inspect_note(note["metadata"], "wrong")["reason"] == "INDEX_DOCUMENT_DIFFERS"
    path.write_text("Edited after indexing")
    assert inspect_note(note["metadata"])["reason"] == "SOURCE_CHANGED_REINDEX_REQUIRED"
    # SQLite-owned exact reads deliberately snapshot the current local version.
    assert current_note(note["metadata"])["document"] == "Edited after indexing"
    path.unlink()
    assert inspect_note(note["metadata"])["reason"] == "MISSING_OR_NONREGULAR_SOURCE"
    assert current_note(note["metadata"]) is None


def test_root_and_symlink_boundaries(note, tmp_path):
    assert inspect_note(note["metadata"], source_root=tmp_path / "elsewhere")["reason"] == "OUTSIDE_SOURCE_ROOT"
    link = tmp_path / "link.md"
    link.symlink_to(note["metadata"]["vault_path"])
    assert inspect_note(changed(note, vault_path=str(link))["metadata"])["reason"] == "MISSING_OR_NONREGULAR_SOURCE"


@pytest.mark.parametrize("failure", ["count", "query"])
def test_backend_failure_is_never_empty_success(note, failure):
    with pytest.raises(SourceAccessError, match="SOURCE_BACKEND_FAILED") as error:
        query_note_sources(Collection([note], failure=failure), "costs")
    assert "private backend details" not in str(error.value)


@pytest.mark.parametrize("broken", [{}, {"documents": [[]], "metadatas": [[{}]], "distances": [[0.1]]},
                                    {"documents": [[None]], "metadatas": [[{}]], "distances": [[0.1]]},
                                    {"documents": [["x"]], "metadatas": [[{}]], "distances": [[float("nan")]]}])
def test_invalid_backend_shape_raises(note, broken):
    collection = Collection([note])
    collection.query = Mock(return_value=broken)
    with pytest.raises(SourceAccessError, match="SOURCE_BACKEND_INVALID"):
        query_note_sources(collection, "costs")


def test_true_zero_and_complete_exclusion(note):
    collection = Collection()
    assert query_note_sources(collection, "costs")["status"] == "NO_MATCHES"
    assert collection.calls == []
    assert query_note_sources(Collection([changed(note, source_available=False)]), "costs")["status"] == "NO_MATCHES"


def test_bounded_window_does_not_claim_no_matches(note):
    excluded = changed(note, source_available=False)
    with pytest.raises(SourceAccessError, match="SOURCE_QUERY_INCOMPLETE"):
        query_note_sources(Collection([excluded] * 8, count=100), "costs", 2)
    result = query_note_sources(Collection([note] + [excluded] * 7, count=100), "costs", 2)
    assert result["status"] == "PARTIAL"
    assert len(result["matches"]) == 1
    assert result["candidate_limit"] == 8


class Server:
    def __init__(self, *args):
        self.callback = None

    def list_tools(self):
        return lambda fn: fn

    def call_tool(self):
        def register(fn):
            self.callback = fn
            return fn
        return register


@pytest.fixture
def adapter():
    chroma = types.ModuleType("chromadb")
    chroma.Collection = Collection
    chroma.PersistentClient = Mock(side_effect=AssertionError("real backend forbidden"))
    mcp = types.ModuleType("mcp")
    mcp.types = types.SimpleNamespace(TextContent=lambda **kw: types.SimpleNamespace(**kw), Tool=object)
    server = types.ModuleType("mcp.server")
    server.Server = Server
    stdio = types.ModuleType("mcp.server.stdio")
    stdio.stdio_server = Mock(side_effect=AssertionError("real server forbidden"))
    spec = importlib.util.spec_from_file_location("safe_search_mcp", Path(__file__).with_name("search_mcp.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"chromadb": chroma, "yaml": types.ModuleType("yaml"),
                                "mcp": mcp, "mcp.server": server, "mcp.server.stdio": stdio}):
        spec.loader.exec_module(module)
    return module


def test_adapter_search_and_error_contract(adapter, note, tmp_path):
    module = adapter
    collection = Collection([changed(note, source_available=False), changed(note, document_type="derived_guideline"), note])

    def search():
        return module.search_papers(collection, "costs", source_root=tmp_path)
    results = search()
    assert [r["arxiv_id"] for r in results] == ["synthetic:paper"]
    assert results[0]["relevance_score"] == 0.875
    assert results[0]["query_status"] == "MATCHES"
    collection.failure = "query"
    with pytest.raises(SourceAccessError):
        search()


def test_adapter_context_and_error_contract(adapter, note, tmp_path):
    module = adapter
    cfg = {"vault_path": str(tmp_path)}
    collection = Collection([changed(note, source_available=False), note])
    context = module.generate_alpha_ideas_text(collection, cfg, "costs")
    assert context.count(note["document"].strip()) == 1
    collection.records = []
    collection.total = 0
    assert "NO_MATCHES" in module.generate_alpha_ideas_text(collection, cfg, "costs")
    collection.failure = "count"
    with pytest.raises(SourceAccessError):
        module.generate_alpha_ideas_text(collection, cfg, "costs")


def test_mcp_dispatch_preserves_error_and_exact_reads(adapter, note, tmp_path):
    module = adapter
    cfg = {"vault_path": str(tmp_path)}
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE papers (arxiv_id, title, categories, published, vault_path, processed, fetched_at)")
    conn.execute("INSERT INTO papers VALUES (?, ?, ?, ?, ?, 1, '2099-01-01')",
                 ("synthetic:paper", "Eligible note", '[]', "2024-01-01", note["metadata"]["vault_path"]))
    module.get_db = lambda cfg: conn
    module.load_config = lambda: cfg
    collection = Collection([note], failure="query")
    module.get_collection = lambda cfg: collection
    server = module.build_server()
    with pytest.raises(SourceAccessError):
        asyncio.run(server.callback("search_papers", {"query": "costs"}))
    assert len(module.list_recent_papers(conn, source_root=tmp_path)) == 1
    Path(note["metadata"]["vault_path"]).unlink()
    assert module.list_recent_papers(conn, source_root=tmp_path) == []
    assert module.get_paper_summary(cfg, "synthetic:paper") is None
