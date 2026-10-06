"""The offline lexical backend through source_access's eligibility contract."""

import pytest

from note_index import NoteIndex
from source_access import query_note_sources


def note(folder, name, body, title="A note"):
    (folder / f"{name}.md").write_text(f"id: {name}\ntitle: {title}\n\n{body}\n")


def test_ranking_counts_query_terms_and_breaks_ties_by_id(tmp_path):
    note(tmp_path, "b-momentum", "momentum winners losers")
    note(tmp_path, "a-momentum", "momentum winners losers")
    note(tmp_path, "reversal", "reversal losers")
    note(tmp_path, "book", "order flow imbalance")
    result = query_note_sources(NoteIndex(tmp_path), "momentum winners", 3, source_root=tmp_path)
    assert [m["metadata"]["arxiv_id"] for m in result["matches"]] == ["a-momentum", "b-momentum", "book"]
    assert [m["distance"] for m in result["matches"]] == [0.0, 0.0, 1.0]


def test_a_note_edited_after_indexing_is_excluded(tmp_path):
    note(tmp_path, "edited", "momentum")
    note(tmp_path, "kept", "momentum")
    index = NoteIndex(tmp_path)
    (tmp_path / "edited.md").write_text("id: edited\ntitle: A note\n\nmomentum, revised\n")
    result = query_note_sources(index, "momentum", 2, source_root=tmp_path)
    assert [m["metadata"]["arxiv_id"] for m in result["matches"]] == ["kept"]
    assert result["excluded"] == [{"source_id": "edited", "reason": "SOURCE_CHANGED_REINDEX_REQUIRED"}]


@pytest.mark.parametrize("text", ["title: no id\n\nbody\n", "id: x\nauthor: y\n\nbody\n", "id:\ntitle: t\n\nbody\n"])
def test_header_must_name_an_id_and_title(tmp_path, text):
    (tmp_path / "bad.md").write_text(text)
    with pytest.raises(ValueError):
        NoteIndex(tmp_path)


def test_duplicate_ids_are_refused(tmp_path):
    note(tmp_path, "one", "x")
    (tmp_path / "two.md").write_text("id: one\ntitle: t\n\nx\n")
    with pytest.raises(ValueError, match="duplicate"):
        NoteIndex(tmp_path)
