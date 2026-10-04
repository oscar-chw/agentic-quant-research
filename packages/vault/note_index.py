"""Standard-library lexical index over a directory of Markdown paper notes.

It implements the count/query backend shape that
``source_access.query_note_sources`` expects, so offline retrieval passes the
same eligibility and digest checks as the optional vector backends. Ranking is
term overlap only; no retrieval-quality claim is made.

A note starts with ``id: <document id>`` and ``title: <title>`` lines, then a
blank line, then free text.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

MAX_NOTES = 512
_TERM = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    return set(_TERM.findall(text.lower()))


def _header(text: str, path: Path) -> dict:
    fields = {}
    for line in text.splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if not sep or key.strip() not in {"id", "title"}:
            raise ValueError(f"{path.name}: header lines must be 'id:' or 'title:'")
        fields[key.strip()] = value.strip()
    if set(fields) != {"id", "title"} or not all(fields.values()):
        raise ValueError(f"{path.name}: note needs an id and a title")
    return fields


class NoteIndex:
    """Snapshot of ``*.md`` notes; digests are taken now and rechecked per query."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        paths = sorted(self.root.glob("*.md"))
        if len(paths) > MAX_NOTES:
            raise ValueError("note directory exceeds the supported note count")
        self.rows = []
        seen = set()
        for path in paths:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            header = _header(text, path)
            if header["id"] in seen:
                raise ValueError("duplicate note id: " + header["id"])
            seen.add(header["id"])
            self.rows.append({
                "document": text,
                "terms": _terms(text),
                "metadata": {
                    "arxiv_id": header["id"],
                    "title": header["title"],
                    "document_type": "paper_note",
                    "source_available": True,
                    "source_status": "present",
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "vault_path": str(path),
                },
            })

    def count(self) -> int:
        return len(self.rows)

    def query(self, *, query_texts, n_results, include):
        (query,) = query_texts
        wanted = _terms(query)
        if not wanted:
            raise ValueError("query has no searchable terms")
        # Distance is the share of query terms a note lacks; ties break by id so
        # the ranking is reproducible.
        ranked = sorted(
            ((1 - len(wanted & row["terms"]) / len(wanted), row) for row in self.rows),
            key=lambda pair: (pair[0], pair[1]["metadata"]["arxiv_id"]),
        )[:n_results]
        return {
            "documents": [[row["document"] for _, row in ranked]],
            "metadatas": [[dict(row["metadata"]) for _, row in ranked]],
            "distances": [[distance for distance, _ in ranked]],
        }
