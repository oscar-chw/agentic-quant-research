"""Offline SQLite ingestion, deduplication and metadata lookup with synthetic papers."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from paper_store import count_total, get_unprocessed, init_db, save_paper


def run_demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    papers = [
        {
            "arxiv_id": "synthetic:market-data",
            "title": "Synthetic order book data quality",
            "authors": ["Demo Author"],
            "abstract": "A fictional study of gaps and event-time ordering.",
            "categories": ["q-fin.TR"],
            "published": "2024-01-03T00:00:00+00:00",
            "pdf_url": "https://example.invalid/market-data",
        },
        {
            "arxiv_id": "synthetic:validation",
            "title": "Synthetic chronological validation",
            "authors": ["Demo Author"],
            "abstract": "A fictional study of time splits and research leakage.",
            "categories": ["q-fin.ST"],
            "published": "2024-01-02T00:00:00+00:00",
            "pdf_url": "https://example.invalid/validation",
        },
        {
            "arxiv_id": "synthetic:costs",
            "title": "Synthetic transaction costs",
            "authors": ["Demo Author"],
            "abstract": "A fictional study of spread and turnover.",
            "categories": ["q-fin.TR"],
            "published": "2024-01-01T00:00:00+00:00",
            "pdf_url": "https://example.invalid/costs",
        },
    ]
    db = output / "papers.sqlite"
    conn = init_db(str(db))
    try:
        for paper in papers + papers:
            save_paper(conn, paper)
        count = count_total(conn)
        pending = get_unprocessed(conn)
        if count != 3 or [p["arxiv_id"] for p in pending] != [
            p["arxiv_id"] for p in papers
        ]:
            raise RuntimeError(
                "Ingestion deduplication or chronological ordering failed"
            )
    finally:
        conn.close()
    # Exercise the persisted database again, rather than only the in-memory objects.
    conn = init_db(str(db))
    try:
        matches = conn.execute(
            "SELECT arxiv_id, title FROM papers WHERE categories LIKE ? ORDER BY published DESC",
            ('%"q-fin.TR"%',),
        ).fetchall()
        if [row[0] for row in matches] != ["synthetic:market-data", "synthetic:costs"]:
            raise RuntimeError("Persisted metadata lookup returned unexpected rows")
    finally:
        conn.close()
    summary = {
        "data": "three fictional metadata records; no downloaded papers",
        "insert_attempts": 6,
        "unique_rows": count,
        "duplicate_rows_ignored": 3,
        "lookup": "SQLite category filter, not semantic search",
        "q_fin_tr_matches": [{"id": r[0], "title": r[1]} for r in matches],
        "pending_records": len(pending),
        "reopen_verified": True,
    }
    (output / "demo-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="New directory for retained artifacts"
    )
    args = parser.parse_args()
    if args.output:
        summary = run_demo(args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="vault-demo-") as temp:
            summary = run_demo(Path(temp) / "demo")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
