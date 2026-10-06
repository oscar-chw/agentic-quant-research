"""SQLite paper metadata store, shared by online ingestion and the offline demo.

Extracted from fetch.py without changing its schema or insert behavior.
No network, vector model, or enrichment dependencies are imported here.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id    TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            authors     TEXT NOT NULL,
            abstract    TEXT NOT NULL,
            categories  TEXT NOT NULL,
            published   TEXT NOT NULL,
            pdf_url     TEXT NOT NULL,
            fetched_at  TEXT NOT NULL,
            processed   INTEGER DEFAULT 0,
            vault_path  TEXT
        )
    """)
    conn.commit()
    return conn


def already_fetched(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    return row is not None


def save_paper(conn: sqlite3.Connection, paper: dict) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO papers
            (arxiv_id, title, authors, abstract, categories, published, pdf_url, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            paper["arxiv_id"],
            paper["title"],
            json.dumps(paper["authors"]),
            paper["abstract"],
            json.dumps(paper["categories"]),
            paper["published"],
            paper["pdf_url"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def get_unprocessed(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT arxiv_id, title, authors, abstract, categories, published, pdf_url
        FROM papers WHERE processed = 0
        ORDER BY published DESC
    """).fetchall()
    return [
        {
            "arxiv_id": r[0],
            "title": r[1],
            "authors": json.loads(r[2]),
            "abstract": r[3],
            "categories": json.loads(r[4]),
            "published": r[5],
            "pdf_url": r[6],
        }
        for r in rows
    ]


def count_total(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
