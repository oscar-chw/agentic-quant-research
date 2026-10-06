"""Run with python -m unittest test_paper_store; no optional services required."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from paper_demo import run_demo
from paper_store import (
    already_fetched,
    count_total,
    get_unprocessed,
    init_db,
    save_paper,
)


class PaperStoreTests(unittest.TestCase):
    def test_demo_and_existing_output_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "demo"
            self.assertEqual(run_demo(output)["unique_rows"], 3)
            before = (output / "papers.sqlite").read_bytes()
            with self.assertRaises(FileExistsError):
                run_demo(output)
            self.assertEqual(before, (output / "papers.sqlite").read_bytes())

    def test_empty_database_and_quoted_id_are_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            conn = init_db(str(Path(temp) / "papers.sqlite"))
            try:
                self.assertEqual(get_unprocessed(conn), [])
                self.assertFalse(already_fetched(conn, "missing"))
                paper = {
                    "arxiv_id": "x'); DROP TABLE papers; --",
                    "title": "Original",
                    "authors": ["Demo"],
                    "abstract": "Synthetic",
                    "categories": [],
                    "published": "2024-01-01",
                    "pdf_url": "https://example.invalid",
                }
                save_paper(conn, paper)
                save_paper(conn, {**paper, "title": "Replacement"})
                self.assertTrue(already_fetched(conn, paper["arxiv_id"]))
                self.assertEqual(count_total(conn), 1)
                self.assertEqual(get_unprocessed(conn)[0]["title"], "Original")
                conn.execute("UPDATE papers SET processed = 1")
                conn.commit()
                self.assertEqual(get_unprocessed(conn), [])
            finally:
                conn.close()

    def test_existing_schema_and_rows_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "papers.sqlite")
            conn = init_db(path)
            schema = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='papers'"
            ).fetchone()[0]
            conn.close()
            conn = sqlite3.connect(path)
            conn.execute(
                "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    "legacy",
                    "Title",
                    "[]",
                    "Abstract",
                    "[]",
                    "2024-01-01",
                    "",
                    "",
                    1,
                    "kept",
                ),
            )
            conn.commit()
            conn.close()
            conn = init_db(path)
            try:
                self.assertEqual(
                    conn.execute(
                        "SELECT sql FROM sqlite_master WHERE name='papers'"
                    ).fetchone()[0],
                    schema,
                )
                self.assertEqual(
                    conn.execute("SELECT processed, vault_path FROM papers").fetchone(),
                    (1, "kept"),
                )
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
