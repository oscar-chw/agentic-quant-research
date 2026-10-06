from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from imc4_analysis.analyzer import analyze
from imc4_analysis.cli import main, _write_exclusive
from imc4_analysis.report import render
from test_analyzer import demo, records, packed


class ReportTests(unittest.TestCase):
    def test_cli_demo_and_deterministic_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/"a", Path(directory)/"b"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--out", str(first)]), 0)
                self.assertEqual(main(["demo", "--out", str(second)]), 0)
            receipt = json.loads((first/"complete.json").read_text())
            for name in ("report.json", "report.html", "complete.json"):
                self.assertEqual((first/name).read_bytes(), (second/name).read_bytes())
            for name, metadata in receipt["files"].items():
                content = (first/name).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["sha256"])
                self.assertEqual(len(content), metadata["bytes"])
            output = json.loads((first/"report.json").read_text())
            self.assertEqual(output["code_identity"], receipt["code_identity"])
            self.assertEqual(output["summary"]["pnl"], "-32")
            self.assertEqual(output["details"]["mode"], "sampled")

    def test_existing_output_is_refused_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"exists"
            target.mkdir()
            (target/"keep.txt").write_text("preserve me")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["demo", "--out", str(target)]), 2)
            self.assertEqual(list(target.iterdir()), [target/"keep.txt"])
            self.assertEqual((target/"keep.txt").read_text(), "preserve me")

    def test_invalid_input_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"bad.jsonl", Path(directory)/"out"
            source.write_text("{}\n")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["analyze", str(source), "--out", str(target)]), 2)
            self.assertFalse(target.exists())

    def test_interrupted_write_has_no_completion_and_retry_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"partial"
            def failing_write(path, content):
                if path.name == "report.html":
                    raise OSError("injected report write interruption")
                _write_exclusive(path, content)
            with patch("imc4_analysis.cli._write_exclusive", side_effect=failing_write), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["demo", "--out", str(target)]), 2)
            self.assertTrue((target/"report.json").exists())
            self.assertFalse((target/"complete.json").exists())
            preserved = (target/"report.json").read_bytes()
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["demo", "--out", str(target)]), 2)
            self.assertEqual((target/"report.json").read_bytes(), preserved)

    def test_untrusted_labels_are_escaped_and_no_external_assets(self):
        rows = records()
        hostile = '<img src=x onerror="alert(1)">'
        rows[0]["label"] = hostile
        rows[-1]["note"] = "<script>alert(2)</script>"
        html = render(analyze(packed(rows)))
        self.assertNotIn(hostile, html)
        self.assertIn("&lt;img", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("https://", html)
        self.assertIn("Content-Security-Policy", html)

    def test_ui_links_to_breach_and_exposes_markout_and_staleness(self):
        report = analyze(demo())
        html = render(report)
        index = report["summary"]["first_breach_index"]
        self.assertIn(f'href="#event-{index}"', html)
        self.assertIn(f'id="event-{index}"', html)
        self.assertIn("bad-fill", html)
        self.assertIn("LIMIT BREACH", html)
        self.assertIn("-14 / -42", html)
        self.assertIn("unavailable_future", html)
        self.assertIn("VALUATION UNAVAILABLE", html)


if __name__ == "__main__":
    unittest.main()
