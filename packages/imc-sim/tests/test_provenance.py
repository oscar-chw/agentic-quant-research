from pathlib import Path
import tempfile
import unittest

from imc4_analysis.analyzer import analyze
from imc4_analysis.provenance import code_identity
from test_analyzer import demo


class ProvenanceTests(unittest.TestCase):
    def test_changed_code_or_path_changes_identity_but_bytecode_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "module.py"
            file.write_text("x = 1\n")
            first = code_identity(root)["sha256"]
            (root / "ignored.pyc").write_bytes(b"cache")
            self.assertEqual(first, code_identity(root)["sha256"])
            file.write_text("x = 2\n")
            second = code_identity(root)["sha256"]
            self.assertNotEqual(first, second)
            file.rename(root / "renamed.py")
            self.assertNotEqual(second, code_identity(root)["sha256"])

    def test_report_identifies_actual_package_python_sources(self):
        identity = analyze(demo())["code_identity"]
        self.assertEqual(identity, code_identity())
        self.assertIn("provenance.py", [f["path"] for f in identity["files"]])
        self.assertIn("analyzer.py", [f["path"] for f in identity["files"]])
