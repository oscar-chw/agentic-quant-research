"""Offline acceptance tests; these import the installed application/libraries."""

import copy
import csv
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from qrae.artifacts import ArtifactError
from qrae.kernel import KernelError, verify_run
from replay.book import NotCovered

from quantos_showcase.pipeline import (
    canonical,
    collect_snapshot,
    derive_prices,
    digest,
    fixture_messages,
    run_demo,
    write_json,
)


class ConnectedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="quantos-integration-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def test_persisted_quotes_replay_to_hand_calculated_midpoints(self):
        collect_snapshot(fixture_messages(), self.root / "collector")
        derive_prices(self.root / "collector", self.root / "prices")
        with (self.root / "prices/prices.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 80)
        self.assertEqual(rows[0]["price"], "0.471")
        self.assertEqual(rows[1]["price"], "0.4715")
        diagnostics = json.loads((self.root / "prices/diagnostics.json").read_bytes())
        self.assertEqual(diagnostics[0]["spread"], "0.002")
        self.assertEqual(diagnostics[0]["top_imbalance"], "0.2")

    def test_normalized_prices_are_repeatable_and_input_sensitive(self):
        messages = fixture_messages()
        hashes = []
        for name in ["a", "b", "changed"]:
            events = copy.deepcopy(messages)
            if name == "changed":
                events[-1]["message"]["price_changes"][-1]["price"] = "0.49"
            collect_snapshot(events, self.root / name)
            derivation = derive_prices(self.root / name, self.root / (name + "-prices"))
            hashes.append(derivation["normalized_price_sha256"])
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])

    def test_delayed_and_unknown_instrument_inputs_are_rejected(self):
        events = fixture_messages()
        events[10]["received_ms"] += 1
        with self.assertRaisesRegex(ValueError, "zero latency"):
            collect_snapshot(events, self.root / "late")
        self.assertFalse((self.root / "late").exists())
        events = fixture_messages()
        events[10]["message"]["price_changes"][0]["asset_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown instrument"):
            collect_snapshot(events, self.root / "unknown")

    def test_raw_tampering_is_rejected_before_price_export(self):
        collect_snapshot(fixture_messages(), self.root / "collector")
        raw = self.root / "collector/raw_messages.jsonl"
        raw.write_bytes(raw.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "raw hash"):
            derive_prices(self.root / "collector", self.root / "prices")
        self.assertFalse((self.root / "prices").exists())

    def test_replay_refuses_coverage_gap_and_reconnect_without_fresh_base(self):
        for gap in [True, False]:
            source = self.root / str(gap)
            snapshot = collect_snapshot(fixture_messages(), source)
            name = next(name for name in snapshot["parts"] if name.endswith("coverage.jsonl"))
            path = source / name
            first = json.loads(path.read_bytes().splitlines()[0])
            end = first["end_ms"]
            first["end_ms"] = snapshot["decision_times_ms"][39]
            rows = [first]
            if not gap:
                # Reconnection has coverage but cannot reuse the pre-gap book.
                rows.append(
                    first
                    | {"start_ms": snapshot["decision_times_ms"][40], "end_ms": end}
                )
            path.write_bytes(b"".join(canonical(row) for row in rows))
            snapshot["parts"][name] = digest(path.read_bytes())
            write_json(source / "snapshot.json", snapshot)
            with self.assertRaises(NotCovered):
                derive_prices(source, self.root / (str(gap) + "-prices"))

    def test_installed_cross_package_run_portable_verification_and_tamper(self):
        output = self.root / "demo"
        summary = run_demo(output, observed=datetime(2026, 9, 13, tzinfo=timezone.utc))
        self.assertEqual(summary["cross_repository_data_flow"], "PASS_SYNTHETIC_ONLY")
        self.assertEqual(summary["evidence_tier"], "E0")
        self.assertFalse(summary["live_trading_authorized"])
        self.assertTrue(summary["idempotent_replay"])
        self.assertEqual(summary["references"][0]["arxiv_id"], "2505.15155v2")
        run = output / summary["run_dir"]
        provenance = json.loads((run / "snapshot/provenance.json").read_bytes())
        self.assertIn("LOCAL_SYNTHETIC", json.dumps(provenance))
        self.assertIn("urn:qrae:synthetic:", json.dumps(provenance))
        # Portable artifact verification must not consult the temporary source catalog.
        shutil.rmtree(output / "workspace/catalog")
        self.assertEqual(verify_run(run)["status"], "HUMAN_REVIEW")
        corrupted = self.root / "tampered-copy"
        shutil.copytree(run, corrupted)
        result = corrupted / "result.json"
        result.write_bytes(result.read_bytes() + b" ")
        with self.assertRaises((ArtifactError, KernelError)):
            verify_run(corrupted)
        before = (output / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run_demo(output)
        self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_source_change_reaches_experiment_returns(self):
        observed = datetime(2026, 9, 13, tzinfo=timezone.utc)
        baseline = run_demo(self.root / "baseline", observed=observed)
        changed = fixture_messages()
        changed[-1]["message"]["price_changes"][-1]["price"] = "0.49"
        with patch("quantos_showcase.pipeline.fixture_messages", return_value=changed):
            candidate = run_demo(self.root / "candidate", observed=observed)
        self.assertNotEqual(baseline["snapshot_id"], candidate["snapshot_id"])
        self.assertNotEqual(baseline["manifest_sha256"], candidate["manifest_sha256"])
        original_series = (
            self.root / "baseline" / baseline["run_dir"] / "backtest/series.jsonl"
        ).read_bytes()
        changed_series = (
            self.root / "candidate" / candidate["run_dir"] / "backtest/series.jsonl"
        ).read_bytes()
        self.assertNotEqual(original_series, changed_series)


if __name__ == "__main__":
    unittest.main()
