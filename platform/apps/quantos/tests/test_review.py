"""Report acceptance from current artifacts, including publication refusal."""
import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.parse import unquote

import pytest
from qrae.artifacts import ArtifactError
from qrae.kernel import KernelError

from quantos_showcase.pipeline import run_demo
from quantos_showcase.review import export_review


@pytest.fixture
def demo(tmp_path):
    output = tmp_path / "run"
    summary = run_demo(output, observed=datetime(2026, 9, 14, tzinfo=timezone.utc))
    return output, summary


def test_report_matches_verified_accounting_and_sources(demo):
    output, summary = demo
    review = json.loads((output / "review.json").read_text())
    result = json.loads((output / summary["run_dir"] / "result.json").read_text())
    text = (output / "REPORT.md").read_text()
    assert review["schema_version"] == "quantos.current-review.v1"
    assert review["manifest_sha256"] == summary["manifest_sha256"]
    assert review["decision"] == "REVISE"
    assert "-3.33%" in text
    assert review["splits"]["test"]["metrics"] == result["metrics"]["test"]
    assert review["splits"]["test"]["metrics"]["net_compounded_return"] == pytest.approx(-0.033255261397489466)
    assert [review["splits"][s]["metrics"]["observations"] for s in ("train", "validation", "test")] == [45, 15, 15]
    assert review["validation"]["checks"]["price_to_position_replay"]["observations_recomputed"] == 75
    assert review["input"]["events"] == 80
    assert review["clocks"]["catalog_availability_basis"] == "OBSERVED_AT_IMPORT"
    assert review["splits"]["train"]["end_exit_time"] < review["splits"]["validation"]["start_exit_time"]
    sources = review["source_eligibility"]
    assert [r["source_id"] for r in sources["eligible"]] == ["2505.15155v2"]
    assert {r["reason"] for r in sources["excluded"]} == {
        "NOT_PAPER_NOTE", "SOURCE_UNAVAILABLE_OR_UNKNOWN", "MISSING_OR_NONREGULAR_SOURCE"}
    assert len(re.findall(r'\]\(([^)]+)\)', text)) >= 10
    for href in re.findall(r'\]\(([^)]+)\)', text):
        if not href.startswith("https://"):
            assert (output / unquote(href)).is_file(), href
    for relative, expected in review["artifact_sha256"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        export_review(output)


@pytest.mark.parametrize("target", ["result", "note", "raw", "derivation"])
def test_changed_evidence_prevents_new_report_publication(demo, target):
    output, summary = demo
    path = {
        "result": output / summary["run_dir"] / "result.json",
        "note": output / "workspace/notes/workflow.md",
        "raw": output / "workspace/collector/raw_messages.jsonl",
        "derivation": output / "workspace/data/derivation.json",
    }[target]
    if target == "derivation":
        value = json.loads(path.read_text())
        value["parameters"]["clock_rule"] = "wrong clock"
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    destination = output / "re-export"
    old_report = (output / "REPORT.md").read_bytes()
    with pytest.raises((ValueError, KernelError, ArtifactError)):
        export_review(output, destination=destination)
    assert not destination.exists()
    assert (output / "REPORT.md").read_bytes() == old_report
