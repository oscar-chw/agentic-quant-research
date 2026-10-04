"""Native artifacts, snapshot staleness, partial results and immutable indexes."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from qrae.artifacts import canonical_json_bytes, sha256_bytes

from quantos_showcase.feed import run_feed
from quantos_showcase.run_index import build_index, verify_index

EXAMPLES = Path(__file__).parents[1] / "examples/feed"


def selection(tmp_path, entries):
    path = tmp_path / "selection.json"
    path.write_bytes(
        canonical_json_bytes({"schema": "quantos-run-selection/v1", "entries": entries})
    )
    return path


@pytest.fixture
def feed(tmp_path):
    result = run_feed(
        EXAMPLES / "archive.jsonl", EXAMPLES / "spec.json", tmp_path / "feed"
    )
    folder = tmp_path / "feed/admissions/runs" / result["run_id"]
    return {"id": "quotes", "kind": "quantos-feed", "path": str(folder)}, result


def test_build_links_native_features_and_recheck_read_only(tmp_path, feed):
    entry, source = feed
    request = selection(tmp_path, [entry])
    result = build_index(request, tmp_path / "index", "review")
    folder = Path(result["directory"])
    index = json.loads((folder / "index.json").read_bytes())
    row = index["entries"][0]
    assert row["verification"] == "NATIVE_RECOMPUTED"
    assert (
        row["metrics"]["splits"]["train"]["models"]["weighted_midpoint"]["mae_ticks"]
        == "50"
    )
    assert index["combined_score"] is None
    assert (folder / row["report"]).resolve() == Path(source["report"]).resolve()
    assert all((folder / path).is_file() for path in row["evidence"])
    before = {
        str(p): sha256_bytes(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert verify_index(tmp_path / "index", "review")["status"] == "VERIFIED_INDEX"
    after = {
        str(p): sha256_bytes(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert before == after
    with pytest.raises(FileExistsError):
        build_index(request, tmp_path / "index", "review")


def test_changed_native_source_marks_saved_index_stale(tmp_path, feed):
    entry, _ = feed
    build_index(selection(tmp_path, [entry]), tmp_path / "index", "review")
    raw = Path(entry["path"]) / "archive.jsonl"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ValueError, match="STALE"):
        verify_index(tmp_path / "index", "review")


def test_one_refused_entry_does_not_hide_valid_entry(tmp_path, feed):
    entry, _ = feed
    missing = {
        "id": "missing",
        "kind": "quantos-feed",
        "path": "absent/admissions/runs/feed-missing",
    }
    result = build_index(
        selection(tmp_path, [entry, missing]), tmp_path / "index", "partial"
    )
    assert result["status"] == "PARTIAL"
    rows = json.loads((Path(result["directory"]) / "index.json").read_bytes())[
        "entries"
    ]
    assert [r["result_state"] for r in rows] == ["COMPLETED", "REFUSED"]


def test_optional_dependency_unavailable_is_explicit(tmp_path):
    request = selection(
        tmp_path, [{"id": "factor", "kind": "factor-trial", "path": "trial"}]
    )
    with patch(
        "quantos_showcase.run_adapters._factor",
        side_effect=ModuleNotFoundError("missing", name="factor_research"),
    ):
        result = build_index(request, tmp_path / "index", "missing-library")
    row = json.loads((Path(result["directory"]) / "index.json").read_bytes())[
        "entries"
    ][0]
    assert row["result_state"] == "UNAVAILABLE" and row["verification"] == "NOT_RUN"
    assert "factor_research" in row["reason"] and result["status"] == "PARTIAL"


def test_unsafe_schema_links_and_incomplete_output(tmp_path, feed):
    entry, _ = feed
    linked = tmp_path / "link"
    linked.symlink_to(Path(entry["path"]))
    result = build_index(
        selection(tmp_path, [entry | {"path": str(linked)}]),
        tmp_path / "index",
        "linked",
    )
    assert result["status"] == "PARTIAL"
    with pytest.raises(ValueError, match="unsupported"):
        build_index(
            selection(tmp_path, [entry | {"kind": "shell-command"}]),
            tmp_path / "index",
            "unknown",
        )
    assert not (tmp_path / "index/runs/unknown").exists()
    folder = Path(result["directory"])
    (folder / "manifest.json").unlink()
    with pytest.raises(ValueError, match="INCOMPLETE"):
        verify_index(tmp_path / "index", "linked")


def test_index_payload_tamper_and_changed_adapter(tmp_path, feed):
    entry, _ = feed
    result = build_index(selection(tmp_path, [entry]), tmp_path / "index", "review")
    with patch("quantos_showcase.run_index.code_identity", return_value="0" * 64):
        with pytest.raises(ValueError, match="code differs"):
            verify_index(tmp_path / "index", "review")
    path = Path(result["directory"]) / "index.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_index(tmp_path / "index", "review")


def test_failed_factor_trial_remains_visible(tmp_path):
    pytest.importorskip("factor_research")
    from factor_research.runs import run_trial
    from factor_research.synthetic import make_fixture

    make_fixture(tmp_path / "fixture")
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("wrong,columns\n")
    trial = run_trial(
        invalid, tmp_path / "fixture/config.json", tmp_path / "trials", "failed"
    )
    assert trial["status"] == "FAILED"
    result = build_index(
        selection(
            tmp_path,
            [
                {
                    "id": "failed-trial",
                    "kind": "factor-trial",
                    "path": trial["directory"],
                }
            ],
        ),
        tmp_path / "index",
        "failure-review",
    )
    row = json.loads((Path(result["directory"]) / "index.json").read_bytes())[
        "entries"
    ][0]
    assert (
        row["result_state"] == "FAILED"
        and row["verification"] == "NATIVE_INTEGRITY_ONLY"
    )
    assert row["metrics"] is None
    assert (
        verify_index(tmp_path / "index", "failure-review")["status"] == "VERIFIED_INDEX"
    )
