"""A rehashed human report must still agree with the native numerical result."""

import json

import pytest
from qrae.artifacts import sha256_bytes

from quantos_showcase.run_adapters import inspect_run


def test_rehashed_factor_markdown_is_refused(tmp_path):
    pytest.importorskip("factor_research")
    from factor_research.runs import run_trial
    from factor_research.synthetic import make_fixture

    make_fixture(tmp_path / "fixture")
    result = run_trial(
        tmp_path / "fixture/panel.csv",
        tmp_path / "fixture/config.json",
        tmp_path / "runs",
        "baseline",
    )
    folder = tmp_path / "runs/baseline"
    assert result["status"] == "SUCCESS"
    (folder / "report.md").write_text("A made-up profitable result.\n")
    receipt = json.loads((folder / "completion.json").read_bytes())
    receipt["files"]["report.md"] = sha256_bytes((folder / "report.md").read_bytes())
    (folder / "completion.json").write_text(json.dumps(receipt))
    row = inspect_run(
        {"id": "factor", "kind": "factor-trial", "path": str(folder)}, tmp_path
    )
    assert row["result_state"] == "REFUSED"
    assert "factor rendered report differs" in row["reason"]
