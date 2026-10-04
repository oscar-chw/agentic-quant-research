"""Native equivalence, admission and finite-attempt failure contracts."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from qrae.artifacts import ArtifactStore, canonical_json_bytes, sha256_bytes

from quantos_showcase import execution
from quantos_showcase.run_adapters import inspect_run
from quantos_showcase.run_index import build_index, verify_index

pytest.importorskip("imc4_analysis")
from imc4_analysis import cli, provenance  # noqa: E402

EXAMPLES = Path(__file__).parents[1] / "examples/execution"


def hashes(folder):
    return {
        p.relative_to(folder).as_posix(): sha256_bytes(p.read_bytes())
        for p in folder.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def inputs(tmp_path):
    base = tmp_path / "inputs"
    base.mkdir()
    for name in ("request.json", "source.md"):
        (base / name).write_bytes((EXAMPLES / name).read_bytes())
    scenario = json.loads((EXAMPLES / "scenario.json").read_bytes())
    scenario["scenarios"] = scenario["scenarios"][:1]
    scenario["executions"] = scenario["executions"][:1]
    (base / "scenario.json").write_bytes(canonical_json_bytes(scenario))
    return base / "request.json", base / "scenario.json", tmp_path / "attempts"


def run(inputs, identity="first"):
    return execution.run_execution(*inputs, identity)


def rehash_manifest(folder):
    value = json.loads((folder / "manifest.json").read_bytes())
    for record in value["artifacts"]:
        raw = (folder / record["path"]).read_bytes()
        record.update(sha256=sha256_bytes(raw), bytes=len(raw))
    (folder / "manifest.json").write_bytes(canonical_json_bytes(value))


def test_registered_native_path_matches_direct_and_read_only_recheck(inputs, tmp_path):
    original = cli.write_study
    calls = []

    def observe(raw, output):
        if output.parent == inputs[2] / "runs/first":
            registration = json.loads(
                (output.parent / "registration.json").read_bytes()
            )
            assert registration["scenario_sha256"] == sha256_bytes(raw)
            assert (output.parent / "request.json").read_bytes() == inputs[
                0
            ].read_bytes()
            assert (output.parent / "source.md").read_bytes() == inputs[0].with_name(
                "source.md"
            ).read_bytes()
        calls.append(output)
        return original(raw, output)

    with patch.object(cli, "write_study", side_effect=observe):
        result = run(inputs)
    assert len(calls) == 2  # Native run and temporary verification, same owner path.
    assert result["execution"] == "COMPLETED" and result["counts"]["runs"] == 2
    folder = Path(result["directory"])
    direct = tmp_path / "direct"
    original(inputs[1].read_bytes(), direct)
    assert hashes(folder / "native") == hashes(direct)
    before = hashes(inputs[2])
    assert execution.verify_execution(inputs[2], "first") == result
    assert hashes(inputs[2]) == before
    assert all(
        (folder / href).is_file()
        for href in re.findall(r"\]\(([^)]+)\)", (folder / "REPORT.md").read_text())
    )


@pytest.mark.parametrize(
    "mutation", ["missing", "inventory", "terminal", "clock", "kind", "cap"]
)
def test_invalid_native_input_refused_before_attempt_or_execution(inputs, mutation):
    raw = json.loads(inputs[1].read_bytes())
    scenario = raw["scenarios"][0]
    if mutation == "missing":
        del scenario["policy"]["tick_size"]
    elif mutation == "inventory":
        scenario["initial_position"] = 100000
    elif mutation == "terminal":
        scenario["steps"][-1]["buyer_units"] = 1
    elif mutation == "clock":
        scenario["steps"][1]["timestamp"] = 10
    elif mutation == "kind":
        raw["data_kind"] = "observed"
    else:
        raw["scenarios"] = [dict(scenario, id="case_" + str(i)) for i in range(8)]
        raw["executions"] *= 2
        raw["executions"][1] = dict(raw["executions"][1], id="second")
    inputs[1].write_bytes(canonical_json_bytes(raw))
    with patch.object(
        cli, "write_study", side_effect=AssertionError("execution must not start")
    ):
        with pytest.raises(ValueError):
            run(inputs)
    assert not inputs[2].exists()


@pytest.mark.parametrize("mutation", ["digest", "escape", "link", "oversize", "method"])
def test_source_and_method_admission(inputs, tmp_path, mutation):
    value = json.loads(inputs[0].read_bytes())
    if mutation == "digest":
        value["source"]["sha256"] = "0" * 64
    elif mutation == "escape":
        value["source"]["path"] = "../outside.md"
    elif mutation == "link":
        link = inputs[0].parent / "link.md"
        link.symlink_to(inputs[0].with_name("source.md"))
        value["source"]["path"] = "link.md"
    elif mutation == "oversize":
        inputs[0].with_name("source.md").write_bytes(b"x" * (execution.MAX_INPUT + 1))
    else:
        value["method"] = "arbitrary-provider"
    inputs[0].write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError):
        run(inputs)
    assert not inputs[2].exists()


def test_existing_and_concurrent_attempts_preserved(inputs):
    result = run(inputs)
    before = hashes(inputs[2])
    with pytest.raises(FileExistsError):
        run(inputs)
    assert hashes(inputs[2]) == before
    with patch("quantos_showcase.execution.reserve", return_value=None):
        with pytest.raises(FileExistsError, match="concurrent"):
            run(inputs, "concurrent")
    assert hashes(inputs[2]) == before
    assert Path(result["report"]).is_file()


@pytest.mark.parametrize("phase", ["partial_native", "final_commit"])
def test_interruption_is_incomplete_and_restart_is_new_attempt(inputs, tmp_path, phase):
    def interrupted(raw, output):
        output.mkdir()
        (output / "scenario.json").write_bytes(raw)
        raise KeyboardInterrupt

    target = (
        patch.object(cli, "write_study", side_effect=interrupted)
        if phase == "partial_native"
        else patch.object(
            ArtifactStore, "commit_manifest", side_effect=KeyboardInterrupt
        )
    )
    with target, pytest.raises(KeyboardInterrupt):
        run(inputs, "interrupted")
    old = inputs[2] / "runs/interrupted"
    assert (old / "registration.json").is_file()
    assert not (old / "manifest.json").exists()
    if phase == "final_commit":
        assert (old / "native/complete.json").is_file()
    before = hashes(old)
    with pytest.raises(ValueError, match="INCOMPLETE"):
        execution.verify_execution(inputs[2], "interrupted")
    with pytest.raises(FileExistsError):
        run(inputs, "interrupted")
    result = run(inputs, "fresh")
    direct = tmp_path / "direct"
    cli.write_study(inputs[1].read_bytes(), direct)
    assert hashes(Path(result["directory"]) / "native") == hashes(direct)
    assert hashes(old) == before
    selected = inspect_run(
        {"id": "old", "kind": "quantos-execution", "path": str(old)}, tmp_path
    )
    assert selected["result_state"] == "REFUSED" and "INCOMPLETE" in selected["reason"]


def test_rehashed_native_trace_still_fails_recompute(inputs):
    result = run(inputs)
    folder = Path(result["directory"])
    name = result["rows"][0]["run"] + "/trace.json"
    trace_path = folder / "native" / name
    trace = json.loads(trace_path.read_bytes())
    trace["policy"] = "invented-policy"
    trace_path.write_bytes(cli.json_bytes(trace))
    receipt = json.loads((folder / "native/complete.json").read_bytes())
    raw = trace_path.read_bytes()
    receipt["files"][name] = {"sha256": sha256_bytes(raw), "bytes": len(raw)}
    (folder / "native/complete.json").write_bytes(cli.json_bytes(receipt))
    rehash_manifest(folder)
    with pytest.raises(ValueError, match="native recomputation differs"):
        execution.verify_execution(inputs[2], "first")


@pytest.mark.parametrize(
    "name",
    ["request.json", "scenario.json", "source.md", "registration.json", "REPORT.md"],
)
def test_changed_wrapper_payload_fails_even_after_manifest_update(inputs, name):
    result = run(inputs)
    folder = Path(result["directory"])
    path = folder / name
    path.write_bytes(path.read_bytes() + b" ")
    rehash_manifest(folder)
    with pytest.raises(ValueError):
        execution.verify_execution(inputs[2], "first")


def test_code_identity_and_unavailable_dependency(inputs, tmp_path):
    result = run(inputs)
    with patch(
        "quantos_showcase.execution.implementation_identity", return_value="0" * 64
    ):
        with pytest.raises(ValueError, match="identity"):
            execution.verify_execution(inputs[2], "first")
    with patch.object(provenance, "code_identity", return_value={"sha256": "0" * 64}):
        with pytest.raises(ValueError, match="accepted"):
            execution.verify_execution(inputs[2], "first")
    with patch(
        "quantos_showcase.execution.native",
        side_effect=ModuleNotFoundError("absent", name="imc4_analysis"),
    ):
        row = inspect_run(
            {"id": "run", "kind": "quantos-execution", "path": result["directory"]},
            tmp_path,
        )
        assert row["result_state"] == "UNAVAILABLE"
        assert (
            execution.main(["verify", "--store", str(inputs[2]), "--run-id", "first"])
            == 2
        )


def test_index_navigation_and_unexpected_files(inputs, tmp_path):
    result = run(inputs)
    selection = tmp_path / "selection.json"
    selection.write_bytes(
        canonical_json_bytes(
            {
                "schema": "quantos-run-selection/v1",
                "entries": [
                    {
                        "id": "execution",
                        "kind": "quantos-execution",
                        "path": result["directory"],
                    }
                ],
            }
        )
    )
    index = build_index(selection, tmp_path / "index", "review")
    assert index["status"] == "VERIFIED_INDEX"
    assert verify_index(tmp_path / "index", "review")["status"] == "VERIFIED_INDEX"
    row = json.loads((Path(index["directory"]) / "index.json").read_bytes())["entries"][
        0
    ]
    assert row["metrics"]["native_rows"] == result["rows"]
    assert row["qualifications"]["resumable_service"] is False
    assert any(Path(name).name == "source.md" for name in row["evidence"])
    assert any(method["module"] == "imc4_analysis.quoting" for method in row["methods"])
    (Path(result["directory"]) / "unclaimed.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="STALE"):
        verify_index(tmp_path / "index", "review")


def test_unsafe_id_linked_store_and_escaped_question(inputs, tmp_path):
    for identity in ("../bad", "with.dot", "", "a" * 65):
        with pytest.raises(ValueError):
            run(inputs, identity)
    target = tmp_path / "target"
    target.mkdir()
    inputs[2].symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="unlinked"):
        run(inputs)
    assert list(target.iterdir()) == []
    request = json.loads(inputs[0].read_bytes())
    request["question"] = '<script>alert("x")</script>'
    report = execution.render(request, {"rows": [], "assumptions": "synthetic"})
    assert "<script>" not in report and "&lt;script&gt;" in report
