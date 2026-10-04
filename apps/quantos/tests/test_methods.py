"""Exercise registration ordering and native replay, including unfavorable trials."""

import json
import shutil
from pathlib import Path

import pytest
from qrae.artifacts import canonical_json_bytes, sha256_bytes

from quantos_showcase import methods

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture
def inputs(tmp_path):
    shutil.copytree(EXAMPLES / "methods", tmp_path / "inputs")
    for name in ("archive.jsonl", "spec.json"):
        shutil.copy2(EXAMPLES / "feed" / name, tmp_path / "inputs" / name)
    return tmp_path / "inputs"


def run(inputs, root, name="trial"):
    return methods.run_method(
        inputs / "card.json", inputs / "archive.jsonl", inputs / "spec.json", root, name
    )


def test_registered_before_execution_and_negative_native_replay(
    inputs, tmp_path, monkeypatch
):
    root = tmp_path / "store"
    native = methods.feed.run_feed
    calls = []

    def observed(archive, spec, store):
        folder = root / "methods/runs/trial"
        assert (folder / "registration.json").is_file()
        assert (folder / "source-0.md").is_file()
        assert archive == folder / "archive.jsonl" and spec == folder / "spec.json"
        assert not (folder / "outcome.json").exists()
        assert not (folder / "manifest.json").exists()
        calls.append(True)
        return native(archive, spec, store)

    monkeypatch.setattr(methods.feed, "run_feed", observed)
    result = run(inputs, root)
    assert calls == [True]
    assert (
        result["execution"] == "COMPLETED"
        and result["verification"] == "NATIVE_RECOMPUTE"
    )
    assert result["decision"]["status"] == "NOT_SUPPORTED"
    assert result["decision"]["improvement_ticks"] == "-50"
    assert result["decision"]["evaluated_pairs"] == 1
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (inputs / "derivation.md").unlink()
    (inputs / "archive.jsonl").write_text("Original changed after freeze")
    assert methods.verify_method(root, "trial") == result
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run(inputs, root)


def test_card_controls_decision_without_changing_native_engine(inputs, tmp_path):
    card = json.loads((inputs / "card.json").read_text())
    card["decision"]["min_pairs"] = 2
    (inputs / "card.json").write_text(json.dumps(card))
    result = run(inputs, tmp_path / "store")
    assert result["decision"]["status"] == "INSUFFICIENT"
    assert result["decision"]["candidate_mae_ticks"] == "50"


@pytest.mark.parametrize(
    "target", ["source-0.md", "archive.jsonl", "outcome.json", "REPORT.md"]
)
def test_tampering_refused(inputs, tmp_path, target):
    root = tmp_path / "store"
    result = run(inputs, root)
    path = Path(result["directory"]) / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        methods.verify_method(root, "trial")


def test_code_change_and_interrupted_output_refused(inputs, tmp_path, monkeypatch):
    root = tmp_path / "store"
    result = run(inputs, root)
    with monkeypatch.context() as m:
        m.setattr(methods, "implementation_identity", lambda: "f" * 64)
        with pytest.raises(ValueError, match="identity"):
            methods.verify_method(root, "trial")
    (Path(result["directory"]) / "manifest.json").unlink()
    with pytest.raises(ValueError, match="INCOMPLETE"):
        methods.verify_method(root, "trial")


def test_rehashed_false_decision_still_fails_native_rule(inputs, tmp_path):
    root = tmp_path / "store"
    result = run(inputs, root)
    folder = Path(result["directory"])
    card = json.loads((folder / "card.json").read_text())
    outcome = json.loads((folder / "outcome.json").read_text())
    outcome["decision"]["status"] = "SUPPORTS_RULE"
    (folder / "outcome.json").write_bytes(canonical_json_bytes(outcome))
    (folder / "REPORT.md").write_text(methods.render(card, outcome))
    manifest = json.loads((folder / "manifest.json").read_text())
    for entry in manifest["artifacts"]:
        raw = (folder / entry["path"]).read_bytes()
        entry.update(bytes=len(raw), sha256=sha256_bytes(raw))
    (folder / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="native method decision"):
        methods.verify_method(root, "trial")


def test_failed_attempt_and_interrupt_retained(inputs, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("controlled native failure")

    monkeypatch.setattr(methods.feed, "run_feed", fail)
    root = tmp_path / "store"
    result = run(inputs, root)
    assert result["execution"] == "FAILED" and result["decision"] is None
    assert result["verification"] == "FAILURE_INTEGRITY_ONLY"
    assert "controlled native failure" in Path(result["report"]).read_text()
    assert methods.verify_method(root, "trial") == result

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(methods.feed, "run_feed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(inputs, root, "interrupted")
    folder = root / "methods/runs/interrupted"
    assert (folder / "registration.json").exists() and not (
        folder / "manifest.json"
    ).exists()
    with pytest.raises(ValueError, match="INCOMPLETE"):
        methods.verify_method(root, "interrupted")


def test_invalid_sources_and_unsupported_method_refuse_before_execution(
    inputs, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        methods.feed, "run_feed", lambda *a: pytest.fail("must not execute")
    )
    root = tmp_path / "store"
    (inputs / "derivation.md").write_text("stale")
    with pytest.raises(ValueError, match="digest"):
        run(inputs, root)
    assert not root.exists()
    card = json.loads((inputs / "card.json").read_text())
    card["method"] = "shell-command"
    (inputs / "card.json").write_text(json.dumps(card))
    with pytest.raises(ValueError, match="unsupported"):
        run(inputs, root)
    assert not root.exists()
