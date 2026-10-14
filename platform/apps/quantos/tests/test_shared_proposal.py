"""Shared source bytes must enter the real QuantOS proposal prompt."""
import hashlib
import json
from pathlib import Path

import pytest
from shared_manifest import SharedManifestError

from quantos_showcase import loop, shared_proposal

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/loop"


def corpus(root, body):
    root.mkdir(exist_ok=True)
    (root / "method.md").write_text(body)
    row = {"id": "test-method", "document_type": "method_note", "title": "Method",
           "path": "method.md", "sha256": hashlib.sha256(body.encode()).hexdigest(),
           "source_refs": [{"uri": "https://example.invalid/source", "kind": "public_code", "revision": "test"}],
           "rights": {"license": "MIT", "basis": "original_summary", "redistribution": "public"}}
    raw = json.dumps({"schema": "quant-shared-notes/v1", "revision": "test", "documents": [row]}).encode()
    (root / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare(root, digest, output):
    return shared_proposal.prepare(EXAMPLES / "campaign.json", EXAMPLES / "synthetic-contract.json",
                                   root, digest, ["test-method"], output)


def test_actual_prompt_contains_source_body_and_tracks_changed_input(tmp_path):
    root = tmp_path / "notes"
    body = "Original method: use only information available before the decision.\n"
    digest = corpus(root, body)
    first = prepare(root, digest, tmp_path / "first")
    prompt = json.loads((tmp_path / "first/prompt.json").read_bytes())
    assert prompt["task"] == "PROPOSE_HYPOTHESES"
    assert prompt["notes"][0]["text"] == body
    assert prompt["notes"][0]["document_type"] == "method_note"
    assert "arxiv_id" not in prompt["notes"][0]
    assert first["state"] == "PROMPT_READY" and first["execution"] == "NOT_EXECUTED"
    assert "response" not in first
    assert first["sources"][0]["schema"] == "vault-method-citation/v1"
    assert first["prompt_sha256"] == hashlib.sha256((tmp_path / "first/prompt.json").read_bytes()).hexdigest()
    changed = body + "Changed source: fit preprocessing using the training split only.\n"
    second = prepare(root, corpus(root, changed), tmp_path / "second")
    assert json.loads((tmp_path / "second/prompt.json").read_bytes())["notes"][0]["text"] == changed
    assert first["prompt_sha256"] != second["prompt_sha256"]
    assert first["sources"][0]["source_sha256"] != second["sources"][0]["source_sha256"]


def test_stale_source_refuses_before_creating_task_output(tmp_path):
    root = tmp_path / "notes"
    digest = corpus(root, "original")
    (root / "method.md").write_text("tampered")
    with pytest.raises(SharedManifestError, match="SHARED_DOCUMENT_STALE"):
        prepare(root, digest, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_existing_task_directory_is_not_overwritten(tmp_path):
    root = tmp_path / "notes"
    digest = corpus(root, "original")
    output = tmp_path / "run"
    prepare(root, digest, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(root, digest, output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_real_proposal_validator_preserves_method_note_identity(tmp_path):
    root = tmp_path / "notes"
    digest = corpus(root, "Use only information available before the decision.")
    from shared_manifest import load_shared_note
    note = load_shared_note(root, "test-method", expected_manifest_sha256=digest)["matches"][0]
    campaign = loop.read_campaign((EXAMPLES / "campaign.json").read_bytes())
    proposal = {"id": "shared-pit", "signal": "momentum", "lookback": 2,
                "hypothesis": "Lagged momentum may persist under an availability cutoff.",
                "mechanism": "Use only prior available bars; evaluate later observations separately.",
                "assumptions": ["Source availability times are correct."],
                "limitations": ["A proposed hypothesis is not evidence of predictive value."],
                "citation": "test-method"}
    rows, valid = loop.check_proposals(json.dumps({"hypotheses": [proposal]}), campaign, [note])
    assert valid == [proposal]
    assert rows[0]["status"] == "PROPOSED"
    assert rows[0]["citation"] == "test-method"
    assert rows[0]["citation_type"] == "method_note"


def test_note_id_must_fit_existing_card_contract_before_prompt(tmp_path):
    root = tmp_path / "notes"
    corpus(root, "original")
    manifest = json.loads((root / "manifest.json").read_bytes())
    note_id = "n" * 65
    manifest["documents"][0]["id"] = note_id
    raw = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    from shared_manifest import load_shared_note
    assert load_shared_note(root, note_id, expected_manifest_sha256=digest)["status"] == "MATCHES"
    with pytest.raises(ValueError, match="64"):
        shared_proposal.prepare(EXAMPLES / "campaign.json", EXAMPLES / "synthetic-contract.json",
                                root, digest, [note_id], tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
