"""Native selection at the real signing boundary; subprocess output is a test double."""

import copy
import json
import subprocess
from types import SimpleNamespace

import pytest
from qrae.llm import ReplayProvider
from test_shared_loop import EXAMPLES, NOTE_ID, folder_for, proposal_replay
from test_shared_loop import shared as shared

from quantos_showcase import loop


def native_run(shared, **overrides):
    options = {"shared_root": shared["root"], "manifest_sha256": shared["digest"],
               "note_ids": [NOTE_ID], "native_critic": True}
    options.update(overrides)
    return loop.run_loop(shared["campaign"], shared["tmp"] / "panel/panel.csv",
                         EXAMPLES / "synthetic-contract.json", None, shared["tmp"] / "store",
                         "shared", ReplayProvider(proposal_replay(shared)),
                         receipt_keys=shared["tmp"] / "keys", **options)


def payload(classification="HYPOTHESIS"):
    return json.dumps({"summary": "Synthetic subprocess test double, no model execution.",
                       "requested_transition": "NONE", "artifact_refs": [],
                       "claims": [{"text": "Synthetic evidence only.", "classification": classification,
                                   "artifact_refs": []}]}).encode()


def test_explicit_native_critic_uses_fixed_broker_once_and_keeps_test_sealed(shared, monkeypatch):
    calls = []

    def subprocess_double(command, **options):
        calls.append((command, options))
        assert options["timeout"] == 120 and options["shell"] is False
        assert command[:2] == ["codex", "exec"]
        assert "--ignore-rules" in command  # Existing fixed command is not changed by selection.
        assert b'"test"' not in options["input"]
        return SimpleNamespace(returncode=0, stdout=payload(), stderr=b"")

    monkeypatch.setattr(loop.codex_broker, "_safe_subprocess_runner", subprocess_double)
    monkeypatch.setattr(loop, "broker_runner", lambda *_: pytest.fail("provider critic called"))
    monkeypatch.setattr(loop, "open_test", lambda *_: pytest.fail("native critic opened holdout"))
    ledger = native_run(shared)
    assert len(calls) == 1 and ledger["critic_mode"] == "native_codex"
    row = ledger["hypotheses"][0]
    assert row["status"] == "AWAITING_HUMAN" and ledger["counts"]["promoted"] == 0
    assert row["decision"]["split_status"]["test"] == "SEALED"
    assert row["decision"]["metrics"]["test"] is None
    folder = folder_for(shared)
    assert not (folder / "gate").exists()
    assert not (folder / "decisions").exists()
    envelope = json.loads((folder / "envelope.json").read_bytes())
    assert "sealed" in envelope["contracts"]["decision"]
    assert "only at the human gate" not in envelope["contracts"]["decision"]
    evidence = json.loads((folder / "critic/evidence/METHOD_1.json").read_bytes())
    assert set(evidence["result"]) == {"validation_status", "validation_metrics"}
    assert evidence["method_note"]["text"] == (folder / "cards/runs/METHOD_1/source-0.md").read_text()
    assert evidence["method_note"]["source"] == json.loads(
        (folder / "cards/runs/METHOD_1/card.json").read_bytes())["sources"][0]
    assert evidence["critic_policy"] == {"mode": "native_codex", "max_calls": 1,
                                          "timeout_seconds": 120, "holdout": "sealed"}
    assert loop.verify(folder, receipt_keys=shared["tmp"] / "keys")["verified"]
    with pytest.raises(ValueError, match="sealed"):
        loop.gate(folder, promote=["METHOD_1"])


@pytest.mark.parametrize("failure,reason", [
    (subprocess.TimeoutExpired(["codex", "exec"], 120), "CODEX_TIMEOUT"),
    (FileNotFoundError("test executable absent"), "CODEX_UNAVAILABLE"),
    (RuntimeError("test runner failure"), "RUNNER_FAILURE"),
    (b'{"requested_transition":"PROMOTED"}', "MALFORMED_OR_FORBIDDEN_OUTPUT"),
])
def test_native_failure_is_signed_once_without_fallback_or_holdout(shared, monkeypatch, failure, reason):
    calls = []

    def subprocess_double(*args, **kwargs):
        calls.append(1)
        if isinstance(failure, Exception):
            raise failure
        return SimpleNamespace(returncode=0, stdout=failure, stderr=b"")

    monkeypatch.setattr(loop.codex_broker, "_safe_subprocess_runner", subprocess_double)
    ledger = native_run(shared)
    assert calls == [1]
    row = ledger["hypotheses"][0]
    assert row["status"] == "CRITIC_UNUSABLE" and row["critic"]["reason_code"] == reason
    assert row["decision"]["metrics"]["test"] is None
    assert loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")["verified"]


@pytest.mark.parametrize("mutation", ["mode", "decision", "critic", "evidence"])
def test_native_verification_refuses_tampered_mode_evidence_or_verdict(shared, monkeypatch, mutation):
    monkeypatch.setattr(loop.codex_broker, "_safe_subprocess_runner",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=payload(), stderr=b""))
    ledger = native_run(shared)
    folder = folder_for(shared)
    if mutation == "evidence":
        p = folder / "critic/evidence/METHOD_1.json"
        p.write_bytes(p.read_bytes().replace(b'"max_calls":1', b'"max_calls":2'))
    else:
        forged = copy.deepcopy(ledger)
        if mutation == "mode":
            forged.pop("critic_mode")
        elif mutation == "decision":
            forged["hypotheses"][0]["decision"]["metrics"]["test"] = {"rank_ic_mean": 1.0}
        else:
            forged["hypotheses"][0]["critic"]["verdict"] = "CRITIC_REJECTED"
        (folder / "ledger.json").write_text(json.dumps(forged))
    with pytest.raises(ValueError):
        loop.verify(folder, receipt_keys=shared["tmp"] / "keys")


def test_native_rejects_more_than_one_candidate_before_any_provider_call(shared, monkeypatch):
    campaign = json.loads(shared["campaign"].read_text())
    campaign["k"] = 2
    shared["campaign"].write_text(json.dumps(campaign))
    monkeypatch.setattr(ReplayProvider, "complete", lambda *a: pytest.fail("proposal was read"))
    with pytest.raises(ValueError, match="one"):
        native_run(shared)
    assert not folder_for(shared).exists()


def test_native_cli_opt_in_is_separate_from_proposal_provider(shared, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(loop.codex_broker, "_safe_subprocess_runner",
                        lambda *a, **k: calls.append(1) or SimpleNamespace(returncode=0, stdout=payload(), stderr=b""))
    code = loop.main(["run", "--campaign", str(shared["campaign"]), "--prices", str(shared["tmp"] / "panel/panel.csv"),
                      "--contract", str(EXAMPLES / "synthetic-contract.json"), "--store", str(shared["tmp"] / "store"),
                      "--run-id", "shared", "--shared-root", str(shared["root"]), "--manifest-sha256", shared["digest"],
                      "--note-id", NOTE_ID, "--replay", str(proposal_replay(shared)),
                      "--receipt-keys", str(shared["tmp"] / "keys"), "--native-critic"])
    assert code == 0, capsys.readouterr().err
    assert calls == [1]


def test_interrupted_native_attempt_cannot_be_reissued_under_the_same_run_id(shared, monkeypatch):
    calls = []

    def interrupted(*a, **k):
        calls.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(loop.codex_broker, "_safe_subprocess_runner", interrupted)
    with pytest.raises(KeyboardInterrupt):
        native_run(shared)
    assert folder_for(shared).is_dir(), "an interrupted attempt must reserve its immutable run id"
    assert not (folder_for(shared) / "ledger.json").exists()
    with pytest.raises(FileExistsError):
        native_run(shared)
    assert calls == [1]


def test_native_cli_refuses_live_proposals_before_provider_construction(shared, monkeypatch):
    monkeypatch.setattr(loop, "provider_for", lambda *_: pytest.fail("live provider constructed"))
    assert loop.main(["run", "--campaign", str(shared["campaign"]), "--prices", "unused.csv",
                      "--contract", "unused.json", "--store", str(shared["tmp"] / "store"),
                      "--run-id", "shared", "--shared-root", str(shared["root"]),
                      "--live", "--native-critic"]) == 2


def test_native_rejection_never_opens_the_test_split(shared, monkeypatch):
    monkeypatch.setattr(loop.codex_broker, "_safe_subprocess_runner",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=payload("REJECTED"), stderr=b""))
    ledger = native_run(shared)
    assert ledger["counts"]["rejected_by_critic"] == 1
    assert ledger["hypotheses"][0]["decision"]["metrics"]["test"] is None
    assert loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")["verified"]


def test_native_rejects_altered_prices_before_any_provider_call(shared, monkeypatch):
    prices = shared["tmp"] / "panel/panel.csv"
    raw = prices.read_bytes()
    assert b',100,group_0' in raw
    prices.write_bytes(raw.replace(b',100,group_0', b',999,group_0', 1))
    monkeypatch.setattr(ReplayProvider, "complete", lambda *a: pytest.fail("proposal was read"))
    with pytest.raises(ValueError, match="synthetic"):
        native_run(shared)
    assert not folder_for(shared).exists()


def test_interactive_gate_refuses_native_sealed_run(shared, monkeypatch):
    monkeypatch.setattr(loop.codex_broker, "_safe_subprocess_runner",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=payload(), stderr=b""))
    native_run(shared)
    folder = folder_for(shared)
    with pytest.raises(ValueError, match="sealed"):
        loop.gate(folder, ask=lambda _: pytest.fail("native run reached legacy interactive gate"))
    assert not (folder / "decisions").exists()
    assert loop.status(folder)["counts"]["awaiting_human"] == 1
    assert loop.gate(folder, reject=["METHOD_1"])["counts"]["rejected_by_human"] == 1
    assert loop.verify(folder, receipt_keys=shared["tmp"] / "keys")["verified"]
