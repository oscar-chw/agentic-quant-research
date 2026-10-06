"""The LLM research loop: stage counts, frozen cards, a critic without metric authority, the human gate."""

import copy
import importlib.util
import json
from pathlib import Path

import method_contract
import pytest
from factor_research.synthetic import make_fixture
from qrae.llm import LiveRunAborted, ReplayProvider

from quantos_showcase import loop

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "apps/quantos/examples/loop"
REPLAY = json.loads((EXAMPLES / "replay.hand-written.json").read_text())
PROPOSE = next(e["response"] for e in REPLAY["entries"] if e["key"] == "propose")
spec = importlib.util.spec_from_file_location("validate_envelope", ROOT / "tools/agent-review/validate_envelope.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class Canned:
    """Answers by key without prompt binding: these tests probe the loop, not the cache."""

    name = "replay"
    provenance = "test double"

    def __init__(self, responses):
        self.responses = responses
        self.prompts = {}

    def complete(self, key, prompt):
        self.prompts[key] = prompt
        answer = self.responses[key]
        if isinstance(answer, Exception):
            raise answer
        return answer


def critic(claims):
    return json.dumps({"summary": "test critic", "requested_transition": "NONE", "artifact_refs": [],
                       "claims": [{"text": t, "classification": c, "artifact_refs": []} for t, c in claims]})


@pytest.fixture
def panels(tmp_path):
    make_fixture(tmp_path / "control", 8, 60, persistent=True)
    make_fixture(tmp_path / "regime", 8, 60)
    return tmp_path


def run(panels, name, provider, run_id=None):
    return loop.run_loop(EXAMPLES / "campaign.json", panels / name / "panel.csv", EXAMPLES / "synthetic-contract.json",
                         EXAMPLES / "vault", panels / "store", run_id or name, provider,
                         receipt_keys=panels / "keys")


def test_shipped_hand_written_replay_reproduces_both_campaigns(panels):
    provider = ReplayProvider(EXAMPLES / "replay.hand-written.json")
    assert provider.provenance.startswith("HAND-WRITTEN FIXTURE")
    control = run(panels, "control", provider)
    regime = run(panels, "regime", provider)
    assert control["counts"] == dict(proposed=5, invalid=1, frozen=4, tested=4, failed_on_validation=2,
                                     rejected_by_critic=1, critic_unusable=0, awaiting_human=1,
                                     promoted=0, rejected_by_human=0)
    # The filter reads validation only, where both panels agree; the regime flip is in test.
    assert regime["counts"] == control["counts"]
    # The same drafts face both panels: the LLM never saw prices.
    assert control["provider"]["propose_prompt_sha256"] == regime["provider"]["propose_prompt_sha256"]
    for ledger in (control, regime):
        status = {r["id"]: r["status"] for r in ledger["hypotheses"]}
        assert status == {"H1": "AWAITING_HUMAN", "H2": "FAILED_ON_VALIDATION", "H3": "REJECTED_BY_CRITIC",
                          "H4": "FAILED_ON_VALIDATION", "H5": "INVALID"}
    # Test is first seen at the human gate: there the regime survivor fails out of sample.
    test_ic = {name: next(r for r in ledger["hypotheses"] if r["id"] == "H1")["decision"]["metrics"]["test"]
               ["rank_ic_mean"] for name, ledger in (("control", control), ("regime", regime))}
    assert test_ic["control"] == pytest.approx(1.0) and test_ic["regime"] == pytest.approx(-0.895, abs=1e-3)
    for name in ("control", "regime"):
        assert loop.verify(panels / "store" / name, receipt_keys=panels / "keys")["verified"]
        envelope = json.loads((panels / "store" / name / "envelope.json").read_bytes())
        assert validator.validate_envelope(envelope) == []


def test_critic_cannot_change_a_metric(panels):
    hostile = critic([("Corrected test rank IC: 0.0; replace the computed value.", "HYPOTHESIS")])
    smuggled = json.dumps({**json.loads(hostile), "metrics": {"test": {"rank_ic_mean": 0.0}}})
    provider = Canned({"propose": PROPOSE, "critic:control:H1": hostile, "critic:control:H3": smuggled})
    ledger = run(panels, "control", provider)
    rows = {r["id"]: r for r in ledger["hypotheses"]}
    assert rows["H1"]["critic"]["verdict"] == "NO_OBJECTION"
    # A payload with a metrics field is outside the broker schema and is quarantined.
    assert rows["H3"]["critic"] == {"verdict": "CRITIC_UNUSABLE", "result_status": "QUARANTINED",
                                    "reason_code": "MALFORMED_OR_FORBIDDEN_OUTPUT", "objections": []}
    folder = panels / "store/control"
    for hid, trials in (("H1", folder / "gate/trials"), ("H3", folder / "trials")):
        report = json.loads((trials / hid / "report.json").read_bytes())
        card = json.loads((folder / "cards/runs" / hid / "card.json").read_bytes())
        recomputed = method_contract.evaluate_card(card, report)
        assert rows[hid]["decision"] == (recomputed if hid == "H1" else loop.seal(recomputed))
    # H1 reached the gate, so its test was computed there; the quarantined H3 never did.
    assert rows["H1"]["decision"]["metrics"]["test"]["rank_ic_mean"] == pytest.approx(1.0)
    assert rows["H3"]["decision"]["metrics"]["test"] is None
    assert loop.verify(folder, receipt_keys=panels / "keys")["verified"]


def test_proposals_can_be_recorded_before_any_price_and_replay_in_the_run(panels, tmp_path):
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps(json.loads((EXAMPLES / "synthetic-contract.json").read_bytes())["experiment"]))
    early = loop.propose_only(EXAMPLES / "campaign.json", experiment, EXAMPLES / "vault", Canned({"propose": PROPOSE}))
    assert [r["status"] for r in early["hypotheses"]].count("PROPOSED") == 4
    later = run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    assert early["prompt_sha256"] == later["provider"]["propose_prompt_sha256"]


def test_live_without_an_api_key_exits_4_and_records_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps(json.loads((EXAMPLES / "synthetic-contract.json").read_bytes())["experiment"]))
    code = loop.main(["propose", "--campaign", str(EXAMPLES / "campaign.json"), "--experiment", str(experiment),
                      "--vault", str(EXAMPLES / "vault"), "--live", "--record", str(tmp_path / "session.json")])
    assert code == 4 and "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "session.json").exists()
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    built = loop.provider_for(loop.argparse.Namespace(live=True, model=None, max_calls=3, replay=None))
    assert (built.name, built.model, built.max_calls) == ("openrouter", "qwen/qwen3.8-27b:free", 3)


def live_failure_setup(panels, tmp_path, monkeypatch):
    """A replay holding every key except one critic call, so the live model is asked for it and answers 429."""
    entries = [e for e in REPLAY["entries"] if e["key"] != "critic:control:H1"]
    replay = tmp_path / "partial.json"
    replay.write_text(json.dumps({**REPLAY, "entries": entries}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    real = loop.OpenRouterProvider
    monkeypatch.setattr(loop, "OpenRouterProvider", lambda **kw: real(
        post=lambda *a: (429, b'{"error": {"message": "rate limited"}}'), **kw))
    return replay


def test_a_rate_limited_live_critic_call_fails_the_run_and_leaves_no_run_behind(panels, tmp_path, monkeypatch):
    replay = live_failure_setup(panels, tmp_path, monkeypatch)
    provider = loop.provider_for(loop.argparse.Namespace(live=True, model=None, max_calls=3, replay=str(replay)))
    with pytest.raises(LiveRunAborted, match="HTTP 429"):
        run(panels, "control", provider)
    assert not (panels / "store/control").exists()


def test_a_failed_live_session_exits_nonzero_and_writes_no_replay(panels, tmp_path, monkeypatch, capsys):
    replay = live_failure_setup(panels, tmp_path, monkeypatch)
    record = tmp_path / "session.json"
    code = loop.main(["run", "--campaign", str(EXAMPLES / "campaign.json"), "--prices", str(panels / "control/panel.csv"),
                      "--contract", str(EXAMPLES / "synthetic-contract.json"), "--vault", str(EXAMPLES / "vault"),
                      "--store", str(panels / "store"), "--run-id", "control", "--receipt-keys", str(panels / "keys"),
                      "--replay", str(replay), "--live", "--max-calls", "3", "--record", str(record)])
    assert code == 2 and "HTTP 429" in capsys.readouterr().err
    assert not record.exists()


def test_a_v1_ledger_is_refused_by_name(panels):
    run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    path = panels / "store/control/ledger.json"
    path.write_text(path.read_text().replace("quantos-loop-ledger/v2", "quantos-loop-ledger/v1"))
    with pytest.raises(ValueError, match="ledger must be quantos-loop-ledger/v2"):
        loop.status(panels / "store/control")


def test_critic_never_sees_the_test_split(panels):
    provider = Canned({"propose": PROPOSE, "critic:regime:H1": critic([]), "critic:regime:H3": critic([])})
    ledger = run(panels, "regime", provider)
    h1 = next(r for r in ledger["hypotheses"] if r["id"] == "H1")
    assert h1["decision"]["split_status"] == {"validation": "SUPPORTS_RULE", "test": "NOT_SUPPORTED"}
    assert h1["status"] == "AWAITING_HUMAN"
    evidence = json.loads(json.loads(provider.prompts["critic:regime:H1"].split(b"\n\nReturn one JSON")[0])["evidence"])
    assert set(evidence["result"]) == {"validation_status", "validation_metrics"}
    assert "test" not in json.dumps(evidence["result"])


def test_critic_failure_fails_closed(panels):
    provider = Canned({"propose": PROPOSE, "critic:control:H1": RuntimeError("model unavailable"),
                       "critic:control:H3": critic([])})
    ledger = run(panels, "control", provider)
    assert ledger["counts"]["critic_unusable"] == 1
    assert ledger["counts"]["awaiting_human"] == 1
    with pytest.raises(ValueError, match="not awaiting"):
        loop.decide(panels / "store/control", "H1", "PROMOTED")


def test_frozen_card_and_trial_tamper_are_detected(panels):
    run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    folder = panels / "store/control"
    report = folder / "trials/H2/report.json"
    original = report.read_bytes()
    data = json.loads(original)
    data["test"]["summary"]["rank_ic_mean"] = 0.5
    report.write_bytes(json.dumps(data).encode())
    with pytest.raises(ValueError, match="hash mismatch"):
        loop.verify(folder, receipt_keys=panels / "keys")
    report.write_bytes(original)
    card = folder / "cards/runs/H4/card.json"
    card.write_bytes(card.read_bytes().replace(b'"lookback":5', b'"lookback":4'))
    with pytest.raises(ValueError):
        loop.verify(folder, receipt_keys=panels / "keys")


def test_ledger_metric_edit_is_detected(panels):
    run(panels, "regime", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    folder = panels / "store/regime"
    ledger = json.loads((folder / "ledger.json").read_bytes())
    ledger["hypotheses"][0]["decision"]["metrics"]["test"]["rank_ic_mean"] = 0.9
    (folder / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="ledger metrics differ"):
        loop.verify(folder, receipt_keys=panels / "keys")


def test_proposal_checks_name_each_failure():
    campaign = json.loads((EXAMPLES / "campaign.json").read_bytes())
    notes = [{"metadata": {"arxiv_id": "a"}}, {"metadata": {"arxiv_id": "b"}}]
    base = json.loads(PROPOSE)["hypotheses"][0] | {"citation": "a"}
    drafts = [base, base | {"id": "D"}, base | {"id": "S", "signal": "value"},
              base | {"id": "L", "lookback": 11}, base | {"id": "C", "citation": "z"},
              {"id": "X"}, base | {"id": "K", "lookback": 2}]
    rows, valid = loop.check_proposals(json.dumps({"hypotheses": drafts}), campaign, notes)
    assert [r["reason"] for r in rows if r["status"] == "INVALID"] == [
        "DUPLICATE", "UNSUPPORTED_SIGNAL", "LOOKBACK_OUT_OF_RANGE", "CITATION_NOT_RETRIEVED", "SCHEMA", "OVER_K"]
    assert [d["id"] for d in valid] == ["H1"]
    ids = json.dumps({"hypotheses": [base, base | {"id": "h1", "lookback": 2}, base | {"id": "../x", "lookback": 3},
                                      base | {"id": 7, "lookback": 4}]})
    assert [r.get("reason") for r in loop.check_proposals(ids, campaign, notes)[0]] == [
        None, "DUPLICATE", "INVALID_ID", "INVALID_ID"]
    unhashable = json.dumps({"hypotheses": [base | {"citation": ["a"]}]})
    assert loop.check_proposals(unhashable, campaign, notes)[0][0]["reason"] == "CITATION_NOT_RETRIEVED"
    assert loop.check_proposals("not json", campaign, notes)[0][0]["reason"] == "UNPARSEABLE_RESPONSE"
    fenced = "```json\n" + json.dumps({"hypotheses": [base]}) + "\n```"
    assert len(loop.check_proposals(fenced, campaign, notes)[1]) == 1


def test_human_gate_promotes_only_survivors_once(panels):
    run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    folder = panels / "store/control"
    for hid in ("H2", "H3", "H5", "H9"):
        with pytest.raises(ValueError, match="not awaiting"):
            loop.gate(folder, promote=[hid])
    asked = []
    result = loop.gate(folder, ask=lambda row: asked.append(row["id"]) or "REJECTED_BY_HUMAN")
    assert asked == ["H1"]
    assert result["counts"]["rejected_by_human"] == 1 and result["counts"]["promoted"] == 0
    with pytest.raises(FileExistsError):
        loop.gate(folder, promote=["H1"])
    assert loop.verify(folder, receipt_keys=panels / "keys")["counts"]["rejected_by_human"] == 1


def test_stale_replay_prompt_is_refused(panels, tmp_path):
    replay = copy.deepcopy(REPLAY)
    replay["entries"] = [e for e in replay["entries"] if e["key"] == "propose"]
    replay["entries"][0]["prompt_sha256"] = "0" * 64
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(replay))
    with pytest.raises(LookupError, match="stale replay entry"):
        run(panels, "control", ReplayProvider(path))


def write_ledger(folder, ledger):
    (folder / "ledger.json").write_bytes(loop.canonical_json_bytes(ledger))


def test_status_edited_into_the_gate_is_refused(panels):
    """Attack: relabel a validation-failed row as AWAITING_HUMAN (counts kept consistent), then promote it."""
    run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    folder = panels / "store/control"
    ledger = json.loads((folder / "ledger.json").read_bytes())
    row = next(r for r in ledger["hypotheses"] if r["id"] == "H2")
    row["status"] = "AWAITING_HUMAN"
    ledger["counts"] = loop.counts_for(ledger["hypotheses"])
    write_ledger(folder, ledger)
    loop.gate(folder, promote=["H2"])
    with pytest.raises(ValueError, match="status|reached the gate"):
        loop.verify(folder, receipt_keys=panels / "keys")


def sealed_run(panels):
    run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    folder = panels / "store/control"
    assert loop.verify(folder, receipt_keys=panels / "keys")["verified"]
    return folder


def test_a_loser_forced_through_the_gate_is_refused_even_after_its_trial_is_deleted(panels):
    """Attack: compute H2's test anyway, then remove only the trial folder and keep its numbers."""
    import shutil
    folder = sealed_run(panels)
    ledger = loop.read_ledger(folder)
    h2 = next(r for r in ledger["hypotheses"] if r["id"] == "H2")
    loop.open_test(folder, "H2", h2["decision"])
    assert (folder / "gate/outcomes/H2.json").is_file()
    with pytest.raises(ValueError, match="gate/trials holds"):
        loop.verify(folder, receipt_keys=panels / "keys")
    shutil.rmtree(folder / "gate/trials/H2")
    with pytest.raises(ValueError, match="gate/prepared holds"):
        loop.verify(folder, receipt_keys=panels / "keys")
    shutil.rmtree(folder / "gate/prepared/H2")
    with pytest.raises(ValueError, match="gate/outcomes holds"):
        loop.verify(folder, receipt_keys=panels / "keys")


def test_a_stray_gate_outcome_is_refused(panels):
    folder = sealed_run(panels)
    (folder / "gate/outcomes/H2.json").write_bytes((folder / "gate/outcomes/H1.json").read_bytes())
    with pytest.raises(ValueError, match="gate/outcomes holds"):
        loop.verify(folder, receipt_keys=panels / "keys")


def test_test_numbers_written_into_a_losers_outcome_are_refused(panels):
    folder = sealed_run(panels)
    path = folder / "outcomes/H2.json"
    outcome = json.loads(path.read_bytes())
    outcome["decision"]["metrics"]["test"] = {"rank_ic_mean": 0.895, "valid_ic_dates": 19, "net_mean": 0.001}
    path.write_bytes(loop.canonical_json_bytes(outcome))
    with pytest.raises(ValueError, match="outcomes/H2.json differs"):
        loop.verify(folder, receipt_keys=panels / "keys")


def test_test_numbers_written_into_the_report_are_refused(panels):
    folder = sealed_run(panels)
    report = folder / "REPORT.md"
    text = report.read_text()
    row = "| H2 | reversal | 1 | jegadeesh-1990 | -1.000 | sealed | sealed |"
    assert row in text
    report.write_text(text.replace(row, "| H2 | reversal | 1 | jegadeesh-1990 | -1.000 | 0.895 | 0.00100 |"))
    with pytest.raises(ValueError, match="REPORT.md"):
        loop.verify(folder, receipt_keys=panels / "keys")


@pytest.mark.parametrize("forge", [
    lambda r: r | {"decision": "SHIPPED"},
    lambda r: r | {"hypothesis": "H3"},
    lambda r: r | {"ledger_sha256": "0" * 64},
])
def test_forged_decision_records_are_refused(panels, forge):
    run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    folder = panels / "store/control"
    record = loop.decide(folder, "H1", "REJECTED_BY_HUMAN")
    (folder / "decisions/H1.json").write_bytes(loop.canonical_json_bytes(forge(record)))
    with pytest.raises(ValueError, match="decision"):
        loop.verify(folder, receipt_keys=panels / "keys")


def test_trial_swapped_from_other_data_is_refused(panels):
    """Attack: replace a trial with an internally consistent one built from different prices."""
    provider = ReplayProvider(EXAMPLES / "replay.hand-written.json")
    run(panels, "control", provider)
    run(panels, "regime", provider)
    control, regime = panels / "store/control", panels / "store/regime"
    import shutil
    # The panels differ only in the test split, so the swap that matters is the gate's trial.
    shutil.rmtree(control / "gate/trials/H1")
    shutil.copytree(regime / "gate/trials/H1", control / "gate/trials/H1")
    ledger = json.loads((control / "ledger.json").read_bytes())
    other = json.loads((regime / "ledger.json").read_bytes())
    swapped = next(r for r in other["hypotheses"] if r["id"] == "H1")["decision"]
    next(r for r in ledger["hypotheses"] if r["id"] == "H1")["decision"] = swapped
    write_ledger(control, ledger)
    with pytest.raises(ValueError, match="frozen card"):
        loop.verify(control, receipt_keys=panels / "keys")


def test_a_pre_gate_trial_must_not_have_seen_test(panels):
    """Attack: replace a sealed trial with the full-price one, which carries test numbers."""
    run(panels, "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"))
    control = panels / "store/control"
    import shutil
    shutil.rmtree(control / "trials/H1")
    shutil.copytree(control / "gate/trials/H1", control / "trials/H1")
    with pytest.raises(ValueError, match="frozen card"):
        loop.verify(control, receipt_keys=panels / "keys")


def test_test_metrics_exist_only_for_gate_survivors(panels):
    """The test split is sealed until the gate: no card that stops earlier has a test number,
    a test row in its trial's prices, or a trial on the full prices."""
    provider = ReplayProvider(EXAMPLES / "replay.hand-written.json")
    for name in ("control", "regime"):
        ledger = run(panels, name, provider)
        folder = panels / "store" / name
        test_start = loop.datetime.fromisoformat("2024-02-10T16:00:00+00:00")
        for r in ledger["hypotheses"]:
            if "card_sha256" not in r:
                continue
            trial = folder / "trials" / r["id"]
            dates = {loop.datetime.fromisoformat(line.split(",")[0].replace("Z", "+00:00"))
                     for line in (trial / "panel.csv").read_text().splitlines()[1:]}
            assert dates and max(dates) < test_start, r["id"]
            report = json.loads((trial / "report.json").read_bytes())
            assert report["test"]["summary"]["valid_ic_dates"] == 0, r["id"]
            if r["id"] == "H1":
                assert r["status"] == "AWAITING_HUMAN"
                assert r["decision"]["metrics"]["test"]["valid_ic_dates"] > 0
                assert (folder / "gate/trials/H1/report.json").is_file()
            else:
                assert r["decision"]["metrics"]["test"] is None, r["id"]
                assert r["decision"]["split_status"]["test"] == "SEALED"
                assert not (folder / "gate/trials" / r["id"]).exists()
                outcome = json.loads((folder / "outcomes" / f"{r['id']}.json").read_bytes())
                assert outcome["decision"]["metrics"]["test"] is None
        report = (folder / "REPORT.md").read_text()
        assert "| H2 | reversal | 1 | jegadeesh-1990 | -1.000 | sealed | sealed |" in report
        assert loop.verify(folder, receipt_keys=panels / "keys")["verified"]


def test_a_run_that_fails_midway_leaves_no_run_directory(panels):
    with pytest.raises(LookupError):
        run(panels, "control", Canned({"propose": PROPOSE, "critic:control:H1": LookupError("replay miss")}))
    assert not (panels / "store/control").exists()
    with pytest.raises(LookupError):
        run(panels, "regime", Canned({"propose": LookupError("replay miss")}))
    assert not (panels / "store/regime").exists()
