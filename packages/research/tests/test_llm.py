"""Replay binding, the claude -p transport (faked, never called) and the broker seam."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from qrae import codex_broker
from qrae.llm import (
    CLAUDE_COMMAND,
    ClaudeCliProvider,
    ReplayMiss,
    ReplayProvider,
    ReplayThenLive,
    broker_runner,
    prompt_sha256,
    strip_fence,
)


def cli_json(text, models=("claude-test-1",), is_error=False):
    return json.dumps({"type": "result", "is_error": is_error, "result": text,
                       "modelUsage": {m: {"outputTokens": 1} for m in models}}).encode()


def replay_file(tmp_path, entries, provenance="HAND-WRITTEN FIXTURE: test"):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({"schema": "qrae.llm-replay/v1", "provenance": provenance, "entries": entries}))
    return path


def test_replay_is_bound_to_the_exact_prompt(tmp_path):
    path = replay_file(tmp_path, [{"key": "k", "prompt_sha256": prompt_sha256(b"ask"), "response": "answer"}])
    provider = ReplayProvider(path)
    assert provider.complete("k", b"ask") == "answer"
    with pytest.raises(ReplayMiss, match="stale"):
        provider.complete("k", b"ask, reworded")
    with pytest.raises(ReplayMiss, match="no replay entry"):
        provider.complete("other", b"ask")


@pytest.mark.parametrize("body", [
    {"schema": "qrae.llm-replay/v1", "provenance": "", "entries": []},
    {"schema": "other", "provenance": "x", "entries": []},
    {"schema": "qrae.llm-replay/v1", "provenance": "x", "entries": [
        {"key": "a", "prompt_sha256": "0", "response": ""}, {"key": "a", "prompt_sha256": "0", "response": ""}]},
])
def test_replay_file_must_state_provenance_and_unique_keys(tmp_path, body):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        ReplayProvider(path)


def test_claude_cli_runs_without_tools_and_records_a_replayable_session(tmp_path):
    calls = []

    def fake_run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout=cli_json("drafted"))

    provider = ClaudeCliProvider(runner=fake_run)
    assert provider.complete("k", b"prompt") == "drafted"
    command, options = calls[0]
    assert command == list(CLAUDE_COMMAND) and command[command.index("--tools") + 1] == ""
    assert options["input"] == b"prompt" and options["shell"] is False
    provider.save(tmp_path / "session.json")
    assert ReplayProvider(tmp_path / "session.json").complete("k", b"prompt") == "drafted"
    with pytest.raises(FileExistsError):
        provider.save(tmp_path / "session.json")


def test_claude_cli_failure_is_an_error_not_an_answer():
    provider = ClaudeCliProvider(runner=lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"partial"))
    with pytest.raises(RuntimeError):
        provider.complete("k", b"prompt")
    assert provider.recorded == []


def test_claude_cli_pins_the_model_and_labels_output_as_real():
    calls = []
    provider = ClaudeCliProvider(model="claude-test-1", runner=lambda c, **o: calls.append(c) or SimpleNamespace(
        returncode=0, stdout=cli_json("x")))
    assert provider.complete("k", b"p") == "x"
    assert calls[0][-2:] == ["--model", "claude-test-1"] and calls[0][:len(CLAUDE_COMMAND)] == list(CLAUDE_COMMAND)
    assert provider.provenance.startswith("REAL LLM OUTPUT") and "CLI-reported model ids: claude-test-1" in provider.provenance


def test_claude_cli_refuses_an_answer_from_another_model_or_an_error():
    other = ClaudeCliProvider(model="claude-test-1", runner=lambda c, **o: SimpleNamespace(
        returncode=0, stdout=cli_json("x", models=("claude-test-2",))))
    with pytest.raises(RuntimeError, match="not the pinned claude-test-1"):
        other.complete("k", b"p")
    failed = ClaudeCliProvider(runner=lambda c, **o: SimpleNamespace(returncode=0, stdout=cli_json("oops", is_error=True)))
    with pytest.raises(RuntimeError, match="reported an error"):
        failed.complete("k", b"p")
    plain = ClaudeCliProvider(runner=lambda c, **o: SimpleNamespace(returncode=0, stdout=b"not json"))
    with pytest.raises(RuntimeError, match="JSON result"):
        plain.complete("k", b"p")
    assert other.recorded == failed.recorded == plain.recorded == []


def test_claude_cli_budget_counts_failed_attempts_and_stops_before_calling():
    calls = []

    def fail(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"API Error: 429 rate limit")

    provider = ClaudeCliProvider(runner=fail, max_calls=1)
    with pytest.raises(RuntimeError, match="429"):
        provider.complete("a", b"p")
    with pytest.raises(RuntimeError, match="budget of 1 exhausted"):
        provider.complete("b", b"p")
    assert len(calls) == 1


def test_replay_then_live_asks_the_model_only_for_unrecorded_keys(tmp_path):
    path = replay_file(tmp_path, [{"key": "propose", "prompt_sha256": prompt_sha256(b"ask"), "response": "drafts"}],
                       provenance="REAL LLM OUTPUT: earlier session")
    calls = []
    live = ClaudeCliProvider(runner=lambda c, **o: calls.append(o["input"]) or SimpleNamespace(
        returncode=0, stdout=cli_json("critique")))
    provider = ReplayThenLive(ReplayProvider(path), live)
    assert provider.complete("propose", b"ask") == "drafts" and calls == []
    assert provider.complete("critic:H1", b"evidence") == "critique" and calls == [b"evidence"]
    with pytest.raises(ReplayMiss, match="stale"):  # a changed prompt is never sent live instead
        provider.complete("propose", b"ask, reworded")
    assert len(calls) == 1
    provider.save(tmp_path / "merged.json")
    merged = ReplayProvider(tmp_path / "merged.json")
    assert set(merged.entries) == {"propose", "critic:H1"} and "earlier session; then REAL LLM" in merged.provenance


def test_fence_stripping_touches_only_one_outer_fence():
    assert strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fence('  {"a": "```"}  ') == '{"a": "```"}'


def test_broker_seam_keeps_schema_and_transition_checks(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "evidence").mkdir(parents=True)
    (workspace / "evidence/e.json").write_text('{"metric": 1}')
    now = datetime.now(timezone.utc)
    order = codex_broker.prepare_work_order(
        workspace, task_id="t1", run_id="r1", task_kind="ADVERSARIAL_CRITIC", input_path="evidence/e.json",
        allowed_paths=["evidence/e.json"], expires_at=now + timedelta(hours=1), now=now)
    promote = json.dumps({"summary": "s", "claims": [], "artifact_refs": [], "requested_transition": "PROMOTE"})

    class One:
        def complete(self, key, prompt):
            assert key == "critic:t1" and b"ADVERSARIAL_CRITIC" in prompt
            # the provider sees the output schema Codex would get as --output-schema
            assert b'"requested_transition":{"const":"NONE"' in prompt
            return "```json\n" + promote + "\n```"

    result = codex_broker.run_work_order(order, workspace, "outbox", enabled=True, now=now,
                                         receipt_key_root=tmp_path / "keys", runner=broker_runner(One(), "critic:t1"))
    assert (result["status"], result["reason_code"]) == ("QUARANTINED", "MALFORMED_OR_FORBIDDEN_OUTPUT")


def test_receipt_key_root_can_be_kept_out_of_the_user_profile(monkeypatch, tmp_path):
    monkeypatch.undo()  # drop conftest's patch: exercise the real default lookup
    monkeypatch.setenv("QRAE_RECEIPT_KEY_ROOT", str(tmp_path / "keys"))
    assert codex_broker._default_receipt_key_root() == tmp_path / "keys"
