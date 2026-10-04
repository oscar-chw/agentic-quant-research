"""Replay binding, the OpenRouter transport (faked, never called) and the broker seam."""

import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from qrae import codex_broker, llm
from qrae.llm import (
    MAX_RESPONSE_BYTES,
    MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    MissingApiKey,
    OpenRouterProvider,
    ReplayMiss,
    ReplayProvider,
    ReplayThenLive,
    broker_runner,
    prompt_sha256,
    strip_fence,
)

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


KEY = "fake-openrouter-test-key-never-sent"
PINNED = "qwen/qwen3.8-27b:free"


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)


def completion(text="drafted", model="qwen/qwen3.8-27b", finish="stop", **extra):
    choice = {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}
    return json.dumps({"id": "gen-1", "provider": "SomeHost", "model": model, "choices": [choice], **extra}).encode()


def fake_post(*replies):
    """Answers each POST with the next (status, body) and records what was sent; never touches the network."""
    sent, queue = [], list(replies)

    def post(url, body, headers, timeout):
        sent.append({"url": url, "body": json.loads(body), "headers": headers, "timeout": timeout})
        return queue.pop(0)

    post.sent = sent
    return post


def test_openrouter_sends_the_pinned_request_and_records_a_replayable_session(tmp_path):
    post = fake_post((200, completion("drafted")))
    provider = OpenRouterProvider(post=post)
    assert provider.model == OPENROUTER_MODEL == PINNED
    assert provider.complete("k", "prompt é".encode()) == "drafted"
    [call] = post.sent
    assert call["url"] == OPENROUTER_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert call["body"] == {"model": PINNED, "messages": [{"role": "user", "content": "prompt é"}],
                            "temperature": 0, "max_tokens": MAX_TOKENS, "reasoning": {"effort": "none"}}
    assert 0 < MAX_TOKENS <= 8192 and call["timeout"] == provider.timeout
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    provider.save(tmp_path / "session.json")
    assert ReplayProvider(tmp_path / "session.json").complete("k", "prompt é".encode()) == "drafted"
    with pytest.raises(FileExistsError):
        provider.save(tmp_path / "session.json")


def test_openrouter_records_model_provider_and_id_but_never_the_key(tmp_path):
    provider = OpenRouterProvider(post=fake_post((200, completion("x"))))
    assert provider.complete("propose", b"p") == "x"
    assert provider.provenance.startswith("REAL LLM OUTPUT")
    assert "propose=qwen/qwen3.8-27b/SomeHost/gen-1" in provider.provenance
    assert "temperature 0" in provider.provenance and "reasoning effort none" in provider.provenance
    provider.save(tmp_path / "session.json")
    saved = (tmp_path / "session.json").read_text()
    assert KEY not in saved and KEY not in provider.provenance and KEY not in repr(provider)
    assert "openrouter-test-key" not in saved


def test_openrouter_accepts_the_pinned_id_with_and_without_free():
    for answered in (PINNED, "qwen/qwen3.8-27b"):
        provider = OpenRouterProvider(post=fake_post((200, completion("x", model=answered))))
        assert provider.complete("k", b"p") == "x"


@pytest.mark.parametrize("answered", ["qwen/qwen3.8-14b", "qwen/qwen3.8-27b:nitro", "free", None, ""])
def test_openrouter_refuses_an_answer_from_another_model(answered):
    provider = OpenRouterProvider(post=fake_post((200, completion("x", model=answered))))
    with pytest.raises(RuntimeError, match="not the pinned qwen/qwen3.8-27b:free"):
        provider.complete("k", b"p")
    assert provider.recorded == [] and provider.responses == []


@pytest.mark.parametrize("status, body, match", [
    (429, json.dumps({"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-min"}}).encode(),
     "HTTP 429: Rate limit exceeded: free-models-per-min"),
    (401, json.dumps({"error": {"code": 401, "message": "No auth credentials found"}}).encode(),
     "HTTP 401: No auth credentials found"),
    (502, b"<html>Bad gateway</html>", "HTTP 502: <html>Bad gateway</html>"),
    (200, json.dumps({"error": {"code": 500, "message": "upstream down"}}).encode(), "reported an error"),
    (200, completion(error={"message": "late"}), "reported an error"),
    (200, json.dumps({"id": "g", "model": "qwen/qwen3.8-27b", "choices": [
        {"finish_reason": "error", "error": {"code": 502, "message": "provider died"},
         "message": {"content": "partial"}}]}).encode(), "error in the choice"),
    (200, completion("cut off mid", finish="length"), "finish_reason is 'length'"),
    (200, completion("x", finish=None), "finish_reason is None"),
    (200, completion(""), "no content"),
    (200, completion("   \n"), "no content"),
    (200, completion(None), "no content"),
    (200, completion(["x"]), "no content"),
    (200, b"not json", "JSON object"),
    (200, b"[1]", "JSON object"),
    (200, json.dumps({"model": "qwen/qwen3.8-27b", "choices": []}).encode(), "chat completion"),
    (200, b"x" * (MAX_RESPONSE_BYTES + 1), "size cap"),
])
def test_openrouter_refuses_and_records_nothing(status, body, match):
    provider = OpenRouterProvider(post=fake_post((status, body)))
    with pytest.raises(RuntimeError, match=match):
        provider.complete("k", b"p")
    assert provider.recorded == [] and provider.responses == []


def test_openrouter_size_cap_is_inclusive():
    padded = completion("x")
    padded = padded[:-1] + b" " * (MAX_RESPONSE_BYTES - len(padded)) + b"}"
    assert len(padded) == MAX_RESPONSE_BYTES
    assert OpenRouterProvider(post=fake_post((200, padded))).complete("k", b"p") == "x"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_openrouter_without_a_key_fails_loudly_before_any_call(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", value)
    post = fake_post()
    with pytest.raises(MissingApiKey, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(post=post)
    assert post.sent == []


def test_openrouter_budget_counts_failed_attempts_and_stops_before_calling():
    post = fake_post((429, b'{"error": {"message": "Rate limit exceeded"}}'), (200, completion("unused")))
    provider = OpenRouterProvider(post=post, max_calls=1)
    with pytest.raises(RuntimeError, match="429"):
        provider.complete("a", b"p")
    with pytest.raises(RuntimeError, match="budget of 1 exhausted"):
        provider.complete("b", b"p")
    assert len(post.sent) == 1 and provider.calls == 1


def test_openrouter_default_transport_reports_a_network_failure(monkeypatch):
    def refuse(request, timeout):
        assert request.get_method() == "POST" and request.full_url == OPENROUTER_URL and timeout == 7
        raise llm.urllib.error.URLError("no route")

    monkeypatch.setattr(llm.urllib.request, "urlopen", refuse)
    provider = OpenRouterProvider(timeout=7)
    with pytest.raises(RuntimeError, match="request failed: .*no route"):
        provider.complete("k", b"p")
    assert provider.recorded == []


def test_openrouter_default_transport_returns_an_http_error_status(monkeypatch):
    body = b'{"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}}'

    def limited(request, timeout):
        raise llm.urllib.error.HTTPError(OPENROUTER_URL, 429, "Too Many Requests", {}, io.BytesIO(body))

    monkeypatch.setattr(llm.urllib.request, "urlopen", limited)
    provider = OpenRouterProvider()
    with pytest.raises(RuntimeError, match="HTTP 429: Rate limit exceeded: free-models-per-day"):
        provider.complete("k", b"p")
    assert provider.recorded == []


def test_replay_then_live_asks_the_model_only_for_unrecorded_keys(tmp_path):
    path = replay_file(tmp_path, [{"key": "propose", "prompt_sha256": prompt_sha256(b"ask"), "response": "drafts"}],
                       provenance="REAL LLM OUTPUT: earlier session")
    post = fake_post((200, completion("critique")))
    provider = ReplayThenLive(ReplayProvider(path), OpenRouterProvider(post=post))
    assert provider.complete("propose", b"ask") == "drafts" and post.sent == []
    assert provider.complete("critic:H1", b"evidence") == "critique"
    assert [c["body"]["messages"][0]["content"] for c in post.sent] == ["evidence"]
    with pytest.raises(ReplayMiss, match="stale"):  # a changed prompt is never sent live instead
        provider.complete("propose", b"ask, reworded")
    assert len(post.sent) == 1
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
