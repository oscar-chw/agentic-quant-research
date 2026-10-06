"""LLM transport for drafting tasks: a replay cache by default, a pinned open-weight model on request.

The live transport is OpenRouter's OpenAI-compatible chat completions endpoint,
serving the open weights ``Qwen/Qwen3.8-27B`` as ``qwen/qwen3.8-27b:free``. No
Anthropic or OpenAI model is used.

Neither transport carries research authority. Callers validate every response
against a fixed schema, deterministic code computes every metric, and a human
decides promotion. A replay entry is bound to the SHA-256 of the exact prompt,
so a changed prompt fails loudly instead of silently reusing an old answer.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .codex_broker import CODEX_OUTPUT_SCHEMA
from .polymarket import _RejectRedirects

REPLAY_SCHEMA = "qrae.llm-replay/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Pinned by protocol v2 amendment 5 (results/forward-2026-09/protocol.json), with the weights' HF revision.
OPENROUTER_MODEL = "qwen/qwen3.8-27b:free"
MAX_TOKENS = 8192
# The listing (model-evidence/openrouter-qwen38-free.json) offers efforts xhigh, medium and low,
# default xhigh, and not "none", which could be refused or silently run at xhigh. "low" is the
# smallest offered; exclude keeps the reasoning text out of the response, though it still counts
# against max_tokens.
REASONING = {"effort": "low", "exclude": True}
MAX_RESPONSE_BYTES = 256 * 1024
_FENCE = re.compile(r"\A```(?:json)?\s*\n(.*)\n```\s*\Z", re.S)


class ReplayMiss(LookupError):
    """No replay entry for this key, or the prompt changed since it was written."""


def prompt_sha256(prompt: bytes) -> str:
    return hashlib.sha256(prompt).hexdigest()


def strip_fence(text: str) -> str:
    """Models often wrap JSON in one Markdown fence; nothing else is rewritten."""
    match = _FENCE.match(text.strip())
    return match.group(1) if match else text.strip()


class ReplayProvider:
    name = "replay"

    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"schema", "provenance", "entries"} \
                or data["schema"] != REPLAY_SCHEMA:
            raise ValueError("replay file must be an exact " + REPLAY_SCHEMA + " object")
        if not isinstance(data["provenance"], str) or not data["provenance"].strip():
            raise ValueError("replay file must say where its responses came from")
        self.provenance = data["provenance"]
        self.entries = {}
        for entry in data["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"key", "prompt_sha256", "response"} \
                    or entry["key"] in self.entries:
                raise ValueError("replay entries need a unique key, prompt_sha256 and response")
            self.entries[entry["key"]] = entry

    def complete(self, key: str, prompt: bytes) -> str:
        entry = self.entries.get(key)
        if entry is None:
            raise ReplayMiss(f"no replay entry for {key!r}")
        actual = prompt_sha256(prompt)
        if entry["prompt_sha256"] != actual:
            raise ReplayMiss(f"stale replay entry for {key!r}: prompt sha256 is {actual}")
        return entry["response"]


class MissingApiKey(RuntimeError):
    """OPENROUTER_API_KEY is unset or empty. A live run must fail here, never skip the arm silently."""


class LiveCallFailed(RuntimeError):
    """A live call produced no usable answer: transport failure, HTTP error, refused body or spent budget."""


class LiveRunAborted(Exception):
    """A live call failed inside the broker seam. Not a RuntimeError, so the broker's runner-failure
    catch cannot turn it into a normal CRITIC_UNUSABLE run whose replay lacks the failed call."""


# One opener, no redirect handler: urllib would re-send the Authorization header to the Location host.
_OPENER = urllib.request.build_opener(_RejectRedirects())


def _post(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes]:
    """POST with the standard library; returns (status, body), an HTTP error's included."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


class OpenRouterProvider:
    """Calls one pinned open-weight model on OpenRouter; records what it returns.

    The request settings are fixed (temperature 0, reasoning effort low and excluded
    from the response, bounded ``max_tokens``) and pre-registered in protocol v2. An
    answer is refused, never recorded, on a non-200 status, an ``error`` in the body or in the choice, a
    ``finish_reason`` other than ``stop`` (a truncated answer is not an answer),
    empty content, a body over the size cap, or a response ``model`` that is
    neither the pinned id nor the pinned id without ``:free``. Each accepted
    response's ``model``, ``provider`` and ``id`` go into the provenance.
    ``max_calls`` is a hard budget on attempts, failed ones included, checked
    before each call. The key comes from OPENROUTER_API_KEY only and is never
    written into the provenance or a replay.
    """

    name = "openrouter"

    def __init__(self, *, timeout: int = 300, post: Callable = _post,
                 model: str | None = None, max_calls: int | None = None):
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise MissingApiKey("OPENROUTER_API_KEY is not set; no live call was made")
        self._key = key
        self.timeout = timeout
        self.post = post
        self.model = model or OPENROUTER_MODEL
        self.max_calls = max_calls
        self.calls = 0
        self.recorded: list[dict] = []
        self.responses: list[str] = []

    @property
    def provenance(self) -> str:
        seen = "; ".join(self.responses) or "none yet"
        return (f"REAL LLM OUTPUT from OpenRouter {OPENROUTER_URL} (pinned model {self.model}; "
                f"temperature 0, reasoning effort low (excluded), max_tokens {MAX_TOKENS}; "
                f"responses as key=model/provider/id: {seen})")

    def request_body(self, prompt: bytes) -> bytes:
        return json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt.decode("utf-8")}],
                           "temperature": 0, "max_tokens": MAX_TOKENS,
                           "reasoning": REASONING}).encode("utf-8")

    def complete(self, key: str, prompt: bytes) -> str:
        try:
            return self._complete(key, prompt)
        except LiveCallFailed:
            raise
        except RuntimeError as exc:
            raise LiveCallFailed(str(exc)) from exc

    def _complete(self, key: str, prompt: bytes) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise RuntimeError(f"live call budget of {self.max_calls} exhausted before {key!r}")
        self.calls += 1
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        status, raw = self.post(OPENROUTER_URL, self.request_body(prompt), headers, self.timeout)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RuntimeError("OpenRouter response exceeds the size cap")
        try:
            reply = json.loads(raw.decode("utf-8"))
        except ValueError:
            reply = None
        if status != 200:
            # The tail of the error message says why (429: the free tier's rate limit).
            error = reply.get("error") if isinstance(reply, dict) else None
            detail = error.get("message") if isinstance(error, dict) else raw.decode("utf-8", "replace")
            raise RuntimeError(f"OpenRouter returned HTTP {status}: {str(detail).strip()[-300:]}")
        if not isinstance(reply, dict):
            raise RuntimeError("OpenRouter did not return a JSON object")
        if reply.get("error"):
            raise RuntimeError(f"OpenRouter reported an error: {str(reply['error'])[:300]}")
        try:
            choice = reply["choices"][0]
            choice_error, finish, text = choice.get("error"), choice.get("finish_reason"), choice["message"]["content"]
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError("OpenRouter did not return a chat completion") from exc
        if choice_error:
            raise RuntimeError(f"OpenRouter reported an error in the choice: {str(choice_error)[:300]}")
        if finish != "stop":
            raise RuntimeError(f"OpenRouter finish_reason is {finish!r}, not 'stop'; refusing")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("OpenRouter returned no content")
        answered = reply.get("model")
        if answered not in {self.model, self.model.removesuffix(":free")}:
            raise RuntimeError(f"OpenRouter answered with {answered!r}, not the pinned {self.model}; refusing")
        self.responses.append(f"{key}={answered}/{reply.get('provider')}/{reply.get('id')}")
        self.recorded.append({"key": key, "prompt_sha256": prompt_sha256(prompt), "response": text})
        return text

    def save(self, path: str | Path) -> None:
        """Write the session as a replay file; refuses to overwrite an earlier one."""
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = {"schema": REPLAY_SCHEMA, "provenance": f"{self.provenance}, {stamp}",
                "entries": self.recorded}
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(body, handle, indent=2, sort_keys=True)
            handle.write("\n")


class ReplayThenLive:
    """Answers recorded keys from a replay and asks the live model only for the rest.

    This lets proposals be recorded live and committed before the data that will
    judge them exists, with the critic called live later. A recorded key whose
    prompt changed is still an error, never a silent live call. ``save`` writes the
    replayed and the new entries together, with both provenances.
    """

    name = "replay+openrouter"

    def __init__(self, replay: ReplayProvider, live: OpenRouterProvider):
        self.replay, self.live = replay, live

    @property
    def provenance(self) -> str:
        return f"{self.replay.provenance}; then {self.live.provenance}"

    def complete(self, key: str, prompt: bytes) -> str:
        if key in self.replay.entries:
            return self.replay.complete(key, prompt)
        return self.live.complete(key, prompt)

    def save(self, path: str | Path) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = {"schema": REPLAY_SCHEMA, "provenance": f"{self.provenance}, {stamp}",
                "entries": sorted([*self.replay.entries.values(), *self.live.recorded], key=lambda e: e["key"])}
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(body, handle, indent=2, sort_keys=True)
            handle.write("\n")


def broker_runner(provider, key: str) -> Callable:
    """Adapt a provider to ``codex_broker.run_work_order``'s runner seam.

    The broker still builds the bounded prompt, validates the schema, refuses
    any state transition and signs the receipt; only the transport changes. The
    broker's Codex argument vector is ignored because the provider owns its own,
    so the output schema is appended to the prompt instead.
    """

    def run(_command, *, input, **_options):
        # Codex receives the output schema as a file argument; a provider gets
        # it in the prompt, or a real model has no way to know the exact shape.
        prompt = input + b"\n\nReturn one JSON object that validates against this JSON Schema:\n" \
            + json.dumps(CODEX_OUTPUT_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            text = strip_fence(provider.complete(key, prompt))
        except LiveCallFailed as exc:
            raise LiveRunAborted(str(exc)) from exc
        return SimpleNamespace(returncode=0, stdout=text.encode("utf-8"), stderr=b"")

    return run
