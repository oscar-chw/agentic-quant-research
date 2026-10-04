"""LLM transport for drafting tasks: a replay cache by default, ``claude -p`` on request.

Neither transport carries research authority. Callers validate every response
against a fixed schema, deterministic code computes every metric, and a human
decides promotion. A replay entry is bound to the SHA-256 of the exact prompt,
so a changed prompt fails loudly instead of silently reusing an old answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .codex_broker import CODEX_OUTPUT_SCHEMA

REPLAY_SCHEMA = "qrae.llm-replay/v1"
# JSON output, because it names the model that actually answered (``modelUsage``).
CLAUDE_COMMAND = ("claude", "-p", "--tools", "", "--no-session-persistence", "--output-format", "json")
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


class ClaudeCliProvider:
    """Calls the local ``claude -p`` with tools disabled; records what it returns.

    ``model`` pins the model by its full id: an answer whose CLI-reported models
    (``modelUsage``) do not include it is refused, never recorded. Every reported
    id goes into the provenance. ``max_calls`` is a hard budget on attempts,
    failed ones included, checked before each call.
    """

    name = "claude-cli"

    def __init__(self, *, timeout: int = 300, runner: Callable = subprocess.run,
                 model: str | None = None, max_calls: int | None = None):
        self.timeout = timeout
        self.runner = runner
        self.model = model
        self.max_calls = max_calls
        self.calls = 0
        self.recorded: list[dict] = []
        self.reported_models: set[str] = set()

    @property
    def provenance(self) -> str:
        reported = ", ".join(sorted(self.reported_models)) or "none yet"
        return (f"REAL LLM OUTPUT from `claude -p` (requested model {self.model or 'CLI default'}; "
                f"CLI-reported model ids: {reported}) with tools disabled")

    def command(self) -> list[str]:
        return list(CLAUDE_COMMAND) + (["--model", self.model] if self.model else [])

    def complete(self, key: str, prompt: bytes) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise RuntimeError(f"live call budget of {self.max_calls} exhausted before {key!r}")
        self.calls += 1
        completed = self.runner(self.command(), input=prompt, capture_output=True,
                                timeout=self.timeout, check=False, shell=False)
        if completed.returncode != 0:
            # The tail of stderr says why (a 429 usage-window error, a logged-out CLI).
            detail = (getattr(completed, "stderr", None) or b"").decode("utf-8", "replace").strip()[-300:]
            raise RuntimeError(f"claude -p exited {completed.returncode}: {detail}")
        if len(completed.stdout) > MAX_RESPONSE_BYTES:
            raise RuntimeError("claude -p response exceeds the size cap")
        try:
            reply = json.loads(completed.stdout.decode("utf-8"))
            text, models = reply["result"], set(reply.get("modelUsage") or {})
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("claude -p did not return its JSON result object") from exc
        if reply.get("is_error") or not isinstance(text, str):
            raise RuntimeError(f"claude -p reported an error: {str(text)[:300]}")
        if self.model and self.model not in models:
            raise RuntimeError(f"claude -p answered with {sorted(models)}, not the pinned {self.model}; refusing")
        self.reported_models |= models
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
    """Answers recorded keys from a replay and asks ``claude -p`` only for the rest.

    This lets proposals be recorded live and committed before the data that will
    judge them exists, with the critic called live later. A recorded key whose
    prompt changed is still an error, never a silent live call. ``save`` writes the
    replayed and the new entries together, with both provenances.
    """

    name = "replay+claude-cli"

    def __init__(self, replay: ReplayProvider, live: ClaudeCliProvider):
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
        text = strip_fence(provider.complete(key, prompt))
        return SimpleNamespace(returncode=0, stdout=text.encode("utf-8"), stderr=b"")

    return run
