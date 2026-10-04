"""Tamper-evident, event-sourced state for local QuantOS research runs.

The log is intentionally small and standard-library-only.  Every line is one
canonical JSON event whose digest commits to the prior event.  The class never
rewrites a log; callers must start a new run/attempt after a terminal state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTERED = "REGISTERED"
SOURCE_RESOLVED = "SOURCE_RESOLVED"
SNAPSHOT_FROZEN = "SNAPSHOT_FROZEN"
DATA_QUALITY_PASSED = "DATA_QUALITY_PASSED"
QUARANTINED = "QUARANTINED"
CHEAP_FALSIFICATION = "CHEAP_FALSIFICATION"
BASELINE_COMPLETE = "BASELINE_COMPLETE"
EXPERIMENT_COMPLETE = "EXPERIMENT_COMPLETE"
INDEPENDENT_VALIDATION = "INDEPENDENT_VALIDATION"
REPORT_READY = "REPORT_READY"
HUMAN_REVIEW = "HUMAN_REVIEW"

CANONICAL_RUN_STATES = (
    REGISTERED,
    SOURCE_RESOLVED,
    SNAPSHOT_FROZEN,
    DATA_QUALITY_PASSED,
    QUARANTINED,
    CHEAP_FALSIFICATION,
    BASELINE_COMPLETE,
    EXPERIMENT_COMPLETE,
    INDEPENDENT_VALIDATION,
    REPORT_READY,
    HUMAN_REVIEW,
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    REGISTERED: frozenset({SOURCE_RESOLVED}),
    SOURCE_RESOLVED: frozenset({SNAPSHOT_FROZEN}),
    SNAPSHOT_FROZEN: frozenset({DATA_QUALITY_PASSED, QUARANTINED}),
    DATA_QUALITY_PASSED: frozenset({CHEAP_FALSIFICATION}),
    QUARANTINED: frozenset(),
    CHEAP_FALSIFICATION: frozenset({BASELINE_COMPLETE}),
    BASELINE_COMPLETE: frozenset({EXPERIMENT_COMPLETE}),
    EXPERIMENT_COMPLETE: frozenset({INDEPENDENT_VALIDATION}),
    INDEPENDENT_VALIDATION: frozenset({REPORT_READY}),
    REPORT_READY: frozenset({HUMAN_REVIEW}),
    HUMAN_REVIEW: frozenset(),
}

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "run_id",
        "occurred_at",
        "from_state",
        "to_state",
        "actor_hash",
        "artifact_hashes",
        "payload",
        "previous_hash",
        "event_hash",
    }
)


class StateLogError(RuntimeError):
    """Base exception for run-state log failures."""


class StateIntegrityError(StateLogError):
    """The persisted event stream is malformed or has been modified."""


class InvalidTransitionError(StateLogError):
    """A state is unknown or a transition is not allowed."""


class IdempotencyConflictError(StateLogError):
    """A retried operation differs from the already-persisted event."""


class StateValidationError(StateLogError, ValueError):
    """An input cannot be represented safely in the event log."""


@dataclass(frozen=True)
class StateEvent:
    schema_version: int
    sequence: int
    run_id: str
    occurred_at: str
    from_state: str | None
    to_state: str
    actor_hash: str
    artifact_hashes: dict[str, str]
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateEvent:
        return cls(**{name: value[name] for name in _EVENT_FIELDS})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "occurred_at": self.occurred_at,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "actor_hash": self.actor_hash,
            "artifact_hashes": dict(self.artifact_hashes),
            "payload": dict(self.payload),
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }


def hash_actor(actor: str) -> str:
    """Return the stable SHA-256 identity stored instead of a raw actor name."""

    if not isinstance(actor, str) or not actor.strip():
        raise StateValidationError("actor must be a non-empty string")
    return hashlib.sha256(actor.encode("utf-8")).hexdigest()


def hash_artifact(content: bytes) -> str:
    """Return a SHA-256 content digest suitable for ``artifact_hashes``."""

    if not isinstance(content, bytes):
        raise StateValidationError("artifact content must be bytes")
    return hashlib.sha256(content).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise StateValidationError(f"value is not canonical JSON: {exc}") from exc


def _normalize_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise StateValidationError("payload must be a JSON object")
    # Round-tripping also detaches caller-owned nested containers.
    return json.loads(_canonical_json(dict(payload)))


def _normalize_artifact_hashes(
    artifact_hashes: Mapping[str, str] | None,
) -> dict[str, str]:
    if artifact_hashes is None:
        return {}
    if not isinstance(artifact_hashes, Mapping):
        raise StateValidationError("artifact_hashes must be an object")
    normalized: dict[str, str] = {}
    for name, digest in artifact_hashes.items():
        if not isinstance(name, str) or not name:
            raise StateValidationError("artifact hash names must be non-empty strings")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower()):
            raise StateValidationError(f"artifact hash for {name!r} is not SHA-256")
        normalized[name] = digest.lower()
    return dict(sorted(normalized.items()))


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise StateIntegrityError("occurred_at must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StateIntegrityError("occurred_at is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise StateIntegrityError("occurred_at must include a timezone")


def _event_hash(event_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(event_without_hash).encode("utf-8")
    ).hexdigest()


def _new_event(
    *,
    sequence: int,
    run_id: str,
    from_state: str | None,
    to_state: str,
    actor_hash_value: str,
    artifact_hashes: dict[str, str],
    payload: dict[str, Any],
    previous_hash: str,
    clock: Callable[[], str],
) -> StateEvent:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "run_id": run_id,
        "occurred_at": clock(),
        "from_state": from_state,
        "to_state": to_state,
        "actor_hash": actor_hash_value,
        "artifact_hashes": artifact_hashes,
        "payload": payload,
        "previous_hash": previous_hash,
    }
    return StateEvent.from_dict({**base, "event_hash": _event_hash(base)})


def _event_bytes(event: StateEvent) -> bytes:
    return (_canonical_json(event.to_dict()) + "\n").encode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("failed to write state event")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry where the platform permits directory fsync."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_atomically(path: Path, data: bytes) -> bool:
    """Publish a fully-fsynced genesis log without replacing an existing log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            # Linking is an atomic create-if-absent operation on the same volume.
            os.link(temporary, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


class RunStateLog:
    """Append-only JSONL state machine for one research run."""

    def __init__(
        self, path: str | os.PathLike[str], *, clock: Callable[[], str] = _utc_now
    ):
        self.path = Path(path)
        self._clock = clock

    @classmethod
    def initialize(
        cls,
        path: str | os.PathLike[str],
        *,
        run_id: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        clock: Callable[[], str] = _utc_now,
    ) -> RunStateLog:
        """Atomically create a run or return an identical initialized run."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise StateValidationError("run_id must be a non-empty string")
        log = cls(path, clock=clock)
        normalized_payload = _normalize_payload(payload)
        normalized_artifacts = _normalize_artifact_hashes(artifact_hashes)
        actor_digest = hash_actor(actor)
        genesis = _new_event(
            sequence=0,
            run_id=run_id,
            from_state=None,
            to_state=REGISTERED,
            actor_hash_value=actor_digest,
            artifact_hashes=normalized_artifacts,
            payload=normalized_payload,
            previous_hash=GENESIS_HASH,
            clock=clock,
        )
        with _path_lock(log.path):
            if _create_atomically(log.path, _event_bytes(genesis)):
                return log
            events = log.replay()
            persisted = events[0]
            identical = (
                persisted.run_id == run_id
                and persisted.actor_hash == actor_digest
                and persisted.artifact_hashes == normalized_artifacts
                and persisted.payload == normalized_payload
            )
            if identical:
                return log
            raise IdempotencyConflictError(
                "state log already exists with a different initialization event"
            )

    def replay(self) -> list[StateEvent]:
        """Validate and return the complete event stream.

        Validation covers canonical byte encoding, schema, state transitions,
        sequence continuity, actor/artifact digests, and the full hash chain.
        """

        try:
            raw_log = self.path.read_bytes()
        except FileNotFoundError as exc:
            raise StateLogError(f"state log does not exist: {self.path}") from exc
        if not raw_log:
            raise StateIntegrityError("state log is empty")
        raw_lines = raw_log.splitlines(keepends=True)
        if any(not line.endswith(b"\n") for line in raw_lines):
            raise StateIntegrityError("state log contains a truncated event")

        events: list[StateEvent] = []
        expected_previous = GENESIS_HASH
        expected_state: str | None = None
        expected_run_id: str | None = None

        for sequence, raw_line in enumerate(raw_lines):
            try:
                decoded = raw_line[:-1].decode("utf-8")
                value = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StateIntegrityError(
                    f"event {sequence} is not valid JSON"
                ) from exc
            if not isinstance(value, dict) or frozenset(value) != _EVENT_FIELDS:
                raise StateIntegrityError(f"event {sequence} has an invalid schema")
            try:
                canonical_line = (_canonical_json(value) + "\n").encode("utf-8")
            except StateValidationError as exc:
                raise StateIntegrityError(
                    f"event {sequence} is not canonical JSON"
                ) from exc
            if raw_line != canonical_line:
                raise StateIntegrityError(
                    f"event {sequence} is not canonically encoded"
                )

            if value["schema_version"] != SCHEMA_VERSION:
                raise StateIntegrityError(f"event {sequence} has an unsupported schema")
            if type(value["sequence"]) is not int or value["sequence"] != sequence:
                raise StateIntegrityError(f"event {sequence} has a broken sequence")
            if not isinstance(value["run_id"], str) or not value["run_id"]:
                raise StateIntegrityError(f"event {sequence} has an invalid run_id")
            if expected_run_id is None:
                expected_run_id = value["run_id"]
            elif value["run_id"] != expected_run_id:
                raise StateIntegrityError(f"event {sequence} changes run_id")
            _validate_timestamp(value["occurred_at"])
            if not isinstance(value["actor_hash"], str) or not _SHA256_RE.fullmatch(
                value["actor_hash"]
            ):
                raise StateIntegrityError(f"event {sequence} has an invalid actor hash")
            try:
                artifacts = _normalize_artifact_hashes(value["artifact_hashes"])
                payload = _normalize_payload(value["payload"])
            except StateValidationError as exc:
                raise StateIntegrityError(
                    f"event {sequence} has invalid evidence"
                ) from exc
            if artifacts != value["artifact_hashes"] or payload != value["payload"]:
                raise StateIntegrityError(
                    f"event {sequence} has non-canonical evidence"
                )
            if value["previous_hash"] != expected_previous:
                raise StateIntegrityError(f"event {sequence} breaks the hash chain")
            if not isinstance(value["event_hash"], str) or not _SHA256_RE.fullmatch(
                value["event_hash"]
            ):
                raise StateIntegrityError(f"event {sequence} has an invalid event hash")
            unhashed = {key: item for key, item in value.items() if key != "event_hash"}
            if value["event_hash"] != _event_hash(unhashed):
                raise StateIntegrityError(f"event {sequence} digest does not match")

            to_state = value["to_state"]
            if to_state not in ALLOWED_TRANSITIONS:
                raise StateIntegrityError(f"event {sequence} uses an unknown state")
            if sequence == 0:
                if value["from_state"] is not None or to_state != REGISTERED:
                    raise StateIntegrityError(
                        "genesis event must initialize REGISTERED"
                    )
            else:
                if expected_state is None:
                    raise StateIntegrityError(f"event {sequence} has no prior state")
                if value["from_state"] != expected_state:
                    raise StateIntegrityError(
                        f"event {sequence} has the wrong source state"
                    )
                if to_state not in ALLOWED_TRANSITIONS[expected_state]:
                    raise StateIntegrityError(
                        f"event {sequence} is an invalid transition"
                    )

            event = StateEvent.from_dict(value)
            events.append(event)
            expected_previous = event.event_hash
            expected_state = event.to_state

        return events

    @property
    def current_state(self) -> str:
        return self.replay()[-1].to_state

    @property
    def run_id(self) -> str:
        return self.replay()[0].run_id

    def transition(
        self,
        to_state: str,
        *,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
    ) -> StateEvent:
        """Append one valid transition, or return an identical retried event."""

        if to_state not in ALLOWED_TRANSITIONS:
            raise InvalidTransitionError(
                f"unknown or prohibited run state: {to_state!r}"
            )
        normalized_payload = _normalize_payload(payload)
        normalized_artifacts = _normalize_artifact_hashes(artifact_hashes)
        actor_digest = hash_actor(actor)

        with _path_lock(self.path):
            events = self.replay()
            last = events[-1]
            if to_state == last.to_state:
                if (
                    last.actor_hash == actor_digest
                    and last.artifact_hashes == normalized_artifacts
                    and last.payload == normalized_payload
                ):
                    return last
                raise IdempotencyConflictError(
                    "same transition was retried with different event content"
                )
            if to_state not in ALLOWED_TRANSITIONS[last.to_state]:
                raise InvalidTransitionError(
                    f"transition {last.to_state} -> {to_state} is not allowed"
                )
            event = _new_event(
                sequence=len(events),
                run_id=last.run_id,
                from_state=last.to_state,
                to_state=to_state,
                actor_hash_value=actor_digest,
                artifact_hashes=normalized_artifacts,
                payload=normalized_payload,
                previous_hash=last.event_hash,
                clock=self._clock,
            )
            flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags)
            try:
                _write_all(descriptor, _event_bytes(event))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return event


__all__ = [
    "ALLOWED_TRANSITIONS",
    "CANONICAL_RUN_STATES",
    "GENESIS_HASH",
    "IdempotencyConflictError",
    "InvalidTransitionError",
    "RunStateLog",
    "StateEvent",
    "StateIntegrityError",
    "StateLogError",
    "StateValidationError",
    "hash_actor",
    "hash_artifact",
]
