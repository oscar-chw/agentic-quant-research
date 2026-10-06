import hashlib
import json
from itertools import pairwise

import pytest

from qrae.state import (
    ALLOWED_TRANSITIONS,
    CANONICAL_RUN_STATES,
    IdempotencyConflictError,
    InvalidTransitionError,
    RunStateLog,
    StateIntegrityError,
    StateValidationError,
    hash_actor,
    hash_artifact,
)

ACTOR = "local-test-runner"
ARTIFACT = hashlib.sha256(b"snapshot-v1").hexdigest()


def initialized_log(tmp_path):
    return RunStateLog.initialize(
        tmp_path / "run.jsonl",
        run_id="run-001",
        actor=ACTOR,
        payload={"mandate": "read-only research"},
        artifact_hashes={"mandate.json": ARTIFACT},
    )


def test_happy_path_is_canonical_hash_chained_and_has_no_live_states(tmp_path):
    log = initialized_log(tmp_path)
    path = log.path
    states = [
        "SOURCE_RESOLVED",
        "SNAPSHOT_FROZEN",
        "DATA_QUALITY_PASSED",
        "CHEAP_FALSIFICATION",
        "BASELINE_COMPLETE",
        "EXPERIMENT_COMPLETE",
        "INDEPENDENT_VALIDATION",
        "REPORT_READY",
        "HUMAN_REVIEW",
    ]
    for state in states:
        log.transition(
            state,
            actor=ACTOR,
            payload={"gate": state.lower()},
            artifact_hashes={f"{state.lower()}.json": ARTIFACT},
        )

    events = log.replay()
    assert [event.to_state for event in events] == ["REGISTERED", *states]
    assert log.current_state == "HUMAN_REVIEW"
    assert log.run_id == "run-001"
    assert "LIVE" not in CANONICAL_RUN_STATES
    assert "SHADOW" not in CANONICAL_RUN_STATES
    assert set(ALLOWED_TRANSITIONS) == set(CANONICAL_RUN_STATES)
    assert events[0].actor_hash == hash_actor(ACTOR)
    assert ACTOR.encode() not in path.read_bytes()
    for prior, current in pairwise(events):
        assert current.previous_hash == prior.event_hash
    with pytest.raises(InvalidTransitionError):
        log.transition("PAPER_CANDIDATE", actor=ACTOR)


def test_quarantined_is_terminal_and_live_states_are_rejected(tmp_path):
    log = initialized_log(tmp_path)
    log.transition("SOURCE_RESOLVED", actor=ACTOR)
    log.transition("SNAPSHOT_FROZEN", actor=ACTOR)
    log.transition("QUARANTINED", actor=ACTOR, payload={"reason": "stale"})

    with pytest.raises(InvalidTransitionError):
        log.transition("DATA_QUALITY_PASSED", actor=ACTOR)
    with pytest.raises(InvalidTransitionError):
        log.transition("LIVE", actor=ACTOR)
    assert log.current_state == "QUARANTINED"
    assert len(log.replay()) == 4


def test_skips_and_post_terminal_transitions_are_rejected_without_append(tmp_path):
    log = initialized_log(tmp_path)
    before = log.path.read_bytes()
    with pytest.raises(InvalidTransitionError):
        log.transition("REPORT_READY", actor=ACTOR)
    assert log.path.read_bytes() == before


def test_transition_retry_is_idempotent_only_for_identical_content(tmp_path):
    log = initialized_log(tmp_path)
    kwargs = {
        "actor": ACTOR,
        "payload": {"source": "public", "count": 3},
        "artifact_hashes": {"source-plan.json": ARTIFACT},
    }
    first = log.transition("SOURCE_RESOLVED", **kwargs)
    size = log.path.stat().st_size
    retried = log.transition("SOURCE_RESOLVED", **kwargs)
    assert retried == first
    assert log.path.stat().st_size == size

    with pytest.raises(IdempotencyConflictError):
        log.transition(
            "SOURCE_RESOLVED",
            actor=ACTOR,
            payload={"source": "changed"},
            artifact_hashes={"source-plan.json": ARTIFACT},
        )
    assert log.path.stat().st_size == size


def test_atomic_initialization_is_idempotent_and_never_overwrites(tmp_path):
    path = tmp_path / "run.jsonl"
    first = RunStateLog.initialize(path, run_id="same", actor=ACTOR, payload={"x": 1})
    original = path.read_bytes()
    second = RunStateLog.initialize(path, run_id="same", actor=ACTOR, payload={"x": 1})
    assert second.replay() == first.replay()
    assert path.read_bytes() == original

    first.transition("SOURCE_RESOLVED", actor=ACTOR)
    progressed = path.read_bytes()
    resumed = RunStateLog.initialize(path, run_id="same", actor=ACTOR, payload={"x": 1})
    assert resumed.current_state == "SOURCE_RESOLVED"
    assert path.read_bytes() == progressed

    with pytest.raises(IdempotencyConflictError):
        RunStateLog.initialize(path, run_id="other", actor=ACTOR, payload={"x": 1})
    assert path.read_bytes() == progressed


def test_payload_tampering_and_chain_tampering_are_detected(tmp_path):
    log = initialized_log(tmp_path)
    log.transition("SOURCE_RESOLVED", actor=ACTOR, payload={"source": "A"})
    lines = log.path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[1])
    event["payload"]["source"] = "B"
    lines[1] = json.dumps(
        event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(StateIntegrityError, match="digest"):
        log.replay()


def test_truncation_and_noncanonical_reencoding_are_detected(tmp_path):
    log = initialized_log(tmp_path)
    original = log.path.read_bytes()
    log.path.write_bytes(original.rstrip(b"\n"))
    with pytest.raises(StateIntegrityError, match="truncated"):
        log.replay()

    log.path.write_bytes(b" " + original)
    with pytest.raises(StateIntegrityError, match="canonically"):
        log.replay()


def test_actor_and_artifact_hash_validation(tmp_path):
    assert hash_artifact(b"snapshot-v1") == ARTIFACT
    with pytest.raises(StateValidationError):
        RunStateLog.initialize(tmp_path / "bad-actor.jsonl", run_id="r", actor="")
    log = initialized_log(tmp_path)
    with pytest.raises(StateValidationError):
        log.transition(
            "SOURCE_RESOLVED",
            actor=ACTOR,
            artifact_hashes={"snapshot": "not-a-sha256"},
        )


def test_initial_and_append_writes_call_fsync(tmp_path, monkeypatch):
    calls = []
    real_fsync = __import__("os").fsync

    def recording_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr("qrae.state.os.fsync", recording_fsync)
    log = RunStateLog.initialize(tmp_path / "fsync.jsonl", run_id="r", actor=ACTOR)
    after_initialize = len(calls)
    log.transition("SOURCE_RESOLVED", actor=ACTOR)
    assert after_initialize >= 1
    assert len(calls) > after_initialize
