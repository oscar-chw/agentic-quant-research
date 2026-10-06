import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import qrae.codex_broker as broker
from qrae.codex_broker import (
    BrokerValidationError,
    OutboxConflictError,
    prepare_work_order,
    run_work_order,
    validate_result_envelope,
    validate_work_order,
)

NOW = datetime(2026, 7, 11, 1, 0, tzinfo=timezone.utc)


def make_order(root: Path, *, task_id: str = "task-001", **overrides):
    packet = root / "runs" / "run-001" / "packet.md"
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text("# Evidence\nNo executable instructions.\n", encoding="utf-8")
    values = {
        "task_id": task_id,
        "run_id": "run-001",
        "task_kind": "ADVERSARIAL_CRITIC",
        "input_path": "runs/run-001/packet.md",
        "allowed_paths": ["runs/run-001"],
        "expires_at": NOW + timedelta(hours=1),
        "now": NOW,
    }
    values.update(overrides)
    return prepare_work_order(root, **values)


def valid_payload():
    return {
        "summary": "The evidence remains insufficient for promotion.",
        "claims": [
            {
                "text": "Depth sensitivity is not yet demonstrated.",
                "classification": "UNSUPPORTED",
                "artifact_refs": ["runs/run-001/packet.md"],
            }
        ],
        "artifact_refs": ["runs/run-001/packet.md"],
        "requested_transition": "NONE",
    }


def test_output_schema_const_has_explicit_structured_output_type():
    assert broker.CODEX_OUTPUT_SCHEMA["properties"]["requested_transition"] == {
        "type": "string",
        "const": "NONE",
    }


def test_prepare_and_validate_bind_immutable_input(tmp_path):
    order = make_order(tmp_path)
    assert order["schema_version"] == "qrae.codex-work-order.v2"
    assert order["run_manifest"] is None
    assert validate_work_order(order, tmp_path, now=NOW) == order
    assert len(order["input"]["sha256"]) == 64
    assert len(order["work_order_sha256"]) == 64

    (tmp_path / order["input"]["path"]).write_text("mutated", encoding="utf-8")
    with pytest.raises(BrokerValidationError, match=r"input (size|hash) changed"):
        validate_work_order(order, tmp_path, now=NOW)


def test_run_rehashes_captured_input_after_work_order_validation(tmp_path, monkeypatch):
    order = make_order(tmp_path)
    packet = tmp_path / order["input"]["path"]
    original_validate = broker.validate_work_order
    runner_called = False

    def validate_then_swap(*args, **kwargs):
        validated = original_validate(*args, **kwargs)
        packet.write_text("mutated after validation", encoding="utf-8")
        return validated

    def forbidden_runner(*args, **kwargs):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("unverified evidence reached Codex")

    monkeypatch.setattr(broker, "validate_work_order", validate_then_swap)
    with pytest.raises(BrokerValidationError, match=r"input (size|hash) changed"):
        run_work_order(
            order,
            tmp_path,
            "outbox",
            enabled=True,
            runner=forbidden_runner,
            now=NOW,
        )
    assert not runner_called


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"input_path": "../packet.md"}, "relative path"),
        ({"task_kind": "RUN_COMMAND"}, "unsupported task_kind"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"expires_at": NOW - timedelta(seconds=1)}, "after issued_at"),
    ],
)
def test_prepare_rejects_path_task_expiry_and_budget(tmp_path, changes, message):
    with pytest.raises(BrokerValidationError, match=message):
        make_order(tmp_path, **changes)


@pytest.mark.parametrize(
    "secret_text",
    [
        "api_key = abcdefghijklmnopqrstuvwxyz",
        "clientSecret = abcdefghijklmnopqrstuvwxyz",
        "authToken = abcdefghijklmnopqrstuvwxyz",
        "accessKey = abcdefghijklmnopqrstuvwxyz",
        "x_api_key = abcdefghijklmnopqrstuvwxyz",
        "aws_secret_access_key = abcdefghijklmnopqrstuvwxyz",
        "client_secret = abcdefghijklmnopqrstuvwxyz",
        '"x_api_key": "abcdefghijklmnopqrstuvwxyz"',
        '"aws_secret_access_key": "abcdefghijklmnopqrstuvwxyz"',
        '"client_secret": "abcdefghijklmnopqrstuvwxyz"',
    ],
)
def test_prepare_rejects_secret_like_input(tmp_path, secret_text):
    order = make_order(tmp_path)
    packet = tmp_path / order["input"]["path"]
    packet.write_text(secret_text, encoding="utf-8")
    with pytest.raises(BrokerValidationError, match="secret-like"):
        prepare_work_order(
            tmp_path,
            task_id="task-secret",
            run_id="run-001",
            task_kind="REPORT_DRAFT",
            input_path="runs/run-001/packet.md",
            allowed_paths=["runs/run-001"],
            expires_at=NOW + timedelta(hours=1),
            now=NOW,
        )


def test_default_is_deferred_atomic_and_idempotent(tmp_path):
    order = make_order(tmp_path)
    first = run_work_order(order, tmp_path, "outbox", now=NOW)
    assert first["status"] == "DEFERRED_NO_CODEX"
    assert first["requested_transition"] == "NONE"
    path = tmp_path / "outbox" / "run-001" / "task-001.json"
    assert json.loads(path.read_text(encoding="utf-8")) == first
    assert not list(path.parent.glob("*.tmp"))

    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("idempotent replay invoked Codex")

    second = run_work_order(
        order, tmp_path, "outbox", enabled=True, runner=should_not_run, now=NOW
    )
    assert second == first
    assert not called


def test_existing_ready_result_is_fully_revalidated(tmp_path):
    order = make_order(tmp_path)

    def valid_runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(valid_payload()), stderr=""
        )

    run_work_order(
        order, tmp_path, "outbox", enabled=True, runner=valid_runner, now=NOW
    )
    result_path = tmp_path / "outbox" / "run-001" / "task-001.json"
    forged = json.loads(result_path.read_text(encoding="utf-8"))
    forged["payload"]["summary"] = ""
    forged["payload_sha256"] = hashlib.sha256(
        json.dumps(
            forged["payload"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    result_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(OutboxConflictError, match="failed validation"):
        run_work_order(order, tmp_path, "outbox", now=NOW)


def test_runner_uses_fixed_no_shell_command_and_accepts_bounded_draft(
    tmp_path, monkeypatch
):
    order = make_order(tmp_path)
    observed = {}
    monkeypatch.setenv("EXCHANGE_API_KEY", "must-not-cross-boundary")

    def fake_runner(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(valid_payload()), stderr=""
        )

    result = run_work_order(
        order, tmp_path, "outbox", enabled=True, runner=fake_runner, now=NOW
    )
    assert result["status"] == "DRAFT_READY"
    assert result["requested_transition"] == "NONE"
    assert validate_result_envelope(result, order) == result
    assert observed["command"][:6] == [
        "codex",
        "exec",
        "--strict-config",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
    ]
    assert "--sandbox" not in observed["command"]
    assert "project_doc_max_bytes=0" in observed["command"]
    assert 'web_search="disabled"' in observed["command"]
    assert 'default_permissions="qrae_evidence"' in observed["command"]
    assert (
        'permissions.qrae_evidence.filesystem={":minimal"="read",'
        '":workspace_roots"={"."="read"}}'
    ) in observed["command"]
    assert (
        observed["command"][observed["command"].index("--disable") + 1] == "shell_tool"
    )
    assert observed["command"][-1] == "-"
    assert observed["shell"] is False
    assert observed["check"] is False
    assert observed["capture_output"] is True
    assert "EXCHANGE_API_KEY" not in observed["env"]
    assert observed["cwd"] != str(tmp_path)
    assert observed["command"][observed["command"].index("--cd") + 1] == observed["cwd"]
    assert observed["env"]["USERPROFILE"].startswith(observed["cwd"])
    assert b"No executable instructions" in observed["input"]
    assert "No executable instructions" not in " ".join(observed["command"])


@pytest.mark.parametrize(
    "runner, reason",
    [
        (
            lambda command, **kwargs: subprocess.CompletedProcess(
                command, 0, stdout="not-json", stderr=""
            ),
            "MALFORMED_OR_FORBIDDEN_OUTPUT",
        ),
        (
            lambda command, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(command, kwargs["timeout"])
            ),
            "CODEX_TIMEOUT",
        ),
    ],
)
def test_malformed_and_timeout_results_are_quarantined(tmp_path, runner, reason):
    order = make_order(tmp_path)
    result = run_work_order(
        order, tmp_path, "outbox", enabled=True, runner=runner, now=NOW
    )
    assert result["status"] == "QUARANTINED"
    assert result["reason_code"] == reason
    assert result["payload"] is None
    assert result["requested_transition"] == "NONE"


def test_promotion_and_secret_output_are_quarantined(tmp_path):
    for task_id, mutate in (
        (
            "promote",
            lambda payload: payload.update(requested_transition="REPORT_READY"),
        ),
        ("secret", lambda payload: payload.update(summary="api_key=abcdefghijklmnop")),
    ):
        order = make_order(tmp_path, task_id=task_id)
        payload = valid_payload()
        mutate(payload)

        def fake_runner(command, payload=payload, **kwargs):
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(payload), stderr=""
            )

        result = run_work_order(
            order, tmp_path, "outbox", enabled=True, runner=fake_runner, now=NOW
        )
        assert result["status"] == "QUARANTINED"
        assert result["requested_transition"] == "NONE"


def test_result_envelope_rejects_any_state_promotion(tmp_path):
    order = make_order(tmp_path)
    result = run_work_order(order, tmp_path, "outbox", now=NOW)
    promoted = copy.deepcopy(result)
    promoted["requested_transition"] = "REPORT_READY"
    with pytest.raises(BrokerValidationError, match="cannot promote"):
        validate_result_envelope(promoted, order)

    ready_order = make_order(tmp_path, task_id="ready")

    def valid_runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(valid_payload()), stderr=""
        )

    ready = run_work_order(
        ready_order,
        tmp_path,
        "outbox",
        enabled=True,
        runner=valid_runner,
        now=NOW,
    )
    nested = copy.deepcopy(ready)
    nested["payload"]["requested_transition"] = "REPORT_READY"
    nested["payload_sha256"] = hashlib.sha256(
        json.dumps(
            nested["payload"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(BrokerValidationError, match="payload cannot promote"):
        validate_result_envelope(nested, ready_order)


def test_unavailable_runner_defers_without_exposing_error_output(tmp_path):
    order = make_order(tmp_path)

    def unavailable(command, **kwargs):
        raise FileNotFoundError("local codex executable unavailable")

    result = run_work_order(
        order, tmp_path, "outbox", enabled=True, runner=unavailable, now=NOW
    )
    assert result["status"] == "DEFERRED_NO_CODEX"
    assert result["reason_code"] == "CODEX_UNAVAILABLE"
    assert result["payload"] is None


def test_result_directory_link_cannot_redirect_outbox_outside_workspace(tmp_path):
    order = make_order(tmp_path)
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    link = outbox / order["run_id"]
    if os.name == "nt":
        created = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(external)],
            capture_output=True,
            check=False,
        )
        if created.returncode != 0:
            pytest.skip("directory junctions are unavailable on this platform")
    else:
        link.symlink_to(external, target_is_directory=True)

    with pytest.raises(BrokerValidationError, match="escapes"):
        run_work_order(order, tmp_path, outbox, now=NOW)
    assert not (external / f"{order['task_id']}.json").exists()


def test_subprocess_output_is_terminated_at_streaming_limit(tmp_path):
    script = (
        "import sys,time; "
        "sys.stdout.buffer.write(b'x' * (4 * 1024 * 1024)); "
        "sys.stdout.buffer.flush(); time.sleep(10)"
    )
    started = time.monotonic()
    with pytest.raises(broker.CodexOutputTooLarge):
        broker._bounded_subprocess_run(
            [sys.executable, "-c", script],
            input_bytes=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ,
            max_output_bytes=1024,
        )
    assert time.monotonic() - started < 5
