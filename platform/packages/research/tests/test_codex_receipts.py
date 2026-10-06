from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import qrae.codex_broker as broker
from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.codex_broker import (
    RECEIPT_SCHEMA,
    RESULT_SCHEMA,
    BrokerValidationError,
    prepare_work_order,
    run_work_order,
    validate_result_envelope,
    validate_result_receipt,
)

NOW = datetime(2026, 7, 11, 1, 0, tzinfo=timezone.utc)


def _payload(run_id: str = "run-001") -> dict[str, object]:
    evidence = f"runs/{run_id}/packet.md"
    return {
        "summary": "The evidence remains insufficient for promotion.",
        "claims": [
            {
                "text": "Execution sensitivity has not been demonstrated.",
                "classification": "UNSUPPORTED",
                "artifact_refs": [evidence],
            }
        ],
        "artifact_refs": [evidence],
        "requested_transition": "NONE",
    }


def _ready_result(
    workspace: Path,
    key_root: Path,
    *,
    task_id: str = "task-001",
    run_id: str = "run-001",
) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    evidence = workspace / "runs" / run_id / "packet.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("# Evidence\nNo executable instructions.\n", encoding="utf-8")
    relative = evidence.relative_to(workspace).as_posix()
    order = prepare_work_order(
        workspace,
        task_id=task_id,
        run_id=run_id,
        task_kind="ADVERSARIAL_CRITIC",
        input_path=relative,
        allowed_paths=[relative],
        expires_at=NOW + timedelta(hours=1),
        now=NOW,
    )

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(_payload(run_id)), stderr=""
        )

    outbox = workspace / "outbox"
    result = run_work_order(
        order,
        workspace,
        outbox,
        enabled=True,
        runner=runner,
        now=NOW,
        receipt_key_root=key_root,
    )
    receipt_path = outbox / run_id / f"{task_id}.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return order, result, receipt, receipt_path


def test_hmac_receipt_authenticates_exact_result_without_exporting_key(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key_root = tmp_path / "receipt-keys"
    order, result, receipt, receipt_path = _ready_result(workspace, key_root)

    assert result["schema_version"] == RESULT_SCHEMA == "qrae.codex-result.v2"
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["result_sha256"] == sha256_bytes(canonical_json_bytes(result))
    assert (
        validate_result_receipt(
            receipt,
            result,
            order,
            workspace,
            receipt_key_root=key_root,
        )
        == receipt
    )
    assert receipt_path.read_bytes() == canonical_json_bytes(receipt)
    assert len(list(key_root.glob("*.key"))) == 1
    assert not list(workspace.rglob("*.key"))
    assert set(receipt) == {
        "schema_version",
        "key_id",
        "run_id",
        "task_id",
        "work_order_sha256",
        "result_sha256",
        "evidence_sha256",
        "produced_at",
        "hmac_sha256",
    }


@pytest.mark.parametrize("target", ["result", "receipt_hash", "receipt_mac"])
def test_receipt_rejects_result_and_receipt_tampering(tmp_path: Path, target: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key_root = tmp_path / "receipt-keys"
    order, result, receipt, _ = _ready_result(workspace, key_root)
    changed_result = copy.deepcopy(result)
    changed_receipt = copy.deepcopy(receipt)

    if target == "result":
        changed_result["payload"]["summary"] = "Tampered draft"
    elif target == "receipt_hash":
        changed_receipt["result_sha256"] = "0" * 64
    else:
        changed_receipt["hmac_sha256"] = "0" * 64

    with pytest.raises(
        BrokerValidationError, match=r"receipt (result hash|authentication)"
    ):
        validate_result_receipt(
            changed_receipt,
            changed_result,
            order,
            workspace,
            receipt_key_root=key_root,
        )


def test_receipt_replay_is_idempotent_but_cross_task_and_cross_workspace_fail(
    tmp_path: Path,
):
    key_root = tmp_path / "receipt-keys"
    workspace = tmp_path / "workspace-a"
    workspace.mkdir()
    order, result, receipt, _ = _ready_result(workspace, key_root)

    called = False

    def forbidden_runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("authenticated replay invoked Codex")

    replay = run_work_order(
        order,
        workspace,
        workspace / "outbox",
        enabled=True,
        runner=forbidden_runner,
        now=NOW,
        receipt_key_root=key_root,
    )
    assert replay == result
    assert called is False

    other_order, other_result, _, _ = _ready_result(
        workspace, key_root, task_id="task-002"
    )
    with pytest.raises(BrokerValidationError, match="receipt task_id does not match"):
        validate_result_receipt(
            receipt,
            other_result,
            other_order,
            workspace,
            receipt_key_root=key_root,
        )

    other_workspace = tmp_path / "workspace-b"
    other_workspace.mkdir()
    _ready_result(
        other_workspace,
        key_root,
        task_id="workspace-b-task",
        run_id="workspace-b-run",
    )
    replay_evidence = other_workspace / "runs" / "run-001" / "packet.md"
    replay_evidence.parent.mkdir(parents=True)
    replay_evidence.write_text(
        "# Evidence\nNo executable instructions.\n", encoding="utf-8"
    )
    with pytest.raises(BrokerValidationError, match="receipt key id mismatch"):
        validate_result_receipt(
            receipt,
            result,
            order,
            other_workspace,
            receipt_key_root=key_root,
        )


def test_receipt_key_store_must_be_outside_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "runs" / "run-001" / "packet.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    order = prepare_work_order(
        workspace,
        task_id="task-001",
        run_id="run-001",
        task_kind="REPORT_DRAFT",
        input_path="runs/run-001/packet.md",
        allowed_paths=["runs/run-001/packet.md"],
        expires_at=NOW + timedelta(hours=1),
        now=NOW,
    )

    with pytest.raises(BrokerValidationError, match="outside workspace_root"):
        run_work_order(
            order,
            workspace,
            workspace / "outbox",
            now=NOW,
            receipt_key_root=workspace / "keys",
        )
    assert not (workspace / "outbox" / "run-001" / "task-001.json").exists()


def test_interrupted_signed_pair_replay_recovers_from_authenticated_journal(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key_root = tmp_path / "receipt-keys"
    order, result, receipt, receipt_path = _ready_result(workspace, key_root)
    result_path = receipt_path.with_name("task-001.json")
    pending_path = receipt_path.with_name(".task-001.pending.json")
    result_path.unlink()
    receipt_path.unlink()
    pending_path.write_bytes(
        canonical_json_bytes({"result": result, "receipt": receipt})
    )

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("pending replay invoked Codex")

    recovered = run_work_order(
        order,
        workspace,
        workspace / "outbox",
        enabled=True,
        runner=forbidden_runner,
        now=NOW,
        receipt_key_root=key_root,
    )
    assert recovered == result
    assert result_path.read_bytes() == canonical_json_bytes(result)
    assert receipt_path.read_bytes() == canonical_json_bytes(receipt)
    assert not pending_path.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="qrae.codex-result.v1"), "schema"),
        (lambda value: value.update(reason_code="CODEX_DISABLED"), "inconsistent"),
        (lambda value: value.update(codex_exit_code=7), "terminal policy"),
        (
            lambda value: value.update(checks=list(reversed(value["checks"]))),
            "terminal policy",
        ),
        (lambda value: value.update(produced_at="2026-07-11T00:59:59Z"), "lifetime"),
        (
            lambda value: value.update(produced_at="2026-07-11T01:00:00+00:00"),
            "canonical UTC seconds",
        ),
    ],
)
def test_result_v2_rejects_inconsistent_terminal_envelopes(
    tmp_path: Path, mutation, message: str
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order, result, _, _ = _ready_result(workspace, tmp_path / "receipt-keys")
    changed = copy.deepcopy(result)
    mutation(changed)

    with pytest.raises(BrokerValidationError, match=message):
        validate_result_envelope(changed, order, workspace)


def test_result_v2_enforces_payload_policy_and_non_promotion(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    order, result, _, _ = _ready_result(workspace, tmp_path / "receipt-keys")

    non_ready = copy.deepcopy(result)
    non_ready.update(
        status="DEFERRED_NO_CODEX",
        reason_code="CODEX_DISABLED",
        codex_exit_code=None,
        checks=[
            {"code": "WORK_ORDER_VALID", "status": "PASS"},
            {"code": "STATE_PROMOTION_FORBIDDEN", "status": "PASS"},
            {"code": "CODEX_AVAILABLE", "status": "NOT_RUN"},
        ],
    )
    with pytest.raises(BrokerValidationError, match="non-ready results cannot carry"):
        validate_result_envelope(non_ready, order, workspace)

    promoted = copy.deepcopy(result)
    promoted["requested_transition"] = "REPORT_READY"
    with pytest.raises(BrokerValidationError, match="cannot promote"):
        validate_result_envelope(promoted, order, workspace)

    nested = copy.deepcopy(result)
    nested["payload"]["requested_transition"] = "REPORT_READY"
    with pytest.raises(BrokerValidationError, match="cannot promote"):
        validate_result_envelope(nested, order, workspace)


def test_result_finishing_after_expiry_is_quarantined_not_signed_as_ready(
    tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key_root = tmp_path / "receipt-keys"
    evidence = workspace / "runs" / "run-001" / "packet.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    order = prepare_work_order(
        workspace,
        task_id="expiry-task",
        run_id="run-001",
        task_kind="ADVERSARIAL_CRITIC",
        input_path="runs/run-001/packet.md",
        allowed_paths=["runs/run-001/packet.md"],
        expires_at=NOW + timedelta(seconds=2),
        now=NOW,
    )
    clock = iter((NOW, NOW + timedelta(seconds=3)))
    monkeypatch.setattr(broker, "_utc_now", lambda: next(clock))

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(_payload()), stderr=""
        )

    result = run_work_order(
        order,
        workspace,
        workspace / "outbox",
        enabled=True,
        runner=runner,
        receipt_key_root=key_root,
    )
    assert result["status"] == "QUARANTINED"
    assert result["reason_code"] == "WORK_ORDER_EXPIRED_DURING_RUN"
    assert result["payload"] is None
    assert result["requested_transition"] == "NONE"
    receipt = json.loads(
        (workspace / "outbox" / "run-001" / "expiry-task.receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        validate_result_receipt(
            receipt,
            result,
            order,
            workspace,
            receipt_key_root=key_root,
        )
        == receipt
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_broker_rejects_fifo_evidence_without_blocking(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "runs" / "run-001" / "packet.md"
    evidence.parent.mkdir(parents=True)
    os.mkfifo(evidence)

    with pytest.raises(BrokerValidationError, match="regular file"):
        prepare_work_order(
            workspace,
            task_id="fifo-task",
            run_id="run-001",
            task_kind="ADVERSARIAL_CRITIC",
            input_path="runs/run-001/packet.md",
            allowed_paths=["runs/run-001/packet.md"],
            expires_at=NOW + timedelta(minutes=1),
            now=NOW,
        )


@pytest.mark.parametrize("manifest_existed_at_prepare", [True, False])
def test_runner_time_manifest_substitution_is_never_signed(
    tmp_path: Path, manifest_existed_at_prepare: bool
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "runs" / "run-001" / "packet.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    manifest = evidence.parent / "manifest.json"
    if manifest_existed_at_prepare:
        manifest.write_text("original manifest\n", encoding="utf-8")
    order = prepare_work_order(
        workspace,
        task_id="manifest-race",
        run_id="run-001",
        task_kind="ADVERSARIAL_CRITIC",
        input_path="runs/run-001/packet.md",
        allowed_paths=["runs/run-001/packet.md"],
        expires_at=NOW + timedelta(minutes=1),
        now=NOW,
    )

    def runner(command, **kwargs):
        manifest.write_text("substituted manifest\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(_payload()), stderr=""
        )

    with pytest.raises(
        BrokerValidationError,
        match=r"run manifest (differs from work order|appeared after work-order validation)",
    ):
        run_work_order(
            order,
            workspace,
            workspace / "outbox",
            enabled=True,
            runner=runner,
            now=NOW,
            receipt_key_root=tmp_path / "receipt-keys",
        )
    assert not (workspace / "outbox" / "run-001" / "manifest-race.json").exists()
    assert not (
        workspace / "outbox" / "run-001" / "manifest-race.receipt.json"
    ).exists()
