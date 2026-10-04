import hashlib
import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae.artifacts import canonical_json_bytes
from qrae.cli import main as cli_main
from qrae.kernel import run_price_baseline
from qrae.research_workflow import (
    ResearchWorkflowError,
    prepare_review_task,
    run_research_draft,
    verify_research_draft_workflow,
)

NOW = datetime(2026, 7, 11, 4, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]


def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "workspace"
    source = REPO / "examples" / "price_baseline"
    destination = root / "examples" / "price_baseline"
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    output = root / ".quantos"
    key_root = root / "receipt-keys"
    return root, output, key_root


def ready_runner(command, **kwargs):
    prompt = json.loads(kwargs["input"].decode("utf-8"))
    payload = {
        "summary": "The E0 mechanics are coherent, but no historical alpha is proven.",
        "claims": [
            {
                "text": "The source run remains capped at E0.",
                "classification": "HYPOTHESIS",
                "artifact_refs": [prompt["input_path"]],
            }
        ],
        "artifact_refs": [prompt["input_path"]],
        "requested_transition": "NONE",
    }
    return subprocess.CompletedProcess(
        command, 0, stdout=json.dumps(payload), stderr=""
    )


def test_one_command_disabled_codex_preserves_verified_research(tmp_path):
    root, output, keys = workspace(tmp_path)
    result = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-disabled-001",
        receipt_key_root=keys,
        now=NOW,
    )

    assert result["status"] == "CODEX_DEFERRED"
    assert result["codex_status"] == "DEFERRED_NO_CODEX"
    assert result["review_bundle_id"] is None
    assert result["state_transition_authorized"] is False
    assert (
        verify_research_draft_workflow(result["workflow_path"], receipt_key_root=keys)[
            "workflow_id"
        ]
        == result["workflow_id"]
    )
    assert (output / "runs" / result["run_id"] / "manifest.json").is_file()


def test_one_command_ready_codex_ingests_and_verifies_bundle(tmp_path):
    root, output, keys = workspace(tmp_path)
    result = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-ready-001",
        enable_codex=True,
        runner=ready_runner,
        receipt_key_root=keys,
        now=NOW,
    )

    assert result["status"] == "REVIEW_READY"
    assert result["codex_status"] == "DRAFT_READY"
    assert len(result["review_bundle_id"]) == 64
    assert (
        output / "codex-reviews" / result["run_id"] / "report-ready-001" / "bundle.json"
    ).is_file()
    draft = output / "codex-drafts" / result["run_id"] / "report-ready-001.md"
    assert draft.is_file()
    assert "no historical alpha is proven" in draft.read_text(encoding="utf-8")
    verified = verify_research_draft_workflow(
        result["workflow_path"], receipt_key_root=keys
    )
    assert verified["review_bundle_id"] == result["review_bundle_id"]


def test_workflow_receipt_is_idempotent_and_does_not_reinvoke_codex(tmp_path):
    root, output, keys = workspace(tmp_path)
    first = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-idempotent-001",
        enable_codex=True,
        runner=ready_runner,
        receipt_key_root=keys,
        now=NOW,
    )
    called = False

    def forbidden_runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("idempotent replay invoked Codex")

    second = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-idempotent-001",
        enable_codex=True,
        runner=forbidden_runner,
        receipt_key_root=keys,
        now=NOW,
    )
    assert second["workflow_id"] == first["workflow_id"]
    assert second["idempotent_replay"] is True
    assert not called


def test_concurrent_identical_workflow_invokes_codex_once(tmp_path):
    root, output, keys = workspace(tmp_path)
    work_order = root / "examples" / "price_baseline" / "work_order.json"
    start_barrier = threading.Barrier(2)
    runner_calls = 0
    runner_guard = threading.Lock()
    second_runner_called = threading.Event()

    def blocked_runner(command, **kwargs):
        nonlocal runner_calls
        with runner_guard:
            runner_calls += 1
            if runner_calls == 2:
                second_runner_called.set()
        second_runner_called.wait(timeout=0.25)
        return ready_runner(command, **kwargs)

    def run_workflow():
        start_barrier.wait(timeout=5)
        return run_research_draft(
            work_order,
            workspace=root,
            output_root=output,
            task_id="report-concurrent-001",
            enable_codex=True,
            runner=blocked_runner,
            receipt_key_root=keys,
            now=NOW,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result()
            for future in (pool.submit(run_workflow), pool.submit(run_workflow))
        ]

    assert runner_calls == 1
    assert len({result["workflow_id"] for result in results}) == 1


def test_concurrent_cold_workflows_share_output_without_manifest_race(tmp_path):
    root, output, keys = workspace(tmp_path)
    work_order = root / "examples" / "price_baseline" / "work_order.json"
    alternate_order = root / "alternate-work-order.json"
    shutil.copy2(work_order, alternate_order)
    start_barrier = threading.Barrier(2)
    runner_calls = 0
    runner_guard = threading.Lock()

    def counted_runner(command, **kwargs):
        nonlocal runner_calls
        with runner_guard:
            runner_calls += 1
        return ready_runner(command, **kwargs)

    def run_workflow(path: Path, task_id: str):
        start_barrier.wait(timeout=5)
        return run_research_draft(
            path,
            workspace=root,
            output_root=output,
            task_id=task_id,
            enable_codex=True,
            runner=counted_runner,
            receipt_key_root=keys,
            now=NOW,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(run_workflow, work_order, "report-concurrent-a"),
            pool.submit(run_workflow, alternate_order, "report-concurrent-b"),
        )
        results = [future.result() for future in futures]

    assert runner_calls == 2
    assert len({result["run_id"] for result in results}) == 1
    assert len({result["workflow_id"] for result in results}) == 2


@pytest.mark.parametrize(
    "changed",
    [
        {"enable_codex": True},
        {"task_kind": "ADVERSARIAL_CRITIC"},
        {"input_name": "claim_ledger.json"},
        {"timeout_seconds": 121},
    ],
)
def test_task_id_replay_rejects_changed_invocation(tmp_path, changed):
    root, output, keys = workspace(tmp_path)
    common = {
        "workspace": root,
        "output_root": output,
        "task_id": "report-invocation-001",
        "receipt_key_root": keys,
        "now": NOW,
    }
    work_order = root / "examples" / "price_baseline" / "work_order.json"
    run_research_draft(work_order, **common)

    with pytest.raises(ResearchWorkflowError, match="WORKFLOW_CONFLICT"):
        run_research_draft(work_order, **common, **changed)


def test_completed_broker_pair_recovers_missing_workflow_after_expiry(tmp_path):
    root, output, keys = workspace(tmp_path)
    first = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-recovery-001",
        enable_codex=True,
        runner=ready_runner,
        receipt_key_root=keys,
        expires_minutes=1,
        now=NOW,
    )
    Path(first["workflow_path"]).unlink()

    recovered = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-recovery-001",
        enable_codex=True,
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recovery invoked Codex")
        ),
        receipt_key_root=keys,
        expires_minutes=1,
        now=NOW + timedelta(hours=1),
    )
    assert recovered["workflow_id"] == first["workflow_id"]
    assert recovered["recovered"] is True


def test_expired_prepared_only_task_returns_typed_retry_instruction(tmp_path):
    root, output, keys = workspace(tmp_path)
    work_order = root / "examples" / "price_baseline" / "work_order.json"
    run = run_price_baseline(work_order, workspace=root, output_root=output, now=NOW)
    prepare_review_task(
        run["run_dir"],
        task_id="report-prepared-expired-001",
        task_kind="REPORT_DRAFT",
        expires_minutes=1,
        now=NOW,
    )

    with pytest.raises(ResearchWorkflowError, match="WORKFLOW_RETRY_REQUIRED") as exc:
        run_research_draft(
            work_order,
            workspace=root,
            output_root=output,
            task_id="report-prepared-expired-001",
            receipt_key_root=keys,
            expires_minutes=1,
            now=NOW + timedelta(hours=1),
        )
    assert "task_id=report-prepared-expired-001-retry-" in str(exc.value)


def test_workflow_receipt_tamper_fails_closed(tmp_path):
    root, output, keys = workspace(tmp_path)
    result = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-tamper-001",
        receipt_key_root=keys,
        now=NOW,
    )
    path = Path(result["workflow_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["task"]["reason_code"] = "NONE"
    path.write_bytes(canonical_json_bytes(value))

    with pytest.raises(ResearchWorkflowError, match="broker artifacts differ"):
        verify_research_draft_workflow(path, receipt_key_root=keys)


def test_recomputed_enable_codex_invocation_tamper_fails_closed(tmp_path):
    root, output, keys = workspace(tmp_path)
    result = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-enable-tamper-001",
        receipt_key_root=keys,
        now=NOW,
    )
    workflow_path = Path(result["workflow_path"])
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    invocation_path = output / workflow["task"]["invocation_path"]
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    invocation["enable_codex"] = True
    invocation.pop("invocation_id")
    invocation["invocation_id"] = hashlib.sha256(
        canonical_json_bytes(invocation)
    ).hexdigest()
    invocation_bytes = canonical_json_bytes(invocation)
    invocation_path.write_bytes(invocation_bytes)
    workflow["task"]["invocation_id"] = invocation["invocation_id"]
    workflow["task"]["invocation_sha256"] = hashlib.sha256(invocation_bytes).hexdigest()
    workflow.pop("workflow_id")
    workflow["workflow_id"] = hashlib.sha256(canonical_json_bytes(workflow)).hexdigest()
    workflow_path.write_bytes(canonical_json_bytes(workflow))

    with pytest.raises(ResearchWorkflowError, match="invocation result mode differs"):
        verify_research_draft_workflow(workflow_path, receipt_key_root=keys)


def test_derived_markdown_draft_tamper_fails_closed(tmp_path):
    root, output, keys = workspace(tmp_path)
    result = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-draft-tamper-001",
        enable_codex=True,
        runner=ready_runner,
        receipt_key_root=keys,
        now=NOW,
    )
    draft = output / "codex-drafts" / result["run_id"] / "report-draft-tamper-001.md"
    draft.write_text("forged\n", encoding="utf-8")
    with pytest.raises(ResearchWorkflowError, match="draft artifact differs"):
        verify_research_draft_workflow(result["workflow_path"], receipt_key_root=keys)


def test_workflow_verifier_rejects_oversized_artifact_before_parsing(tmp_path):
    root, output, keys = workspace(tmp_path)
    result = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-oversize-001",
        receipt_key_root=keys,
        now=NOW,
    )
    workflow = Path(result["workflow_path"])
    value = json.loads(workflow.read_text(encoding="utf-8"))
    order_path = output / value["task"]["work_order_path"]
    order_path.write_bytes(b"{" + b" " * (2 * 1024 * 1024) + b"}")

    with pytest.raises(ResearchWorkflowError, match="WORKFLOW_SIZE"):
        verify_research_draft_workflow(workflow, receipt_key_root=keys)


def test_derived_markdown_escapes_active_model_markup(tmp_path):
    root, output, keys = workspace(tmp_path)

    def markup_runner(command, **kwargs):
        prompt = json.loads(kwargs["input"].decode("utf-8"))
        payload = {
            "summary": (
                "![remote](https://example.test/pixel) <img src=x>\n"
                "# Forged authority\n"
                "- Live trading authorized: true\n"
                "1. Deploy capital\n"
                "Forged heading\n"
                "===\r# Carriage-return authority\r- Live trading authorized: true"
            ),
            "claims": [],
            "artifact_refs": [prompt["input_path"]],
            "requested_transition": "NONE",
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        )

    result = run_research_draft(
        root / "examples" / "price_baseline" / "work_order.json",
        workspace=root,
        output_root=output,
        task_id="report-markup-001",
        enable_codex=True,
        runner=markup_runner,
        receipt_key_root=keys,
        now=NOW,
    )
    draft = (
        output / "codex-drafts" / result["run_id"] / "report-markup-001.md"
    ).read_text(encoding="utf-8")
    assert "![remote](" not in draft
    assert "<img src=x>" not in draft
    assert "\\<img src=x\\>" in draft
    assert "\\!\\[remote\\]\\(" in draft
    assert "\n# Forged authority" not in draft
    assert "\n- Live trading authorized: true" not in draft
    assert "\n1. Deploy capital" not in draft
    assert "\n===" not in draft
    assert "\n\\# Forged authority" in draft
    assert "\n\\- Live trading authorized" in draft
    assert "\n1\\. Deploy capital" in draft
    assert "\n\\===" in draft
    assert "\r" not in draft
    assert "\n\\# Carriage-return authority" in draft


def test_research_draft_cli_runs_and_verifies_without_codex(tmp_path, capsys):
    root, output, keys = workspace(tmp_path)
    code = cli_main(
        [
            "research-draft",
            "--work-order",
            str(root / "examples" / "price_baseline" / "work_order.json"),
            "--workspace",
            str(root),
            "--output-root",
            str(output),
            "--task-id",
            "report-cli-001",
            "--receipt-key-root",
            str(keys),
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["status"] == "CODEX_DEFERRED"

    code = cli_main(
        [
            "research-draft-verify",
            "--workflow",
            result["workflow_path"],
            "--receipt-key-root",
            str(keys),
        ]
    )
    assert code == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["workflow_id"] == result["workflow_id"]
