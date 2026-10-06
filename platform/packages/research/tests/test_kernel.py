from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae import kernel as kernel_module
from qrae.artifacts import (
    ArtifactError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from qrae.cli import main as cli_main
from qrae.kernel import KernelError, run_price_baseline, verify_run
from qrae.state import RunStateLog

UTC = timezone.utc
NOW = datetime(2026, 7, 11, tzinfo=UTC)


def write_prices(
    path: Path, *, invalid_availability: bool = False, constant_prices: bool = False
) -> None:
    rows = ["event_time,available_time,instrument,price"]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for index in range(40):
        event = start + timedelta(days=index)
        available = (
            event + timedelta(days=1) if invalid_availability and index == 0 else event
        )
        rows.append(
            f"{event.isoformat().replace('+00:00', 'Z')},"
            f"{available.isoformat().replace('+00:00', 'Z')},TEST,{100 if constant_prices else 100 + index}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_order(
    workspace: Path,
    order_id: str,
    *,
    invalid_availability: bool = False,
    constant_prices: bool = False,
    evidence: str = "E0",
    min_test_observations: int = 5,
) -> Path:
    dataset = workspace / "data" / f"{order_id}.csv"
    write_prices(
        dataset,
        invalid_availability=invalid_availability,
        constant_prices=constant_prices,
    )
    payload = {
        "schema_version": "1.0",
        "work_order_id": order_id,
        "kind": "PRICE_BASELINE",
        "market_family": "crypto",
        "created_at": "2026-07-10T00:00:00Z",
        "expires_at": "2026-07-12T00:00:00Z",
        "cutoff": "2024-02-09T00:00:00Z",
        "dataset": {
            "path": f"data/{order_id}.csv",
            "sha256": sha256_file(dataset),
            "format": "csv",
        },
        "hypothesis": {
            "mechanism": "Delayed adjustment creates persistence.",
            "prediction": "Lagged returns predict the next return sign.",
            "horizon": "one observation",
            "falsifier": "Locked after-cost test return is non-positive.",
        },
        "experiment": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "min_test_observations": min_test_observations,
        },
        "strategy": {"kind": "lagged_momentum", "lookback": 3, "cost_bps": 5.0},
        "evidence_ceiling": evidence,
        "allow_live_trading": False,
    }
    path = workspace / f"{order_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_kernel_writes_verified_e0_bundle_and_replays_idempotently(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "quantos"
    order = write_order(workspace, "baseline-001")

    first = run_price_baseline(order, workspace=workspace, output_root=output, now=NOW)
    second = run_price_baseline(order, workspace=workspace, output_root=output, now=NOW)

    assert first["status"] == "HUMAN_REVIEW"
    assert first["evidence_tier"] == "E0"
    assert first["proposed_decision"] == "REVISE"
    assert first["live_trading_authorized"] is False
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert first["manifest_sha256"] == second["manifest_sha256"]

    run_dir = Path(first["run_dir"])
    verified = verify_run(run_dir)
    assert verified["status"] == "HUMAN_REVIEW"
    assert verified["state_events"] == 10
    assert RunStateLog(run_dir / "state.jsonl").current_state == "HUMAN_REVIEW"
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["execution_scope"].startswith("bar-level delayed fill")
    assert result["metrics"]["test"]["observations"] >= 5
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert "live-ready" in decision["prohibited_claims"]
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["validator"]["uses_generation_objects"] is False
    assert validation["validator"]["external_independent_reproduction"] is False


def test_artifact_replay_validator_does_not_reuse_generation_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original_metrics = kernel_module.metrics

    def forged_metrics(points):
        value = original_metrics(points)
        value["win_rate"] = 0.123456789
        return value

    monkeypatch.setattr(kernel_module, "metrics", forged_metrics)
    with pytest.raises(KernelError, match="artifact-replay metric recomputation"):
        run_price_baseline(
            write_order(workspace, "forged-generation-metrics"),
            workspace=workspace,
            output_root=tmp_path / "quantos",
            now=NOW,
        )


def test_point_in_time_failure_is_quarantined_without_replacing_latest(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "quantos"
    good = run_price_baseline(
        write_order(workspace, "good"), workspace=workspace, output_root=output, now=NOW
    )
    latest_before = (output / "latest.json").read_bytes()

    bad = run_price_baseline(
        write_order(workspace, "bad", invalid_availability=True),
        workspace=workspace,
        output_root=output,
        now=NOW,
    )

    assert bad["status"] == "QUARANTINED"
    assert bad["error_code"] == "PIT_NOT_AVAILABLE"
    assert bad["live_trading_authorized"] is False
    assert (output / "latest.json").read_bytes() == latest_before
    assert json.loads((output / "latest.json").read_text())["run_id"] == good["run_id"]
    assert verify_run(bad["run_dir"])["status"] == "QUARANTINED"


@pytest.mark.parametrize(
    ("order_kwargs", "error_code"),
    [
        ({"min_test_observations": 1_000_000}, "DQ_INSUFFICIENT_TEST"),
        ({"constant_prices": True}, "DQ_CONSTANT_PRICES"),
    ],
)
def test_failed_data_admission_gates_quarantine_before_metrics(
    tmp_path: Path, order_kwargs: dict[str, object], error_code: str
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = run_price_baseline(
        write_order(workspace, f"gate-{error_code.lower()}", **order_kwargs),
        workspace=workspace,
        output_root=tmp_path / "quantos",
        now=NOW,
    )

    assert result["status"] == "QUARANTINED"
    assert result["error_code"] == error_code
    assert not (Path(result["run_dir"]) / "result.json").exists()


def test_manifest_tampering_blocks_verification(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = run_price_baseline(
        write_order(workspace, "tamper"),
        workspace=workspace,
        output_root=tmp_path / "quantos",
        now=NOW,
    )
    (Path(result["run_dir"]) / "result.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="verification failed"):
        verify_run(result["run_dir"])


def test_forged_manifest_cannot_omit_required_result(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "quantos"
    result = run_price_baseline(
        write_order(workspace, "omit-result"),
        workspace=workspace,
        output_root=output,
        now=NOW,
    )
    run_dir = Path(result["run_dir"])
    (output / "latest.json").unlink()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["path"] != "result.json"
    ]
    (run_dir / "result.json").unlink()
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(KernelError, match="manifest artifact contract"):
        verify_run(run_dir)


def test_forged_result_and_rehashed_manifest_fail_fresh_artifact_replay(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "quantos"
    result = run_price_baseline(
        write_order(workspace, "alter-result"),
        workspace=workspace,
        output_root=output,
        now=NOW,
    )
    run_dir = Path(result["run_dir"])
    (output / "latest.json").unlink()
    result_path = run_dir / "result.json"
    altered = json.loads(result_path.read_text(encoding="utf-8"))
    altered["metrics"]["test"]["net_compounded_return"] = 999.0
    altered_bytes = canonical_json_bytes(altered)
    result_path.write_bytes(altered_bytes)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        if item["path"] == "result.json":
            item["sha256"] = sha256_bytes(altered_bytes)
            item["bytes"] = len(altered_bytes)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(KernelError, match="state commitment"):
        verify_run(run_dir)


def test_latest_pointer_anchors_manifest_and_is_repaired_if_missing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "quantos"
    order = write_order(workspace, "latest-anchor")
    result = run_price_baseline(order, workspace=workspace, output_root=output, now=NOW)
    latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    latest["manifest_sha256"] = "0" * 64
    (output / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(Exception, match="anchor"):
        verify_run(result["run_dir"])

    (output / "latest.json").unlink()
    replay = run_price_baseline(order, workspace=workspace, output_root=output, now=NOW)
    assert replay["idempotent_replay"] is True
    assert (output / "latest.json").is_file()


def test_missing_latest_is_not_repaired_from_a_tampered_run(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "quantos"
    order = write_order(workspace, "latest-tampered")
    result = run_price_baseline(order, workspace=workspace, output_root=output, now=NOW)
    (output / "latest.json").unlink()
    (Path(result["run_dir"]) / "report.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="verification failed"):
        run_price_baseline(order, workspace=workspace, output_root=output, now=NOW)
    assert not (output / "latest.json").exists()


def test_cli_runs_example_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    workspace = Path(__file__).resolve().parents[1]
    code = cli_main(
        [
            "run",
            "--work-order",
            str(workspace / "examples" / "price_baseline" / "work_order.json"),
            "--workspace",
            str(workspace),
            "--output-root",
            str(tmp_path / "quantos"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["evidence_tier"] == "E0"
    assert payload["status"] == "HUMAN_REVIEW"
    assert payload["live_trading_authorized"] is False


def test_codex_cli_boundary_prepares_then_defers_without_state_promotion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    completed = run_price_baseline(
        write_order(workspace, "codex-boundary"),
        workspace=workspace,
        output_root=tmp_path / "quantos",
        now=NOW,
    )
    run_dir = Path(completed["run_dir"])
    state_before = (run_dir / "state.jsonl").read_bytes()

    prepare_code = cli_main(
        [
            "codex-prepare",
            "--run-dir",
            str(run_dir),
            "--task-id",
            "critic-001",
            "--task-kind",
            "ADVERSARIAL_CRITIC",
        ]
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepare_code == 0
    assert prepared["state_transition_authorized"] is False

    run_code = cli_main(
        [
            "codex-run",
            "--work-order",
            prepared["work_order"],
            "--workspace",
            prepared["workspace"],
        ]
    )
    deferred = json.loads(capsys.readouterr().out)
    assert run_code == 0
    assert deferred["status"] == "DEFERRED_NO_CODEX"
    assert deferred["requested_transition"] == "NONE"
    assert (run_dir / "state.jsonl").read_bytes() == state_before
    assert not (run_dir / "codex-work-orders").exists()
    assert verify_run(run_dir)["status"] == "HUMAN_REVIEW"
