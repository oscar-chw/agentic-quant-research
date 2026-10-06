from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae.artifacts import canonical_json_bytes, sha256_bytes, sha256_file
from qrae.cli import main as cli_main
from qrae.codex_broker import BrokerValidationError, prepare_work_order, run_work_order
from qrae.codex_reviews import (
    BUNDLE_SCHEMA,
    ReviewBundleError,
    ingest_review_bundle,
    verify_review_bundle,
)
from qrae.kernel import run_price_baseline

UTC = timezone.utc
KERNEL_NOW = datetime(2026, 7, 11, 0, 0, tzinfo=UTC)
REVIEW_NOW = datetime(2026, 7, 11, 1, 0, tzinfo=UTC)


@dataclass(frozen=True)
class ReviewCase:
    output_root: Path
    run_dir: Path
    run_id: str
    task_id: str
    order: dict[str, object]
    key_root: Path

    @property
    def order_path(self) -> Path:
        return (
            self.output_root
            / "codex-work-orders"
            / self.run_id
            / f"{self.task_id}.json"
        )

    @property
    def result_path(self) -> Path:
        return self.output_root / "codex-outbox" / self.run_id / f"{self.task_id}.json"

    @property
    def receipt_path(self) -> Path:
        return (
            self.output_root
            / "codex-outbox"
            / self.run_id
            / f"{self.task_id}.receipt.json"
        )

    @property
    def bundle_dir(self) -> Path:
        return self.output_root / "codex-reviews" / self.run_id / self.task_id


def _write_kernel_order(workspace: Path) -> Path:
    dataset = workspace / "data" / "prices.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = ["event_time,available_time,instrument,price"]
    for index in range(40):
        timestamp = (start + timedelta(days=index)).isoformat().replace("+00:00", "Z")
        rows.append(f"{timestamp},{timestamp},TEST,{100 + index}")
    dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
    order = {
        "schema_version": "1.0",
        "work_order_id": "review-source-001",
        "kind": "PRICE_BASELINE",
        "market_family": "crypto",
        "created_at": "2026-07-10T00:00:00Z",
        "expires_at": "2026-07-12T00:00:00Z",
        "cutoff": "2024-02-09T00:00:00Z",
        "dataset": {
            "path": "data/prices.csv",
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
            "min_test_observations": 5,
        },
        "strategy": {"kind": "lagged_momentum", "lookback": 3, "cost_bps": 5.0},
        "evidence_ceiling": "E0",
        "allow_live_trading": False,
    }
    path = workspace / "work-order.json"
    path.write_text(json.dumps(order), encoding="utf-8")
    return path


def _draft_payload(run_id: str, *, transition: str = "NONE") -> dict[str, object]:
    report = f"runs/{run_id}/report.md"
    decision = f"runs/{run_id}/decision.json"
    return {
        "summary": "The deterministic E0 result remains research-only.",
        "claims": [
            {
                "text": "No historical-alpha conclusion is supported.",
                "classification": "UNSUPPORTED",
                "artifact_refs": [decision],
            }
        ],
        "artifact_refs": [report, decision],
        "requested_transition": transition,
    }


def _make_review_case(
    tmp_path: Path,
    *,
    task_id: str = "critic-001",
    transition: str = "NONE",
    allowed_paths: list[str] | None = None,
    payload: dict[str, object] | None = None,
) -> ReviewCase:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    output_root = tmp_path / "quantos"
    completed = run_price_baseline(
        _write_kernel_order(workspace),
        workspace=workspace,
        output_root=output_root,
        now=KERNEL_NOW,
    )
    run_dir = Path(completed["run_dir"])
    run_id = completed["run_id"]
    prefix = f"runs/{run_id}"
    order = prepare_work_order(
        output_root,
        task_id=task_id,
        run_id=run_id,
        task_kind="ADVERSARIAL_CRITIC",
        input_path=f"{prefix}/report.md",
        allowed_paths=allowed_paths
        or [f"{prefix}/decision.json", f"{prefix}/report.md"],
        expires_at=REVIEW_NOW + timedelta(hours=1),
        now=REVIEW_NOW,
    )
    order_path = output_root / "codex-work-orders" / run_id / f"{task_id}.json"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.write_bytes(canonical_json_bytes(order))
    draft = payload or _draft_payload(run_id, transition=transition)

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(draft), stderr=""
        )

    key_root = tmp_path / "receipt-keys"
    run_work_order(
        order,
        output_root,
        output_root / "codex-outbox",
        enabled=True,
        runner=runner,
        now=REVIEW_NOW,
        receipt_key_root=key_root,
    )
    return ReviewCase(output_root, run_dir, run_id, task_id, order, key_root)


@pytest.fixture
def review_case(tmp_path: Path) -> ReviewCase:
    return _make_review_case(tmp_path)


def test_ingests_canonical_draft_with_run_hash_and_evidence_bindings(
    review_case: ReviewCase,
):
    state_before = (review_case.run_dir / "state.jsonl").read_bytes()
    manifest_before = (review_case.run_dir / "manifest.json").read_bytes()
    latest_before = (review_case.output_root / "latest.json").read_bytes()
    metrics_before = (review_case.run_dir / "result.json").read_bytes()

    ingested = ingest_review_bundle(
        review_case.run_dir,
        task_id=review_case.task_id,
        receipt_key_root=review_case.key_root,
    )

    assert review_case.order["schema_version"] == "qrae.codex-work-order.v2"
    assert review_case.order["run_manifest"]["sha256"] == sha256_bytes(manifest_before)
    assert ingested["status"] == "DRAFT_INGESTED"
    assert ingested["schema_version"] == BUNDLE_SCHEMA
    assert ingested["run_id"] == review_case.run_id
    assert ingested["state_transition_authorized"] is False
    assert {path.name for path in review_case.bundle_dir.iterdir()} == {
        "work_order.json",
        "result.json",
        "receipt.json",
        "bundle.json",
    }
    descriptor_raw = (review_case.bundle_dir / "bundle.json").read_bytes()
    descriptor = json.loads(descriptor_raw)
    assert descriptor_raw == canonical_json_bytes(descriptor)
    assert descriptor["kind"] == "CODEX_DRAFT_REVIEW"
    assert descriptor["authority"] == {
        "requested_transition": "NONE",
        "metric_authority": False,
        "state_transition_authorized": False,
        "capital_authorized": False,
        "live_trading_authorized": False,
    }
    assert descriptor["run"]["manifest_sha256"] == sha256_bytes(manifest_before)
    assert descriptor["task"]["result_sha256"] == sha256_file(review_case.result_path)
    assert descriptor["task"]["receipt_sha256"] == sha256_file(review_case.receipt_path)
    evidence = {row["path"]: row for row in descriptor["evidence_bindings"]}
    report = f"runs/{review_case.run_id}/report.md"
    decision = f"runs/{review_case.run_id}/decision.json"
    manifest = f"runs/{review_case.run_id}/manifest.json"
    assert set(evidence) == {report, decision, manifest}
    assert evidence[report]["roles"] == ["INPUT", "PAYLOAD"]
    assert evidence[decision]["roles"] == ["CLAIM:000", "PAYLOAD"]
    assert evidence[manifest]["roles"] == ["RUN_MANIFEST"]
    assert evidence[report]["sha256"] == sha256_file(review_case.run_dir / "report.md")

    assert (review_case.run_dir / "state.jsonl").read_bytes() == state_before
    assert (review_case.run_dir / "manifest.json").read_bytes() == manifest_before
    assert (review_case.output_root / "latest.json").read_bytes() == latest_before
    assert (review_case.run_dir / "result.json").read_bytes() == metrics_before


def test_review_ingestion_is_immutable_idempotent_and_outbox_independent(
    review_case: ReviewCase,
):
    first = ingest_review_bundle(
        review_case.run_dir,
        task_id=review_case.task_id,
        receipt_key_root=review_case.key_root,
    )
    frozen = {path.name: path.read_bytes() for path in review_case.bundle_dir.iterdir()}
    review_case.order_path.unlink()
    review_case.result_path.unlink()
    review_case.receipt_path.unlink()

    second = ingest_review_bundle(
        review_case.run_dir,
        task_id=review_case.task_id,
        receipt_key_root=review_case.key_root,
    )
    verified = verify_review_bundle(
        review_case.bundle_dir, receipt_key_root=review_case.key_root
    )
    assert second == first == verified
    assert {
        path.name: path.read_bytes() for path in review_case.bundle_dir.iterdir()
    } == frozen
    assert not list(review_case.bundle_dir.glob("*.tmp"))


def test_concurrent_ingestion_converges_on_one_immutable_bundle(
    review_case: ReviewCase,
):
    def ingest(_: int) -> dict[str, object]:
        return ingest_review_bundle(
            review_case.run_dir,
            task_id=review_case.task_id,
            receipt_key_root=review_case.key_root,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(ingest, range(4)))

    assert len({outcome["bundle_id"] for outcome in outcomes}) == 1
    assert {path.name for path in review_case.bundle_dir.iterdir()} == {
        "work_order.json",
        "result.json",
        "receipt.json",
        "bundle.json",
    }


def test_cli_ingests_and_verifies_review_bundle(review_case: ReviewCase, capsys):
    assert (
        cli_main(
            [
                "codex-ingest",
                "--run-dir",
                str(review_case.run_dir),
                "--task-id",
                review_case.task_id,
                "--receipt-key-root",
                str(review_case.key_root),
            ]
        )
        == 0
    )
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["ok"] is True
    assert ingested["status"] == "DRAFT_INGESTED"

    assert (
        cli_main(
            [
                "codex-review-verify",
                "--bundle-dir",
                str(review_case.bundle_dir),
                "--receipt-key-root",
                str(review_case.key_root),
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["ok"] is True
    assert verified["bundle_id"] == ingested["bundle_id"]


def test_ingestion_rejects_cross_run_evidence_even_when_broker_allowed(
    tmp_path: Path,
):
    case = _make_review_case(tmp_path)
    cross = case.output_root / "runs" / "other-run" / "packet.md"
    cross.parent.mkdir(parents=True)
    cross.write_text("other evidence\n", encoding="utf-8")
    prefix = f"runs/{case.run_id}"
    broad_order = prepare_work_order(
        case.output_root,
        task_id=case.task_id,
        run_id=case.run_id,
        task_kind="ADVERSARIAL_CRITIC",
        input_path=f"{prefix}/report.md",
        allowed_paths=["runs"],
        expires_at=REVIEW_NOW + timedelta(hours=1),
        now=REVIEW_NOW,
    )
    case.order_path.write_bytes(canonical_json_bytes(broad_order))
    case.result_path.unlink()
    case.receipt_path.unlink()
    payload = _draft_payload(case.run_id)
    payload["artifact_refs"].append("runs/other-run/packet.md")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        )

    run_work_order(
        broad_order,
        case.output_root,
        case.output_root / "codex-outbox",
        enabled=True,
        runner=runner,
        now=REVIEW_NOW,
        receipt_key_root=case.key_root,
    )
    with pytest.raises(ReviewBundleError, match="outside the source run"):
        ingest_review_bundle(
            case.run_dir, task_id=case.task_id, receipt_key_root=case.key_root
        )


def test_ingestion_rejects_authenticated_result_bound_to_another_run(tmp_path: Path):
    case = _make_review_case(tmp_path)
    foreign_run = "foreign-run"
    report = f"runs/{case.run_id}/report.md"
    foreign_order = prepare_work_order(
        case.output_root,
        task_id=case.task_id,
        run_id=foreign_run,
        task_kind="ADVERSARIAL_CRITIC",
        input_path=report,
        allowed_paths=[report],
        expires_at=REVIEW_NOW + timedelta(hours=1),
        now=REVIEW_NOW,
    )
    case.order_path.write_bytes(canonical_json_bytes(foreign_order))
    case.result_path.unlink()
    case.receipt_path.unlink()

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "summary": "Foreign review",
                    "claims": [],
                    "artifact_refs": [report],
                    "requested_transition": "NONE",
                }
            ),
            stderr="",
        )

    run_work_order(
        foreign_order,
        case.output_root,
        case.output_root / "codex-outbox",
        enabled=True,
        runner=runner,
        now=REVIEW_NOW,
        receipt_key_root=case.key_root,
    )
    foreign_outbox = case.output_root / "codex-outbox" / foreign_run
    case.result_path.parent.mkdir(parents=True, exist_ok=True)
    case.result_path.write_bytes((foreign_outbox / f"{case.task_id}.json").read_bytes())
    case.receipt_path.write_bytes(
        (foreign_outbox / f"{case.task_id}.receipt.json").read_bytes()
    )

    with pytest.raises(ReviewBundleError, match="task or run identity differs"):
        ingest_review_bundle(
            case.run_dir, task_id=case.task_id, receipt_key_root=case.key_root
        )


def test_non_ready_or_promoting_codex_output_cannot_create_review_bundle(
    tmp_path: Path,
):
    promoted = _make_review_case(tmp_path / "promoted", transition="REPORT_READY")
    state_before = (promoted.run_dir / "state.jsonl").read_bytes()
    with pytest.raises(ReviewBundleError, match="only DRAFT_READY"):
        ingest_review_bundle(
            promoted.run_dir,
            task_id=promoted.task_id,
            receipt_key_root=promoted.key_root,
        )
    assert (promoted.run_dir / "state.jsonl").read_bytes() == state_before
    assert not (promoted.bundle_dir / "bundle.json").exists()

    deferred = _make_review_case(tmp_path / "deferred")
    deferred.result_path.unlink()
    deferred.receipt_path.unlink()
    run_work_order(
        deferred.order,
        deferred.output_root,
        deferred.output_root / "codex-outbox",
        now=REVIEW_NOW,
        receipt_key_root=deferred.key_root,
    )
    with pytest.raises(ReviewBundleError, match="only DRAFT_READY"):
        ingest_review_bundle(
            deferred.run_dir,
            task_id=deferred.task_id,
            receipt_key_root=deferred.key_root,
        )


@pytest.mark.parametrize("target", ["result", "receipt", "authority", "source", "key"])
def test_review_verification_fails_closed_on_tampering(
    review_case: ReviewCase, target: str
):
    ingest_review_bundle(
        review_case.run_dir,
        task_id=review_case.task_id,
        receipt_key_root=review_case.key_root,
    )
    if target == "result":
        result_path = review_case.bundle_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["payload"]["summary"] = "tampered"
        result_path.write_bytes(canonical_json_bytes(result))
    elif target == "receipt":
        receipt_path = review_case.bundle_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["hmac_sha256"] = "0" * 64
        receipt_path.write_bytes(canonical_json_bytes(receipt))
    elif target == "authority":
        descriptor_path = review_case.bundle_dir / "bundle.json"
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor["authority"]["state_transition_authorized"] = True
        descriptor_path.write_bytes(canonical_json_bytes(descriptor))
    elif target == "source":
        (review_case.run_dir / "report.md").write_text("tampered\n", encoding="utf-8")
    else:
        next(review_case.key_root.glob("*.key")).unlink()

    with pytest.raises(ReviewBundleError):
        verify_review_bundle(
            review_case.bundle_dir, receipt_key_root=review_case.key_root
        )


def test_ingestion_does_not_overwrite_a_tampered_existing_bundle(
    review_case: ReviewCase,
):
    ingest_review_bundle(
        review_case.run_dir,
        task_id=review_case.task_id,
        receipt_key_root=review_case.key_root,
    )
    descriptor = review_case.bundle_dir / "bundle.json"
    descriptor.write_bytes(b"{}\n")

    with pytest.raises(ReviewBundleError):
        ingest_review_bundle(
            review_case.run_dir,
            task_id=review_case.task_id,
            receipt_key_root=review_case.key_root,
        )
    assert descriptor.read_bytes() == b"{}\n"


def test_signed_evidence_root_rejects_manifest_substitution_before_ingestion(
    review_case: ReviewCase,
):
    manifest_path = review_case.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["kernel_version"] = "forged-version"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    latest_path = review_case.output_root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["manifest_sha256"] = sha256_file(manifest_path)
    latest_path.write_bytes(canonical_json_bytes(latest))

    runner_called = False

    def forbidden_runner(*args, **kwargs):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("changed prepare-time manifest reached Codex")

    with pytest.raises(BrokerValidationError, match="manifest differs from work order"):
        run_work_order(
            review_case.order,
            review_case.output_root,
            review_case.output_root / "codex-outbox",
            enabled=True,
            runner=forbidden_runner,
            now=REVIEW_NOW,
            receipt_key_root=review_case.key_root,
        )
    assert runner_called is False

    with pytest.raises(ReviewBundleError, match="manifest differs from work order"):
        ingest_review_bundle(
            review_case.run_dir,
            task_id=review_case.task_id,
            receipt_key_root=review_case.key_root,
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_ingestion_rejects_fifo_source_without_blocking(review_case: ReviewCase):
    review_case.result_path.unlink()
    os.mkfifo(review_case.result_path)

    with pytest.raises(ReviewBundleError, match="bounded regular file"):
        ingest_review_bundle(
            review_case.run_dir,
            task_id=review_case.task_id,
            receipt_key_root=review_case.key_root,
        )
