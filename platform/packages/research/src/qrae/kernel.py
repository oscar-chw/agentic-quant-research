"""Deterministic local-first QuantOS research kernel."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import __version__
from .artifacts import (
    ArtifactRecord,
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_manifest,
)
from .contracts import (
    WorkOrder,
    canonical_json,
    canonical_sha256,
    load_work_order,
    resolve_dataset_provenance,
)
from .data_catalog import CatalogError, verify_provenance_envelope
from .price_baseline import (
    DataQualityError,
    load_price_rows,
    metrics,
    run_lagged_momentum,
    split_points,
)
from .price_reference import PriceReferenceError, reference_baseline
from .state import (
    BASELINE_COMPLETE,
    CHEAP_FALSIFICATION,
    DATA_QUALITY_PASSED,
    EXPERIMENT_COMPLETE,
    HUMAN_REVIEW,
    INDEPENDENT_VALIDATION,
    QUARANTINED,
    REPORT_READY,
    SNAPSHOT_FROZEN,
    SOURCE_RESOLVED,
    IdempotencyConflictError,
    RunStateLog,
    hash_actor,
)

UTC = timezone.utc
KERNEL_SCHEMA_VERSION = "1.0"
KERNEL_ACTOR = "qrae.local-kernel"

_COMPLETED_ARTIFACTS = frozenset(
    {
        "backtest/series.jsonl",
        "cheap_falsification.json",
        "claim_ledger.json",
        "data_quality.json",
        "decision.json",
        "knowledge.json",
        "report.md",
        "result.json",
        "snapshot/manifest.json",
        "snapshot/prices.csv",
        "state.jsonl",
        "summary.json",
        "validation.json",
        "work_order.json",
    }
)
_QUARANTINED_ARTIFACTS = frozenset(
    {
        "data_quality.json",
        "report.md",
        "snapshot/manifest.json",
        "snapshot/prices.csv",
        "state.jsonl",
        "summary.json",
        "work_order.json",
    }
)
_PROVENANCE_ARTIFACT = "snapshot/provenance.json"


class KernelError(RuntimeError):
    """A deterministic run or verification gate failed."""


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _order_payload(order: WorkOrder) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "path": order.dataset.path,
        "sha256": order.dataset.sha256,
        "format": order.dataset.format,
    }
    if order.dataset.catalog_snapshot is not None:
        dataset["catalog_snapshot"] = asdict(order.dataset.catalog_snapshot)
    return {
        "schema_version": order.schema_version,
        "work_order_id": order.work_order_id,
        "kind": order.kind,
        "market_family": order.market_family,
        "created_at": _utc(order.created_at),
        "expires_at": _utc(order.expires_at),
        "cutoff": _utc(order.cutoff),
        "dataset": dataset,
        "hypothesis": asdict(order.hypothesis),
        "experiment": asdict(order.experiment),
        "strategy": asdict(order.strategy),
        "evidence_ceiling": order.evidence_ceiling,
        "allow_live_trading": False,
    }


def _run_id(order_payload: Mapping[str, Any]) -> str:
    digest = canonical_sha256(
        {
            "kernel_schema_version": KERNEL_SCHEMA_VERSION,
            "kernel_version": __version__,
            "work_order": order_payload,
        }
    )
    return f"{order_payload['work_order_id']}-{digest[:16]}"


def _advance(
    state: RunStateLog,
    target: str,
    *,
    payload: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> None:
    events = state.replay()
    for event in events:
        if event.to_state == target:
            if (
                event.actor_hash != hash_actor(KERNEL_ACTOR)
                or event.payload != dict(payload)
                or event.artifact_hashes != dict(sorted(artifact_hashes.items()))
            ):
                raise IdempotencyConflictError(
                    f"persisted {target} transition differs from retry"
                )
            return
    state.transition(
        target, actor=KERNEL_ACTOR, payload=payload, artifact_hashes=artifact_hashes
    )


def _artifact_record(path: Path, relative: str) -> ArtifactRecord:
    return ArtifactRecord(
        relative.replace("\\", "/"), sha256_file(path), path.stat().st_size
    )


def _series_bytes(points: list[Any]) -> bytes:
    return b"".join(canonical_json_bytes(point.as_dict()) for point in points)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KernelError(f"persisted series field {field} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise KernelError(f"persisted series field {field} is not finite")
    return number


def _load_persisted_series(path: Path) -> list[dict[str, Any]]:
    expected = {
        "exit_time",
        "gross_return",
        "cost_return",
        "net_return",
        "turnover",
        "active_instruments",
        "flat_entry_cost_return",
        "flat_entry_turnover",
        "flat_exit_cost_return",
        "flat_exit_turnover",
    }
    output: list[dict[str, Any]] = []
    previous_time: datetime | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise KernelError("cannot read persisted backtest series") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KernelError(
                f"persisted series line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict) or set(value) != expected:
            raise KernelError(
                f"persisted series line {line_number} has an invalid schema"
            )
        try:
            exit_time = datetime.fromisoformat(
                str(value["exit_time"]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise KernelError(
                f"persisted series line {line_number} has an invalid exit_time"
            ) from exc
        if exit_time.tzinfo is None or (
            previous_time is not None and exit_time <= previous_time
        ):
            raise KernelError(
                "persisted series exit times must be timezone-aware and strictly increasing"
            )
        previous_time = exit_time
        point = {
            "exit_time": value["exit_time"],
            "gross_return": _finite_number(value["gross_return"], "gross_return"),
            "cost_return": _finite_number(value["cost_return"], "cost_return"),
            "net_return": _finite_number(value["net_return"], "net_return"),
            "turnover": _finite_number(value["turnover"], "turnover"),
            "active_instruments": value["active_instruments"],
        }
        if (
            isinstance(point["active_instruments"], bool)
            or not isinstance(point["active_instruments"], int)
            or point["active_instruments"] < 1
        ):
            raise KernelError(
                "persisted series active_instruments must be a positive integer"
            )
        for field in (
            "flat_entry_cost_return",
            "flat_entry_turnover",
            "flat_exit_cost_return",
            "flat_exit_turnover",
        ):
            point[field] = _finite_number(value[field], field)
        if (
            abs(point["gross_return"] - point["cost_return"] - point["net_return"])
            > 1e-12
        ):
            raise KernelError(
                f"persisted series accounting mismatch at line {line_number}"
            )
        output.append(point)
    if len(output) < 3:
        raise KernelError("persisted backtest series has too few observations")
    return output


def _split_persisted_series(
    points: list[dict[str, Any]], train_fraction: float, validation_fraction: float
) -> dict[str, list[dict[str, Any]]]:
    count = len(points)
    train_end = max(1, int(count * train_fraction))
    validation_end = max(
        train_end + 1, int(count * (train_fraction + validation_fraction))
    )
    validation_end = min(validation_end, count - 1)
    if train_end >= validation_end or validation_end >= count:
        raise KernelError("persisted experiment cannot form three chronological splits")
    boundaries = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, count),
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for name, (start, end) in boundaries.items():
        segment = [dict(point) for point in points[start:end]]
        first = segment[0]
        first["cost_return"] = first["flat_entry_cost_return"]
        first["turnover"] = first["flat_entry_turnover"]
        first["net_return"] = first["gross_return"] - first["cost_return"]
        last = segment[-1]
        last["cost_return"] += last["flat_exit_cost_return"]
        last["turnover"] += last["flat_exit_turnover"]
        last["net_return"] = last["gross_return"] - last["cost_return"]
        output[name] = segment
    return output


def _absorbing_compound(returns: list[float]) -> tuple[float, bool]:
    equity = 1.0
    bankrupt = False
    for value in returns:
        if bankrupt or value <= -1.0:
            equity = 0.0
            bankrupt = True
        else:
            equity *= 1.0 + value
    return equity - 1.0, bankrupt


def _replay_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    gross = [point["gross_return"] for point in points]
    net = [point["net_return"] for point in points]
    costs = [point["cost_return"] for point in points]
    gross_compounded, _ = _absorbing_compound(gross)
    net_compounded, bankrupt = _absorbing_compound(net)
    mean = statistics.fmean(net) if net else 0.0
    sample_std = statistics.stdev(net) if len(net) > 1 else 0.0
    ratio = mean / sample_std if sample_std > 0 else None
    equity = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in net:
        equity = 0.0 if equity == 0.0 or value <= -1.0 else equity * (1.0 + value)
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
    return {
        "observations": len(points),
        "gross_compounded_return": gross_compounded,
        "net_compounded_return": net_compounded,
        "mean_net_return_per_observation": mean,
        "sample_std_net_return_per_observation": sample_std,
        "mean_over_sample_std_not_annualized": ratio,
        "maximum_drawdown": maximum_drawdown,
        "win_rate": sum(value > 0 for value in net) / len(net) if net else None,
        "turnover_units": sum(point["turnover"] for point in points),
        "cost_return_sum": sum(costs),
        "bankrupt_path": bankrupt,
        "units": {
            "returns": "decimal return",
            "cost_return_sum": "decimal return summed across observations",
            "turnover_units": "absolute position change, equal-weight portfolio",
            "ratio": "per-observation, not annualized",
        },
    }


def _equivalent_metrics(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _equivalent_metrics(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    return actual == expected


def _frozen_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise KernelError(f"persisted {field} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise KernelError(f"persisted {field} is not a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise KernelError(f"persisted {field} must use UTC")
    return parsed.astimezone(UTC)


def _load_json_artifact(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KernelError(f"persisted {label} is invalid JSON") from exc
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise KernelError(f"persisted {label} is not finite JSON") from exc
    if not isinstance(value, dict) or raw != canonical:
        raise KernelError(f"persisted {label} must be canonical JSON")
    return value


def _frozen_relative_path(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise KernelError(f"persisted {field} is not a normalized relative path")
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or PureWindowsPath(value).drive
        or str(posix) != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise KernelError(f"persisted {field} is not a normalized relative path")
    return value


def _validate_frozen_dataset(
    run_dir: Path,
    *,
    expected_hash: Any | None = None,
) -> dict[str, Any]:
    work_order = _load_json_artifact(run_dir / "work_order.json", "work order")
    snapshot_manifest = _load_json_artifact(
        run_dir / "snapshot" / "manifest.json", "snapshot manifest"
    )
    dataset = work_order.get("dataset")
    if not isinstance(dataset, dict):
        raise KernelError("persisted work-order dataset is invalid")
    dataset_hash = dataset.get("sha256")
    _frozen_relative_path(dataset.get("path"), "dataset path")
    if (
        work_order.get("evidence_ceiling") != "E0"
        or work_order.get("allow_live_trading") is not False
    ):
        raise KernelError("persisted work order violates the research-only ceiling")
    if expected_hash is not None and expected_hash != dataset_hash:
        raise KernelError("artifact-replay result dataset hash differs from work order")
    prices_path = run_dir / "snapshot" / "prices.csv"
    if (
        not isinstance(dataset_hash, str)
        or len(dataset_hash) != 64
        or any(character not in "0123456789abcdef" for character in dataset_hash)
        or not prices_path.is_file()
        or sha256_file(prices_path) != dataset_hash
    ):
        raise KernelError("artifact-replay dataset hash validation failed")

    base_manifest_fields = {
        "schema_version",
        "source_relative_path",
        "snapshot_path",
        "sha256",
        "cutoff",
        "market_family",
        "format",
    }
    reference = dataset.get("catalog_snapshot")
    provenance_required = reference is not None
    expected_manifest_fields = set(base_manifest_fields)
    if provenance_required:
        expected_manifest_fields.update(
            {"provenance_path", "provenance_sha256", "catalog_snapshot_id"}
        )
    if set(snapshot_manifest) != expected_manifest_fields:
        raise KernelError("persisted snapshot manifest fields are invalid")
    if (
        snapshot_manifest.get("schema_version") != "1.0"
        or snapshot_manifest.get("source_relative_path") != dataset.get("path")
        or snapshot_manifest.get("snapshot_path") != "snapshot/prices.csv"
        or snapshot_manifest.get("sha256") != dataset_hash
        or snapshot_manifest.get("cutoff") != work_order.get("cutoff")
        or snapshot_manifest.get("market_family") != work_order.get("market_family")
        or snapshot_manifest.get("format") != dataset.get("format")
    ):
        raise KernelError("persisted snapshot manifest binding differs")

    provenance_sha256: str | None = None
    if provenance_required:
        if (
            work_order.get("schema_version") != "1.1"
            or not isinstance(reference, dict)
            or set(reference) != {"catalog_root", "snapshot_id"}
        ):
            raise KernelError("persisted catalog snapshot reference is invalid")
        _frozen_relative_path(reference.get("catalog_root"), "catalog_root")
        if (
            not isinstance(reference.get("snapshot_id"), str)
            or len(reference["snapshot_id"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in reference["snapshot_id"]
            )
        ):
            raise KernelError("persisted catalog snapshot id is invalid")
        provenance_path = run_dir / _PROVENANCE_ARTIFACT
        provenance = _load_json_artifact(provenance_path, "catalog provenance")
        try:
            provenance = verify_provenance_envelope(provenance)
        except CatalogError as exc:
            raise KernelError(
                f"persisted catalog provenance is invalid: {exc.code}"
            ) from exc
        provenance_sha256 = sha256_file(provenance_path)
        selected = provenance["selected_object"]
        source_snapshot = provenance["snapshot"]
        if (
            reference.get("snapshot_id") != source_snapshot.get("snapshot_id")
            or selected.get("sha256") != dataset_hash
            or selected.get("bytes") != prices_path.stat().st_size
            or provenance["adapter"].get("market_family")
            != work_order.get("market_family")
            or source_snapshot.get("normalized_schema") != "qrae.price-bars.v1"
            or source_snapshot.get("evidence_ceiling") != "E0"
            or snapshot_manifest.get("provenance_path") != _PROVENANCE_ARTIFACT
            or snapshot_manifest.get("provenance_sha256") != provenance_sha256
            or snapshot_manifest.get("catalog_snapshot_id")
            != reference.get("snapshot_id")
            or _frozen_utc(
                source_snapshot.get("available_at"), "provenance available_at"
            )
            > _frozen_utc(work_order.get("cutoff"), "work-order cutoff")
        ):
            raise KernelError("persisted catalog provenance binding differs")
    elif (
        work_order.get("schema_version") != "1.0"
        or (run_dir / _PROVENANCE_ARTIFACT).exists()
    ):
        raise KernelError("legacy work order has an unexpected provenance artifact")
    return {
        "work_order": work_order,
        "snapshot_manifest": snapshot_manifest,
        "dataset_sha256": dataset_hash,
        "provenance_required": provenance_required,
        "provenance_sha256": provenance_sha256,
    }


def _artifact_replay_validation(
    run_dir: Path, result: Mapping[str, Any], *, schema_version: str = "1.1"
) -> dict[str, Any]:
    if schema_version not in {"1.0", "1.1"}:
        raise KernelError("unsupported persisted validation schema")
    experiment = result.get("experiment")
    if not isinstance(experiment, dict):
        raise KernelError("persisted result experiment is invalid")
    train_fraction = _finite_number(experiment.get("train_fraction"), "train_fraction")
    validation_fraction = _finite_number(
        experiment.get("validation_fraction"), "validation_fraction"
    )
    if (
        not 0 < train_fraction < 1
        or not 0 < validation_fraction < 1
        or train_fraction + validation_fraction >= 1
    ):
        raise KernelError("persisted result experiment fractions are invalid")
    points = _load_persisted_series(run_dir / "backtest" / "series.jsonl")
    splits = _split_persisted_series(points, train_fraction, validation_fraction)
    recomputed = {name: _replay_metrics(segment) for name, segment in splits.items()}
    if not _equivalent_metrics(result.get("metrics"), recomputed):
        raise KernelError(
            "artifact-replay metric recomputation differs from result bundle"
        )

    frozen = _validate_frozen_dataset(
        run_dir, expected_hash=result.get("dataset_sha256")
    )
    expected_provenance = frozen["provenance_sha256"]
    if result.get("dataset_provenance_sha256") != expected_provenance:
        raise KernelError("artifact-replay result provenance hash differs")
    order = frozen["work_order"]
    if result.get("strategy") != order.get("strategy") or experiment != order.get(
        "experiment"
    ):
        raise KernelError("price-reference registered strategy or experiment differs")
    try:
        reference = reference_baseline(run_dir / "snapshot" / "prices.csv", order)
    except PriceReferenceError as exc:
        raise KernelError(f"price-reference input validation failed: {exc}") from exc
    expected_points = reference["series"]
    if len(points) != len(expected_points):
        raise KernelError("price-reference observation count differs")
    for index, (actual, expected) in enumerate(zip(points, expected_points, strict=True)):
        if not _equivalent_metrics(actual, expected):
            raise KernelError(f"price-reference series differs at observation {index}")
    reference_metrics = {
        name: _replay_metrics(segment) for name, segment in reference["splits"].items()
    }
    if not _equivalent_metrics(result.get("metrics"), reference_metrics):
        raise KernelError("price-reference split accounting differs")
    # Legacy receipts retain their exact shape and bytes. Fresh verification
    # still runs the reference above; only new receipts record the added check.
    reference_checks = (
        {}
        if schema_version == "1.0"
        else {
            "price_to_position_replay": {
                "status": "PASS",
                "reference_version": "qrae.price-reference.v1",
                "arithmetic": "rational prices and positions; float serialization",
                "observations_recomputed": len(expected_points),
                "instruments": reference["instruments"],
                "tolerance": "relative and absolute 1e-12",
                "split_entry_and_liquidation": True,
            }
        }
    )
    return {
        "schema_version": schema_version,
        "status": "PASS",
        "validator": {
            "kind": "persisted_artifact_replay",
            "uses_generation_objects": False,
            "external_independent_reproduction": False,
        },
        "checks": {
            "dataset_hash": True,
            "catalog_provenance": {
                "status": "PASS" if frozen["provenance_required"] else "NOT_APPLICABLE",
                "frozen_replay": True,
                "external_catalog_required": False,
            },
            "serialized_series_schema": True,
            "chronological_unique_series": True,
            "chronological_disjoint_splits": True,
            "accounting_reconciliation": {
                "status": "PASS",
                "observations_recomputed": len(points),
                "tolerance": "1e-12",
            },
            "metric_recomputation": "PASS",
            **reference_checks,
            "live_trading_disabled": result.get("live_trading_authorized") is False,
        },
        "limitations": [
            "internal persisted-artifact replay; external independent reproduction not performed",
            "single registered split",
            "bar-level execution only",
            "simplified linear transaction cost",
            "no capacity, impact, borrow, funding, or market-specific settlement model",
        ],
    }


def _decision(order: WorkOrder, result: Mapping[str, Any]) -> dict[str, Any]:
    if order.evidence_ceiling != "E0":
        raise KernelError(
            "unsupported evidence ceiling escaped the V0 work-order contract"
        )
    decision = "REVISE"
    reason = (
        "The registered evidence ceiling is E0; mechanics may be inspected but no "
        "historical-alpha claim is permitted without licensed historical point-in-time provenance "
        "and compatible temporal validation."
    )
    evidence_tier = "E0"
    return {
        "schema_version": "1.0",
        "run_id": result["run_id"],
        "proposed_decision": decision,
        "authorized_state": "HUMAN_REVIEW",
        "evidence_tier": evidence_tier,
        "reason": reason,
        "human_approval_required": True,
        "live_trading_authorized": False,
        "permitted_claims": [
            "mechanical baseline result"
            if evidence_tier == "E0"
            else "preliminary historical E1 evidence",
            "after-cost result under the registered simplified cost model",
        ],
        "prohibited_claims": [
            "profitable strategy",
            "execution-realistic fills",
            "paper-ready",
            "live-ready",
            "capital-approved",
        ],
        "confidence_vector": {
            "data_integrity": "PASS",
            "mechanism": "HYPOTHESIS",
            "statistical_validity": "PRELIMINARY"
            if evidence_tier == "E1"
            else "UNPROVEN",
            "oos_robustness": "SINGLE_SPLIT_ONLY"
            if evidence_tier == "E1"
            else "UNPROVEN",
            "execution_realism": "BAR_LEVEL_ONLY",
            "cost_capacity": "SIMPLIFIED_COST_ONLY",
            "regime_stability": "UNTESTED",
            "model_calibration": "NOT_APPLICABLE",
            "operational_security": "RESEARCH_ONLY",
            "independent_reproduction": "REQUIRED",
        },
    }


def _report(
    order: WorkOrder, result: Mapping[str, Any], decision: Mapping[str, Any]
) -> str:
    lines = [
        f"# QuantOS baseline report: {result['run_id']}",
        "",
        "## Status",
        "",
        f"- Proposed decision: `{decision['proposed_decision']}`",
        f"- Evidence tier: `{decision['evidence_tier']}`",
        "- Human approval required: `true`",
        "- Live trading authorized: `false`",
        f"- Catalog provenance frozen: `{'true' if order.dataset.catalog_snapshot is not None else 'false'}`",
        "",
        "## Registered hypothesis",
        "",
        f"- Mechanism: {order.hypothesis.mechanism}",
        f"- Prediction: {order.hypothesis.prediction}",
        f"- Horizon: {order.hypothesis.horizon}",
        f"- Falsifier: {order.hypothesis.falsifier}",
        "",
        "## Method",
        "",
        f"A `{order.strategy.kind}` baseline uses only information available at each event time, a lookback of {order.strategy.lookback}, fills no earlier than the following observation, and measures the subsequent holding return. Costs are {order.strategy.cost_bps} basis points per unit of position change, including split-local entry and terminal liquidation.",
        f"The chronological fractions are train={order.experiment.train_fraction}, validation={order.experiment.validation_fraction}, and test={1.0 - order.experiment.train_fraction - order.experiment.validation_fraction:.6g}.",
        "No hyperparameter selection or annualization is performed.",
        "",
        "## Results",
        "",
        "| Split | Observations | Gross compounded | Net compounded | Maximum drawdown | Per-observation mean/std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        item = result["metrics"][split]
        ratio = item["mean_over_sample_std_not_annualized"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        lines.append(
            f"| {split} | {item['observations']} | {item['gross_compounded_return']:.6f} | "
            f"{item['net_compounded_return']:.6f} | {item['maximum_drawdown']:.6f} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"{decision['reason']}",
            "This is a deterministic bar-level baseline, not event-driven fill evidence. It does not prove queue priority, slippage, market impact, capacity, paper performance, or live profitability.",
            "",
            "## Next gate",
            "",
            "Run independent reproduction, walk-forward or nested temporal validation, parameter and regime perturbations, richer cost/capacity analysis, and market-specific execution replay before raising the evidence tier.",
        ]
    )
    return "\n".join(lines) + "\n"


def _claim_ledger(
    run_id: str,
    decision: Mapping[str, Any],
    *,
    provenance_bound: bool,
) -> dict[str, Any]:
    claims = [
        {
            "claim_id": "C001",
            "text": "The registered dataset passed the implemented schema, timing, cutoff, duplicate, monotonicity, and price checks.",
            "status": "VERIFIED",
            "artifact": "data_quality.json",
        },
        {
            "claim_id": "C002",
            "text": "The result is preliminary historical evidence under the registered bar-level and cost assumptions.",
            "status": "VERIFIED" if decision["evidence_tier"] == "E1" else "HYPOTHESIS",
            "artifact": "result.json",
        },
        {
            "claim_id": "C003",
            "text": "The strategy is ready for live trading or capital allocation.",
            "status": "REJECTED",
            "artifact": "decision.json",
        },
    ]
    if provenance_bound:
        claims.append(
            {
                "claim_id": "C004",
                "text": "The frozen normalized dataset is hash-bound to the registered catalog snapshot and was available by the work-order cutoff.",
                "status": "VERIFIED",
                "artifact": _PROVENANCE_ARTIFACT,
            }
        )
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "claims": claims,
    }


def _quarantine(
    *,
    store: ArtifactStore,
    state: RunStateLog,
    records: list[ArtifactRecord],
    error: DataQualityError,
) -> dict[str, Any]:
    failure = {
        "schema_version": "1.0",
        "status": "QUARANTINED",
        "code": error.code,
        "message": str(error),
        "details": error.details,
    }
    failure_record = store.write_json("data_quality.json", failure)
    records.append(failure_record)
    _advance(
        state,
        QUARANTINED,
        payload={"code": error.code},
        artifact_hashes={failure_record.path: failure_record.sha256},
    )
    report_record = store.write_text(
        "report.md",
        "# QuantOS quarantined run\n\n"
        f"- Code: `{error.code}`\n"
        f"- Reason: {error}\n"
        "- No metric, research promotion, paper status, or live action is authorized.\n",
    )
    records.append(report_record)
    provenance_record = next(
        (record for record in records if record.path == _PROVENANCE_ARTIFACT), None
    )
    summary = {
        "schema_version": "1.0",
        "run_id": store.run_id,
        "status": "QUARANTINED",
        "evidence_tier": "E0",
        "proposed_decision": "REJECT",
        "live_trading_authorized": False,
        "error_code": error.code,
    }
    if provenance_record is not None:
        summary["dataset_provenance_sha256"] = provenance_record.sha256
    summary_record = store.write_json("summary.json", summary)
    records.append(summary_record)
    state_record = _artifact_record(state.path, "state.jsonl")
    records.append(state_record)
    metadata: dict[str, Any] = {"status": "QUARANTINED", "evidence_tier": "E0"}
    if provenance_record is not None:
        metadata["dataset_provenance_sha256"] = provenance_record.sha256
    manifest = store.commit_manifest(records, metadata)
    return {
        **summary,
        "manifest_sha256": manifest.sha256,
        "run_dir": str(store.run_dir),
    }


def run_price_baseline(
    work_order_path: str | Path,
    *,
    workspace: str | Path,
    output_root: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Execute one immutable, research-only, point-in-time baseline run."""

    workspace_path = Path(workspace).resolve()
    order = load_work_order(work_order_path, workspace_path, now=now)
    source = (workspace_path / order.dataset.path).resolve()
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise KernelError("validated dataset became unreadable before freeze") from exc
    if sha256_bytes(source_bytes) != order.dataset.sha256:
        raise KernelError("dataset changed after work-order validation")
    provenance = resolve_dataset_provenance(
        order.dataset,
        workspace=workspace_path,
        cutoff=order.cutoff,
        market_family=order.market_family,
    )
    payload = _order_payload(order)
    run_id = _run_id(payload)
    store = ArtifactStore(Path(output_root), run_id)
    existing_manifest = store.run_dir / "manifest.json"
    if existing_manifest.exists():
        verified = verify_run(store.run_dir)
        summary = json.loads(
            (store.run_dir / "summary.json").read_text(encoding="utf-8")
        )
        manifest_record = _artifact_record(existing_manifest, "manifest.json")
        latest_path = store.root / "latest.json"
        if verified["status"] == HUMAN_REVIEW and not latest_path.exists():
            store.publish_latest(manifest_record)
            verify_run(store.run_dir)
        return {
            **summary,
            "manifest_sha256": manifest_record.sha256,
            "run_dir": str(store.run_dir),
            "idempotent_replay": True,
        }

    records: list[ArtifactRecord] = []
    canonical_order = canonical_json(payload).encode("utf-8") + b"\n"
    order_record = store.write_bytes("work_order.json", canonical_order)
    records.append(order_record)
    state = RunStateLog.initialize(
        store.run_dir / "state.jsonl",
        run_id=run_id,
        actor=KERNEL_ACTOR,
        payload={"work_order_sha256": order_record.sha256},
        artifact_hashes={order_record.path: order_record.sha256},
    )
    source_hashes = {"dataset_source": order.dataset.sha256}
    source_payload: dict[str, Any] = {"dataset_path": order.dataset.path}
    if provenance is not None:
        source_hashes.update(
            {
                "catalog_snapshot": provenance["snapshot"]["snapshot_id"],
                "adapter_contract": provenance["adapter"]["contract_sha256"],
            }
        )
        source_payload["catalog_snapshot_id"] = provenance["snapshot"]["snapshot_id"]
    _advance(
        state,
        SOURCE_RESOLVED,
        payload=source_payload,
        artifact_hashes=source_hashes,
    )

    snapshot_record = store.write_bytes("snapshot/prices.csv", source_bytes)
    if snapshot_record.sha256 != order.dataset.sha256:
        raise KernelError("frozen snapshot hash differs from validated source")
    records.append(snapshot_record)
    provenance_record: ArtifactRecord | None = None
    if provenance is not None:
        provenance_record = store.write_json(_PROVENANCE_ARTIFACT, provenance)
        records.append(provenance_record)
    snapshot = {
        "schema_version": "1.0",
        "source_relative_path": order.dataset.path,
        "snapshot_path": snapshot_record.path,
        "sha256": snapshot_record.sha256,
        "cutoff": _utc(order.cutoff),
        "market_family": order.market_family,
        "format": order.dataset.format,
    }
    if provenance is not None and provenance_record is not None:
        snapshot.update(
            {
                "provenance_path": provenance_record.path,
                "provenance_sha256": provenance_record.sha256,
                "catalog_snapshot_id": provenance["snapshot"]["snapshot_id"],
            }
        )
    snapshot_manifest = store.write_json("snapshot/manifest.json", snapshot)
    records.append(snapshot_manifest)
    frozen_hashes = {
        snapshot_record.path: snapshot_record.sha256,
        snapshot_manifest.path: snapshot_manifest.sha256,
    }
    if provenance_record is not None:
        frozen_hashes[provenance_record.path] = provenance_record.sha256
    _advance(
        state,
        SNAPSHOT_FROZEN,
        payload={"cutoff": _utc(order.cutoff)},
        artifact_hashes=frozen_hashes,
    )

    try:
        rows, data_quality = load_price_rows(
            store.run_dir / snapshot_record.path, order.cutoff
        )
        points = run_lagged_momentum(
            rows, order.strategy.lookback, order.strategy.cost_bps
        )
        splits = split_points(
            points,
            order.experiment.train_fraction,
            order.experiment.validation_fraction,
        )
        if len(splits["test"]) < order.experiment.min_test_observations:
            raise DataQualityError(
                "DQ_INSUFFICIENT_TEST",
                "locked chronological test split is smaller than min_test_observations",
                {
                    "actual": len(splits["test"]),
                    "required": order.experiment.min_test_observations,
                },
            )
        if len({row.price for row in rows}) <= 1:
            raise DataQualityError(
                "DQ_CONSTANT_PRICES",
                "constant prices cannot support the registered directional baseline",
            )
    except DataQualityError as error:
        return _quarantine(store=store, state=state, records=records, error=error)

    data_quality_record = store.write_json("data_quality.json", data_quality)
    records.append(data_quality_record)
    _advance(
        state,
        DATA_QUALITY_PASSED,
        payload={
            "rows": data_quality["row_count"],
            "instruments": data_quality["instrument_count"],
        },
        artifact_hashes={data_quality_record.path: data_quality_record.sha256},
    )
    cheap_test: dict[str, Any] = {
        "checks": {
            "nonconstant_prices": len({row.price for row in rows}) > 1,
            "next_bar_signal_ordering": True,
            "cost_model_registered": True,
            "chronological_split_registered": True,
        },
    }
    cheap_test["status"] = "PASS" if all(cheap_test["checks"].values()) else "FAIL"
    cheap_record = store.write_json("cheap_falsification.json", cheap_test)
    records.append(cheap_record)
    _advance(
        state,
        CHEAP_FALSIFICATION,
        payload={"status": cheap_test["status"]},
        artifact_hashes={cheap_record.path: cheap_record.sha256},
    )

    series_record = store.write_bytes("backtest/series.jsonl", _series_bytes(points))
    records.append(series_record)
    _advance(
        state,
        BASELINE_COMPLETE,
        payload={"observations": len(points), "strategy": order.strategy.kind},
        artifact_hashes={series_record.path: series_record.sha256},
    )
    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "work_order_id": order.work_order_id,
        "market_family": order.market_family,
        "dataset_sha256": order.dataset.sha256,
        "cutoff": _utc(order.cutoff),
        "strategy": asdict(order.strategy),
        "experiment": asdict(order.experiment),
        "metrics": {name: metrics(segment) for name, segment in splits.items()},
        "metric_scope": "equal-weight portfolio, per observation, not annualized",
        "execution_scope": "bar-level delayed fill and following-observation holding return; no queue/fill/cancel model",
        "live_trading_authorized": False,
    }
    if provenance_record is not None:
        result["dataset_provenance_sha256"] = provenance_record.sha256
    result_record = store.write_json("result.json", result)
    records.append(result_record)
    _advance(
        state,
        EXPERIMENT_COMPLETE,
        payload={"splits": {name: len(segment) for name, segment in splits.items()}},
        artifact_hashes={result_record.path: result_record.sha256},
    )

    persisted_result = json.loads(
        (store.run_dir / result_record.path).read_text(encoding="utf-8")
    )
    validation = _artifact_replay_validation(store.run_dir, persisted_result)
    if not all(
        value is True or value == "PASS" or isinstance(value, dict)
        for value in validation["checks"].values()
    ):
        raise KernelError("validation check failed")
    validation_record = store.write_json("validation.json", validation)
    records.append(validation_record)
    _advance(
        state,
        INDEPENDENT_VALIDATION,
        payload={
            "status": "PASS",
            "scope": "persisted_artifact_replay",
            "external_independent_reproduction": False,
            "limitations": len(validation["limitations"]),
        },
        artifact_hashes={validation_record.path: validation_record.sha256},
    )

    decision = _decision(order, result)
    decision_record = store.write_json("decision.json", decision)
    claim_record = store.write_json(
        "claim_ledger.json",
        _claim_ledger(run_id, decision, provenance_bound=provenance_record is not None),
    )
    knowledge = {
        "schema_version": "1.0",
        "knowledge_id": f"hypothesis-{run_id}",
        "type": "experiment_result",
        "status": decision["proposed_decision"],
        "as_of": _utc(order.cutoff),
        "evidence_tier": decision["evidence_tier"],
        "visibility": "private",
        "hypothesis": asdict(order.hypothesis),
        "artifacts": [
            "work_order.json",
            "snapshot/manifest.json",
            *([_PROVENANCE_ARTIFACT] if provenance_record is not None else []),
            "result.json",
            "validation.json",
            "decision.json",
        ],
        "invalidation_rule": "Invalidate if snapshot hash, timing, accounting, or registered experiment contract fails.",
    }
    knowledge_record = store.write_json("knowledge.json", knowledge)
    records.extend([decision_record, claim_record, knowledge_record])
    report_record = store.write_text("report.md", _report(order, result, decision))
    records.append(report_record)
    _advance(
        state,
        REPORT_READY,
        payload={"report_kind": "baseline"},
        artifact_hashes={
            report_record.path: report_record.sha256,
            claim_record.path: claim_record.sha256,
            knowledge_record.path: knowledge_record.sha256,
        },
    )
    _advance(
        state,
        HUMAN_REVIEW,
        payload={
            "proposed_decision": decision["proposed_decision"],
            "human_approval_required": True,
        },
        artifact_hashes={decision_record.path: decision_record.sha256},
    )
    summary = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "HUMAN_REVIEW",
        "evidence_tier": decision["evidence_tier"],
        "proposed_decision": decision["proposed_decision"],
        "live_trading_authorized": False,
    }
    if provenance_record is not None:
        summary["dataset_provenance_sha256"] = provenance_record.sha256
    summary_record = store.write_json("summary.json", summary)
    records.append(summary_record)
    records.append(_artifact_record(state.path, "state.jsonl"))
    manifest_metadata: dict[str, Any] = {
        "status": "HUMAN_REVIEW",
        "evidence_tier": decision["evidence_tier"],
        "kernel_version": __version__,
    }
    if provenance_record is not None:
        manifest_metadata["dataset_provenance_sha256"] = provenance_record.sha256
    manifest = store.commit_manifest(records, manifest_metadata)
    verified = verify_run(store.run_dir)
    if verified["status"] != "HUMAN_REVIEW":
        raise KernelError("completed run did not verify at HUMAN_REVIEW")
    store.publish_latest(manifest)
    verify_run(store.run_dir)
    return {
        **summary,
        "manifest_sha256": manifest.sha256,
        "run_dir": str(store.run_dir),
        "idempotent_replay": False,
    }


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    """Verify artifact hashes, state integrity, and summary/state agreement."""

    path = Path(run_dir).resolve()
    manifest = verify_manifest(path)
    state = RunStateLog(path / "state.jsonl")
    events = state.replay()
    final_state = events[-1].to_state
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KernelError("terminal summary is invalid JSON") from exc
    if summary.get("run_id") != path.name or manifest.get("run_id") != path.name:
        raise KernelError("run identity mismatch")
    if summary.get("status") != final_state:
        raise KernelError("summary status differs from event-sourced state")
    if summary.get("live_trading_authorized") is not False:
        raise KernelError("research run must not authorize live trading")
    if final_state not in {HUMAN_REVIEW, QUARANTINED}:
        raise KernelError(f"run is incomplete at {final_state}")
    frozen = _validate_frozen_dataset(path)
    declared_paths = frozenset(item["path"] for item in manifest["artifacts"])
    expected_paths = (
        _COMPLETED_ARTIFACTS if final_state == HUMAN_REVIEW else _QUARANTINED_ARTIFACTS
    )
    if frozen["provenance_required"]:
        expected_paths = expected_paths | {_PROVENANCE_ARTIFACT}
    if declared_paths != expected_paths:
        missing = sorted(expected_paths - declared_paths)
        unexpected = sorted(declared_paths - expected_paths)
        detail = (missing or unexpected)[0]
        raise KernelError(
            f"manifest artifact contract is incomplete or unexpected: {detail}"
        )
    manifest_hashes = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    for event in events:
        for artifact_path, digest in event.artifact_hashes.items():
            if (
                artifact_path in manifest_hashes
                and manifest_hashes[artifact_path] != digest
            ):
                raise KernelError(
                    f"manifest hash differs from the state commitment: {artifact_path}"
                )
    metadata = manifest.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("status") != final_state
        or metadata.get("evidence_tier") != summary.get("evidence_tier")
        or metadata.get("dataset_provenance_sha256") != frozen["provenance_sha256"]
        or summary.get("dataset_provenance_sha256") != frozen["provenance_sha256"]
    ):
        raise KernelError("manifest metadata differs from the terminal summary")
    if final_state == HUMAN_REVIEW:
        try:
            result = json.loads((path / "result.json").read_text(encoding="utf-8"))
            persisted_validation = json.loads(
                (path / "validation.json").read_text(encoding="utf-8")
            )
            decision = json.loads((path / "decision.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KernelError(
                "completed run contains invalid result artifacts"
            ) from exc
        work_order = frozen["work_order"]
        work_order_dataset = (
            work_order.get("dataset") if isinstance(work_order, dict) else None
        )
        if (
            not isinstance(result, dict)
            or result.get("run_id") != path.name
            or result.get("live_trading_authorized") is not False
            or not isinstance(decision, dict)
            or decision.get("run_id") != path.name
            or decision.get("evidence_tier") != summary.get("evidence_tier")
            or decision.get("proposed_decision") != summary.get("proposed_decision")
            or decision.get("live_trading_authorized") is not False
            or not isinstance(work_order, dict)
            or not isinstance(work_order_dataset, dict)
            or result.get("work_order_id") != work_order.get("work_order_id")
            or result.get("market_family") != work_order.get("market_family")
            or result.get("dataset_sha256") != work_order_dataset.get("sha256")
            or result.get("cutoff") != work_order.get("cutoff")
            or result.get("experiment") != work_order.get("experiment")
            or result.get("strategy") != work_order.get("strategy")
            or work_order.get("evidence_ceiling") != "E0"
            or summary.get("evidence_tier") != "E0"
        ):
            raise KernelError("completed result or decision contract is inconsistent")
        if not isinstance(persisted_validation, dict):
            raise KernelError("persisted validation is not an object")
        replayed_validation = _artifact_replay_validation(
            path, result, schema_version=persisted_validation.get("schema_version", "")
        )
        if canonical_json(replayed_validation) != canonical_json(persisted_validation):
            raise KernelError("persisted validation differs from fresh artifact replay")
    latest_path = path.parent.parent / "latest.json"
    if latest_path.exists() or latest_path.is_symlink():
        if latest_path.is_symlink() or not latest_path.is_file():
            raise KernelError("latest pointer must be a regular JSON file")
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KernelError("latest pointer is invalid JSON") from exc
        if (
            not isinstance(latest, dict)
            or set(latest)
            != {"schema_version", "run_id", "manifest_path", "manifest_sha256"}
            or latest.get("schema_version") != "1.0"
            or not isinstance(latest.get("run_id"), str)
            or latest.get("manifest_path")
            != f"runs/{latest.get('run_id')}/manifest.json"
            or not isinstance(latest.get("manifest_sha256"), str)
            or len(latest["manifest_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in latest["manifest_sha256"]
            )
        ):
            raise KernelError("latest pointer schema is invalid")
        if latest.get("run_id") == path.name:
            expected_path = f"runs/{path.name}/manifest.json"
            if (
                latest.get("schema_version") != "1.0"
                or latest.get("manifest_path") != expected_path
                or latest.get("manifest_sha256") != sha256_file(path / "manifest.json")
            ):
                raise KernelError("latest pointer does not anchor the run manifest")
    return {
        "run_id": path.name,
        "status": final_state,
        "evidence_tier": summary.get("evidence_tier"),
        "manifest_artifacts": len(manifest.get("artifacts", [])),
        "state_events": len(state.replay()),
    }


__all__ = ["KernelError", "run_price_baseline", "verify_run"]
