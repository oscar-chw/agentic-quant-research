#!/usr/bin/env python3
"""Validate a QuantOS evidence envelope without third-party dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.0"
MAX_ENVELOPE_BYTES = 1_048_576
MODES = frozenset({"frame", "audit", "report"})
MODE_ACTORS = {"frame": "research", "audit": "audit", "report": "report"}
DECISIONS = frozenset(
    {"REJECT", "RESEARCH", "REVISE", "DEEPEN", "REPORT", "PAPER_CANDIDATE"}
)
EVIDENCE_TIERS = ("E0", "E1", "E2", "E3", "E4", "E5")
CHECK_STATUSES = frozenset({"pass", "fail", "unavailable", "not_applicable"})
CHECK_SEVERITIES = frozenset({"info", "warning", "hard"})
CONFIDENCE_COMPONENTS = (
    "C_data",
    "C_mechanism",
    "C_stat",
    "C_generalization",
    "C_model",
    "C_execution",
    "C_capacity",
    "C_risk",
    "C_ops",
    "C_repro",
    "C_governance",
)
CONTRACT_FIELDS = frozenset(
    {
        "source",
        "snapshot",
        "feature",
        "label",
        "hypothesis",
        "strategy",
        "model",
        "experiment",
        "replay",
        "result",
        "decision",
        "report",
        "eval_case",
    }
)
RUN_STATES = (
    "REGISTERED",
    "SOURCE_RESOLVED",
    "SNAPSHOT_FROZEN",
    "DATA_QUALITY_PASSED",
    "QUARANTINED",
    "CHEAP_FALSIFICATION",
    "BASELINE_COMPLETE",
    "EXPERIMENT_COMPLETE",
    "INDEPENDENT_VALIDATION",
    "REPORT_READY",
    "HUMAN_REVIEW",
)
ALLOWED_TRANSITIONS = {
    "REGISTERED": frozenset({"SOURCE_RESOLVED"}),
    "SOURCE_RESOLVED": frozenset({"SNAPSHOT_FROZEN"}),
    "SNAPSHOT_FROZEN": frozenset({"DATA_QUALITY_PASSED", "QUARANTINED"}),
    "DATA_QUALITY_PASSED": frozenset({"CHEAP_FALSIFICATION"}),
    "QUARANTINED": frozenset(),
    "CHEAP_FALSIFICATION": frozenset({"BASELINE_COMPLETE"}),
    "BASELINE_COMPLETE": frozenset({"EXPERIMENT_COMPLETE"}),
    "EXPERIMENT_COMPLETE": frozenset({"INDEPENDENT_VALIDATION"}),
    "INDEPENDENT_VALIDATION": frozenset({"REPORT_READY"}),
    "REPORT_READY": frozenset({"HUMAN_REVIEW"}),
    "HUMAN_REVIEW": frozenset(),
}
BLOCKER_PREFIXES = frozenset(
    {
        "DQ",
        "PIT",
        "STAT",
        "ML",
        "UNIT",
        "ACCT",
        "EXEC",
        "RISK",
        "STATE",
        "OPS",
        "REPORT",
        "CLAIM",
        "SEC",
        "SCOPE",
    }
)
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "skill_version",
        "mode",
        "actor_role",
        "facts",
        "assumptions",
        "unknowns",
        "input_artifacts",
        "contracts",
        "run_state_before",
        "requested_state",
        "run_state_after",
        "checks",
        "evidence_tier",
        "confidence_vector",
        "decision",
        "hard_blocks",
        "permitted_claims",
        "prohibited_claims",
        "next_test",
        "knowledge_records",
        "numeric_results",
        "safety",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCKER_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_ABSOLUTE_LOCAL_PATH_RE = re.compile(
    r"(?:^|[\s'\"`(])(?:[A-Za-z]:[\\/]|/(?:home|Users|private|tmp|var)/)"
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)['\"]?(?:api[_ -]?key|access[_ -]?token|password|secret|"
        r"client[_ -]?secret|private[_ -]?key|wallet[_ -]?key|authorization)['\"]?"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{12,}"
    ),
)
_OVERCLAIM_PATTERNS = (
    re.compile(r"(?i)\bproduction[- ]ready\b"),
    re.compile(r"(?i)\bready\s+for\s+production\b"),
    re.compile(r"(?i)\bfund[- ]ready\b"),
    re.compile(r"(?i)\bdeployable\b"),
    re.compile(r"(?i)\blive\s+(?:sharpe|performance|profit|pnl)\b"),
    re.compile(r"(?i)\bguaranteed(?:\s+profit)?\b"),
    re.compile(r"(?i)\brisk[- ]free\b"),
)
_E0_EMPIRICAL_PATTERNS = (
    re.compile(
        r"(?i)\b(?:backtest(?:ed)?|historical|out[- ]of[- ]sample|oos)\s+"
        r"(?:sharpe|sortino|calmar|return|performance|profit|pnl|drawdown)\b"
    ),
    re.compile(
        r"(?i)\b(?:sharpe|sortino|calmar|information\s+ratio|annualized\s+return|"
        r"max(?:imum)?\s+drawdown)\s*(?:of|=|:)?\s*[-+]?\d"
    ),
)
_ACTION_COMPLETION_PATTERNS = (
    re.compile(
        r"(?i)\b(?:placed|submitted|executed|cancelled|signed)\s+(?:a\s+)?live\s+order\b"
    ),
    re.compile(
        r"(?i)\blive\s+order\s+(?:was|is|has been)\s+"
        r"(?:placed|submitted|executed|cancelled|signed)\b"
    ),
    re.compile(
        r"(?i)\b(?:started|exposed|deployed|opened)\s+(?:a\s+)?"
        r"(?:public|unauthenticated)\s+(?:codex|mcp).*(?:endpoint|listener)\b"
    ),
)


def _strict_keys(
    value: Any, expected: set[str] | frozenset[str], field: str, errors: list[str]
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{field}:TYPE")
        return None
    actual = set(value)
    if actual != set(expected):
        errors.append(f"{field}:FIELDS")
    return value


def _string_list(value: Any, field: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{field}:STRING_LIST")
        return None
    if any(not item.strip() for item in value):
        errors.append(f"{field}:EMPTY_STRING")
    if len(value) != len(set(value)):
        errors.append(f"{field}:DUPLICATE")
    return value


def _walk_strings(value: Any):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            stack.extend(item)


def _portable_artifact_path(value: str) -> bool:
    if "://" in value:
        return value.startswith(("artifact://", "https://", "urn:"))
    normalized = value.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", normalized) or ":" in normalized:
        return False
    path = PurePosixPath(normalized)
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts


def _utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifacts(
    artifacts: list[Any], artifact_root: Path | None, errors: list[str]
) -> None:
    if artifact_root is None:
        errors.append("evidence_tier:ARTIFACT_VERIFICATION_REQUIRED")
        return
    try:
        root = artifact_root.resolve(strict=True)
        root_metadata = root.lstat()
    except OSError:
        errors.append("artifact_root:UNAVAILABLE")
        return
    root_attributes = int(getattr(root_metadata, "st_file_attributes", 0))
    if not root.is_dir() or root.is_symlink() or bool(root_attributes & 0x400):
        errors.append("artifact_root:UNSAFE")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            continue
        relative = artifact.get("path")
        if not isinstance(relative, str) or "://" in relative:
            errors.append(f"input_artifacts[{index}]:UNVERIFIABLE")
            continue
        try:
            candidate = (root / relative.replace("\\", "/")).resolve(strict=True)
            candidate.relative_to(root)
            metadata = candidate.lstat()
        except (OSError, ValueError):
            errors.append(f"input_artifacts[{index}]:UNAVAILABLE")
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or bool(attributes & 0x400)
        ):
            errors.append(f"input_artifacts[{index}]:UNSAFE")
            continue
        if _artifact_digest(candidate) != artifact.get("sha256"):
            errors.append(f"input_artifacts[{index}]:HASH_MISMATCH")


def validate_envelope(value: Any, artifact_root: Path | None = None) -> list[str]:
    """Return stable validation errors; an empty list means valid."""

    errors: list[str] = []
    envelope = _strict_keys(value, TOP_LEVEL_FIELDS, "envelope", errors)
    if envelope is None:
        return errors

    if envelope.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version:VALUE")
    task_id = envelope.get("task_id")
    if not isinstance(task_id, str) or not _IDENTIFIER_RE.fullmatch(task_id):
        errors.append("task_id:VALUE")
    skill_version = envelope.get("skill_version")
    if not isinstance(skill_version, str) or not _SEMVER_RE.fullmatch(skill_version):
        errors.append("skill_version:VALUE")
    mode = envelope.get("mode")
    if mode not in MODES:
        errors.append("mode:VALUE")
    actor_role = envelope.get("actor_role")
    if actor_role not in MODE_ACTORS.values():
        errors.append("actor_role:VALUE")
    elif mode in MODE_ACTORS and actor_role != MODE_ACTORS[mode]:
        errors.append("actor_role:MODE_MISMATCH")

    text_lists: dict[str, list[str]] = {}
    for field in ("facts", "assumptions", "unknowns"):
        text_lists[field] = _string_list(envelope.get(field), field, errors) or []

    artifacts = envelope.get("input_artifacts")
    artifact_paths: set[str] = set()
    if not isinstance(artifacts, list):
        errors.append("input_artifacts:TYPE")
    else:
        artifact_fields = {
            "path",
            "cutoff",
            "sha256",
            "schema",
            "visibility",
            "license",
        }
        for index, raw in enumerate(artifacts):
            artifact = _strict_keys(
                raw, artifact_fields, f"input_artifacts[{index}]", errors
            )
            if artifact is None:
                continue
            path = artifact.get("path")
            if not isinstance(path, str) or not _portable_artifact_path(path):
                errors.append(f"input_artifacts[{index}].path:VALUE")
            else:
                artifact_paths.add(path)
            cutoff = artifact.get("cutoff")
            if not _utc_timestamp(cutoff):
                errors.append(f"input_artifacts[{index}].cutoff:VALUE")
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                errors.append(f"input_artifacts[{index}].sha256:VALUE")
            for field in ("schema", "visibility", "license"):
                item = artifact.get(field)
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"input_artifacts[{index}].{field}:VALUE")

    contracts = envelope.get("contracts")
    if not isinstance(contracts, Mapping):
        errors.append("contracts:TYPE")
    else:
        for key, item in contracts.items():
            if not isinstance(key, str) or key not in CONTRACT_FIELDS:
                errors.append("contracts:KEY")
            if isinstance(item, str):
                valid_item = bool(item.strip())
            elif isinstance(item, list):
                valid_item = bool(item) and all(
                    isinstance(member, str) and member.strip() for member in item
                )
            else:
                valid_item = False
            if not valid_item:
                errors.append(f"contracts.{key}:VALUE")

    before = envelope.get("run_state_before")
    requested = envelope.get("requested_state")
    after = envelope.get("run_state_after")
    if before not in RUN_STATES:
        errors.append("run_state_before:VALUE")
    if requested not in (*RUN_STATES, "NONE"):
        errors.append("requested_state:VALUE")
    if after not in RUN_STATES:
        errors.append("run_state_after:VALUE")

    hard_blocks = _string_list(envelope.get("hard_blocks"), "hard_blocks", errors)
    hard_blocks = hard_blocks or []
    for block in hard_blocks:
        prefix = block.split("_", 1)[0]
        if not _BLOCKER_RE.fullmatch(block) or prefix not in BLOCKER_PREFIXES:
            errors.append("hard_blocks:CODE")

    if before in RUN_STATES and after in RUN_STATES and after != before:
        errors.append("run_state_after:STATE_CHANGE_FORBIDDEN")
        if after not in ALLOWED_TRANSITIONS[before]:
            errors.append("run_state_after:INVALID_TRANSITION")
    if before in RUN_STATES and requested in RUN_STATES:
        invalid_requested = requested not in ALLOWED_TRANSITIONS[before]
        if invalid_requested:
            if "STATE_INVALID_TRANSITION" not in hard_blocks:
                errors.append("requested_state:MISSING_BLOCK")
        elif "STATE_INVALID_TRANSITION" in hard_blocks:
            errors.append("requested_state:SPURIOUS_BLOCK")
    elif requested == "NONE" and after != before:
        errors.append("requested_state:NONE_CHANGED_STATE")

    checks = envelope.get("checks")
    failed_hard: list[str] = []
    check_evidence_refs: set[str] = set()
    if not isinstance(checks, list):
        errors.append("checks:TYPE")
    else:
        check_fields = {"id", "status", "evidence_ref", "severity", "reason"}
        seen: set[str] = set()
        for index, raw in enumerate(checks):
            check = _strict_keys(raw, check_fields, f"checks[{index}]", errors)
            if check is None:
                continue
            check_id = check.get("id")
            if not isinstance(check_id, str) or not _BLOCKER_RE.fullmatch(check_id):
                errors.append(f"checks[{index}].id:VALUE")
            elif check_id in seen:
                errors.append("checks:DUPLICATE_ID")
            else:
                seen.add(check_id)
            if check.get("status") not in CHECK_STATUSES:
                errors.append(f"checks[{index}].status:VALUE")
            if check.get("severity") not in CHECK_SEVERITIES:
                errors.append(f"checks[{index}].severity:VALUE")
            for field in ("evidence_ref", "reason"):
                item = check.get(field)
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"checks[{index}].{field}:VALUE")
            evidence_ref = check.get("evidence_ref")
            if isinstance(evidence_ref, str) and evidence_ref.strip():
                check_evidence_refs.add(evidence_ref)
            if check.get("status") == "fail" and check.get("severity") == "hard":
                if isinstance(check_id, str):
                    failed_hard.append(check_id)
    if set(failed_hard) != set(hard_blocks):
        errors.append("hard_blocks:CHECK_MISMATCH")

    tier = envelope.get("evidence_tier")
    if tier not in EVIDENCE_TIERS:
        errors.append("evidence_tier:VALUE")
    elif tier in {"E3", "E4", "E5"}:
        errors.append("evidence_tier:SKILL_CEILING")
    confidence = envelope.get("confidence_vector")
    mapped_confidence_blocks: set[str] = set()
    if not isinstance(confidence, Mapping) or set(confidence) != set(
        CONFIDENCE_COMPONENTS
    ):
        errors.append("confidence_vector:COMPONENTS")
    else:
        component_fields = {
            "status",
            "evidence_refs",
            "tier",
            "rubric",
            "staleness",
            "blocker",
        }
        for name in CONFIDENCE_COMPONENTS:
            component = _strict_keys(
                confidence.get(name),
                component_fields,
                f"confidence_vector.{name}",
                errors,
            )
            if component is None:
                continue
            if component.get("status") not in CHECK_STATUSES:
                errors.append(f"confidence_vector.{name}.status:VALUE")
            evidence_refs = (
                _string_list(
                    component.get("evidence_refs"),
                    f"confidence_vector.{name}.evidence_refs",
                    errors,
                )
                or []
            )
            if component.get("status") == "pass" and not evidence_refs:
                errors.append(f"confidence_vector.{name}.evidence_refs:REQUIRED")
            if component.get("status") == "pass" and any(
                ref not in artifact_paths | check_evidence_refs for ref in evidence_refs
            ):
                errors.append(f"confidence_vector.{name}.evidence_refs:UNBOUND")
            component_tier = component.get("tier")
            if component_tier not in EVIDENCE_TIERS:
                errors.append(f"confidence_vector.{name}.tier:VALUE")
            elif tier in EVIDENCE_TIERS and EVIDENCE_TIERS.index(
                component_tier
            ) > EVIDENCE_TIERS.index(tier):
                errors.append(f"confidence_vector.{name}.tier:CEILING")
            for field in ("rubric", "staleness"):
                item = component.get(field)
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"confidence_vector.{name}.{field}:VALUE")
            blocker = component.get("blocker")
            if blocker is not None and blocker not in hard_blocks:
                errors.append(f"confidence_vector.{name}.blocker:VALUE")
            elif blocker is not None:
                mapped_confidence_blocks.add(blocker)
                if component.get("status") != "fail":
                    errors.append(f"confidence_vector.{name}.blocker:STATUS")
    if set(hard_blocks) != mapped_confidence_blocks:
        errors.append("hard_blocks:CONFIDENCE_MISMATCH")

    decision = envelope.get("decision")
    if decision not in DECISIONS:
        errors.append("decision:VALUE")
    if decision == "PAPER_CANDIDATE":
        errors.append("decision:EVIDENCE_CEILING")
    if hard_blocks and decision not in {"REJECT", "REVISE"}:
        errors.append("decision:HARD_BLOCKED")

    if tier in {"E1", "E2"}:
        artifact_items = artifacts if isinstance(artifacts, list) else []
        if not artifact_items:
            errors.append("evidence_tier:ARTIFACTS_REQUIRED")
        else:
            _verify_artifacts(artifact_items, artifact_root, errors)
        passing_checks = {
            item.get("id")
            for item in checks or []
            if isinstance(item, Mapping) and item.get("status") == "pass"
        }
        required_checks = {
            "PIT_POINT_IN_TIME",
            "ACCT_RECONCILIATION",
            "STAT_CHRONOLOGICAL_OOS",
            "EXEC_COST_MODEL",
        }
        if tier == "E2":
            required_checks |= {"EXEC_EVENT_REPLAY", "ACCT_LIFECYCLE_RECONCILIATION"}
        for check_id in sorted(required_checks - passing_checks):
            errors.append(f"evidence_tier:MISSING_CHECK:{check_id}")
        for item in checks or []:
            if (
                isinstance(item, Mapping)
                and item.get("id") in required_checks
                and item.get("status") == "pass"
                and item.get("evidence_ref") not in artifact_paths
            ):
                errors.append(f"evidence_tier:UNBOUND_CHECK:{item.get('id')}")

    permitted = _string_list(
        envelope.get("permitted_claims"), "permitted_claims", errors
    )
    prohibited = _string_list(
        envelope.get("prohibited_claims"), "prohibited_claims", errors
    )
    permitted = permitted or []
    prohibited = prohibited or []
    if {item.casefold() for item in permitted} & {
        item.casefold() for item in prohibited
    }:
        errors.append("claims:OVERLAP")
    if tier in {"E0", "E1", "E2"}:
        if any(
            pattern.search(claim)
            for claim in text_lists["facts"]
            for pattern in _OVERCLAIM_PATTERNS
        ):
            errors.append("facts:EVIDENCE_CEILING")
        if tier == "E0" and any(
            pattern.search(claim)
            for claim in [*text_lists["facts"], *permitted]
            for pattern in _E0_EMPIRICAL_PATTERNS
        ):
            errors.append("claims:E0_EMPIRICAL")
        for claim in permitted:
            if any(pattern.search(claim) for pattern in _OVERCLAIM_PATTERNS):
                errors.append("permitted_claims:EVIDENCE_CEILING")
                break
    for text in [*text_lists["facts"], *permitted]:
        if any(pattern.search(text) for pattern in _ACTION_COMPLETION_PATTERNS):
            errors.append("claims:LIVE_ACTION_COMPLETION")
            break

    next_test = _strict_keys(
        envelope.get("next_test"),
        {"description", "required_inputs", "stop_rule"},
        "next_test",
        errors,
    )
    if next_test is not None:
        for field in ("description", "stop_rule"):
            item = next_test.get(field)
            if not isinstance(item, str) or not item.strip():
                errors.append(f"next_test.{field}:VALUE")
        _string_list(
            next_test.get("required_inputs"), "next_test.required_inputs", errors
        )

    records = envelope.get("knowledge_records")
    if not isinstance(records, list):
        errors.append("knowledge_records:TYPE")
    else:
        record_fields = {"id", "type", "status", "provenance"}
        for index, raw in enumerate(records):
            record = _strict_keys(
                raw, record_fields, f"knowledge_records[{index}]", errors
            )
            if record is None:
                continue
            for field in ("id", "type", "status"):
                item = record.get(field)
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"knowledge_records[{index}].{field}:VALUE")
            _string_list(
                record.get("provenance"),
                f"knowledge_records[{index}].provenance",
                errors,
            )

    numeric = envelope.get("numeric_results")
    if not isinstance(numeric, Mapping):
        errors.append("numeric_results:TYPE")
    else:
        for key, item in numeric.items():
            if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
                errors.append("numeric_results:KEY")
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
            ):
                errors.append(f"numeric_results.{key}:VALUE")

    safety = _strict_keys(
        envelope.get("safety"),
        {
            "secret_detected",
            "secret_echoed",
            "live_action_requested",
            "live_action_executed",
            "public_endpoint_requested",
            "public_endpoint_started",
        },
        "safety",
        errors,
    )
    if safety is not None:
        for field, item in safety.items():
            if not isinstance(item, bool):
                errors.append(f"safety.{field}:TYPE")
        if safety.get("secret_echoed") is not False:
            errors.append("safety.secret_echoed:FORBIDDEN")
        if safety.get("live_action_executed") is not False:
            errors.append("safety.live_action_executed:FORBIDDEN")
        if safety.get("public_endpoint_started") is not False:
            errors.append("safety.public_endpoint_started:FORBIDDEN")
        expected_safety_blocks = {
            "SEC_SECRET": safety.get("secret_detected") is True,
            "SEC_LIVE_ACTION": safety.get("live_action_requested") is True,
            "SEC_PUBLIC_ENDPOINT": safety.get("public_endpoint_requested") is True,
        }
        for code, expected in expected_safety_blocks.items():
            if (code in hard_blocks) != expected:
                errors.append(f"safety:{code}_MISMATCH")

    for text in _walk_strings(envelope):
        if _ABSOLUTE_LOCAL_PATH_RE.search(text):
            errors.append("envelope:ABSOLUTE_LOCAL_PATH")
            break
    serialized = json.dumps(
        envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if any(pattern.search(serialized) for pattern in _SECRET_VALUE_PATTERNS):
        errors.append("envelope:SECRET_VALUE")

    return sorted(set(errors))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_json(path: str) -> Any:
    if path == "-":
        raw = sys.stdin.buffer.read(MAX_ENVELOPE_BYTES + 1)
    else:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_ENVELOPE_BYTES + 1)
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise ValueError("input too large")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("envelope", help="JSON envelope path, or - for stdin")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="trusted root used to rehash relative input artifacts for E1/E2",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable status"
    )
    args = parser.parse_args(argv)
    try:
        value = _load_json(args.envelope)
        errors = validate_envelope(value, artifact_root=args.artifact_root)
    except (OSError, UnicodeError, ValueError, RecursionError, json.JSONDecodeError):
        errors = ["input:UNREADABLE_JSON"]
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors}, separators=(",", ":")))
    elif errors:
        print("INVALID " + " ".join(errors))
    else:
        print("VALID")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
