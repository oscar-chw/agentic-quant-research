"""Read-only adapters to native quant results; numerical logic stays with owners."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

from qrae.path_safety import has_link_or_reparse_component
from qrae.quote_features import strict_json

MAX_JSON = 16 * 1024 * 1024
MAX_EVIDENCE = 128 * 1024 * 1024
KINDS = {
    "quantos-feed",
    "quantos-execution",
    "quantos-persisted",
    "factor-trial",
    "imc4-import",
}


class Evidence:
    """Bind only selected files, rejecting links and bounded by total input bytes."""

    def __init__(self):
        self.files = {}
        self.total = 0

    def bind(self, path, role, *, content=False):
        path = Path(path).absolute()
        if has_link_or_reparse_component(path) or not path.is_file():
            raise ValueError("regular, unlinked evidence required: " + str(path))
        name = str(path.resolve())
        digest, chunks, size = hashlib.sha256(), [], 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                if size > MAX_JSON:
                    raise ValueError("evidence file exceeds byte cap")
                digest.update(chunk)
                if content:
                    chunks.append(chunk)
        item = {"sha256": digest.hexdigest(), "bytes": size, "role": role}
        if name in self.files:
            old = self.files[name]
            if (old["sha256"], old["bytes"]) != (item["sha256"], size):
                raise ValueError("evidence changed during verification")
        else:
            if len(self.files) >= 256 or self.total + size > MAX_EVIDENCE:
                raise ValueError("entry exceeds 256 files / 128 MiB")
            self.files[name] = item
            self.total += size
        return b"".join(chunks) if content else item

    def json(self, path, role="artifact"):
        return strict_json(self.bind(path, role, content=True))

    def flat(self, folder):
        folder = Path(folder)
        if has_link_or_reparse_component(folder) or not folder.is_dir():
            raise ValueError("unlinked run directory required")
        paths = sorted(folder.iterdir())
        if len(paths) > 256:
            raise ValueError("run exceeds file cap")
        for path in paths:
            self.bind(path, "native artifact")

    def methods(self, names):
        result = []
        for name, purpose in names:
            module = importlib.import_module(name)  # Names fixed in this module.
            path = Path(module.__file__)
            self.bind(path, "installed method code")
            result.append(
                {"module": name, "path": str(path.resolve()), "purpose": purpose}
            )
        return result

    def unchanged(self):
        for name, old in list(self.files.items()):
            self.bind(Path(name), old["role"])


def _feed(path, evidence):
    from quantos_showcase.feed import verify_feed

    if path.parent.name != "runs" or path.parent.parent.name != "admissions":
        raise ValueError("feed admission directory required")
    root = path.parents[2]
    evidence.flat(path)
    reference = evidence.json(path / "research-ref.json")
    # IDs are validated by the owner before their paths are consumed here.
    result = verify_feed(root, path.name, True)
    features = root / "features/runs" / result["feature_id"]
    experiment = root / "experiments/runs" / result["research_run_id"]
    evidence.flat(features)
    evidence.flat(experiment)
    report = evidence.json(experiment / "report.json")
    if reference["feature_id"] != result["feature_id"]:
        raise ValueError("feature reference mismatch")
    methods = evidence.methods(
        [
            ("replay.admission", "Archived-message admission and continuity"),
            ("replay.feed", "Aggregate book update semantics"),
            ("replay.book", "Integer book state"),
            ("qrae.quote_features", "As-of features and fixed forecast errors"),
            (
                "qrae.quote_workflow",
                "Feature materialization and immutable experiments",
            ),
            ("quantos_showcase.feed", "Raw-to-report verification"),
        ]
    )
    return dict(
        home="QuantOS",
        result_state="COMPLETED",
        verification="NATIVE_RECOMPUTED",
        scope="Next-decision quote forecast; rational ticks, not trading returns",
        report=str(experiment / "REPORT.md"),
        methods=methods,
        metrics={"price_tick": report["price_tick"], "splits": report["splits"]},
        qualifications={
            k: report[k]
            for k in (
                "data_kind",
                "evidence_tier",
                "availability",
                "profitability_evaluated",
                "exchange_completeness_verified",
            )
        },
    )


def _factor(path, evidence):
    from factor_research.reports import human_report
    from factor_research.runs import verify_run

    evidence.flat(path)
    result = verify_run(path, recompute=True)
    methods = evidence.methods(
        [
            ("factor_research.runs", "Immutable trial identity and verification"),
            (
                "factor_research.evaluator",
                "Feature, rank-IC, split selection and cost calculation",
            ),
            ("factor_research.prepare", "Explicit source/clock/universe admission"),
        ]
    )
    if result["status"] == "FAILED":
        return dict(
            home="Factor research",
            result_state="FAILED",
            verification="NATIVE_INTEGRITY_ONLY",
            scope="Recorded failed trial; no successful research result",
            report=str(path / "failure.json"),
            methods=methods,
            metrics=None,
            qualifications={
                "failure": evidence.json(path / "failure.json"),
                "failure_reexecuted": False,
            },
        )
    report = evidence.json(path / "report.json")
    if human_report(report).encode() != evidence.bind(
        path / "report.md", "native report", content=True
    ):
        raise ValueError("factor rendered report differs")
    return dict(
        home="Factor research",
        result_state="COMPLETED",
        verification="NATIVE_RECOMPUTED",
        scope="Rank correlation and independent interval sleeves; no compounded portfolio comparison",
        report=str(path / "report.md"),
        methods=methods,
        metrics={
            "selection": report["selection"],
            "test": report["test"]["summary"] if report["test"] else None,
            "units": report["units"],
        },
        qualifications={
            "limitations": report["limitations"],
            "data_source": report.get(
                "data_source",
                "synthetic/offline panel; historical authenticity unverified",
            ),
        },
    )


def _imc(path, evidence):
    from imc4_analysis import __version__
    from imc4_analysis.analyzer import analyze
    from imc4_analysis.community_import import normalize_community_log
    from imc4_analysis.provenance import code_identity
    from imc4_analysis.report import render

    members = {
        "raw-source.log",
        "import-config.json",
        "normalized.jsonl",
        "import-receipt.json",
        "report.json",
        "report.html",
    }
    evidence.flat(path)
    completion = evidence.json(path / "complete.json")
    if (
        completion.get("schema") != "imc4-analysis-completion/v1"
        or set(completion["files"]) != members
        or {p.name for p in path.iterdir()} != members | {"complete.json"}
    ):
        raise ValueError(
            "complete imported IMC bundle required; plain analysis needs retained source"
        )
    for name, commitment in completion["files"].items():
        actual = evidence.bind(path / name, "native artifact")
        if commitment != {k: actual[k] for k in ("sha256", "bytes")}:
            raise ValueError("IMC completion hash mismatch: " + name)
    report = evidence.json(path / "report.json")
    if (
        report.get("schema") != "imc4-analysis-report/v1"
        or report["code_identity"] != code_identity()
    ):
        raise ValueError("IMC recomputation requires original installed code")
    for name in (
        "tool_version",
        "code_identity",
        "runtime",
        "source_sha256",
        "import_provenance",
    ):
        if completion[name] != report[name]:
            raise ValueError("IMC completion/report identity mismatch")
    normalized, provenance = normalize_community_log(
        evidence.bind(path / "raw-source.log", "raw source", content=True),
        evidence.bind(path / "import-config.json", "import contract", content=True),
    )
    if (
        normalized
        != evidence.bind(path / "normalized.jsonl", "normalized source", content=True)
        or provenance != report["import_provenance"]
    ):
        raise ValueError("IMC import recomputation differs")
    actual = analyze(
        normalized,
        detail=report["details"]["mode"],
        sample_limit=report["details"]["sample_limit"],
    )
    actual["import_provenance"] = provenance
    if {k: v for k, v in actual.items() if k != "runtime"} != {
        k: v for k, v in report.items() if k != "runtime"
    }:
        raise ValueError("IMC analysis recomputation differs")
    expected_receipt = dict(
        provenance,
        code_identity=report["code_identity"],
        runtime=report["runtime"],
        tool_version=__version__,
    )
    if evidence.json(path / "import-receipt.json") != expected_receipt:
        raise ValueError("IMC import receipt differs")
    if render(report).encode() != evidence.bind(
        path / "report.html", "native report", content=True
    ):
        raise ValueError("IMC rendered report differs")
    methods = evidence.methods(
        [
            (
                "imc4_analysis.community_import",
                "Declared competition-log conversion and fill identity",
            ),
            (
                "imc4_analysis.analyzer",
                "Cash, inventory, mark freshness and fill markouts",
            ),
            ("imc4_analysis.report", "Native diagnostic view"),
        ]
    )
    return dict(
        home="IMC 4",
        result_state="COMPLETED",
        verification="NATIVE_RECOMPUTED",
        scope="Imported competition accounting and markouts on an opaque source clock",
        report=str(path / "report.html"),
        methods=methods,
        metrics={
            "summary": report["summary"],
            "currency": report["currency"],
            "timestamp_unit": report["timestamp_unit"],
        },
        qualifications={
            "data_kind": report["data_kind"],
            "history": report["provenance"],
            "import_limits": provenance["limitations"],
            "runtime_metadata": "retained as reported; not rerun equality across interpreters",
        },
    )


def _execution(path, evidence):
    from quantos_showcase.execution import verify_execution

    if path.parent.name != "runs":
        raise ValueError("execution attempt directory required")
    result = verify_execution(path.parents[1], path.name, evidence=evidence)
    methods = evidence.methods(
        [
            (
                "imc4_analysis.quoting",
                "Inventory reservation price, ticks, lots and live capacity",
            ),
            (
                "imc4_analysis.quote_study",
                "Native finite decisions, cancellation, matching and fills",
            ),
            (
                "imc4_analysis.analyzer",
                "Native cash, positions, fees and marked valuation",
            ),
            ("source_access", "Current local source-note eligibility"),
            (
                "quantos_showcase.execution",
                "Registered finite attempt and exact native replay",
            ),
        ]
    )
    return dict(
        home="QuantOS",
        result_state="COMPLETED",
        verification=result["verification"],
        scope="Native IMC synthetic quote study; finite new-attempt lifecycle, no pooled P&L",
        report=result["report"],
        methods=methods,
        metrics={"native_rows": result["rows"], "counts": result["counts"]},
        qualifications={
            "data_kind": "synthetic",
            "lifecycle": result["lifecycle"],
            "market_quality": "NOT_RUN",
            "resumable_service": False,
            "source_authenticity": "not verified",
        },
    )


def _persisted(path, evidence):
    from quantos_showcase.persisted import verify

    if path.parent.name != "runs" or path.parent.parent.name != "persisted":
        raise ValueError("persisted wrapper directory required")
    result = verify(path.parents[2], path.name, evidence=evidence)
    if result["execution"] != "COMPLETED":
        return {
            "home": "QuantOS",
            "result_state": result["execution"],
            "verification": result["verification"],
            "reason": "Native/wrapper lifecycle: " + result["execution"],
        }
    methods = evidence.methods(
        [
            ("imc4_analysis.quoting", "Native reservation quotes and live capacity"),
            ("imc4_analysis.quote_study", "Native one-step execution and effects"),
            (
                "imc4_analysis.persistence",
                "Committed suffix recovery and native publication",
            ),
            ("imc4_analysis.analyzer", "Sole native monetary accounting report"),
            ("quantos_showcase.persisted", "Frozen context and wrapper finalization"),
            ("source_access", "Current frozen note eligibility"),
        ]
    )
    return dict(
        home="QuantOS",
        result_state="COMPLETED",
        verification=result["verification"],
        scope="Native persisted synthetic quote study; state verification without policy replay",
        report=result["report"],
        methods=methods,
        metrics={"native_rows": result["rows"]},
        qualifications={
            "data_kind": "synthetic",
            "lifecycle": result["lifecycle"],
            "market_quality": "NOT_RUN",
            "power_loss": "NOT_RUN",
        },
    )


def inspect_run(entry, base):
    evidence = Evidence()
    try:

        def input_path(value):
            candidate = base / value
            if has_link_or_reparse_component(candidate):
                raise ValueError("unlinked selected input required")
            # Relative references may be anchored at a not-yet-created index
            # directory. Collapse dot components only after checking for links.
            return Path(os.path.abspath(candidate))

        path = input_path(entry["path"])
        if entry["kind"] == "quantos-feed":
            row = _feed(path, evidence)
        elif entry["kind"] == "quantos-persisted":
            row = _persisted(path, evidence)
        elif entry["kind"] == "quantos-execution":
            row = _execution(path, evidence)
        elif entry["kind"] == "factor-trial":
            row = _factor(path, evidence)
        else:
            row = _imc(path, evidence)
        evidence.unchanged()
        return row | {
            "id": entry["id"],
            "kind": entry["kind"],
            "evidence": evidence.files,
        }
    except ImportError as exc:
        return {
            "id": entry["id"],
            "kind": entry["kind"],
            "result_state": "UNAVAILABLE",
            "verification": "NOT_RUN",
            "reason": "required optional package unavailable: "
            + (exc.name or "unknown"),
            "evidence": evidence.files,
        }
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        return {
            "id": entry["id"],
            "kind": entry["kind"],
            "result_state": "REFUSED",
            "verification": "FAILED",
            "reason": type(exc).__name__ + ": " + str(exc),
            "evidence": evidence.files,
        }
