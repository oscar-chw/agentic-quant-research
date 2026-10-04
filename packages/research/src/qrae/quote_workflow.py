"""Compose quote features and fixed experiments using the existing immutable store."""

from __future__ import annotations

from pathlib import Path

from . import quote_features
from .artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    sha256_bytes,
    verify_manifest,
)
from .path_safety import has_link_or_reparse_component
from .quote_features import (
    evaluate,
    feature_spec,
    materialize,
    read_quotes,
    strict_json,
)

METHOD_URL = "https://www.ma.imperial.ac.uk/~ajacquie/Gatheral60/Slides/Gatheral60%20-%20Stoikov.pdf"
FEATURE_FILES = {"quotes.csv", "feature-spec.json", "features.jsonl"}
RUN_FILES = {
    "experiment-spec.json",
    "features-ref.json",
    "report.json",
    "evaluations.jsonl",
    "REPORT.md",
}
MAX_FEATURE_BYTES = 64 * 1024 * 1024


def code_identity() -> str:
    package = Path(__file__).parent
    names = (
        "quote_features.py",
        "quote_states.py",
        "quote_workflow.py",
        "artifacts.py",
        "contracts.py",
        "path_safety.py",
        "__init__.py",
    )
    return sha256_bytes(
        canonical_json_bytes(
            {n: sha256_bytes((package / n).read_bytes()) for n in names}
        )
    )


def read_input(path: Path, cap: int) -> bytes:
    if has_link_or_reparse_component(path) or not path.is_file():
        raise ValueError("regular input without link components required")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("input exceeds declared byte cap")
    return raw


def jsonl(rows: list[dict]) -> bytes:
    raw = b"".join(canonical_json_bytes(row) for row in rows)
    if len(raw) > MAX_FEATURE_BYTES:
        raise ValueError("materialized output exceeds 64 MiB")
    return raw


def checked_bundle(folder: Path, members: set[str], kind: str) -> dict:
    if has_link_or_reparse_component(folder) or any(
        has_link_or_reparse_component(p) for p in folder.iterdir()
    ):
        raise ValueError("linked bundle refused")
    if not (folder / "manifest.json").is_file():
        raise ValueError(
            "INCOMPLETE bundle: no final manifest; preserve and use a new store"
        )
    if {p.name for p in folder.iterdir()} != members | {"manifest.json"}:
        raise ValueError("unexpected bundle contents")
    manifest = verify_manifest(folder)
    if manifest["metadata"].get("kind") != kind:
        raise ValueError("bundle kind mismatch")
    return manifest


def reserve(root: Path, run_id: str):
    if has_link_or_reparse_component(root):
        raise ValueError("linked store refused")
    parent = root / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    if has_link_or_reparse_component(parent):
        raise ValueError("linked store refused")
    folder = parent / run_id
    try:
        folder.mkdir()
    except FileExistsError:
        return None
    return ArtifactStore(root, run_id)


def save_features(raw: bytes, spec: dict, root: Path) -> tuple[dict, list[dict], bool]:
    fspec = feature_spec(spec)
    identity = {
        "source_sha256": sha256_bytes(raw),
        "spec_sha256": sha256_bytes(canonical_json_bytes(fspec)),
        "code_sha256": code_identity(),
    }
    key = "features-" + sha256_bytes(canonical_json_bytes(identity))
    folder = root / "features/runs" / key
    if folder.exists():
        manifest = checked_bundle(folder, FEATURE_FILES, "quote-features/v1")
        if manifest["metadata"]["identity"] != identity:
            raise ValueError("feature cache identity mismatch")
        rows = [
            strict_json(line)
            for line in read_input(
                folder / "features.jsonl", MAX_FEATURE_BYTES
            ).splitlines()
        ]
        reused = True
    else:
        rows = materialize(read_quotes(raw, fspec), fspec)
        payload = jsonl(rows)
        store = reserve(root / "features", key)
        if store is None:
            # A concurrent writer is not implicitly complete.
            return save_features(raw, spec, root)
        records = [
            store.write_bytes("quotes.csv", raw),
            store.write_json("feature-spec.json", fspec),
            store.write_bytes("features.jsonl", payload),
        ]
        store.commit_manifest(
            records, {"kind": "quote-features/v1", "identity": identity}
        )
        reused = False
    return (
        {
            "feature_id": key,
            "manifest_sha256": sha256_bytes((folder / "manifest.json").read_bytes()),
        },
        rows,
        reused,
    )


def human_report(report: dict, reference: dict) -> str:
    lines = [
        "# Availability-aware quote forecast",
        "",
        "A fixed weighted-midpoint forecast is compared with the current midpoint for the next decision-time quote.",
        "E0: supplied clocks are unaudited; quote-prediction errors are not trading returns.",
        "",
        "| Split | Evaluated / possible pairs | Midpoint MAE (ticks) | Weighted midpoint MAE (ticks) |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["splits"].items():
        lines.append(
            f"| {name} | {row['evaluated_pairs']} / {row['candidate_pairs']} | {row['models']['midpoint']['mae_ticks']} | {row['models']['weighted_midpoint']['mae_ticks']} |"
        )
    lines += [
        "",
        "Feature materialization: `" + reference["feature_id"] + "`.",
        "Price per tick: "
        + report["price_tick"]
        + ". Rational values such as 1/3 are exact.",
        "Rows omitted for missing/stale quotes are counted in report.json and evaluations.jsonl.",
        "Each split excludes its outgoing boundary pair. No fitting, tuning or winner selection occurs.",
        "",
        "[Weighted-midpoint reference]("
        + METHOD_URL
        + "). This uses the simple weighted midpoint, not the calibrated micro-price.",
        "Freshness checks do not prove feed connectivity. Only complete top-of-book snapshots are accepted; no delta recovery or exchange sequence-gap detection is claimed.",
        "Single instrument; bounded in-memory sort/sweep. No live execution, autonomous model calls, transaction-cost backtest or historical authenticity verification.",
    ]
    if report["schema"] == "quote-forecast-report/v2":
        lines[-2] = (
            "Collector-ordered quote states carry explicit gaps and continuity segments. Pairs crossing a declared break are excluded even if the break falls between decision times. Collector record IDs are not exchange sequence numbers; unreported loss and source-clock authenticity remain unverified."
        )
        lines += [
            "",
            "Raw archive SHA256: `" + report["source_sha256"] + "`.",
            "Collector adapter SHA256: `" + report["adapter_sha256"] + "`.",
        ]
    return "\n".join(lines) + "\n"


def run_quotes(quotes: Path, specification: Path, root: Path) -> dict:
    if has_link_or_reparse_component(root):
        raise ValueError("linked store refused")
    raw = read_input(quotes, quote_features.MAX_INPUT)
    spec_raw = read_input(specification, 64 * 1024)
    spec = strict_json(spec_raw)
    feature_spec(spec)  # Validate before creating any output.
    reference, features, reused = save_features(raw, spec, root)
    identity = {
        "features": reference,
        "spec_sha256": sha256_bytes(spec_raw),
        "code_sha256": code_identity(),
    }
    key = "experiment-" + sha256_bytes(canonical_json_bytes(identity))
    folder = root / "experiments/runs" / key
    if folder.exists():
        result = verify_quote_run(root, key)
        return result | {"features_reused": reused, "experiment_reused": True}
    report, pairs = evaluate(features, spec)
    # Include all evaluations, even excluded ones; both models are always reported.
    payload = jsonl(pairs)
    store = reserve(root / "experiments", key)
    if store is None:
        result = verify_quote_run(root, key)
        return result | {"features_reused": reused, "experiment_reused": True}
    records = [
        store.write_bytes("experiment-spec.json", spec_raw),
        store.write_json("features-ref.json", reference),
        store.write_json("report.json", report),
        store.write_bytes("evaluations.jsonl", payload),
        store.write_text("REPORT.md", human_report(report, reference)),
    ]
    store.commit_manifest(
        records, {"kind": "quote-experiment/v1", "identity": identity}
    )
    return verify_quote_run(root, key) | {
        "features_reused": reused,
        "experiment_reused": False,
    }


def verify_quote_run(root: Path, run_id: str, recompute: bool = False) -> dict:
    import re

    if not re.fullmatch(
        r"experiment-[0-9a-f]{64}", run_id
    ) or has_link_or_reparse_component(root):
        raise ValueError("safe content-addressed run and unlinked store required")
    folder = root / "experiments/runs" / run_id
    manifest = checked_bundle(folder, RUN_FILES, "quote-experiment/v1")
    identity = manifest["metadata"]["identity"]
    if run_id != "experiment-" + sha256_bytes(canonical_json_bytes(identity)):
        raise ValueError("experiment ID mismatch")
    reference = strict_json((folder / "features-ref.json").read_bytes())
    if reference != identity["features"] or not re.fullmatch(
        r"features-[0-9a-f]{64}", reference["feature_id"]
    ):
        raise ValueError("feature reference identity mismatch")
    feature_folder = root / "features/runs" / reference["feature_id"]
    fm = checked_bundle(feature_folder, FEATURE_FILES, "quote-features/v1")
    if (
        sha256_bytes((feature_folder / "manifest.json").read_bytes())
        != reference["manifest_sha256"]
    ):
        raise ValueError("feature manifest identity mismatch")
    fi = fm["metadata"]["identity"]
    spec_raw = (folder / "experiment-spec.json").read_bytes()
    spec = strict_json(spec_raw)
    expected = feature_spec(spec)
    fraw = (feature_folder / "feature-spec.json").read_bytes()
    raw = (feature_folder / "quotes.csv").read_bytes()
    if (
        identity["spec_sha256"] != sha256_bytes(spec_raw)
        or fi["source_sha256"] != sha256_bytes(raw)
        or fi["spec_sha256"] != sha256_bytes(fraw)
        or strict_json(fraw) != expected
        or fi["code_sha256"] != identity["code_sha256"]
        or reference["feature_id"]
        != "features-" + sha256_bytes(canonical_json_bytes(fi))
    ):
        raise ValueError("source/spec/code identity mismatch")
    if recompute:
        if code_identity() != identity["code_sha256"]:
            raise ValueError("recompute requires original code")
        features = materialize(read_quotes(raw, expected), expected)
        report, pairs = evaluate(features, spec)
        if (
            jsonl(features) != (feature_folder / "features.jsonl").read_bytes()
            or canonical_json_bytes(report) != (folder / "report.json").read_bytes()
            or jsonl(pairs) != (folder / "evaluations.jsonl").read_bytes()
            or human_report(report, reference).encode()
            != (folder / "REPORT.md").read_bytes()
        ):
            raise ValueError("recomputed artifacts differ")
    return {
        "status": "VERIFIED",
        "run_id": run_id,
        "directory": str(folder),
        "feature_id": reference["feature_id"],
        "recomputed": recompute,
        "evidence_tier": "E0",
    }
