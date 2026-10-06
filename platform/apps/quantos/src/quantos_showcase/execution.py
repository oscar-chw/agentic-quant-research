"""Finite IMC quote studies with source registration and immutable attempts."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
from pathlib import Path

import source_access
from qrae import artifacts, path_safety, quote_workflow
from qrae.artifacts import ArtifactRecord, canonical_json_bytes, sha256_bytes
from qrae.quote_features import strict_json
from qrae.quote_workflow import checked_bundle, read_input, reserve

from quantos_showcase import run_adapters
from quantos_showcase.run_adapters import Evidence

KIND = "quantos-execution/v1"
METHOD = "imc4-inventory-quote-study/v1"
IMC_VERSION = "0.4.0"
IMC_CODE = "536964bdd251c77576ed6953bd8befabc843b05813e8e205e5c11bd387abfcd4"
MAX_INPUT = 256 * 1024
MEMBERS = {
    "request.json",
    "scenario.json",
    "source.md",
    "registration.json",
    "REPORT.md",
    "native",
}


def native():
    """Import only the fixed, accepted optional implementation; no live clients."""
    from imc4_analysis import __version__, cli, provenance, quote_study

    if __version__ != IMC_VERSION or provenance.code_identity()["sha256"] != IMC_CODE:
        raise ValueError(
            "execution requires accepted repaired imc4-analysis 0.4.0 code; older attempts require their original pinned environment"
        )
    return cli, quote_study


def implementation_identity():
    modules = {
        "execution": Path(__file__),
        "source_access": Path(source_access.__file__),
        "artifacts": Path(artifacts.__file__),
        "path_safety": Path(path_safety.__file__),
        "quote_workflow": Path(quote_workflow.__file__),
        "run_adapters": Path(run_adapters.__file__),
    }
    return sha256_bytes(
        canonical_json_bytes(
            {k: sha256_bytes(p.read_bytes()) for k, p in modules.items()}
        )
    )


def attempt_folder(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id
    ):
        raise ValueError("run ID must be 1..64 letters, digits, underscores or hyphens")
    folder = Path(root) / "runs" / run_id
    if path_safety.has_link_or_reparse_component(folder):
        raise ValueError("unlinked attempt store required")
    return folder.absolute()


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {
        "schema",
        "method",
        "question",
        "source",
    }:
        raise ValueError("exact execution request fields required")
    if (
        request["schema"] != "quantos-execution-request/v1"
        or request["method"] != METHOD
    ):
        raise ValueError("fixed IMC quote-study method required")
    if (
        not isinstance(request["question"], str)
        or not 1 <= len(request["question"].strip()) <= 4000
    ):
        raise ValueError("bounded research question required")
    source = request["source"]
    if not isinstance(source, dict) or set(source) != {"id", "path", "sha256"}:
        raise ValueError("explicit source id, path and digest required")
    if not isinstance(source["id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", source["id"]
    ):
        raise ValueError("safe source ID required")
    name = source["path"]
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= 1024
        or "\\" in name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        or any(ord(c) < 32 for c in name)
    ):
        raise ValueError("source path must stay within the request directory")
    digest = source["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("explicit source digest required")
    return request


def read_source(request, base, *, frozen=False):
    path = base / ("source.md" if frozen else request["source"]["path"])
    raw = read_input(path, MAX_INPUT)
    if not raw.decode("utf-8").strip():
        raise ValueError("nonempty UTF-8 source note required")
    decision = source_access.inspect_note(
        {
            "document_type": "paper_note",
            "arxiv_id": request["source"]["id"],
            "source_available": True,
            "source_status": "present",
            "vault_path": str(path),
            "source_sha256": request["source"]["sha256"],
        },
        raw.decode("utf-8"),
        source_root=base,
    )
    if not decision["eligible"]:
        raise ValueError("source note refused: " + decision["reason"])
    return raw


def admit(raw):
    _, study_module = native()
    study = study_module.load_study(raw)
    runs = 2 * len(study["scenarios"]) * len(study["executions"])
    steps = (
        2 * len(study["executions"]) * sum(len(s["steps"]) for s in study["scenarios"])
    )
    if runs > 18 or steps > 1024:
        raise ValueError("run cap: at most 18 native runs and 1024 total steps")
    return {"runs": runs, "steps": steps}


def registration(request_raw, scenario_raw, note):
    native()
    return {
        "schema": "quantos-execution-registration/v1",
        "method": METHOD,
        "wrapper_sha256": implementation_identity(),
        "native_version": IMC_VERSION,
        "native_code_sha256": IMC_CODE,
        "request_sha256": sha256_bytes(request_raw),
        "scenario_sha256": sha256_bytes(scenario_raw),
        "source_sha256": sha256_bytes(note),
        "lifecycle": "finite-new-attempt-only/v1",
        "scope": "Local snapshots before native execution; no external preregistration or source-authenticity proof",
    }


def payloads(folder, evidence):
    """Bound traversal before following any bundle contents."""
    if path_safety.has_link_or_reparse_component(folder) or not folder.is_dir():
        raise ValueError("unlinked bundle directory required")
    result, nodes = {}, 0
    for parent, directories, files in os.walk(folder, followlinks=False):
        for name in sorted(directories + files):
            path = Path(parent) / name
            nodes += 1
            if nodes > 256 or path_safety.has_link_or_reparse_component(path):
                raise ValueError("bundle link or 256-node cap exceeded")
        for name in sorted(files):
            path = Path(parent) / name
            result[path.relative_to(folder).as_posix()] = evidence.bind(
                path, "native execution artifact", content=True
            )
    return result


def render(request, comparison, *, product="native", persisted=False):
    lines = [
        "# Registered native quote execution",
        "",
        "Execution: **COMPLETED / SYNTHETIC**.",
        "",
        "<p>" + html.escape(request["question"]) + "</p>",
        "",
        "[Frozen request](request.json) · [Source note](source.md) · [Scenario and policy settings](scenario.json) · [Registration](registration.json)",
        "",
        f"[Native comparison]({product}/comparison.md) · [Machine-readable comparison]({product}/comparison.json)",
        "",
        "Both policies use the native IMC implementation. The analyzer owns cash, fees and marked P&L; no results are pooled with other applications.",
        "",
        "| Scenario / execution | Policy | Marked P&L | Terminal position | Decisions, orders, fills and inventory | Native report |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in comparison["rows"]:
        name = row["run"]
        lines.append(
            f"| {row['scenario']} / {row['execution']['id']} | {row['policy']} | {row['summary']['pnl']} | {row['summary']['positions']['ASSET']} | [Trace]({product}/{name}/trace.json) · [Events]({product}/{name}/normalized.jsonl) | [Inspect]({product}/{name}/report.html) |"
        )
    lines += [
        "",
        "## Lifecycle and interpretation",
        "",
        "One finite attempt starts from the declared initial cash and inventory. Orders expire at the terminal tick and remaining inventory is marked, not liquidated.",
        "A final QuantOS manifest certifies completed output. Any interruption before that manifest leaves an incomplete attempt, including when native reports already exist. Preserve it and use a new run ID to restart from the same inputs; no fills or state are resumed or appended.",
        "Verification reruns the native writer in temporary storage and compares every payload byte. Saved attempts are never modified. Exact installed native code and runtime output are required; verification across different Python/runtime metadata is not qualified.",
        "No persistent service, market-performance conclusion, official matching parity or live execution is established. A local note digest establishes context identity, not original-source truth or publication rights.",
        "",
        comparison["assumptions"],
        "",
    ]
    if persisted:
        lines = [
            line.replace(
                "A final QuantOS manifest certifies completed output. Any interruption before that manifest leaves an incomplete attempt, including when native reports already exist. Preserve it and use a new run ID to restart from the same inputs; no fills or state are resumed or appended.",
                "Native committed checkpoints resume only remaining economic steps. Native COMPLETE and QuantOS completion are separate: a missing wrapper report/receipt can be finalized after native state verification with zero simulation. Exact complete report-only output can be reconciled; partial or different wrapper files are preserved/refused.",
            ).replace(
                "Verification reruns the native writer in temporary storage and compares every payload byte. Saved attempts are never modified. Exact installed native code and runtime output are required; verification across different Python/runtime metadata is not qualified.",
                "Default verification audits native committed state and product commitments without replaying policy. Explicit recompute delegates to native verification. The native operational state is path/identity bound; do not relocate it or infer power-loss recovery from process-exit tests.",
            )
            for line in lines
        ]
    return "\n".join(lines)


def run_execution(request_path, scenario_path, root, run_id):
    folder = attempt_folder(root, run_id)
    if folder.exists():
        raise FileExistsError(
            "attempt exists; verify or restart with a new ID; never resume in place"
        )
    request_raw = read_input(request_path, 64 * 1024)
    request = validate_request(strict_json(request_raw))
    note = read_source(request, request_path.absolute().parent)
    raw = read_input(scenario_path, MAX_INPUT)
    admit(raw)
    record = registration(request_raw, raw, note)
    store = reserve(Path(root), run_id)
    if store is None:
        raise FileExistsError("concurrent attempt writer; prior output preserved")
    records = [
        store.write_bytes("request.json", request_raw),
        store.write_bytes("scenario.json", raw),
        store.write_bytes("source.md", note),
        store.write_json("registration.json", record),
    ]
    # Do not catch interruptions or mark partial native output as completed.
    cli, _ = native()
    comparison = cli.write_study(raw, folder / "native")
    evidence = Evidence()
    for name, content in payloads(folder / "native", evidence).items():
        records.append(
            ArtifactRecord("native/" + name, sha256_bytes(content), len(content))
        )
    records.append(store.write_text("REPORT.md", render(request, comparison)))
    store.commit_manifest(
        records,
        {
            "kind": KIND,
            "registration_sha256": sha256_bytes(canonical_json_bytes(record)),
        },
    )
    return verify_execution(Path(root), run_id)


def verify_execution(root, run_id, *, evidence=None):
    folder = attempt_folder(root, run_id)
    if not (folder / "manifest.json").is_file():
        raise ValueError(
            "INCOMPLETE attempt: no final QuantOS manifest; preserve and use a new ID"
        )
    evidence = evidence if evidence is not None else Evidence()
    all_files = payloads(folder, evidence)
    manifest = checked_bundle(folder, MEMBERS, KIND)
    request_raw, raw, note = (
        all_files[n] for n in ("request.json", "scenario.json", "source.md")
    )
    request = validate_request(strict_json(request_raw))
    if read_source(request, folder, frozen=True) != note:
        raise ValueError("source changed during verification")
    counts = admit(raw)
    record = registration(request_raw, raw, note)
    if (
        manifest["run_id"] != run_id
        or canonical_json_bytes(record) != all_files["registration.json"]
        or manifest["metadata"]
        != {
            "kind": KIND,
            "registration_sha256": sha256_bytes(all_files["registration.json"]),
        }
    ):
        raise ValueError(
            "registration/input/code identity mismatch; older attempts require their original pinned environment"
        )
    cli, _ = native()
    with tempfile.TemporaryDirectory(prefix="quantos-native-check-") as temp:
        expected_path = Path(temp).resolve() / "native"
        comparison = cli.write_study(raw, expected_path)
        expected = payloads(expected_path, Evidence())
        actual = {
            n.removeprefix("native/"): b
            for n, b in all_files.items()
            if n.startswith("native/")
        }
        if actual != expected:
            raise ValueError(
                "native recomputation differs from saved execution payloads"
            )
    if render(request, comparison).encode() != all_files["REPORT.md"]:
        raise ValueError("execution report differs")
    evidence.unchanged()
    return {
        "execution": "COMPLETED",
        "verification": "NATIVE_PAYLOAD_RECOMPUTED",
        "method": METHOD,
        "run_id": run_id,
        "native_version": IMC_VERSION,
        "data_kind": "synthetic",
        "counts": counts,
        "rows": comparison["rows"],
        "directory": str(folder),
        "report": str(folder / "REPORT.md"),
        "lifecycle": record["lifecycle"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--store", type=Path, required=True)
        sub.add_argument("--run-id", required=True)
        if name == "run":
            sub.add_argument("--request", type=Path, required=True)
            sub.add_argument("--scenario", type=Path, required=True)
    for name in ("persist-run", "checkpoint", "resume", "verify-persisted"):
        sub = commands.add_parser(name)
        sub.add_argument("--store", type=Path, required=True)
        sub.add_argument("--run-id", required=True)
        if name == "persist-run":
            sub.add_argument("--request", type=Path, required=True)
            sub.add_argument("--scenario", type=Path, required=True)
        if name in ("persist-run", "resume"):
            sub.add_argument("--stop-after", type=int)
        if name == "resume":
            sub.add_argument("--token", type=Path, required=True)
        if name == "verify-persisted":
            sub.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command not in ("run", "verify"):
            from quantos_showcase import persisted

            if args.command == "persist-run":
                result = persisted.run(
                    args.request,
                    args.scenario,
                    args.store,
                    args.run_id,
                    stop_after_commits=args.stop_after,
                )
            elif args.command == "checkpoint":
                result = persisted.checkpoint(args.store, args.run_id)
            elif args.command == "resume":
                result = persisted.resume(
                    args.store,
                    args.run_id,
                    strict_json(read_input(args.token, 65536)),
                    stop_after_commits=args.stop_after,
                )
            else:
                result = persisted.verify(
                    args.store,
                    args.run_id,
                    mode="recompute" if args.recompute else "state",
                )
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0

        result = (
            run_execution(args.request, args.scenario, args.store, args.run_id)
            if args.command == "run"
            else verify_execution(args.store, args.run_id)
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except ImportError as exc:
        print(
            json.dumps(
                {
                    "execution": "UNAVAILABLE",
                    "error": "required optional package: " + (exc.name or "unknown"),
                }
            )
        )
        return 2
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(
            json.dumps(
                {
                    "execution": "REFUSED",
                    "code": getattr(exc, "code", "REFUSED"),
                    "error": str(exc),
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
