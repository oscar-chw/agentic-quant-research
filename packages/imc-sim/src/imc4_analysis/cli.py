"""Offline entry points with refusal to overwrite and a completion receipt."""
import argparse
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import sys

from . import __version__
from .analyzer import analyze
from .community_import import normalize_community_log
from .contracts import InputError, MAX_BYTES
from .report import render
from .quote_study import MAX_STUDY_BYTES, run_study


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write_exclusive(path, content):
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_report(report, output, import_artifacts=None, *, _before_write=None, _after_write=None):
    """A missing complete.json means incomplete; an existing directory is never reused."""
    numerical = json_bytes(report)
    visual = render(report).encode("utf-8")
    payloads = {"report.json": numerical, "report.html": visual}
    if import_artifacts is not None:
        if set(import_artifacts) != {"raw-source.log", "import-config.json", "normalized.jsonl", "import-receipt.json"}:
            raise InputError("import bundle must preserve raw source, config, normalized input and receipt")
        provenance = report["import_provenance"]
        for name, field in [("raw-source.log", "raw_source_sha256"), ("import-config.json", "config_sha256"), ("normalized.jsonl", "normalized_sha256")]:
            if hashlib.sha256(import_artifacts[name]).hexdigest() != provenance[field]:
                raise InputError(f"import artifact commitment mismatch: {name}")
        if provenance["normalized_sha256"] != report["source_sha256"]:
            raise InputError("normalized import is not the analyzed input")
        payloads.update(import_artifacts)
    receipt = {"schema": "imc4-analysis-completion/v1", "tool_version": __version__,
               "code_identity": report["code_identity"], "runtime": report["runtime"],
               "source_sha256": report["source_sha256"],
               "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                         for name, content in payloads.items()}}
    if import_artifacts is not None:
        receipt["import_provenance"] = report["import_provenance"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in list(payloads.items()) + [("complete.json", json_bytes(receipt))]:
        if _before_write is not None: _before_write(output / name, content)
        _write_exclusive(output / name, content)
        if _after_write is not None: _after_write(output / name)
    # Receipt is last: an interrupted run never produces a valid completed bundle.
    return receipt


def write_study(raw, output):
    """Complete finite computation before exclusive output creation; receipt last."""
    comparison, runs = run_study(raw)
    return write_study_results(raw, comparison, runs, output)


def study_markdown_bytes(comparison):
    lines = ["# Controlled quote-policy study", "", comparison["assumptions"], "",
             "Inventory square time integrates post-fill inventory squared over each one-tick interval. Terminal inventory is marked, not forcibly liquidated.", "",
             "| Scenario | Execution | Policy | P&L | Fees | Terminal inventory | Peak absolute inventory | Inventory square time | Fill events | Pending-cancel filled units | Report |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in comparison["rows"]:
        name, summary, metrics = row["run"], row["summary"], row["metrics"]
        lines.append(f"| {row['scenario']} | {row['execution']['id']} | {row['policy']} | {summary['pnl']} | {summary['total_fees']} | {summary['positions']['ASSET']} | {metrics['peak_absolute_inventory']} | {metrics['inventory_square_time']} | {summary['fill_count']} | {metrics['pending_cancel_fill_units']} | [Inspect]({name}/report.html) |")
    lines.extend(["", "Inspect each report's equity, inventory and fill-quality tables alongside its trace.json. Declared current marks drive decisions; future marks appear only in the analyzer's retrospective markouts.", "",
                  "No aggregate winner is selected. Queue and cancellation variants change one execution assumption at a time; these deterministic scenarios do not estimate market uncertainty or alpha."])
    return ("\n".join(lines)+"\n").encode()


def write_study_results(raw, comparison, runs, output, *, _on_create=None, _before_write=None, _after_write=None):
    """Publish supplied native results only; exclusive creation, root receipt last.

    The caller supplies results bound to raw. This function performs no study,
    transition, policy, matching or analysis computation and no partial recovery.
    """
    def write(path, content):
        if _before_write is not None: _before_write(path, content)
        _write_exclusive(path, content)
        if _after_write is not None: _after_write(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if _on_create is not None: _on_create(output)
    write(output / "scenario.json", raw)
    write(output / "comparison.json", json_bytes(comparison))
    for row in comparison["rows"]:
        name, summary, metrics = row["run"], row["summary"], row["metrics"]
        directory = output / name
        write_report(runs[name]["report"], directory, _before_write=_before_write, _after_write=_after_write)
        write(directory / "normalized.jsonl", runs[name]["normalized"])
        write(directory / "trace.json", json_bytes(dict(policy=runs[name]["policy"],
                         policy_settings=runs[name]["policy_settings"], exogenous_sha256=row["exogenous_sha256"],
                         normalized_sha256=row["normalized_sha256"], steps=runs[name]["trace"])))
    write(output / "comparison.md", study_markdown_bytes(comparison))
    # Root receipt also commits normalized/trace files and each nested report receipt.
    receipt = dict(schema="imc4-quote-study-completion/v1", tool_version=__version__,
                   code_identity=comparison["code_identity"], runtime=comparison["runtime"],
                   source_sha256=comparison["source_sha256"], files={})
    for path in sorted(output.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            receipt["files"][str(path.relative_to(output))] = dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    write(output / "complete.json", json_bytes(receipt))
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline normalized run analyzer; new post-competition tooling.")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "analyze", "import-log"):
        command = subcommands.add_parser(name)
        if name in ("analyze", "import-log"):
            command.add_argument("input", type=Path)
        if name == "import-log":
            command.add_argument("--config", type=Path, required=True, help="explicit format, identity, timing, fees and opening state")
        command.add_argument("--out", type=Path, required=True, help="new directory; existing paths refused")
        command.add_argument("--detail", choices=("sampled", "full"), default="sampled")
        command.add_argument("--sample-limit", type=int, default=200, help="10–1000 rows per kind; full mode ignores sampling")
    study_command = subcommands.add_parser("quote-study", help="bounded synthetic inventory-policy and execution comparison")
    study_command.add_argument("input", type=Path, nargs="?", help="explicit synthetic scenario JSON; defaults to the packaged frozen example")
    study_command.add_argument("--out", type=Path, required=True)
    for operation in ("persist-run", "checkpoint", "resume", "verify-attempt"):
        command = subcommands.add_parser(operation, help="bounded local persisted simulation")
        if operation == "persist-run": command.add_argument("input", type=Path, nargs="?")
        command.add_argument("--store", type=Path, required=True)
        command.add_argument("--attempt", required=True)
        command.add_argument("--context-sha256")
        if operation in ("persist-run", "resume"): command.add_argument("--stop-after", type=int)
        if operation == "resume": command.add_argument("--token", type=Path, required=True)
        if operation == "verify-attempt": command.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    if args.command in ("persist-run", "checkpoint", "resume", "verify-attempt"):
        from . import persistence
        try:
            options = dict(context_sha256=args.context_sha256)
            if args.command == "persist-run":
                if args.input is None:
                    raw = resources.files("imc4_analysis").joinpath("data/quoting_scenarios.json").read_bytes()
                else:
                    with args.input.open("rb") as f: raw = f.read(MAX_STUDY_BYTES + 1)
                result = persistence.run(raw, args.store, args.attempt, stop_after_commits=args.stop_after, **options)
            elif args.command == "checkpoint":
                result = persistence.checkpoint(args.store, args.attempt, **options)
            elif args.command == "resume":
                with args.token.open("rb") as f: raw_token = f.read(32769)
                if len(raw_token) > 32768: raise InputError("token file exceeds32KiB")
                from .contracts import no_duplicate_keys
                token = json.loads(raw_token, object_pairs_hook=no_duplicate_keys)
                if isinstance(token, dict) and "token" in token: token = token["token"]
                result = persistence.resume(args.store, args.attempt, token, stop_after_commits=args.stop_after, **options)
            else:
                result = persistence.verify(args.store, args.attempt, mode="recompute" if args.recompute else "state", **options)
            print(json.dumps(result, sort_keys=True))
            return 0
        except (InputError, OSError, ValueError) as exc:
            print(json.dumps(dict(status="REFUSED", code=getattr(exc,"code","INVALID_INPUT"), message=str(exc)), sort_keys=True))
            return 2

    try:
        if args.command == "quote-study":
            if args.input is None:
                data = resources.files("imc4_analysis").joinpath("data/quoting_scenarios.json").read_bytes()
            else:
                with args.input.open("rb") as handle:
                    data = handle.read(MAX_STUDY_BYTES + 1)
            write_study(data, args.out)
            print(f"Study complete: {args.out / 'comparison.md'}")
            return 0
        if args.command == "demo":
            data = resources.files("imc4_analysis").joinpath("data/synthetic.jsonl").read_bytes()
        else:
            with args.input.open("rb") as handle:
                data = handle.read(MAX_BYTES + 1)
        artifacts = None
        provenance = None
        if args.command == "import-log":
            raw = data
            with args.config.open("rb") as handle:
                configuration = handle.read(65537)
            data, provenance = normalize_community_log(raw, configuration)
        report = analyze(data, detail=args.detail, sample_limit=args.sample_limit)
        if provenance is not None:
            report["import_provenance"] = provenance
            artifacts = {"raw-source.log": raw, "import-config.json": configuration, "normalized.jsonl": data,
                         "import-receipt.json": json_bytes(dict(provenance, code_identity=report["code_identity"], runtime=report["runtime"], tool_version=__version__))}
        write_report(report, args.out, artifacts)
    except (InputError, OSError) as exc:
        print(f"imc4-analyze: {exc}", file=sys.stderr)
        return 2
    print(f"Report complete: {args.out / 'report.html'}")
    return 0
