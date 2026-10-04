"""Freeze source-backed hypotheses before invoking the existing quote workflow."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import method_contract
import source_access
from replay.admission import MAX_INPUT, adapter_identity
from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.quote_features import strict_json
from qrae.quote_workflow import checked_bundle, code_identity, read_input, reserve

from quantos_showcase import feed

KIND = "quantos-method-trial/v1"
FILES = {
    "card.json",
    "archive.jsonl",
    "spec.json",
    "registration.json",
    "outcome.json",
    "REPORT.md",
}


def implementation_identity():
    modules = (
        Path(__file__),
        Path(method_contract.__file__),
        Path(source_access.__file__),
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "modules": {p.name: sha256_bytes(p.read_bytes()) for p in modules},
                "feed": feed.application_identity(),
                "adapter": adapter_identity(),
                "research": code_identity(),
            }
        )
    )


def registration(card_raw, raw, spec_raw, sources):
    return {
        "schema": "quantos-method-registration/v1",
        "code_sha256": implementation_identity(),
        "card_sha256": sha256_bytes(card_raw),
        "archive_sha256": sha256_bytes(raw),
        "spec_sha256": sha256_bytes(spec_raw),
        "sources": {n: sha256_bytes(b) for n, b in sources.items()},
        "scope": "Local snapshots written before native invocation; no external timestamp or prior-inspection proof",
        "source_authenticity_verified": False,
        "publication_rights_verified": False,
    }


def native_outcome(card, research_root, result, *, recompute=True):
    verified = feed.verify_feed(research_root, result["run_id"], recompute)
    report = strict_json(
        read_input(Path(verified["report"]).with_name("report.json"), 128 * 1024)
    )
    return {
        "schema": KIND,
        "execution": "COMPLETED",
        "native": {k: verified[k] for k in ("run_id", "research_run_id", "feature_id")},
        "decision": method_contract.evaluate_card(card, report),
        "data_qualification": {
            key: report[key]
            for key in (
                "data_kind",
                "availability",
                "historically_point_in_time_verified",
                "profitability_evaluated",
                "target",
            )
        },
    }


def render(card, outcome):
    # Escape authored prose rather than letting a source card inject HTML/links.
    esc = html.escape
    lines = [
        "# Source-backed quote research",
        "",
        "Execution: **" + outcome["execution"] + "**.",
        "",
        "Method: `" + card["method"] + "`. Card: [frozen contract](card.json).",
        "",
        "## Hypothesis and mechanism",
        "",
        "<p>" + esc(card["hypothesis"]) + "</p>",
        "",
        "<p>" + esc(card["mechanism"]) + "</p>",
        "",
        "## Assumptions and limits",
        "",
    ]
    lines += [
        "<p>" + esc(x) + "</p>" for x in card["assumptions"] + card["limitations"]
    ]
    lines += ["", "## Declared sources", ""]
    for i, source in enumerate(card["sources"]):
        lines += [
            f"- [{source['id']}](source-{i}.md): {source['kind']}, {source['relationship']}; `{source['sha256']}`."
        ]
    lines += [
        "",
        "These are checked local notes. Links and hashes do not establish original source claims or publication rights.",
        "",
        "## Result",
        "",
        "```json",
        json.dumps(outcome, indent=2, sort_keys=True),
        "```",
        "",
    ]
    if outcome["execution"] == "COMPLETED":
        native = outcome["native"]
        lines += [
            "[Native forecast report](../../../research/experiments/runs/"
            + native["research_run_id"]
            + "/REPORT.md)",
            "",
        ]
    lines += [
        "Strict improvement is required; equality does not support the rule. Low pair count or coverage is INSUFFICIENT.",
        "The rule is descriptive, without an uncertainty estimate or multiple-testing correction. It is not an alpha acceptance gate.",
        "Local before-execution registration does not establish an untouched holdout. The shipped example uses previously inspected synthetic data.",
        "No fees, execution model, trading returns, paper-source authenticity or external preregistration are qualified.",
        "",
    ]
    return "\n".join(lines)


def trial_folder(root, trial_id):
    method_contract.safe_id(trial_id)
    return root / "methods/runs" / trial_id


def run_method(card_path, archive, specification, root, trial_id):
    folder = trial_folder(root, trial_id)
    if folder.exists():
        raise FileExistsError(
            "trial exists; verify or use a new ID; prior attempts are retained"
        )
    card_raw = read_input(card_path, 64 * 1024)
    card = method_contract.validate_card(strict_json(card_raw))
    if card["method"] != method_contract.QUOTE_METHOD:
        raise ValueError("quantos-method runs quote cards; factor cards run in quantos-loop")
    sources = method_contract.read_sources(card, card_path.absolute().parent)
    raw, spec_raw = read_input(archive, MAX_INPUT), read_input(specification, 64 * 1024)
    feed.prepare(
        raw, spec_raw
    )  # Refuse invalid input before registration; no experiment execution.
    record = registration(card_raw, raw, spec_raw, sources)
    store = reserve(root / "methods", trial_id)
    if store is None:
        raise FileExistsError("concurrent trial writer; output preserved")
    records = [
        store.write_bytes("card.json", card_raw),
        store.write_bytes("archive.jsonl", raw),
        store.write_bytes("spec.json", spec_raw),
    ]
    records += [store.write_bytes(name, data) for name, data in sources.items()]
    records.append(store.write_json("registration.json", record))
    # Registration and snapshots exist before the numerical workflow is invoked.
    try:
        result = feed.run_feed(
            folder / "archive.jsonl", folder / "spec.json", root / "research"
        )
        outcome = native_outcome(card, root / "research", result, recompute=False)
    except Exception as exc:
        # An ordinary failed attempt is retained; process interruption leaves no final manifest.
        outcome = {
            "schema": KIND,
            "execution": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "decision": None,
        }
    records += [
        store.write_json("outcome.json", outcome),
        store.write_text("REPORT.md", render(card, outcome)),
    ]
    store.commit_manifest(
        records,
        {
            "kind": KIND,
            "registration_sha256": sha256_bytes(canonical_json_bytes(record)),
        },
    )
    return verify_method(root, trial_id)


def verify_method(root, trial_id):
    folder = trial_folder(root, trial_id)
    card_raw = read_input(folder / "card.json", 64 * 1024)
    card = method_contract.validate_card(strict_json(card_raw))
    sources = method_contract.read_sources(card, folder, frozen=True)
    manifest = checked_bundle(folder, FILES | sources.keys(), KIND)
    raw, spec_raw = (
        read_input(folder / "archive.jsonl", MAX_INPUT),
        read_input(folder / "spec.json", 64 * 1024),
    )
    record_raw = read_input(folder / "registration.json", 64 * 1024)
    record = strict_json(record_raw)
    if (
        manifest["run_id"] != trial_id
        or record != registration(card_raw, raw, spec_raw, sources)
        or manifest["metadata"]["registration_sha256"] != sha256_bytes(record_raw)
    ):
        raise ValueError("registration/input/code identity mismatch")
    outcome_raw = read_input(folder / "outcome.json", 128 * 1024)
    outcome = strict_json(outcome_raw)
    if outcome.get("schema") != KIND:
        raise ValueError("method outcome schema mismatch")
    if outcome["execution"] == "COMPLETED":
        expected = native_outcome(card, root / "research", outcome["native"])
        admission = root / "research/admissions/runs" / expected["native"]["run_id"]
        if raw != read_input(
            admission / "archive.jsonl", MAX_INPUT
        ) or spec_raw != read_input(admission / "requested-spec.json", 64 * 1024):
            raise ValueError("registered inputs differ from executed inputs")
        if canonical_json_bytes(expected) != outcome_raw:
            raise ValueError("native method decision differs")
        verification = "NATIVE_RECOMPUTE"
    elif (
        outcome["execution"] == "FAILED"
        and set(outcome) == {"schema", "execution", "error_type", "error", "decision"}
        and outcome["decision"] is None
    ):
        verification = "FAILURE_INTEGRITY_ONLY"
    else:
        raise ValueError("unsupported method outcome")
    if render(card, outcome).encode() != read_input(folder / "REPORT.md", 128 * 1024):
        raise ValueError("method report differs")
    return {
        "execution": outcome["execution"],
        "verification": verification,
        "decision": outcome["decision"],
        "directory": str(folder),
        "report": str(folder / "REPORT.md"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--store", type=Path, required=True)
        sub.add_argument("--trial-id", required=True)
        if name == "run":
            for field in ("card", "archive", "spec"):
                sub.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            run_method(args.card, args.archive, args.spec, args.store, args.trial_id)
            if args.command == "run"
            else verify_method(args.store, args.trial_id)
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        # A completed unfavorable research result is a successful software execution.
        return 0 if result["execution"] == "COMPLETED" else 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"execution": "REFUSED", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
