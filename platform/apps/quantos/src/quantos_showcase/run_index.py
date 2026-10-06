"""Immutable navigation across selected native quant runs; no combined score."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from qrae import artifacts, contracts, path_safety, quote_features, quote_workflow
from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.quote_features import strict_json
from qrae.quote_workflow import checked_bundle, read_input, reserve

from quantos_showcase import run_adapters
from quantos_showcase.run_adapters import KINDS, inspect_run

MEMBERS = {"request.json", "selection.json", "index.json", "INDEX.md"}
SCHEMA = "quantos-run-selection/v1"


def code_identity():
    names = {
        "run_index.py": Path(__file__),
        "run_adapters.py": Path(run_adapters.__file__),
        "qrae.artifacts": Path(artifacts.__file__),
        "qrae.path_safety": Path(path_safety.__file__),
        "qrae.quote_features": Path(quote_features.__file__),
        "qrae.quote_workflow": Path(quote_workflow.__file__),
        "qrae.contracts": Path(contracts.__file__),
    }
    return sha256_bytes(
        canonical_json_bytes(
            {n: sha256_bytes(p.read_bytes()) for n, p in names.items()}
        )
    )


def validate_selection(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "entries"}
        or value["schema"] != SCHEMA
    ):
        raise ValueError("exact quantos-run-selection/v1 object required")
    entries = value["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
        raise ValueError("select 1..16 explicit native runs")
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            raise ValueError("unsupported native run kind")
        if set(entry) != {"id", "kind", "path"}:
            raise ValueError("exact fields required for selected run kind")
        name = entry["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name)
            or name in ids
        ):
            raise ValueError("unique safe entry IDs required")
        ids.add(name)
        text = entry["path"]
        if (
            not isinstance(text, str)
            or not 1 <= len(text) <= 4096
            or any(ord(c) < 32 for c in text)
        ):
            raise ValueError("explicit local path required")
    return value


def relative(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def normalized_selection(request, origin, destination):
    rows = []
    for entry in request["entries"]:
        row = dict(entry)
        path = origin / row["path"]
        # Do not resolve away an input link before the native adapter rejects it.
        if path_safety.has_link_or_reparse_component(path):
            row["path"] = str(path.absolute())
        else:
            row["path"] = relative(path.absolute(), destination)
        rows.append(row)
    return {"schema": SCHEMA, "entries": rows}


def collect(selection, folder):
    rows = []
    for entry in selection["entries"]:
        row = inspect_run(entry, folder)
        if "report" in row:
            row["report"] = relative(row["report"], folder)
            for method in row["methods"]:
                method["path"] = relative(method["path"], folder)
        row["evidence"] = {
            relative(name, folder): item
            for name, item in sorted(row["evidence"].items())
        }
        rows.append(row)
    partial = any(
        r["result_state"] in {"REFUSED", "UNAVAILABLE"}
        or (r["kind"] == "quantos-persisted" and r["result_state"] != "COMPLETED")
        for r in rows
    )
    return {
        "schema": "quantos-run-index/v1",
        "status": "PARTIAL" if partial else "VERIFIED_INDEX",
        "entries": rows,
        "combined_score": None,
        "scope": "Selected runs only, checked at build/recheck; no discovery or completeness claim",
        "qualification": "Native replay/integrity scopes are per entry; no historical authorship, source authenticity or trading-quality qualification",
    }


def render(index):
    lines = [
        "# QuantOS research result index",
        "",
        "Status: **" + index["status"] + "**.",
        "",
        "Follow each result into its native report, input evidence and installed method code.",
        "Values retain their own units and accounting. There is no cross-project score.",
        "This is a snapshot of the selected runs; recheck before relying on changed files.",
        "",
        "| Entry | Kind | Result | Verification scope |",
        "|---|---|---|---|",
    ]
    for row in index["entries"]:
        label = (
            "[" + row["id"] + "](" + quote(row["report"], safe="/") + ")"
            if "report" in row
            else row["id"]
        )
        lines.append(
            f"| {label} | {row['kind']} | {row['result_state']} | {row['verification']} |"
        )
    for row in index["entries"]:
        lines += ["", "## " + row["id"], ""]
        if "report" not in row:
            lines += [
                "**" + row["result_state"] + "**",
                "",
                "```json",
                json.dumps({"reason": row["reason"]}, ensure_ascii=False),
                "```",
            ]
        else:
            lines += [
                row["scope"],
                "",
                "[Open native report](" + quote(row["report"], safe="/") + ")",
                "",
                "### Native result and qualifications",
                "",
                "```json",
                json.dumps(
                    {
                        "metrics": row["metrics"],
                        "qualifications": row["qualifications"],
                    },
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "```",
                "",
                "### Method implementation",
                "",
            ]
            for method in row["methods"]:
                lines.append(
                    "- ["
                    + method["module"]
                    + "]("
                    + quote(method["path"], safe="/")
                    + ") — "
                    + method["purpose"]
                )
        lines += [
            "",
            "### Checked evidence",
            "",
            "| File | Role | SHA256 |",
            "|---|---|---|",
        ]
        for name, item in row["evidence"].items():
            label = (
                Path(name)
                .name.replace("[", "\\[")
                .replace("]", "\\]")
                .replace("|", "\\|")
            )
            lines.append(
                f"| [{label}]({quote(name, safe='/')}) | {item['role']} | `{item['sha256']}` |"
            )
    lines += ["", index["scope"], "", index["qualification"], ""]
    return "\n".join(lines)


def index_folder(root, index_id):
    if not isinstance(index_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", index_id
    ):
        raise ValueError("safe 1..64 character index ID required")
    folder = root / "runs" / index_id
    if path_safety.has_link_or_reparse_component(folder):
        raise ValueError("unlinked index store required")
    return folder.absolute()


def build_index(request_path, root, index_id):
    folder = index_folder(root, index_id)
    if folder.exists():
        raise FileExistsError("index exists; recheck it or use a new ID")
    request_raw = read_input(request_path, 128 * 1024)
    request = validate_selection(strict_json(request_raw))
    origin = request_path.absolute().parent
    selection = normalized_selection(request, origin, folder)
    index = collect(selection, folder)
    store = reserve(root, index_id)
    if store is None:
        raise FileExistsError("concurrent index writer; existing output preserved")
    records = [
        store.write_bytes("request.json", request_raw),
        store.write_json("selection.json", selection),
        store.write_json("index.json", index),
        store.write_text("INDEX.md", render(index)),
    ]
    store.commit_manifest(
        records,
        {
            "kind": "quantos-run-index/v1",
            "identity": {
                "code_sha256": code_identity(),
                "request_sha256": sha256_bytes(request_raw),
                "selection_sha256": sha256_bytes(canonical_json_bytes(selection)),
                "request_base": relative(origin, folder),
            },
        },
    )
    return {
        "status": index["status"],
        "directory": str(folder),
        "index": str(folder / "INDEX.md"),
        "entries": len(index["entries"]),
        "index_current": True,
    }


def verify_index(root, index_id):
    folder = index_folder(root, index_id)
    manifest = checked_bundle(folder, MEMBERS, "quantos-run-index/v1")
    identity = manifest["metadata"]["identity"]
    if manifest["run_id"] != index_id or identity["code_sha256"] != code_identity():
        raise ValueError("index ID/code differs; rebuild with a new ID")
    request_raw = read_input(folder / "request.json", 128 * 1024)
    request = validate_selection(strict_json(request_raw))
    selection_raw = read_input(folder / "selection.json", 128 * 1024)
    selection = validate_selection(strict_json(selection_raw))
    if (
        identity["request_sha256"] != sha256_bytes(request_raw)
        or identity["selection_sha256"] != sha256_bytes(selection_raw)
        or selection
        != normalized_selection(request, folder / identity["request_base"], folder)
    ):
        raise ValueError("index selection identity differs")
    actual = collect(selection, folder)
    if canonical_json_bytes(actual) != read_input(
        folder / "index.json", 16 * 1024 * 1024
    ) or render(actual).encode() != read_input(folder / "INDEX.md", 16 * 1024 * 1024):
        raise ValueError(
            "STALE index: selected source, result, method code or verification changed"
        )
    return {
        "status": actual["status"],
        "directory": str(folder),
        "index": str(folder / "INDEX.md"),
        "entries": len(actual["entries"]),
        "index_current": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--store", type=Path, required=True)
        sub.add_argument("--index-id", required=True)
        if name == "build":
            sub.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            build_index(args.selection, args.store, args.index_id)
            if args.command == "build"
            else verify_index(args.store, args.index_id)
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result["status"] == "VERIFIED_INDEX" else 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
