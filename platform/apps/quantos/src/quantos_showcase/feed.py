"""Connect archived feed admission to installed reusable quote research."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

from replay.admission import MAX_INPUT, adapter_identity, admit
from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.quote_features import COLUMNS, feature_spec, read_quotes, strict_json
from qrae.quote_workflow import (
    checked_bundle,
    code_identity,
    read_input,
    reserve,
    run_quotes,
    verify_quote_run,
)

MEMBERS = {
    "archive.jsonl",
    "requested-spec.json",
    "quotes.csv",
    "quote-spec.json",
    "admission.json",
    "research-ref.json",
}


def application_identity():
    return sha256_bytes(Path(__file__).read_bytes())


def prepare(raw, spec_raw):
    spec = strict_json(spec_raw)
    feature_spec(spec)
    if spec["schema"] != "quote-spec/v1":
        raise ValueError(
            "request uses quote-spec/v1; admission creates source-bound v2"
        )
    header, rows, admission = admit(raw)
    if (
        spec["instrument"] != header["instrument"]
        or spec["data_kind"] != header["data_kind"]
        or spec["price_tick"] != "0.0001"
    ):
        raise ValueError(
            "spec must match archive instrument/data kind and exact 0.0001 price tick"
        )
    bound = spec | dict(
        schema="quote-spec/v2",
        sequence_kind="collector-record-order",
        source_sha256=admission["source_sha256"],
        adapter_sha256=admission["adapter_sha256"],
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=(*COLUMNS, "status", "segment"), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    quotes = stream.getvalue().encode()
    read_quotes(quotes, feature_spec(bound))  # Consumer contract before any writes.
    return quotes, canonical_json_bytes(bound), admission


def run_feed(archive: Path, specification: Path, root: Path):
    raw = read_input(archive, MAX_INPUT)
    spec_raw = read_input(specification, 64 * 1024)
    quotes, bound, admission = prepare(raw, spec_raw)
    identity = dict(
        source_sha256=sha256_bytes(raw),
        spec_sha256=sha256_bytes(spec_raw),
        adapter_sha256=adapter_identity(),
        application_sha256=application_identity(),
        research_code_sha256=code_identity(),
    )
    run_id = "feed-" + sha256_bytes(canonical_json_bytes(identity))
    store = reserve(root / "admissions", run_id)
    if store is None:
        return verify_feed(root, run_id) | {"admission_reused": True}
    folder = root / "admissions/runs" / run_id
    records = [
        store.write_bytes("archive.jsonl", raw),
        store.write_bytes("requested-spec.json", spec_raw),
        store.write_bytes("quotes.csv", quotes),
        store.write_bytes("quote-spec.json", bound),
        store.write_json("admission.json", admission),
    ]
    result = run_quotes(folder / "quotes.csv", folder / "quote-spec.json", root)
    reference = {k: result[k] for k in ("run_id", "feature_id")}
    records.append(store.write_json("research-ref.json", reference))
    store.commit_manifest(records, {"kind": "feed-research/v1", "identity": identity})
    return verify_feed(root, run_id) | {"admission_reused": False}


def verify_feed(root: Path, run_id: str, recompute=False):
    if not re.fullmatch(r"feed-[0-9a-f]{64}", run_id):
        raise ValueError("safe content-addressed feed ID required")
    folder = root / "admissions/runs" / run_id
    manifest = checked_bundle(folder, MEMBERS, "feed-research/v1")
    identity = manifest["metadata"]["identity"]
    if run_id != "feed-" + sha256_bytes(canonical_json_bytes(identity)):
        raise ValueError("feed identity mismatch")
    raw = read_input(folder / "archive.jsonl", MAX_INPUT)
    spec_raw = read_input(folder / "requested-spec.json", 64 * 1024)
    admission = strict_json(read_input(folder / "admission.json", 64 * 1024))
    bound_raw = read_input(folder / "quote-spec.json", 64 * 1024)
    bound = strict_json(bound_raw)
    expected = strict_json(spec_raw) | dict(
        schema="quote-spec/v2",
        sequence_kind="collector-record-order",
        source_sha256=sha256_bytes(raw),
        adapter_sha256=identity["adapter_sha256"],
    )
    feature_spec(expected)
    if (
        identity["source_sha256"] != sha256_bytes(raw)
        or identity["spec_sha256"] != sha256_bytes(spec_raw)
        or bound != expected
        or admission["source_sha256"] != identity["source_sha256"]
        or admission["adapter_sha256"] != identity["adapter_sha256"]
    ):
        raise ValueError("archive/spec/admission identity mismatch")
    reference = strict_json(read_input(folder / "research-ref.json", 64 * 1024))
    result = verify_quote_run(root, reference["run_id"], recompute)
    if result["feature_id"] != reference["feature_id"]:
        raise ValueError("research feature identity mismatch")
    feature_folder = root / "features/runs" / result["feature_id"]
    experiment = root / "experiments/runs" / reference["run_id"]
    research_manifest = strict_json((experiment / "manifest.json").read_bytes())
    if (
        read_input(folder / "quotes.csv", MAX_INPUT)
        != read_input(feature_folder / "quotes.csv", MAX_INPUT)
        or bound_raw != read_input(experiment / "experiment-spec.json", 64 * 1024)
        or research_manifest["metadata"]["identity"]["code_sha256"]
        != identity["research_code_sha256"]
    ):
        raise ValueError("admission/research source binding mismatch")
    if recompute:
        if (
            adapter_identity() != identity["adapter_sha256"]
            or application_identity() != identity["application_sha256"]
        ):
            raise ValueError("recompute requires original adapter/application code")
        quotes, bound_again, admission_again = prepare(raw, spec_raw)
        if (
            quotes != (folder / "quotes.csv").read_bytes()
            or bound_again != bound_raw
            or canonical_json_bytes(admission_again)
            != (folder / "admission.json").read_bytes()
        ):
            raise ValueError("recomputed admission differs")
    return dict(
        status="VERIFIED",
        run_id=run_id,
        research_run_id=reference["run_id"],
        feature_id=result["feature_id"],
        report=str(experiment / "REPORT.md"),
        recomputed=recompute,
        evidence_tier="E0",
        exchange_completeness_verified=False,
    )


def bridge_demo(snapshot_folder: Path, destination: Path):
    """Explicitly synthetic bridge: no availability is inferred for real archives."""
    snapshot_raw = read_input(snapshot_folder / "snapshot.json", 256 * 1024)
    snapshot = strict_json(snapshot_raw)
    raw = read_input(snapshot_folder / "raw_messages.jsonl", MAX_INPUT)
    if (
        snapshot.get("schema_version") != "quantos.collector-fixture.v1"
        or snapshot.get("synthetic") is not True
        or snapshot.get("raw_sha256") != sha256_bytes(raw)
    ):
        raise ValueError("verified synthetic collector fixture required")
    header = dict(
        schema="quant-feed-archive/v1",
        instrument=snapshot["instrument"],
        data_kind="synthetic",
        clock_evidence="SYNTHETIC_ZERO_LATENCY",
        origin=dict(
            raw_sha256=sha256_bytes(raw), snapshot_sha256=sha256_bytes(snapshot_raw)
        ),
    )
    output, times = [canonical_json_bytes(header)], []
    for index, line in enumerate(raw.splitlines()):
        row = strict_json(line)
        if not isinstance(row, dict) or set(row) != {
            "source_event_ms",
            "received_ms",
            "message",
        }:
            raise ValueError("exact legacy synthetic envelope required")
        stamp = row["source_event_ms"]
        if (
            type(stamp) is not int
            or stamp != row["received_ms"]
            or str(stamp) != row["message"].get("timestamp")
            or times
            and stamp <= times[-1]
        ):
            raise ValueError("legacy zero-latency, increasing clock contract required")
        times.append(stamp)
        output.append(
            canonical_json_bytes(
                dict(
                    record_id=index,
                    kind="message",
                    received_ms=stamp,
                    available_ms=stamp,
                    message=row["message"],
                )
            )
        )
    if not 30 <= len(times) <= 512 or times != snapshot.get("decision_times_ms"):
        raise ValueError("legacy fixture observation grid mismatch")
    payload = b"".join(output)
    admit(payload)
    # Parent components are checked by the shared path contract; exclusive create
    # preserves any existing archive. No implicit rewrite of the legacy source.
    from qrae.path_safety import has_link_or_reparse_component

    if has_link_or_reparse_component(destination):
        raise ValueError("unlinked destination required")
    with destination.open("xb") as stream:
        stream.write(payload)
    return dict(
        status="BRIDGED_SYNTHETIC",
        archive=str(destination),
        source_raw_sha256=sha256_bytes(raw),
        source_snapshot_sha256=sha256_bytes(snapshot_raw),
        archive_sha256=sha256_bytes(payload),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--store", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--store", type=Path, required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--recompute", action="store_true")
    bridge = commands.add_parser("from-demo")
    bridge.add_argument("--snapshot", type=Path, required=True)
    bridge.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_feed(args.archive, args.spec, args.store)
        elif args.command == "verify":
            result = verify_feed(args.store, args.run_id, args.recompute)
        else:
            result = bridge_demo(args.snapshot, args.archive)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, csv.Error) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
