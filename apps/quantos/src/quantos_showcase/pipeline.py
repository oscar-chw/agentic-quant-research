"""Connected, offline E0 demo. Synthetic quotes are never executable fill prices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import paper_store
from qrae.data_catalog import DataCatalog
from qrae.kernel import run_price_baseline, verify_run
from replay.book import PRICE_SCALE, SIDE_ASK, SIDE_BID, Level, reconstruct_many
from replay.feed import Feed

from quantos_showcase.review import export_review

UTC = timezone.utc
INSTRUMENT = "synthetic-quote-A"
PARAMETERS = {"observations": 80, "interval_ms": 60000, "start_ms": 1704067200000}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def stamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat().replace("+00:00", "Z")


def fixture_messages() -> list[dict]:
    """Fixed, bounded zero-latency toy feed; the shape follows the collector API."""
    messages = []
    previous = None
    for i in range(PARAMETERS["observations"]):
        ts = PARAMETERS["start_ms"] + i * PARAMETERS["interval_ms"]
        # A deterministic sawtooth, not fitted or sampled market observations.
        bid = 4700 + 8 * (i % 11) - 3 * (i % 7)
        ask = bid + 20

        def price(tick):
            return str(Decimal(tick) / PRICE_SCALE)

        if previous is None:
            message = {
                "event_type": "book",
                "asset_id": INSTRUMENT,
                "timestamp": str(ts),
                "bids": [{"price": price(bid), "size": "12.00"}],
                "asks": [{"price": price(ask), "size": "8.00"}],
            }
        else:
            changes = [
                {"asset_id": INSTRUMENT, "side": side, "price": price(tick), "size": size}
                for side, tick, size in [
                    ("BUY", previous[0], "0"),
                    ("SELL", previous[1], "0"),
                    ("BUY", bid, "12.00"),
                    ("SELL", ask, "8.00"),
                ]
            ]
            message = {
                "event_type": "price_change",
                "timestamp": str(ts),
                "price_changes": changes,
            }
        messages.append({"source_event_ms": ts, "received_ms": ts, "message": message})
        previous = bid, ask
    return messages


def collect_snapshot(messages: list[dict], output: Path) -> dict:
    """Run the replay feed over the messages and persist the history it yields
    (keyframes, deltas, coverage) as hashed JSONL parts, keeping the raw bytes
    and both source clocks."""
    if not 30 <= len(messages) <= 512:
        raise ValueError("demo requires 30..512 observations")
    previous = -1
    for event in messages:
        ts = event["source_event_ms"]
        if type(ts) is not int or ts <= previous:
            raise ValueError("source times must be strictly increasing integers")
        if event["received_ms"] != ts:
            raise ValueError(
                "this synthetic adapter requires zero latency; delayed data is unsupported"
            )
        if int(event["message"]["timestamp"]) != ts:
            raise ValueError("source clock disagrees with message")
        previous = ts
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw_messages.jsonl").write_bytes(
        b"".join(canonical(event) for event in messages)
    )
    feed = Feed()
    streams = {"keyframes": [], "deltas": [], "coverage": []}
    for event in messages:
        update = feed.apply(event["message"])
        if update.disagreed:
            raise ValueError("collector rejected inconsistent feed state")
        for delta in update.deltas:
            if delta.asset != INSTRUMENT:
                raise ValueError("unknown instrument")
            streams["deltas"].append(
                {"asset": delta.asset, "ts_ms": delta.ts_ms, "side": delta.side,
                 "tick": delta.tick, "size_e2": delta.size_e2}
            )
        for frame in update.keyframes:
            if frame.asset != INSTRUMENT:
                raise ValueError("unknown instrument")
            streams["keyframes"].extend(
                {"asset": frame.asset, "ts_ms": frame.ts_ms, "side": level.side,
                 "tick": level.tick, "size_e2": level.size_e2}
                for level in frame.levels
            )
    # One instrument, arriving continuously, closed when this single batch ends.
    streams["coverage"].append(
        {"asset": INSTRUMENT, "start_ms": messages[0]["source_event_ms"],
         "end_ms": messages[-1]["source_event_ms"], "reason": "shutdown"}
    )
    (output / "history").mkdir()
    parts = {}
    for name, rows in streams.items():
        path = output / "history" / f"{name}.jsonl"
        path.write_bytes(b"".join(canonical(row) for row in rows))
        parts[str(path.relative_to(output))] = digest(path.read_bytes())
    snapshot = {
        "schema_version": "quantos.collector-fixture.v1",
        "synthetic": True,
        "instrument": INSTRUMENT,
        "decision_times_ms": [event["source_event_ms"] for event in messages],
        "raw_sha256": digest((output / "raw_messages.jsonl").read_bytes()),
        "parts": parts,
    }
    write_json(output / "snapshot.json", snapshot)
    return snapshot


def derive_prices(source: Path, destination: Path) -> dict:
    """Read the hashed persisted parts, replay covered books, export exact midpoints."""
    snapshot = json.loads((source / "snapshot.json").read_bytes())
    if snapshot["raw_sha256"] != digest((source / "raw_messages.jsonl").read_bytes()):
        raise ValueError("collector raw hash mismatch")
    streams = defaultdict(list)
    for name, expected in snapshot["parts"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or path.is_symlink():
            raise ValueError("collector part escapes source")
        if digest(path.read_bytes()) != expected:
            raise ValueError("collector part hash mismatch")
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if any(row["asset"] != snapshot["instrument"] for row in rows):
            raise ValueError("unknown instrument in collector part")
        streams[Path(name).stem].extend(rows)
    frames = defaultdict(list)
    for row in streams["keyframes"]:
        frames[row["ts_ms"]].append(Level(row["side"], row["tick"], row["size_e2"]))
    keyframes = sorted(frames.items())
    # Rows at the same timestamp must preserve their input order (delete then set).
    deltas = [
        (r["ts_ms"], r["side"], r["tick"], r["size_e2"])
        for r in sorted(streams["deltas"], key=lambda r: r["ts_ms"])
    ]
    coverage = [(r["start_ms"], r["end_ms"]) for r in streams["coverage"]]
    records, diagnostics = [], []
    times = snapshot["decision_times_ms"]
    books = reconstruct_many(keyframes, deltas, times, coverage)
    for ts, book in zip(times, books, strict=True):
        bids = [
            (tick, size)
            for (side, tick), size in book.levels.items()
            if side == SIDE_BID
        ]
        asks = [
            (tick, size)
            for (side, tick), size in book.levels.items()
            if side == SIDE_ASK
        ]
        if not bids or not asks:
            raise ValueError("two-sided book required")
        bid, bid_size = max(bids)
        ask, ask_size = min(asks)
        if bid <= 0 or ask <= bid:
            raise ValueError("positive uncrossed book required")
        mid = Decimal(bid + ask) / (2 * PRICE_SCALE)
        records.append([stamp(ts), stamp(ts), snapshot["instrument"], str(mid)])
        diagnostics.append(
            {
                "event_time": stamp(ts),
                "spread": str(Decimal(ask - bid) / PRICE_SCALE),
                "top_imbalance": str(
                    Decimal(bid_size - ask_size) / (bid_size + ask_size)
                ),
            }
        )
    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / "prices.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["event_time", "available_time", "instrument", "price"])
        writer.writerows(records)
    write_json(destination / "diagnostics.json", diagnostics)
    # Normalized row hash: independent of part names and row order.
    normalized = {
        name: sorted(rows, key=lambda row: canonical(row))
        for name, rows in sorted(streams.items())
    }
    derivation = {
        "schema_version": "qrae.synthetic-derivation.v1",
        "synthetic": True,
        "market_family": "prediction-synthetic",
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "parameters": PARAMETERS
        | {
            "price_rule": "(best_bid_tick+best_ask_tick)/(2*10000)",
            "clock_rule": "synthetic source_event_ms=received_ms=decision_ms",
        },
        "collector_raw_sha256": snapshot["raw_sha256"],
        "collector_normalized_sha256": digest(canonical(normalized)),
        "normalized_price_sha256": digest(csv_path.read_bytes()),
    }
    write_json(destination / "derivation.json", derivation)
    return derivation


def reference_metadata() -> dict:
    """Public citation checked 2026-09-13; abstract field is our bounded paraphrase."""
    return {
        "arxiv_id": "2505.15155v2",
        "title": "R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization",
        "authors": [
            "Yuante Li",
            "Xu Yang",
            "Xiao Yang",
            "Minrui Xu",
            "Xisen Wang",
            "Weiqing Liu",
            "Jiang Bian",
        ],
        "abstract": "Reading note, not the original abstract: this paper separates hypothesis formation and implementation, with experimental feedback between them. Here it is workflow background only; this fixture does not implement or reproduce the paper's factor-model search or results.",
        "categories": ["q-fin.CP", "cs.AI"],
        "published": "2025-09-25T10:13:08Z",
        "pdf_url": "https://arxiv.org/pdf/2505.15155v2",
    }


def _package_version(name):
    # The monorepo check runs from source via PYTHONPATH, where no distribution
    # metadata exists; record that honestly instead of failing the demo.
    try:
        return version(name)
    except PackageNotFoundError:
        return "uninstalled-source-tree"


def run_demo(output: Path, *, observed: datetime | None = None) -> dict:
    """Run from installed dependencies into one new, disposable directory."""
    from qrae.synthetic_import import import_synthetic_prices

    observed = observed or datetime.now(UTC)
    if observed.tzinfo is None:
        raise ValueError("observation clock must be timezone-aware")
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    collect_snapshot(fixture_messages(), workspace / "collector")
    derivation = derive_prices(workspace / "collector", workspace / "data")
    snapshot = import_synthetic_prices(
        DataCatalog(workspace / "catalog"),
        workspace / "data/prices.csv",
        derivation,
        observed,
    )
    conn = paper_store.init_db(str(workspace / "knowledge.sqlite"))
    try:
        paper_store.save_paper(conn, reference_metadata())
        paper_store.save_paper(conn, reference_metadata())
        references = paper_store.get_unprocessed(conn)
        if paper_store.count_total(conn) != 1:
            raise RuntimeError("reference deduplication failed")
    finally:
        conn.close()
    order = {
        "schema_version": "1.1",
        "work_order_id": "connected-quote-baseline",
        "kind": "PRICE_BASELINE",
        "market_family": "prediction-synthetic",
        "created_at": observed.isoformat(),
        "expires_at": (observed + timedelta(days=1)).isoformat(),
        "cutoff": observed.isoformat(),
        "dataset": {
            "path": "data/prices.csv",
            "sha256": snapshot["normalized_sha256"],
            "format": "csv",
            "catalog_snapshot": {
                "catalog_root": "catalog",
                "snapshot_id": snapshot["snapshot_id"],
            },
        },
        "hypothesis": {
            "mechanism": "Toy quote paths may exhibit short-lived directional persistence; a software fixture hypothesis only.",
            "prediction": "A lagged positive midpoint return predicts a positive next-observation midpoint return, and vice versa.",
            "horizon": "one synthetic minute",
            "falsifier": "Locked test after-cost return is non-positive.",
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
    order_path = workspace / "work_order.json"
    write_json(order_path, order)
    first = run_price_baseline(
        order_path, workspace=workspace, output_root=output / "research", now=observed
    )
    replay = run_price_baseline(
        order_path, workspace=workspace, output_root=output / "research", now=observed
    )
    verified = verify_run(first["run_dir"])
    if (
        not replay["idempotent_replay"]
        or first["manifest_sha256"] != replay["manifest_sha256"]
    ):
        raise RuntimeError("QRAE replay mismatch")
    if first["evidence_tier"] != "E0" or first["live_trading_authorized"]:
        raise RuntimeError("synthetic authority exceeded")
    summary = {
        "schema_version": "quantos.connected-demo.v1",
        "synthetic": True,
        "cross_repository_data_flow": "PASS_SYNTHETIC_ONLY",
        "status": verified["status"],
        "snapshot_id": snapshot["snapshot_id"],
        "manifest_sha256": first["manifest_sha256"],
        "run_dir": str(Path(first["run_dir"]).relative_to(output)),
        "normalized_price_sha256": derivation["normalized_price_sha256"],
        "observations": PARAMETERS["observations"],
        "evidence_tier": "E0",
        "idempotent_replay": True,
        "live_trading_authorized": False,
        "references": references,
        "packages": {
            name: _package_version(name)
            for name in [
                "quantos-showcase",
                "quant-marketdata",
                "qrae-rd",
                "quant-paper-store",
            ]
        },
        "limits": [
            "synthetic zero-latency single instrument only",
            "midpoints are not executable fills",
            "lagged momentum is the sole consumed strategy feature; spread/imbalance are diagnostics",
            "vault citation joins the review report, not automatic hypothesis generation",
            "no autonomous model calls, feature search, live data, execution or profitability evidence",
        ],
    }
    write_json(output / "summary.json", summary)
    reference = references[0]
    notes = workspace / "notes"
    notes.mkdir()
    note_text = f"# Workflow reading note\n\n{reference['abstract']}\n"
    (notes / "workflow.md").write_text(note_text, encoding="utf-8")
    meta = {"arxiv_id": reference["arxiv_id"], "title": reference["title"],
            "source_url": f"https://arxiv.org/abs/{reference['arxiv_id']}",
            "vault_path": "workspace/notes/workflow.md", "document_type": "paper_note",
            "source_available": True, "source_status": "present", "source_sha256": digest(note_text.encode())}
    source_records = [{"metadata": meta, "document": note_text}]
    for fields in [
        {"arxiv_id": "fixture:derived", "document_type": "derived_guideline"},
        {"arxiv_id": "fixture:unavailable", "source_available": False, "source_status": "unlisted"},
        {"arxiv_id": "fixture:deleted", "vault_path": "workspace/notes/nonexistent.md"},
    ]:
        source_records.append({"metadata": meta | fields, "document": note_text})
    write_json(workspace / "sources.json", {"schema_version": "quantos.source-fixtures.v1",
               "required_source_ids": [reference["arxiv_id"]], "records": source_records})
    export_review(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing paths are refused",
    )
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
