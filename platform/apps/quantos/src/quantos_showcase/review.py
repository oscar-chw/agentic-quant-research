"""Current-run review export, verified against artifacts and local note digests."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from qrae.kernel import verify_run
from source_access import select_note_sources


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_bytes())


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("review source escapes output directory")
    return path


def export_review(output: Path, *, destination: Path | None = None) -> dict:
    """Reverify before export; no report is published on a failed evidence check.

    Eligibility is an as-of-export check of local note bytes. Re-export to a new
    destination to check changed sources; a static report cannot promise freshness.
    """
    output = output.resolve()
    destination = (destination or output).resolve()
    targets = [destination / "review.json", destination / "REPORT.md"]
    if any(path.exists() for path in targets):
        raise FileExistsError("review output exists; choose a new destination")
    summary = _json(output / "summary.json")
    run = _inside(output, summary["run_dir"])
    verified = verify_run(run)
    manifest_sha256 = _hash(run / "manifest.json")
    if manifest_sha256 != summary["manifest_sha256"]:
        raise ValueError("summary and verified run differ")
    paths = [output / "summary.json", run / "manifest.json"]
    manifest = _json(run / "manifest.json")
    paths += [_inside(run, a["path"]) for a in manifest["artifacts"]]
    result = _json(run / "result.json")
    validation = _json(run / "validation.json")
    decision = _json(run / "decision.json")
    provenance = _json(run / "snapshot/provenance.json")
    lineage = _json(output / "workspace/collector/snapshot.json")
    derivation = _json(output / "workspace/data/derivation.json")
    if not summary["synthetic"] or summary["evidence_tier"] != "E0" or result["live_trading_authorized"]:
        raise ValueError("this report supports only the synthetic E0 workflow")
    collector = output / "workspace/collector"
    raw = collector / "raw_messages.jsonl"
    if _hash(raw) != lineage["raw_sha256"] or lineage["raw_sha256"] != derivation["collector_raw_sha256"]:
        raise ValueError("collector raw lineage differs")
    for relative, expected in lineage["parts"].items():
        part = _inside(collector, relative)
        if _hash(part) != expected:
            raise ValueError("collector part lineage differs")
        paths.append(part)
    prices = output / "workspace/data/prices.csv"
    if _hash(prices) != result["dataset_sha256"] or result["dataset_sha256"] != derivation["normalized_price_sha256"]:
        raise ValueError("normalized price lineage differs")
    if provenance["synthetic_derivation"] != derivation:
        raise ValueError("frozen and current derivation differ")
    raw_records = [json.loads(line) for line in raw.read_text().splitlines()]
    if [r["received_ms"] for r in raw_records] != lineage["decision_times_ms"] or any(
        r["source_event_ms"] != r["received_ms"] for r in raw_records
    ):
        raise ValueError("zero-latency fixture clocks differ")
    sources_path = output / "workspace/sources.json"
    sources = _json(sources_path)
    records = []
    for record in sources["records"]:
        meta = dict(record["metadata"])
        meta["vault_path"] = str(_inside(output, meta["vault_path"]))
        records.append({**record, "metadata": meta})
    eligibility = select_note_sources(records, source_root=output / "workspace/notes")
    actual = {r["metadata"]["arxiv_id"] for r in eligibility["matches"]}
    if not set(sources["required_source_ids"]).issubset(actual) or not actual:
        raise ValueError("required source is ineligible; refresh its evidence before export")
    paths += [raw, prices, sources_path, collector / "snapshot.json", output / "workspace/data/derivation.json",
              output / "workspace/data/diagnostics.json"]
    paths += [Path(r["metadata"]["vault_path"]) for r in eligibility["matches"]]
    bindings = {str(p.relative_to(output)): _hash(p) for p in paths}
    series = [json.loads(line) for line in (run / "backtest/series.jsonl").read_text().splitlines()]
    splits, offset = {}, 0
    for name in ("train", "validation", "test"):
        count = result["metrics"][name]["observations"]
        rows = series[offset:offset + count]
        splits[name] = {"start_exit_time": rows[0]["exit_time"], "end_exit_time": rows[-1]["exit_time"],
                        "metrics": result["metrics"][name]}
        offset += count
    if offset != len(series):
        raise ValueError("reported split counts do not exhaust the holding series")
    eligible = [{"source_id": r["metadata"]["arxiv_id"], "title": r["metadata"]["title"],
                 "path": str(Path(r["metadata"]["vault_path"]).relative_to(output)),
                 "source_url": r["metadata"]["source_url"], **r["eligibility"]}
                for r in eligibility["matches"]]
    review = {
        "schema_version": "quantos.current-review.v1", "verified_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": summary["run_dir"], "manifest_sha256": manifest_sha256,
        "status": verified["status"], "decision": decision["proposed_decision"], "evidence_tier": "E0",
        "synthetic": True, "live_trading_authorized": False, "packages": summary["packages"],
        "input": {"events": len(raw_records), "raw_bytes": raw.stat().st_size, "raw_sha256": _hash(raw),
                  "normalized_bytes": prices.stat().st_size, "normalized_sha256": _hash(prices),
                  "origin": provenance["adapter"]["source_kind"], "snapshot_id": summary["snapshot_id"]},
        "clocks": {"first_source_event_ms": raw_records[0]["source_event_ms"],
                   "last_source_event_ms": raw_records[-1]["source_event_ms"],
                   "event_receipt_decision_relation": "source_event_ms = received_ms = decision_ms (synthetic)",
                   "bar_availability": "available_time = event_time (synthetic only)",
                   "catalog_availability_basis": provenance["snapshot"]["availability_basis"],
                   "import_observed_at": provenance["snapshot"]["available_at"], "cutoff": result["cutoff"]},
        "validation": validation, "strategy": result["strategy"], "experiment": result["experiment"],
        "metric_scope": result["metric_scope"], "execution_scope": result["execution_scope"], "splits": splits,
        "source_eligibility": {"status": eligibility["status"], "scanned": eligibility["scanned"],
                               "eligible": eligible, "excluded": eligibility["excluded"],
                               "checked_as_of": "export; local note bytes only, not original paper claims"},
        "artifact_sha256": bindings, "limits": summary["limits"],
    }

    def link(label, relative):
        path = _inside(output, relative)
        return f"[{label}]({quote(os.path.relpath(path, destination), safe='/')})"

    ref = validation["checks"]["price_to_position_replay"]
    test = splits["test"]["metrics"]
    falsified = test["net_compounded_return"] <= 0
    verdict = "met" if falsified else "not met in this single fixture"
    lines = ["# QuantOS current-run review", "",
             f"**{review['status']} / {review['decision']} / E0.** The locked synthetic test returned "
             f"**{test['net_compounded_return']:.2%} after costs** over {test['observations']} holdings. "
             f"The registered non-positive-return falsifier was {verdict}. This is a mechanical research result.", "",
             "Use this report to inspect where the data came from, whether the accounting replay passed, "
             "and which source notes were eligible before considering another experiment.", "",
             f"Verified at {review['verified_at']}. Manifest `{review['manifest_sha256']}`. "
             "Re-export to a new destination to recheck current artifacts and notes.", "",
             f"{link('Machine-readable review', str((destination / 'review.json').relative_to(output)))} · "
             f"{link('Kernel report', summary['run_dir'] + '/report.md')} · "
             f"{link('Decision', summary['run_dir'] + '/decision.json')}", "",
             "## Input and clocks", "",
             f"{len(raw_records)} deterministic synthetic quote messages, one instrument, "
             f"{review['input']['raw_bytes']:,} raw bytes; {review['input']['normalized_bytes']:,} normalized CSV bytes. "
             "Midpoint quotes are not executable fills.", "",
             f"- {link('Raw messages', 'workspace/collector/raw_messages.jsonl')}: `{_hash(raw)}`",
             f"- {link('Normalized prices', 'workspace/data/prices.csv')}: `{_hash(prices)}`",
             f"- {link('Collector lineage', 'workspace/collector/snapshot.json')} · "
             f"{link('Derivation', 'workspace/data/derivation.json')} · "
             f"{link('Frozen catalog provenance', summary['run_dir'] + '/snapshot/provenance.json')}", "",
             "| Clock | Recorded meaning |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in review["clocks"].items()]
    lines += ["", "## Accounting, splits and costs", "",
              f"Independent price-to-position reference: **{ref['status']}**, {ref['observations_recomputed']} holdings, "
              f"{ref['reference_version']}; {ref['arithmetic']}, tolerance {ref['tolerance']}. "
              "This is a separate implementation in the same package; external reproduction remains unrun.", "",
              f"{link('Validation receipt', summary['run_dir'] + '/validation.json')} · "
              f"{link('Registered work order', summary['run_dir'] + '/work_order.json')} · "
              f"{link('Holding series', summary['run_dir'] + '/backtest/series.jsonl')}", "",
              f"Lagged momentum, lookback {result['strategy']['lookback']}; {result['strategy']['cost_bps']:g} bps "
              "per unit absolute position change. Signals use prior prices, enter on the following observation "
              "and earn the next holding return. Each split starts flat and pays entry and terminal liquidation costs. "
              "Costs below are a sum of decimal return deductions, expressed as percent; they are not the difference of compounded gross and net returns.", "",
              "| Split | Holding exit range (UTC) | Holdings | Gross compounded | Net compounded | Summed costs | Turnover units | Max drawdown |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, split in splits.items():
        m = split["metrics"]
        lines.append(f"| {name} | {split['start_exit_time']} to {split['end_exit_time']} | {m['observations']} | "
                     f"{m['gross_compounded_return']:.4%} | {m['net_compounded_return']:.4%} | {m['cost_return_sum']:.4%} | "
                     f"{m['turnover_units']:g} | {m['maximum_drawdown']:.4%} |")
    lines += ["", result["metric_scope"] + ". " + result["execution_scope"] + ".", "",
              "## Source eligibility", "",
              f"{len(eligible)} eligible note out of {eligibility['scanned']} explicit records. "
              "The other records are controlled exclusion fixtures. Eligibility checks type, availability, identity, "
              "regular local file and matching indexed/current digest. This checks note context; it does not validate original-paper claims.", ""]
    for source in eligible:
        lines.append(f"- {link(source['source_id'] + ' reading note', source['path'])} · "
                     f"[{source['title']}]({source['source_url']}); `{source['source_sha256']}`. Workflow background only.")
    lines += ["", "| Excluded fixture | Reason |", "|---|---|"]
    lines += [f"| {r['source_id']} | {r['reason']} |" for r in eligibility["excluded"]]
    lines += ["", "## Limits", ""] + [f"- {limit}" for limit in review["limits"]]
    lines += ["- No real Chroma retrieval or model quality evaluation; optional adapters have controlled fake tests only.",
              "- All artifact digests and package versions are recorded in review.json; this static file is an as-of-export receipt.", ""]
    markdown = "\n".join(lines)
    verify_run(run)
    if any(_hash(_inside(output, path)) != expected for path, expected in bindings.items()):
        raise ValueError("evidence changed during export")
    destination.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(review, sort_keys=True, indent=2, allow_nan=False) + "\n")
    targets[1].write_text(markdown)
    return review
