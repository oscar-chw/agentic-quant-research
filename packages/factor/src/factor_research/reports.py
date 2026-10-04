"""Bounded human-readable diagnostics; complete numerical detail stays in JSON."""
import json


def human_report(report):
    number = lambda v: "unavailable" if v is None else f"{v:.8g}"
    lines = ["# Offline factor experiment", "", "New platform-independent research tooling. Synthetic fixtures are not market evidence.",
             "", f"Selected baseline: **{report['selection']['selected'] or 'none'}**. Selection uses validation IC only.",
             "", f"Panel rows: {report['observation_rows']}; missing rows: {report['missing_panel_rows']}; "
             f"blank prices: {report['null_prices']}; delayed rows: {report['delayed_rows']}.",
             "", "| Baseline | Split | Mean rank IC | Valid IC dates | Coverage | Mean gross | Mean net | Mean turnover |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    if "data_source" in report:
        source=report["data_source"]
        lines[2:2]=["**Availability: "+source["availability_status"]+". Historical point-in-time verification: NO.**", "",
                    *source["warnings"], ""]
    parts = [(factor, split, result) for factor, results in report["development"].items() for split, result in results.items()]
    if report["test"] is not None:
        parts.append((report["selection"]["selected"], "test (now exposed)", report["test"]))
    for factor, split, part in parts:
        s = part["summary"]
        lines.append(f"| {factor} | {split} | {number(s['rank_ic_mean'])} | {s['valid_ic_dates']} | "
                     f"{number(s['ic_coverage_mean'])} | {number(s['gross_mean'])} | {number(s['net_mean'])} | {number(s['turnover_mean'])} |")
    lines += ["", "IC is average-tie Spearman correlation across eligible assets for each date. Coverage uses the fixed configured universe.",
              "Gross/net are average interval P&L per fixed unit gross notional; turnover includes full entry and price-adjusted exit. "
              "The book is liquidated each interval. Costs are applied to both sides. No compounding or cross-date netting.",
              "Missing held-asset exit marks suppress the entire split's aggregate portfolio score; they do not reweight survivors.",
              "", "## Held-out interval diagnostics (maximum 40 rows)", ""]
    if report["test"] is None:
        lines.append("NOT_RUN: no valid validation IC was available to select a baseline.")
    else:
        daily=report["test"]["daily"]
        shown=daily if len(daily)<=40 else daily[:20]+daily[-20:]
        lines += ["| Decision UTC | Rank IC | IC pairs | Portfolio state | Net |", "|---|---:|---:|---|---:|"]
        for d in shown:
            lines.append(f"| {d['time']} | {number(d['rank_ic'])} | {d['ic_pair_count']} | {d['portfolio']['status']} | {number(d['portfolio']['net'])} |")
        if len(shown)<len(daily):
            lines.append(f"\nShowing first/last 20 of {len(daily)} intervals. Full numerical rows are retained in report.json.")
    lines += ["", "## Provenance", "", "```json", json.dumps(report["provenance"],sort_keys=True,indent=2), "```", "",
              "## Limits", ""] + ["- "+v for v in report["limitations"]]
    return "\n".join(lines)+"\n"
