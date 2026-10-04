"""Small importer baseline, using only the original synthetic example.

Run from the installed environment. No upstream software/data or speedup claim.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from imc4_analysis.analyzer import analyze
from imc4_analysis.cli import json_bytes
from imc4_analysis.community_import import normalize_community_log, _remove_object_trailing_commas
from imc4_analysis.report import render


def fixture(cycles):
    examples = Path(__file__).resolve().parents[1] / "examples/community-log"
    raw = (examples / "sample.log").read_text()
    sandbox, body = raw.split("Activities log:\n")
    activity, trades = body.split("Trade History:\n")
    activity = activity.strip().splitlines()
    trades = json.loads(_remove_object_trailing_commas(trades))
    expanded_activity, expanded_trades = [activity[0]], []
    for cycle in range(cycles):
        for line in activity[1:]:
            cells = line.split(";")
            cells[1] = str(int(cells[1]) + 300 * cycle)
            expanded_activity.append(";".join(cells))
        expanded_trades.extend(dict(row, timestamp=row["timestamp"] + 300 * cycle) for row in trades)
    generated = sandbox + "Activities log:\n" + "\n".join(expanded_activity) + "\n\nTrade History:\n" + json.dumps(expanded_trades)
    return generated.encode(), (examples / "import.json").read_bytes()


def run_case(cycles):
    raw, config = fixture(cycles)
    timings = []
    for repeat in range(3):
        gc.collect()
        started = time.perf_counter()
        normalized, provenance = normalize_community_log(raw, config)
        report = analyze(normalized, detail="sampled")
        timings.append(time.perf_counter() - started)
        summary = report["summary"]
        # Integer ledger per cycle: cash -99, one ALPHA retained, fees 4.
        assert summary["cash"] == str(1000 - 99 * cycles)
        assert summary["positions"] == {"ALPHA": cycles, "BETA": 0}
        assert summary["equity"] == str(1000 + 3 * cycles)
        assert summary["pnl"] == str(3 * cycles)
        assert summary["total_fees"] == str(4 * cycles)
        assert provenance["own_trade_rows"] == 4 * cycles
        assert provenance["market_trade_rows_excluded"] == 2 * cycles
        assert summary["event_count"] == 10 * cycles
        assert report["details"]["mode"] == "sampled"
    report["import_provenance"] = provenance
    serialized, html = json_bytes(report), render(report).encode()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(cycles=cycles, raw_rows=12 * cycles, normalized_events=10 * cycles,
                raw_bytes=len(raw), normalized_bytes=len(normalized),
                raw_sha256=hashlib.sha256(raw).hexdigest(), normalized_sha256=hashlib.sha256(normalized).hexdigest(),
                import_and_analysis_seconds=timings, report_json_bytes=len(serialized), report_html_bytes=len(html),
                summary=summary, details=report["details"], code_identity=report["code_identity"],
                process_peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
                rss_scope="Fresh process including fixture generation, three import/analyze calls and rendering; not per-call allocation.",
                correctness="PASS: independent per-cycle integer ledger and explicit own/market counts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    if args.case:
        (args.out / f"case-{args.case}.json").write_bytes(json_bytes(run_case(args.case)))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cases = []
    for cycles in (1, 100, 800):
        subprocess.run([sys.executable, __file__, "--out", str(args.out), "--case", str(cycles)], check=True, timeout=60)
        cases.append(json.loads((args.out / f"case-{cycles}.json").read_text()))
    receipt = dict(python=sys.version, platform=platform.platform(), executable=sys.executable,
                   total_elapsed_seconds=time.perf_counter() - started, cases=cases,
                   scope="Original two-instrument synthetic example repeated at distinct timestamps; three sequential in-memory import+analysis calls in a fresh process per case; default sampled detail. No cold-disk, concurrency, official-format or upstream execution claim; unchanged caps.")
    (args.out / "measurement.json").write_bytes(json_bytes(receipt))
    print(json.dumps({"cases": len(cases), "seconds": receipt["total_elapsed_seconds"], "status": "PASS"}))


if __name__ == "__main__":
    main()
