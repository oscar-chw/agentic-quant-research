"""Bounded synthetic scaling receipts; no market data or performance claims.

Run with the installed environment. Each case runs in a fresh child process.
"""
import argparse
import cProfile
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import resource
import subprocess
import sys
import time
import tracemalloc

from imc4_analysis.analyzer import analyze
from imc4_analysis.cli import json_bytes, write_report
from imc4_analysis.report import render


def fixture(event_count, instrument_count):
    assert event_count % 4 == 0
    symbols = [f"PRODUCT_{i:02d}" for i in range(instrument_count)]
    config = dict(type="config", schema="imc4-analysis/v1", label="Synthetic scaling probe",
                  data_kind="synthetic", currency="TEST_CCY", timestamp_unit="synthetic_tick",
                  start_timestamp=0, initial_cash="1000000", mark_max_age=instrument_count * 8,
                  markout_horizon=2, instruments={s:dict(initial_position=0, limit=1) for s in symbols})
    rows = [config]
    for cycle in range(event_count // 4):
        symbol = symbols[cycle % instrument_count]
        price = 100 + 10 * (cycle % instrument_count)
        timestamp = cycle * 4
        common = dict(sequence=0, instrument=symbol)
        rows.extend([dict(common, type="quote", timestamp=timestamp, mark=str(price)),
                     dict(common, type="fill", timestamp=timestamp+1, fill_id=f"{cycle}-buy", side="buy", units=1, price=str(price)),
                     dict(common, type="quote", timestamp=timestamp+2, mark=str(price+1)),
                     dict(common, type="fill", timestamp=timestamp+3, fill_id=f"{cycle}-sell", side="sell", units=1, price=str(price+1))])
    return ("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n").encode()


def oracle(report, events):
    """Independent integer closed-cycle ledger, not the production valuation code."""
    summary = report["summary"]
    assert summary["event_count"] == events
    assert summary["fill_count"] == events // 2
    assert summary["pnl"] == str(events // 4)
    assert summary["cash"] == summary["equity"] == str(1_000_000 + events // 4)
    assert all(q == 0 for q in summary["positions"].values())
    assert summary["breach_observations"] == summary["unavailable_valuation_snapshots"] == 0
    assert summary["markout_status_counts"]["available"] == events // 2 - 1
    assert summary["markout_status_counts"]["unavailable_future"] == 1
    for row in report["snapshots"]:
        index = row["index"]
        expected = 1_000_000 if not index else 1_000_000 + (index-1)//4 + int((index-1)%4 >= 2)
        assert row["equity"] == str(expected), (index, row["equity"], expected)
    for fill in report["fills"]:
        if fill["markout"]["status"] == "available":
            assert fill["markout"]["per_unit"] == ("1" if fill["side"] == "buy" else "0")


def run_case(events, instruments, detail, directory):
    data = fixture(events, instruments)
    raw_times = []
    report = None
    for repeat in range(3):
        report = None
        gc.collect()
        start = time.perf_counter()
        report = analyze(data, detail=detail)
        raw_times.append(time.perf_counter()-start)
        oracle(report, events)
    start = time.perf_counter()
    numerical = json_bytes(report)
    json_seconds = time.perf_counter()-start
    numerical_bytes = len(numerical)
    numerical = None
    start = time.perf_counter()
    html = render(report).encode()
    html_seconds = time.perf_counter()-start
    html_bytes = len(html)
    html = None
    summary = report["summary"]
    details = report["details"]
    identity = report["code_identity"]
    report = None
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    measured = analyze(data, detail=detail)
    traced_seconds = time.perf_counter()-start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    oracle(measured, events)
    output = dict(events=events, instruments=instruments, detail=detail, input_bytes=len(data), code_identity=identity,
                  input_sha256=hashlib.sha256(data).hexdigest(), analysis_seconds=raw_times,
                  analysis_events_per_second=[events/t for t in raw_times],
                  json_serialization_seconds=json_seconds, html_render_seconds=html_seconds,
                  json_serialized_bytes=numerical_bytes, html_serialized_bytes=html_bytes,
                  traced_analysis_seconds=traced_seconds, tracemalloc_peak_bytes=peak,
                  tracemalloc_current_bytes=current, summary=summary, details=details,
                  correctness="PASS: independent integer cycle ledger, retained snapshot values and signed markouts",
                  cache_state="fresh process per case; 3 sequential analyses of the same in-memory bytes; no result cache; no cold-disk claim")
    if events == 10000 and instruments == 8 and detail == "sampled":
        bundle = directory / "scale-demo"
        write_report(measured, bundle)
        output["saved_bundle"] = {p.name:p.stat().st_size for p in bundle.iterdir()}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    output["process_peak_rss_scope"] = "fresh case process peak including generation, analyses, serialization, tracemalloc and optional report write; not per-function allocation"
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", nargs=3)
    args = parser.parse_args()
    if args.case:
        events, instruments, detail = args.case
        output = run_case(int(events), int(instruments), detail, args.out)
        destination = args.out / f"case-{events}-{instruments}-{detail}.json"
        destination.write_bytes(json_bytes(output))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Profile the full-detail mode before making any further optimization choice.
    data = fixture(10000, 8)
    profile = cProfile.Profile()
    report = profile.runcall(analyze, data, detail="full")
    oracle(report, 10000)
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)
    (args.out/"profile-full-10000-8.txt").write_text(stream.getvalue())
    report = None
    cases = [(1000, 2, "sampled"), (5000, 2, "sampled"), (10000, 2, "sampled"),
             (10000, 2, "full"), (10000, 8, "sampled"), (10000, 8, "full")]
    results = []
    for events, instruments, detail in cases:
        if time.perf_counter()-started > 240:
            raise RuntimeError("benchmark time budget exhausted; completed receipts retained")
        subprocess.run([sys.executable, __file__, "--out", str(args.out), "--case", str(events), str(instruments), detail], check=True, timeout=60)
        results.append(json.loads((args.out/f"case-{events}-{instruments}-{detail}.json").read_text()))
    # Exact aggregate equality across retention modes on identical generated bytes.
    for instruments in (2, 8):
        pair = [r for r in results if r["events"] == 10000 and r["instruments"] == instruments]
        assert pair[0]["input_sha256"] == pair[1]["input_sha256"]
        assert pair[0]["summary"] == pair[1]["summary"]
    result = dict(platform=platform.platform(), python=sys.version, executable=sys.executable,
                  clock="time.perf_counter", data="deterministic synthetic closed-cycle log; no market data",
                  total_elapsed_seconds=time.perf_counter()-started, cases=results,
                  mode_parity="PASS: identical input hashes and exact full/sampled aggregate summaries",
                  limits="10,000-event / 16-MiB input cap unchanged; HTML <=200 rows per table; no extrapolation or speedup claim")
    (args.out/"measurement.json").write_bytes(json_bytes(result))
    print(json.dumps({"cases":len(results), "seconds":result["total_elapsed_seconds"], "status":result["mode_parity"]}))


if __name__ == "__main__":
    main()
