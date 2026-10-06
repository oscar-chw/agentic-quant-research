"""Measure Python and C++ replay, scalar and batch, in one session, one machine.

Every implementation runs in N fresh processes with R timed repetitions each,
interleaved round by round (rotating the order) so slow drift in machine state
lands on all of them alike. The headline is the median of per-process medians,
the aggregation every results file uses. Raw nanoseconds, machine
and compiler details go to a new bench/results-<date>.json; a results file
already there is never overwritten (the next run that day writes results-<date>-2.json).

Run through `make bench`, which builds the binary and the pybind11 module and
exports the workload. Six implementations by default: Python scalar and batch
(replay.book), C++ scalar and batch called from C++, and C++ called from Python
through the module (python-binding-batch: the module alone; python-native-batch:
replay.book.reconstruct_many(..., native=True) in the Python batch scope).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMPLS = ["python-scalar", "python-batch", "cpp-batch", "cpp-scalar", "python-native-batch", "python-binding-batch"]
RATIOS = [("python-scalar", "cpp-batch"), ("python-batch", "cpp-batch"), ("python-scalar", "python-batch"),
          ("cpp-scalar", "cpp-batch"), ("python-batch", "python-native-batch"),
          ("python-batch", "python-binding-batch"), ("python-native-batch", "cpp-batch"),
          ("python-binding-batch", "cpp-batch")]


def run(argv) -> dict:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    if done.returncode:
        sys.exit(f"benchmark process failed ({done.returncode}): {' '.join(argv)}\n{done.stderr}")
    return json.loads(done.stdout)


def first_line(argv) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout.splitlines()[0].strip()


def write_new(directory: Path, date: str, text: str) -> Path:
    """Write results-<date>.json, or -2, -3, ... if taken; never replace a committed file."""
    for n in range(1, 1000):
        path = directory / (f"results-{date}.json" if n == 1 else f"results-{date}-{n}.json")
        try:
            with path.open("x") as out:
                out.write(text)
            return path
        except FileExistsError:
            continue
    sys.exit(f"no free results-{date}-N.json name in {directory}")


def machine(python: str, cxx: str, cxxflags: str) -> dict:
    if sys.platform == "darwin":
        cpu = first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), platform.processor())
    return {"cpu": cpu, "os": platform.platform(), "arch": platform.machine(),
            "python": first_line([python, "-c", "import sys; print(sys.version)"]),
            "compiler": first_line([cxx, "--version"]), "cxxflags": cxxflags}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, help="directory holding the reference replay package")
    parser.add_argument("--python", default=sys.executable, help="interpreter that runs the reference")
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxxflags", required=True)
    parser.add_argument("--bench-bin", default="build/replay_bench")
    parser.add_argument("--workload", default="build/workload.txt")
    parser.add_argument("--expected", default="build/expected.json")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--impls", nargs="+", choices=IMPLS, default=IMPLS)
    parser.add_argument("--ext-cxxflags", help="flags the pybind11 module was built with")
    args = parser.parse_args()
    if args.processes < 5:
        sys.exit("report the median of at least 5 runs")

    commands = {
        "python-scalar": [args.python, "bench/py_bench.py", "--ref", args.ref, "--mode", "scalar"],
        "python-batch": [args.python, "bench/py_bench.py", "--ref", args.ref, "--mode", "batch"],
        "cpp-batch": [args.bench_bin, "--mode", "batch"],
        "cpp-scalar": [args.bench_bin, "--mode", "scalar"],
        "python-native-batch": [args.python, "bench/py_bench.py", "--ref", args.ref, "--mode", "native-batch"],
        "python-binding-batch": [args.python, "bench/py_bench.py", "--ref", args.ref, "--mode", "binding-batch"],
    }
    commands = {name: commands[name] for name in args.impls}
    impls = list(commands)
    raw = {name: [] for name in impls}
    meta = {}
    for round_index in range(args.processes):
        order = impls[round_index % len(impls):] + impls[:round_index % len(impls)]
        for name in order:
            result = run(commands[name] + ["--reps", str(args.reps), args.workload, args.expected])
            raw[name].append(result["nanoseconds"])
            meta[name] = {k: result[k] for k in ("queries", "delta_rows", "output_levels")}
        print(f"round {round_index + 1}/{args.processes} done", file=sys.stderr)

    summary = {}
    for name, processes in raw.items():
        medians = [statistics.median(p) / 1e9 for p in processes]
        flat = [ns / 1e9 for p in processes for ns in p]
        summary[name] = {"median_of_process_medians_seconds": statistics.median(medians),
                         "process_medians_seconds": medians, "min_seconds": min(flat), "max_seconds": max(flat)}
    head = {name: s["median_of_process_medians_seconds"] for name, s in summary.items()}
    ratios = {f"{a}_over_{b}".replace("-", "_"): head[a] / head[b] for a, b in RATIOS if a in head and b in head}
    if len({json.dumps(m, sort_keys=True) for m in meta.values()}) != 1:
        sys.exit(f"implementations disagree on the workload: {meta}")

    via_module = [name for name in impls if name in ("python-native-batch", "python-binding-batch")]
    today = dt.date.today().isoformat()
    report = {
        "schema": "replay-native-benchmark/v1", "date": today, "synthetic": True,
        "workload": {"messages": 100000, "seed": 240914, **next(iter(meta.values())),
                     "source": "python/export_workload.py (parameters and digests: docs/evidence/replay-benchmark.json)"},
        "scope": "per repetition: API validation/preparation + replay + 64 independent books + all sorted output "
                 "levels; workload loading and answer checking excluded for every implementation"
                 + (f"; {' and '.join(via_module)} (through the pybind11 module) also convert every input row to "
                    "C++ and every output level back to Python" if via_module else ""),
        "every_repetition_checked_against_reference_output": True,
        "processes_per_implementation": args.processes, "repetitions_per_process": args.reps,
        "aggregation": "median of per-process medians",
        "machine": {**machine(args.python, args.cxx, args.cxxflags),
                    **({"ext_cxxflags": args.ext_cxxflags} if args.ext_cxxflags else {})},
        # Absolute paths are machine-specific (and name the user): keep the
        # interpreter's basename and show the reference directory as <ref>.
        "commands": {name: " ".join([Path(argv[0]).name if argv[0].startswith("/") else argv[0]] +
                                    ["<ref>" if a == args.ref else a for a in argv[1:]])
                     for name, argv in commands.items()},
        "summary": summary, "ratios": ratios, "raw_nanoseconds": raw,
    }
    out = write_new(HERE, today, json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out.relative_to(ROOT)), "median_seconds": head, "ratios": ratios}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
