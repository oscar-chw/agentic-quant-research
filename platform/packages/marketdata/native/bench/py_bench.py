"""Time the Python reference on the exported workload.

The timed body is replay(): run the reference API
(scalar: one reconstruct per query; batch: one reconstruct_many), then build
every book's sorted level list. Loading the workload and checking the answer
are excluded, as they are for the C++ harness (replay_bench.cpp).

Two modes cross into C++ through the pybind11 module (build/python), with all
conversion inside the timed body: native-batch is reconstruct_many(...,
native=True) then the same sorted lists, the exact Python-batch scope;
binding-batch calls the module directly, whose answer already is those lists.

    py_bench.py --ref <dir holding replay/> --mode scalar|batch|native-batch|binding-batch --reps N <workload.txt> <expected.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from replay_ref import DEFAULT_REF, canonical, import_reference, read_workload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--mode", choices=["scalar", "batch", "native-batch", "binding-batch"], required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("workload", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    reference = import_reference(args.ref)
    frames, deltas, coverage, queries = read_workload(args.workload, reference.Level)
    expected = args.expected.read_bytes()
    if args.mode == "scalar":
        execute = lambda qs, cv: [reference.reconstruct(frames, deltas, q, cv) for q in qs]  # noqa: E731
    elif args.mode == "batch":
        execute = lambda qs, cv: reference.reconstruct_many(frames, deltas, qs, cv, native=False)  # noqa: E731
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "python"))
        import asof_replay_native  # noqa: PLC0415 -- built by `make ext`
        if args.mode == "native-batch":
            execute = lambda qs, cv: reference.reconstruct_many(frames, deltas, qs, cv, native=True)  # noqa: E731
        else:
            execute = None  # the module already answers with the sorted lists

    def replay():
        if execute is None:
            return asof_replay_native.reconstruct_many(frames, deltas, queries, coverage)
        return [sorted(book.multiset()) for book in execute(queries, coverage)]

    nanoseconds = []
    for _ in range(args.reps):
        start = time.perf_counter_ns()
        result = replay()
        nanoseconds.append(time.perf_counter_ns() - start)
        if canonical({"query_times_ms": list(queries), "books": result}) != expected:
            sys.exit("output differs from the exported reference answer")
    print(json.dumps({"impl": "python", "mode": args.mode, "queries": len(queries), "delta_rows": len(deltas),
                      "output_levels": sum(len(levels) for levels in result), "nanoseconds": nanoseconds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
