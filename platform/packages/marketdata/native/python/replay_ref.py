"""Shared plumbing between the Python reference and the C++ port.

The Python module replay.book (packages/marketdata/replay) is the reference
implementation, and replay.feed turns messages into its history; this module
only imports them, reads and writes the text interchange format that
include/asof/replay/workload.hpp documents, and renders canonical output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# This folder sits at packages/marketdata/native/; the reference package is
# packages/marketdata/replay/. Override with --ref when running from anywhere else.
DEFAULT_REF = Path(__file__).resolve().parents[2]


def import_reference(ref: Path):
    """Import replay.book from the directory holding the replay package, refusing a missing path.

    A missing reference must fail loudly: a parity check that silently found no
    reference would report nothing and look like a pass.
    """
    ref = Path(ref).resolve()
    if not (ref / "replay" / "book.py").is_file():
        sys.exit(f"reference not found: {ref}/replay/book.py (pass --ref)")
    sys.path.insert(0, str(ref))
    from replay import book  # noqa: PLC0415 -- path is only known at run time
    return book


def canonical(value) -> bytes:
    """Canonical JSON: the exact bytes every implementation's output is compared and hashed as."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def books_output(queries, books) -> bytes:
    return canonical({"query_times_ms": list(queries), "books": [sorted(b.multiset()) for b in books]})


def _levels(levels) -> str:
    return " ".join([str(len(levels))] + [f"{s} {t} {z}" for s, t, z in levels])


def write_case(out, name, frames, deltas, coverage, queries) -> None:
    out.write(f"case {name}\n")
    for ts, levels in frames:
        out.write(f"K {ts} {_levels([(lv.side, lv.tick, lv.size_e2) for lv in levels])}\n")
    for ts, side, tick, size in deltas:
        out.write(f"D {ts} {side} {tick} {size}\n")
    for start, end in coverage:
        out.write(f"C {start} {'-' if end is None else end}\n")
    out.write("Q " + " ".join(str(q) for q in queries) + "\n")


def write_batch_books(out, books) -> None:
    out.write("batch books\n")
    for book in books:
        out.write(f"B {_levels(sorted(book.multiset()))}\n")


def write_error(out, prefix, error) -> None:
    out.write(f"{prefix} error {type(error).__name__} {error}\n")


def write_scalar_book(out, book) -> None:
    out.write(f"scalar book {_levels(sorted(book.multiset()))}\n")


def read_workload(path: Path, level_type):
    """Load a single-case workload into the reference's own Python types.

    Timing excludes this for every implementation.
    """
    frames, deltas, coverage, queries = [], [], [], []
    with open(path) as handle:
        for line in handle:
            tag, *fields = line.split()
            if tag == "K":
                ts, n, rest = int(fields[0]), int(fields[1]), [int(x) for x in fields[2:]]
                assert len(rest) == 3 * n
                frames.append((ts, [level_type(*rest[i:i + 3]) for i in range(0, len(rest), 3)]))
            elif tag == "D":
                deltas.append(tuple(int(x) for x in fields))
            elif tag == "C":
                coverage.append((int(fields[0]), None if fields[1] == "-" else int(fields[1])))
            elif tag == "Q":
                queries.extend(int(x) for x in fields)
    return frames, deltas, coverage, queries
