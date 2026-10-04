"""The SYNTHETIC 100,000-message benchmark workload, end to end in pure Python.

replay.feed must turn the generator's messages into the published history, and
replay.book must answer its 64 queries with the published bytes: the same digest
the C++ port's parity check (`make parity`) compares against, here without C++.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native/python"))
from export_workload import EVENTS, OUTPUT_SHA256, RAW_SHA256, build_history, sample_queries  # noqa: E402
from replay_ref import books_output  # noqa: E402

from replay.book import reconstruct, reconstruct_many  # noqa: E402


def test_the_benchmark_workload_reproduces_the_published_digests():
    frames, deltas, coverage, raw_sha = build_history(EVENTS)
    assert raw_sha == RAW_SHA256
    assert (len(frames), len(deltas)) == (100, 399_600)
    queries = sample_queries(EVENTS, frames, coverage)
    output = books_output(queries, reconstruct_many(frames, deltas, queries, coverage, native=False))
    assert (len(output), hashlib.sha256(output).hexdigest()) == (2782, OUTPUT_SHA256)
    # The scalar reference agrees on a spread of the same queries (all 64 take about a second).
    sample = queries[::8]
    assert books_output(sample, [reconstruct(frames, deltas, q, coverage) for q in sample]) == books_output(
        sample, reconstruct_many(frames, deltas, sample, coverage, native=False))
