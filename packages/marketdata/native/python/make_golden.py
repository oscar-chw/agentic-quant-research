"""Record golden outcomes from the Python reference for the C++ differential test.

Each case is a history plus queries, and what the reference did with them:
reconstruct_many's books or its first error, and reconstruct's answer or
refusal for every query on its own. The random cases are built to hit what
hand-written cases miss: coverage gaps, overlapping and open epochs,
keyframes sharing a timestamp, duplicate and size-0 snapshot levels, deltas
tying with keyframes, repeated and out-of-range queries, and unsorted input.

Deterministic: the same reference produces the same file byte for byte, which
is how parity.py detects a reference that has drifted from the committed file.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from replay_ref import (DEFAULT_REF, import_reference, write_batch_books, write_case,
                        write_error, write_scalar_book)

RANDOM_CASES = 240


def fixed_cases(Level):
    """The fixtures of the Python suites, with every query list they use."""
    frames = [(10, [Level(0, 100, 10), Level(1, 110, 20)]),
              (30, [Level(0, 101, 12)]), (30, [Level(0, 102, 13), Level(1, 112, 14)])]
    deltas = [(10, 0, 100, 999), (15, 0, 100, 0), (15, 0, 100, 11), (20, 1, 110, 0),
              (30, 0, 102, 999), (35, 1, 113, 7), (100, 0, 999, 999)]
    yield "many-fixture", frames, deltas, [(10, None)], [35, 10, 15, 30, 20, 15, 10]
    yield "many-gap", frames, deltas, [(10, 20), (25, 40)], [40, 20, 30, 10, 25, 21]
    yield "many-overlap", frames, deltas, [(0, None), (25, 35)], [35, 20, 36, 30, 100, 25]
    yield "many-unsorted", list(reversed(frames)), deltas, [(10, None)], [35, 10]
    yield "reconnect", [(1000, [Level(0, 4000, 999)]), (3100, [Level(0, 4900, 100)]),
                        (5000, [Level(1, 9000, 999)])], \
        [(2900, 0, 4500, 999), (3100, 0, 4900, 999), (3200, 1, 5100, 200), (3200, 1, 5100, 0),
         (3200, 1, 5100, 300), (4000, 0, 4600, 999)], [(1000, 2000), (3000, None)], [3500, 3000, 2000, 5000]
    yield "zero-level-snapshot", [(1, [Level(0, 100, 0)])], [], [(1, 1)], [1]


def random_case(seed: int, reference):
    Level = reference.Level

    def answerable(frames, deltas, q, coverage):
        try:
            reference.reconstruct(frames, deltas, q, coverage)
        except reference.NotCovered:
            return False
        return True

    rng = random.Random(seed)
    steps = rng.randrange(5, 40)
    frames, deltas = [], []
    for t in range(steps):
        if rng.random() < 0.15 or (t == 0 and rng.random() < 0.7):
            for _ in range(1 + (rng.random() < 0.2)):  # sometimes two keyframes share t
                frames.append((t, [Level(rng.randrange(2), rng.randrange(100, 112), rng.randrange(0, 5))
                                   for _ in range(rng.randrange(0, 6))]))  # may repeat a key or carry size 0
        for _ in range(rng.randrange(0, 4)):
            deltas.append((t, rng.randrange(2), rng.randrange(100, 112), rng.choice([0, 0, 1, 2, 3, 50])))
    # Usually one wide epoch, plus up to two more that overlap it or leave gaps.
    coverage = [(rng.randrange(-2, 3), None if rng.random() < 0.5 else steps)] if rng.random() < 0.8 else []
    for _ in range(rng.randrange(0, 3)):
        start = rng.randrange(-2, steps)
        coverage.append((start, None if rng.random() < 0.3 else start + rng.randrange(0, steps)))
    queries = [rng.randrange(-3, steps + 3) for _ in range(rng.randrange(1, 12))]
    if rng.random() < 0.7:
        # Most cases keep only answerable queries, so the batch path returns
        # books often enough to be compared level by level, not just refused.
        queries = [q for q in queries if answerable(frames, deltas, q, coverage)] or queries
    if rng.random() < 0.1:
        rng.shuffle(deltas if deltas and rng.random() < 0.5 else frames)
    return f"random-{seed}", frames, deltas, coverage, queries


def record(out, reference, name, frames, deltas, coverage, queries) -> None:
    write_case(out, name, frames, deltas, coverage, queries)
    try:
        # native=False: the oracle must be pure Python even if ASOF_REPLAY_NATIVE=1 is set.
        write_batch_books(out, reference.reconstruct_many(frames, deltas, queries, coverage, native=False))
    except (reference.NotCovered, ValueError) as error:
        write_error(out, "batch", error)
    for q in queries:
        try:
            write_scalar_book(out, reference.reconstruct(frames, deltas, q, coverage))
        except reference.NotCovered as error:
            write_error(out, "scalar", error)
    out.write("end\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF, help="directory holding the reference replay package")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = import_reference(args.ref)
    with args.out.open("w") as out:
        out.write("# Golden outcomes recorded from the Python reference (replay.book). SYNTHETIC inputs.\n")
        out.write("# Regenerate: make golden. Do not edit by hand.\n")
        for case in fixed_cases(reference.Level):
            record(out, reference, *case)
        for seed in range(RANDOM_CASES):
            record(out, reference, *random_case(seed, reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
