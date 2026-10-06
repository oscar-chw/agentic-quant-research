"""Export the SYNTHETIC 100,000-message replay workload and its reference answer.

The workload is fixed by its parameters (docs/evidence/replay-benchmark.json):

* messages: seed 240914, 100,000 messages for one synthetic asset, a two-level
  book (one bid, one ask), 1 ms steps, a full snapshot every 1,000 messages;
* history: each message goes through replay.feed.Feed, which yields the
  keyframes and delta rows a recorder would persist;
* coverage: the single interval for one continuously arriving asset, closed
  at the last message: [first message, last message];
* queries: seed 240914 + 4, 64 query times (both ends twice, 60 sampled).

Two digests pin it, so a drifted generator fails here instead of producing a
plausible but different benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from replay_ref import DEFAULT_REF, books_output, import_reference, write_case

START = 1704067200000
SEED = 240914
EVENTS = 100_000
# RAW_SHA256: the message stream's digest. OUTPUT_SHA256: the reference answer's
# digest. Both are listed in docs/evidence/replay-benchmark.json.
RAW_SHA256 = "67d36ed8600fa58b09aef5e1e20e7ff3ba44bffe571e26f7cfbee7366843aebc"
OUTPUT_SHA256 = "279cc86976de46b9d48938599e4e255288a09b945dadca08dc3c2c7393583849"


def messages(n: int):
    """Seeded synthetic messages; each snapshot repeats the previous book."""
    rng = random.Random(SEED)
    previous = None
    for i in range(n):
        ts = START + i
        bid = previous if i % 1000 == 0 and previous is not None else 4700 + rng.randrange(100)
        ask = bid + 20
        price = lambda tick: f"{tick // 10000}.{tick % 10000:04d}"  # noqa: E731
        if i % 1000 == 0:
            msg = {"event_type": "book", "asset_id": "synthetic-A", "timestamp": str(ts),
                   "bids": [{"price": price(bid), "size": "12"}],
                   "asks": [{"price": price(ask), "size": "8"}]}
        else:
            entries = [("BUY", previous, "0"), ("SELL", previous + 20, "0"),
                       ("BUY", bid, "12"), ("SELL", ask, "8")]
            msg = {"event_type": "price_change", "timestamp": str(ts),
                   "price_changes": [{"asset_id": "synthetic-A", "side": side, "price": price(tick), "size": size}
                                     for side, tick, size in entries]}
        yield msg
        previous = bid


def build_history(n: int):
    from replay.feed import Feed  # noqa: PLC0415 -- importable once the reference is on the path

    feed, raw = Feed(), hashlib.sha256()
    frames, deltas = [], []
    for msg in messages(n):
        raw.update((json.dumps(msg, sort_keys=True, separators=(",", ":")) + "\n").encode())
        update = feed.apply(msg)
        assert not update.disagreed
        deltas.extend((d.ts_ms, d.side, d.tick, d.size_e2) for d in update.deltas)
        frames.extend((k.ts_ms, list(k.levels)) for k in update.keyframes)
    coverage = [(START, START + n - 1)]
    return frames, deltas, coverage, raw.hexdigest()


def sample_queries(n: int, frames, coverage):
    """Both ends twice plus 60 sampled message times, shuffled (seed 240914 + 4)."""
    start, end, interval = frames[0][0], coverage[-1][1], 1
    rng = random.Random(SEED + 4)
    queries = [end, start] + [start + i * interval for i in rng.sample(range(1, n - 1), 60)] + [start, end]
    rng.shuffle(queries)
    assert len(queries) == 64
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF, help="directory holding the reference replay package")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()

    reference = import_reference(args.ref)
    frames, deltas, coverage, raw_sha = build_history(EVENTS)
    if raw_sha != RAW_SHA256:
        sys.exit(f"generator drift: message stream sha256 {raw_sha} != published {RAW_SHA256}")
    assert len(deltas) == (EVENTS - EVENTS // 1000) * 4 and len(frames) == EVENTS // 1000
    queries = sample_queries(EVENTS, frames, coverage)

    books = reference.reconstruct_many(frames, deltas, queries, coverage, native=False)  # the reference, never C++
    scalar = [reference.reconstruct(frames, deltas, q, coverage) for q in queries]
    output = books_output(queries, books)
    if output != books_output(queries, scalar):
        sys.exit("reference disagreement: reconstruct_many != reconstruct")
    digest = hashlib.sha256(output).hexdigest()
    if digest != OUTPUT_SHA256:
        sys.exit(f"reference output sha256 {digest} != published {OUTPUT_SHA256}")

    args.workload.parent.mkdir(parents=True, exist_ok=True)
    with args.workload.open("w") as out:
        out.write(f"# SYNTHETIC replay workload: {EVENTS} messages, seed {SEED}, {len(queries)} queries\n")
        write_case(out, "synthetic-100000", frames, deltas, coverage, queries)
        out.write("end\n")
    args.expected.write_bytes(output)
    print(json.dumps({"synthetic": True, "messages": EVENTS, "keyframes": len(frames), "delta_rows": len(deltas),
                      "queries": len(queries), "message_sha256": raw_sha, "output_sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
