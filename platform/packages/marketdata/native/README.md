> Package reference. Start at the [root README](../../../README.md); the [docs index](../../../docs/README.md) lists every page.

# replay-native: C++20 order-book replay

Rebuilds the order book as it stood at each moment from recorded messages, in C++20.
It ports the Python reference [`replay.book`](../replay/book.py) and has to match it byte for byte before any timing counts.

The port covers the scalar `reconstruct` and the one-sweep batch `reconstruct_many`. All benchmark data is
**SYNTHETIC**. A pybind11 module exposes it to Python as an opt-in path of
`replay.book.reconstruct_many`.

**For a Python caller, the port is ~2.3× faster than Python batch, end to end**
(`native=True` 8.482 ms, the module alone 8.571 ms, vs 19.524 ms; conversion included). The 25.10× figure is C++ batch called from C++ against Python batch (`python_batch_over_cpp_batch`).
This is order-book replay on a SYNTHETIC workload; it is not part of the
repository's research loop.

Written Oct 2026 with AI coding agents under Oscar's design and review.

## The problem

To answer "what did the book look like at time T", replay applies
aggregate-size deltas to the last keyframe inside the coverage epoch that
contains T. It refuses (`NotCovered`) when nothing was being recorded at T, or
when no keyframe exists in that epoch. The Python batch path already beats its
scalar loop by 41.66× (813.541 ms → 19.524 ms, 100,000 synthetic messages,
64 queries; Benchmark below). This port finds out what a native implementation
of the same contract costs, with no change in behaviour.

## Approach (methods and algorithms)

- **Same semantics, case by case.** Each delta replaces the aggregate size at
  its level, and size 0 deletes the level. A keyframe already holds every
  change at its own timestamp, so equal-time deltas are skipped. When coverage
  intervals overlap, the latest containing epoch wins, and endpoints are
  inclusive. In a snapshot, the last duplicate wins and size-0 rows are kept.
  Queries are preflighted in caller order, so the first refusal reported is the
  same one Python reports, with identical message text. The batch path rejects
  unsorted histories (`std::invalid_argument`, Python's `ValueError`). The
  scalar path tolerates them, as Python's does. Results are independent books.
  Python models no crossed-book or locked-book check, so the port adds none.
- **Python binding.** [`bindings/py_replay.cpp`](bindings/py_replay.cpp) takes
  `replay.book`'s own inputs and returns each book's sorted `(side, tick, size_e2)`
  tuples. `replay.book.reconstruct_many(..., native=True)`, or
  `ASOF_REPLAY_NATIVE=1` with `native` unset, routes through it and wraps the
  answer in `Book`s; refusals come back as `replay.book.NotCovered` with the same
  text. Rows are converted with the CPython API, not pybind11's generic list
  casters, because conversion is most of the cost. Values C++ cannot hold
  exactly raise instead of being coerced: a float or NaN query time raises
  `TypeError`; an int outside int64 (int32 for ticks) or a side other than 0/1
  raises `ValueError`.
- **Layout.** A book is one sorted `std::vector<Level>` keyed by (side, tick).
  Real books hold tens of levels, so a binary search plus a short move beats a
  hash map. Copying a snapshot is one allocation, and the sorted output costs
  nothing extra. Prices are integer ticks and sizes are integer hundredths, so
  there are no floats and parity needs no tolerance.
- **Batch algorithm.** Validate, then preflight every query (O(Q·C + Q log K)).
  Then visit the queries in time order with one forward delta cursor: O(D)
  replay in total, against O(Q·D) for the scalar loop.

## Results

**Parity** (`make parity`). The C++ batch and scalar drivers both write the
same 2,782 bytes as the Python reference, sha256 `279cc869…3849`, the
`output_sha256` listed with the workload's parameters in
[`docs/evidence/replay-benchmark.json`](../../../docs/evidence/replay-benchmark.json).
It was first recorded from the Python implementation this package shipped until
2026-10-04; `replay.book`, written fresh to the same contract, reproduces it, and
[`../tests/test_workload.py`](../tests/test_workload.py) pins it without C++. The exported
message stream is pinned the same way (sha256 `67d36ed8…aebc`). [`tests/golden/reference_cases.txt`](tests/golden/reference_cases.txt)
holds 247 cases recorded from the Python reference: 7 hand-written fixtures
and 240 randomized histories with gaps, overlapping epochs, keyframes
sharing a timestamp, duplicate snapshot rows and unsorted input. Across those
cases, C++ matches every recorded batch outcome and every per-query scalar
outcome. `make parity` re-records the file to confirm it is current; recorded
from `replay.book`, every case came out unchanged and only the header comment
moved.

**Failure probe.** Changing `lower_bound` to `upper_bound` in `Book::apply`
(an off-by-one in level lookup) turned 13 of 27 C++ tests red, and both parity
checks failed. Skipping one fewer equal-time delta after a keyframe (`<=` to
`<`) turned 7 tests red. Both mutants were reverted.

**Benchmark** (`make bench`, or `make native-bench` from the repository root).
SYNTHETIC workload: 100,000 messages, 399,600 delta rows, 100 keyframes, 64
queries, 128 output levels. Each implementation ran in 5 fresh processes with
5 timed repetitions each, interleaved round by round, and every repetition was
checked against the reference output. Figures are medians of the per-process
medians, truncated rather than rounded. All six come from one session on
2026-10-04 on an Apple M2 Max with Python 3.11.15 and Apple clang 21,
`-std=c++20 -O2`; the module is a release build (`-DNDEBUG`, recorded as
`ext_cxxflags`; without it pybind11 checks the GIL on every reference-count
change). Raw data: [`bench/results-2026-10-04.json`](bench/results-2026-10-04.json).

| implementation | median | C++ batch is faster by |
|---|---:|---:|
| Python scalar (`reconstruct` × 64) | 813.541 ms | 1046.07× |
| Python batch (`reconstruct_many`) | 19.524 ms | 25.10× |
| C++ scalar (`reconstruct` × 64) | 10.897 ms | 14.01× |
| C++ via Python: the module alone | 8.571 ms | 11.02× |
| C++ via Python: `replay.book.reconstruct_many(..., native=True)` | 8.482 ms | 10.90× |
| C++ batch (`reconstruct_many`), called from C++ | 0.777 ms | 1× |

From Python the port is ~2.3× faster than Python batch: 2.30× through
`native=True` and 2.27× for the module alone, two figures whose per-process
medians overlap, so their order is noise. A C++ caller gets 25.10× over Python batch. Derived by subtraction, not measured
directly: of the 8.482 ms through `replay.book`, the C++ replay as timed from
C++ is 0.777 ms, leaving 7.705 ms (90.8%) for everything on the Python side
of the call, which includes argument validation and preparation as well as
converting the data across the binding.

Timing scope, the same for every implementation. Timed:
API validation and preparation, the replay, 64 independent books, every
sorted output level, and freeing the previous repetition's output. The
via-Python rows also time converting all 399,600 delta rows, 100 keyframes
and 64 queries to C++ and every output level back. Not timed: loading the
workload and checking the answer.

**History: the ratios fell because the Python reference got faster.** The
2026-10-03 files, [`bench/results-2026-10-03.json`](bench/results-2026-10-03.json)
and [`bench/results-binding-2026-10-03.json`](bench/results-binding-2026-10-03.json),
timed the same workload against the Python implementation this package
shipped then. Its batch path took 47.641 and 48.529 ms, so C++ batch was
60.01× faster and `native=True` 5.48×. `replay.book` replaced it on
2026-10-04; its batch path takes 19.524 ms. The C++ timings barely moved
(batch 0.793 and 0.790 ms then, 0.777 ms now), so 60.01× → 25.10× and
5.48× → ~2.3× measure a faster baseline, not a slower port. The old files are
kept unchanged apart from a `history_note` field.

**Python-side parity** (`make pytest`, [`tests/test_binding.py`](tests/test_binding.py)).
20 tests compare the module and the `native=True` path with pure-Python
`reconstruct_many` and `reconstruct`. They cover the 6 fixtures and 240
recorded random histories above, and hypothesis-generated histories: sorted,
unsorted, and across the full int64/int32 range. The generated histories are
biased towards answerable queries, and each property asserts a minimum share
of examples that compare non-empty books. With hypothesis 6.168.3 that share
is 202 of 400 sorted and 85 of 150 full-range; of 150 unsorted, 16 compare
books and 109 are refused as unsorted. To reprint the counts:
`PYTHONPATH=build/python:..:python:tests python -c "import test_binding as t; print(t.run_property(t.histories(**t.SMALL), 400))"`
(`t.WIDE` for full range, `sort=False` for unsorted).
Books must match level for level and in order, and refusals must match in
type and message. One regression test uses a level whose attribute getter
clears the caller's input lists during conversion. The module must give
`replay.book`'s answer and must not read freed memory; the first version of the
binding aborted the interpreter on this test. Running the replay suite
(`packages/marketdata/tests`) with `ASOF_REPLAY_NATIVE=1` gives 1 failed,
39 passed. The failure is the test that passes fractional and NaN query times,
which the native path refuses by design, so pure Python stays the default.

**Binding probe.** Reversing the order in which the module returns a book's
levels turned 9 of the 20 Python tests red; the change was reverted.

## How to run (under 5 minutes)

From the repository root, `make native-build native-test` builds everything
and runs `test`, `parity` and `pytest` below; `bash scripts/check.sh` does the
same when clang++ and pybind11 are present. Here, with `PYTHON` naming an
interpreter that has pybind11, hypothesis and pytest:

```sh
make test           # C++ unit + golden differential tests; clang++ only, no Python
make parity         # C++ vs Python on the 100,000-message workload
make ext            # the pybind11 module -> build/python/
make pytest         # the module vs pure Python (fixtures + hypothesis)
make bench          # all six timings in one session -> a new bench/results-<date>[-N].json (about 30 s)
```

The Python targets import the reference package `replay` from `..`
(`packages/marketdata`). Pass `REF=<directory holding replay/>` to use
another one.

## Architecture

```
include/asof/replay/  book.hpp (Level, Book) · replay.hpp (reconstruct, reconstruct_many,
                      NotCovered) · workload.hpp (text interchange, canonical JSON)
src/                  the three implementations
tests/                check.hpp (tiny harness) · test_book · test_replay (the hand-written
                      fixtures) · test_golden (differential) · golden/reference_cases.txt
tools/replay_cli.cpp  parity driver: workload in, canonical JSON out
bench/                replay_bench.cpp · py_bench.py · run_bench.py · results-*.json
python/               export_workload.py (generator → workload + reference answer)
                      make_golden.py · parity.py · replay_ref.py (shared plumbing)
bindings/py_replay.cpp  pybind11 module asof_replay_native, built by `make ext`;
                      replay.book.reconstruct_many(..., native=True) calls it
tests/test_binding.py module vs pure Python (pytest + hypothesis)
```

## Limits

- SYNTHETIC only: one instrument, a two-level book, 1 ms steps. Deeper books
  and real feed data are not measured.
- Called from Python, the Python side of the call (by subtraction: validation,
  preparation and crossing the binding) takes 90.8% of the time above and is
  paid on every call; nearly all of what crosses is the 399,600 delta rows.
  Keeping the history in C++ across calls could avoid that; nothing here does
  it yet.
- Query times are `int64` milliseconds. Python also accepts fractional and NaN
  query times, which the Python suite exercises; the port refuses them, so the
  native path is opt-in.
- The module is built with `-undefined dynamic_lookup` on macOS and has only
  been built and run on macOS (arm64).
- The timings come from one machine, one session per JSON file. OS caches and
  CPU frequency were not controlled.

## What I learned

Lessons from what this folder records (confirmed by the author, 2026-10-03).

- **Where the time goes decides what a port is worth.** A Python caller gets
  ~2.3× over Python batch; only a C++ caller sees 25.10× over Python batch. By subtraction,
  7.705 of the 8.482 ms (90.8%) is spent on the Python side of the call. Source:
  [bench/results-2026-10-04.json](bench/results-2026-10-04.json) and Results above.
  The same held against the Python reference of 2026-10-03 (5.48×, 61.40×, 91%:
  [bench/results-binding-2026-10-03.json](bench/results-binding-2026-10-03.json)).
- **A faster path that accepts less input stays opt-in.** The port takes
  `int64` millisecond query times; Python also accepts fractional and NaN ones,
  which a replay test exercises, so pure Python remains the default.
  Source: Limits above.
- **Byte-identical canonical output is the parity test that scales.** One
  digest over 2,782 bytes of canonical JSON on the 100,000-message workload,
  plus 247 recorded reference cases, checks the whole port at once instead of
  spot values. Source: Results above and `tests/golden/reference_cases.txt`.
