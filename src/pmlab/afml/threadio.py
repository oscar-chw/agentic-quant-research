"""ch.20, the third kind of parallelism: I/O, where threads are the right answer and processes are not.

The book splits *compute* into molecules across processes because the interpreter lock serialises Python bytecode
(20.5, and `pmlab.afml.parallel`). Waiting is the opposite case. A parquet read spends its time inside pyarrow, a
backfill spends its time inside a socket, and both drop the lock while they wait, so threads in one process overlap
the waiting at no cost in memory and with no pickling of the result. A process pool would pay 0.8 GB and a round trip
through pickle to win nothing.

What is here:
- `map_threads`  the primitive: a callable over items, results in the order the items were given, never in the order
                 they finished, so nothing downstream can depend on which read came back first;
- `read_days`    days of one history kind read at once (`pmlab.store.read_day` per day);
- `read_parquet` arbitrary parquet paths, columns pushed down;
- `gather`       the same overlap for HTTP backfills and any other waiting call.

Order is the whole discipline. `pmlab.store.read_day` is deterministic per day, so a threaded read gives byte-identical
frames to a serial one; the only way to lose that is to reduce results as they arrive, which is why nothing here has a
completion-order path. Where numbers are later summed across days, the caller concatenates this list, in this order.

THREADS is deliberately larger than the core budget of docs/hardware.md: a thread waiting on a disk or a socket is not
running, so it does not count against `pmlab.afml.threadcaps.RESEARCH_CORES`. What *would* count is pyarrow's own
decode pool inside each read, which the pinned OMP/BLAS variables already cap.

This module lives under pmlab.afml because it is research-side: scripts/confirm.py registers every module of pmlab
outside afml and dashboard, and the confirmation period runs to 2026-10-24.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

THREADS = 8
"""Concurrent waiting calls. Threads blocked on a read or a socket hold no core, so this is above the worker cap."""

MIN_ITEMS = 3
"""Fewer items than this run here: two day files showed no gain over reading them one after the other, and a pool
that cannot pay for itself is a slower way to get the same answer."""


def map_threads(func, items, max_workers: int | None = None, threads: int = THREADS) -> list:
    """`func` over `items` on a thread pool, results in the order of `items`.

    `max_workers` is taken literally, including 0 and 1, which both mean "run it here": a caller sizing the pool
    from its own work (`len(paths) // 10`) must not get eight threads because the arithmetic came out zero. One
    worker, one item, or fewer than MIN_ITEMS runs everything in this thread — the debugging path
    `parallel.process_jobs_` is for compute, and the honest answer when there is nothing to overlap.

    The caller sees the first item's exception either way, but not at the same moment: serially the loop stops there,
    while the pool has already submitted every item and runs the rest to completion before the exception surfaces.
    Only the side effects differ; the answer does not.

    This only pays where the call really waits. Measured on this machine: `store.read_day` over 60 Binance kline days
    is **1.9x** (900 ms to 480 ms, cold and warm alike, same days both ways), because pyarrow drops the interpreter
    lock for the read and most of the conversion; `pandas.read_parquet` with a filter over 30 small event-day files
    is 0.95x-1.17x, because those few megabytes are already in the page cache and the time goes to pandas holding the
    lock. docs/performance.md keeps both numbers, and `pmlab.afml.models.load_events` stayed serial because of them.
    """
    items = list(items)
    n = min(threads if max_workers is None else int(max_workers), len(items))
    if n <= 1 or len(items) < MIN_ITEMS:
        return [func(x) for x in items]
    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(func, items))


def read_days(kind: str, days, max_workers: int | None = None) -> list:
    """One frame per day of a history kind, in the order given (`pmlab.store.read_day`).

    Byte-identical to reading them one after another: read_day is a pure function of the day's files, and the results
    keep the caller's order.

    Known and deferred: `pmlab.store.read_day`'s own cache evicts with `_cache.pop(next(iter(_cache)))` after the
    insert, which is two bytecodes apart and therefore a race between threads — two of them can choose the same
    victim (`KeyError`) or one can iterate while the other inserts (`RuntimeError: dictionary changed size`). The
    cache holds nine entries, so every read past the ninth evicts. It was not reproducible through this path (188
    days, six passes, nothing), only in the isolated shape, and the fix is a lock inside `read_day` —
    `src/pmlab/store.py` is registered code and hashed until 2026-10-24, so it waits, with the specification in
    tests/test_io_threads.py and in docs/performance.md.
    """
    from pmlab import store

    return map_threads(lambda d: store.read_day(kind, d), days, max_workers)


def read_parquet(paths, columns=None, max_workers: int | None = None) -> list:
    """One frame per parquet path, in the order given."""
    import pandas as pd

    return map_threads(lambda p: pd.read_parquet(p, columns=columns), paths, max_workers)


def gather(func, items, max_workers: int | None = None) -> list:
    """`func` over `items` for calls that wait on the network, results in the order of `items`.

    The same primitive as `map_threads`, named for what an HTTP backfill is doing: the wall time of N requests
    becomes roughly the slowest one rather than their sum, and nothing about the result depends on which returned
    first.

    **`func` has to be safe to call concurrently, and `pmlab.http.get_json` is not.** Its per-host spacing is an
    unsynchronised read-modify-write of a module dict and its `requests.Session` is shared, which is what its own
    first line is about: "Polite single-threaded HTTP: Cloudflare bans on concurrency, not volume". N threads through
    that check in the same instant issue N simultaneous requests, and the failure is a ban rather than an exception.
    `pmlab/http.py` is registered code, frozen until 2026-10-24; once its wait-and-stamp takes a lock, the backfill's
    `fetch_tape` can move here from its four processes.

    There is no timeout here and `Executor.__exit__` waits, so a hung socket holds this call — and a Ctrl-C raised in
    the main thread — until it returns. `func` carries its own timeout; `pmlab.http` passes `timeout=30`.
    """
    return map_threads(func, items, max_workers)
