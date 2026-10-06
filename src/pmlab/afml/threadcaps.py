"""ch.20, the half of the thread question the book leaves implicit: native threads, and why they are pinned before
numpy loads.

The book's molecules are *processes* (20.5) because the interpreter lock serialises Python bytecode, so threads buy
concurrency and not parallelism for pure Python. The libraries underneath are the opposite case: BLAS behind numpy,
OpenMP behind scikit-learn and xgboost, and pyarrow's own pool are already threaded and release the lock, and each
one defaults to a thread per core. Four workers each taking twelve BLAS threads asks for 48 runnable threads on
twelve cores: they thrash, and the process that trades loses the core docs/hardware.md reserves for it. The binding
rule is therefore the product and not either factor:

    workers x threads_per_worker <= RESEARCH_CORES          (12 cores less the live app's one)

Every one of those libraries reads its thread count from the environment once, when it loads. `pin()` must run before
numpy is imported, so this module imports nothing but the standard library and is the first import of any entry point
that starts a pool. Children started with the *spawn* context inherit ``os.environ`` and read the pinned values at
their own numpy import, which is why `pmlab.afml.parallel.process_jobs` defaults to spawn; a forked child inherits an
already-initialised BLAS instead, and only `limit()` (threadpoolctl) can still move it.

`pin()` reports rather than promises: `numpy_loaded()` says whether it was already too late and `observed()` reads the
thread pools that actually exist, so a test can measure the cap instead of asserting the variable it set.

This module lives under pmlab.afml because everything here is research-side: scripts/confirm.py registers every
module of pmlab outside afml and dashboard, and the confirmation period runs to 2026-10-24.
"""
from __future__ import annotations

import contextlib
import os
import sys

MACHINE_CORES = 12
"""Cores docs/hardware.md describes (Apple M2 Max, 8 performance + 4 efficiency). A test ties this to the page."""

CORES = min(MACHINE_CORES, os.cpu_count() or MACHINE_CORES)
"""Cores to plan against: the page's number, or fewer when the machine really has fewer — a container with a CPU
quota, or CI. Planning 4 workers x 2 threads against 3 real cores would oversubscribe the very thing this module
exists to protect."""

LIVE_CORES = 1
"""Kept for the live paper app and its watchdog: "a research run must not starve the trading process"."""

RESEARCH_CORES = CORES - LIVE_CORES
"""What a research run may use at once, counting native threads as well as worker processes."""

WORKERS = 4
"""Worker processes for a research stage (docs/hardware.md): 4 x 0.8 GB of working set fits 32 GB."""

VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS")
"""The environment variables OpenMP, OpenBLAS, MKL, Apple's Accelerate and numexpr read when they load."""


def threads_for(workers: int = WORKERS, cores: int = RESEARCH_CORES) -> int:
    """Native threads each of `workers` processes may take without the product exceeding `cores`, or 1, whichever is
    larger. Above `cores` workers there is no such number, and `plan()["oversubscribed"]` is how a caller learns it."""
    return max(1, int(cores) // max(1, int(workers)))


def workers_for(threads: int = 1, cores: int = RESEARCH_CORES, cap: int = WORKERS) -> int:
    """Worker processes allowed when each takes `threads` native threads, never above `cap` (memory, not cores)."""
    return max(1, min(int(cap), int(cores) // max(1, int(threads))))


def oversubscribes(workers: int, threads: int, cores: int = RESEARCH_CORES) -> bool:
    """True when `workers` processes of `threads` native threads each would ask for more than `cores`."""
    return int(workers) * int(threads) > int(cores)


def numpy_loaded() -> bool:
    """Whether numpy is already imported in this process, i.e. whether pin() is too late to bind BLAS."""
    return "numpy" in sys.modules


_PINNED: dict[str, str] = {}
"""What this module put in the environment, so a later pin may lower its own value without touching an operator's."""


def pin(threads: int = 1, override: bool = False) -> dict[str, str]:
    """Set VARS to `threads` in os.environ and return what they now hold.

    A value that was already in the environment before this module touched it is the operator's and is left alone
    unless `override`; a value this module set itself is replaced, so a stage that wants a tighter cap than the last
    one gets it. Spawned children inherit the result; this process only binds it if numpy has not loaded yet
    (see `numpy_loaded`, `limit`).

    Which variable binds what, measured here (tests/test_parallel_threads.py): numpy on this machine is built against
    Apple's Accelerate, and of the five only VECLIB_MAXIMUM_THREADS changes the threads a matmul actually starts.
    OMP_NUM_THREADS is what scikit-learn's and xgboost's OpenMP read, and the OpenBLAS and MKL names are for a wheel
    built against either — so all five are set, and none of them is assumed to be the one that matters.
    """
    value = str(max(1, int(threads)))
    for name in VARS:
        current = os.environ.get(name)
        if override or not current or _PINNED.get(name) == current:
            os.environ[name] = value
            _PINNED[name] = value
    return {name: os.environ[name] for name in VARS}


def pin_report(threads: int = 1, override: bool = False) -> dict:
    """pin(), and whether it bound anything *here* or only what this process spawns from now on.

    `bound_here` is false once numpy has loaded: the variables still change, and a spawned child will read them, but
    this process's own BLAS was sized at import and only `limit()` can still move it. A caller that needs the cap in
    its own process — a forked worker, a stage that pins late — has to look at this rather than at the returned
    environment, which is identical either way.
    """
    too_late = numpy_loaded()
    return {"env": pin(threads, override), "threads": max(1, int(threads)), "bound_here": not too_late,
            "note": "numpy was already imported: this binds spawned children only, use limit() here" if too_late
                    else "bound before numpy loaded"}


def observed() -> list[dict]:
    """The native thread pools that exist in this process right now (threadpoolctl), one entry per library."""
    import threadpoolctl

    return threadpoolctl.threadpool_info()


def observed_threads() -> int | None:
    """The largest native thread pool threadpoolctl can see, or **None** when it can see nothing at all.

    None is not "one thread". numpy on this machine is built against Apple's Accelerate, which threadpoolctl does not
    report, so a process that has loaded numpy and nothing else answers None however many threads its matmul starts.
    A caller that compares this to a cap would be asserting nothing; measure `cores_used()` instead, which is what
    scripts/perf.py and tests/test_parallel_threads.py both do.
    """
    sizes = [int(p.get("num_threads", 1)) for p in observed()]
    return max(sizes) if sizes else None


def cores_used(seconds: float = 0.6, n: int = 1200) -> float:
    """CPU seconds this process burns per wall second while multiplying: how many cores it is actually taking.

    The supported way to check a cap. It is a measurement of the process, not a question to a library, so it works
    for Accelerate, OpenBLAS, MKL and OpenMP alike, and it is the quantity "oversubscribe" is about. The first
    product is outside the clock so the native pool exists before anything is counted.
    """
    import numpy as np
    import psutil
    import time

    a = np.random.default_rng(0).standard_normal((n, n))
    a @ a
    me = psutil.Process()
    c0, t0 = me.cpu_times(), time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ a
    c1, t1 = me.cpu_times(), time.perf_counter()
    return ((c1.user + c1.system) - (c0.user + c0.system)) / (t1 - t0)


@contextlib.contextmanager
def limit(threads: int = 1):
    """Cap every loaded native thread pool for the duration of the block (threadpoolctl).

    This is the only lever left once numpy has loaded — in the parent process, or in a forked child, where the
    environment was read before the fork.
    """
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=max(1, int(threads))):
        yield


def plan(workers: int = WORKERS, cores: int = RESEARCH_CORES) -> dict:
    """The thread budget of a stage: workers, threads each, the product, and the cores it leaves the live app.

    `threads_for` floors at one thread, so above `cores` workers the product no longer fits however few threads each
    takes — `plan(16)` on this machine is 16 of 11. `oversubscribed` says so rather than letting the caller read
    `total <= research_cores` off a dict that never promised it.
    """
    threads = threads_for(workers, cores)
    total = int(workers) * threads
    return {"workers": int(workers), "threads_per_worker": threads, "total": total,
            "research_cores": int(cores), "machine_cores": CORES, "reserved_for_live": CORES - int(cores),
            "oversubscribed": oversubscribes(workers, threads, cores)}
