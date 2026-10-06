"""ch.20  Multiprocessing and vectorization, sized for this machine (docs/hardware.md: 12 cores, 32 GB, 4 workers).

The book's engine is written for a 24-core server or a cluster; the logic is hardware-agnostic, so it is implemented
here with the worker count set by docs/hardware.md rather than by the book's default of 24.

- lin_parts          20.4.1  the N+1 indices that cut a list of atoms into equal molecules
- nested_parts       20.4.2  the same for a triangular double loop, so every molecule carries the same number of
                     atomic tasks (the positive root of the quadratic in the section, rounded to whole rows)
- cartesian_product  20.2    the grid of a dict of lists, its dimension read off the input at runtime
- expand_call        20.5.3  a job (a dict) becomes a call: pop the callback, pass the rest as keywords
- process_jobs_      20.5.1  the serial path, kept because a bug in a worker is hard to see (the chapter's note on
                     bugs that change when observed)
- process_jobs       20.5.2/20.5.5  molecules in a process pool, each result reduced as it arrives
- mp_pandas_obj      20.5.1  the whole thing: atoms -> molecules -> jobs -> outputs, stitched or reduced
- top_eigen_columns,
  get_pcs, pcs_from_files  20.6  P = Z W = sum_b Z_b W_b, one block of columns per molecule, summed on the fly, so
                     peak memory is one block rather than the whole matrix

The workers are processes, not threads, because the interpreter lock serialises Python bytecode. The libraries they
call are the other way round — BLAS, OpenMP and pyarrow are already threaded — so a pool caps its workers' *native*
threads before it starts them (pmlab.afml.threadcaps): four workers times their thread counts has to stay inside the
cores docs/hardware.md leaves the live app, and the product, not either factor, is what oversubscribes a machine.

Docstrings cite section numbers of the book; they do not reproduce its text.
"""
from __future__ import annotations

import copy
import itertools
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab.afml import threadcaps

WORKERS = threadcaps.WORKERS
"""Parallel workers on this machine: docs/hardware.md caps research runs at 4 of the 12 cores, so the live app keeps
a core and 4 x 0.8 GB of worker working set fits in 32 GB."""


def lin_parts(n_atoms: int, n_parts: int) -> list[int]:
    """20.4.1: the N+1 indices enclosing equal partitions of n_atoms atoms, N = min(n_parts, n_atoms).

    Consecutive molecules differ in size by at most one atom.
    """
    n_atoms = int(n_atoms)
    n = max(1, min(int(n_parts), n_atoms))
    return [int(x) for x in np.ceil(np.linspace(0, n_atoms, n + 1)).astype(int)]


def nested_parts(n_atoms: int, n_parts: int, upper_triang: bool = False) -> list[int]:
    """20.4.2: the row indices that cut the lower triangle {(i, j) | 1 <= j <= i <= N} into molecules of equal work.

    Rows 1..N hold N(N+1)/2 atomic tasks; a molecule of M should hold N(N+1)/(2M). Ending subset m at row r_m makes it
    (r_m + r_{m-1} + 1)(r_m - r_{m-1})/2 tasks, whose positive root is
    r_m = (-1 + sqrt(1 + 4 (r_{m-1}^2 + r_{m-1} + N(N+1)/M))) / 2, rounded to a whole row (so sizes deviate slightly).
    With upper_triang the first rows are the heaviest, so the differences are reversed.
    """
    n_atoms = int(n_atoms)
    n = max(1, min(int(n_parts), n_atoms))
    parts = [0.0]
    for _ in range(n):
        root = 1 + 4 * (parts[-1] ** 2 + parts[-1] + n_atoms * (n_atoms + 1.0) / n)
        parts.append((-1 + root ** 0.5) / 2.0)
    edges = np.round(parts).astype(int)
    if upper_triang:
        edges = np.append([0], np.cumsum(np.diff(edges)[::-1]))
    return [int(x) for x in edges]


def cartesian_product(dict0: dict) -> pd.DataFrame:
    """20.2: every combination of the values of dict0, one column per key, the number of dimensions read off dict0.

    The loops of the un-vectorised version are replaced by a compiled iterator (itertools.product), so the same call
    handles 3 or 100 keys without a change.
    """
    keys = list(dict0)
    rows = list(itertools.product(*[dict0[k] for k in keys]))
    return pd.DataFrame(rows, columns=keys)


def expand_call(job: dict):
    """20.5.3: turn a job (a dict holding the callback, its keyword arguments and one molecule) into the call."""
    job = dict(job)
    func = job.pop("func")
    return func(**job)


def report_progress(done: int, total: int, t0: float) -> str:
    """20.5.2: how much is finished and how many minutes are left at the rate observed so far."""
    share = done / total if total else 1.0
    minutes = (time.time() - t0) / 60.0
    left = minutes * (1 / share - 1) if share > 0 else float("nan")
    return f"{share * 100:.1f}% done after {minutes:.2f} min, {left:.2f} min to go"


def _seed(redux_init, redux_in_place: bool):
    """The accumulator a reduction starts from, or None for the book's own start (20.9: the first output becomes it).

    An explicit seed is what a caller needs when the reduction has a side effect — a progress line, an accounting
    total — that must run for every molecule including the first, which the book's start would skip.
    """
    if redux_init is None:
        return None
    return copy.deepcopy(redux_init) if redux_in_place else redux_init


def _reduce(out, result, redux, redux_args: dict | None, redux_in_place: bool):
    """20.5.5: fold one molecule's result into the running output as it arrives."""
    if out is None:
        # 20.9 copies the first output before it becomes the accumulator: an in-place reduction would otherwise
        # mutate whatever the first molecule returned, which can be an object its caller still holds.
        return copy.deepcopy(result) if redux_in_place else result
    if redux_in_place:
        redux(out, result, **(redux_args or {}))
        return out
    return redux(out, result, **(redux_args or {}))


def process_jobs_(jobs: list[dict], redux=None, redux_args: dict | None = None, redux_in_place: bool = False,
                  verbose: bool = False, redux_init=None):
    """20.5.1: run the jobs in this process, in order. The debugging path, and what a single-core run does."""
    t0, out, results = time.time(), _seed(redux_init, redux_in_place), []
    for done, job in enumerate(jobs, 1):
        result = expand_call(job)
        if redux is None:
            results.append(result)
        else:
            out = _reduce(out, result, redux, redux_args, redux_in_place)
        if verbose:
            print(report_progress(done, len(jobs), t0))
    return results if redux is None else out


REDUX_ORDERS = ("arrival", "atom")
"""How a reduction folds molecules: as they come back, or always in the order their atoms were submitted."""


def process_jobs(jobs: list[dict], num_threads: int = WORKERS, redux=None, redux_args: dict | None = None,
                 redux_in_place: bool = False, verbose: bool = False, mp_context: str = "spawn",
                 pin_threads: bool = True, redux_order: str = "arrival", redux_init=None):
    """20.5.2 and 20.5.5: molecules in a process pool, results reduced as they come back rather than collected first.

    With redux the parent holds one reduced output instead of one output per molecule, which is what keeps a large
    stitched result inside the 32 GB of docs/hardware.md.

    `redux_order` is the determinism switch, and it matters because float addition is not associative:

    - "arrival" is the book's, and the default: fold each molecule the moment it returns. Peak memory is one reduced
      output, and the answer's last digits depend on which worker finished first — measured at 1.1e-13 on this
      project's data, which is nothing and is still enough to make two runs of the same script disagree.
    - "atom" folds in submission order instead, holding back a molecule that finished early until its predecessors
      have gone in. The result is **bit-identical to `process_jobs_`**, run after run and whatever the worker count,
      and peak memory is the reduced output plus the molecules still out of order — bounded by the pool, not by the
      job. Every reduction whose number is published or compared uses this one.

    `pin_threads` caps the *native* thread pool of each worker before the pool starts, so `num_threads` processes of
    BLAS or OpenMP threads each stay inside the cores docs/hardware.md leaves the live app (pmlab.afml.threadcaps).
    It works because the default context is spawn: a spawned child reads os.environ at its own numpy import. A forked
    child inherits an already-initialised BLAS instead, so with mp_context="fork" the caller has to hold
    `threadcaps.limit()` around the work itself.
    """
    if redux_order not in REDUX_ORDERS:
        raise ValueError(f"redux_order is one of {REDUX_ORDERS}, got {redux_order!r}")
    if num_threads <= 1:
        return process_jobs_(jobs, redux, redux_args, redux_in_place, verbose=verbose, redux_init=redux_init)
    if pin_threads:
        if mp_context == "fork":
            # A forked child inherits an already-initialised BLAS, so the environment this would set reaches nobody
            # and each worker would run an uncapped BLAS on all twelve cores — the exact failure threadcaps exists
            # to prevent, and silent. The caller holds threadcaps.limit() around the work instead.
            raise ValueError("pin_threads cannot reach a forked child: use mp_context='spawn', or pass "
                             "pin_threads=False and hold pmlab.afml.threadcaps.limit() around the work")
        threadcaps.pin(threadcaps.threads_for(num_threads))
    t0, out = time.time(), _seed(redux_init, redux_in_place)
    results: list = [None] * len(jobs)
    held: dict[int, object] = {}                          # finished early, waiting for the molecules before them
    nxt = 0
    with ProcessPoolExecutor(num_threads, mp_context=mp.get_context(mp_context)) as pool:
        futures = {pool.submit(expand_call, job): i for i, job in enumerate(jobs)}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if redux is None:
                    results[futures[future]] = result      # submission order, not completion order
                elif redux_order == "arrival":
                    out = _reduce(out, result, redux, redux_args, redux_in_place)
                else:
                    held[futures[future]] = result
                    while nxt in held:
                        out = _reduce(out, held.pop(nxt), redux, redux_args, redux_in_place)
                        nxt += 1
                if verbose:
                    print(report_progress(done, len(jobs), t0))
        except BaseException:
            # a molecule raised: drop the queue rather than running every remaining job before the traceback
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return results if redux is None else out


def _stitch(out: list):
    """20.5.1, fourth step: one list, Series or DataFrame out of the molecular outputs, back in atom order.

    The molecules arrive in submission order (process_jobs), so concatenating them is already atom order; the sort is
    the book's and is stable, so rows sharing an index label keep the order their atoms had.
    """
    if not out:
        return out
    if isinstance(out[0], (pd.DataFrame, pd.Series)):
        return pd.concat(out).sort_index(kind="stable")
    return out          # anything else is the book's list of molecular outputs, one per molecule


def mp_pandas_obj(func, pd_obj: tuple, num_threads: int = WORKERS, mp_batches: int = 1, lin_mols: bool = True,
                  redux=None, redux_args: dict | None = None, redux_in_place: bool = False, verbose: bool = False,
                  mp_context: str = "spawn", pin_threads: bool = True, redux_order: str = "arrival",
                  redux_init=None, **kargs):
    """20.5.1 and 20.5.5: run func over molecules of pd_obj[1], passed under the name pd_obj[0].

    mp_batches > 1 cuts the atoms into more molecules than workers, so a heavy molecule does not leave the other
    workers idle; the queue drains instead of waiting on the worst job. num_threads = 1 runs everything here, which is
    how a failing job is debugged. With redux the outputs are reduced on the fly; without it they are stitched at the
    end, which is only safe while the outputs are small. `redux_order="atom"` makes the on-the-fly reduction
    bit-identical to the serial path (see process_jobs) and is what every published number uses.
    """
    atoms = pd_obj[1]
    if pd_obj[0] in kargs:
        raise ValueError(f"{pd_obj[0]!r} is the molecule's name and cannot also be a callback keyword")
    n_parts = max(1, int(num_threads) * max(1, int(mp_batches)))
    parts = lin_parts(len(atoms), n_parts) if lin_mols else nested_parts(len(atoms), n_parts)
    # nested_parts rounds its roots to whole rows, which can repeat an edge (nested_parts(3, 4) == [0, 2, 2, 3]); an
    # empty molecule is work for nobody and a None for the reduction to trip over.
    parts = [e for i, e in enumerate(parts) if i == 0 or e > parts[i - 1]]
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: atoms[parts[i - 1]:parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)
    out = process_jobs(jobs, num_threads=num_threads, redux=redux, redux_args=redux_args,
                       redux_in_place=redux_in_place, verbose=verbose, mp_context=mp_context, pin_threads=pin_threads,
                       redux_order=redux_order, redux_init=redux_init)
    return _stitch(out) if redux is None else out


def top_eigen_columns(eigen_values, tau: float) -> int:
    """20.6: the number M of leading eigenvectors whose eigenvalues carry at least a share tau of the total."""
    # a rank-deficient covariance gives tiny negative eigenvalues, which would make the cumulative share non-monotone
    lam = np.clip(np.sort(np.asarray(eigen_values, dtype=float))[::-1], 0.0, None)
    total = lam.sum()
    if total <= 0:
        return 0
    return int(min(len(lam), np.searchsorted(np.cumsum(lam) / total, tau) + 1))


def get_pcs(eigen_vectors: pd.DataFrame, molecule: list[str]) -> pd.DataFrame:
    """20.6: the part of P = Z W contributed by the column blocks in molecule, each block read on its own.

    Every file holds one block Z_b of columns; the block is multiplied by the rows of W named by its columns, so the
    blocks add up: P = sum_b Z_b W_b. Only one block is in memory at a time. The atoms are the file names themselves,
    so the book's extra fileNames argument has nothing to do here.
    """
    out = None
    for name in molecule:
        zb = pd.read_parquet(name)
        part = zb.dot(eigen_vectors.loc[zb.columns])
        out = part if out is None else out.add(part, fill_value=0)
    return out


def pcs_from_files(file_names: list[str], eigen_vectors: pd.DataFrame, num_threads: int = WORKERS,
                   mp_batches: int = 1, mp_context: str = "spawn") -> pd.DataFrame:
    """20.6: P = Z W with the columns split across molecules and the partial products summed as they arrive."""
    return mp_pandas_obj(get_pcs, ("molecule", list(file_names)), num_threads=num_threads, mp_batches=mp_batches,
                         redux=pd.DataFrame.add, redux_args={"fill_value": 0}, mp_context=mp_context,
                         eigen_vectors=eigen_vectors)


def write_column_blocks(z: pd.DataFrame, n_blocks: int, folder) -> list[str]:
    """20.6, the set-up: cut Z into n_blocks files of columns, the Z_b that get_pcs reads one at a time."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    edges, names = lin_parts(z.shape[1], n_blocks), []
    for i in range(1, len(edges)):
        path = folder / f"z{i - 1}.parquet"
        z.iloc[:, edges[i - 1]:edges[i]].to_parquet(path)
        names.append(str(path))
    return names
