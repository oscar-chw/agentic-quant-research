"""AFML ch.20 on this machine: partitions that balance, molecules that run in a pool, outputs reduced on the fly.

The jobs are module-level functions because the pool starts its workers with spawn, as scripts/afml_events.py does.
"""
import itertools
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from pmlab.afml import parallel as P

pytestmark = pytest.mark.case


def _sizes(edges):
    return np.diff(edges)


def _triangle_work(edges, n, upper_triang=False):
    """Atomic tasks in each molecule: row i of a lower triangle has i tasks, of an upper triangle n - i + 1."""
    return [sum((n - i + 1) if upper_triang else i for i in range(a + 1, b + 1))
            for a, b in zip(edges[:-1], edges[1:])]


_SHARED = {"seed": 1}


def _shared_dict(molecule):
    """Returns an object the caller keeps: an in-place reduction must not fold into this very dict."""
    a = int(molecule[0])
    return _SHARED if a == 0 else {**_SHARED, a: a}


def _slow_first_molecule(molecule):
    molecule = list(molecule)
    if molecule and molecule[0] == 0:
        time.sleep(0.4)                                      # finishes last, and must still be first in the answer
    return molecule


def _repeated_label(molecule):
    """Many atoms share a label, so a stitch that is not a stable sort shuffles them."""
    molecule = list(molecule)
    return pd.Series(molecule, index=[a % 4 for a in molecule])


def _block_or_none(molecule, columns):
    """Shaped like get_pcs: an empty molecule has nothing to contribute and returns None, which a reduction adds."""
    molecule = list(molecule)
    return None if not molecule else pd.DataFrame(1.0, index=[str(a) for a in molecule], columns=columns)


def _boom_or_sleep(molecule):
    if list(molecule) == [0]:
        raise ValueError("molecule 0")
    time.sleep(2.0)
    return list(molecule)


def _double_molecule(molecule, scale):
    """A vectorised job: the whole molecule at once, as Section 20.3 asks (array work inside a parallel worker)."""
    return pd.Series(np.asarray(molecule, dtype=float) * scale, index=list(molecule))


def _molecule_size(molecule):
    return pd.Series(len(molecule), index=list(molecule))


def _ones_frame(molecule, columns):
    return pd.DataFrame(float(len(molecule)), index=["total"], columns=columns)


def _molecule_squares(molecule):
    return {a: a * a for a in molecule}


def test_linear_partitions_tile_the_atoms_and_differ_by_at_most_one_atom():
    for n_atoms, n_parts in [(20, 6), (1_000, 4), (97, 12), (5, 5)]:
        edges = P.lin_parts(n_atoms, n_parts)
        assert edges[0] == 0 and edges[-1] == n_atoms
        assert len(edges) == n_parts + 1
        assert edges == sorted(edges)
        assert _sizes(edges).max() - _sizes(edges).min() <= 1


def test_there_are_never_more_molecules_than_atoms():
    assert P.lin_parts(3, 10) == [0, 1, 2, 3]


def test_nested_partitions_give_every_molecule_the_same_triangular_work():
    n, m = 200, 6
    edges = P.nested_parts(n, m)
    assert edges[0] == 0 and edges[-1] == n and len(edges) == m + 1
    work = _triangle_work(edges, n)
    assert sum(work) == n * (n + 1) // 2
    target = n * (n + 1) / (2 * m)
    assert max(abs(w - target) for w in work) <= n           # rounding to whole rows costs at most one row


def test_nested_partitions_beat_a_linear_one_on_a_triangular_double_loop():
    n, m = 200, 6
    nested = max(_triangle_work(P.nested_parts(n, m), n))
    linear = max(_triangle_work(P.lin_parts(n, m), n))
    target = n * (n + 1) / (2 * m)
    assert nested < 1.1 * target                             # the heaviest molecule decides the elapsed time
    assert linear > 1.7 * target                             # the last linear molecule carries the widest rows


def test_the_first_nested_edge_is_the_closed_form_positive_root():
    n, m = 200, 6
    r1 = (-1 + np.sqrt(1 + 4 * n * (n + 1) / m)) / 2
    assert P.nested_parts(n, m)[1] == round(r1)
    assert P.nested_parts(n, m)[0] == 0                      # r_m reduces to r_1 when the previous edge is 0


def test_the_upper_triangular_partition_balances_the_heavy_first_rows():
    n, m = 200, 6
    edges = P.nested_parts(n, m, upper_triang=True)
    assert edges[0] == 0 and edges[-1] == n
    work = _triangle_work(edges, n, upper_triang=True)
    target = n * (n + 1) / (2 * m)
    assert max(abs(w - target) for w in work) <= n
    assert _sizes(edges)[0] < _sizes(edges)[-1]              # heavier rows come first, so fewer of them per molecule


def test_the_cartesian_product_reads_its_dimensions_off_the_input():
    grid = {"a": [0, 1], "b": [0, 1], "c": [0, 1]}
    out = P.cartesian_product(grid)
    assert list(map(tuple, out.to_numpy())) == [(a, b, c) for a in grid["a"] for b in grid["b"] for c in grid["c"]]
    assert P.cartesian_product({f"x{i}": [0, 1] for i in range(10)}).shape == (1024, 10)


def test_a_job_dictionary_becomes_a_call():
    assert P.expand_call({"func": _double_molecule, "molecule": [1, 2], "scale": 3}).tolist() == [3.0, 6.0]


def test_the_progress_report_states_the_share_done_and_the_time_left():
    """20.5.2: minutes left = elapsed x (1 / share - 1). One minute in at 25% done, three minutes remain."""
    message = P.report_progress(1, 4, time.time() - 60)
    assert "25.0% done" in message and "3.00 min to go" in message
    assert "0.00 min to go" in P.report_progress(4, 4, time.time() - 60)


def test_molecules_in_a_pool_equal_the_single_process_result():
    atoms = list(range(40))
    serial = P.mp_pandas_obj(_double_molecule, ("molecule", atoms), num_threads=1, scale=2.0)
    parallel = P.mp_pandas_obj(_double_molecule, ("molecule", atoms), num_threads=2, scale=2.0)
    assert serial.tolist() == [2.0 * a for a in atoms]
    pd.testing.assert_series_equal(serial, parallel)


def test_more_batches_than_workers_makes_smaller_molecules():
    atoms = list(range(24))
    one = P.mp_pandas_obj(_molecule_size, ("molecule", atoms), num_threads=1, mp_batches=1)
    many = P.mp_pandas_obj(_molecule_size, ("molecule", atoms), num_threads=1, mp_batches=6)
    assert set(one) == {24} and set(many) == {4}


def test_on_the_fly_reduction_equals_reducing_the_collected_outputs():
    atoms, columns = list(range(30)), ["a", "b"]
    edges = P.lin_parts(30, 10)
    parts = [_ones_frame(atoms[a:b], columns) for a, b in zip(edges[:-1], edges[1:])]
    total = sum(parts[1:], parts[0])
    reduced = P.mp_pandas_obj(_ones_frame, ("molecule", atoms), num_threads=2, mp_batches=5,
                              redux=pd.DataFrame.add, redux_args={"fill_value": 0}, columns=columns)
    pd.testing.assert_frame_equal(total, reduced)
    assert reduced.loc["total", "a"] == 30                   # every atom counted once, in whatever order they return


def test_an_in_place_reduction_updates_the_running_output():
    jobs = [{"func": _molecule_squares, "molecule": [i]} for i in range(4)]
    out = P.process_jobs_(jobs, redux=dict.update, redux_in_place=True)
    assert out == {0: 0, 1: 1, 2: 4, 3: 9}


def test_the_variance_share_picks_the_leading_eigenvalues():
    assert P.top_eigen_columns([4.0, 3.0, 2.0, 1.0], 0.0) == 1
    assert P.top_eigen_columns([4.0, 3.0, 2.0, 1.0], 0.4) == 1
    assert P.top_eigen_columns([4.0, 3.0, 2.0, 1.0], 0.7) == 2
    assert P.top_eigen_columns([4.0, 3.0, 2.0, 1.0], 1.0) == 4


def test_column_blocks_project_to_the_same_components_as_the_whole_matrix(tmp_path):
    rng = np.random.default_rng(20)
    z = pd.DataFrame(rng.normal(size=(60, 12)), columns=[f"c{i}" for i in range(12)])
    values, vectors = np.linalg.eigh(np.cov(z.to_numpy(), rowvar=False))
    order = np.argsort(values)[::-1]
    m = P.top_eigen_columns(values, 0.95)
    assert 0 < m < 12                                        # tau leaves something out, so the test is not trivial
    w = pd.DataFrame(vectors[:, order[:m]], index=z.columns)
    names = P.write_column_blocks(z, 4, tmp_path)
    assert len(names) == 4
    blocks = P.pcs_from_files(names, w, num_threads=2)
    pd.testing.assert_frame_equal(z.dot(w).sort_index(), blocks.sort_index()[list(w.columns)])


def test_a_failing_molecule_is_visible_in_the_serial_path():
    """20.5.1: the reason to keep num_threads = 1 is that a worker's traceback is hard to read."""
    def boom(molecule):
        raise ValueError(f"molecule {list(molecule)}")

    with pytest.raises(ValueError, match=r"molecule \[0, 1\]"):
        P.process_jobs_([{"func": boom, "molecule": [0, 1]}])


def test_a_job_that_cannot_be_pickled_never_reaches_a_worker():
    """20.3 and 20.5.4: workers share no memory, so every job crosses to them pickled."""
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        P.mp_pandas_obj(lambda molecule: list(molecule), ("molecule", [0, 1, 2, 3]), num_threads=2)


def test_a_bound_method_pickles_without_help_from_the_engine():
    """20.5.4: the reduction the book registers for bound methods is native in Python 3."""
    assert pickle.loads(pickle.dumps(pd.Series([1, 2]).sum))() == 3


def test_the_engine_does_not_care_what_the_callback_returns():
    out = P.mp_pandas_obj(lambda molecule: list(molecule), ("molecule", list(range(6))), num_threads=1, mp_batches=3)
    assert sorted(itertools.chain.from_iterable(out)) == list(range(6))


def test_molecules_come_back_in_atom_order_however_long_each_one_took():
    """20.5.2 reduces results as they arrive, but the stitched answer is the book's: atom order, not finish order."""
    atoms = list(range(12))
    out = P.mp_pandas_obj(_slow_first_molecule, ("molecule", atoms), num_threads=4)
    assert out == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    many = list(range(60))
    labelled = P.mp_pandas_obj(_repeated_label, ("molecule", many), num_threads=4, mp_batches=3)
    assert labelled.index.tolist() == sorted(a % 4 for a in many)
    assert labelled.tolist() == sorted(many, key=lambda a: (a % 4, a))   # stable: atom order inside a label


def test_a_callback_keyword_cannot_take_the_molecules_name():
    with pytest.raises(ValueError, match="molecule"):
        P.mp_pandas_obj(_molecule_size, ("molecule", [0, 1, 2, 3]), num_threads=1, mp_batches=2, molecule=[99])


def test_a_triangular_partition_never_dispatches_an_empty_molecule():
    """nested_parts rounds its roots, so its edges can repeat: nested_parts(3, 4) == [0, 2, 2, 3]."""
    assert P.nested_parts(3, 4) == [0, 2, 2, 3]
    empty = [(n, k) for n in range(1, 40) for k in (2, 4, 6, 12)
             if any(b == a for a, b in zip(P.nested_parts(n, k)[:-1], P.nested_parts(n, k)[1:]))]
    assert empty, "the rounding has to be able to repeat an edge, or this test proves nothing"
    for n, k in empty[:8]:
        sizes = P.mp_pandas_obj(_molecule_size, ("molecule", list(range(n))), num_threads=1, mp_batches=k,
                                lin_mols=False)
        assert sizes.index.tolist() == list(range(n))        # every atom, once
        assert sizes.min() > 0                               # and no molecule of no atoms
    # a callback shaped like get_pcs returns None for a molecule with nothing in it, and the reduction then dies
    frames = P.mp_pandas_obj(_block_or_none, ("molecule", list(range(3))), num_threads=1, mp_batches=4, lin_mols=False,
                             redux=pd.DataFrame.add, redux_args={"fill_value": 0}, columns=["a"])
    assert sorted(frames.index) == ["0", "1", "2"] and frames["a"].tolist() == [1.0, 1.0, 1.0]


def test_a_molecule_that_raises_cancels_the_queue_instead_of_draining_it():
    jobs = [{"func": _boom_or_sleep, "molecule": [i]} for i in range(16)]
    started = time.time()
    with pytest.raises(ValueError, match="molecule 0"):
        P.process_jobs(jobs, num_threads=2)
    assert time.time() - started < 8.0                        # 15 x 2 s of sleeping would be ~16 s if it drained


def test_the_first_output_is_copied_before_an_in_place_reduction_mutates_it():
    """20.9 deep-copies the first output: the accumulator must not be an object the callback still owns."""
    jobs = [{"func": _shared_dict, "molecule": [i]} for i in range(3)]
    out = P.process_jobs_(jobs, redux=dict.update, redux_in_place=True)
    assert out == {"seed": 1, 1: 1, 2: 2}      # molecule 0 returns _SHARED itself and becomes the seed
    assert _SHARED == {"seed": 1}, "the reduction mutated the callback's own object"
    assert out is not _SHARED


def test_the_eigenvalue_count_never_exceeds_the_eigenvalues_it_was_given():
    assert P.top_eigen_columns([4.0, 3.0, 2.0, 1.0], 1.0) == 4
    assert P.top_eigen_columns([4.0, 3.0, 2.0, 1.0], 1.5) == 4        # used to answer 5
    assert P.top_eigen_columns([4.0, 3.0, 2.0, -1e-17], 0.99) == 3    # a rank-deficient covariance's negative tail


def test_a_serial_run_reports_its_progress_too(capsys):
    """20.5.1 recommends num_threads = 1 for debugging, which is where progress is most wanted."""
    atoms = list(range(8))
    P.mp_pandas_obj(_molecule_size, ("molecule", atoms), num_threads=1, mp_batches=4, verbose=True)
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 4 and printed[0].startswith("25.0% done") and printed[-1].startswith("100.0% done")


def _makespan(costs, workers):
    """Greedy list scheduling: every worker takes the next molecule as it frees up, which is what a pool does."""
    busy = [0.0] * workers
    for c in costs:
        i = busy.index(min(busy))
        busy[i] += c
    return max(busy)


def test_more_batches_halve_the_makespan_of_the_sections_own_uneven_task():
    """20.5.1's worked case: 10 molecules on 10 cores where molecule 1 takes twice as long leaves 9 cores idle
    for half the run, and mpBatches = 10 halves the elapsed time. The partition is the engine's own lin_parts,
    the schedule is the greedy one a pool performs, and the cost of an atom is 1 except in the first tenth of
    them, which costs 2 -- so the claim is arithmetic on the partition and can fail."""
    workers, atoms = 10, 1000
    cost = lambda atom: 2.0 if atom < atoms // 10 else 1.0

    def molecules(batches):
        edges = P.lin_parts(atoms, workers * batches)
        return [sum(cost(a) for a in range(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]

    one, ten = molecules(1), molecules(10)
    assert len(one) == 10 and len(ten) == 100
    assert max(one) == 2 * min(one)                          # the first molecule is the whole of the heavy tenth
    ideal = sum(one) / workers                               # 110 units of work over 10 cores
    assert _makespan(one, workers) == pytest.approx(200.0)
    assert _makespan(ten, workers) == pytest.approx(ideal, rel=0.05)
    assert _makespan(ten, workers) < 0.6 * _makespan(one, workers)
    # the cost of the finer partition is more jobs, which is the trade the section names
    assert len(ten) == 10 * len(one)


# --- the determinism switch (plan/performance.md, node 8) ----------------------------------------------------------

def _slow_first_atom(molecule):
    """Atom 0 sleeps longest, so the molecules come back in roughly the reverse of the order they were submitted."""
    i = int(molecule[0])
    time.sleep(0.30 / (i + 1))
    return float(i) + 1e-9 * i


def _fold_scaled(a: float, b: float) -> float:
    """Deliberately not associative: folding in the wrong order lands somewhere else entirely."""
    return a * 0.5 + b


def _fold_and_count(out: dict, value: float) -> dict:
    return {"sum": out["sum"] + value, "molecules": out["molecules"] + 1}


def test_an_atom_ordered_reduction_gives_the_serial_answer_and_an_arrival_ordered_one_does_not():
    """Float addition is not associative, so the order molecules are folded in is a number, not an implementation
    detail. `redux_order="atom"` holds an early finisher back until its predecessors have gone in."""
    atoms = list(range(8))
    fold = lambda **kw: P.mp_pandas_obj(_slow_first_atom, ("molecule", atoms), mp_batches=len(atoms),
                                        redux=_fold_scaled, **kw)
    serial = fold(num_threads=1)
    assert fold(num_threads=4, redux_order="atom") == serial          # the same bits, not merely close
    assert fold(num_threads=4, redux_order="arrival") != serial       # or this check would have no power


def test_an_unknown_reduction_order_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="redux_order"):
        P.process_jobs([], num_threads=2, redux=_fold_scaled, redux_order="whatever")


def test_a_seeded_accumulator_reaches_the_reduction_for_every_molecule_including_the_first():
    """The book's start makes the first output the accumulator, so a reduction with a side effect — a progress line,
    an accounting total — never sees it. `redux_init` is the seed that fixes that."""
    atoms = list(range(6))
    jobs = [{"func": _slow_first_atom, "molecule": [i]} for i in atoms]
    seeded = P.process_jobs_(jobs, redux=_fold_and_count, redux_init={"sum": 0.0, "molecules": 0})
    assert seeded["molecules"] == len(atoms)
    pooled = P.mp_pandas_obj(_slow_first_atom, ("molecule", atoms), num_threads=4, mp_batches=len(atoms),
                             redux=_fold_and_count, redux_order="atom", redux_init={"sum": 0.0, "molecules": 0})
    assert pooled == seeded
    unseeded = P.process_jobs_(jobs, redux=_fold_scaled)
    assert unseeded == P.mp_pandas_obj(_slow_first_atom, ("molecule", atoms), num_threads=4, mp_batches=len(atoms),
                                       redux=_fold_scaled, redux_order="atom")


def test_a_seed_is_copied_for_an_in_place_reduction_so_the_caller_s_object_is_not_the_accumulator():
    seed = {}
    jobs = [{"func": _molecule_squares, "molecule": [i]} for i in range(3)]
    out = P.process_jobs_(jobs, redux=dict.update, redux_in_place=True, redux_init=seed)
    assert out == {0: 0, 1: 1, 2: 4}
    assert seed == {}
