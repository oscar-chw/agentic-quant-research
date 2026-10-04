"""The pybind11 module against the Python reference (replay.book), from Python.

Every comparison runs the pure-Python reconstruct_many and the C++ path on the
same inputs and requires the same outcome: the same books, or the same
exception type and message. The module's raw output is compared in order with
sorted(book.multiset()), so a mis-ordered level list fails here even though
replay.book's dict-backed Books would hide it.

Run by `make pytest` (or `make native-test` at the repository root), which sets
ASOF_REQUIRE_NATIVE=1 so a module that fails to import is an error there, not a
skip. Run alone without the built module, the file skips with that reason.
"""
from __future__ import annotations

import os
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import replay.book as reference
from make_golden import RANDOM_CASES, fixed_cases, random_case

try:
    import asof_replay_native as native
except ImportError as error:
    if os.environ.get("ASOF_REQUIRE_NATIVE") == "1":
        raise
    pytest.skip(f"asof_replay_native is not built ({error}); run `make native-build`", allow_module_level=True)

Level = reference.Level


def outcome(call):
    """('books', [sorted levels per book]) or (exception type name, message)."""
    try:
        books = call()
    except (reference.NotCovered, ValueError) as error:
        return type(error).__name__, str(error)
    return "books", [sorted(book.multiset()) for book in books]


def raw_outcome(call):
    """The module's own answer, in the order it returned the levels."""
    try:
        return "books", call()
    except native.NotCovered as error:
        return "NotCovered", str(error)
    except ValueError as error:
        return "ValueError", str(error)


def assert_same(frames, deltas, coverage, queries):
    expected = outcome(lambda: reference.reconstruct_many(frames, deltas, queries, coverage, native=False))
    assert outcome(lambda: reference.reconstruct_many(frames, deltas, queries, coverage, native=True)) == expected
    assert raw_outcome(lambda: native.reconstruct_many(frames, deltas, queries, coverage)) == expected
    for q in queries:
        scalar = outcome(lambda q=q: [reference.reconstruct(frames, deltas, q, coverage)])
        if scalar[0] == "books":
            scalar = scalar[0], scalar[1][0]
        assert raw_outcome(lambda q=q: native.reconstruct(frames, deltas, q, coverage)) == scalar
    return expected


FIXED = list(fixed_cases(Level))


def test_fixture_and_random_case_counts():
    # Absence policy: a renamed or emptied generator must not pass by comparing nothing.
    assert len(FIXED) == 6 and RANDOM_CASES == 240


@pytest.mark.parametrize("case", FIXED, ids=[case[0] for case in FIXED])
def test_python_suite_fixtures(case):
    _, frames, deltas, coverage, queries = case
    assert_same(frames, deltas, coverage, queries)


def test_recorded_random_histories():
    kinds = set()
    for seed in range(RANDOM_CASES):
        _, frames, deltas, coverage, queries = random_case(seed, reference)
        kinds.add(assert_same(frames, deltas, coverage, queries)[0])
    # The corpus must reach books and both kinds of refusal, or it proves little.
    assert kinds == {"books", "NotCovered", "ValueError"}


def test_returned_books_are_independent():
    _, frames, deltas, coverage, _ = FIXED[0]
    books = reference.reconstruct_many(frames, deltas, [15, 35, 15], coverage, native=True)
    books[0].apply(0, 100, 1)
    assert books[2].levels == {(0, 100): 11, (1, 110): 20}
    assert len({id(book.levels) for book in books}) == 3


def test_env_var_selects_the_native_path_and_the_flag_overrides_it(monkeypatch):
    # A fractional query time is the observable difference: Python answers it,
    # the C++ path refuses it with TypeError.
    _, frames, deltas, coverage, _ = FIXED[0]
    monkeypatch.setenv("ASOF_REPLAY_NATIVE", "1")
    with pytest.raises(TypeError, match="query time must be an int"):
        reference.reconstruct_many(frames, deltas, [14.5], coverage)
    assert reference.reconstruct_many(frames, deltas, [14.5], coverage, native=False)[0].levels == {
        (0, 100): 10, (1, 110): 20}
    monkeypatch.delenv("ASOF_REPLAY_NATIVE")
    assert reference.reconstruct_many(frames, deltas, [14.5], coverage)[0].levels == {(0, 100): 10, (1, 110): 20}


@pytest.mark.parametrize(("deltas", "queries", "error", "message"), [
    ([], [float("nan")], TypeError, "query time must be an int"),
    ([(15, 2, 100, 1)], [20], ValueError, "side must be 0"),
    ([(15, 0, 2**31, 1)], [20], ValueError, "int32"),
    ([(15, 0, 100, 2**63)], [20], ValueError, "int64"),
    ([(15, 0, 100)], [20], ValueError, "4 items"),
])
def test_values_cpp_cannot_hold_are_refused_not_coerced(deltas, queries, error, message):
    with pytest.raises(error, match=message):
        reference.reconstruct_many([(10, [Level(0, 100, 10)])], deltas, queries, [(10, None)], native=True)


def test_no_queries_never_reads_the_history():
    def history():
        raise AssertionError("history consumed")
        yield  # pragma: no cover

    assert reference.reconstruct_many(history(), history(), [], history(), native=True) == []


def test_level_getter_that_mutates_the_inputs():
    # A level attribute is Python code and may mutate the caller's lists while
    # the module converts them. That must neither read freed memory nor change
    # the answer: replay.book copies its inputs first, and the module must agree.
    frames = [(10, [Level(0, 100, 10)]), (20, [Level(1, 110, 5)])]
    deltas = [(15, 0, 100, 11), (25, 1, 111, 3)]
    coverage = [(10, None)]
    queries = [15, 25]
    expected = outcome(lambda: reference.reconstruct_many(list(frames), list(deltas), queries, coverage, native=False))

    class Clearing:
        tick, size_e2 = 100, 10

        @property
        def side(self):
            for rows in (frames, deltas, coverage, queries):
                rows.clear()  # drops the list's references to every row
            return 0

    frames[0] = (10, [Clearing()])
    assert outcome(lambda: reference.reconstruct_many(frames, deltas, queries, coverage, native=True)) == expected
    assert frames == deltas == coverage == queries == []


# Histories are biased towards answerable queries: an open epoch from the
# first keyframe and queries at or after it, so most examples compare
# non-empty books level by level instead of two identical refusals. Small
# ranges make collisions likely: shared timestamps, repeated levels, deltas
# tying with keyframes, gaps and overlapping epochs.
SMALL = {"time": st.integers(-3, 30), "tick": st.integers(100, 111), "size": st.sampled_from([0, 0, 1, 2, 50])}
I64 = st.integers(-(2**63), 2**63 - 1)
WIDE = {"time": I64, "tick": st.integers(-(2**31), 2**31 - 1), "size": I64}


@st.composite
def histories(draw, time, tick, size, sort=True):
    side = st.integers(0, 1)
    frames = draw(st.lists(st.tuples(time, st.lists(st.builds(Level, side, tick, size), max_size=5)),
                           min_size=1, max_size=6))
    deltas = draw(st.lists(st.tuples(time, side, tick, size), max_size=40))
    if sort:
        frames, deltas = sorted(frames, key=lambda r: r[0]), sorted(deltas, key=lambda r: r[0])
    coverage = draw(st.lists(st.tuples(time, st.none() | time), max_size=3))
    first = min(row[0] for row in frames)
    # sampled_from shrinks towards its first element, so the common case goes first.
    if draw(st.sampled_from([True, True, True, False])):  # an open epoch from the first keyframe
        coverage.insert(draw(st.integers(0, len(coverage))), (first, None))
    stamps = [row[0] for row in frames + deltas]
    query = st.sampled_from(stamps) | st.integers(first, max(stamps))
    if draw(st.sampled_from([False, False, False, True])):  # sometimes a query anywhere at all
        query = query | time
    return frames, deltas, coverage, draw(st.lists(query, min_size=1, max_size=12))


def kind(result):
    """'levels' when some book has levels, else 'empty' books or the refusal type."""
    if result[0] != "books":
        return result[0]
    return "levels" if any(result[1]) else "empty"


def run_property(strategy, examples):
    counts = Counter()

    @settings(max_examples=examples, database=None, derandomize=True, deadline=None)
    @given(strategy)
    def check(case):
        frames, deltas, coverage, queries = case
        counts[kind(assert_same(frames, deltas, coverage, queries))] += 1

    check()
    return counts


def test_random_sorted_histories():
    counts = run_property(histories(**SMALL), 400)
    # Resolution: a large share of examples must compare real books (202 of 400 when
    # written), and refusals must still occur.
    assert counts["levels"] >= 0.4 * sum(counts.values()), counts
    assert counts["NotCovered"] and counts["ValueError"] == 0, counts


def test_random_unsorted_histories():
    counts = run_property(histories(**SMALL, sort=False), 150)
    assert counts["ValueError"] >= 0.3 * sum(counts.values()) and counts["levels"], counts


def test_full_integer_range():
    counts = run_property(histories(**WIDE), 150)
    assert counts["levels"] >= 0.4 * sum(counts.values()), counts
