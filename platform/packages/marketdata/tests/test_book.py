"""replay.book against hand-worked books, the recorded golden cases and generated histories."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from replay.book import SIDE_ASK, SIDE_BID, Level, NotCovered, reconstruct, reconstruct_many

GOLDEN = Path(__file__).resolve().parents[1] / "native/tests/golden/reference_cases.txt"


def levels(book):
    return sorted(book.multiset())


# --- hand-worked books ------------------------------------------------------

FRAMES = [(10, [Level(SIDE_BID, 100, 10), Level(SIDE_ASK, 110, 20)])]


def test_a_delta_replaces_the_aggregate_and_zero_deletes():
    deltas = [(11, SIDE_BID, 100, 7), (12, SIDE_ASK, 110, 0), (13, SIDE_ASK, 111, 5)]
    assert levels(reconstruct(FRAMES, deltas, 11, [(10, None)])) == [(0, 100, 7), (1, 110, 20)]
    assert levels(reconstruct(FRAMES, deltas, 13, [(10, None)])) == [(0, 100, 7), (1, 111, 5)]


def test_deltas_at_the_keyframe_time_are_already_in_it():
    deltas = [(10, SIDE_BID, 100, 999), (10, SIDE_BID, 101, 1)]
    assert levels(reconstruct(FRAMES, deltas, 10, [(10, None)])) == [(0, 100, 10), (1, 110, 20)]


def test_a_snapshot_keeps_its_last_duplicate_and_its_zero_rows():
    frames = [(1, [Level(0, 100, 5), Level(0, 100, 0), Level(1, 101, 0)])]
    assert levels(reconstruct(frames, [], 1, [(1, 1)])) == [(0, 100, 0), (1, 101, 0)]


def test_uncovered_and_baseless_queries_are_refused():
    with pytest.raises(NotCovered, match="^no coverage interval contains 30$"):
        reconstruct(FRAMES, [], 30, [(10, 20)])
    # A reconnect at 25 opens a new epoch; the keyframe at 10 cannot seed it.
    with pytest.raises(NotCovered, match="^covered at 30 but no keyframe in coverage epoch starting 25$"):
        reconstruct_many(FRAMES, [], [15, 30], [(10, 20), (25, None)])
    # Overlapping intervals: the latest-starting one that contains the query wins.
    with pytest.raises(NotCovered, match="epoch starting 12$"):
        reconstruct(FRAMES, [], 15, [(0, None), (12, 18)])


def test_the_first_refusal_in_caller_order_is_reported():
    coverage = [(10, 20), (25, None)]
    with pytest.raises(NotCovered, match="contains 21"):
        reconstruct_many(FRAMES, [], [15, 21, 30], [(10, 20)])
    with pytest.raises(NotCovered, match="covered at 30"):
        reconstruct_many(FRAMES, [], [30, 21], coverage)


def test_unsorted_history_is_refused_by_the_batch_path_only():
    frames = [(30, [Level(0, 102, 1)]), (10, [Level(0, 100, 1)])]
    with pytest.raises(ValueError, match="^keyframes must be sorted by nondecreasing timestamp$"):
        reconstruct_many(frames, [], [35], [(0, None)])
    deltas = [(20, 0, 100, 3), (15, 0, 100, 2)]
    with pytest.raises(ValueError, match="^deltas must be sorted by nondecreasing timestamp$"):
        reconstruct_many(FRAMES, deltas, [25], [(0, None)])
    assert levels(reconstruct(FRAMES, deltas, 25, [(0, None)])) == [(0, 100, 2), (1, 110, 20)]


def test_queries_in_any_order_come_back_in_caller_order_as_independent_books():
    deltas = [(15, 0, 100, 11)]
    books = reconstruct_many(FRAMES, deltas, [15, 10, 15], [(10, None)])
    assert [levels(b) for b in books] == [[(0, 100, 11), (1, 110, 20)], [(0, 100, 10), (1, 110, 20)],
                                          [(0, 100, 11), (1, 110, 20)]]
    books[0].apply(0, 100, 1)
    assert books[2].levels == {(0, 100): 11, (1, 110): 20}
    assert len({id(b.levels) for b in books}) == 3


def test_fractional_and_nan_query_times_are_answered_in_pure_python():
    # The C++ path refuses non-int query times (TypeError), which is why it is opt-in.
    books = reconstruct_many(FRAMES, [(15, 0, 100, 11)], [14.5, 15.5], [(10, None)])
    assert [b.levels for b in books] == [{(0, 100): 10, (1, 110): 20}, {(0, 100): 11, (1, 110): 20}]
    with pytest.raises(NotCovered, match="^no coverage interval contains nan$"):
        reconstruct_many(FRAMES, [], [float("nan")], [(10, None)])


def test_no_queries_never_reads_the_history():
    def history():
        raise AssertionError("history consumed")
        yield  # pragma: no cover

    assert reconstruct_many(history(), history(), [], history(), native=False) == []


# --- golden cases recorded for the C++ port --------------------------------

def parse_levels(fields):
    count, rest = int(fields[0]), [int(x) for x in fields[1:]]
    assert len(rest) == 3 * count
    return [tuple(rest[i:i + 3]) for i in range(0, len(rest), 3)]


def golden_cases(text):
    case = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        tag, *fields = line.split()
        if tag == "case":
            case = {"name": fields[0], "frames": [], "deltas": [], "coverage": [], "queries": [],
                    "batch": None, "scalar": []}
        elif tag == "K":
            case["frames"].append((int(fields[0]), [Level(*row) for row in parse_levels(fields[1:])]))
        elif tag == "D":
            case["deltas"].append(tuple(int(x) for x in fields))
        elif tag == "C":
            case["coverage"].append((int(fields[0]), None if fields[1] == "-" else int(fields[1])))
        elif tag == "Q":
            case["queries"].extend(int(x) for x in fields)
        elif tag == "batch" and fields[0] == "books":
            case["batch"] = ("books", [])
        elif tag == "B":
            case["batch"][1].append(parse_levels(fields))
        elif tag in ("batch", "scalar") and fields[0] == "error":
            outcome = (fields[1], " ".join(fields[2:]))
            if tag == "batch":
                case["batch"] = outcome
            else:
                case["scalar"].append(outcome)
        elif tag == "scalar":
            case["scalar"].append(("books", parse_levels(fields[1:])))
        elif tag == "end":
            yield case


def outcome(call):
    try:
        return "books", call()
    except (NotCovered, ValueError) as error:
        return type(error).__name__, str(error)


def test_every_recorded_golden_outcome():
    cases = list(golden_cases(GOLDEN.read_text()))
    assert len(cases) == 247  # 7 fixtures + 240 random; a truncated file must not pass
    kinds = set()
    for c in cases:
        frames, deltas, coverage, queries = c["frames"], c["deltas"], c["coverage"], c["queries"]
        batch = outcome(lambda: [levels(b) for b in reconstruct_many(frames, deltas, queries, coverage, native=False)])
        assert batch == c["batch"], c["name"]
        kinds.add(batch[0])
        assert len(c["scalar"]) == len(queries), c["name"]
        for q, expected in zip(queries, c["scalar"]):
            assert outcome(lambda q=q: levels(reconstruct(frames, deltas, q, coverage))) == expected, (c["name"], q)
    assert kinds == {"books", "NotCovered", "ValueError"}


# --- generated histories ----------------------------------------------------

TIME = st.integers(-3, 30)
LEVEL = st.builds(Level, st.integers(0, 1), st.integers(100, 111), st.sampled_from([0, 0, 1, 2, 50]))
DELTA = st.tuples(TIME, st.integers(0, 1), st.integers(100, 111), st.sampled_from([0, 0, 1, 2, 50]))


@st.composite
def histories(draw):
    frames = sorted(draw(st.lists(st.tuples(TIME, st.lists(LEVEL, max_size=5)), min_size=1, max_size=6)),
                    key=lambda row: row[0])
    deltas = sorted(draw(st.lists(DELTA, max_size=40)), key=lambda row: row[0])
    coverage = draw(st.lists(st.tuples(TIME, st.none() | TIME), max_size=3))
    first = frames[0][0]
    # Biased towards answerable queries, so most examples compare real books: usually
    # an open epoch from the first keyframe and queries at or after it.
    if draw(st.sampled_from([True, True, True, False])):
        coverage.append((first, None))
    query = st.integers(first, 30) if draw(st.sampled_from([True, True, True, False])) else TIME
    queries = draw(st.lists(query, min_size=1, max_size=12))
    return frames, deltas, coverage, queries


def test_batch_equals_the_scalar_loop():
    kinds = Counter()

    @settings(max_examples=400, deadline=None, derandomize=True, database=None)
    @given(histories())
    def check(case):
        frames, deltas, coverage, queries = case
        expected = outcome(lambda: [levels(reconstruct(frames, deltas, q, coverage)) for q in queries])
        assert outcome(lambda: [levels(b) for b in reconstruct_many(frames, deltas, queries, coverage)]) == expected
        kinds["levels" if expected[0] == "books" and any(expected[1]) else expected[0]] += 1

    check()
    # Resolution: most examples must compare real books, and refusals must still occur.
    assert kinds["levels"] >= 0.4 * sum(kinds.values()) and kinds["NotCovered"], kinds


@settings(max_examples=200, deadline=None, derandomize=True, database=None)
@given(histories(), st.randoms(use_true_random=False))
def test_a_disordered_history_is_refused(case, rng):
    frames, deltas, coverage, queries = case
    rows = [row[0] for row in deltas]
    if len(set(rows)) < 2:
        return
    shuffled = list(deltas)
    while [row[0] for row in shuffled] == sorted(rows):
        rng.shuffle(shuffled)
    with pytest.raises(ValueError, match="deltas must be sorted"):
        reconstruct_many(frames, shuffled, queries, [(-10, None)])
