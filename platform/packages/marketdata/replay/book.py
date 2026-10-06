"""Point-in-time limit order book replay from keyframes plus deltas.

A recorded history has three parts, all plain Python values:

* keyframes: ``(ts_ms, [Level, ...])`` full book snapshots;
* deltas: ``(ts_ms, side, tick, size_e2)`` rows, each the NEW aggregate size
  resting at one price level (size 0 removes the level), never an increment;
* coverage: ``(start_ms, end_ms | None)`` intervals during which the recorder
  was receiving data. Both endpoints are inclusive; ``None`` is still open.

``reconstruct`` rebuilds one book at a query time; ``reconstruct_many`` answers
many query times with one forward sweep over the deltas. Both refuse with
``NotCovered`` rather than guess: a query outside every coverage interval, or
inside one that has no keyframe at or before the query, has no trustworthy
answer, because a book replayed across a recording gap looks complete and is
not. When intervals overlap, the one that started latest wins: a reconnect
starts a new epoch, and a keyframe from before it must not seed a book after it.

Prices are integer ticks (``PRICE_SCALE`` ticks per unit) and sizes integer
hundredths (``SIZE_SCALE``), so books compare exactly.

``reconstruct_many(..., native=True)``, or ``ASOF_REPLAY_NATIVE=1`` with
``native`` left unset, runs the C++20 port (``packages/marketdata/native``)
through its pybind11 module. It accepts only int64 query times, so pure Python
stays the default.
"""
from __future__ import annotations

import os
from bisect import bisect_right
from itertools import islice
from operator import le
from typing import NamedTuple

SIDE_BID = 0
SIDE_ASK = 1
PRICE_SCALE = 10_000
SIZE_SCALE = 100


class Level(NamedTuple):
    side: int
    tick: int
    size_e2: int


class NotCovered(Exception):
    """The history cannot answer this query time; no book is returned."""


class Book:
    """Aggregate size per ``(side, tick)``. Every answer is an independent copy."""

    __slots__ = ("levels",)

    def __init__(self, levels=None):
        self.levels = dict(levels) if levels else {}

    @classmethod
    def from_snapshot(cls, snapshot):
        """Adopt a keyframe as stored: a repeated level keeps its last row, and a
        size-0 row stays a level (a snapshot is data, not a delete instruction)."""
        book = cls()
        book.levels = {(level.side, level.tick): level.size_e2 for level in snapshot}
        return book

    def apply(self, side, tick, size_e2):
        """Set the aggregate at one level; size 0 removes it."""
        if size_e2 == 0:
            self.levels.pop((side, tick), None)
        else:
            self.levels[(side, tick)] = size_e2

    def multiset(self):
        """Every level as ``(side, tick, size_e2)``; ``sorted()`` gives the canonical order."""
        return [(side, tick, size) for (side, tick), size in self.levels.items()]

    def __eq__(self, other):
        return isinstance(other, Book) and self.levels == other.levels

    def __repr__(self):
        return f"Book({sorted(self.multiset())})"


def _epoch_start(coverage, at_ms):
    start = None
    for begin, end in coverage:
        if begin <= at_ms and (end is None or at_ms <= end) and (start is None or begin > start):
            start = begin
    if start is None:
        raise NotCovered(f"no coverage interval contains {at_ms}")
    return start


def _no_base(at_ms, start):
    return NotCovered(f"covered at {at_ms} but no keyframe in coverage epoch starting {start}")


def reconstruct(keyframes, deltas, at_ms, coverage):
    """The book at ``at_ms``: the last keyframe of its coverage epoch at or before
    ``at_ms``, then every delta after that keyframe up to and including ``at_ms``.

    Deltas stamped with the keyframe's own time are already in it and are skipped.
    This is the plain reference: it rescans every delta and does not require a
    sorted history (it stops at the first keyframe later than ``at_ms``).
    """
    start = _epoch_start(coverage, at_ms)
    base = None
    for frame in keyframes:
        if frame[0] > at_ms:
            break
        if frame[0] >= start:
            base = frame
    if base is None:
        raise _no_base(at_ms, start)
    base_ts = base[0]
    book = Book.from_snapshot(base[1])
    for ts, side, tick, size in deltas:
        if base_ts < ts <= at_ms:
            book.apply(side, tick, size)
    return book


def _require_sorted(times, name):
    if not all(map(le, times, islice(times, 1, None))):
        raise ValueError(f"{name} must be sorted by nondecreasing timestamp")


def reconstruct_many(keyframes, deltas, at_times, coverage, *, native=None):
    """``[reconstruct(keyframes, deltas, t, coverage) for t in at_times]``, in one sweep.

    Keyframes and deltas must be sorted by time (``ValueError`` otherwise); query
    times may repeat and come in any order. Every query is checked in caller order
    before any replay, so the refusal raised is the one the scalar loop would raise
    first, and no partial answer escapes. Inputs are copied before use, and no
    queries means the history is never read.
    """
    if native is None:
        native = os.environ.get("ASOF_REPLAY_NATIVE") == "1"
    if native:
        return _native_many(keyframes, deltas, at_times, coverage)
    queries = list(at_times)
    if not queries:
        return []
    frames, rows, intervals = list(keyframes), list(deltas), list(coverage)
    frame_times = [frame[0] for frame in frames]
    delta_times = [row[0] for row in rows]
    _require_sorted(frame_times, "keyframes")
    _require_sorted(delta_times, "deltas")

    bases = []
    for at_ms in queries:
        start = _epoch_start(intervals, at_ms)
        index = bisect_right(frame_times, at_ms) - 1
        if index < 0 or frame_times[index] < start:
            raise _no_base(at_ms, start)
        bases.append(index)

    # In time order the chosen keyframe never moves back, so one cursor walks the
    # deltas once: O(D) replay in total instead of O(Q * D) for the scalar loop.
    results = [None] * len(queries)
    levels, base, cursor = None, -1, 0
    for position in sorted(range(len(queries)), key=queries.__getitem__):
        at_ms, index = queries[position], bases[position]
        if index != base:
            base = index
            levels = Book.from_snapshot(frames[index][1]).levels
            cursor = bisect_right(delta_times, frame_times[index], cursor)
        end = bisect_right(delta_times, at_ms, cursor)
        for _, side, tick, size in rows[cursor:end]:  # a slice: islice would rescan from 0
            if size == 0:
                levels.pop((side, tick), None)
            else:
                levels[(side, tick)] = size
        cursor = end
        results[position] = Book(levels)
    return results


def _native_many(keyframes, deltas, at_times, coverage):
    import asof_replay_native as native  # noqa: PLC0415 -- optional, built by `make native-build`

    try:
        answers = native.reconstruct_many(keyframes, deltas, at_times, coverage)
    except native.NotCovered as error:
        raise NotCovered(str(error)) from None
    return [Book({(side, tick): size for side, tick, size in levels}) for levels in answers]
