"""replay.feed: venue messages to keyframes, deltas and live books."""
from __future__ import annotations

import pytest

from replay.book import Level, reconstruct_many
from replay.feed import Feed, scale_price, scale_size


def book(ts, bids, asks, asset="A"):
    return {"event_type": "book", "asset_id": asset, "timestamp": str(ts),
            "bids": [{"price": p, "size": s} for p, s in bids],
            "asks": [{"price": p, "size": s} for p, s in asks]}


def change(ts, *entries, asset="A"):
    return {"event_type": "price_change", "timestamp": str(ts),
            "price_changes": [{"asset_id": asset, "side": side, "price": p, "size": s} for side, p, s in entries]}


def test_exact_scaling_refuses_values_off_the_grid():
    assert scale_price("0.4700") == 4700 and scale_price("1") == 10_000
    assert scale_size("12") == 1200 and scale_size("0.25") == 25
    for bad in ("0.00001", "abc", "", "NaN"):
        with pytest.raises(ValueError):
            scale_price(bad)
    with pytest.raises(ValueError, match="1/100"):
        scale_size("0.001")
    with pytest.raises(ValueError, match="decimal string"):
        scale_size(3)


def test_a_snapshot_becomes_a_sorted_keyframe_without_zero_rows():
    feed = Feed()
    update = feed.apply(book(5, [("0.49", "3"), ("0.48", "0"), ("0.47", "1"), ("0.47", "2")], [("0.51", "1")]))
    (frame,) = update.keyframes
    assert (frame.asset, frame.ts_ms) == ("A", 5) and update.deltas == []
    assert frame.levels == (Level(0, 4700, 200), Level(0, 4900, 300), Level(1, 5100, 100))
    assert feed.books["A"].levels == {(0, 4700): 200, (0, 4900): 300, (1, 5100): 100}


def test_changes_replace_aggregates_in_listed_order():
    feed = Feed()
    feed.apply(book(0, [("0.49", "3")], [("0.51", "1")]))
    update = feed.apply(change(7, ("BUY", "0.49", "0"), ("BUY", "0.49", "5"), ("SELL", "0.52", "2")))
    assert [(d.ts_ms, d.side, d.tick, d.size_e2) for d in update.deltas] == [
        (7, 0, 4900, 0), (7, 0, 4900, 500), (7, 1, 5200, 200)]
    assert feed.books["A"].levels == {(0, 4900): 500, (1, 5100): 100, (1, 5200): 200}


def test_a_snapshot_that_disagrees_with_the_live_book_is_reported():
    feed = Feed()
    assert feed.apply(book(0, [("0.49", "3")], [("0.51", "1")])).disagreed == []
    feed.apply(change(1, ("BUY", "0.49", "4")))
    assert feed.apply(book(2, [("0.49", "4")], [("0.51", "1")])).disagreed == []
    assert feed.apply(book(3, [("0.50", "4")], [("0.51", "1")])).disagreed == ["A"]
    assert feed.books["A"].levels == {(0, 5000): 400, (1, 5100): 100}


def test_live_books_equal_the_replayed_history():
    feed, frames, deltas = Feed(), [], []
    messages = [book(0, [("0.49", "3")], [("0.51", "1")]), change(1, ("SELL", "0.51", "0"), ("SELL", "0.53", "2")),
                change(2, ("BUY", "0.50", "1")), book(3, [("0.48", "1")], [("0.53", "2")]), change(4, ("BUY", "0.48", "0"))]
    live = []
    for message in messages:
        update = feed.apply(message)
        frames += [(k.ts_ms, list(k.levels)) for k in update.keyframes]
        deltas += [(d.ts_ms, d.side, d.tick, d.size_e2) for d in update.deltas]
        live.append(dict(feed.books["A"].levels))
    replayed = reconstruct_many(frames, deltas, range(5), [(0, None)])
    assert [b.levels for b in replayed] == live


@pytest.mark.parametrize("message", [
    {"event_type": "trade", "timestamp": "1"},
    "not an object",
    book("x", [], []),
    book(-1, [], []),
    {"event_type": "book", "timestamp": "1", "bids": [], "asks": []},
    {"event_type": "book", "asset_id": "A", "timestamp": "1", "bids": []},
    change(1, ("BID", "0.49", "1")),
    change(1, ("BUY", "0.49", "1"), asset=None),
    {"event_type": "price_change", "timestamp": "1"},
    change(1, ("BUY", "0.00001", "1")),
])
def test_unknown_or_malformed_messages_raise_instead_of_being_skipped(message):
    with pytest.raises(ValueError):
        Feed().apply(message)
