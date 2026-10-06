"""Venue book messages to a replayable history: snapshots become keyframes,
level updates become deltas, and a live book per asset is kept alongside.

Two JSON message shapes are understood; anything else raises ``ValueError``
rather than being skipped, because a silently dropped update corrupts every
book after it:

* ``{"event_type": "book", "asset_id": A, "timestamp": "<ms>",
  "bids": [{"price": "0.49", "size": "3"}, ...], "asks": [...]}``
  is a full snapshot of asset A. It replaces the live book and is recorded as a
  keyframe. On a repeated price the later row wins; a size-0 row rests nothing
  and is dropped.
* ``{"event_type": "price_change", "timestamp": "<ms>", "price_changes":
  [{"asset_id": A, "side": "BUY" | "SELL", "price": p, "size": s}, ...]}``
  carries the new aggregate size at each listed level (size 0 removes it). Each
  entry becomes one delta, applied in the order listed. ``replay.book`` skips a
  delta stamped with its keyframe's own time as already included, so a change
  stamped with the time of its asset's latest snapshot, which arrived after it,
  is also recorded as a fresh keyframe of the book it produced.

Prices and sizes are decimal strings converted exactly onto the integer grids
of ``replay.book``; a value off the grid is refused, never rounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from replay.book import PRICE_SCALE, SIDE_ASK, SIDE_BID, SIZE_SCALE, Book, Level

SIDES = {"BUY": SIDE_BID, "SELL": SIDE_ASK}


def _exact(text, scale, what):
    if not isinstance(text, str):
        raise ValueError(f"{what} must be a decimal string")
    try:
        value = Fraction(text) * scale
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"{what} is not a decimal: {text!r}") from None
    if value.denominator != 1:
        raise ValueError(f"{what} {text} is not a multiple of 1/{scale}")
    return int(value)


def scale_price(text):
    """``"0.4700"`` -> 4700 ticks."""
    return _exact(text, PRICE_SCALE, "price")


def scale_size(text):
    """``"12"`` -> 1200 hundredths."""
    return _exact(text, SIZE_SCALE, "size")


@dataclass(frozen=True)
class Delta:
    asset: str
    ts_ms: int
    side: int
    tick: int
    size_e2: int


@dataclass(frozen=True)
class Keyframe:
    asset: str
    ts_ms: int
    levels: tuple  # Level rows sorted by (side, tick)


@dataclass
class Update:
    """What one message changed. ``disagreed`` names each asset whose live book
    differed from the snapshot that replaced it (a sign of missed updates)."""

    deltas: list = field(default_factory=list)
    keyframes: list = field(default_factory=list)
    disagreed: list = field(default_factory=list)


def _timestamp(message):
    stamp = message.get("timestamp")
    if not isinstance(stamp, str) or not stamp.isdigit():
        raise ValueError(f"timestamp must be a string of digits, not {stamp!r}")
    return int(stamp)


class Feed:
    """Live books per asset, advanced one message at a time."""

    def __init__(self):
        self.books: dict[str, Book] = {}
        self._snapshot_ts: dict[str, int] = {}

    def apply(self, message) -> Update:
        if not isinstance(message, dict):
            raise ValueError("message must be a JSON object")
        kind = message.get("event_type")
        if kind == "book":
            return self._snapshot(message)
        if kind == "price_change":
            return self._changes(message)
        raise ValueError(f"unsupported event_type {kind!r}")

    def _snapshot(self, message):
        ts, asset = _timestamp(message), message.get("asset_id")
        if not isinstance(asset, str):
            raise ValueError("a book message needs an asset_id")
        resting = {}
        for side, name in ((SIDE_BID, "bids"), (SIDE_ASK, "asks")):
            rows = message.get(name)
            if not isinstance(rows, list):
                raise ValueError(f"a book message needs a {name} list")
            for row in rows:
                resting[(side, scale_price(row["price"]))] = scale_size(row["size"])
        levels = {key: size for key, size in resting.items() if size != 0}
        update = Update(keyframes=[Keyframe(asset, ts, tuple(Level(s, t, z) for (s, t), z in sorted(levels.items())))])
        previous = self.books.get(asset)
        if previous is not None and previous.levels != levels:
            update.disagreed.append(asset)
        self.books[asset] = Book(levels)
        self._snapshot_ts[asset] = ts
        return update

    def _changes(self, message):
        ts, entries = _timestamp(message), message.get("price_changes")
        if not isinstance(entries, list):
            raise ValueError("a price_change message needs a price_changes list")
        update, same_ms = Update(), {}
        for entry in entries:
            asset, side = entry.get("asset_id"), SIDES.get(entry.get("side"))
            if not isinstance(asset, str) or side is None:
                raise ValueError("each price change needs an asset_id and side BUY or SELL")
            delta = Delta(asset, ts, side, scale_price(entry["price"]), scale_size(entry["size"]))
            self.books.setdefault(asset, Book()).apply(delta.side, delta.tick, delta.size_e2)
            update.deltas.append(delta)
            if self._snapshot_ts.get(asset) == ts:
                same_ms[asset] = None
        for asset in same_ms:
            levels = self.books[asset].levels
            update.keyframes.append(Keyframe(asset, ts, tuple(Level(s, t, z) for (s, t), z in sorted(levels.items()))))
        return update
