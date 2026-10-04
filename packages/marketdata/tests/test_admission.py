"""Hand book-state oracle and input-boundary failures for archive admission."""

import copy

import pytest
from replay.admission import admit, canonical

HEADER = {
    "schema": "quant-feed-archive/v1",
    "instrument": "DEMO",
    "data_kind": "synthetic",
    "clock_evidence": "PROVIDED_CLOCKS_UNAUDITED",
}


def book(ts, bid="0.49", ask="0.51"):
    return {
        "event_type": "book",
        "asset_id": "DEMO",
        "timestamp": str(ts),
        "bids": [{"price": bid, "size": "3"}],
        "asks": [{"price": ask, "size": "1"}],
    }


def delta(ts, side="BUY", price="0.49", size="1"):
    return {
        "event_type": "price_change",
        "timestamp": str(ts),
        "price_changes": [
            {"asset_id": "DEMO", "side": side, "price": price, "size": size}
        ],
    }


def raw(items):
    output = [HEADER]
    for i, (received, available, value) in enumerate(items):
        row = {
            "record_id": i,
            "kind": "message" if isinstance(value, dict) else "gap",
            "received_ms": received,
            "available_ms": available,
        }
        row["message" if isinstance(value, dict) else "reason"] = value
        output.append(row)
    return b"".join(canonical(row) for row in output)


def test_hand_book_aggregate_delete_gap_and_late_state():
    _, rows, counts = admit(
        raw(
            [
                (0, 0, delta(0)),
                (1, 2, book(1)),
                (12, 15, delta(10)),
                (21, 21, "disconnect"),
                (22, 22, delta(22)),
                (23, 23, book(20)),
                (25, 25, book(25)),
                (32, 35, delta(30, "SELL", "0.51", "0")),
                (40, 40, delta(40, "SELL", "0.52", "1")),
                (45, 45, delta(29)),
                (50, 50, book(50, "0.50", "0.52")),
                (70, 70, book(70, "0.51", "0.53")),
            ]
        )
    )
    assert [r["status"] for r in rows] == [
        "AWAITING_SNAPSHOT",
        "AVAILABLE",
        "AVAILABLE",
        "GAP",
        "AWAITING_SNAPSHOT",
        "OUT_OF_ORDER",
        "AVAILABLE",
        "NO_TWO_SIDED_DEPTH",
        "AVAILABLE",
        "OUT_OF_ORDER",
        "AVAILABLE",
        "AVAILABLE",
    ]
    assert rows[1]["bid_size"] == 300
    assert rows[2]["bid_size"] == 100  # aggregate replaces 300, never adds.
    assert rows[2]["event_ms"] == 10 and rows[2]["available_ms"] == 15
    assert rows[8]["ask_tick"] == 5200  # deletion of 5100 was applied.
    assert rows[6]["segment"] > rows[2]["segment"]
    assert rows[8]["segment"] > rows[6]["segment"]
    assert rows[11]["segment"] > rows[10]["segment"]
    assert counts["counts"]["snapshot_disagreement"] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(timestamp="-1"),
        lambda m: m.update(timestamp="2"),
        lambda m: m.update(event_type="unknown"),
        lambda m: m["price_changes"][0].update(side="BID"),
        lambda m: m["price_changes"][0].update(asset_id="OTHER"),
        lambda m: m["price_changes"][0].update(price="0.00001"),
        lambda m: m["price_changes"][0].update(size="0.001"),
        lambda m: m["price_changes"][0].update(size="NaN"),
    ],
)
def test_malformed_message_refuses_instead_of_skipping(mutate):
    msg = delta(1)
    mutate(msg)
    with pytest.raises(ValueError):
        admit(raw([(1, 1, msg)]))


def test_order_clock_duplicate_json_and_snapshot_boundaries():
    for archive in [
        raw([(10, 20, book(10)), (9, 21, book(9))]),
        raw([(10, 20, book(10)), (11, 19, book(11))]),
        raw([(10, 9, book(10))]),
        raw([(1, 1, book(1))]).replace(b'"record_id":0', b'"record_id":1'),
        raw([(1, 1, book(1))]).replace(
            b'"record_id":0', b'"record_id":0,"record_id":0'
        ),
    ]:
        with pytest.raises(ValueError):
            admit(archive)
    msg = book(0)
    msg["bids"].append(copy.deepcopy(msg["bids"][0]))
    with pytest.raises(ValueError, match="duplicate snapshot"):
        admit(raw([(0, 0, msg)]))


def test_crossed_book_and_future_suffix_invariance():
    prefix = [(0, 0, book(0)), (1, 1, delta(1, price="0.52"))]
    _, before, _ = admit(raw(prefix))
    _, after, _ = admit(raw([*prefix, (2, 2, book(2))]))
    assert before == after[:2]
    assert before[1]["status"] == "CROSSED_BOOK"
    assert before[1]["bid_tick"] is None
    assert before[1]["segment"] > before[0]["segment"]
