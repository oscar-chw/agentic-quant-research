"""Hand-calculated economics, brute-force temporal oracle, and artifact recovery."""

import csv
import io
import random
from fractions import Fraction
from unittest.mock import patch

import pytest

from qrae.artifacts import ArtifactStore, canonical_json_bytes
from qrae.quote_features import (
    COLUMNS,
    evaluate,
    feature_spec,
    materialize,
    read_quotes,
)
from qrae.quote_workflow import run_quotes, verify_quote_run

SPEC = dict(
    schema="quote-spec/v1",
    instrument="DEMO",
    data_kind="synthetic",
    price_tick="0.01",
    start_ms=0,
    step_ms=10,
    count=12,
    max_age_ms=25,
    train_count=4,
    validation_count=4,
)
DATA = [
    (0, 0, 0, 0, 0, 99, 101, 3, 1),
    (10, 20, 25, 1, 0, 100, 102, 1, 1),
    (20, 35, 35, 2, 0, 100, 102, 1, 1),
    (30, 30, 30, 3, 0, 101, 103, 3, 1),
    (30, 45, 45, 3, 1, 101, 103, 1, 3),
    (50, 55, 60, 4, 0, 103, 105, 1, 1),
    (90, 90, 90, 5, 0, 105, 107, 2, 2),
]


def raw(data=DATA):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(("DEMO", *r) for r in data)
    return stream.getvalue().encode()


def oracle(rows, spec):
    result = []
    for index in range(spec["count"]):
        now = spec["start_ms"] + index * spec["step_ms"]
        eligible = [r for r in rows if r["available_ms"] <= now]
        if not eligible:
            result.append({"decision_ms": now, "status": "NO_QUOTE"})
            continue
        row = max(eligible, key=lambda r: (r["sequence"], r["revision"]))
        common = {
            "decision_ms": now,
            **{
                k: row[k]
                for k in (
                    "event_ms",
                    "received_ms",
                    "available_ms",
                    "sequence",
                    "revision",
                )
            },
        }
        if row["event_ms"] < now - spec["max_age_ms"]:
            result.append(common | {"status": "STALE"})
            continue
        bid, ask, qb, qa = (
            row[k] for k in ("bid_tick", "ask_tick", "bid_size", "ask_size")
        )
        if min(qb, qa) == 0:
            result.append(common | {"status": "NO_TWO_SIDED_DEPTH"})
            continue
        result.append(
            common
            | dict(
                status="AVAILABLE",
                midpoint_ticks=str(Fraction(bid + ask, 2)),
                spread_ticks=ask - bid,
                imbalance=str(Fraction(qb - qa, qb + qa)),
                weighted_midpoint_ticks=str(Fraction(ask * qb + bid * qa, qb + qa)),
            )
        )
    return result


def test_delays_revisions_stale_and_independent_ledger():
    fs = feature_spec(SPEC)
    rows = read_quotes(raw(), fs)
    features = materialize(rows, fs)
    assert features == oracle(rows, fs)
    assert [features[i]["sequence"] for i in (0, 1, 2, 3, 4, 5, 6)] == [
        0,
        0,
        0,
        3,
        3,
        3,
        4,
    ]
    assert features[4]["weighted_midpoint_ticks"] == "205/2"
    assert features[5]["weighted_midpoint_ticks"] == "203/2"
    assert features[8]["status"] == "STALE"
    report, pairs = evaluate(features, SPEC)
    assert report["splits"]["train"]["models"]["midpoint"]["mae_ticks"] == "2/3"
    assert (
        report["splits"]["train"]["models"]["weighted_midpoint"]["mae_ticks"] == "5/6"
    )
    assert (
        report["splits"]["validation"]["models"]["weighted_midpoint"]["mae_ticks"]
        == "1"
    )
    assert report["splits"]["test"]["evaluated_pairs"] == 2
    assert len(pairs) == 9
    assert all(p["decision_ms"] not in (30, 70, 110) for p in pairs)


def test_future_changes_do_not_rewrite_features_or_earlier_splits():
    fs = feature_spec(SPEC)
    original = materialize(read_quotes(raw(), fs), fs)
    changed = [
        *DATA[:-1],
        (90, 90, 90, 5, 0, 205, 207, 5, 1),
        (120, 125, 130, 6, 0, 299, 301, 1, 1),
    ]
    new = materialize(read_quotes(raw(changed), fs), fs)
    assert original[:9] == new[:9]
    before, _ = evaluate(original, SPEC)
    after, _ = evaluate(new, SPEC)
    assert before["splits"]["train"] == after["splits"]["train"]
    assert before["splits"]["validation"] == after["splits"]["validation"]


def test_sweep_matches_bruteforce_for_shuffled_delayed_revised_streams():
    rng = random.Random(417)
    for _ in range(25):
        data = []
        for seq in range(30):
            event = seq * 5
            received = event + rng.randrange(25)
            available = received + rng.randrange(10)
            bid = 100 + rng.randrange(10)
            qb, qa = rng.randrange(5), rng.randrange(5)
            data.append((event, received, available, seq, 0, bid, bid + 2, qb, qa))
            if seq % 3 == 0:
                data.append(
                    (event, received + 8, available + 8, seq, 1, bid, bid + 3, qa, qb)
                )
        rng.shuffle(data)
        fs = feature_spec(SPEC | dict(count=20))
        rows = read_quotes(raw(data), fs)
        assert materialize(rows, fs) == oracle(rows, fs)


@pytest.mark.parametrize(
    "change,reason",
    [
        (dict(received_ms=0, event_ms=1), "clocks"),
        (dict(available_ms=0, received_ms=1), "clocks"),
        (dict(bid_tick=102), "uncrossed"),
        (dict(instrument="OTHER"), "instrument"),
        (dict(sequence=-1), "unsigned"),
        (dict(bid_size="NaN"), "unsigned"),
    ],
)
def test_malformed_quotes_refuse(change, reason):
    lines = list(csv.DictReader(io.StringIO(raw().decode())))
    lines[0].update(change)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(lines)
    with pytest.raises(ValueError, match=reason):
        read_quotes(stream.getvalue().encode(), feature_spec(SPEC))


def test_identity_conflicts_duplicates_no_quote_zero_depth():
    fs = feature_spec(SPEC)
    assert read_quotes(raw([*DATA, DATA[0]]), fs) == read_quotes(raw(), fs)
    with pytest.raises(ValueError, match="conflicting"):
        read_quotes(raw([*DATA, (0, 0, 0, 0, 0, 99, 103, 3, 1)]), fs)
    with pytest.raises(ValueError, match="revision"):
        read_quotes(raw([*DATA, (31, 45, 45, 3, 2, 101, 103, 1, 3)]), fs)
    rows = read_quotes(raw([(0, 1, 5, 0, 0, 99, 101, 0, 1)]), fs)
    features = materialize(rows, fs)
    assert features[0]["status"] == "NO_QUOTE"
    assert features[1]["status"] == "NO_TWO_SIDED_DEPTH"
    report, _ = evaluate(features, SPEC)
    assert report["splits"]["train"]["models"]["midpoint"]["mae_ticks"] is None


def inputs(tmp_path):
    quotes = tmp_path / "quotes.csv"
    spec = tmp_path / "spec.json"
    quotes.write_bytes(raw())
    spec.write_bytes(canonical_json_bytes(SPEC))
    return quotes, spec, tmp_path / "store"


def test_materialization_reused_across_experiments_and_source_changes_invalidate(
    tmp_path,
):
    quotes, spec, store = inputs(tmp_path)
    first = run_quotes(quotes, spec, store)
    assert verify_quote_run(store, first["run_id"], True)["recomputed"]
    second = run_quotes(quotes, spec, store)
    assert second["features_reused"] and second["experiment_reused"]
    spec.write_bytes(
        canonical_json_bytes(SPEC | dict(train_count=6, validation_count=2))
    )
    third = run_quotes(quotes, spec, store)
    assert third["features_reused"] and third["run_id"] != first["run_id"]
    assert len(list((store / "features/runs").iterdir())) == 1
    quotes.write_bytes(raw([*DATA, (120, 120, 120, 6, 0, 107, 109, 1, 1)]))
    fourth = run_quotes(quotes, spec, store)
    assert fourth["feature_id"] != first["feature_id"]


def test_tamper_and_interrupted_materialization_refuse(tmp_path):
    quotes, spec, store = inputs(tmp_path)
    first = run_quotes(quotes, spec, store)
    feature = store / "features/runs" / first["feature_id"] / "features.jsonl"
    feature.write_bytes(feature.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_quote_run(store, first["run_id"])
    interrupted = tmp_path / "interrupted"
    with patch.object(
        ArtifactStore, "commit_manifest", side_effect=OSError("interrupted")
    ):
        with pytest.raises(OSError):
            run_quotes(quotes, spec, interrupted)
    with pytest.raises(ValueError, match="INCOMPLETE"):
        run_quotes(quotes, spec, interrupted)


def test_spec_refusal_before_output_and_linked_paths(tmp_path):
    quotes, spec, store = inputs(tmp_path)
    spec.write_bytes(canonical_json_bytes(SPEC | dict(train_count=10)))
    with pytest.raises(ValueError):
        run_quotes(quotes, spec, store)
    assert not store.exists()
    spec.write_bytes(canonical_json_bytes(SPEC))
    alias = tmp_path / "alias"
    alias.symlink_to(quotes)
    with pytest.raises(ValueError, match="link"):
        run_quotes(alias, spec, store)
    spec.write_text('{"schema":"quote-spec/v1","schema":"quote-spec/v1"}')
    with pytest.raises(ValueError, match="duplicate"):
        run_quotes(quotes, spec, store)
