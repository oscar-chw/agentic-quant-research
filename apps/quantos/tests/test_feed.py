"""Exercise actual collector -> shared features -> experiment artifact bindings."""

import json
from pathlib import Path

import pytest
from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.quote_features import (
    evaluate,
    feature_spec,
    materialize,
    read_quotes,
    strict_json,
)

from quantos_showcase.feed import bridge_demo, prepare, run_feed, verify_feed

EXAMPLES = Path(__file__).parents[1] / "examples/feed"


def test_hand_features_and_gap_entirely_between_decisions():
    quotes, bound, _ = prepare(
        (EXAMPLES / "archive.jsonl").read_bytes(), (EXAMPLES / "spec.json").read_bytes()
    )
    spec = strict_json(bound)
    features = materialize(read_quotes(quotes, feature_spec(spec)), spec)
    assert [r["status"] for r in features] == ["AVAILABLE"] * 10 + ["GAP", "GAP"]
    assert features[0]["weighted_midpoint_ticks"] == "5050"
    assert (
        features[1]["weighted_midpoint_ticks"] == "5050"
    )  # update unavailable until 15.
    assert features[2]["weighted_midpoint_ticks"] == "5000"
    assert features[4]["midpoint_ticks"] == "5050"  # ask deletion/replacement.
    assert features[5]["midpoint_ticks"] == "5100"
    report, pairs = evaluate(features, spec)
    outcomes = {p["decision_ms"]: p for p in pairs}
    for decision in (20, 40, 60):
        assert outcomes[decision]["reason"] == "continuity_break"
    assert outcomes[90]["reason"] == "target:GAP"
    assert report["splits"]["train"]["models"]["weighted_midpoint"]["mae_ticks"] == "50"
    assert report["splits"]["train"]["models"]["midpoint"]["mae_ticks"] == "0"
    # Future change cannot rewrite earlier features or the train split.
    lines = (EXAMPLES / "archive.jsonl").read_bytes().splitlines()
    item = json.loads(lines[-2])
    item["message"]["bids"][0]["size"] = "11"
    lines[-2] = canonical_json_bytes(item).rstrip()
    new_quotes, new_bound, _ = prepare(
        b"\n".join(lines) + b"\n", (EXAMPLES / "spec.json").read_bytes()
    )
    new_spec = strict_json(new_bound)
    changed = materialize(read_quotes(new_quotes, feature_spec(new_spec)), new_spec)
    assert features[:7] == changed[:7]
    assert (
        report["splits"]["train"] == evaluate(changed, new_spec)[0]["splits"]["train"]
    )


def test_complete_chain_reuse_raw_change_tamper_and_interruption(tmp_path):
    store = tmp_path / "store"
    result = run_feed(EXAMPLES / "archive.jsonl", EXAMPLES / "spec.json", store)
    assert verify_feed(store, result["run_id"], True)["recomputed"]
    again = run_feed(EXAMPLES / "archive.jsonl", EXAMPLES / "spec.json", store)
    assert again["admission_reused"] and again["run_id"] == result["run_id"]
    # Even formatting-only raw changes are preserved as a new source identity.
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(
        (EXAMPLES / "archive.jsonl").read_bytes().replace(b'"kind":', b'"kind": ')
    )
    new = run_feed(changed, EXAMPLES / "spec.json", store)
    assert (
        new["run_id"] != result["run_id"] and new["feature_id"] != result["feature_id"]
    )
    folder = store / "admissions/runs" / result["run_id"]
    original = (folder / "archive.jsonl").read_bytes()
    (folder / "archive.jsonl").write_bytes(original + b" ")
    with pytest.raises(ValueError):
        verify_feed(store, result["run_id"])
    (folder / "archive.jsonl").write_bytes(original)
    (folder / "manifest.json").unlink()
    with pytest.raises(ValueError, match="INCOMPLETE"):
        run_feed(EXAMPLES / "archive.jsonl", EXAMPLES / "spec.json", store)
    assert (folder / "archive.jsonl").read_bytes() == original


def test_reject_malformed_before_writes_and_linked_paths(tmp_path):
    bad = tmp_path / "bad.json"
    spec = json.loads((EXAMPLES / "spec.json").read_bytes())
    spec["price_tick"] = "0.01"
    bad.write_bytes(canonical_json_bytes(spec))
    with pytest.raises(ValueError, match="price tick"):
        run_feed(EXAMPLES / "archive.jsonl", bad, tmp_path / "store")
    assert not (tmp_path / "store").exists()
    link = tmp_path / "linked"
    link.symlink_to(EXAMPLES / "archive.jsonl")
    with pytest.raises(ValueError, match="link"):
        run_feed(link, EXAMPLES / "spec.json", tmp_path / "store")


def test_bridge_reads_actual_persisted_demo_source(tmp_path):
    from quantos_showcase.pipeline import collect_snapshot, fixture_messages

    source, archive = tmp_path / "snapshot", tmp_path / "archive.jsonl"
    collect_snapshot(fixture_messages(), source)
    bridge = bridge_demo(source, archive)
    assert bridge["source_raw_sha256"] == sha256_bytes(
        (source / "raw_messages.jsonl").read_bytes()
    )
    assert (
        json.loads(archive.read_bytes().splitlines()[0])["origin"]["snapshot_sha256"]
        == bridge["source_snapshot_sha256"]
    )
    result = run_feed(archive, EXAMPLES / "demo-spec.json", tmp_path / "store")
    assert (
        verify_feed(tmp_path / "store", result["run_id"], True)["status"] == "VERIFIED"
    )
    with pytest.raises(FileExistsError):
        bridge_demo(source, archive)
    snapshot = json.loads((source / "snapshot.json").read_bytes())
    snapshot["synthetic"] = False
    (source / "snapshot.json").write_bytes(canonical_json_bytes(snapshot))
    with pytest.raises(ValueError, match="synthetic"):
        bridge_demo(source, tmp_path / "other.jsonl")
