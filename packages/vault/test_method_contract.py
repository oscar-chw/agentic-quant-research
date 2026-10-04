"""Hand oracles for rules, source identity and incomplete evidence."""

import copy
import hashlib

import pytest

from method_contract import evaluate_card, read_sources, validate_card


@pytest.fixture
def card(tmp_path):
    raw = b"Original local algebra; no predictive claim established.\n"
    (tmp_path / "note.md").write_bytes(raw)
    return {
        "schema": "vault-method/v1",
        "id": "imbalance",
        "method": "quote-weighted-midpoint/v1",
        "hypothesis": "Lower error than midpoint",
        "mechanism": "Displayed size imbalance",
        "assumptions": ["Useful displayed liquidity"],
        "limitations": ["No fill model"],
        "sources": [
            {
                "id": "original-note",
                "kind": "local_derivation",
                "path": "note.md",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "relationship": "derivation",
                "use_scope": "local_review",
            }
        ],
        "decision": {
            "split": "validation",
            "metric": "mae_ticks",
            "candidate": "weighted_midpoint",
            "baseline": "midpoint",
            "min_pairs": 2,
            "min_coverage": "1/2",
            "min_improvement_ticks": "1/3",
        },
    }


def report(candidate="1/3", baseline="1", count=2, coverage="2/3"):
    return {
        "schema": "quote-forecast-report/v2",
        "splits": {
            "validation": {
                "candidate_pairs": 3,
                "evaluated_pairs": count,
                "coverage": coverage,
                "models": {
                    "weighted_midpoint": {"mae_ticks": candidate},
                    "midpoint": {"mae_ticks": baseline},
                },
            }
        },
    }


@pytest.mark.parametrize(
    ("candidate", "baseline", "status", "improvement"),
    [
        ("1/3", "1", "SUPPORTS_RULE", "2/3"),
        ("2/3", "1", "NOT_SUPPORTED", "1/3"),
        ("1", "1", "NOT_SUPPORTED", "0"),
        ("3/2", "1", "NOT_SUPPORTED", "-1/2"),
    ],
)
def test_hand_exact_rule(card, candidate, baseline, status, improvement):
    out = evaluate_card(card, report(candidate, baseline))
    assert out["status"] == status and out["improvement_ticks"] == improvement
    assert not out["profitability_evaluated"] and not out["untouched_holdout_verified"]


def test_insufficient_count_coverage_and_empty(card):
    assert (
        evaluate_card(card, report(count=1, coverage="1/3"))["status"] == "INSUFFICIENT"
    )
    card["decision"]["min_pairs"] = 1
    assert (
        evaluate_card(card, report(count=1, coverage="1/3"))["status"] == "INSUFFICIENT"
    )
    out = evaluate_card(card, report(None, None, 0, "0"))
    assert out["status"] == "INSUFFICIENT" and out["improvement_ticks"] is None
    with pytest.raises(ValueError, match="coverage"):
        evaluate_card(card, report(coverage="1"))


def test_sources_original_and_paper_share_digest_policy(card, tmp_path):
    assert (
        read_sources(card, tmp_path)["source-0.md"]
        == (tmp_path / "note.md").read_bytes()
    )
    card["sources"][0]["kind"] = "paper_note"
    assert read_sources(card, tmp_path)
    (tmp_path / "note.md").write_text("Changed")
    with pytest.raises(ValueError, match="digest"):
        read_sources(card, tmp_path)
    (tmp_path / "note.md").unlink()
    with pytest.raises(ValueError, match="regular"):
        read_sources(card, tmp_path)


def test_linked_parent_and_size_refused(card, tmp_path):
    (tmp_path / "actual").mkdir()
    (tmp_path / "actual/note.md").write_bytes((tmp_path / "note.md").read_bytes())
    (tmp_path / "alias").symlink_to(tmp_path / "actual", target_is_directory=True)
    card["sources"][0]["path"] = "alias/note.md"
    with pytest.raises(ValueError, match="unlinked"):
        read_sources(card, tmp_path)
    card["sources"][0]["path"] = "note.md"
    (tmp_path / "note.md").write_bytes(b"x" * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="256 KiB"):
        read_sources(card, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"method": "subprocess.run"},
        {"assumptions": []},
        {"sources": []},
        {"unknown": True},
        {"id": "../escape"},
    ],
)
def test_unsupported_card(card, change):
    with pytest.raises(ValueError):
        validate_card(card | change)


@pytest.mark.parametrize("value", [True, 0, -1, 100001])
def test_min_pairs_integer(card, value):
    card["decision"]["min_pairs"] = value
    with pytest.raises(ValueError):
        validate_card(card)


def test_no_rights_promotion_or_path_escape(card):
    other = copy.deepcopy(card)
    other["sources"][0]["use_scope"] = "public"
    with pytest.raises(ValueError, match="rights"):
        validate_card(other)
    card["sources"][0]["path"] = "../note.md"
    with pytest.raises(ValueError, match="within"):
        validate_card(card)


def factor_card(card, **rule):
    splits = {s: {"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"} for s in ("train", "validation", "test")}
    return {**card, "method": "factor-rank-ic/v1", "decision": {
        "signal": "momentum", "lookback": 2, "cost_bps": 10, "splits": splits, "metric": "rank_ic_mean",
        "min_rank_ic": "1/10", "min_valid_dates": 3, "min_net_mean": "0", **rule}}


def factor_report(validation, test, signal="momentum"):
    def summary(ic, dates, net=0.001):
        return {"rank_ic_mean": ic, "valid_ic_dates": dates, "net_mean": net}
    return {"schema_version": "factor-report/v2", "selection": {"selected": signal},
            "development": {signal: {"validation": {"summary": summary(*validation)}}},
            "test": {"summary": summary(*test)}}


@pytest.mark.parametrize(("validation", "test", "status"), [
    ((0.2, 5), (0.1, 5), "SUPPORTS_RULE"),
    ((0.2, 5), (0.09, 5), "NOT_SUPPORTED"),
    ((0.05, 5), (0.5, 5), "NOT_SUPPORTED"),
    ((0.9, 2), (0.9, 5), "INSUFFICIENT"),
    ((0.9, 5), (0.9, 2), "INSUFFICIENT"),
    ((0.9, 5), (0.9, 5, -0.001), "NOT_SUPPORTED"),
    ((0.9, 5), (0.9, 5, 0.0), "NOT_SUPPORTED"),
    ((0.9, 5), (0.9, 5, None), "NOT_SUPPORTED"),
])
def test_factor_rule_needs_both_splits_to_clear_the_frozen_bar(card, validation, test, status):
    result = evaluate_card(factor_card(card), factor_report(validation, test))
    assert result["status"] == status
    assert result["metrics"]["test"]["rank_ic_mean"] == test[0]


@pytest.mark.parametrize(("validation", "test", "split_status"), [
    ((0.2, 5), (0.05, 5), {"validation": "SUPPORTS_RULE", "test": "NOT_SUPPORTED"}),
    ((0.05, 5), (0.5, 2), {"validation": "NOT_SUPPORTED", "test": "INSUFFICIENT"}),
])
def test_factor_rule_reports_each_split_on_its_own(card, validation, test, split_status):
    assert evaluate_card(factor_card(card), factor_report(validation, test))["split_status"] == split_status


def test_factor_report_must_be_the_cards_own_signal(card):
    with pytest.raises(ValueError, match="exactly the card's signal"):
        evaluate_card(factor_card(card), factor_report((0.5, 5), (0.5, 5), signal="reversal"))


@pytest.mark.parametrize("rule", [
    {"signal": "value"}, {"lookback": 0}, {"cost_bps": 1.5}, {"min_rank_ic": "0"},
    {"min_rank_ic": "2"}, {"min_valid_dates": 0}, {"min_net_mean": "-1/100"}, {"splits": {"train": {}}}, {"metric": "sharpe"},
])
def test_factor_card_rule_fields_are_bounded(card, rule):
    with pytest.raises(ValueError):
        validate_card(factor_card(card, **rule))
