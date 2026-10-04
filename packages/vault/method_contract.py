"""Offline, source-bound research cards; no retrieval, model or execution engine."""

from __future__ import annotations

import hashlib
import re
from fractions import Fraction
from pathlib import Path

from source_access import inspect_note

MAX_NOTE_BYTES = 256 * 1024
CARD_FIELDS = {
    "schema",
    "id",
    "method",
    "hypothesis",
    "mechanism",
    "assumptions",
    "limitations",
    "sources",
    "decision",
}
DECISION_FIELDS = {
    "split",
    "metric",
    "candidate",
    "baseline",
    "min_pairs",
    "min_coverage",
    "min_improvement_ticks",
}
QUOTE_METHOD = "quote-weighted-midpoint/v1"
FACTOR_METHOD = "factor-rank-ic/v1"
# A factor card fixes the signal, its parameters, the splits and the bar it must
# clear before any evaluation; the factor lab supplies the numbers afterwards.
FACTOR_DECISION_FIELDS = {
    "signal",
    "lookback",
    "cost_bps",
    "splits",
    "metric",
    "min_rank_ic",
    "min_valid_dates",
    "min_net_mean",
}


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value
    ):
        raise ValueError("safe 1..64 character identifier required")
    return value


def rational(value):
    if (
        not isinstance(value, str)
        or len(value) > 32
        or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value)
    ):
        raise ValueError("bounded nonnegative canonical rational string required")
    result = Fraction(value)
    if str(result) != value:
        raise ValueError("canonical rational string required")
    return result


def validate_card(card):
    if (
        not isinstance(card, dict)
        or set(card) != CARD_FIELDS
        or card["schema"] != "vault-method/v1"
    ):
        raise ValueError("exact vault-method/v1 card required")
    safe_id(card["id"])
    if card["method"] not in {QUOTE_METHOD, FACTOR_METHOD}:
        raise ValueError("unsupported method; cards cannot supply executable code")
    for key in ("hypothesis", "mechanism"):
        if not isinstance(card[key], str) or not 1 <= len(card[key].strip()) <= 4000:
            raise ValueError("explicit bounded hypothesis and mechanism required")
    for key in ("assumptions", "limitations"):
        if (
            not isinstance(card[key], list)
            or not 1 <= len(card[key]) <= 16
            or any(
                not isinstance(item, str) or not 1 <= len(item.strip()) <= 1000
                for item in card[key]
            )
        ):
            raise ValueError("explicit bounded assumptions and limitations required")
    if card["method"] == QUOTE_METHOD:
        _validate_quote_rule(card["decision"])
    else:
        _validate_factor_rule(card["decision"])
    sources = card["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError("1..8 explicit source notes required")
    ids = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {
            "id",
            "kind",
            "path",
            "sha256",
            "relationship",
            "use_scope",
        }:
            raise ValueError("exact source fields required")
        identity = safe_id(source["id"])
        if identity in ids:
            raise ValueError("duplicate source identity")
        ids.add(identity)
        if source["kind"] not in {"local_derivation", "paper_note"} or source[
            "relationship"
        ] not in {"derivation", "background"}:
            raise ValueError("explicit source kind and relationship required")
        if source["use_scope"] != "local_review":
            raise ValueError(
                "this card grants no publication or source redistribution rights"
            )
        path = source["path"]
        if (
            not isinstance(path, str)
            or not 1 <= len(path) <= 1024
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or any(ord(c) < 32 for c in path)
        ):
            raise ValueError("source path must stay within the card directory")
        if not isinstance(source["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", source["sha256"]
        ):
            raise ValueError("explicit source digest required")
    return card


def _validate_quote_rule(rule):
    if not isinstance(rule, dict) or set(rule) != DECISION_FIELDS:
        raise ValueError("exact decision fields required")
    if rule["split"] not in {"train", "validation", "test"} or (
        rule["metric"],
        rule["candidate"],
        rule["baseline"],
    ) != ("mae_ticks", "weighted_midpoint", "midpoint"):
        raise ValueError("fixed native MAE comparison required")
    if type(rule["min_pairs"]) is not int or not 1 <= rule["min_pairs"] <= 100_000:
        raise ValueError("min_pairs must be an integer in 1..100000")
    if not 0 < rational(rule["min_coverage"]) <= 1:
        raise ValueError("coverage must be in (0, 1]")
    rational(rule["min_improvement_ticks"])


def _validate_factor_rule(rule):
    if not isinstance(rule, dict) or set(rule) != FACTOR_DECISION_FIELDS:
        raise ValueError("exact factor decision fields required")
    if rule["signal"] not in {"momentum", "reversal"} or rule["metric"] != "rank_ic_mean":
        raise ValueError("factor cards compare the lab's momentum/reversal rank IC only")
    if type(rule["lookback"]) is not int or not 1 <= rule["lookback"] <= 250:
        raise ValueError("lookback must be an integer in 1..250")
    if type(rule["cost_bps"]) is not int or not 0 <= rule["cost_bps"] <= 10_000:
        raise ValueError("cost_bps must be an integer in 0..10000")
    if type(rule["min_valid_dates"]) is not int or not 1 <= rule["min_valid_dates"] <= 5000:
        raise ValueError("min_valid_dates must be an integer in 1..5000")
    if not 0 < rational(rule["min_rank_ic"]) <= 1:
        raise ValueError("min_rank_ic must be in (0, 1]")
    rational(rule["min_net_mean"])
    splits = rule["splits"]
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"} or any(
        not isinstance(b, dict) or set(b) != {"start", "end"}
        or not all(isinstance(t, str) and 1 <= len(t) <= 40 for t in b.values())
        for b in splits.values()
    ):
        raise ValueError("explicit train/validation/test start/end timestamps required")


def _linked(path):
    # Keep this standard-library-only; the vault has no dependency on QRAE.
    for node in (path, *path.parents):
        if node.is_symlink():
            return True
        try:
            if getattr(node.lstat(), "st_file_attributes", 0) & 0x400:
                return True
        except FileNotFoundError:
            continue
    return False


def read_sources(card, base: Path, *, frozen=False):
    """Read only declared current notes; returned bytes are the snapshots to retain."""
    validate_card(card)
    snapshots = {}
    for i, source in enumerate(card["sources"]):
        name = f"source-{i}.md"
        path = base / (name if frozen else source["path"])
        if (
            _linked(path)
            or not path.is_file()
            or not path.resolve().is_relative_to(base.resolve())
        ):
            raise ValueError(
                "source must be a regular unlinked file within its directory"
            )
        with path.open("rb") as handle:
            raw = handle.read(MAX_NOTE_BYTES + 1)
        if len(raw) > MAX_NOTE_BYTES:
            raise ValueError("source note exceeds 256 KiB")
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("source digest differs; revise the card explicitly")
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValueError("source note must contain text")
        if source["kind"] == "paper_note":
            decision = inspect_note(
                {
                    "document_type": "paper_note",
                    "arxiv_id": source["id"],
                    "source_available": True,
                    "source_status": "present",
                    "source_sha256": source["sha256"],
                    "vault_path": str(path),
                },
                text,
                source_root=base,
            )
            if not decision["eligible"]:
                raise ValueError("paper note ineligible: " + decision["reason"])
        snapshots[name] = raw
    return snapshots


def evaluate_card(card, report):
    """Exact descriptive rule; SUPPORTS_RULE is not statistical or economic proof."""
    validate_card(card)
    if card["method"] == FACTOR_METHOD:
        return _evaluate_factor(card["decision"], report)
    if report.get("schema") not in {
        "quote-forecast-report/v1",
        "quote-forecast-report/v2",
    }:
        raise ValueError("native quote forecast report required")
    rule = card["decision"]
    metrics = report["splits"][rule["split"]]
    count, possible = metrics["evaluated_pairs"], metrics["candidate_pairs"]
    if (
        type(count) is not int
        or type(possible) is not int
        or not 0 <= count <= possible
        or possible < 1
    ):
        raise ValueError("invalid native pair counts")
    coverage = Fraction(count, possible)
    if rational(metrics["coverage"]) != coverage:
        raise ValueError("native coverage differs from counts")
    candidate = metrics["models"][rule["candidate"]]["mae_ticks"]
    baseline = metrics["models"][rule["baseline"]]["mae_ticks"]
    if count:
        improvement = rational(baseline) - rational(candidate)
    elif candidate is not None or baseline is not None:
        raise ValueError("empty evaluation must retain null errors")
    else:
        improvement = None
    enough = count >= rule["min_pairs"] and coverage >= rational(rule["min_coverage"])
    status = (
        "INSUFFICIENT"
        if not enough
        else (
            "SUPPORTS_RULE"
            if improvement > rational(rule["min_improvement_ticks"])
            else "NOT_SUPPORTED"
        )
    )
    return {
        "status": status,
        "split": rule["split"],
        "metric": "mae_ticks",
        "candidate_mae_ticks": candidate,
        "baseline_mae_ticks": baseline,
        "improvement_ticks": str(improvement) if improvement is not None else None,
        "evaluated_pairs": count,
        "candidate_pairs": possible,
        "coverage": str(coverage),
        "decision_rule": rule,
        "statistical_significance_evaluated": False,
        "profitability_evaluated": False,
        "untouched_holdout_verified": False,
    }


def _evaluate_factor(rule, report):
    """Validation and the once-exposed test split must both clear the card's bars:
    mean rank IC and mean net-of-cost sleeve P&L, on enough valid dates."""
    if report.get("schema_version") not in {"factor-report/v1", "factor-report/v2"}:
        raise ValueError("native factor report required")
    signal = rule["signal"]
    if report["selection"]["selected"] not in {signal, None} or list(report["development"]) != [signal]:
        raise ValueError("factor report must evaluate exactly the card's signal")
    summaries = {"validation": report["development"][signal]["validation"]["summary"],
                 "test": report["test"]["summary"] if report["test"] is not None else None}
    ic_bar, net_bar = rational(rule["min_rank_ic"]), rational(rule["min_net_mean"])
    metrics, split_status = {}, {}
    for split, summary in summaries.items():
        summary = summary or {"rank_ic_mean": None, "valid_ic_dates": 0, "net_mean": None}
        ic, dates, net = summary["rank_ic_mean"], summary["valid_ic_dates"], summary["net_mean"]
        metrics[split] = {"rank_ic_mean": ic, "valid_ic_dates": dates, "net_mean": net}
        split_status[split] = ("INSUFFICIENT" if dates < rule["min_valid_dates"] else "SUPPORTS_RULE"
                               if ic >= ic_bar and net is not None and net > net_bar else "NOT_SUPPORTED")
    statuses = set(split_status.values())
    return {
        "status": "INSUFFICIENT" if "INSUFFICIENT" in statuses else "NOT_SUPPORTED"
                  if "NOT_SUPPORTED" in statuses else "SUPPORTS_RULE",
        # Per split, so a caller can filter on validation and keep test for a later gate.
        "split_status": split_status,
        "metric": "rank_ic_mean",
        "signal": signal,
        "lookback": rule["lookback"],
        "metrics": metrics,
        "decision_rule": rule,
        "statistical_significance_evaluated": False,
        "profitability_evaluated": False,
        "untouched_holdout_verified": False,
    }
