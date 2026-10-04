"""Causal marked valuations plus separately computed retrospective markouts."""
from bisect import bisect_right
from decimal import Decimal, localcontext
import hashlib
import platform

from . import __version__
from .balances import apply_fill
from .contracts import InputError, integer, load, money
from .provenance import code_identity


def even_indices(length, limit):
    """Deterministic positions including endpoints; no claim to preserve extrema."""
    if length <= limit:
        return set(range(length))
    return {i * (length - 1) // (limit - 1) for i in range(limit)}


def mark_state(mark, timestamp, max_age):
    if mark is None:
        return {"status": "missing", "mark": None, "timestamp": None, "sequence": None,
                "age": None, "bid": None, "ask": None, "spread": None}
    age = timestamp - mark["timestamp"]
    return {"status": "fresh" if age <= max_age else "stale", "mark": money(mark["mark"]),
            "timestamp": mark["timestamp"], "sequence": mark["sequence"], "age": age,
            "bid": money(mark["bid"]), "ask": money(mark["ask"]),
            "spread": money(mark["ask"] - mark["bid"]) if mark["bid"] is not None else None}


def analyze(data, detail="full", sample_limit=200):
    if detail not in ("full", "sampled"):
        raise InputError("detail must be full or sampled")
    integer(sample_limit, "sample_limit", 10, 1000)
    config, events = load(data)
    with localcontext() as context:
        context.prec = 50
        return _analyze(config, events, data, detail, sample_limit)


def _analyze(config, events, data, detail, sample_limit):
    cash = config["initial_cash"]
    positions = {s: v["initial_position"] for s, v in config["instruments"].items()}
    marks = {}
    quote_history = {s: [] for s in positions}
    for symbol, spec in config["instruments"].items():
        if "initial_mark" in spec:
            mark = {"timestamp": config["start_timestamp"], "sequence": -1,
                    "mark": spec["initial_mark"], "bid": None, "ask": None}
            marks[symbol] = mark
            quote_history[symbol].append(mark)
    opening_equity = cash + sum((v["initial_position"] * v.get("initial_mark", Decimal(0))
                                for v in config["instruments"].values()), Decimal(0))
    snapshots, fills = {}, []
    total_fees = Decimal(0)
    selected = even_indices(len(events) + 1, sample_limit - 2) if detail == "sampled" else None
    previous = None
    breach_count = unavailable_count = 0
    breach_by_instrument = {s: 0 for s in positions}
    first_breach = first_unavailable = None

    def snapshot(event, delta_cash, index):
        nonlocal previous, breach_count, unavailable_count, first_breach, first_unavailable
        valuations = {s: mark_state(marks.get(s), event["timestamp"], config["mark_max_age"]) for s in positions}
        issues = [{"instrument": s, "status": v["status"], "age": v["age"]}
                  for s, v in valuations.items() if positions[s] and v["status"] != "fresh"]
        missing = any(positions[s] and v["mark"] is None for s, v in valuations.items())
        position_value = None if missing else sum((positions[s] * marks[s]["mark"]
                                                   for s in positions if positions[s]), Decimal(0))
        last_known = None if missing else cash + position_value
        equity = None if issues else last_known
        previous_equity = Decimal(previous["equity"]) if previous and previous["equity"] is not None else None
        previous_value = Decimal(previous["position_value"]) if previous and previous["position_value"] is not None else None
        valid_value = position_value if not issues else None
        breaches = [{"instrument": s, "position": q, "limit": config["instruments"][s]["limit"]}
                    for s, q in positions.items() if abs(q) > config["instruments"][s]["limit"]]
        row = {"index": index, "timestamp": event["timestamp"], "sequence": event["sequence"],
               "type": event["type"], "instrument": event.get("instrument"), "fill_id": event.get("fill_id"),
               "note": event.get("note", ""), "cash": money(cash), "cash_change": money(delta_cash),
               "positions": dict(positions), "marks": valuations, "position_value": money(valid_value),
               "position_value_change": money(valid_value - previous_value) if valid_value is not None and previous_value is not None else None,
               "equity": money(equity), "equity_change": money(equity - previous_equity) if equity is not None and previous_equity is not None else None,
               "pnl": money(equity - opening_equity) if equity is not None else None,
               "last_known_equity": money(last_known), "valuation_issues": issues, "limit_breaches": breaches}
        breach_count += len(breaches)
        unavailable_count += bool(issues)
        for breach in breaches:
            breach_by_instrument[breach["instrument"]] += 1
        if breaches and first_breach is None:
            first_breach = index
        if issues and first_unavailable is None:
            first_unavailable = index
        if selected is None or index in selected or index in (first_breach, first_unavailable):
            snapshots[index] = row
        previous = row
        return row

    snapshot({"type": "initial", "timestamp": config["start_timestamp"], "sequence": -1}, Decimal(0), 0)
    for index, event in enumerate(events, 1):
        symbol = event["instrument"]
        delta_cash = Decimal(0)
        if event["type"] == "quote":
            marks[symbol] = event
            quote_history[symbol].append(event)
        else:
            cash, positions[symbol], total_fees, delta_cash = apply_fill(
                cash, positions[symbol], total_fees, side=event["side"],
                units=event["units"], price=event["price"], fee=event["fee"])
            current = mark_state(marks.get(symbol), event["timestamp"], config["mark_max_age"])
            fills.append({"fill_id": event["fill_id"], "timestamp": event["timestamp"], "sequence": event["sequence"],
                          "snapshot_index": index, "instrument": symbol, "side": event["side"],
                          "units": event["units"], "price": money(event["price"]), "fee": money(event["fee"]),
                          "at_fill_mark": current, "note": event.get("note", "")})
        snapshot(event, delta_cash, index)

    # Future information is used only here, after causal snapshots are frozen.
    end_timestamp = previous["timestamp"]
    histories = {s: [(m["timestamp"], m["sequence"]) for m in history] for s, history in quote_history.items()}
    markout_counts = {s: 0 for s in ("available", "unavailable_future", "missing", "stale")}
    adverse_count = 0
    first_adverse = None
    for fill_index, fill in enumerate(fills):
        target = fill["timestamp"] + config["markout_horizon"]
        result = {"target_timestamp": target, "status": "unavailable_future", "mark": None,
                  "mark_timestamp": None, "mark_sequence": None, "mark_age": None,
                  "per_unit": None, "gross_total": None}
        if target <= end_timestamp:
            symbol = fill["instrument"]
            at = bisect_right(histories[symbol], (target, 10**15)) - 1
            mark = quote_history[symbol][at] if at >= 0 else None
            status = mark_state(mark, target, config["mark_max_age"])
            result.update(status="available" if status["status"] == "fresh" else status["status"],
                          mark=status["mark"], mark_timestamp=status["timestamp"],
                          mark_sequence=status["sequence"], mark_age=status["age"])
            if status["status"] == "fresh":
                per_unit = (mark["mark"] - Decimal(fill["price"])) * (1 if fill["side"] == "buy" else -1)
                result.update(per_unit=money(per_unit), gross_total=money(per_unit * fill["units"]))
        fill["markout"] = result
        markout_counts[result["status"]] += 1
        if result["gross_total"] is not None and Decimal(result["gross_total"]) < 0:
            adverse_count += 1
            if first_adverse is None:
                first_adverse = fill_index

    fill_count = len(fills)
    if detail == "sampled":
        selected_fills = even_indices(len(fills), sample_limit - 1)
        if first_adverse is not None:
            selected_fills.add(first_adverse)
        fills = [f for i, f in enumerate(fills) if i in selected_fills]

    instruments = {s: {"initial_position": spec["initial_position"], "limit": spec["limit"],
                       "initial_mark": money(spec.get("initial_mark"))} for s, spec in config["instruments"].items()}
    return {"schema": "imc4-analysis-report/v1", "tool_version": __version__,
            "code_identity": code_identity(), "runtime": {"implementation": platform.python_implementation(), "python_version": platform.python_version()},
            "source_sha256": hashlib.sha256(data).hexdigest(), "source_bytes": len(data),
            "label": config["label"], "data_kind": config["data_kind"],
            "provenance": "New post-competition tooling; original team sources remain unrecovered. No official-log or simulator equivalence.",
            "currency": config["currency"], "timestamp_unit": config["timestamp_unit"],
            "mark_max_age": config["mark_max_age"], "markout_horizon": config["markout_horizon"],
            "instruments": instruments, "opening_equity": money(opening_equity),
            "summary": {"event_count": len(events), "fill_count": fill_count, "total_fees": money(total_fees),
                        "cash": money(cash), "positions": dict(positions), "equity": previous["equity"],
                        "pnl": previous["pnl"], "last_known_equity": previous["last_known_equity"],
                        "breach_observations": breach_count, "breach_observations_by_instrument": breach_by_instrument,
                        "unavailable_valuation_snapshots": unavailable_count, "first_breach_index": first_breach,
                        "first_unavailable_valuation_index": first_unavailable,
                        "markout_status_counts": markout_counts, "adverse_markout_fills": adverse_count},
            "details": {"mode": detail, "sample_limit": sample_limit, "total_snapshots": len(events) + 1,
                        "retained_snapshots": len(snapshots), "retained_fills": len(fills),
                        "selection": "All rows" if detail == "full" else "Evenly spaced rows plus first breach, first unavailable valuation and first adverse fill; not extrema preserving. Aggregates use every event."},
            "snapshots": [snapshots[i] for i in sorted(snapshots)], "fills": fills}
