"""Finite synthetic execution study; the existing analyzer owns cash valuation."""
import copy
from decimal import Decimal, localcontext
import hashlib
import json
import re

from .analyzer import analyze
from .balances import apply_fill
from .contracts import InputError, decimal, integer, keys, money, no_duplicate_keys
from .quoting import quote_orders, validate_order, validate_settings, _state as validate_observation

MAX_STUDY_BYTES = 256 * 1024


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", value):
        raise InputError("scenario/execution id must be a safe lowercase identifier (1-40 characters)")
    return value


def load_study(raw, *, _check_policy=True):
    if len(raw) > MAX_STUDY_BYTES:
        raise InputError("study exceeds 256 KiB")
    try:
        study = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys,
                           parse_constant=lambda value: (_ for _ in ()).throw(InputError("nonfinite study value")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(f"invalid study JSON: {exc}") from exc
    keys(study, ["schema", "data_kind", "scenarios", "executions"])
    if study["schema"] != "imc4-quote-study/v1" or study["data_kind"] != "synthetic":
        raise InputError("this controlled study accepts imc4-quote-study/v1 synthetic inputs only")
    for group, limit in (("scenarios", 8), ("executions", 4)):
        if not isinstance(study[group], list) or not 1 <= len(study[group]) <= limit:
            raise InputError(f"{group} must contain 1 to {limit} entries")
        identities = set()
        for record in study[group]:
            if not isinstance(record, dict): raise InputError("study entries must be objects")
            identity = _identifier(record.get("id"))
            if identity in identities: raise InputError("duplicate study identifier")
            identities.add(identity)
    for execution in study["executions"]:
        keys(execution, ["id", "queue_ahead_units", "cancel_delay_steps"])
        integer(execution["queue_ahead_units"], "queue_ahead_units", 0, 10**6)
        integer(execution["cancel_delay_steps"], "cancel_delay_steps", 0, 2)
    for scenario in study["scenarios"]:
        keys(scenario, ["id", "policy", "initial_cash", "initial_position", "fee_per_unit", "end_timestamp", "steps"])
        settings = validate_settings(scenario["policy"])
        decimal(scenario["initial_cash"], "initial_cash")
        decimal(scenario["fee_per_unit"], "fee_per_unit", nonnegative=True)
        end = integer(scenario["end_timestamp"], "end_timestamp", 1, 64)
        if end > settings["max_horizon"]:
            raise InputError("scenario horizon exceeds policy maximum")
        if not isinstance(scenario["steps"], list) or len(scenario["steps"]) != end + 1:
            raise InputError("scenario requires one contiguous observation per tick, including terminal mark")
        for index, step in enumerate(scenario["steps"]):
            keys(step, ["timestamp", "fair_price", "volatility", "seller_floor", "seller_units", "buyer_ceiling", "buyer_units"])
            if type(step["timestamp"]) is not int or step["timestamp"] != index:
                raise InputError("study timestamps must be contiguous integer ticks starting at zero")
            for name in ("fair_price", "seller_floor", "buyer_ceiling"):
                decimal(step[name], name, positive=True)
            decimal(step["volatility"], "volatility", nonnegative=True)
            for name in ("seller_units", "buyer_units"):
                integer(step[name], name, 0, 10**6)
                if step[name] % settings["lot_size"]:
                    raise InputError("external quantities must be whole lots")
            if index == end and (step["seller_units"] or step["buyer_units"]):
                raise InputError("terminal tick is valuation/expiry only; demand must be zero")
        for execution in study["executions"]:
            if execution["queue_ahead_units"] % settings["lot_size"]:
                raise InputError("queue_ahead_units must be whole lots")
        observation = {"fair_price": scenario["steps"][0]["fair_price"],
                       "volatility": scenario["steps"][0]["volatility"], "inventory": scenario["initial_position"],
                       "remaining_horizon": end, "outstanding_buy": 0, "outstanding_sell": 0}
        if _check_policy:
            quote_orders(scenario["policy"], observation)
        else:
            # Resume admission checks feasibility without generating an old decision.
            validate_observation(settings, observation)
    return study


BOUNDARY_SCHEMA = "imc4-quote-boundary/v1"
COUNTERS = ("peak", "area", "pending_units", "partials", "submitted", "rejected")


def _configuration(scenario, policy_name):
    if policy_name not in ("symmetric", "inventory"):
        raise InputError("unknown quote policy")
    config = copy.deepcopy(scenario["policy"])
    if policy_name == "symmetric":
        config["inventory_penalty"] = "0"
    return config


def _binding(scenario, execution, policy_name):
    return hashlib.sha256(encoded(dict(scenario=scenario, execution=execution, policy=policy_name))).hexdigest()


def _genesis(scenario, execution, policy_name):
    return dict(schema=BOUNDARY_SCHEMA,
                run=f"{scenario['id']}--{execution['id']}--{policy_name}",
                input_sha256=_binding(scenario, execution, policy_name), next_step=0,
                inventory=scenario["initial_position"], cash=money(Decimal(scenario["initial_cash"])),
                total_fees="0", live=[],
                counters=dict(peak=abs(scenario["initial_position"]), area=0, pending_units=0,
                              partials=0, submitted=0, rejected=0),
                event_count=0, last_event_key=None)


def initialize_scenario(scenario, execution, policy_name):
    """Genesis for fixed inputs admitted by load_study; returns plain JSON state."""
    state = _genesis(scenario, execution, policy_name)
    validate_boundary(scenario, execution, policy_name, state)
    return state


def _derived_decimal(value, name, bound, nonnegative=False):
    # A bounded derived balance is not a new normalized-input monetary value.
    if not isinstance(value, str) or not re.fullmatch(r"-?(0|[1-9][0-9]{0,24})(\.[0-9]{1,8})?", value):
        raise InputError(f"invalid canonical derived {name}")
    number = Decimal(value)
    if money(number) != value or abs(number) > bound or (nonnegative and number < 0):
        raise InputError(f"invalid derived {name}")
    return number


def validate_boundary(scenario, execution, policy_name, state):
    """Check structure and fixed-input binding, never assert history authenticity.

    No policy/matcher call or event-prefix replay is needed to load bounded state.
    A later journal must separately verify the lineage of a plausible state.
    """
    with localcontext() as context:
        context.prec = 80
        _validate_boundary(scenario, execution, policy_name, state)


def _validate_boundary(scenario, execution, policy_name, state):
    keys(state, ["schema", "run", "input_sha256", "next_step", "inventory", "cash", "total_fees",
                 "live", "counters", "event_count", "last_event_key"])
    config = validate_settings(_configuration(scenario, policy_name))
    if (state["schema"] != BOUNDARY_SCHEMA or
            state["run"] != f"{scenario['id']}--{execution['id']}--{policy_name}" or
            state["input_sha256"] != _binding(scenario, execution, policy_name)):
        raise InputError("boundary schema or fixed input binding mismatch")
    end = scenario["end_timestamp"]
    n = integer(state["next_step"], "next_step", 0, end + 1)
    q = integer(state["inventory"], "inventory", config["min_position"], config["max_position"])
    if q % config["lot_size"]:
        raise InputError("boundary inventory must be whole lots")
    max_live = 2 * (execution["cancel_delay_steps"] + 1)
    # Conservative admitted-step bound, deliberately above input-only 1e12.
    max_fills = min(n, end) * max_live
    fee_bound = max_fills * 10**6 * Decimal(scenario["fee_per_unit"])
    cash_bound = abs(Decimal(scenario["initial_cash"])) + max_fills * 10**6 * Decimal(10**12) + fee_bound
    _derived_decimal(state["cash"], "cash", cash_bound)
    _derived_decimal(state["total_fees"], "total_fees", fee_bound, nonnegative=True)
    live = state["live"]
    if not isinstance(live, list) or len(live) > max_live or (n in (0, end + 1) and live):
        raise InputError("invalid boundary live order count")
    seen, previous = set(), None
    outstanding = {"buy": 0, "sell": 0}
    for order in live:
        keys(order, ["order_id", "side", "price", "units", "remaining_units", "created", "cancel_due"])
        side = order["side"]
        if side not in ("buy", "sell"):
            raise InputError("invalid live order side")
        created = integer(order["created"], "created", 0, n - 1)
        identity = f"{policy_name}-{created}-{side}"
        ordering = (created, 0 if side == "buy" else 1)
        if order["order_id"] != identity or identity in seen or (previous is not None and ordering <= previous):
            raise InputError("invalid live order identity/order")
        seen.add(identity); previous = ordering
        units = integer(order["units"], "original units", 1, config["order_size"])
        remaining = integer(order["remaining_units"], "remaining units", 1, units)
        if units % config["lot_size"] or remaining % config["lot_size"]:
            raise InputError("live units must be whole lots")
        price = decimal(order["price"], "live price", positive=True)
        if price % config["tick_size"]:
            raise InputError("live price must be on tick")
        due = order["cancel_due"]
        if due is None:
            if created != n - 1:
                raise InputError("old live order lacks cancellation clock")
        else:
            integer(due, "cancel_due", n, end + 2)
            if created >= n - 1 or due != created + 1 + execution["cancel_delay_steps"]:
                raise InputError("invalid cancellation clock")
        outstanding[side] += remaining
    if q + outstanding["buy"] > config["max_position"] or q - outstanding["sell"] < config["min_position"]:
        raise InputError("infeasible gross outstanding capacity")
    keys(state["counters"], COUNTERS)
    c = state["counters"]
    limit = max(abs(config["min_position"]), abs(config["max_position"]))
    integer(c["peak"], "peak", max(abs(q), abs(scenario["initial_position"])), limit)
    integer(c["area"], "area", 0, min(n, end) * limit**2)
    integer(c["pending_units"], "pending_units", 0, max_fills * 10**6)
    integer(c["partials"], "partials", 0, max_fills)
    for name in ("submitted", "rejected"):
        integer(c[name], name, 0, 2 * min(n, end))
    if c["submitted"] + c["rejected"] > 2 * min(n, end):
        raise InputError("invalid order counters")
    integer(state["event_count"], "event_count", n, n + max_fills)
    last = state["last_event_key"]
    if n == 0:
        if state != _genesis(scenario, execution, policy_name):
            raise InputError("boundary genesis mismatch")
    else:
        if not isinstance(last, list) or len(last) != 2:
            raise InputError("invalid last event key")
        integer(last[0], "last event tick", n - 1, n - 1)
        integer(last[1], "last event sequence", 0, 0 if n == end + 1 else max_live)
        if state["event_count"] < n + last[1]:
            raise InputError("event count/sequence mismatch")


def advance_step(scenario, execution, policy_name, state, step):
    """Return a new complete boundary and one delta, without mutating any input."""
    validate_boundary(scenario, execution, policy_name, state)
    n = state["next_step"]
    if n >= len(scenario["steps"]):
        raise InputError("scenario is already complete")
    if encoded(step) != encoded(scenario["steps"][n]):
        raise InputError("step is not the fixed next input")
    result = copy.deepcopy(state)
    config = _configuration(scenario, policy_name)
    q, end = result["inventory"], scenario["end_timestamp"]
    live, events = result["live"], []
    peak, area, pending_units, partials, submitted, rejected = (result["counters"][name] for name in COUNTERS)
    cash, total_fees, projected_q = Decimal(result["cash"]), Decimal(result["total_fees"]), q

    def observed(step):
        return dict(fair_price=step["fair_price"], volatility=step["volatility"], inventory=q,
                    remaining_horizon=end-step["timestamp"],
                    outstanding_buy=sum(o["remaining_units"] for o in live if o["side"] == "buy"),
                    outstanding_sell=sum(o["remaining_units"] for o in live if o["side"] == "sell"))

    with localcontext() as context:
        context.prec = 80
        fee = decimal(scenario["fee_per_unit"], "fee_per_unit", nonnegative=True)
        t = step["timestamp"]
        entry = dict(timestamp=t, cancellations=[], accepted=[], rejected=[], fills=[], matches=[])
        # Venue acknowledgements precede new requests; pending orders remain executable.
        for order in list(live):
            if t == end or (order["cancel_due"] is not None and order["cancel_due"] <= t):
                entry["cancellations"].append(dict(order_id=order["order_id"], units=order["remaining_units"],
                                                   status="expired" if t == end else "acknowledged"))
                live.remove(order)
        for order in list(live):
            if order["cancel_due"] is None:
                order["cancel_due"] = t + execution["cancel_delay_steps"]
                entry["cancellations"].append(dict(order_id=order["order_id"], units=order["remaining_units"],
                                                   status="requested", due=order["cancel_due"]))
                if order["cancel_due"] == t:
                    entry["cancellations"].append(dict(order_id=order["order_id"], units=order["remaining_units"], status="acknowledged"))
                    live.remove(order)
        observed_state = observed(step)
        decision = quote_orders(config, observed_state)
        entry.update(state_before=dict(observed_state), decision=decision, exogenous=dict(step))
        for intent in decision["orders"]:
            try:
                validate_order(intent, config, observed(step), live)
            except InputError as exc:
                entry["rejected"].append(dict(intent=intent, reason=str(exc))); rejected += 1
                continue
            order = dict(intent, order_id=f"{policy_name}-{t}-{intent['side']}", created=t,
                         remaining_units=intent["units"], cancel_due=None)
            live.append(order); submitted += 1
            entry["accepted"].append(dict(order))
        events.append(dict(type="quote", timestamp=t, sequence=0, instrument="ASSET", mark=step["fair_price"]))
        sequence = 0
        for side, flow_name, price_name in (("buy", "seller_units", "seller_floor"), ("sell", "buyer_units", "buyer_ceiling")):
            budget = max(0, step[flow_name] - execution["queue_ahead_units"])
            original_budget = budget
            boundary = decimal(step[price_name], price_name, positive=True)
            eligible = [o for o in live if o["side"] == side and
                        ((side == "buy" and Decimal(o["price"]) >= boundary) or
                         (side == "sell" and Decimal(o["price"]) <= boundary))]
            eligible.sort(key=lambda o: ((-1 if side == "buy" else 1)*Decimal(o["price"]), o["created"]))
            for order in eligible:
                units = min(budget, order["remaining_units"])
                if units == 0: break
                before = order["remaining_units"]
                pending = order["cancel_due"] is not None
                q += units if side == "buy" else -units
                if not config["min_position"] <= q <= config["max_position"]:
                    raise InputError("execution violated hard position bounds")
                budget -= units; order["remaining_units"] -= units
                sequence += 1; peak = max(peak, abs(q)); pending_units += units if pending else 0
                partials += int(units < before)
                fill = dict(type="fill", timestamp=t, sequence=sequence, instrument="ASSET",
                            fill_id=f"{order['order_id']}-fill-{t}", side=side, units=units,
                            price=order["price"], fee=money(fee*units))
                events.append(fill)
                cash, projected_q, total_fees, _ = apply_fill(
                    cash, projected_q, total_fees, side=side, units=units,
                    price=Decimal(fill["price"]), fee=Decimal(fill["fee"]))
                if projected_q != q:
                    raise InputError("execution/accounting position mismatch")
                entry["fills"].append(dict(fill, order_id=order["order_id"], inventory_after=q,
                                            pending_cancel=pending, order_units_before=before))
                if order["remaining_units"] == 0: live.remove(order)
            entry["matches"].append(dict(own_side=side, external_units=step[flow_name],
                                          residual_after_queue=original_budget, filled_units=original_budget-budget))
        area += q*q if t < end else 0
        entry.update(inventory_after=q, live_after=copy.deepcopy(live))
    result.update(next_step=n + 1, inventory=q, cash=money(cash), total_fees=money(total_fees),
                  counters=dict(zip(COUNTERS, (peak, area, pending_units, partials, submitted, rejected))),
                  event_count=state["event_count"] + len(events), last_event_key=[n, events[-1]["sequence"]])
    validate_boundary(scenario, execution, policy_name, result)
    return result, dict(trace=entry, events=events)


def finish_scenario(scenario, execution, policy_name, state, deltas):
    """Construct the existing report from deltas; never replay decisions/matches."""
    validate_boundary(scenario, execution, policy_name, state)
    if state["next_step"] != len(scenario["steps"]) or len(deltas) != state["next_step"]:
        raise InputError("finalization requires terminal state and complete delta history")
    events, trace = [], []
    for t, delta in enumerate(deltas):
        keys(delta, ["trace", "events"])
        if delta["trace"]["timestamp"] != t or not delta["events"]:
            raise InputError("noncontiguous trace delta")
        for sequence, event in enumerate(delta["events"]):
            if (event["timestamp"], event["sequence"]) != (t, sequence):
                raise InputError("noncontiguous event delta")
        events.extend(copy.deepcopy(delta["events"]))
        trace.append(copy.deepcopy(delta["trace"]))
    if len(events) != state["event_count"] or [events[-1]["timestamp"], events[-1]["sequence"]] != state["last_event_key"]:
        raise InputError("terminal event count/key mismatch")
    if trace[-1]["live_after"] != state["live"] or trace[-1]["inventory_after"] != state["inventory"]:
        raise InputError("terminal trace/state mismatch")
    config = _configuration(scenario, policy_name)
    q = scenario["initial_position"]
    config_row = dict(type="config", schema="imc4-analysis/v1", label=f"Synthetic {scenario['id']} / {execution['id']} / {policy_name}",
                      data_kind="synthetic", currency="SYNTHETIC_CCY", timestamp_unit="synthetic_tick", start_timestamp=0,
                      initial_cash=scenario["initial_cash"], mark_max_age=1, markout_horizon=1,
                      instruments={"ASSET":dict(initial_position=q, initial_mark=scenario["steps"][0]["fair_price"],
                                                limit=max(abs(config["min_position"]), abs(config["max_position"])))})
    normalized = b"".join(encoded(row) for row in [config_row] + events)
    report = analyze(normalized, detail="full")
    summary = report["summary"]
    if (summary["cash"], summary["positions"]["ASSET"], summary["total_fees"]) != (state["cash"], state["inventory"], state["total_fees"]):
        raise InputError("terminal accounting projection mismatch")
    peak, area, pending_units, partials, submitted, rejected = (state["counters"][name] for name in COUNTERS)
    live = state["live"]
    metrics = dict(peak_absolute_inventory=peak, inventory_square_time=area,
                   inventory_square_time_unit="units_squared_times_synthetic_ticks", pending_cancel_fill_units=pending_units,
                   partial_fill_events=partials, accepted_orders=submitted, rejected_orders=rejected,
                   terminal_live_orders=len(live), terminal_liquidation="none; remaining inventory marked")
    return dict(policy=policy_name, policy_settings=config, trace=trace, normalized=normalized, report=report, metrics=metrics)



def run_scenario(scenario, execution, policy_name):
    """Run admitted fixed inputs through the same explicit complete-step seam."""
    state = initialize_scenario(scenario, execution, policy_name)
    deltas = []
    for step in scenario["steps"]:
        state, delta = advance_step(scenario, execution, policy_name, state, step)
        deltas.append(delta)
    return finish_scenario(scenario, execution, policy_name, state, deltas)


def finish_study(raw, runs):
    """Build the existing comparison from supplied native runs; no simulation."""
    study = json.loads(raw)
    rows = []
    expected_names = [f"{s['id']}--{e['id']}--{p}" for s in study["scenarios"]
                      for e in study["executions"] for p in ("symmetric", "inventory")]
    if list(runs) != expected_names:
        raise InputError("study run identities/order mismatch")
    for scenario in study["scenarios"]:
        exogenous_hash = hashlib.sha256(encoded(scenario["steps"])).hexdigest()
        for execution in study["executions"]:
            for policy in ("symmetric", "inventory"):
                name = f"{scenario['id']}--{execution['id']}--{policy}"
                run = runs[name]
                rows.append(dict(run=name, scenario=scenario["id"], execution=execution, policy=policy,
                                 exogenous_sha256=exogenous_hash, normalized_sha256=run["report"]["source_sha256"],
                                 summary=run["report"]["summary"], metrics=run["metrics"]))
    first = next(iter(runs.values()))["report"]
    comparison = dict(schema="imc4-quote-study-result/v1", source_sha256=hashlib.sha256(raw).hexdigest(),
                      code_identity=first["code_identity"], runtime=first["runtime"], rows=rows,
                      assumptions="Synthetic willingness-price demand with queue haircut, own price/time allocation and specified cancellation delay. Same external inputs, different endogenous orders/fills. No queue reconstruction, impact, fitted arrivals, cash/margin constraint or terminal liquidation. No market-alpha or official-engine claim.")
    return comparison, runs


def run_study(raw):
    study = load_study(raw)
    runs = {}
    for scenario in study["scenarios"]:
        for execution in study["executions"]:
            for policy in ("symmetric", "inventory"):
                name = f"{scenario['id']}--{execution['id']}--{policy}"
                runs[name] = run_scenario(scenario, execution, policy)
    return finish_study(raw, runs)
