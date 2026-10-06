"""Read-only projection of committed effects; never generates policy/matches."""
import copy
from decimal import Decimal, localcontext

from .balances import apply_fill
from .contracts import InputError, keys, load, money
from . import quote_study as qs


def project(scenario, execution, policy, previous, trace, events):
    """Validate effect lineage and derive the complete next boundary.

    This checks recorded effects against native admission/accounting primitives;
    it does not authenticate a hostile rewrite of all records and commitments.
    """
    with localcontext() as context:
        context.prec = 80
        return _project(scenario, execution, policy, previous, trace, events)


def _project(scenario, execution, policy, previous, trace, events):
    qs.validate_boundary(scenario, execution, policy, previous)
    t = previous["next_step"]; end = scenario["end_timestamp"]
    if t > end: raise InputError("effect after terminal")
    step = scenario["steps"][t]
    keys(trace, ["timestamp", "cancellations", "accepted", "rejected", "fills", "matches",
                 "state_before", "decision", "exogenous", "inventory_after", "live_after"])
    if trace["timestamp"] != t or qs.encoded(trace["exogenous"]) != qs.encoded(step):
        raise InputError("effect input/clock mismatch")
    state = copy.deepcopy(previous); live = state["live"]; c = state["counters"]
    config = qs._configuration(scenario, policy)
    cancellations = []
    for order in list(live):
        if t == end or (order["cancel_due"] is not None and order["cancel_due"] <= t):
            cancellations.append(dict(order_id=order["order_id"], units=order["remaining_units"],
                                      status="expired" if t == end else "acknowledged"))
            live.remove(order)
    for order in list(live):
        if order["cancel_due"] is None:
            order["cancel_due"] = t + execution["cancel_delay_steps"]
            cancellations.append(dict(order_id=order["order_id"], units=order["remaining_units"],
                                      status="requested", due=order["cancel_due"]))
            if order["cancel_due"] == t:
                cancellations.append(dict(order_id=order["order_id"], units=order["remaining_units"], status="acknowledged"))
                live.remove(order)
    if cancellations != trace["cancellations"]: raise InputError("cancellation effect mismatch")
    q = previous["inventory"]
    def observed():
        return dict(fair_price=step["fair_price"], volatility=step["volatility"], inventory=q,
                    remaining_horizon=end-t, outstanding_buy=sum(o["remaining_units"] for o in live if o["side"]=="buy"),
                    outstanding_sell=sum(o["remaining_units"] for o in live if o["side"]=="sell"))
    if observed() != trace["state_before"]: raise InputError("decision observation mismatch")
    decision = trace["decision"]
    keys(decision,["reservation_price","inventory_adjustment","capacity_units","orders","suppressed","status"])
    if not isinstance(decision["orders"],list) or len(decision["orders"])>2:
        raise InputError("invalid recorded intents")
    if t==end and decision["orders"]:raise InputError("recorded terminal order")
    accepted, rejected = [], []
    for intent in decision["orders"]:
        try: qs.validate_order(intent,config,observed(),live)
        except InputError as exc:
            rejected.append(dict(intent=intent,reason=str(exc)));c["rejected"]+=1
        else:
            order=dict(intent,order_id=f"{policy}-{t}-{intent['side']}",created=t,
                       remaining_units=intent["units"],cancel_due=None)
            accepted.append(copy.deepcopy(order));live.append(order);c["submitted"]+=1
    if accepted!=trace["accepted"] or rejected!=trace["rejected"]:raise InputError("order acceptance effect mismatch")
    # Reuse the normalized parser on this block for monetary/type validation.
    config_row=dict(type="config",schema="imc4-analysis/v1",label="Persisted effect audit",data_kind="synthetic",
                    currency="SYNTHETIC_CCY",timestamp_unit="synthetic_tick",start_timestamp=0,
                    initial_cash=scenario["initial_cash"],mark_max_age=1,markout_horizon=1,
                    instruments={"ASSET":dict(initial_position=scenario["initial_position"],
                        initial_mark=scenario["steps"][0]["fair_price"],limit=max(abs(config["min_position"]),abs(config["max_position"])))})
    load(b"".join(qs.encoded(e) for e in [config_row]+events))
    if not events or events[0]!=dict(type="quote",timestamp=t,sequence=0,instrument="ASSET",mark=step["fair_price"]):
        raise InputError("normalized mark mismatch")
    if len(events)-1!=len(trace["fills"]):raise InputError("fill event count mismatch")
    cash,fees=Decimal(state["cash"]),Decimal(state["total_fees"])
    totals={"buy":0,"sell":0};last_side="buy"
    for sequence,(event,detail) in enumerate(zip(events[1:],trace["fills"]),1):
        if event["type"]!="fill" or event["timestamp"]!=t or event["sequence"]!=sequence:
            raise InputError("fill clock mismatch")
        side=event["side"]
        if last_side=="sell" and side=="buy":raise InputError("fill side order mismatch")
        last_side=side
        found=[o for o in live if o["order_id"]==detail["order_id"]]
        if len(found)!=1:raise InputError("fill has no live order")
        order=found[0];units=event["units"];remaining=order["remaining_units"]
        if (side!=order["side"] or event["price"]!=order["price"] or units>remaining or
            event["fill_id"]!=f"{order['order_id']}-fill-{t}" or
            event["fee"]!=money(Decimal(scenario["fee_per_unit"])*units)):
            raise InputError("fill order/quantity/fee mismatch")
        price=Decimal(event["price"])
        if (side=="buy" and price<Decimal(step["seller_floor"])) or (side=="sell" and price>Decimal(step["buyer_ceiling"])):
            raise InputError("fill outside admitted demand price")
        pending=order["cancel_due"] is not None
        cash,q,fees,_=apply_fill(cash,q,fees,side=side,units=units,price=price,fee=Decimal(event["fee"]))
        expected=dict(event,order_id=order["order_id"],inventory_after=q,pending_cancel=pending,order_units_before=remaining)
        if detail!=expected:raise InputError("fill detail/projection mismatch")
        order["remaining_units"]-=units;totals[side]+=units
        c["peak"]=max(c["peak"],abs(q));c["pending_units"]+=units if pending else 0;c["partials"]+=int(units<remaining)
        if order["remaining_units"]==0:live.remove(order)
    matches=[]
    for side,flow in [("buy","seller_units"),("sell","buyer_units")]:
        budget=max(0,step[flow]-execution["queue_ahead_units"])
        if totals[side]>budget:raise InputError("fill exceeds side flow budget")
        matches.append(dict(own_side=side,external_units=step[flow],residual_after_queue=budget,filled_units=totals[side]))
    if matches!=trace["matches"]:raise InputError("side flow summary mismatch")
    c["area"]+=q*q if t<end else 0
    state.update(next_step=t+1,inventory=q,cash=money(cash),total_fees=money(fees),
                 event_count=previous["event_count"]+len(events),last_event_key=[t,len(events)-1])
    if q!=trace["inventory_after"] or live!=trace["live_after"]:raise InputError("live/inventory effect mismatch")
    qs.validate_boundary(scenario,execution,policy,state)
    return state
