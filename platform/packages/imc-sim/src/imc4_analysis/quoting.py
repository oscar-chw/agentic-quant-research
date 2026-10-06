"""Pure fixed-spread reservation-price policy; no market/clock/client access."""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext

from .contracts import InputError, decimal, integer, keys, money


def validate_settings(settings):
    keys(settings, ["inventory_penalty", "half_spread", "tick_size", "lot_size", "order_size",
                    "min_position", "max_position", "max_horizon", "time_unit", "volatility_unit", "penalty_unit"])
    expected = {"time_unit": "synthetic_tick", "volatility_unit": "price_per_sqrt_tick", "penalty_unit": "inverse_currency"}
    for name, value in expected.items():
        if settings[name] != value:
            raise InputError(f"{name} must be {value}; no implicit unit conversion")
    result = dict(settings)
    for name in ("inventory_penalty", "half_spread", "tick_size"):
        result[name] = decimal(settings[name], name, positive=name != "inventory_penalty", nonnegative=True)
    for name in ("lot_size", "order_size", "max_horizon"):
        integer(settings[name], name, 1, 10**6)
    for name in ("min_position", "max_position"):
        integer(settings[name], name, -10**6, 10**6)
    if settings["min_position"] >= settings["max_position"]:
        raise InputError("position bounds must define a nonempty interval")
    if settings["order_size"] % settings["lot_size"]:
        raise InputError("order_size must be a multiple of lot_size")
    return result


def _state(settings, state):
    keys(state, ["fair_price", "inventory", "volatility", "remaining_horizon", "outstanding_buy", "outstanding_sell"])
    result = dict(state)
    result["fair_price"] = decimal(state["fair_price"], "fair_price", positive=True)
    result["volatility"] = decimal(state["volatility"], "volatility", nonnegative=True)
    integer(state["remaining_horizon"], "remaining_horizon", 0, settings["max_horizon"])
    q = integer(state["inventory"], "inventory", settings["min_position"], settings["max_position"])
    for name in ("inventory", "outstanding_buy", "outstanding_sell"):
        if name != "inventory": integer(state[name], name, 0, 2 * 10**6)
        if state[name] % settings["lot_size"]:
            raise InputError(f"{name} must be a multiple of lot_size")
    buy = settings["max_position"] - q - state["outstanding_buy"]
    sell = q - settings["min_position"] - state["outstanding_sell"]
    if buy < 0 or sell < 0:
        raise InputError("infeasible outstanding orders: worst-case inventory exceeds a bound")
    return result, {"buy": buy, "sell": sell}


def quote_orders(settings, state):
    """r = fair - gamma*q*sigma**2*tau; round outward and reserve live capacity.

    Pending cancellations count as outstanding. At horizon zero, emit no orders.
    This deliberately leaves half-spread fixed: no fitted arrival/optimality claim.
    """
    config = validate_settings(settings)
    observed, capacity = _state(config, state)
    with localcontext() as context:
        context.prec = 80
        adjustment = (Decimal(observed["inventory"]) * config["inventory_penalty"] *
                      observed["volatility"] ** 2 * observed["remaining_horizon"])
        reservation = observed["fair_price"] - adjustment
        result = {"reservation_price": money(reservation), "inventory_adjustment": money(adjustment),
                  "capacity_units": capacity, "orders": [], "suppressed": {},
                  "status": "horizon_closed" if observed["remaining_horizon"] == 0 else "quoting"}
        for side, sign, rounding in (("buy", -1, ROUND_FLOOR), ("sell", 1, ROUND_CEILING)):
            if observed["remaining_horizon"] == 0:
                result["suppressed"][side] = "horizon_closed"
                continue
            units = min(config["order_size"], capacity[side] // config["lot_size"] * config["lot_size"])
            price = ((reservation + sign * config["half_spread"]) / config["tick_size"]).to_integral_value(rounding=rounding) * config["tick_size"]
            if units == 0:
                result["suppressed"][side] = "no_lot_capacity"
            elif price <= 0:
                result["suppressed"][side] = "nonpositive_price"
            else:
                decimal(money(price), "rounded quote price", positive=True)
                result["orders"].append({"side": side, "price": money(price), "units": units})
        return result


def validate_order(order, settings, state, live_orders=()):
    """Pre-trade refusal; no mutation, execution, netting or cancellation inference."""
    config = validate_settings(settings)
    observed, capacity = _state(config, state)
    keys(order, ["side", "price", "units"])
    if order["side"] not in ("buy", "sell"):
        raise InputError("order side must be buy or sell")
    if observed["remaining_horizon"] == 0:
        raise InputError("order after quoting horizon")
    units = integer(order["units"], "order units", 1, 10**6)
    if units % config["lot_size"]:
        raise InputError("order units must be a multiple of lot_size")
    if units > capacity[order["side"]]:
        raise InputError("order exceeds capacity including outstanding orders")
    price = decimal(order["price"], "order price", positive=True)
    with localcontext() as context:
        context.prec = 80
        if price % config["tick_size"]:
            raise InputError("order price is off tick")
    for other in live_orders:
        if other["side"] != order["side"]:
            other_price = decimal(other["price"], "live order price", positive=True)
            if (order["side"] == "buy" and price >= other_price) or (order["side"] == "sell" and price <= other_price):
                raise InputError("order would cross a live own order; await cancellation")
    return order
