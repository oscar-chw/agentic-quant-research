import copy
import unittest

from imc4_analysis.contracts import InputError
from imc4_analysis.quoting import quote_orders, validate_order


def settings(**changes):
    result = dict(inventory_penalty="1", half_spread="1", tick_size="1", lot_size=1, order_size=2,
                  min_position=-10, max_position=10, max_horizon=10, time_unit="synthetic_tick",
                  volatility_unit="price_per_sqrt_tick", penalty_unit="inverse_currency")
    result.update(changes)
    return result


def state(**changes):
    result = dict(fair_price="100", inventory=0, volatility="1", remaining_horizon=2,
                  outstanding_buy=0, outstanding_sell=0)
    result.update(changes)
    return result


class QuotingTests(unittest.TestCase):
    def test_zero_inventory_penalty_or_volatility_has_no_adjustment(self):
        for config, observed in ((settings(), state()), (settings(inventory_penalty="0"), state(inventory=3)),
                                 (settings(), state(inventory=3, volatility="0"))):
            result = quote_orders(config, observed)
            self.assertEqual(result["reservation_price"], "100")
            self.assertEqual([o["price"] for o in result["orders"]], ["99", "101"])

    def test_sign_variance_and_horizon_units(self):
        self.assertEqual(quote_orders(settings(), state(inventory=3, volatility="2"))["inventory_adjustment"], "24")
        self.assertEqual(quote_orders(settings(), state(inventory=-3, volatility="2"))["reservation_price"], "124")
        a = quote_orders(settings(), state(inventory=1, volatility="0.5", remaining_horizon=4))
        b = quote_orders(settings(), state(inventory=1, volatility="1", remaining_horizon=1))
        self.assertEqual(a["reservation_price"], b["reservation_price"])
        self.assertEqual(a["reservation_price"], "99")

    def test_outward_tick_rounding(self):
        result = quote_orders(settings(tick_size="0.5", half_spread="0.6"), state(fair_price="100.2"))
        self.assertEqual([o["price"] for o in result["orders"]], ["99.5", "101"])

    def test_capacity_reserves_own_orders_without_opposite_netting(self):
        config = settings(min_position=-6, max_position=6)
        result = quote_orders(config, state(inventory=4, outstanding_buy=1, outstanding_sell=6))
        self.assertEqual(result["capacity_units"], {"buy": 1, "sell": 4})
        self.assertEqual([o["units"] for o in result["orders"]], [1, 2])
        result = quote_orders(config, state(inventory=4, outstanding_buy=2, outstanding_sell=6))
        self.assertEqual([o["side"] for o in result["orders"]], ["sell"])
        with self.assertRaisesRegex(InputError, "infeasible outstanding"):
            quote_orders(config, state(inventory=4, outstanding_buy=3))

    def test_lots_and_asymmetric_bounds(self):
        result = quote_orders(settings(lot_size=2, order_size=4, min_position=-5, max_position=5), state(inventory=2))
        self.assertEqual([o["units"] for o in result["orders"]], [2, 4])
        at_upper = quote_orders(settings(min_position=0, max_position=6), state(inventory=6))
        self.assertEqual([o["side"] for o in at_upper["orders"]], ["sell"])

    def test_horizon_zero_stops_orders_and_bounds_refuse(self):
        self.assertEqual(quote_orders(settings(), state(remaining_horizon=0))["orders"], [])
        for bad in (-1, 11, None, True):
            with self.subTest(value=bad), self.assertRaises(InputError):
                quote_orders(settings(), state(remaining_horizon=bad))

    def test_missing_invalid_values_and_units_refuse(self):
        observed = state(); del observed["volatility"]
        with self.assertRaises(InputError): quote_orders(settings(), observed)
        for field, value in (("fair_price", None), ("fair_price", 100.0), ("volatility", "-1"), ("inventory", 11)):
            with self.subTest(field=field), self.assertRaises(InputError):
                quote_orders(settings(), state(**{field:value}))
        for change in (dict(volatility_unit="percent_per_day"), dict(time_unit="seconds"), dict(penalty_unit="unitless"),
                       dict(inventory_penalty="-1"), dict(half_spread="0"), dict(tick_size="0"), dict(lot_size=2, order_size=3)):
            with self.subTest(change=change), self.assertRaises(InputError): quote_orders(settings(**change), state())

    def test_nonpositive_bid_suppressed(self):
        result = quote_orders(settings(tick_size="0.5"), state(fair_price="0.5"))
        self.assertEqual(result["suppressed"]["buy"], "nonpositive_price")
        self.assertEqual(result["orders"], [{"side":"sell", "price":"1.5", "units":2}])

    def test_infeasible_off_tick_off_lot_and_missing_orders_refuse(self):
        config, observed = settings(lot_size=2, order_size=2), state(inventory=8)
        for order in ({"side":"buy","price":"99","units":4}, {"side":"buy","price":"99.5","units":2},
                      {"side":"buy","price":"99","units":1}, {"side":"unknown","price":"99","units":2},
                      {"side":"buy","price":None,"units":2}, {"side":"buy","units":2}):
            with self.subTest(order=order), self.assertRaises(InputError): validate_order(order, config, observed)
        with self.assertRaisesRegex(InputError, "horizon"):
            validate_order({"side":"sell","price":"101","units":2}, config, state(remaining_horizon=0))

    def test_self_crossing_live_orders_refused_and_inputs_unchanged(self):
        config, observed = settings(), state(outstanding_sell=1)
        before = copy.deepcopy((config, observed))
        with self.assertRaisesRegex(InputError, "cross a live own"):
            validate_order({"side":"buy","price":"100","units":1}, config, observed, [{"side":"sell","price":"100"}])
        quote_orders(config, observed)
        self.assertEqual((config, observed), before)
