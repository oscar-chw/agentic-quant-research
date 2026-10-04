import copy
from importlib import resources
import json
import unittest

from imc4_analysis.analyzer import analyze
from imc4_analysis.contracts import InputError, MAX_BYTES, MAX_EVENTS


def demo():
    return resources.files("imc4_analysis").joinpath("data/synthetic.jsonl").read_bytes()


def records():
    return [json.loads(line) for line in demo().splitlines()]


def packed(rows):
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


class AccountingTests(unittest.TestCase):
    def test_sampled_detail_preserves_exact_summary_and_retained_rows(self):
        rows = records()
        for timestamp in range(12, 112):
            rows.append(dict(type="quote", timestamp=timestamp, sequence=0, instrument="ALPHA", mark=str(95 + timestamp % 7)))
        full = analyze(packed(rows))
        sampled = analyze(packed(rows), detail="sampled", sample_limit=10)
        self.assertEqual(sampled["summary"], full["summary"])
        self.assertLessEqual(len(sampled["snapshots"]), 10)
        self.assertLessEqual(len(sampled["fills"]), 10)
        by_index = {r["index"]: r for r in full["snapshots"]}
        for row in sampled["snapshots"]:
            self.assertEqual(row, by_index[row["index"]])
        kept = {r["index"] for r in sampled["snapshots"]}
        self.assertIn(sampled["summary"]["first_breach_index"], kept)
        self.assertIn(sampled["summary"]["first_unavailable_valuation_index"], kept)
        self.assertIn(len(rows)-1, kept)

    def test_hand_calculated_roundtrip_then_open_position(self):
        report = analyze(demo())
        by_time = {r["timestamp"]: r for r in report["snapshots"]}
        # Hand ledger: 1000 - 2*101 - 1 = 797; +2*105 -1 = 1006.
        self.assertEqual(by_time[1]["cash"], "797")
        self.assertEqual(by_time[1]["equity"], "997")
        self.assertEqual(by_time[2]["equity"], "1005")
        self.assertEqual(by_time[3]["cash"], "1006")
        self.assertEqual(by_time[3]["pnl"], "6")
        self.assertEqual(by_time[3]["positions"]["ALPHA"], 0)
        # Then -3*110 + 98 = -232 cash; remaining 2*97 = 194 marked.
        self.assertEqual(report["summary"]["cash"], "774")
        self.assertEqual(report["summary"]["positions"]["ALPHA"], 2)
        self.assertEqual(report["summary"]["equity"], "968")
        self.assertEqual(report["summary"]["pnl"], "-32")
        self.assertEqual(report["summary"]["total_fees"], "2")
        self.assertEqual(by_time[5]["cash_change"], "-330")
        self.assertEqual(by_time[5]["position_value_change"], "300")
        self.assertEqual(by_time[5]["equity_change"], "-30")

    def test_exact_decimal_accounting(self):
        rows = records()[:3]
        rows[0]["initial_cash"] = "0.3"
        rows[1] = dict(type="quote", timestamp=0, sequence=0, instrument="ALPHA", mark="0.1")
        rows[2].update(units=2, price="0.1", fee="0.01")
        r = analyze(packed(rows))["summary"]
        self.assertEqual(r["cash"], "0.09")
        self.assertEqual(r["equity"], "0.29")
        self.assertEqual(r["pnl"], "-0.01")

    def test_buy_sell_markout_sign_and_unavailable_tail(self):
        fills = {f["fill_id"]: f for f in analyze(demo())["fills"]}
        self.assertEqual(fills["roundtrip-buy"]["markout"]["per_unit"], "3")
        self.assertEqual(fills["roundtrip-sell"]["markout"]["per_unit"], "5")
        self.assertEqual(fills["bad-fill"]["markout"]["gross_total"], "-42")
        self.assertEqual(fills["bad-fill"]["markout"]["mark_age"], 1)
        tail = fills["partial-exit"]["markout"]
        self.assertEqual(tail["status"], "unavailable_future")
        self.assertIsNone(tail["gross_total"])

    def test_adverse_sell_markout(self):
        rows = records()[:2]
        rows += [dict(type="fill", timestamp=1, sequence=0, instrument="ALPHA", fill_id="short", side="sell", units=1, price="100"),
                 dict(type="quote", timestamp=3, sequence=0, instrument="ALPHA", mark="104")]
        report = analyze(packed(rows))
        self.assertEqual(report["fills"][0]["markout"]["per_unit"], "-4")
        self.assertEqual(report["summary"]["equity"], "996")

    def test_stale_mark_cannot_be_reported_as_valid_equity(self):
        result = analyze(demo())
        boundary = next(r for r in result["snapshots"] if r["timestamp"] == 8)
        self.assertEqual(boundary["marks"]["ALPHA"]["status"], "fresh")
        self.assertEqual(boundary["equity"], "964")
        row = next(r for r in result["snapshots"] if r["timestamp"] == 9)
        self.assertEqual(row["marks"]["ALPHA"]["age"], 3)
        self.assertEqual(row["marks"]["ALPHA"]["status"], "stale")
        self.assertIsNone(row["equity"])
        self.assertIsNone(row["pnl"])
        self.assertEqual(row["last_known_equity"], "964")

    def test_missing_mark_does_not_borrow_a_future_quote(self):
        rows = records()
        del rows[1]
        report = analyze(packed(rows))
        first_fill = report["snapshots"][1]
        self.assertEqual(first_fill["cash"], "797")
        self.assertIsNone(first_fill["equity"])
        self.assertIsNone(first_fill["last_known_equity"])
        self.assertEqual(first_fill["valuation_issues"][0]["status"], "missing")

    def test_future_extension_preserves_every_prior_snapshot(self):
        rows = records()
        prefix = analyze(packed(rows[:7]))
        extended = analyze(packed(rows))
        self.assertEqual(prefix["snapshots"], extended["snapshots"][:len(prefix["snapshots"])])
        self.assertEqual(prefix["fills"][-1]["markout"]["status"], "unavailable_future")
        self.assertEqual(extended["fills"][2]["markout"]["status"], "available")

    def test_per_instrument_limit_is_not_netted(self):
        rows = records()[:3]
        rows.append(dict(type="fill", timestamp=1, sequence=1, instrument="BETA", fill_id="b", side="sell", units=2, price="20"))
        r = analyze(packed(rows))
        self.assertEqual(r["snapshots"][-1]["limit_breaches"], [{"instrument": "BETA", "position": -2, "limit": 1}])
        self.assertEqual(r["snapshots"][-2]["limit_breaches"], [])

    def test_initial_position_and_opening_baseline(self):
        rows = records()[:1]
        rows[0]["instruments"]["ALPHA"].update(initial_position=3, initial_mark="100")
        r = analyze(packed(rows))
        self.assertEqual(r["opening_equity"], "1300")
        self.assertEqual(r["summary"]["pnl"], "0")
        self.assertEqual(r["summary"]["breach_observations"], 1)

    def test_markout_missing_and_stale(self):
        for state in ("missing", "stale"):
            with self.subTest(state=state):
                rows = records()[:3]
                rows[0]["mark_max_age"] = 0
                if state == "missing":
                    del rows[1]
                rows.append(dict(type="quote", timestamp=3, sequence=0, instrument="BETA", mark="20"))
                markout = analyze(packed(rows))["fills"][0]["markout"]
                self.assertEqual(markout["status"], state)
                self.assertIsNone(markout["gross_total"])

    def test_markout_does_not_use_first_quote_after_target(self):
        rows = records()[:3]
        rows[0]["mark_max_age"] = 10
        rows.append(dict(type="quote", timestamp=4, sequence=0, instrument="ALPHA", mark="500"))
        result = analyze(packed(rows))["fills"][0]["markout"]
        self.assertEqual(result["target_timestamp"], 3)
        self.assertEqual(result["mark"], "100")
        self.assertEqual(result["per_unit"], "-1")

    def test_same_timestamp_availability_follows_sequence(self):
        rows = records()[:3]
        rows[1]["timestamp"] = 1
        rows[1]["sequence"] = 1
        rows[2]["sequence"] = 0
        rows[1], rows[2] = rows[2], rows[1]
        r = analyze(packed(rows))
        self.assertIsNone(r["snapshots"][1]["equity"])
        self.assertEqual(r["snapshots"][2]["equity"], "997")


class RefusalTests(unittest.TestCase):
    def test_explicit_input_caps_refused(self):
        with self.assertRaisesRegex(InputError, "exceeds"):
            analyze(b"x" * (MAX_BYTES + 1))
        rows = records()[:1]
        rows.extend(dict(type="quote", timestamp=i, sequence=0, instrument="ALPHA", mark="100") for i in range(MAX_EVENTS+1))
        with self.assertRaisesRegex(InputError, "at most"):
            analyze(packed(rows))

    def test_all_duplicate_fill_ids_refused(self):
        for conflicting in (False, True):
            rows = records()[:3]
            duplicate = copy.deepcopy(rows[-1])
            duplicate["timestamp"] = 2
            if conflicting:
                duplicate["price"] = "999"
            rows.append(duplicate)
            with self.subTest(conflicting=conflicting), self.assertRaisesRegex(InputError, "duplicate fill identity"):
                analyze(packed(rows))

    def test_out_of_order_and_repeated_keys_refused(self):
        for key in ((0, 0), (0, -1)):
            rows = records()[:3]
            rows[2]["timestamp"], rows[2]["sequence"] = key
            with self.subTest(key=key), self.assertRaises(InputError):
                analyze(packed(rows))

    def test_invalid_units_refused(self):
        for value in (0, -1, True, 1.5, "1", 1000001):
            rows = records()[:3]
            rows[-1]["units"] = value
            with self.subTest(value=value), self.assertRaisesRegex(InputError, "units"):
                analyze(packed(rows))

    def test_invalid_monetary_input_refused(self):
        for field, value in (("price", 1.25), ("price", "NaN"), ("price", "Infinity"), ("price", "0"),
                             ("price", "1e3"), ("price", True), ("fee", "-1")):
            rows = records()[:3]
            rows[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(InputError):
                analyze(packed(rows))

    def test_held_opening_position_requires_mark(self):
        rows = records()[:1]
        rows[0]["instruments"]["ALPHA"]["initial_position"] = 1
        with self.assertRaisesRegex(InputError, "requires initial_mark"):
            analyze(packed(rows))

    def test_crossed_or_incomplete_quote_refused(self):
        for change in ({"bid": "105"}, {"mark": "100"}, {"ask": None}):
            rows = records()[:2]
            rows[1].update(change)
            with self.subTest(change=change), self.assertRaises(InputError):
                analyze(packed(rows))

    def test_unknown_field_instrument_schema_and_duplicate_json_keys(self):
        for row_index, field, value in ((0, "schema", "unknown"), (1, "instrument", "UNKNOWN"), (1, "typo", 1)):
            rows = records()[:2]
            rows[row_index][field] = value
            with self.subTest(field=field), self.assertRaises(InputError):
                analyze(packed(rows))
        with self.assertRaisesRegex(InputError, "duplicate JSON key"):
            analyze(b'{"type":"config","type":"config"}\n')

    def test_blank_lines_and_empty_input_refused(self):
        for data in (b"", b"\n", demo()+b"\n"):
            with self.subTest(data=data[:10]), self.assertRaises(InputError):
                analyze(data)


if __name__ == "__main__":
    unittest.main()
