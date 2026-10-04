from contextlib import redirect_stderr, redirect_stdout
import copy
from decimal import Decimal
import hashlib
from importlib import resources
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from imc4_analysis.analyzer import analyze
from imc4_analysis.cli import main, write_study
from imc4_analysis.contracts import InputError
from imc4_analysis.quote_study import encoded, load_study, run_scenario, run_study


def fixture():
    return resources.files("imc4_analysis").joinpath("data/quoting_scenarios.json").read_bytes()


class QuoteStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = fixture()
        cls.study = load_study(cls.raw)
        cls.comparison, cls.runs = run_study(cls.raw)

    def test_independent_hand_ledger_and_inventory_time(self):
        expected = {"symmetric": ("900.9",1,"998.9","-1.1","0.1",2),
                    "inventory": ("1000.8",0,"1000.8","0.8","0.2",1)}
        for policy, oracle in expected.items():
            run = self.runs[f"hand_ledger--immediate--{policy}"]
            s = run["report"]["summary"]
            self.assertEqual((s["cash"],s["positions"]["ASSET"],s["equity"],s["pnl"],s["total_fees"],run["metrics"]["inventory_square_time"]), oracle)
        decision = self.runs["hand_ledger--immediate--inventory"]["trace"][1]["decision"]
        self.assertEqual(decision["reservation_price"], "99")
        self.assertEqual([o["price"] for o in decision["orders"]], ["98","100"])

    def test_independent_cash_inventory_conservation_all_scenarios(self):
        for row in self.comparison["rows"]:
            scenario = next(s for s in self.study["scenarios"] if s["id"] == row["scenario"])
            cash, q = Decimal(scenario["initial_cash"]), scenario["initial_position"]
            fees = Decimal(0)
            run = self.runs[row["run"]]
            for step in run["trace"]:
                for fill in step["fills"]:
                    signed = fill["units"] if fill["side"] == "buy" else -fill["units"]
                    cash -= signed*Decimal(fill["price"]) + Decimal(fill["fee"])
                    fees += Decimal(fill["fee"]); q += signed
                    self.assertEqual(q, fill["inventory_after"])
                self.assertEqual(q, step["inventory_after"])
            summary = run["report"]["summary"]
            self.assertEqual(Decimal(summary["cash"]), cash)
            self.assertEqual(summary["positions"]["ASSET"], q)
            self.assertEqual(Decimal(summary["total_fees"]), fees)
            self.assertEqual(Decimal(summary["equity"]), cash+q*Decimal(scenario["steps"][-1]["fair_price"]))

    def test_same_external_inputs_and_deterministic_runs(self):
        for index in range(0,len(self.comparison["rows"]),2):
            a,b=self.comparison["rows"][index:index+2]
            self.assertEqual(a["exogenous_sha256"],b["exogenous_sha256"])
            self.assertEqual([s["exogenous"] for s in self.runs[a["run"]]["trace"]], [s["exogenous"] for s in self.runs[b["run"]]["trace"]])
        comparison, runs = run_study(self.raw)
        self.assertEqual(comparison, self.comparison)
        self.assertEqual(runs, self.runs)

    def test_future_prices_and_current_demand_unavailable_to_policy(self):
        scenario = copy.deepcopy(self.study["scenarios"][1]); execution = self.study["executions"][0]
        original = run_scenario(scenario, execution, "inventory")
        scenario["steps"][5]["fair_price"] = "200"
        changed = run_scenario(scenario, execution, "inventory")
        self.assertEqual(original["trace"][:5], changed["trace"][:5])
        scenario = copy.deepcopy(self.study["scenarios"][1]); scenario["steps"][0]["seller_units"] = 0
        changed = run_scenario(scenario, execution, "inventory")
        self.assertEqual(original["trace"][0]["decision"],changed["trace"][0]["decision"])
        self.assertEqual(original["trace"][0]["accepted"],changed["trace"][0]["accepted"])

    def test_flow_budget_partial_fills_and_pending_cancel_execution(self):
        for run in self.runs.values():
            for step in run["trace"]:
                for match in step["matches"]:
                    filled=sum(f["units"] for f in step["fills"] if f["side"] == match["own_side"])
                    self.assertEqual(filled,match["filled_units"])
                    self.assertLessEqual(filled,match["residual_after_queue"])
        self.assertTrue(any(r["metrics"]["partial_fill_events"]>0 for name,r in self.runs.items() if "queue_2" in name))
        self.assertTrue(any(r["metrics"]["pending_cancel_fill_units"]>0 for name,r in self.runs.items() if "cancel_delay_1" in name))
        hand_queued=self.runs["hand_ledger--queue_2--inventory"]
        self.assertEqual(hand_queued["report"]["summary"]["fill_count"],0)

    def test_live_and_pending_quantities_stay_feasible_terminal_expires(self):
        for run in self.runs.values():
            cfg=run["policy_settings"]
            for step in run["trace"]:
                q=step["inventory_after"]
                buy=sum(o["remaining_units"] for o in step["live_after"] if o["side"]=="buy")
                sell=sum(o["remaining_units"] for o in step["live_after"] if o["side"]=="sell")
                self.assertLessEqual(q+buy,cfg["max_position"])
                self.assertGreaterEqual(q-sell,cfg["min_position"])
            self.assertEqual(run["trace"][-1]["decision"]["orders"],[])
            self.assertEqual(run["metrics"]["terminal_live_orders"],0)
            self.assertEqual(run["report"]["summary"]["breach_observations"],0)

    def test_unfavorable_scenario_is_retained(self):
        rising_s=self.runs["rising--immediate--symmetric"]; rising_i=self.runs["rising--immediate--inventory"]
        self.assertLess(Decimal(rising_i["report"]["summary"]["pnl"]),Decimal(rising_s["report"]["summary"]["pnl"]))
        self.assertLess(rising_i["metrics"]["inventory_square_time"],rising_s["metrics"]["inventory_square_time"])
        self.assertLess(Decimal(self.runs["falling--immediate--symmetric"]["report"]["summary"]["pnl"]),0)

    def test_missing_bad_units_infeasible_horizon_and_non_synthetic_refused(self):
        changes=[lambda s:s.update(data_kind="observed"),lambda s:s["scenarios"][0]["steps"][0].pop("fair_price"),
                 lambda s:s["scenarios"][0]["policy"].update(volatility_unit="percent"),
                 lambda s:s["scenarios"][0].update(initial_position=3),
                 lambda s:s["scenarios"][0].update(end_timestamp=65),
                 lambda s:s["scenarios"][0]["steps"][-1].update(buyer_units=1),
                 lambda s:s["executions"][0].update(cancel_delay_steps=3),
                 lambda s:s["executions"][0].update(id="../escape")]
        for change in changes:
            changed=copy.deepcopy(self.study);change(changed)
            with self.subTest(change=change),self.assertRaises(InputError):load_study(encoded(changed))
        with self.assertRaises(InputError):load_study(b"x"*(256*1024+1))

    def test_installed_cli_bundle_commits_all_trace_normalized_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"study"
            with redirect_stdout(io.StringIO()): self.assertEqual(main(["quote-study","--out",str(target)]),0)
            receipt=json.loads((target/"complete.json").read_text())
            self.assertEqual(receipt["source_sha256"],hashlib.sha256(self.raw).hexdigest())
            paths={str(p.relative_to(target)) for p in target.rglob("*") if p.is_file() and p!=target/"complete.json"}
            self.assertEqual(set(receipt["files"]),paths)
            for name,d in receipt["files"].items():self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(),d["sha256"])
            for row in self.comparison["rows"]:
                normalized=(target/row["run"]/"normalized.jsonl").read_bytes()
                self.assertEqual(hashlib.sha256(normalized).hexdigest(),row["normalized_sha256"])
                self.assertEqual(analyze(normalized)["summary"],row["summary"])
            with redirect_stderr(io.StringIO()):self.assertEqual(main(["quote-study","--out",str(target)]),2)

    def test_partial_write_is_incomplete_and_invalid_input_creates_nothing(self):
        from imc4_analysis import cli
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"partial"
            original=cli._write_exclusive
            def fail_trace(path,content):
                if path.name=="trace.json":raise OSError("injected trace write failure")
                return original(path,content)
            with mock.patch.object(cli,"_write_exclusive",side_effect=fail_trace),self.assertRaises(OSError):write_study(self.raw,target)
            self.assertFalse((target/"complete.json").exists())
            bad=Path(directory)/"invalid"
            with self.assertRaises(InputError):write_study(b"{}",bad)
            self.assertFalse(bad.exists())
