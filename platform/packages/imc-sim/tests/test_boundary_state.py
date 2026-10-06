"""In-memory boundary restoration and the independent accounting seam."""
from contextlib import ExitStack
import copy
from decimal import Decimal, localcontext
from fractions import Fraction
from importlib import resources
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from imc4_analysis import analyzer, balances, cli, quote_study as qs, quoting
from imc4_analysis.contracts import InputError


def inputs():
    raw = resources.files("imc4_analysis").joinpath("data/quoting_scenarios.json").read_bytes()
    return raw, qs.load_study(raw)


def boundaries(scenario, execution, policy="inventory"):
    state = qs.initialize_scenario(scenario, execution, policy)
    states, deltas = [state], []
    for step in scenario["steps"]:
        state, delta = qs.advance_step(scenario, execution, policy, state, step)
        states.append(state); deltas.append(delta)
    return states, deltas


class BoundaryStateTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.study = inputs()
        self.scenario = self.study["scenarios"][1]
        self.execution = self.study["executions"][2]
        self.states, self.deltas = boundaries(self.scenario, self.execution)

    def test_restored_pending_cancel_and_partial_remainder(self):
        # Hand amounts agree with the separately hash-bound root Fraction oracle.
        s3, s4 = self.states[4], self.states[5]
        self.assertEqual((s3["inventory"], s3["cash"], s3["total_fees"]), (2,"801.8","0.2"))
        self.assertEqual([(o["order_id"],o["remaining_units"],o["cancel_due"]) for o in s3["live"]],
                         [("inventory-2-sell",2,4),("inventory-3-sell",2,None)])
        self.assertEqual((s4["inventory"],s4["cash"],s4["total_fees"]),(-1,"1100.5","0.5"))
        self.assertEqual([(o["order_id"],o["units"],o["remaining_units"]) for o in s4["live"]],[("inventory-4-sell",2,1)])
        restored = json.loads(json.dumps(s4))
        before = copy.deepcopy((restored, self.deltas[:5], self.scenario, self.execution))
        result, delta = qs.advance_step(self.scenario,self.execution,"inventory",restored,self.scenario["steps"][5])
        self.assertEqual(delta["trace"]["fills"][0]["units"],1)
        self.assertEqual((restored,self.deltas[:5],self.scenario,self.execution),before)
        self.assertEqual(result,self.states[6])
        run=qs.finish_scenario(self.scenario,self.execution,"inventory",self.states[-1],self.deltas)
        self.assertEqual(run["report"]["summary"]["pnl"],"-14.6")

    def test_structure_cursor_remainder_and_cancellation_refusals_are_unchanged(self):
        mutations = [lambda s:s.update(schema="unknown"), lambda s:s.update(history=[]),
                     lambda s:s.update(input_sha256="0"*64),lambda s:s.update(run="other"),
                     lambda s:s.update(next_step=True),lambda s:s.update(next_step=6),
                     lambda s:s.update(last_event_key=[3,7]),lambda s:s.update(event_count=0),
                     lambda s:s.update(cash="NaN"),lambda s:s.update(cash="801.80"),
                     lambda s:s.update(total_fees="-1"),lambda s:s["counters"].update(peak=0),
                     lambda s:s["live"][0].update(remaining_units=0),
                     lambda s:s["live"][0].update(remaining_units=3),
                     lambda s:s["live"][0].update(remaining_units=True),
                     lambda s:s["live"][0].update(cancel_due=3),
                     lambda s:s["live"][0].update(cancel_due=5),
                     lambda s:s["live"][0].update(cancel_due=None),
                     lambda s:s["live"][0].update(created=4),
                     lambda s:s["live"].reverse()]
        for i, change in enumerate(mutations):
            with self.subTest(case=i):
                state=copy.deepcopy(self.states[4]); change(state); before=copy.deepcopy(state)
                with self.assertRaises(InputError): qs.validate_boundary(self.scenario,self.execution,"inventory",state)
                self.assertEqual(state,before)
        state=copy.deepcopy(self.states[4]); before=copy.deepcopy(state)
        with self.assertRaises(InputError):qs.advance_step(self.scenario,self.execution,"inventory",state,self.scenario["steps"][5])
        self.assertEqual(state,before)
        with self.assertRaises(InputError):qs.advance_step(self.scenario,self.execution,"inventory",self.states[-1],self.scenario["steps"][-1])
        bad=copy.deepcopy(self.states[-1]);bad["live"]=copy.deepcopy(self.states[5]["live"])
        with self.assertRaises(InputError):qs.validate_boundary(self.scenario,self.execution,"inventory",bad)

    def test_mid_step_failure_never_mutates_prior_state_inputs_or_deltas(self):
        state=copy.deepcopy(self.states[4]); prior=copy.deepcopy((state,self.scenario,self.execution,self.deltas))
        # Accounting failure occurs after cancels, acceptance and a tentative fill.
        with mock.patch.object(qs,"apply_fill",side_effect=RuntimeError("injected after matching")):
            with self.assertRaises(RuntimeError):qs.advance_step(self.scenario,self.execution,"inventory",state,self.scenario["steps"][4])
        self.assertEqual((state,self.scenario,self.execution,self.deltas),prior)

    def test_independent_accumulated_cash_above_input_limit(self):
        study=copy.deepcopy(self.study)
        scenario=study["scenarios"][0];scenario["initial_cash"]="1000000000000"
        admitted=qs.load_study(qs.encoded(study));scenario=admitted["scenarios"][0]
        states,deltas=boundaries(scenario,admitted["executions"][0])
        run=qs.finish_scenario(scenario,admitted["executions"][0],"inventory",states[-1],deltas)
        # Buy1@99 fee.1, sell1@100 fee.1: exact independent rational ledger.
        expected=Fraction(10**12)-99-Fraction(1,10)+100-Fraction(1,10)
        self.assertEqual(Fraction(states[-1]["cash"]),expected)
        self.assertEqual(states[-1]["cash"],"1000000000000.8")
        self.assertEqual(run["report"]["summary"]["cash"],states[-1]["cash"])
        qs.validate_boundary(scenario,admitted["executions"][0],"inventory",json.loads(json.dumps(states[-1])))

    def test_shared_balance_precision_is_independent_of_ambient_precision(self):
        with localcontext() as ctx:
            ctx.prec=3
            actual=balances.apply_fill(Decimal("1000000000000.01"),4,Decimal("0.07"),
                side="sell",units=3,price=Decimal("123456789.12345678"),fee=Decimal("0.03"))
        expected=Fraction("1000000000000.01")+3*Fraction("123456789.12345678")-Fraction("0.03")
        self.assertEqual(Fraction(actual[0]),expected)
        self.assertEqual(actual[1],1);self.assertEqual(actual[2],Decimal("0.10"))
        normal=qs.run_study(self.raw)
        with localcontext() as ctx:
            ctx.prec=3
            low=qs.run_study(self.raw)
        self.assertEqual(low,normal)
        self.assertIs(analyzer.apply_fill,balances.apply_fill)
        self.assertIs(qs.apply_fill,balances.apply_fill)

    def test_structural_validation_does_not_claim_history_authenticity(self):
        state=copy.deepcopy(self.states[4]);state["cash"]="801.7"
        qs.validate_boundary(self.scenario,self.execution,"inventory",state)
        terminal=copy.deepcopy(self.states[-1]);terminal["cash"]="1200"
        with self.assertRaises(InputError):qs.finish_scenario(self.scenario,self.execution,"inventory",terminal,self.deltas)

    def test_finalization_requires_complete_contiguous_history(self):
        with self.assertRaises(InputError):qs.finish_scenario(self.scenario,self.execution,"inventory",self.states[4],self.deltas[:4])
        bad=copy.deepcopy(self.deltas);bad[4]["events"][0]["sequence"]=1
        with self.assertRaises(InputError):qs.finish_scenario(self.scenario,self.execution,"inventory",self.states[-1],bad)

    def test_publication_only_spy_and_nonmutating_existing_output_refusal(self):
        comparison,runs=qs.run_study(self.raw)
        before=copy.deepcopy((comparison,runs))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"supplied"
            with ExitStack() as stack:
                for module,names in [(cli,["run_study","analyze"]),(qs,["run_study","run_scenario","advance_step","quote_orders","validate_order","finish_scenario","finish_study","analyze"]),(quoting,["quote_orders","validate_order"])]:
                    for name in names: stack.enter_context(mock.patch.object(module,name,side_effect=AssertionError("computation during publication")))
                cli.write_study_results(self.raw,comparison,runs,out)
                files={str(p.relative_to(out)):p.read_bytes() for p in out.rglob("*") if p.is_file()}
                self.assertEqual(len(files),94)
                with self.assertRaises(FileExistsError):cli.write_study_results(self.raw,comparison,runs,out)
                self.assertEqual(files,{str(p.relative_to(out)):p.read_bytes() for p in out.rglob("*") if p.is_file()})
            self.assertEqual((comparison,runs),before)


if __name__ == "__main__":
    unittest.main()
