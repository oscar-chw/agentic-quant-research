from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest

from factor_research.allocation import solve_allocation
from factor_research.allocation_workflow import compare_allocations, run_allocation, transition, verify_allocation
from factor_research.artifacts import json_bytes
from factor_research.contracts import read_config, read_panel
from factor_research.evaluator import evaluate
from factor_research.prepare import prepare_prices
from factor_research.runs import run_trial

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'


def tiny(**changes):
    args=dict(utility=[.1,-.1],covariance=[[.5,0],[0,.5]],previous=[0,0],costs=[.01,.01],
              exit_costs=[.01,.01],lower=[-.5,-.5],upper=[.5,.5],gross_limit=1,
              target_net=0,risk_aversion=1,ridge=.5)
    args.update(changes)
    return solve_allocation(**args)


def explicit_utility(w, p, mu, variance, gamma, ridge, c, h):
    # Independent plain sum/formulation used by the finite enumeration oracle.
    return sum(mu[i]*w[i]-(gamma*variance[i]+ridge)*w[i]**2/2-c[i]*abs(w[i]-p[i])-h[i]*abs(w[i]) for i in range(len(w)))


class ConvexAllocationTests(unittest.TestCase):
    def test_hand_optimum_and_certificate(self):
        result=tiny()
        self.assertAlmostEqual(result['weights'][0],float(Fraction(2,25)),places=8)
        self.assertAlmostEqual(result['weights'][1],-.08,places=8)
        self.assertAlmostEqual(result['metrics']['objective_utility'],.0064,places=10)
        self.assertLessEqual(max(result['certificate']['residuals'].values()),1e-9)
        self.assertLessEqual(abs(result['certificate']['duality_gap']),result['certificate']['gap_tolerance'])

    def test_independent_one_dimensional_grid(self):
        grid=[Fraction(i,1000) for i in range(-500,501)]
        objective=lambda x:Fraction(1,5)*x-x*x-Fraction(1,25)*abs(x)
        best=max(grid,key=objective)
        self.assertEqual(best,Fraction(2,25))
        self.assertAlmostEqual(tiny()['weights'][0],float(best),places=8)

    def test_gross_cap(self):
        result=tiny(gross_limit=.12)
        self.assertAlmostEqual(result['weights'][0],.06,places=8)
        self.assertAlmostEqual(result['metrics']['objective_utility'],.006,places=9)
        self.assertGreater(result['certificate']['gross_multiplier'],0)

    def test_position_concentration_cap(self):
        result=tiny(lower=[-.04,-.04],upper=[.04,.04])
        self.assertEqual(result['weights'],[.04,-.04])

    def test_zero_signal_and_high_cost_from_cash(self):
        self.assertEqual(tiny(utility=[0,0])['weights'],[0,0])
        self.assertEqual(tiny(costs=[.2,.2],exit_costs=[.2,.2])['weights'],[0,0])

    def test_high_cost_respects_prior_holdings(self):
        result=tiny(previous=[.03,-.03],costs=[.2,.2],exit_costs=[.2,.2])
        self.assertEqual(result['weights'],[.03,-.03])
        self.assertEqual(result['trades'],[0,0])

    def test_zero_psd_variance_with_explicit_ridge(self):
        result=tiny(covariance=[[0,0],[0,0]])
        self.assertTrue(all(math.isfinite(w) for w in result['weights']))

    def test_invalid_covariance(self):
        matrices=[[[1,0]], [[-.1,0],[0,1]], [[1,.1],[.1,1]], [[1,0],[.1,1]], [[float('nan'),0],[0,1]]]
        for matrix in matrices:
            with self.subTest(matrix=matrix),self.assertRaises(ValueError):tiny(covariance=matrix)

    def test_infeasible_net_gross_and_bounds(self):
        for changes in [dict(lower=[.2,.2]),dict(lower=[.3,-.5],upper=[.5,-.2],gross_limit=.5),dict(lower=[.6,-.5])]:
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'infeasible'):tiny(**changes)

    def test_forced_bounds_and_nonzero_net(self):
        result=tiny(lower=[.05,-.2],upper=[.2,.2],target_net=.1,gross_limit=.3)
        self.assertAlmostEqual(sum(result['weights']),.1,places=8)
        self.assertGreaterEqual(result['weights'][0],.05)

    def test_three_asset_grid_and_certificate(self):
        args=dict(utility=[.04,-.02,.01],covariance=[[.1,0,0],[0,.2,0],[0,0,.3]],previous=[.01,-.02,.01],
                  costs=[.002]*3,exit_costs=[.002]*3,lower=[-.2]*3,upper=[.2]*3,gross_limit=.4,risk_aversion=2,ridge=.5)
        result=solve_allocation(**args)
        best=-float('inf')
        for ix in range(-40,41):
            for iy in range(-40,41):
                w=[ix/200,iy/200,-(ix+iy)/200]
                if abs(w[2])<=.2 and sum(abs(v) for v in w)<=.4:
                    best=max(best,explicit_utility(w,args['previous'],args['utility'],[.1,.2,.3],2,.5,args['costs'],args['exit_costs']))
        self.assertGreaterEqual(result['metrics']['objective_utility']+1e-9,best)
        self.assertLess(abs(result['certificate']['duality_gap']),1e-8)

    def test_invalid_parameters(self):
        for changes in [dict(ridge=0),dict(costs=[-.1,.1]),dict(utility=[float('inf'),0]),dict(gross_limit=-1)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):tiny(**changes)

    def test_hand_cash_shares_and_full_costs(self):
        state=transition([.08,-.08],[.03,-.03],1000,1000,[100,100],[110,90],.01)
        self.assertEqual(state['target_shares'],[.8,-.8])
        for actual,expected in zip(state['trade_shares'],[.5,-.5]):self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(state['cash_after_rebalance'],999)
        self.assertAlmostEqual(state['gross_pnl'],16)
        self.assertAlmostEqual(state['terminal_fee'],1.6)
        self.assertAlmostEqual(state['net_pnl'],13.4)
        self.assertAlmostEqual(state['terminal_cash'],1013.4)

    def test_cash_refusal_and_missing_exit(self):
        self.assertEqual(transition([1,1],[0,0],1000,1000,[100,100],[110,90],0)['status'],'CASH_REFUSED')
        result=transition([.08,-.08],[.03,-.03],1000,1000,[100,100],[None,90],.01)
        self.assertEqual(result['status'],'MISSING_EXIT_MARK')
        self.assertIsNone(result['net_pnl'])
        self.assertEqual(result['target_shares'],[.8,-.8])


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.panel=(EXAMPLES/'price-import/manual-panel.csv').read_bytes()
        self.config=(EXAMPLES/'price-import/manual-config.json').read_bytes()
        self.scenario=(EXAMPLES/'allocation/scenario.json').read_bytes()

    def compare(self,panel=None,scenario=None):
        raw=self.panel if panel is None else panel
        config=read_config(self.config)
        research=evaluate(read_panel(raw,config),config)
        return compare_allocations(raw,self.config,research,self.scenario if scenario is None else scenario)

    def test_scenario_tradeoffs_and_unfavorable_outcome(self):
        report=self.compare();cases={c['name']:c for c in report['cases']}
        low,high=cases['low-cost'],cases['high-cost']
        self.assertEqual(high['allocation']['trades'],[0,0,0])
        self.assertLess(high['allocation_state']['net_pnl'],0)
        self.assertLess(high['baseline_state']['net_pnl'],0)
        self.assertTrue(low['baseline_feasible'])
        self.assertGreaterEqual(low['objective_advantage_if_comparable'],-1e-8)
        self.assertLess(low['allocation']['metrics']['variance'],low['baseline_metrics']['variance'])
        self.assertFalse(cases['tight-gross']['baseline_feasible'])
        self.assertIsNone(cases['tight-gross']['objective_advantage_if_comparable'])

    def test_future_mutation_cannot_change_decision_or_training_risk(self):
        raw=self.panel.replace(b'2024-01-11T16:00:00Z,114,',b'2024-01-11T16:00:00Z,11,')
        # Rewrite only A's future mark, using its full CSV row to avoid time fields.
        raw=self.panel.replace(b'2024-01-11T16:00:00Z,A,2024-01-11T16:00:00Z,2024-01-11T16:00:00Z,114,',
                               b'2024-01-11T16:00:00Z,A,2024-01-11T16:00:00Z,2024-01-11T16:00:00Z,11,')
        before,after=self.compare(),self.compare(raw)
        self.assertEqual(before['risk_covariance'],after['risk_covariance'])
        self.assertEqual(before['selected_factor'],after['selected_factor'])
        for a,b in zip(before['cases'],after['cases']):self.assertEqual(a['allocation'],b['allocation'])
        self.assertNotEqual(before['cases'][0]['allocation_state']['net_pnl'],after['cases'][0]['allocation_state']['net_pnl'])

    def test_missing_exit_suppresses_comparison_without_reweighting(self):
        original=b'2024-01-11T16:00:00Z,A,2024-01-11T16:00:00Z,2024-01-11T16:00:00Z,114,'
        raw=self.panel.replace(original,original.replace(b'114,',b','))
        before,after=self.compare(),self.compare(raw)
        self.assertEqual(before['cases'][0]['allocation'],after['cases'][0]['allocation'])
        self.assertEqual(after['cases'][0]['status'],'INCOMPLETE_COMPARISON')
        self.assertIsNone(after['cases'][0]['realized_pnl_difference'])

    def test_missing_current_mark_on_holdings_refused(self):
        original=b'2024-01-10T16:00:00Z,A,2024-01-10T16:00:00Z,2024-01-10T16:00:00Z,112,'
        with self.assertRaisesRegex(ValueError,'missing current mark'):
            self.compare(self.panel.replace(original,original.replace(b'112,',b',')))

    def test_no_test_end_exit_or_late_policy_freeze(self):
        for name,value in [('decision_time','2024-01-12T16:00:00Z'),('policy_fixed_by','2024-01-05T16:00:00Z')]:
            scenario=json.loads(self.scenario);scenario[name]=value
            with self.assertRaises(ValueError):self.compare(scenario=json_bytes(scenario))

    def test_installed_style_bundle_recompute_repeat_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prepared=root/'prepared';store=root/'research';output=root/'allocation'
            prepare_prices(EXAMPLES/'price-import/prices.csv',EXAMPLES/'price-import/contract.json',prepared)
            run_trial(None,None,store,'input',prepared=prepared)
            result=run_allocation(store/'input',EXAMPLES/'allocation/scenario.json',output)
            self.assertTrue(result['recomputed'])
            before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
            self.assertTrue(run_allocation(store/'input',EXAMPLES/'allocation/scenario.json',output)['reused'])
            self.assertEqual(before,{str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()})
            changed=root/'changed.json';scenario=json.loads(self.scenario);scenario['score_scale']=.01;changed.write_bytes(json_bytes(scenario))
            with self.assertRaisesRegex(ValueError,'conflicting'):run_allocation(store/'input',changed,output)
            (output/'allocation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_allocation(output)


if __name__=='__main__':unittest.main()
