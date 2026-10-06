"""Independent saved-scenario arithmetic; imports no factor_research code."""
import json
import math
from pathlib import Path
import sys


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
        raise AssertionError((actual, expected))


def check_state(state, weights, report, fee):
    if state['status'] != 'COMPLETE':
        raise AssertionError('reference demo requires complete outcomes')
    capital = report['reference_capital']
    prices, exits, previous = report['current_prices'], report['exit_prices'], report['prior_weights']
    cash = report['prior_cash']
    gross, opening_fee, closing_fee, closing_value, turnover = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, weight in enumerate(weights):
        if prices[i] is None:
            assert previous[i] == weight == 0
            continue
        old = previous[i]*capital/prices[i]
        new = weight*capital/prices[i]
        trade = new-old
        close(state['prior_shares'][i], old)
        close(state['target_shares'][i], new)
        close(state['trade_shares'][i], trade)
        turnover += abs(trade)*prices[i]
        opening_fee += fee*abs(trade)*prices[i]
        cash -= trade*prices[i]+fee*abs(trade)*prices[i]
        if new:
            gross += new*(exits[i]-prices[i])
            closing_fee += fee*abs(new)*exits[i]
            closing_value += new*exits[i]
    close(state['rebalance_turnover_dollars'], turnover)
    close(state['rebalance_cost_dollars'], opening_fee)
    close(state['cash_after_rebalance'], cash)
    close(state['gross_pnl'], gross)
    close(state['terminal_fee'], closing_fee)
    close(state['net_pnl'], gross-opening_fee-closing_fee)
    close(state['terminal_cash'], cash+closing_value-closing_fee)
    close(state['terminal_cash']-capital, state['net_pnl'])
    assert all(v == 0 for v in state['terminal_shares'])


def check(folder):
    report = json.loads((Path(folder)/'allocation.json').read_text())
    checked = []
    for case in report['cases']:
        assert case['status'] == 'COMPLETE_COMPARISON'
        assumptions = case['assumptions']
        fee = assumptions['cost_bps']/10000
        for weights, metrics, state in (
            (case['allocation']['weights'], case['allocation']['metrics'], case['allocation_state']),
            (case['baseline_weights'], case['baseline_metrics'], case['baseline_state']),
        ):
            variance = sum(weight**2*report['risk_covariance'][i][i] for i, weight in enumerate(weights))
            score = sum(weight*report['utility_coefficients'][i] for i, weight in enumerate(weights))
            turnover = sum(abs(weight-report['prior_weights'][i]) for i, weight in enumerate(weights))
            gross = sum(abs(weight) for weight in weights)
            objective = score-report['risk_aversion']*variance/2-report['ridge']*sum(w*w for w in weights)/2-fee*(turnover+gross)
            close(metrics['variance'], variance)
            close(metrics['rebalance_turnover'], turnover)
            close(metrics['objective_utility'], objective)
            check_state(state, weights, report, fee)
        weights = case['allocation']['weights']
        scenario = json.loads((Path(folder)/'scenario.json').read_text())
        close(sum(weights), scenario['target_net'])
        assert sum(abs(w) for w in weights) <= assumptions['gross_limit']+1e-9
        assert all(lo-1e-9 <= w <= hi+1e-9 for w, lo, hi in zip(weights, case['lower'], case['upper']))
        checked.append(case['name'])
    return dict(status='PASS_INDEPENDENT_ARITHMETIC', cases=checked,
                limitation='Saved-input arithmetic and feasibility check, not an independent general solver or empirical validation.')


if __name__ == '__main__':
    print(json.dumps(check(sys.argv[1]), sort_keys=True, indent=2))
