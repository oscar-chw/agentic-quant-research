"""A bounded one-period scenario adapter over a verified research experiment."""
import json
import math
from pathlib import Path
import statistics

from .allocation import diagonal_covariance, finite, residuals, solve_allocation, terms
from .artifacts import json_bytes, read_bounded, sha, sync_directory, write_once
from .contracts import decode_json, iso, read_config, read_panel, timestamp
from .evaluator import known_price, target_weights
from .prepare import exact
from .runs import code_identity, verify_run


def training_risk(panel, config):
    start, end = config.splits['train']
    variances, counts = [], []
    for asset in config.assets:
        returns = []
        for i in range(start+1, end+1):
            t0, t1 = config.calendar[i-1], config.calendar[i]
            p0, p1 = known_price(panel, t0, asset, t0), known_price(panel, t1, asset, t1)
            if p0 is not None and p1 is not None:
                value = p1/p0-1
                if not math.isfinite(value):
                    raise ValueError("nonfinite training return")
                returns.append(value)
        if len(returns) < 2:
            raise ValueError("at least two training returns per asset required for risk")
        variances.append(statistics.variance(returns))
        counts.append(len(returns))
    return [[v if i == j else 0.0 for j in range(len(variances))] for i, v in enumerate(variances)], counts


def transition(weights, previous, cash, capital, entry, exit_prices, fee):
    """Fractional-share rebalance, then terminal liquidation; fees are real cash."""
    previous_shares, shares, trades = [], [], []
    for w, p, price in zip(weights, previous, entry):
        if price is None:
            if w != 0 or p != 0:
                raise ValueError("cannot value/trade nonzero holdings with missing current mark")
            previous_shares.append(0.0); shares.append(0.0); trades.append(0.0)
        else:
            old, new = p*capital/price, w*capital/price
            previous_shares.append(old); shares.append(new); trades.append(new-old)
    trade_cash = math.fsum(z*price for z, price in zip(trades, entry) if price is not None)
    turnover = math.fsum(abs(z)*price for z, price in zip(trades, entry) if price is not None)
    entry_fee = turnover*fee
    after_cash = cash-trade_cash-entry_fee
    prior_equity = cash+math.fsum(p*capital for p in previous)
    if abs(prior_equity-capital) > 1e-8*capital:
        raise ValueError("prior cash and holdings must reconcile to reference capital")
    if after_cash < -1e-8:
        return dict(status="CASH_REFUSED", accepted=False, reason="rebalance would make cash negative",
                    cash_before=cash, prior_shares=previous_shares)
    equity_after = after_cash+math.fsum(q*price for q, price in zip(shares, entry) if price is not None)
    conservation = abs(equity_after-(prior_equity-entry_fee))
    if conservation > 1e-8*capital:
        raise ValueError("cash/position conservation failed")
    result = dict(accepted=True, cash_before=cash, prior_shares=previous_shares, target_shares=shares,
                  trade_shares=trades, rebalance_turnover_dollars=turnover, rebalance_cost_dollars=entry_fee,
                  cash_after_rebalance=after_cash, equity_after_rebalance=equity_after,
                  cash_conservation_residual=conservation)
    if any(q != 0 and price is None for q, price in zip(shares, exit_prices)):
        return dict(result, status="MISSING_EXIT_MARK", gross_pnl=None, terminal_fee=None,
                    net_pnl=None, terminal_cash=None, terminal_shares=None)
    gross = math.fsum(q*(end-start) for q, start, end in zip(shares, entry, exit_prices) if q != 0)
    closing_value = math.fsum(q*price for q, price in zip(shares, exit_prices) if q != 0)
    closing_turnover = math.fsum(abs(q)*price for q, price in zip(shares, exit_prices) if q != 0)
    terminal_fee = closing_turnover*fee
    terminal_cash = after_cash+closing_value-terminal_fee
    net = gross-entry_fee-terminal_fee
    if abs(terminal_cash-prior_equity-net) > 1e-8*capital:
        raise ValueError("terminal cash/P&L reconciliation failed")
    return dict(result, status="COMPLETE", gross_pnl=gross, terminal_fee=terminal_fee,
                terminal_turnover_dollars=closing_turnover, net_pnl=net, terminal_cash=terminal_cash,
                terminal_shares=[0.0]*len(shares))


def compare_allocations(panel_raw, config_raw, research, scenario_raw):
    config = read_config(config_raw)
    panel = read_panel(panel_raw, config)
    scenario = decode_json(scenario_raw)
    exact(scenario, {'schema_version', 'decision_time', 'policy_fixed_by', 'reference_capital',
                     'prior_weights', 'prior_cash', 'score_scale', 'risk_aversion', 'ridge', 'target_net', 'cases'}, 'allocation scenario')
    if scenario['schema_version'] != 'factor-allocation-scenarios/v1':
        raise ValueError("unsupported allocation scenario")
    at = timestamp(scenario['decision_time'])
    try:
        index = config.calendar.index(at)
    except ValueError as exc:
        raise ValueError("allocation decision must be in the research calendar") from exc
    test_start, test_end = config.splits['test']
    if not test_start <= index < test_end:
        raise ValueError("allocation decision/exit must stay inside test boundaries")
    if timestamp(scenario['policy_fixed_by']) > config.calendar[config.splits['train'][1]]:
        raise ValueError("score/risk/cost policy must be declared fixed by training end")
    selected = research['selection']['selected']
    if selected is None or research['selection'].get('uses_test') is not False:
        raise ValueError("a validation-selected factor with no test selection is required")
    if len(config.assets) > 64:
        raise ValueError("allocation scenario supports at most 64 assets")
    row = research['feature_rows'][selected][index]
    if timestamp(row['time']) != at:
        raise ValueError("research feature clock mismatch")
    signals = row['values']
    if set(signals) != set(config.assets) or set(scenario['prior_weights']) != set(config.assets):
        raise ValueError("signal/prior holdings must match the frozen asset universe")
    capital = finite(scenario['reference_capital'], 'reference capital')
    cash = finite(scenario['prior_cash'], 'prior cash')
    scale = finite(scenario['score_scale'], 'score scale')
    if capital <= 0 or cash < 0 or scale < 0:
        raise ValueError("positive capital and nonnegative cash/score scale required")
    previous = [finite(scenario['prior_weights'][a], 'prior weight') for a in config.assets]
    if abs(cash+math.fsum(previous)*capital-capital) > 1e-8*capital:
        raise ValueError("prior cash/holdings do not reconcile to capital")
    entry_map = {a: known_price(panel, at, a, at) for a in config.assets}
    if any(p != 0 and entry_map[a] is None for a, p in zip(config.assets, previous)):
        raise ValueError("missing current mark on prior holdings; decision refused")
    baseline_map = target_weights(signals, entry_map)
    baseline = [baseline_map[a] for a in config.assets]
    maximum = max(abs(w) for w in baseline)
    scores = [w/maximum if maximum else 0.0 for w in baseline]
    utility = [scale*s for s in scores]
    covariance, counts = training_risk(panel, config)
    variances = diagonal_covariance(covariance, len(config.assets))
    later = config.calendar[index+1]
    entry = [entry_map[a] for a in config.assets]
    exits = [known_price(panel, later, a, later) for a in config.assets]
    if not isinstance(scenario['cases'], list) or not 1 <= len(scenario['cases']) <= 8:
        raise ValueError("provide 1..8 fixed scenario cases")
    names = []
    cases = []
    for case in scenario['cases']:
        exact(case, {'name', 'cost_bps', 'gross_limit', 'position_cap'}, 'scenario case')
        if not isinstance(case['name'], str) or not case['name'] or case['name'] in names:
            raise ValueError("case names must be nonempty and unique")
        names.append(case['name'])
        fee = finite(case['cost_bps'], 'cost bps')/10000
        cap = finite(case['position_cap'], 'position cap')
        if not 0 <= fee <= 1 or not 0 <= cap <= 1:
            raise ValueError("cost must be 0..10000 bps and position cap 0..1")
        lower = [-cap if signals[a] is not None and entry_map[a] is not None else 0.0 for a in config.assets]
        upper = [-lo for lo in lower]
        costs = [fee]*len(config.assets)
        try:
            result = solve_allocation(utility, covariance, previous, costs, costs, lower, upper,
                                      case['gross_limit'], scenario['target_net'], scenario['risk_aversion'], scenario['ridge'])
        except ValueError as exc:
            cases.append(dict(name=case['name'], status="REFUSED", reason=str(exc), constraints=case))
            continue
        baseline_residuals = residuals(baseline, lower, upper, case['gross_limit'], scenario['target_net'])
        baseline_feasible = max(baseline_residuals.values()) <= 1e-9
        baseline_metrics = terms(baseline, previous, utility, variances, scenario['risk_aversion'], scenario['ridge'], costs, costs)
        policy_state = transition(result['weights'], previous, cash, capital, entry, exits, fee)
        baseline_state = transition(baseline, previous, cash, capital, entry, exits, fee)
        both_complete = policy_state['status'] == baseline_state['status'] == 'COMPLETE'
        cases.append(dict(name=case['name'], status="COMPLETE_COMPARISON" if both_complete else "INCOMPLETE_COMPARISON",
                          assumptions=case, lower=lower, upper=upper, allocation=result,
                          allocation_state=policy_state, baseline_weights=baseline, baseline_metrics=baseline_metrics,
                          baseline_constraint_residuals=baseline_residuals, baseline_feasible=baseline_feasible,
                          baseline_state=baseline_state,
                          objective_advantage_if_comparable=result['metrics']['objective_utility']-baseline_metrics['objective_utility'] if baseline_feasible else None,
                          realized_pnl_difference=policy_state['net_pnl']-baseline_state['net_pnl'] if both_complete else None))
    return dict(schema_version="factor-allocation-report/v1", assets=list(config.assets), selected_factor=selected,
                decision_time=iso(at), exit_time=iso(later), reference_capital=capital, prior_cash=cash,
                prior_weights=previous, current_prices=entry, exit_prices=exits, normalized_rank_scores=scores,
                utility_coefficients=utility, utility_kind="SCORE_UTILITY_NOT_EXPECTED_RETURN", score_scale=scale,
                policy_fixed_by=scenario['policy_fixed_by'], risk_covariance=covariance, risk_samples=counts,
                risk_training_end=iso(config.calendar[config.splits['train'][1]]), risk_aversion=scenario['risk_aversion'],
                ridge=scenario['ridge'], cases=cases, source_assumptions=research.get('data_source'),
                limitations=["Diagonal training covariance; correlations and empirical predictive calibration are not qualified.",
                             "Utility scaling and fixed-by timestamp are scenario declarations, not historical calibration evidence.",
                             "Constraints use fixed reference capital, not fee-reduced equity; cash is checked separately.",
                             "Hypothetical fractional close fills and terminal liquidation; no borrow/funding/impact/queue model.",
                             "One-period state transition, not a restartable multi-period controller or live deployment.",
                             "Adverse outcomes and incomplete exit marks remain visible; no market/employer equivalence claim."])


def allocation_markdown(report):
    fmt = lambda v: 'unavailable' if v is None else f'{v:.8g}'
    lines = ['# Constrained allocation scenario', '',
             '**Score utility is not calibrated expected return. All outcomes below are hypothetical.**', '',
             f"Factor: {report['selected_factor']}; decision: {report['decision_time']}; exit: {report['exit_time']}.",
             f"Reference capital: {report['reference_capital']}; training risk samples per asset: {report['risk_samples']}.", '',
             '| Case | Policy gross | Rebalance turnover | Variance | Policy utility | Rank utility | Rank feasible | Policy net P&L | Rank net P&L |',
             '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for case in report['cases']:
        if 'allocation' not in case:
            lines.append(f"| {case['name']} | REFUSED: {case['reason']} | | | | | | | |")
            continue
        result, baseline = case['allocation'], case['baseline_metrics']
        metrics = result['metrics']
        lines.append(f"| {case['name']} | {fmt(sum(abs(w) for w in result['weights']))} | {fmt(metrics['rebalance_turnover'])} | "
                     f"{fmt(metrics['variance'])} | {fmt(metrics['objective_utility'])} | {fmt(baseline['objective_utility'])} | "
                     f"{case['baseline_feasible']} | {fmt(case['allocation_state'].get('net_pnl'))} | {fmt(case['baseline_state'].get('net_pnl'))} |")
    lines += ['', 'Gross/turnover are fractions of fixed reference capital; variance is one-period return variance. P&L is in the declared capital currency.',
              'The cost objective includes current-mark liquidation fees as a proxy; outcomes use actual next-mark fees. An infeasible rank baseline is flagged, not promoted as a constrained alternative.', '',
              '## Decisions and state', '']
    for case in report['cases']:
        if 'allocation' in case:
            lines += [f"### {case['name']}", '', f"Target weights: {case['allocation']['weights']}",
                      f"Trade shares: {case['allocation_state'].get('trade_shares')}",
                      f"Cash after rebalance: {case['allocation_state'].get('cash_after_rebalance')}",
                      f"Outcome state: {case['allocation_state']['status']}; certificate: {case['allocation']['certificate']}", '']
    lines += ['## Limits', '']+['- '+text for text in report['limitations']]
    if report['source_assumptions']:
        lines += ['', 'Source availability: '+report['source_assumptions']['availability_status']+'; historical point-in-time verification: NO.']
    return '\n'.join(lines)+'\n'


def verify_allocation(folder, recompute=True):
    folder = Path(folder)
    manifest = decode_json((folder/'completion.json').read_bytes())
    if manifest.get('schema_version') != 'factor-allocation-bundle/v1':
        raise ValueError("invalid allocation completion schema")
    expected = {'scenario.json', 'allocation.json', 'allocation.md', 'identity.json'}
    if set(manifest['files']) != expected or {p.name for p in folder.iterdir()} != expected|{'completion.json', 'research'}:
        raise ValueError("unexpected allocation bundle members")
    if folder.is_symlink() or any(p.is_symlink() for p in folder.iterdir()):
        raise ValueError("symlinked allocation bundle")
    for name, digest in manifest['files'].items():
        if sha((folder/name).read_bytes()) != digest:
            raise ValueError("allocation artifact hash mismatch")
    identity = decode_json((folder/'identity.json').read_bytes())
    source = verify_run(folder/'research', recompute=recompute)
    if source['status'] != 'SUCCESS' or sha((folder/'research/completion.json').read_bytes()) != identity['research_completion_sha256']:
        raise ValueError("research identity mismatch")
    if sha((folder/'scenario.json').read_bytes()) != identity['scenario_sha256']:
        raise ValueError("scenario identity mismatch")
    if recompute:
        if code_identity()['code_sha256'] != identity['code_sha256']:
            raise ValueError("allocation recompute requires original code")
        actual = compare_allocations((folder/'research/panel.csv').read_bytes(), (folder/'research/config.json').read_bytes(),
                                     decode_json((folder/'research/report.json').read_bytes()), (folder/'scenario.json').read_bytes())
        if json_bytes(actual) != (folder/'allocation.json').read_bytes():
            raise ValueError("allocation recomputation differs")
    return dict(status='VERIFIED_ALLOCATION', directory=str(folder), identity=identity, recomputed=recompute)


def run_allocation(research_folder, scenario_path, destination):
    research_folder, destination = Path(research_folder), Path(destination)
    source = verify_run(research_folder, recompute=True)
    if source['status'] != 'SUCCESS':
        raise ValueError("allocation requires a successful current-code research run")
    scenario_raw = read_bounded(Path(scenario_path))
    identity = dict(research_completion_sha256=sha((research_folder/'completion.json').read_bytes()),
                    scenario_sha256=sha(scenario_raw), code_sha256=code_identity()['code_sha256'])
    if destination.exists():
        old = verify_allocation(destination)
        if old['identity'] != identity:
            raise ValueError("conflicting allocation destination; prior evidence preserved")
        return dict(old, reused=True)
    snapshot = {p.name: read_bounded(p) for p in research_folder.iterdir()}
    report = compare_allocations(snapshot['panel.csv'], snapshot['config.json'], decode_json(snapshot['report.json']), scenario_raw)
    destination.mkdir(parents=True)
    (destination/'research').mkdir()
    for name, raw in snapshot.items():
        write_once(destination/'research'/name, raw)
    payloads = {'scenario.json': scenario_raw, 'identity.json': json_bytes(identity),
                'allocation.json': json_bytes(report), 'allocation.md': allocation_markdown(report).encode()}
    for name, raw in payloads.items():
        write_once(destination/name, raw)
    write_once(destination/'completion.json', json_bytes(dict(schema_version='factor-allocation-bundle/v1', files={k: sha(v) for k, v in payloads.items()})))
    sync_directory(destination/'research'); sync_directory(destination)
    return dict(verify_allocation(destination), reused=False)
