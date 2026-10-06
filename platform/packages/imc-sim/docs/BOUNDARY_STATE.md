# Complete-step state and in-memory restoration

Version0.3.1 exposes one complete-step transition for the synthetic study. It preserves the quoting equations, pending-order capacity, cancellation acknowledgements, price/time matching, partial fills and unfavorable scenarios of0.3.0. This is an in-memory restoration interface, not durable resume or a recovery CLI.

`initialize_scenario`, `validate_boundary`, `advance_step`, `finish_scenario` and `finish_study` live in `imc4_analysis.quote_study`. Inputs are fixed scenario/execution records admitted by `load_study`; policy is `symmetric` or `inventory`. `advance_step(scenario, execution, policy, state, step)` takes the exact next row and returns a new state and `{trace, events}` delta without changing its inputs, including on failure. One-shot entry points use that same transition. Genesis precedes tick0; the terminal step expires orders and emits the last mark. There is no mid-fill boundary.

State schema `imc4-quote-boundary/v1` contains run and input hash bindings, next_step, inventory, cash, total_fees, ordered live orders, six counters, event_count and last_event_key. Live records preserve side/price, original units, remaining_units, created, cancel_due and order_id. Old orders remain executable until their acknowledgement: an outstanding request is not a completed cancellation. A partial order retains its remaining quantity, never its original size. The boundary after rising/cancel_delay_1/inventory tick4 has inventory-1/cash1100.5/fees0.5 and exactly one remaining unit of inventory-4-sell; its final marked P&L remains-14.6.

Accounting uses one `balances.apply_fill` primitive under Decimal precision50, shared with the analyzer. Matching retains precision80. Cash is a derived accumulation and can legitimately exceed the normalized input's1e12 limit; a bounded derived-state validator permits it without rounding or clamping. No financing, cash/margin, fee or liquidation policy is added.

State deliberately excludes growing trace/event histories and retrospective markouts. Retain each returned delta separately. Finalization reconstructs normalized input and calls the existing analyzer, which computes markouts only after causal snapshots have been established. A future price diagnostic cannot become information available to a past policy decision. Supplied final results can be written by `cli.write_study_results` without simulation or analysis; `write_study` remains computation followed by that writer. Output creation is exclusive and the root completion receipt is last. This unit does not recover partial files.

Structural checks bind fixed inputs and reject malformed clocks, cursors, quantities and capacities. They do not prove the economic history of an otherwise plausible snapshot. Terminal finalization checks complete contiguous delta clocks/counts and agreement of analyzer cash/inventory/fees with the projection; this also is not journal authentication. Storage, ownership locks, expected-head tokens, crash recovery and power-loss qualification remain unimplemented.

From the monorepo root, after `make install` (or `source scripts/env.sh`, with `$PY` in place of `python`), run this bounded example:

```sh
python - <<'PY'
import json
from importlib.resources import files
from imc4_analysis.quote_study import (load_study, initialize_scenario,
    advance_step, finish_scenario)
raw = files('imc4_analysis').joinpath('data/quoting_scenarios.json').read_bytes()
study = load_study(raw)
s = next(s for s in study['scenarios'] if s['id'] == 'rising')
e = next(e for e in study['executions'] if e['id'] == 'cancel_delay_1')
state, deltas = initialize_scenario(s, e, 'inventory'), []
for step in s['steps'][:5]:
    state, delta = advance_step(s, e, 'inventory', state, step)
    deltas.append(delta)
state = json.loads(json.dumps(state))  # bounded snapshot; history stays separate
assert state['live'][0]['remaining_units'] == 1
for step in s['steps'][state['next_step']:]:
    state, delta = advance_step(s, e, 'inventory', state, step)
    deltas.append(delta)
run = finish_scenario(s, e, 'inventory', state, deltas)
assert run['report']['summary']['pnl'] == '-14.6'
print('In-memory restoration:', run['report']['summary'])
PY
```

At most six live orders fit the admitted cancellation delay, so snapshot size is bounded independently of the event prefix. Boundary input binding currently hashes the finite fixed scenario; it does not provide constant-time restoration or a throughput guarantee. Final report construction processes the retained events and computes retrospective markouts once.
