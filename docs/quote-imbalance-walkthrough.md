# Quote-imbalance walkthrough: reject a plausible idea for the right reason

**Question:** does displayed size imbalance improve the next-quote forecast, and
can a reviewer reconstruct the answer without trusting the headline?

This is a separate SYNTHETIC experiment, outside the research loop
([how a hypothesis dies](how-a-hypothesis-dies.md)). Run `bash scripts/demo.sh --quotes <new-dir>`
from the repository root; the quote trial lands in `<new-dir>/quotes/methods/runs/imbalance/`. The fixture is deliberately
small, SYNTHETIC and already inspected. This is an engineering walkthrough, not
a research discovery. Allow roughly five minutes for the four decisions below.

## 1. Read the hypothesis before the result

Open `card.json` and `registration.json` in that trial directory.
The candidate is a fixed weighted midpoint, the baseline the ordinary midpoint,
and the primary split is validation. The rule requires strictly lower MAE on
identical eligible pairs. The card freezes assumptions and source-note hashes.

**Why this matters:** changing the comparison rule after seeing results makes
research harder to audit. Local before-execution registration records this call's
ordering; it cannot prove that nobody previously inspected the data.

## 2. Inspect the unfavorable result

Open `REPORT.md`. Candidate MAE is **50 ticks**, baseline MAE **0**, and the
decision is **NOT_SUPPORTED**. The 0.0001 price tick makes 50 ticks a 0.0050
price difference, not a 50-unit trading loss. Only **one of three** validation
pairs is usable. The one-pair threshold is explicitly a fixture threshold.

**Why this matters:** a correct system must make an unsuccessful experiment as
easy to inspect as a successful one. Low forecast error is also a different
objective from executable net P&L.

## 3. Explain why observations disappear

Follow the native report link, then inspect its `report.json` and
`evaluations.jsonl`. The archive contains declared discontinuities. A pair
crossing a break is excluded even if both grid endpoints look valid. Missing or
stale observations remain unavailable. Read the [feed contract](../apps/quantos/docs/feed-workflow.md)
alongside the [as-of evaluator](../packages/research/src/qrae/quote_features.py).

**Why this matters:** quietly bridging a data outage can create a fictitious
research path. Event time alone cannot establish what information was available.

## 4. Challenge the result

`quantos-method verify` replays the calculation and checks the source/input/code
bindings. The [application tests](../apps/quantos/tests/test_methods.py) show that
an altered decision remains detectable even when its report hashes are updated.
They also show that changing/deleting the original note after freezing does not
erase the saved evidence.

To explore a different rule, copy the example card and its `derivation.md` into
one new folder, change `min_pairs` from 1 to 2 and run with a **new trial ID**.
The decision becomes **INSUFFICIENT**. Existing native features and experiments
are reused; a stronger sample requirement does not fabricate more data.

## Resume an execution without changing its economics

From the repository root, after `source scripts/env.sh`:

```sh
$PY -m quantos_showcase.execution persist-run --request apps/quantos/examples/execution/request.json \
  --scenario apps/quantos/examples/execution/scenario.json --store work/execution --run-id inventory-study --stop-after 67
$PY -m quantos_showcase.execution checkpoint --store work/execution --run-id inventory-study > work/checkpoint.json
$PY -m quantos_showcase.execution resume --store work/execution --run-id inventory-study --token work/checkpoint.json
$PY -m quantos_showcase.execution verify-persisted --store work/execution --run-id inventory-study
```

The first command pauses at global step 67. The
native state has inventory 2, cash 801.8 and fees 0.2; a pending sell cancellation is
not yet acknowledged. Resume with `--stop-after 1`, checkpoint again, then finish.
After step 68, inventory is −1, cash 1100.5 and fees 0.5, with one unit left on a
two-unit sell. Recreating the full order or replaying the committed fills would
change exposure or cash. The wrapper calls the native transition for the suffix
and binds the frozen source/method registration as its context.

Use the [persisted operation guide](../apps/quantos/docs/persisted-execution.md)
to inspect tokens, native traces and final reports. Native completion before
wrapper finalization can recover without simulation. Torn wrapper files and
ambiguous native ownership preserve/refuse. Default verification and the
`quantos-persisted` index use stored-state checks; `--recompute` is explicit.
The delayed-inventory result remains −14.6 on the rising synthetic path. Recovery
correctness does not improve the strategy's market evidence.

## A useful next extension

Admit a permitted real archive with verified clock semantics, choose sample and
coverage thresholds before the campaign, and compare across distinct periods.
Retain missingness and every attempted hypothesis. Uncertainty, dependence and
selection controls are prerequisites to a market conclusion.

The [case study](engineering-case-study.md) also explains the replay optimization
and reasons for keeping package ownership explicit.
