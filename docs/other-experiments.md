# Other experiments: quote imbalance and inventory-aware quoting

These are the monorepo's experiments outside the LLM research loop. Both run on
SYNTHETIC data and both keep an unfavourable result. The factor work driven by the
loop is in the [root README](../README.md) and [results/real-2026-10](../results/real-2026-10/README.md).

| Package | Quant problem | Implemented method | Evidence today | Next substantive test |
|---|---|---|---|---|
| apps/quantos with marketdata and research | Does quote size help forecast a short-horizon price? | As-of book reconstruction; fixed weighted-midpoint versus midpoint forecast | Exact fixture comparisons, explicit exclusions and an unfavorable outcome; measured replay optimization | Admit a representative archive, freeze a finite comparison and assess forecast errors with coverage and uncertainty |
| imc-sim | How does inventory-aware quoting change exposure and execution outcomes? | Reservation-price center, discrete order constraints, cancellation/partial-fill state machine and existing cash/markout analyzer | 18 controlled synthetic runs retain lower exposure with worse rising-path P&L and delayed-cancel losses; independent hand ledger | Integrate the same native policy, then qualify observed inputs and execution assumptions before market-performance claims |

Each row is a distinct contribution. Reliable research software is valuable
even when a hypothesis fails. An experiment becomes quantitative research
evidence only within the data, comparison and economic model actually tested.

## The quote-imbalance method: displayed size and next observed price

Let `b, a` be best bid/ask prices and `q_b, q_a` their positive displayed sizes.
The [implemented feature calculation](../packages/research/src/qrae/quote_features.py)
is:

```text
midpoint m          = (a+b)/2
size imbalance I    = (q_b-q_a)/(q_b+q_a)
weighted midpoint w = m + I*(a-b)/2
                    = (a*q_b+b*q_a)/(q_b+q_a)
```

The mechanism proposed in the [local derivation](../apps/quantos/examples/methods/derivation.md)
is that unequal displayed supply may contain information about the next price.
This is a hypothesis to test. The feature measures current top-of-book size;
it is not an order-flow-imbalance measure based on event changes and not a fitted
microprice model. Quotes can disappear without trading, and a price forecast
does not specify an executable order or its fill probability.

The target is the **next decision's as-of observed midpoint**, not an assumed
latent fair price or fixed physical-time price move. Both forecasts are compared
on the same pairs. Event, receipt and availability clocks, quote age and
continuity determine admission. Any excluded pair remains in the coverage
denominator. Split-crossing labels are excluded. Integer ticks and rational
arithmetic make fixture computations auditable.

The [current method card](../apps/quantos/examples/methods/card.json) fixes a
validation MAE comparison before execution. The [native result](evidence/demo.json)
has candidate MAE **50 ticks**, baseline **0**, and **1/3 eligible validation
pairs**. Its status is `NOT_SUPPORTED`. The card's one-pair minimum demonstrates
software behavior; it is not an empirical research threshold. No parameter is
fitted and no untouched market holdout has been assessed.

## Quant development serves the experiment

| Module | Algorithm or representation | Why the quant workflow needs it |
|---|---|---|
| [Market replay](../packages/marketdata/replay/book.py) | Ordered query sweep, baseline lookup and independent output books | Evaluate many decision times consistently without repeatedly replaying the same history |
| [Feature construction](../packages/research/src/qrae/quote_states.py) | Availability-ordered state and explicit coverage/continuity | Prevent delayed observations or broken books from becoming apparent predictive information |
| [Method vault](../packages/vault/method_contract.py) | Structured source, hypothesis, assumption and comparison contract | Keep the tested method identifiable when an attractive result tempts a retrospective explanation |
| [Workflow](../apps/quantos/src/quantos_showcase/methods.py) | Compose owned packages and preserve attempted outcomes | Recompute the experiment and distinguish a rejected hypothesis, insufficient observations and execution failure |
| [Native review adapters](../apps/quantos/src/quantos_showcase/run_adapters.py) | Reuse each project's calculations and units | Inspect forecast error, execution diagnostics and factor results without inventing one common performance score |

The [41.66× batch-over-scalar replay measurement](../packages/marketdata/native/bench/results-2026-10-04.json) supports an
algorithmic improvement on its stated synthetic workload. It does not expand
the statistical evidence for the price hypothesis. Storage, agents and additional
languages earn a place when they remove a measured obstacle to these workflows.

## What a real quote study must establish

The immediate dependency is representative input with credible timing and
continuity. If those clocks are missing, implement the narrow capture/admission
repair first; relabeling old data cannot recover historical availability.

For an admitted dataset, freeze one small campaign: universe, periods, decision
grid/horizon, candidate and baseline, candidate count, eligibility rules and
minimum useful effect. Record every attempted comparison. Keep development and
validation separate from the final evaluation, and disclose prior inspection.

Evaluate effect size alongside coverage, sample dependence and uncertainty.
Predeclare useful checks such as period and spread/imbalance sensitivity; record
additional exploratory slices as further research decisions. The present
one-pair software gate must be replaced with a justified campaign rule before
an empirical claim. Unfavorable or inconclusive outcomes remain valid findings.

Only a subsequent execution experiment can test trading usefulness: declare
order policy, spread/fees, latency, fill assumptions and exposure limits, then
examine net results, turnover and losses. Capacity requires its own liquidity
and impact evidence. These are pending qualifications, not current features.

The [engineering case study](engineering-case-study.md) and [quote walkthrough](quote-imbalance-walkthrough.md)
connect the research and engineering halves.
