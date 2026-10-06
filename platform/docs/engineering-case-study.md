# Engineering case study — make research state explainable

## Problem and constraints

A quant researcher needs to answer two questions together: “what did this method
do?” and “could it have used this information then?” A backtest that executes
successfully can still depend on duplicated data, a stale book or a changing
source. The engineering objective here is reproducible research state under
bounded local resources, with a path to a larger modular system.

The implementation starts with Python, integer/rational price representations,
hashed JSON Lines artifacts and owned packages. No distributed service is needed to prove
the contracts. Venue-specific rules remain at adapters; forecast errors and
competition accounting are not forced into one misleading portfolio score.

## Performance: change the algorithm at the measured bottleneck

Repeated scalar reconstruction revisited history for every query. The batch
implementation sorts requests, finds the relevant baseline by binary search,
sweeps ordered deltas and restores caller order with independent book copies.
It reuses the scalar implementation as a correctness reference.

On 100,000 synthetic messages and 64 requests, the current module takes a median
**813.541 ms** for the scalar loop and **19.524 ms** for the batch call, a **41.66×**
ratio (5 processes × 5 repetitions, one session). The measured stage includes
batch preparation and returned copies, but excludes history loading. This is
not an end-to-end ingestion speedup or an external-market capacity estimate.
[Samples and scope](../packages/marketdata/native/bench/results-2026-10-04.json) · [implementation](../packages/marketdata/replay/book.py)
· [parity tests](../packages/marketdata/tests/test_book.py).
Both paths return the same 2,782 output bytes, the digest pinned in
[evidence/replay-benchmark.json](evidence/replay-benchmark.json) with the workload's seed
and sizes.

The relevant costs are query sorting, history preparation, replayed deltas,
coverage checks and requested output depth. Retaining all requested books still
uses memory. A server or rewritten kernel would be premature until a different
workload identifies loading, serialization or computation as the new bottleneck.

## Research integration: source → hypothesis → decision

The vault checks explicit source notes and owns the method-card schema. The app
freezes notes, inputs and rules before invoking shared feed/features/evaluation.
The included size-imbalance method has a defensible algebraic definition but
performs worse in its fixture: validation MAE **50 versus 0 ticks**, with one of
three usable pairs. The unfavorable result remains visible.

[Method contract](../packages/vault/method_contract.py) ·
[composition](../apps/quantos/src/quantos_showcase/methods.py) ·
[tamper and interruption tests](../apps/quantos/tests/test_methods.py).

**Lesson:** implementation correctness, statistical evidence and economic utility
are separate gates. A method card supplies a research contract; it does not
prove paper authenticity, an untouched holdout or trading profitability.

## Decisions and when to revisit them

| Decision | Reason | Revisit when |
|---|---|---|
| Owned libraries plus thin application | One calculation owner; independent native semantics | Two consumers demonstrate a truly shared domain contract |
| Local immutable artifacts | Straightforward replay and interrupted-run diagnosis | Measured trial volume requires catalog/retention changes |
| Exact arithmetic for reference-sensitive calculations | Eliminate rounding ambiguity in hand checks | Profiling shows cost and an equivalent faster representation is verified |
| As-of availability and explicit gaps | Avoid retroactive information and fictitious continuity | A venue adapter can prove stronger clock/sequence guarantees |

## Evidence of capability and next work

This work demonstrates data-contract design, algorithmic optimization,
reproducible experiments and adversarial validation. The LLM research loop
(`quantos-loop`) is wired end to end, but its demo replays a hand-written
fixture; no live model run is recorded yet.

The factor half has since met real data: the pre-registered Binance daily study
([results/real-2026-10](../results/real-2026-10/README.md)) ran 120 recomputed
trials, and nothing survived the test split. The quote half has not: it still
needs a permitted archive with credible clock and continuity evidence.
[Quote walkthrough](quote-imbalance-walkthrough.md) · [module map](architecture.md).
