# Source-backed research requests

A reading note should lead to an explicit testable method, not become an
unqualified citation next to whichever backtest looks best. `quantos-method`
checks a vault-owned card and its notes, freezes the inputs and decision rule,
then invokes the existing feed/features/forecast implementation. No new model,
numerical engine, retrieval service or provider is introduced.

From the monorepo root, after `make install` (or with `$PY -m quantos_showcase.methods` after `source scripts/env.sh`):

```sh
quantos-method run --card apps/quantos/examples/methods/card.json --archive apps/quantos/examples/feed/archive.jsonl --spec apps/quantos/examples/feed/spec.json --store work/method-research --trial-id weighted-midpoint-example
quantos-method verify --store work/method-research --trial-id weighted-midpoint-example
```

Open `work/method-research/methods/runs/weighted-midpoint-example/REPORT.md`.
The example produces **NOT_SUPPORTED**: candidate MAE 50 ticks, baseline MAE 0,
improvement -50 on one eligible validation pair out of three. A successful
command means software execution completed, including an unfavorable result.
Exit 2 indicates refused input or an execution failure. Raising the card's
minimum count to two produces **INSUFFICIENT**. Use a new trial ID for every
changed request; old failures and interruptions remain visible.

## Contracts and ownership

- `quant-paper-store` owns `method_contract`: source/card validation and the
  exact rational decision rule. Paper notes reuse its existing eligibility
  checks. Original local derivations are labelled separately.
- The application owns registration, immutable trial snapshots and native
  workflow composition. QRAE owns features, splits and forecast arithmetic;
  the market-data package owns feed admission and continuity handling.
- A card declares the hypothesis, mechanism, assumptions, limitations, pinned
  local notes and one fixed weighted-midpoint-versus-midpoint MAE comparison.
  It cannot load arbitrary code, change accounting or call a model.

`min_pairs` and `min_coverage` must both be satisfied. The candidate supports
the descriptive rule only when baseline MAE minus candidate MAE is **strictly
greater** than `min_improvement_ticks`. Equality is NOT_SUPPORTED. An empty or
under-covered evaluation is INSUFFICIENT. Errors are native rational ticks,
with no rounding, combined portfolio score or inference about returns.

Notes are explicitly selected under the card's directory, hash-pinned and
frozen before evaluation. `local_review` states this workflow's scope; it does
not grant publication rights. A checked note is not proof of its claims or of
the original source's authenticity. Source content is never executed.

## Reproducibility and failure behavior

Registration and source/input snapshots exist before native invocation. The
final manifest is written last. An ordinary failed invocation retains its
failure without a decision; verification checks its integrity without claiming
the failed computation succeeded. A process interruption leaves an incomplete
trial, which cannot be overwritten or verified as complete.

Verification reads the frozen files, checks current method/adapter/evaluator
code identity, replays the native feed/forecast, binds its actual inputs back
to the registration, recalculates the decision and checks the human report.
Deleting an original note after freezing does not erase the trial evidence.
Changing the frozen note or reported decision causes refusal. Manifests are
unsigned; they are not authentication against an owner who rewrites everything.

Each trial snapshots its bounded raw input and spec; native feed admission also
retains its own source. This duplication buys a straightforward before-execution
snapshot and costs at most one additional 16 MiB archive per trial. Native
features/experiments are reused by existing content identity. Larger campaigns
should introduce a shared immutable input reference before scaling trial count.

Limits: 64 KiB card, one to eight notes at 256 KiB each, 16 MiB archive and the
existing feed/decision caps. Processing is one local trial per command. The
workflow has no external job queue or hard wall-time supervisor. It is not a
product research-agent runner.

## Research qualification

The shipped original derivation and synthetic archive were already inspected.
The one-pair example threshold demonstrates mechanics only. Local call ordering
does not establish external preregistration, unseen data or a statistically
defensible acceptance threshold. The report preserves this limitation even
if the descriptive rule is supported.

A meaningful market campaign still needs permitted representative data,
preselected sample/coverage requirements, uncertainty under serial dependence,
multiple-testing controls and an untouched evaluation policy. Economic use
also needs execution/fees/latency/capacity modeling. A fixed weighted midpoint
is not a fitted microprice. No empirical market edge is claimed here.
