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

## Shared method-note replay

The separate `quantos-shared-note-demo` command consumes the canonical shared
notes directory by explicit ID and pinned manifest SHA256. For example:

```sh
quantos-shared-note-demo \
  --shared-root /path/to/quant-research-vault/notes \
  --output work/shared-note-runs \
  --run-id pit-cutoff-replay
```

It writes a run record with the exact note and manifest digests, the bound
source ID, and a deterministic **hand-written replay** prompt/response. It
does not call a live model or copy the shared note body into the run store. The
`vault-method-citation/v1` citation adapter retains `document_type=method_note`, source
references and rights; the API's full envelope also includes the checked
document text. This demonstrates source admission and citation plumbing only,
with no retrieval-quality, model-quality, or trading-performance claim.

## Shared notes in the actual proposal input

`quantos-shared-proposal` feeds admitted source bodies into the same
`quantos_showcase.loop.propose_prompt` builder used by the existing research
loop. From the platform root:

```sh
quantos-shared-proposal \
  --campaign apps/quantos/examples/loop/campaign.json \
  --contract apps/quantos/examples/loop/synthetic-contract.json \
  --shared-root /path/to/quant-research-vault/notes \
  --manifest-sha256 c15abda5657cc63a953c7d8b8d34f044f0b6dcad09c253a0b9842e5beea4c70a \
  --note-id quant-pit-cutoffs-v1 --output work/shared-proposal
```

The new output directory contains `prompt.json` (the actual proposal task,
including the source text as untrusted context) and `request.json` (input
digests, typed citations and `PROMPT_READY` / `NOT_EXECUTED`). These are task
input snapshots, not another knowledge index. Changing admitted note text
changes the real task input; a stale digest refuses before output creation.
Legacy paper prompts retain their exact bytes and replay behavior.

This command prepares input only. It neither runs a provider nor validates a
proposal, freezes a method card or backtests a strategy. The separate
hand-written replay above is only a citation demonstration; neither command
establishes model reasoning, retrieval quality or market performance.

## Shared notes through the normal research loop

Campaign admission checks the declared question, signals, retrieval request and
numeric rule before reading dependent price/contract files or calling the proposer.
Use nonempty text for `question` and `retrieval.query`, a nonempty signal list drawn
from `momentum` / `reversal`, integer `retrieval.n_results` in1..128, integer
`max_lookback` in1..250 and integer `rule.min_valid_dates` in1..5000. Boolean values
are not integer settings. These bounds match the existing source-access and native
factor-card contracts; data coverage may still make a valid experiment insufficient.

Malformed input exits2 with `REFUSED` and the offending field. For example,
`max_lookback: 0` is rejected before a proposal call or run reservation. Correct the
campaign and retry the same run ID if no directory was created. A completed run
remains immutable: changed research requests need a new ID. Valid campaigns retain
their original prompt bytes and replay identity.

`quantos-loop run` and `quantos-loop propose` explicitly opt into the shared
corpus with `--shared-root`, `--manifest-sha256`, and repeated `--note-id`.
Select 1..8 distinct IDs, each satisfying the card's 64-character identifier
limit. All three options are required together, and `--vault` is mutually
exclusive with this selection. The existing paper-vault path keeps its exact
prompt and replay bytes. Explicit shared selection follows the supplied ID
order; the campaign's lexical query does not silently select other notes.

After sourcing `scripts/env.sh`, a proposal-only recorded replay can enter the
normal loop with the exact campaign and contract that produced its prompt:

```sh
"$PY" -m quantos_showcase.loop run \
  --campaign work/recorded-campaign.json \
  --prices work/control/panel.csv \
  --contract apps/quantos/examples/loop/synthetic-contract.json \
  --shared-root ../../quant-research-vault/notes \
  --manifest-sha256 c15abda5657cc63a953c7d8b8d34f044f0b6dcad09c253a0b9842e5beea4c70a \
  --note-id quant-pit-cutoffs-v1 \
  --store work/loop --run-id recorded-proposal \
  --replay work/recorded-proposal.json
"$PY" -m quantos_showcase.loop verify --run work/loop/recorded-proposal
```

The optional Python arguments are `shared_root`, `manifest_sha256`, and
`note_ids`; pass `vault=None` to `run_loop` or `propose_only`. Admission checks
all selected notes, rights and manifest/source digests before the proposer is
called. No method note receives an arXiv identifier. The unchanged
`vault-method/v1` card uses its supported `local_derivation` source kind;
`registration.json` carries the corresponding `vault-method-citation/v1`
metadata. The ledger retains the selection, document ID/type, title, both
hashes, source references and rights. Frozen task inputs are run evidence,
not another knowledge index.

Verification binds the selection and retrieval metadata to the exact stored
prompt and raw response, then to each card, source snapshot and registration.
It binds frozen prices and import rules to the recorded input digests, allowing only
the proposal's lookback and signal override, and enforces the prompt's horizon.
The exact campaign and original contract are retained as input snapshots.
It recomputes the existing trial and checks the signed critic result. It
verifies admitted snapshots even if the corpus later changes; a new run must
pass current admission again. Local hashes detect inconsistent edits, not an
owner rewriting all evidence consistently.

`quantos-loop verify` also checks the review-tool export `envelope.json` against
the verified ledger, its exact digest and the hash-bound proposal prompt's
evaluation cutoff. Changing the evidence tier, claims, input references or
requested decision causes refusal, just as editing the Markdown report does.
A missing envelope is refused with its path; restore the original run artifact
and verify again. Verification never rewrites or repairs run evidence. This is
consistency checking of local evidence, not external authentication or proof of
an untouched dataset.

With an offline `ReplayProvider` in a shared-note run, an absent critic entry
produces the broker's signed `QUARANTINED` / `RUNNER_FAILURE` result and a
`CRITIC_UNUSABLE` row. No critic answer is fabricated; test metrics stay
`SEALED`, and promotion is refused. A stale critic replay still aborts, as do
live transport failures under the existing policy. A proposal response and its
provenance are retained exactly, including incomplete prose; no follow-up
model call repairs them. Verification of this workflow establishes execution
and provenance, not model quality or market performance.
