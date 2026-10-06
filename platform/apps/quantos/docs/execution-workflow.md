# Run and inspect a native quote strategy

This guide describes finite `run` / `verify`. For separate native checkpoint/resume commands, see [persisted execution](persisted-execution.md). Both require the exact accepted repaired IMC4 0.4.0 code; older stored attempts retain their original pinned environment and are never migrated.

`quantos-execution` runs the accepted IMC4 inventory-aware quoting study from a fixed scenario and a source-linked research request. It exposes the native sequence from observed state to quote decisions, live-order constraints, accepted orders, cancellation responses, fills, positions and marked results. Both the symmetric and inventory-adjusted policies remain in the comparison.

## Installed operation

Use QuantOS showcase **0.10.0** with its base libraries and the accepted repaired **imc4-analysis 0.4.0** companion in the same Python 3.11 environment. From the monorepo root, `make install` installs both from `packages/imc-sim` and `apps/quantos` (or `source scripts/env.sh` and use `$PY -m quantos_showcase.execution`). The `portfolio` extra pins that companion version. The command checks the native Python fingerprint as well as the version.

From the monorepo root:

```sh
quantos-execution run --request apps/quantos/examples/execution/request.json --scenario apps/quantos/examples/execution/scenario.json --store ./execution-store --run-id first
quantos-execution verify --store ./execution-store --run-id first
```

Open `execution-store/runs/first/REPORT.md`. It links the frozen research question, source note, complete scenario/policy settings and registration to the native comparison and all 18 native reports. Each row links `trace.json` for policy inputs, decisions, accepted/rejected orders, pending cancellations, fills and inventory, and `normalized.jsonl` for the analyzer's event input. Prices, fees and P&L retain the native synthetic units. No cross-project score is calculated.

The shipped scenario is an exact copy of the native packaged fixture, SHA-256 `b5ae49724a60025a1b2db3220d8cc1d930be24a764ab6f5814d2320979f31ea1`. It compares three finite paths, three execution assumptions and two policies. The result is deliberately mixed: on the rising path immediate inventory-adjusted quoting reduces exposure but changes marked P&L from +34.6 to −0.6; delayed cancellation gives the adjusted policy −14.6. These are fixed synthetic outcomes, not estimated market performance.

## State and recovery

| State | Evidence | Supported operation |
|---|---|---|
| Not started | No attempt directory | Run with an unused safe ID |
| Reserved / executing | Input snapshots and registration; no final QuantOS manifest | Allow finite execution to finish, or interrupt the foreground command |
| Incomplete | Any output without a valid final QuantOS manifest | Preserve for diagnosis; use a new ID to start from original initial state |
| Completed | Final manifest and successful native-payload verification | Inspect, verify or add to an index; reuse of its ID is refused |
| Changed / invalid | Digest, identity, output or recomputation check fails | Refuse completed status; preserve evidence |

To interrupt, use the foreground process's normal Ctrl-C. This does not produce a resumable checkpoint. An abrupt interruption before any output leaves no attempt; after reservation it leaves an incomplete attempt, even if the native report directory already contains its own completion receipt. Verification requires the final QuantOS manifest too. No recovery code appends orders, fills or positions to an existing ID.

To restart, explicitly run the same command with `--run-id second`. This is a new independent simulation from the declared opening state. It does not combine the first attempt's fills or claim exactly-once execution across attempts. Input and code commitments allow the two attempts to be compared. Interrupted, repeated and competing uses of the same ID never replace existing attempt files.

Native admission enforces the scenario schema, clocks, units, ticks, lots, terminal demand and initial feasibility. The native validator itself calls the pure quote policy to check initial feasibility; no matching or study output is created during admission. QuantOS adds 256 KiB scenario/note caps, at most 18 native runs and 1,024 total decision/terminal steps. Evidence is bounded by the existing 256-file / 128 MiB policy and a 256-node traversal cap. Larger inputs require a separately reviewed integration scope.

## Source and method binding

The execution request has a fixed method identifier and a local note with its expected digest. The existing vault `source_access.inspect_note` decides note eligibility. Exact request, note and scenario bytes, native version/fingerprint and wrapper/dependency fingerprints are registered before the native writer is invoked.

The existing `vault-method/v1` card evaluates forecast MAE. It is not reused as an execution-result rule: this adapter has a distinct fixed request schema, and does not declare a statistical winner. The note documents the reservation-center model, fixed-spread approximation and execution assumptions. The native `quoting.py`, `quote_study.py` and `analyzer.py` remain the sole implementations of their respective responsibilities. The index links directly to those installed methods.

## Existing index navigation

Create a selection file beside the execution store:

```json
{"schema":"quantos-run-selection/v1","entries":[{"id":"quotes","kind":"quantos-execution","path":"execution-store/runs/first"}]}
```

Then run:

```sh
quantos-index build --selection selection.json --store ./review-index --index-id first-review
quantos-index verify --store ./review-index --index-id first-review
```

The index rechecks the native payloads, binds the frozen source/method/input evidence and retains native units. An incomplete attempt is a refused entry and makes an index partial; an absent companion package is explicitly unavailable. Changing a selected attempt or relevant code makes the old index stale.

## Verification, tradeoff and limits

Verification invokes the same native `write_study` in temporary storage and compares every output byte, including decisions, normalized events, nested reports and completion receipts. It then checks the QuantOS report and registration. This avoids reimplementing a native renderer or transition checker, at the cost of another bounded computation and temporary filesystem writes. Saved attempts are read-only during verification. Exact runtime-dependent bytes are checked, so a different Python/runtime metadata record requires a new run; cross-runtime archival verification is not qualified here.

Admission, complete/direct equivalence, changed/rehashed trace refusal, source and code identity, partial-write interruption, final-commit interruption, new-ID restart and index behavior have focused installed tests. Those checks establish a finite local integration. They do not establish a daemon, incremental/resumable state, production deployment, live orders, calibrated queue/arrival behavior, source authenticity or official competition-engine parity.

The architectural lesson is specific: a native completion receipt marks a completed study, while the application's final manifest also binds its source registration and review. Treating the first marker as a resumable trading checkpoint would misstate both lifecycle and fill identity. The adapter therefore keeps both boundaries explicit.
