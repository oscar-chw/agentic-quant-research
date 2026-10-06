# Pause, resume and inspect the same quote study

`quantos-execution` consumes the accepted repaired IMC4 0.4.0 persistence API. Native code owns the policy, matcher, committed state and monetary accounting. QuantOS freezes the research context before execution, supplies its digest on every native call, and links the native product into a final review.

Use Python 3.11 with the exact accepted packages installed and a local POSIX store without symlink components. The native code digest is `536964bdd251c77576ed6953bd8befabc843b05813e8e205e5c11bd387abfcd4`. From the monorepo root, `make install` provides this combination, and the optional `portfolio` extra pins the same imc4-analysis 0.4.0 (without installing, `source scripts/env.sh` and use `$PY -m quantos_showcase.execution`).

```sh
quantos-execution persist-run --request apps/quantos/examples/execution/request.json --scenario apps/quantos/examples/execution/scenario.json --store /absolute/private/quote-store --run-id demo --stop-after 67
quantos-execution checkpoint --store /absolute/private/quote-store --run-id demo > /absolute/private/checkpoint.json
quantos-execution resume --store /absolute/private/quote-store --run-id demo --token /absolute/private/checkpoint.json --stop-after 1
quantos-execution checkpoint --store /absolute/private/quote-store --run-id demo > /absolute/private/checkpoint.json
quantos-execution resume --store /absolute/private/quote-store --run-id demo --token /absolute/private/checkpoint.json
quantos-execution verify-persisted --store /absolute/private/quote-store --run-id demo
```

These commands have a separate lifecycle from finite `run` / `verify`. Existing finite attempts retain their original meaning. Stored attempts bind exact code and runtime identities; use their original pinned release to verify old results. No migration or implicit restart is performed. IDs use lowercase letters, digits, underscore or hyphen, begin with a letter, and are at most 64 characters.

Python equivalents are `quantos_showcase.persisted.run(request_path, scenario_path, store, run_id, stop_after_commits=67)`, `checkpoint(store, run_id)`, `resume(store, run_id, expected_checkpoint, stop_after_commits=1)` and `verify(store, run_id, mode="state")`. Paths are `pathlib.Path` objects. Resume accepts the typed checkpoint envelope or a full result containing it. Tokens bind both economic generation and operational revision. A missing acknowledgement does not mean a step was uncommitted: read a new checkpoint and explicitly use that token. A stale token refuses.

## One context and one execution state

The wrapper is `STORE/persisted/runs/ID`. It contains `wrapper.lock`, request/scenario/source snapshots, immutable `registration.json`, a `native` store and optional `REPORT.md` / `complete.json`. Its native attempt is exactly `native/ID`; the inspectable product is `native/ID/product`. The registration binds the actual snapshot bytes, fixed method, wrapper/shared code, native identity, absolute path, directory/lock identities and product mapping. It is written and synced before native initialization. Its actual SHA256 is the native `context_sha256`.

Continuation validates the frozen source note with the existing source-access contract. It never consults an edited original note or runs initial policy admission. The wrapper acquires a stable nonblocking lifetime lock before native API calls. Native APIs acquire and own their internal locks. Unexpected files, replaced ownership, symlinks and extra hard-link aliases refuse before mutable native access. This is a cooperating local ownership contract, not authentication against an owner who rewrites all code and evidence.

| Native / wrapper state | Meaning | Next action |
|---|---|---|
| PAUSED_AT_BOUNDARY | Exact committed economic prefix | Resume with current checkpoint |
| COMPUTE_COMPLETE | All economic steps committed; publication pending | Resume to publish with zero policy steps |
| PUBLICATION_INCOMPLETE | Native publication started | Resume only where native ownership/recovery permits |
| NATIVE_COMPLETE_WRAPPER_INCOMPLETE | Native product complete; wrapper finalization missing | Verify native state/product and finish wrapper metadata with zero native run/resume calls |
| COMPLETED | Native completion plus exact wrapper report and receipt | Read-only verification; further resume refuses |
| REFUSED | Ambiguous, corrupt or incompatible evidence | Preserve the attempt; inspect the stated cause |

Finalization exclusively creates the deterministic report and then the completion receipt, flushing and syncing each. Missing files can be created, and an existing exact full report can be reconciled. A partial or differing report/receipt, or a receipt without a report, is preserved and refused. No append, truncation or automatic partial-file repair is supported. Native pre-ownership publication and incomplete forensic-capture refusals propagate. Real process-exit checks cover the boundary before wrapper finalization; they do not establish power-loss recovery.

## Why the restored state changes the next decision

The fixed source note defines the inventory reservation-center model and assumptions. Its policy and the symmetric diagnostic baseline share exogenous inputs while their endogenous orders and fills differ. The persisted state retains live order **remaining** quantities and pending cancellation clocks together with inventory, cash, fees and the event/trace delta. Pending cancellation still reserves executable exposure until acknowledgement.

In the packaged rising / delay1 / inventory run, generation 67 retains cash 801.8, inventory 2 and fees 0.2. `inventory-2-sell` awaits cancellation at tick 4. After one committed step, generation 68 has cash 1100.5, inventory −1 and fees 0.5; `inventory-4-sell` has only one unit left. For the three sold units, native cash increases by 298.7 after 0.3 fees. Restoring the initial two-unit order size would invent exposure. These are assertions against saved native rows, not a parallel accounting implementation.

The installed demonstration resumes exactly one step and then the remaining 58, and compares all 94 product files against the direct native writer. The delayed inventory result **−14.6** remains visible. This synthetic study illustrates execution assumptions and state consistency; it is newly authored work, not recovered IMC4 competition performance or market qualification.

## Review and limits

Default `verify-persisted` uses native state/effect and product verification without replaying completed policy or matching. `--recompute` explicitly requests native independent computation and reports `NATIVE_RECOMPUTED`. Prefix audit work grows with retained history; no constant-time restart or speedup is claimed.

Index the wrapper with `{"id":"quotes","kind":"quantos-persisted","path":"quote-store/persisted/runs/demo"}` in the existing `quantos-run-selection/v1` selection. The index links bounded snapshots, receipts, native product comparison, reports and method code. It does not pass databases, rollback journals, anchors or forensics through the generic evidence reader. Incomplete entries retain their phase and make the index partial; combined_score stays null. The final report links native decision traces, normalized orders/fills and position/accounting reports directly.

Native bounds remain 18 runs, 1024 steps, 256 KiB raw input, 64 MiB database/journal, 128 MiB product and 512 MiB native attempt. Wrapper metadata is separately capped at 4 MiB, with a fixed small file grammar: request/registration/receipt 64 KiB each, scenario/note 256 KiB each, report 1 MiB, zero-byte lock. The generic evidence caps are unchanged. Native state is path and identity bound; a copied product is inspectable output, not a relocated resumable attempt.

The architectural lesson is that native computation, native publication and wrapper completion are separate boundaries. Delegating the first two to their owner allows honest suffix-only recovery without another journal or a second economic interpretation. Conservative refusal leaves torn wrapper publication as an explicit limitation rather than pretending it is handled.
