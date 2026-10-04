# Quote features that are actually used in research

A quote update may describe an earlier market event yet reach a researcher later. Building a signal from its event timestamp alone can expose information too soon. The quote workflow retains event, received and available clocks and selects observations known at each declared decision time.

The data and feature code lives in the installed `qrae` research package. `quantos-quotes` is the application adapter; it does not copy the materializer, accounting or artifact-storage logic. The existing price/cost workflow remains a separate supported contract.

## Run and inspect

After `make install`, run from the monorepo root (or use `$PY -m quantos_showcase.quotes` after `source scripts/env.sh`):

```sh
quantos-quotes run --quotes apps/quantos/examples/quotes/observations.csv --spec apps/quantos/examples/quotes/spec.json --store work/quote-research
```

The response contains `run_id`, `feature_id` and `directory`. Read `REPORT.md`, `report.json` and `evaluations.jsonl` in that experiment directory. To recompute, use the returned ID:

```sh
quantos-quotes verify --store work/quote-research --run-id experiment-REPLACE_WITH_RETURNED_HASH --recompute
```

Repeated identical input verifies and reuses the existing run. Changing only `train_count` or `validation_count` creates a new experiment while reusing the same feature materialization. This is explicit split reuse, not permission to search for a favorable held-out period.

## Contracts and algorithms

CSV columns are exactly `instrument,event_ms,received_ms,available_ms,sequence,revision,bid_tick,ask_tick,bid_size,ask_size`. Clocks and sizes are unsigned integers; times are milliseconds, price fields are integer ticks and sizes share one declared unit within the input. The spec supplies the positive price-per-tick value. Both quote sides must be positive and uncrossed. Missing depth produces an explicit unavailable row.

Rows must satisfy event <= received <= available. Each `(sequence, revision)` identifies one observation. Exact duplicates are idempotent; conflicting rows refuse. A revision keeps its original event clock. At a decision, the highest available sequence/revision supplies the state. A delayed older sequence or revision cannot roll current state backwards. This assumes complete top-of-book snapshots and meaningful venue sequence identifiers; it does not reconstruct deltas or detect missing exchange packets.

An availability sort followed by one sweep costs O(N log N + G), with O(N + G) in-memory storage for N observations and G decisions. Source-time age controls staleness; receiving an old quote does not make it fresh. At each available decision the store writes midpoint, spread, signed size imbalance and weighted midpoint as exact rational tick values.

The forecast reads these stored rows. With bid B, ask A and displayed sizes Qb/Qa, weighted midpoint is `(A*Qb + B*Qa)/(Qb+Qa)`. Equivalently it is midpoint plus half-spread times signed imbalance. This simple estimator is described in [Stoikov's slides](https://www.ma.imperial.ac.uk/~ajacquie/Gatheral60/Slides/Gatheral60%20-%20Stoikov.pdf); it is not the calibrated micro-price model in his paper. No upstream code or data is copied.

Both the current midpoint and weighted midpoint forecast the next decision's as-of observed midpoint. Reports retain MAE, MSE, pair counts, coverage and exclusion reasons for train, validation and test. Pairs crossing split boundaries are excluded. These fixed methods fit no parameters and select no winner. Targets never enter earlier features. No transaction-cost, fill, execution or profitability result follows from forecast error.

## Storage and verification

`features/runs/<feature_id>/` contains the raw CSV, feature specification, complete feature rows and immutable manifest. Identity includes raw source, feature spec and implementing code hashes. `experiments/runs/<run_id>/` contains the full experiment specification, a hash-bound feature reference, both-model report, every evaluated/excluded pair and its manifest. Changing source or feature logic invalidates the materialization; changing split choices alone reuses it. Identical semantics in differently ordered source bytes still have different source identities.

The implementation reuses QRAE's existing immutable ArtifactStore. Manifest absence means incomplete output and is refused on reuse. It never silently repairs or overwrites a previous run. `verify --recompute` reconstructs features and reports using the original code. Hashes detect ordinary corruption; unsigned local manifests do not authenticate a hostile writer or establish historical data rights.

## Evidence and limits

The original seven-row fixture includes a late observation, an older sequence arriving after a newer one, a same-sequence revision and a stale gap. Its training midpoint MAE is 2/3 tick; weighted midpoint MAE is 5/6 tick, so the more elaborate forecast is worse. That negative result remains visible. A brute-force as-of oracle checks the sweep over 25 shuffled delayed/revised streams; additional checks cover conflicts, missing depth, future-row invariance, input/spec changes, cache reuse, interruption and tampering.

Admission limits are 16 MiB CSV, 100,000 input rows and decisions, one instrument and 64 MiB per materialized output. This version uses bounded in-memory JSONL materialization, not an out-of-core feature engine. One 10,000-observation/decision synthetic run on Python 3.11.15/macOS arm64 took 0.262 seconds for read/hash, calculation, writes and verification, with 54.9 MB process peak RSS and 5.37 MB stored output. It excludes interpreter startup and fixture generation from timing; RSS includes them. One measurement establishes no speedup or general capacity.

Reports remain E0 and PROVIDED_CLOCKS_UNAUDITED, with historical point-in-time verification false. The old price workflow still rejects delayed data. This new top-of-book workflow handles explicit availability; a general venue adapter connecting archived feed envelopes, a multi-asset/chunked feature store, research-agent campaigns and broader cross-project execution remain to be implemented and tested.
