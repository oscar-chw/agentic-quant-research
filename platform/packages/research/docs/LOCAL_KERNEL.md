# Local QuantOS kernel

## Purpose

The local kernel is the first executable vertical slice that replaces the old hard-coded idea tournament with data-derived, reproducible research artifacts. It is deliberately narrow: one generic bar-price baseline, one immutable run contract, and one optional local Codex review boundary.

## Work-order contract

Legacy schema `1.0` remains supported. Schema `1.1` adds a required `dataset.catalog_snapshot` object containing a normalized relative `catalog_root` and immutable `snapshot_id`. Both versions require:

- schema version and safe ID;
- market family;
- UTC issue, expiry, dataset cutoff;
- relative dataset path, SHA-256, and CSV format;
- falsifiable mechanism, prediction, horizon, and falsifier;
- chronological train/validation fractions and minimum locked-test observations;
- `lagged_momentum`, lookback, and nonnegative cost in basis points;
- evidence ceiling `E0`; catalog provenance does not itself unlock `E1`;
- `allow_live_trading: false`.

Unknown keys, duplicate JSON keys, absolute/traversal paths, missing files, hash changes, secret-like keys, future issue/cutoff, expiry, non-finite values, unsupported strategies, and live-trading requests fail before execution.

For schema `1.1`, the catalog must resolve the snapshot as of the registered cutoff. Its normalized schema must be `qrae.price-bars.v1`; normalized hash and size must match the local CSV; adapter, snapshot, availability, object path, and evidence bindings are recomputed. Catalog roots containing symlinks or escaping the workspace fail closed.

## Dataset contract

The CSV columns are:

```text
event_time,available_time,instrument,price
```

Timestamps must be timezone-aware UTC. A row must be available no later than its decision/event time and no observation may exceed the registered cutoff. Each instrument must have unique, strictly increasing timestamps and finite positive prices. A multi-instrument V0 run requires one identical event-time grid; staggered panels fail closed rather than receiving implicit full-capital exposure.

## Baseline semantics

At observation `t`, the signal is the sign of `price[t] / price[t-lookback] - 1`. The first eligible fill is `t+1`, and the measured holding return is `price[t+2] / price[t+1] - 1`; same-observation signal/fill is impossible. Cost equals `cost_bps / 10,000 * abs(position[t] - position[t-1])`. Every reported chronological split starts and ends flat, with entry and terminal liquidation costs charged. Aligned instruments are equal weighted, and the registered minimum test count is enforced before metrics are admitted.

Metrics include gross and net compounded return, per-observation mean and sample standard deviation, a non-annualized mean/std ratio, maximum drawdown, win rate, turnover, and summed cost return. Units and scope are stored in `result.json`.

## Run bundle

```text
<output-root>/runs/<run-id>/
  work_order.json
  state.jsonl
  snapshot/prices.csv
  snapshot/manifest.json
  snapshot/provenance.json   # schema 1.1 only
  data_quality.json
  cheap_falsification.json
  backtest/series.jsonl
  result.json
  validation.json
  decision.json
  claim_ledger.json
  knowledge.json
  report.md
  summary.json
  manifest.json
```

The run ID binds the canonical work order and kernel version. Artifacts are write-once; the manifest records hash and size. Schema `1.1` also commits provenance in the state log, result, validation, claim ledger, summary, and run manifest. A completed run remains verifiable after the source catalog is removed. The mutable `latest.json` pointer changes only after a successful verified run, never for quarantine.

## Public snapshot catalog

`DataCatalog` stores contracts and provenance in SQLite while raw and normalized bytes remain immutable SHA-256-addressed files. Adapter hashes bind both the contract and canonical registration time, and registration must precede the snapshot request. Resolution always requires an `as_of` cutoff and rehashes both objects. Registration is transactional and idempotent; older snapshots cannot regress the latest pointer. A non-empty V0.11 catalog lacks this time binding and is refused without deletion; create a new V0.12 catalog and refetch or reimport.

The first adapter fetches one bounded page of public Polymarket Gamma market metadata over HTTPS. It follows no pagination, sends no credentials, accepts only the allowlisted final host and JSON content type, limits time and bytes, rejects future/duplicate/non-finite/crossed records, and records only safe response headers.

`price-bars-import` is an offline bridge for a CSV the user already obtained from an allowed unauthenticated HTTPS source. It accepts an explicit adapter contract, requires the exact four-column price schema, validates every row, and records `OBSERVED_AT_IMPORT`. It never fetches the URL and cannot prove origin, licensing, historical availability, or alpha.

`historical-import` adds a stricter six-family source manifest. It hash-binds price, entitlement, calendar, revision, and family-control files; checks entitlement, point-in-time and per-instrument monotonicity; validates each revision against the latest matching dataset lineage; and freezes an exact-manifest validation bundle as raw plus prices as normalized. Code cannot verify legal truth, and the snapshot remains E0.

## Commands

From the monorepo root, after `make install` (or `source scripts/env.sh` and use `$PY` for `python`):

```text
python -m qrae.cli run --work-order packages/research/examples/price_baseline/work_order.json --workspace packages/research --output-root .quantos
python -m qrae.cli verify --run-dir .quantos/runs/<run-id>
python -m qrae.cli polymarket-fetch --catalog-root .quantos/catalog --request-id pm-YYYYMMDD-001 --limit 100
python -m qrae.cli price-bars-import --catalog-root .quantos/catalog --adapter-contract adapter_contract.json --request-id prices-YYYYMMDD-001 --source-uri https://<allowed-host>/<path> --csv data/prices.csv
python -m qrae.cli historical-import --catalog-root .quantos/catalog --workspace . --manifest data/historical-manifest.json --as-of 2025-01-01T00:00:00Z --request-id history-001
python -m qrae.cli catalog-verify --catalog-root .quantos/catalog
```

Run the complete deterministic-to-Codex path with one command:

```text
python -m qrae.cli research-draft --work-order packages/research/examples/price_baseline/work_order.json --workspace packages/research --output-root .quantos --task-id report-001 [--enable-codex]
python -m qrae.cli research-draft-verify --workflow .quantos/workflows/<run-id>/report-001.json
```

The immutable workflow receipt binds the source-run manifest, invocation (including input, Codex flag, lifetime, timeout, and output budget), broker work order/result/receipt, optional review bundle, and derived Markdown draft. Changed task-ID replays fail. Concurrent workflows sharing one output root are serialized inside one process; cross-process single-flight is not claimed. A completed broker pair recovers a missing receipt; an expired prepared-only task returns a typed successor ID. Disabled or unavailable Codex remains terminal and non-promoting.

Prepare an optional local Codex review after verification:

```text
python -m qrae.cli codex-prepare --run-dir .quantos/runs/<run-id> --task-id critic-001 --task-kind ADVERSARIAL_CRITIC
python -m qrae.cli codex-run --work-order .quantos/codex-work-orders/<run-id>/critic-001.json --workspace .quantos --enable-codex
python -m qrae.cli codex-ingest --run-dir .quantos/runs/<run-id> --task-id critic-001
python -m qrae.cli codex-review-verify --bundle-dir .quantos/codex-reviews/<run-id>/critic-001
```

Omit `--enable-codex` to test the deterministic fallback. Codex work-order schema v2 binds the verified source manifest at preparation and rejects any change before execution. Only an authenticated `DRAFT_READY` result can be ingested; deferred and quarantined results remain outbox records. Result schema v2 enforces exact terminal status/reason/check/exit-code combinations and canonical production time.

The broker journals each result/receipt pair so an interrupted two-file publication can recover without rerunning or signing attacker-supplied output. Its HMAC receipt binds the work order, canonical result, source manifest, and cited evidence path/hash/size set. The key is created outside the workspace; `--receipt-key-root` may select an explicit external key directory on all three receipt-aware commands. Validation never creates a missing key store.

Ingestion stages all four bundle files and atomically publishes the complete directory at `codex-reviews/<run>/<task>`. Reverification requires the frozen bundle, verified source run, and receipt key, but no outbox, network, or Codex call. It checks source-run identity, manifest and evidence hashes, HMAC, canonical JSON, exact inventory, and fixed `requested_transition=NONE` authority. It never changes the run state log, metrics, manifest, or `latest.json`.

A receipt proves only that this local broker boundary signed those bytes. It does not prove which model produced them, that the analysis is correct, or that a human approved it; compromise of the same local user can compromise the key.

Legacy work-order-v1/result-v1 outbox entries have neither the prepare-time run binding, strict terminal contract, nor authenticated receipt and cannot be upgraded safely. Keep them as historical drafts and issue a new task ID or use a fresh output root; the broker never back-signs pre-existing output.

## Evidence ceiling

- `E0`: mechanics or synthetic fixture only; no historical-alpha claim.
- `E1`: still blocked. A current public metadata fetch proves only when QRAE observed it; E1 requires licensed historical data with defensible point-in-time availability and a compatible chronological backtest.

The current kernel cannot produce E2 event-replay, E3 paper/shadow, E4 tiny-live, or E5 scaled-live evidence.

## Independent price-to-position validation

The September 14 change adds an internal reference for the registered lagged-momentum baseline. It reads frozen CSV independently of the production loader and generator, uses rational arithmetic, and checks serialized returns/turnover/flat-boundary fields and independently recomputed split metrics. A consistent but wrong generation series now fails before a completed manifest or latest pointer is published.

New `validation.json` receipts use schema `1.1` and include `checks.price_to_position_replay`. Valid historical `1.0` receipts remain verifiable without modification; fresh verification runs the reference even though the old receipt does not claim that check. Kernel/work-order APIs and immutable artifact sets are unchanged. Wheel hashes identify a build; unreleased version labels alone do not. The tolerance after float serialization is relative/absolute `1e-12`; realistic execution, market authenticity and external independent review remain outside this gate.
