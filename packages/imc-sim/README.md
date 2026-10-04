> Package reference. Start at the [root README](../../README.md); the [docs index](../../docs/README.md) has reading paths.

# Diagnose a saved trading run

Import a saved community-backtester log and inspect cash, configured fees, inventory limits and fill markouts in one offline report. The importer removes manual event-by-event JSONL conversion while keeping identity, timing and fee assumptions explicit. It preserves the original log, import configuration and normalized input alongside the report, with hashes linking all of them to the installed code.

The included import example has two instruments, four own fills and two market trades. After excluding market trades and applying the declared fee of 1 per own fill, it ends at **cash 901, ALPHA position 1, BETA position 0, equity 1003 and marked P&L 3**, from opening cash 1000. Its initial ALPHA buy breaches the declared limit. These are hand-checked synthetic diagnostics, not market performance.

**Post-competition tooling, written after IMC Prosperity 4; it contains none of the team's competition code.** Format support is limited to a declared, pinned Prosperity 4 community writer; this is not an official Prosperity adapter, matching simulator or historical-result claim.

## Install and run

Python 3.9 or newer; no runtime dependencies. From the monorepo root, `make install` installs it offline (setuptools and wheel must already be in the environment) and puts `imc4-analyze` on the environment's PATH; without installing, `source scripts/env.sh` and use `$PY -m imc4_analysis`.

```sh
imc4-analyze import-log packages/imc-sim/examples/community-log/sample.log --config packages/imc-sim/examples/community-log/import.json --out imported-run
source scripts/env.sh && $PY -m pytest -q -p no:cacheprovider packages/imc-sim/tests
```

Open `imported-run/report.html`. The run directory also contains `raw-source.log`, `import-config.json`, `normalized.jsonl`, `import-receipt.json`, `report.json` and the last-written `complete.json`. Use a new output directory for every run.

For your own log, copy [import.json](examples/community-log/import.json) and declare the correct opening cash/positions, limits, day/round, identity and fee policy. Read the [mapping contract](docs/COMMUNITY_IMPORT.md) before selecting the supported format. Missing fees/identity/order facts are not guessed; multi-day logs and ambiguous duplicate own trades are refused. The [pinned source study](docs/UPSTREAM_FORMAT.md) explains exactly what was inspected and its limits.

Existing normalized input and packaged diagnostic demo remain supported:

```sh
imc4-analyze demo --out demo-output
imc4-analyze analyze packages/imc-sim/src/imc4_analysis/data/synthetic.jsonl --out explicit-input-output
```

To exercise a strategy that generates orders and simulated fills, run `imc4-analyze quote-study --out quote-study`. The [inventory-policy study](docs/QUOTE_POLICY.md) derives the reservation-price adjustment, enforces ticks/lots/live-order capacity, and compares symmetric versus adjusted quotes on fixed rising/falling inputs with queue and cancellation-delay sensitivities. Its traces feed this same analyzer; unfavorable results are retained. This is a controlled synthetic study, not observed-market performance.

Open `demo-output/report.html` in a browser. It loads no external assets or services. `report.json` contains exact numerical values; `complete.json` records input identity and output hashes. The CLI refuses any existing output directory. An interrupted write without a complete, matching receipt is incomplete; retry into a new directory and retain the partial evidence. Files are fsynced, but power-loss/directory durability and concurrent filesystem fault injection are not qualified.

Report and completion records also include a deterministic fingerprint of the actual installed package's Python sources: canonical relative-path/per-file-hash manifest, hashed with SHA-256. It includes the fingerprint implementation itself. Scope excludes the fixture (separately input-hashed), package metadata, interpreter and standard library; runtime implementation/version are recorded separately. A version label alone is not used as build identity. Normal unpacked wheel/source installation is supported; source-less/zip imports are not qualified.

Default `--detail sampled` retains at most 200 snapshots and 200 fills. `--sample-limit` accepts 10–1000. `--detail full` retains every row within the **10,000-event, 16-MiB input limit**. HTML always displays at most 200 snapshots and 200 fills, even with full JSON. Both modes compute exact aggregate results from every admitted event.

## Investigate the demo

This separate packaged JSONL demo ends at −32 marked P&L and exercises stale marks and an unfavorable fill. The primary imported-log example above ends at +3 under its declared synthetic fees; neither is a strategy result.

1. Click **Inspect first inventory-limit breach**. Event 6 is the buy of 3 ALPHA units at 110 against a then-available mark of 100 and limit 2. Cash falls 330, marked position value rises 300, so equity falls 30 at that event.
2. Open **Fill quality**. `bad-fill` has a +2-tick as-of mark of 96: per-unit markout −14 and gross total −42. This is a diagnostic of unfavorable execution relative to a later mark; it does not isolate causal adverse selection from spread or market movement.
3. Click the missing/stale valuation link. At tick 9, the held ALPHA mark is age 3 against a maximum age of 2. Valid equity/P&L is unavailable; the separate last-known equity is 964. BETA's newer quote does not refresh ALPHA.
4. Inspect `partial-exit`. The position returns within limit; its target tick lies beyond run end, so future markout remains unavailable.

## Input and accounting

The first JSONL record is `type: config`, `schema: imc4-analysis/v1`. See the packaged [synthetic fixture](src/imc4_analysis/data/synthetic.jsonl) and [contract](CONTRACT.md). Records must be UTF-8, one JSON object per nonblank line. Quotes supply a mark or both bid and ask; fills supply unique identity, side, positive integer units, price and optional nonnegative fee. Initial cash, initial per-instrument positions and limits are explicit. Nonzero opening inventory requires its opening mark.

Money is an integer or a decimal **string**, not a binary JSON float. Computation uses `Decimal` under a bounded high-precision context; JSON emits exact decimal strings. Buy fills subtract price × units plus fee from cash; sell fills add price × units less fee. Equity is cash plus marked positions, and marked P&L is equity less the opening equity. Fees are included. There is no lot-based realized/unrealized decomposition, conversion, financing, FX or derivative-payoff model.

## Time and report boundaries

Events must be strictly increasing by `(timestamp, sequence)`; no sorting or duplicate removal occurs. All duplicate fill IDs, including identical retries, are refused. Quotes become available at their position in this stream. Fill timestamps describe known execution events; decision time and order submission are not inferred. Market timestamps are opaque integer ticks in the declared unit, never assumed to be UTC.

Each valuation uses only previously observed marks, including the current event. A held position with a missing/stale mark invalidates current portfolio valuation. Markout is computed separately after the causal pass, using the last quote at or before `fill timestamp + horizon`; no interpolation or quote after the target. The whole run must reach that target. A carried mark may predate the fill when still within the declared maximum age; its observation time and age remain visible. A later quote can populate a previously unavailable future diagnostic but cannot alter any earlier valuation snapshot.

Sampling retains endpoints/evenly spaced rows plus the first limit breach, first unavailable valuation and first adverse fill. It is **not extrema-preserving** and does not retain all anomalies. Exact totals and per-instrument breach-observation counts are unaffected. Δ columns use the actual preceding event, not the previously displayed sample row. Full JSON within the admitted cap is the raw drill-down route.

## Architecture, tests and measurements

- [contracts.py](src/imc4_analysis/contracts.py): strict normalized input validation, decimal bounds, event order and fill identity.
- [analyzer.py](src/imc4_analysis/analyzer.py): sequential cash/position/mark state, immutable causal snapshots and indexed retrospective markouts.
- [report.py](src/imc4_analysis/report.py): escaped responsive HTML/SVG; no numerical strategy/accounting logic in UI callbacks.
- [cli.py](src/imc4_analysis/cli.py): packaged demo, explicit input path, deterministic reports and completion hashes.
- [community_import.py](src/imc4_analysis/community_import.py): one pinned community log format, explicit mapping policies, market/own distinction and raw/config/normalized commitments; it calls the existing validator/analyzer.
- [Tests](tests): hand ledger, short positions, precision, invalid input, time/missingness, sampled/full parity, escaping, overwrite and interruption refusal.
- [Measurement driver](benchmarks/measure.py): admitted 1k/5k/10k-event synthetic logs, 2/8 instruments, three raw analysis repeats, separate serialization/rendering time, traced allocation and process peak RSS. Each case runs in a fresh process with a 60-second timeout. Generated inputs are not saved as large corpora.

```sh
source scripts/env.sh
$PY packages/imc-sim/benchmarks/measure.py --out measurements
$PY packages/imc-sim/benchmarks/measure_import.py --out import-measurements
```

Full snapshots cost O(events × instruments) retained state. Sampled snapshots bound this part to O(sample limit × instruments), but the parser, quote histories and fill diagnostics still consume O(events). Computation still scans all declared instruments per event; sampling is not a constant-memory streaming parser. Markouts use binary search in per-instrument observation histories. The importer adds a bounded stable event merge and counts market rows toward its 10k raw-row cap. No cap increase, extrapolated scale or speedup is claimed. Benchmark receipts are generated locally instead of embedded as timeless promises.

The import driver repeats the original two-instrument example at distinct timestamps, checking a separate integer ledger for 12, 1,200 and 9,600 raw rows. Each case performs three import-plus-analysis calls in a fresh process with a 60-second timeout, then records report sizes and process peak RSS. It is a synthetic local baseline, with no upstream execution or market-performance claim.

To adapt a recovered team UI later, obtain its parser, manifest, launch path, permitted input fixture and applicable edition/datamodel/rules. Build a separate normalized adapter with source-hash/clock/sign/unit checks and compare against retained official evidence. Existing original submission bytes remain unchanged.
