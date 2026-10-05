> Package reference. Start at the [root README](../../README.md); the [docs index](../../docs/README.md) has reading paths.

# Factor Lab: momentum vs reversal from your own price CSV

Give it a long-form price CSV (one row per asset and timestamp) and a short JSON contract; it compares lagged momentum with reversal on fixed train/validation/test dates and reports rank IC, coverage and modeled costs.
Every run is kept as a hashed trial, with the original CSV, so it can be verified and recomputed later.

## Install and run

From the monorepo root, after `make install` puts `factor-research` on the environment's PATH (or `source scripts/env.sh` and use `$PY -m factor_research` without installing):

```sh
factor-research prepare --prices packages/factor/examples/price-import/prices.csv --contract packages/factor/examples/price-import/contract.json --out work/prepared
factor-research run --prepared work/prepared --store work/runs --run-id csv-demo
factor-research verify work/runs/csv-demo --recompute
factor-research trials work/runs
```

Open `work/runs/csv-demo/report.md`; `report.json` contains the full numerical diagnostics. The hand-authored example has 35 price rows for three assets over 12 declared grid dates, one omitted row and one blank price. It prepares exactly the separately written [manual panel](examples/price-import/manual-panel.csv). The report prominently states **ASSUMED_DELAY; historical point-in-time verification: NO** because the ordinary CSV has no original availability clocks.

For supplied clocks, including one late observation:

```sh
factor-research prepare --prices packages/factor/examples/price-import/prices-with-clocks.csv --contract packages/factor/examples/price-import/contract-with-clocks.json --out work/prepared-clocks
factor-research run --prepared work/prepared-clocks --store work/runs --run-id supplied-clocks
```

The late row is withheld from earlier features; supplied clocks are labeled **PROVIDED_CLOCKS_UNAUDITED**, not historically verified. All examples are synthetic. Runtime dependencies are Python >=3.9 and the standard library; packaging needs setuptools >=70 in the installing environment, because `make install` disables build isolation. Version 0.4.0 passes 82 tests (`packages/factor/tests`) and the example workflow on Python 3.11.15/macOS arm64.

## Constrained holdings and trades

Use the same research run to compare one-period allocations with the existing rank weights:

```sh
factor-research allocate --research-run work/runs/csv-demo --scenario packages/factor/examples/allocation/scenario.json --out work/allocation-demo
factor-research verify-allocation work/allocation-demo
python3 packages/factor/tools/allocation_reference_check.py work/allocation-demo
```

Open `work/allocation-demo/allocation.md`. The decision balances explicitly scaled **score utility, not expected return**, diagonal training risk and proportional costs under net/gross/position constraints. It starts from prior holdings, emits fractional share trades and checks cash conservation. Four fixed scenarios expose the cost/risk tradeoff; high fees retain an adverse outcome and tighter constraints flag the raw rank baseline as infeasible. Missing exits keep realized outcomes unavailable. See [the objective, solver derivation, state equations and limits](docs/ALLOCATION.md).

## Bring your own CSV

Start with [the small import contract](examples/price-import/contract.json). Map every source column explicitly; unknown/unmapped columns, duplicate/conflicting asset-time rows and out-of-contract assets/times are refused. Choose fixed UTC offsets or offsets supplied in ISO strings, an explicit time for date-only rows, or integer Unix timestamps with declared s/ms/us units. No timestamp guessing occurs.

Choose an explicit asset list or a universe known by a training-period cutoff. New test-only assets are refused rather than used to redefine the historical universe. Choose a declared fixed-step calendar to expose entirely missing dates, or an observed-union grid whose inability to detect such missing dates is made visible. Set absolute timestamp split endpoints; there is no automatic split search on held-out results.

Provide both observation/availability fields or explicitly opt into an assumed-delay model. Even a zero-second delay is an assumption and appears in the immutable report. The prepared bundle keeps the raw CSV, raw import contract, panel/config and hash manifest. **Use `run --prepared` with the intact bundle** to retain those records in the trial; the older panel/config route refuses files adjacent to an import manifest. Deliberately copying only the normalized data strips provenance and must not be represented as a verified import.

See [import contract and refusal cases](docs/IMPORT_CONTRACT.md) for precise policies, limits and examples. Identical preparation is deterministic and byte-preserving; changed input/contract cannot overwrite an existing destination. Use a fresh output directory after a deliberate change.

## Evaluator input and time contract

`factor-panel/v1` is CSV with `time,asset,observed_at,available_at,price` and optional `group`. Timestamps must include a timezone and are normalized to UTC; `time <= observed_at <= available_at`. `time` identifies the price observation. Price is positive finite or blank. Duplicate asset/time pairs, unknown assets/calendar entries, invalid clocks and nonfinite prices are refused. Groups are retained metadata; no neutralization is performed.

`factor-config/v1` freezes the full asset universe, ordered calendar, lookback, one-interval horizon, candidate baselines, cost in basis points and ordered train/validation/test endpoints. Calendar intervals are explicit observation-grid steps, not inferred exchange sessions or elapsed 24-hour windows. Splits must have at least two grid points and cannot overlap. Row omissions remain missing; there is no interpolation, forward fill or revised-observation model.

At decision grid index t, momentum is `P(t−1) / P(t−1−lookback) − 1`; reversal negates it. Both observations must be available by t. It does not search backwards for an older substitute. The label is `P(t+1)/P(t)−1`, and each endpoint must have an available close mark at its own timestamp. The last decision of each split is excluded so labels never cross split boundaries. Historical feature inputs may precede the current split, as they would in an online rolling calculation.

Selection uses the highest **validation** mean cross-sectional rank IC; configuration order breaks exact ties. Training is descriptive for these unfitted baselines. Test labels do not enter selection, and only the selected candidate's test result is evaluated. If no validation date has valid IC, test evaluation is absent.

## Diagnostics and portfolio convention

- Rank IC is Spearman correlation computed separately for each date, using ascending average ranks for exact ties. It excludes pairs with missing feature/label values and abstains for fewer than two pairs or a constant ranked vector. Mean IC is an unweighted mean of valid dates; valid-date and asset coverage counts remain visible.
- Eligible current signals are centered cross-sectional ranks and normalized to unit gross exposure. Eligibility uses current information only. Constant/insufficient signals produce a zero-position abstention.
- Every evaluated interval opens an independent fixed-unit-notional long/short sleeve at close t and fully liquidates at close t+1. `q_i = w_i/P_i(t)`, gross P&L is `sum(q_i*(P_i(t+1)-P_i(t)))`, turnover is `sum(abs(q_i)*(P_i(t)+P_i(t+1)))`, and net is gross minus `turnover*cost_bps/10000`. There is no cross-day netting or compounding.
- A missing exit mark on any nonzero position makes that interval's portfolio outcome unavailable and suppresses the whole split's aggregate gross/net/turnover/cost score. It remains in the report as a missing outcome. Zero-position intervals have zero P&L/cost and an explicit abstention state.

## Trial integrity and recovery

`run` reserves an ID and writes the raw input/config snapshots and installed-package source hashes before parsing the experiment. It adds success artifacts or a failure receipt, then writes a completion manifest. Ordinary validation failures return exit code 2 and remain registered. An interruption before completion stays visibly incomplete and cannot be overwritten; use a new ID after inspection. Missing input paths, invalid IDs or oversized files are preflight refusals before registration.

An identical completed run ID is verified and reused without rewriting bytes. Conflicting input/config/code under that ID is refused. `verify` checks input/config/code identity and all committed artifact hashes; `--recompute` additionally requires the original code hash and recreates the numerical report. The tool has no delete/overwrite operation. POSIX directory syncing is locally qualified.

Imported runs use versioned v2 trial/completion/report records and copy the raw price CSV, import contract and import manifest into the trial. Verification regenerates the normalized inputs and verifies the availability assumption. The original direct panel/config route retains its v1 schema; numerical evaluator semantics are unchanged. Raw v1 trials are not evidence of historical data-source clocks. Preparation failures are preflight refusals; experiment failures after registration remain trials.

The trial directory is the state boundary; JSON and Markdown are views of that record. Keep rejected baselines, failed runs and earlier test exposure. Do not cherry-pick run IDs to claim a fresh holdout.

## Verify and measure

```sh
source scripts/env.sh
$PY -m pytest -q -p no:cacheprovider packages/factor/tests
$PY packages/factor/tools/reference_check.py work/runs/csv-demo
$PY packages/factor/tools/measure.py --out work/new-measurements
```

The separate reference checker imports no lab code: pairwise rank counts replace sorting and explicit share/cash flows replace the evaluator's return-based portfolio formula. Tests also cover late availability, null/constant panels, duplicate keys/rows, split isolation, future-price mutation, invalid horizons, run conflicts, retained failures, tampering and interrupted registration.

The optional bounded measurement script runs 10,000 and 100,000 synthetic panel rows through the direct evaluator route. It records exact bytes, feature-cell counts, runtime, process peak RSS and report hashes, then checks numerical values against the separate reference. Human reports show at most 40 test dates. The retained [measurements](docs/MEASUREMENTS.md) describe version 0.1.0's earlier direct route. Import is demonstrated on the useful small CSV, without repeating large benchmarks for a cell-count headline.

See [architecture](docs/ARCHITECTURE.md) and [independent arithmetic](docs/INDEPENDENT_CALCULATIONS.md). It is standalone software written for this repository and contains no WorldQuant BRAIN code or results. No agent framework, paid model or live connection is required.

## Limits

- **Long-form CSV only.** No wide spreadsheets, vendor formats or named time zones; each file is capped at 16 MiB, 200,000 rows, 200 assets and 5,000 grid dates ([import contract](docs/IMPORT_CONTRACT.md)).
- **No market-data corrections.** There is no automatic corporate-action adjustment, exchange calendar, survivorship correction or executable-close assumption.
- **Idealized execution.** Fractional positions and close fills are hypothetical. Borrow, funding, spread/slippage beyond the declared cost, capacity, trading calendars, corporate actions and actual execution access are not qualified.
- **No significance claims.** There are no t-statistic, Sharpe or significance claims from these small dependent samples.
- **One look at test.** Once reported, the test is exposed; retaining a run does not guarantee an untouched holdout across later human experiments.
- **Integrity, not security.** These are application-level immutability and integrity checks, not a signature against a malicious writer replacing an entire directory. POSIX directory syncing is locally qualified; crash/power-loss durability on other filesystems is untested.
- **Qualified on one platform.** Python 3.11.15/macOS arm64; other operating systems remain unqualified. The retained measurements are not current import performance claims.
