# Ordinary price CSV to verified experiment inputs

`prepare --prices <csv> --contract <json> --out <new-directory>` accepts one UTF-8 long-form CSV with one row per asset/observation timestamp. It does not fetch data, transpose wide spreadsheets, infer a vendor format or automatically adjust prices. The supported contract is `factor-price-import/v1`; use the committed [example](../examples/price-import/contract.json) as a starting point.

## Required declarations

| Contract field | Meaning and supported choices |
|---|---|
| `columns` | Logical `time`, `asset`, `price`, optional `group`, and both `observed_at`/`available_at` in supplied-clock mode. Every CSV column must be mapped once. No implicit ignored columns. |
| `timestamps` | Encoding for each mapped time field; independent encodings allow date-only price keys and full availability timestamps. |
| `availability` | `columns`, or `assumed_delay` with explicit `observed_delay_seconds` and `available_delay_seconds`. Integer seconds must satisfy `0 <= observed <= available <= 31536000`. No default delay. |
| `universe` | `explicit` with unique asset names (sorted in generated config), or `known_by_cutoff` with an absolute training-period timestamp. The latter includes only assets with a row whose observation and availability times are at/before that cutoff. It uses presence, not future price performance. Rows for excluded assets are refused. |
| `calendar` | `fixed_step` with absolute start/end and positive integer step_seconds, or `observed_union` with start/end bounds. Fixed grids must land exactly on end, retain missing grid rows and refuse off-grid times. Union grids use sorted observed timestamps and cannot discover a date missing for all assets. Neither is an exchange calendar. |
| `price_semantics` | Kind (`price`, `unadjusted_close`, `adjusted_close`), unit and adjustment_notes. These are user declarations, not transformations or evidence that the price/adjustments are correct. |
| `experiment` | Existing lookback, horizon=1, cost_bps, momentum/reversal candidates and absolute train/validation/test start/end timestamps. Existing validation applies unchanged. |

All split and range endpoints include offsets and must resolve to the generated grid. Calendar/universe construction never chooses boundaries from returns or expands the universe using the test period. Missing asset/grid rows and blank prices stay missing; no imputation or fill-forward occurs. Prices must be positive finite or blank; asset names cannot be empty or have edge whitespace.

## Timestamp encodings

- `{"format":"date","timezone":"UTC","time_of_day":"16:00:00"}`: strict YYYY-MM-DD input plus explicitly declared HH:MM:SS. This assigns an observation timestamp, not an inferred exchange close. Date-only data without the time declaration is refused.
- `{"format":"iso8601","timezone":"offset_in_value"}`: every value must include its own offset. Naive strings are refused.
- `{"format":"iso8601","timezone":"+08:00"}`: naive strings receive this declared fixed offset; aware strings must already match it. A conflicting +00:00 value is refused instead of silently relabeled.
- `{"format":"unix","unit":"ms","timezone":"UTC"}`: signed integer epochs; units are explicitly s, ms or us. No unit guessing or floating epochs. Wrong units fail datetime bounds or the declared date range/grid.

Fixed offsets range through ±14:00. Named zones and DST inference are unsupported; preprocess them with an explicitly audited conversion if needed. Source timestamps after the observation range can be valid availability clocks for late data; observation keys themselves must be inside the declared grid/range.

## Provenance and the supported run route

Prepared output is five files: `raw-prices.csv`, `import-contract.json`, `panel.csv`, `config.json`, `import-manifest.json`. The raw source/contract bytes are retained exactly. Panel rows and inferred sets are canonically sorted; equivalent shuffled input produces identical normalized panel/config but a different raw-input hash and manifest. The manifest includes hashes, missingness and explicit timing/universe/price assumptions.

`run --prepared <directory>` snapshots the bundle into versioned v2 trial/completion/report records. `verify <run-directory> --recompute` validates raw/mapping/output hashes, regenerates normalized inputs and assumptions, then reruns the unchanged numerical evaluator under the original installed-code identity. A verified run uses its own snapshots even if the original prepared directory changes later.

Both JSON and Markdown report either **ASSUMED_DELAY** or **PROVIDED_CLOCKS_UNAUDITED**, with `historically_point_in_time_verified=false`. Supplied clocks may be useful for temporal filtering but are not independent evidence that an old research result was point-in-time valid. No corporate-action, exchange-calendar, survivorship or executable-price assurance follows from importing rows.

The old `run --panel ... --config ...` path remains for explicitly authored v1 experiments, and refuses normalized inputs next to an import manifest to prevent accidental loss of assumptions. Do not intentionally strip/copy the normalized files and present them as source-verified; these unsigned local manifests are not hostile-tamper proof.

## Repeat, refusal and resource boundary

Validation occurs before creating an output directory. Identical destination/input/contract is verified and reused byte-for-byte. A conflict, symlink, incomplete output or tampered member is refused; no overwrite/delete route is provided. A prepare interrupted after directory reservation stays incomplete and requires a fresh directory after inspection. Imported run failures follow the existing immutable trial behavior.

Each raw or generated file is capped at 16 MiB, rows at 200,000, assets at 200 and grid dates at 5,000. Generated outputs that exceed the file cap are refused before publication. Processing uses an in-memory dictionary and sorting; this is a bounded local import, not a streaming warehouse ingester. No large-scale import throughput claim is made in this release.

## Hand-checkable example

The original `Date,Ticker,Close` example has 35 hand-authored rows, three assets and 12 fixed daily grid points. A is absent on January 7 and B is blank on January 8. The assumed zero-delay model reproduces [manual-panel.csv](../examples/price-import/manual-panel.csv) exactly and [manual-config.json](../examples/price-import/manual-config.json) as JSON values. These oracles are separately specified, not generated by the importer.

[prices-with-clocks.csv](../examples/price-import/prices-with-clocks.csv) supplies Seen/Ready columns. The January 6 A price is available only January 8, so it cannot enter the January 7 feature. Tests additionally mutate test-period prices and confirm unchanged earlier features, development results and selected baseline; input order and test-only asset mistakes are tested separately.
