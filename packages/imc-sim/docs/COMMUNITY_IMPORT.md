# Import one saved community backtester log

From the monorepo root after `make install` (or use `$PY -m imc4_analysis` after `source scripts/env.sh`):

```sh
imc4-analyze import-log packages/imc-sim/examples/community-log/sample.log --config packages/imc-sim/examples/community-log/import.json --out imported-run
```

This calls the installed importer (added in `0.2.0`), existing normalized validator, existing analyzer and existing report renderer. No backtester, strategy, API, exchange account or network service is executed. The example inputs are original synthetic records authored for this package; no upstream market data or strategy is redistributed.

## Narrow source contract

Supported selection: `nabayansaha/prosperity4btest-log`, declared producer revision `0094c681f8cd019889761e6431a1a47ea151aaa8`, edition `prosperity4`. See [the five-file primary-source study](UPSTREAM_FORMAT.md). The raw log contains no version/edition header; configuration is a user declaration, not producer authentication. Identically shaped logs from another edition do not become supported by renaming a file. No official equivalence is claimed.

One UTF-8 log contains exactly one `Sandbox logs:`, `Activities log:` and `Trade History:` section in that order. Sandbox text is preserved but not interpreted or executed. Activities use the pinned 17-column semicolon header. Trade history uses the pinned seven-field objects, allowing its writer's trailing comma before a closing object brace; quoted string content is never rewritten. Standard JSON syntax is also accepted. CRLF is handled while the original bytes remain separately hashed.

The raw file must contain one declared activity day, and each section must be nondecreasing in source timestamp. Duplicate product/timestamp activity rows are refused. Merged days are unsupported; export a single day from the declared producer. The adapter does not infer official tick duration, UTC, day rollovers or order latency.

## Explicit configuration

Start from [the example configuration](../examples/community-log/import.json):

| Field | Meaning and refusal boundary |
|---|---|
| `schema` | `imc4-analysis-community-import/v1`; unknown versions refused |
| `source_format`, `producer_revision`, `edition` | Exact narrow selection above, explicitly declared |
| `round`, `day` | Nonempty round label and integer day. The day must match every activity row; the round is not present in the log and cannot be verified from it. |
| `mode` | `own_fills` requires at least one matching own trade; `quotes_only` excludes all fills and requires explicitly zero opening cash/positions |
| `self_id` | Explicit participant identifier. Exactly one of buyer/seller must match to create a fill. Neither match means market-only; both match is ambiguous and refused. |
| `quantity_unit` | `integer_units`; no lot scaling |
| `sequence_policy` | Explicit `activities_then_trades_source_order`. At each timestamp all activity rows precede own fills, preserving row order within each section; resulting sequences start at zero. This follows the pinned producer's activity-before-matching flow, not observed exchange arrival times. |
| `fill_identity_policy` | `source_index_refuse_indistinguishable`. IDs bind the full raw-file hash and trade-array index; they are record identities, not exchange execution IDs. Indistinguishable own trades at identical time/instrument/participants/price/quantity are refused instead of silently counted/deduplicated. |
| `fee_policy` | Required nonnegative `per_fill` and a nonempty `basis`. Fees are a user-declared constant in XIREC per own fill. Zero must be explicitly declared. Variable fees, access fees, financing and conversions require another verified input contract. |
| `analysis` | Existing [normalized config](../CONTRACT.md): opening cash/positions/marks, instrument limits, freshness/horizon, label and data kind. Currency must be `XIREC`, timestamp unit `source_timestamp`. No scaling or currency conversion. |

Use integer or decimal-string money in the configuration. JSON monetary numbers in the source are parsed from their decimal spelling without conversion through binary floats. Integer-only fields remain strict. The normalized output obeys the existing limit of eight fractional digits and other numeric bounds; unsupported spellings or values refuse rather than round silently.

## Data mapping and unavailable facts

| Source | Normalized use |
|---|---|
| Activity `timestamp`, `product`, `mid_price` | Quote/mark event with the explicitly merged sequence. The recorded `mid_price` is used, rather than recomputing a mark from depth. |
| Activity depth columns | Validate complete positive price/volume pairs where present, then retain raw only. Depth/spread reconstruction is not implemented. Blank paired levels are permitted; half-populated pairs refuse. |
| Activity `profit_and_loss` | Validate its numeric spelling but **do not use it** as cash, mark or the accounting oracle. It may refer to different timing or costs. |
| Trade `buyer`, `seller` | Derive buy/sell only by the configured own identity; market rows never become own fills |
| Trade `timestamp`, `symbol`, `quantity`, `price`, `currency` | Own fill fields; positive integer quantity and XIREC price required; undeclared instruments refuse |
| Absent fee, unified sequence and execution ID | Explicit policy above; never represented as observed economic certainty |

Trade-array index IDs change when raw-file bytes change, including an appended future section. This updates provenance only; it does not change earlier cash/inventory/valuation values. Do not use these IDs as trading features or as a cross-file economic deduplication key. Multiple identical legitimate executions cannot be distinguished from duplicated export records in this format; supply a separately identified normalized input instead of bypassing that refusal.

Quotes-only reports prominently show **Strategy P&L: Not evaluated**. Any zero cash/equity/P&L in normalized numerical fields is the explicitly empty-portfolio convention for quote diagnostics, not a reconstructed strategy result. Own-trade mode remains conditional on the declared identity, opening state and fee model; it does not reproduce source-reported or official P&L by assumption.

## Commitments, bounds and failures

The run preserves exact raw log and config bytes plus deterministic `normalized.jsonl`. `import-receipt.json`, `report.json` and `complete.json` bind raw/config/normalized hashes and the actual installed Python source identity. Completion hashes cover all six payload files; the receipt is written last. The CLI refuses existing outputs and invalid input before creating a report directory. A missing or mismatching completion receipt remains incomplete.

Limits stay bounded: raw log ≤16 MiB, config ≤64 KiB, combined activity/trade rows ≤10,000, normalized input ≤16 MiB/10,000 events. Market rows count against the raw-row cap even though they are excluded from own accounting. No clipping or unreported filtering is performed to fit. Larger or multi-day logs need a separately qualified adapter change, not an increased CLI limit.

The synthetic hand case ends cash 901 + one ALPHA at mark 102 = equity 1003. Four fees sum to 4; two other-party market trades are excluded. At +100 source-timestamp units, gross own-fill markouts are 6, 2, 3, then unavailable future. The implementation reuses the accounting/time logic covered by the original 30 tests and adds format, tie, commitment, malformed-row, unit, duplicate and mode checks.
