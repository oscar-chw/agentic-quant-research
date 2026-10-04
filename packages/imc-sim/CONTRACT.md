# Normalized event contract v1

This is a new, tool-owned schema. It does not assert Prosperity 3/4 format equivalence. One run represents one continuous opaque clock; normalize round/day resets into separate runs until a verified adapter exists. All instruments share one explicitly declared currency; instrument quantity is integer units and price is that currency per unit.

## Config — exactly one first record

Required fields: `type="config"`, `schema="imc4-analysis/v1"`, `label`, `data_kind` (`synthetic` or `normalized_offline`), `currency`, `timestamp_unit`, `start_timestamp`, `initial_cash`, `mark_max_age`, `markout_horizon`, `instruments`.

`instruments` maps 1–64 nonempty labels to `{initial_position, limit, initial_mark?}`. Position is an integer in ±1e9; limit is an integer from 0 to 1e9. An initial position may breach its declared limit, which is reported rather than silently clipped. Nonzero initial position requires a positive initial mark. Opening marks are available at `(start_timestamp, -1)`. The opening equity baseline is initial cash plus opening positions at their explicit marks, not a retrospective first quote.

Timestamps/sequences and freshness age are nonnegative integers ≤1e15; horizon is positive ≤1e15. Events cannot predate the configured start. Labels/notes are nonempty text ≤200 characters without ASCII controls. Unknown object fields, duplicate JSON keys, unsupported schema, blank lines and non-UTF-8 input are refused.

## Event fields

| Kind | Common required fields | Additional fields |
|---|---|---|
| `quote` | `type`, integer `timestamp`, integer `sequence`, declared `instrument` | Exactly one of: positive `mark`; or positive `bid` and `ask` with bid ≤ ask. Midpoint of bid/ask is the analysis mark. Optional `note`. |
| `fill` | `type`, integer `timestamp`, integer `sequence`, declared `instrument` | Unique `fill_id`, `side` buy/sell, integer `units` 1..1e6, positive `price`; optional nonnegative `fee` (default 0), `note`. |

All `(timestamp, sequence)` keys are strictly increasing across the run, including across instruments. Sequences can restart at a later timestamp. Every duplicate fill identity is refused, regardless of whether values agree. The analyzer neither deduplicates retries nor guesses whether identical economic values mean the same trade; resolve provenance upstream.

Money inputs accept integers or plain decimal strings, with at most eight fractional digits and absolute value ≤1e12. Scientific notation, floating JSON numbers, NaN, Infinity and booleans are refused. Initial cash may be negative; prices/marks must be positive; fees cannot be negative. Midpoint computation may introduce a ninth decimal place, retained exactly. The admitted 10,000 events, 64 instruments and numeric bounds fit the 50-digit decimal context used for all monetary calculations. Full JSON money values are decimal strings, not rounded floats; SVG geometry alone uses floating-point display coordinates.

## Output validity

`imc4-analysis-report/v1` includes input SHA-256/bytes, tool version, declared units, summary, snapshots, fills and detail-retention policy. Each snapshot records cash, positions, quote provenance/age, marked position value, equity, opening-relative P&L, their applicable changes and breaches. Summary values/counts include every event plus the opening snapshot for opening limit/freshness checks.

`equity`, `pnl` and valid `position_value` are null if a **held** instrument has no fresh mark. Missing marks on flat instruments do not invalidate cash-only equity. `last_known_equity` carries stale known marks for diagnosis, but is null if any held mark is entirely missing. An unavailable-to-valid transition has null equity change because the immediate prior valid valuation is unknown; no value is invented to bridge the gap.

Markouts have `available`, `unavailable_future`, `missing` or `stale` status. A target beyond final event timestamp is unavailable. Otherwise the last observed quote at/before the target (including the latest sequence at the target timestamp) is used only if age ≤max age. Per-unit markout is `(target mark − execution price)` for buys and its negative for sells. Gross total multiplies units; fees are excluded from markout and included in cash accounting. There is no interpolation, decision signal or profitability inference.

`summary.breach_observations` counts each breached instrument at each snapshot; it is not a count of independent breach episodes. `summary.adverse_markout_fills` counts negative gross diagnostics; it does not identify a market-impact or informed-trader mechanism. Sampled retention never changes exact summary values/counts. It can omit extrema and later anomalies, and that limitation is visible in JSON/HTML. Use full JSON within the cap for event-by-event drill-down.

The complete bundle has `report.json`, `report.html`, and a last-written `complete.json` with SHA-256 and size of the two reports. A completion file must parse and match its actual files to be accepted. An absent, partial or mismatching receipt cannot certify a report. No hostile-file, cross-platform power-loss or simultaneous-writer durability certification is claimed.

Both report and completion include `code_identity`: SHA-256 of a canonical JSON array of sorted relative Python-source paths and their SHA-256 digests. Hash the actually imported package directory, including the fingerprint implementation; ignore bytecode/cache files. This identifies an unreleased installed source revision independently of `tool_version`. Fixture input has its own byte hash. Build metadata, interpreter and stdlib are outside the code fingerprint; runtime implementation/version are recorded separately. This is identity and reproducibility evidence, not cryptographic publisher authentication.
