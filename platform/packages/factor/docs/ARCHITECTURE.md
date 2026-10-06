# Actual implementation and boundaries

The original evaluator/trial package was authored and inspected first. Version 0.2.0 adds a deterministic price importer and versioned import provenance while preserving the evaluator. Generated fixture data and wheel outputs are verification artifacts; third-party environments are outside this code audit.

| Module | Actual responsibility | Quant/research use |
|---|---|---|
| [prepare.py](../src/factor_research/prepare.py) | Explicit column/time mapping, calendar/universe policies, canonical prepared bundle and verification | Bring ordinary long-form prices into a repeatable experiment while surfacing assumed availability |
| [artifacts.py](../src/factor_research/artifacts.py) | Existing canonical JSON/hash/exclusive-write primitives, shared inside this package | One owner for bounded file handling and local provenance |
| [contracts.py](../src/factor_research/contracts.py) | CSV/JSON schema, UTC normalization, explicit static universe/calendar and chronological split validation | Refuse ambiguous clocks, duplicate observations and silently changed experiment settings |
| [evaluator.py](../src/factor_research/evaluator.py) | Lagged momentum/reversal; tied ranks; date-wise IC; validation selection; independent fixed-notional sleeves | Compare simple hypotheses and expose regime failure, low coverage or unavailable outcomes |
| [runs.py](../src/factor_research/runs.py) | Exclusive ID reservation, input/config/code identity, success/failure completion, read-back and recomputation | Make a report reproducible and prevent restart/conflict from silently replacing trial evidence |
| [reports.py](../src/factor_research/reports.py) | Bounded Markdown diagnostic view over the same numerical report | Show selection, missingness, units and negative outcomes without hiding the full JSON |
| [synthetic.py](../src/factor_research/synthetic.py) | Deterministic trend/reversal and delayed/missing fixtures | Exercise known failure classes without private data or platform calls |
| [cli.py](../src/factor_research/cli.py) | `prepare`, `from-ohlcv`, `fixture`, `run`, `verify`, `trials`, `allocate`, `verify-allocation` entry points | Prepare, reproduce and inspect an experiment from a fresh install |
| [reference_check.py](../tools/reference_check.py) | Independent pairwise ranks and share/cash-flow arithmetic | Falsify consistent-but-wrong numerical output; does not import the lab evaluator |

Actual prepared-input path: `prepare` -> declared CSV conversion -> reuse config/panel validators -> five-file source bundle; `run --prepared` -> bundle verification and v2 source snapshots -> existing numerical path -> assumption-bearing report/completion. `verify --recompute` regenerates the import from stored raw/mapping bytes before rerunning the evaluator. Direct authored panel/config experiments retain v1 trial/report schemas. Imported trials use v2; prepared manifests and contracts each have their own v1 schema.

The numerical path is unchanged: raw/config validation -> lagged feature matrices -> train/validation evaluation -> selected candidate test -> JSON/Markdown -> completion -> read-back. The importer sorts observation keys and policy-derived sets deterministically; it never chooses splits or a universe using test returns. `known_by_cutoff` only uses observations available by a declared training cutoff. See [input contract](IMPORT_CONTRACT.md).

The evaluator is pure computation over parsed objects. Trial, prepare and fixture operations write local files through one artifact helper. There are no network clients, account lookups, threads, subprocesses or external providers in package runtime. The optional measurement tool launches bounded local child interpreters.

Data is keyed by `(UTC observation time, asset)` in a dictionary, making exact lookups O(1) average. Explicit feature computation is O(F*T*N); cross-sectional sorting is O(F*T*N log N), and report storage is O(F*T*N). The oracle deliberately uses O(N²) pairwise rank counts for independent arithmetic. The simple whole-panel representation suits the measured bounded workload; streaming/columnar output remains proposed if larger permitted inputs justify it. No speedup is claimed.

State is local per-run directories, not a mutable global experiment database. Exclusive directory creation arbitrates run-ID collisions; a missing/invalid completion marker is never success. Fresh IDs preserve interrupted/failed work. This avoids a service/database dependency for the supported single-machine workflow; multi-writer scheduling, durable recovery after every possible disk failure and hostile tamper protection remain unqualified.

Current UI is a generated Markdown report with comparison table, at most 40 test interval rows, provenance and limitations. It is a research/debugging view, not a trading dashboard. Browser interactivity, group neutralization, multi-horizon overlapping portfolios, external datasets, live platform simulation and AI hypothesis generation are proposed or out of scope, not implemented features.
