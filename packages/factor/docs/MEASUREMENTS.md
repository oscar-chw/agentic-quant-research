# Historical version 0.1.0 synthetic baseline — 2026-09-14

These retained measurements describe the original direct panel/config route. They do not measure version 0.2.0 preparation/import and must not be presented as current end-to-end import throughput. The new workflow is covered by a small hand-built equivalence/installed example instead of another headline benchmark.

The installed wheel processed deterministic synthetic panels at two sizes, with 100 configured assets and two baseline factors. Each input deliberately contains one late observation and one blank price. All feature values, date-wise rank ICs, weights and portfolio/summary calculations were checked by the independent reference implementation. No optimization or market-data performance claim is made.

| Quantity | Small workload | Larger workload |
|---|---:|---:|
| Dates × assets | 100 × 100 | 1,000 × 100 |
| CSV rows | 10,000 | 100,000 |
| Physical input bytes | 957,891 | 9,587,778 |
| Evaluated feature cells (includes null/warmup) | 20,000 | 200,000 |
| Non-null feature values | 19,594 | 199,594 |
| Run wall seconds, raw repeat 1 | 0.114386166 | 1.182140833 |
| Run wall seconds, raw repeat 2 | 0.115327291 | 1.173828584 |
| Run wall seconds, raw repeat 3 | 0.113952834 | 1.178613875 |
| Median seconds | 0.114386166 | 1.178613875 |
| Peak process RSS bytes, repeat 1 | 41,369,600 | 196,706,304 |
| Peak process RSS bytes, repeat 2 | 42,106,880 | 197,328,896 |
| Peak process RSS bytes, repeat 3 | 41,533,440 | 198,000,640 |
| Full report JSON bytes | 1,596,419 | 16,110,188 |
| Independent numerical comparisons | 38,259 | 387,759 |
| Maximum absolute numerical delta | 1.33e−15 | 2.00e−15 |

Environment: installed Python 3.11.15, macOS arm64. `time.perf_counter` wraps input reads, hashing, parsing, evaluation, JSON/Markdown serialization, trial writes/sync and final artifact read-back. It excludes interpreter startup, fixture generation and the separate reference checker. Each repeat uses a fresh process and fresh trial ID; OS cache is uncontrolled, with later repeats likely warm. RSS is the fresh child process peak reported by macOS `resource.getrusage(RUSAGE_SELF).ru_maxrss`, not a per-function allocation measurement.

All three report hashes match within each workload. The measurement tree occupied 95,486,848 physical bytes before the final receipt update; this excludes the separately retained environment, demos and profile run. Every child stayed below the 60-second cap. Run `packages/factor/tools/measure.py` on fresh inputs for current measurements.

These are two distinct baselines, not 200,000 researched alphas. Evaluated cells include unavailable/warmup cells; the non-null count is reported separately. The roughly tenfold elapsed-time increase for tenfold rows is an observation over two small local workloads, not a scalability guarantee or a claimed speedup.

## Profile and next measured constraint

A separate cProfile run on the identical 100,000-row input took 3.507 seconds of instrumented time. JSON serialization accumulated about 1.507 seconds, panel parsing 0.798 seconds and numerical evaluation 1.063 seconds; these are nested cumulative times, so they must not be added as disjoint costs. Instrumentation overhead makes this run unsuitable as a throughput repeat.

The full numerical JSON is 16.11 MB and the human view is capped at 40 test dates. A next bounded optimization could investigate less verbose or streamed feature output while preserving exact report semantics and provenance. That change was not attempted here; the current baseline and profile are retained for a fair comparison. No concurrency, native extension, remote compute or larger volume is justified solely by these numbers.
