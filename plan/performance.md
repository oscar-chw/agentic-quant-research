# plan: performance (sanitized copy)

## goal: Fast where it matters, measured rather than asserted: vectorised maths, the book's molecules across processes, threads only where they actually help

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/performance.md
      2026-09-19T08:27:53Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/performance.md
      2026-09-19T08:27:54Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/performance.md
      2026-09-19T08:27:54Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/performance.md
      2026-09-19T08:27:54Z exit 0 in 0s
- [x] 5. The benchmark harness: `scripts/perf.py` with a named benchmark per hot path (replay events/s, bar build rows/s, feature build rows/s, label spans/s, fair price per second, a model fit, a parquet day read, the live loop's per-second work), each with a budget, a warm-up, repeats, and a report; `--check` fails when a budget is missed | needs: 4 | ctx: scripts/perf.py | gate: python scripts/perf.py --check --quick
      2026-09-19T08:48:49Z exit 0 in 31s
      2026-09-19T10:19:15Z exit 1 in 75s
      2026-09-19T10:23:16Z exit 0 in 69s
      2026-09-19T10:41:57Z exit 0 in 53s
- [x] 6. Thread discipline: BLAS and OpenMP thread counts pinned per process before numpy loads, workers capped so 4 × threads never exceeds the cores the live app can spare, and a test that a worker pool does not oversubscribe | needs: 5 | ctx: src/pmlab/afml/parallel.py | gate: python -m pytest -p no:warnings tests/test_parallel_threads.py -q
      2026-09-19T08:56:14Z exit 0 in 9s
      2026-09-19T10:19:26Z exit 0 in 11s
      2026-09-19T10:42:07Z exit 0 in 9s
- [x] 7. Vectorise the hot paths the benchmarks name, keeping results bit-identical where the pipeline compares numbers and documenting where they cannot be (float addition is not associative) | needs: 5 | ctx: src/pmlab/afml, src/pmlab/features.py | gate: python scripts/perf.py --check --vectorised
      2026-09-19T09:01:44Z exit 0 in 22s
      2026-09-19T10:19:52Z exit 0 in 25s
      2026-09-19T10:42:27Z exit 0 in 20s
- [x] 8. Research jobs run through the book's molecules: the event build, feature importance and the replay use `pmlab.afml.parallel` rather than their own pools, with ordered reduction where numbers are compared and checkpoints that survive a kill | needs: 6,7 | ctx: scripts | gate: python scripts/perf.py --check --parallel
      2026-09-19T09:21:00Z exit 0 in 66s
      2026-09-19T10:21:22Z exit 0 in 89s
      2026-09-19T10:43:38Z exit 0 in 70s
- [x] 9. I/O off the critical path: parquet day reads and HTTP backfills on a thread pool, the live loop's disk writes off its own thread, with a test that the trading loop's per-second work never waits on a read | needs: 6 | ctx: src/pmlab/live, src/pmlab/store.py | gate: python -m pytest -p no:warnings tests/test_io_threads.py -q
      2026-09-19T09:32:01Z exit 0 in 7s
      2026-09-19T10:21:32Z exit 0 in 9s
      2026-09-19T10:43:46Z exit 0 in 8s
- [x] 10. The live loop's own budget: per-second work measured end to end (feed to decision to record), a budget set from the measurement, and an alert when a window exceeds it; nothing here may change registered behaviour before 2026-10-24 | needs: 5 | ctx: src/pmlab/live | gate: python scripts/perf.py --check --live
      2026-09-20T05:37:09Z exit 0 in 165s
      2026-09-20T06:21:14Z exit 0 in 178s
- [x] 11. Written down: docs/performance.md — the four kinds of parallelism and which belongs where, the budgets and how to re-measure them, what must stay deterministic and why, and the rule that the trading process always keeps its core | needs: 8,9,10 | gate: test -s docs/performance.md && python scripts/perf.py --check
      2026-09-20T05:41:31Z exit 0 in 258s
