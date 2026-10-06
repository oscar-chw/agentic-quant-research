# plan: code-eval (sanitized copy)

## goal: Evaluate every line of code: line-level test coverage of src/ and scripts/, a line-by-line review of every file by fresh reviewers (registered confirmation code first), every finding fixed with a test or recorded with its reason

- [x] 1. Analyse: what is actually being asked, and what would prove it done | gate: grep -q '^## analysis' plan/code-eval.md
      2026-09-17T15:49:12Z exit 0 in 0s
- [x] 2. Research: prior art, constraints, unknowns resolved or explicitly deferred | needs: 1 | gate: grep -q '^## research' plan/code-eval.md
      2026-09-17T15:49:12Z exit 0 in 0s
- [x] 3. Plan, with acceptance tests written before any implementation | needs: 2 | gate: grep -q '^## plan' plan/code-eval.md
      2026-09-17T15:49:12Z exit 0 in 0s
- [x] 4. Finalise: scope, gates and done-condition agreed | needs: 3 | gate: grep -q '^APPROVED' plan/code-eval.md
      2026-09-17T15:49:12Z exit 0 in 0s
- [x] 5. Coverage of the whole non-integration suite by coverage.py (branches, subprocesses, workers): research/coverage.json and research/coverage.md (per file %, missing lines and branches) | needs: 4 | ctx: pyproject.toml | gate: test -s research/coverage.json && test -s research/coverage.md
      2026-09-17T16:13:31Z exit 0 in 0s
- [x] 6. Review, registered live code: src/pmlab/live (engine, session, feeds, detect, app, datatap, tape, paper, recorder, warmstart, chainlink, validate, replay) | needs: 4 | ctx: src/pmlab/live | gate: test -s research/code_review/live.md && grep -q '^REVIEW DONE' research/code_review/live.md
      2026-09-17T16:18:18Z exit 0 in 0s
- [x] 7. Review, registered data and evaluation code: src/pmlab/live/warehouse.py, store.py, db.py, polymarket.py, backfill.py, dataset.py, http.py, performance.py, evaluation.py, metrics.py, stats.py, confirmation.py, scripts/confirm.py | needs: 4 | ctx: src/pmlab | gate: test -s research/code_review/data_eval.md && grep -q '^REVIEW DONE' research/code_review/data_eval.md
      2026-09-17T16:12:09Z exit 0 in 0s
- [x] 8. Review, registered pricing and strategy code: fair, measure, features, pipeline, backtest, strategies, risk, pq, scoreboard, vol, routes, bo, micro, labels, fees, events, bars and every other src/pmlab module outside live/, afml/ and dashboard/ not in node 7 | needs: 4 | ctx: src/pmlab | gate: test -s research/code_review/pricing.md && grep -q '^REVIEW DONE' research/code_review/pricing.md
      2026-09-17T16:12:34Z exit 0 in 0s
- [x] 9. Review, src/pmlab/afml | needs: 4 | ctx: src/pmlab/afml | gate: test -s research/code_review/afml.md && grep -q '^REVIEW DONE' research/code_review/afml.md
      2026-09-17T16:44:35Z exit 0 in 0s
- [x] 10. Review, src/pmexec and src/pmlab/dashboard (Python and static files) | needs: 4 | ctx: src/pmexec, src/pmlab/dashboard | gate: test -s research/code_review/exec_dashboard.md && grep -q '^REVIEW DONE' research/code_review/exec_dashboard.md
      2026-09-17T16:45:17Z exit 0 in 0s
- [x] 11. Review, scripts/ part 1 (operations and live tooling: deploy, watchdog, integrity, notify, service, preflight, ready, deps, replay_backtest, backfill, history_backfill, fetch_tapes, tape_integrity, socket_experiment, paper_summary, parity, consistency_check, dashboard_visual, lifecycle) | needs: 4 | ctx: scripts | gate: test -s research/code_review/scripts_ops.md && grep -q '^REVIEW DONE' research/code_review/scripts_ops.md
      2026-09-17T16:43:56Z exit 0 in 0s
- [x] 12. Review, scripts/ part 2 (studies: walkforward, backtest, trial_registry, alpha_*, bar_study, lag_study, microstructure, vol_study, pricing_scoreboard, resolution_forensics, fit_live_models, build_lesson_l1) | needs: 4 | ctx: scripts | gate: test -s research/code_review/scripts_studies.md && grep -q '^REVIEW DONE' research/code_review/scripts_studies.md
      2026-09-17T16:41:42Z exit 0 in 0s
- [x] 13. Review, scripts/ part 3 (the book pipeline: market_analysis, afml_events, afml_extensions, feature_importance, afml_fidelity, micro_features, meta_label, afml_sizing, afml_backtest, afml_allocation, afml_pipeline) | needs: 4 | ctx: scripts | gate: test -s research/code_review/scripts_pipeline.md && grep -q '^REVIEW DONE' research/code_review/scripts_pipeline.md
      2026-09-17T16:38:14Z exit 0 in 0s
- [x] 14. Registered-code findings fixed (a failing-then-passing test each), deployed, preflight green, then the confirmation v2 registration (plan/confirm-v2.md node 8) | needs: 6,7,8 | ctx: src/pmlab | gate: python scripts/preflight.py --check
      2026-09-18T02:40:35Z exit 0 in 550s
- [x] 15. Every other finding fixed with a test or recorded with its reason; index research/code_review/README.md closed | needs: 5,9,10,11,12,13,14 | ctx: research/code_review | gate: python scripts/code_review_check.py --check
      2026-09-19T02:29:52Z exit 0 in 2s
      2026-09-19T02:31:41Z exit 0 in 2s
