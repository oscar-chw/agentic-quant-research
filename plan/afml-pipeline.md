# plan: afml-pipeline (sanitized copy)

## goal: The book's research pipeline applied end to end to Polymarket BTC 5-minute markets — market analysis, features, labels, weights, importance, models, bet sizing, backtests, allocation, deployment

- [x] 1. Market analysis (ch. 1–2, 17–19): activity and risk by Taiwan hour, UTC hour and weekday over every day since 2026-03-17 (BTC volume and realised volatility, Polymarket taker volume, trades, buy share, print density, Up rate, fair-price error), by resolution rule regime, bar arrival rates per clock, structural-break and entropy summaries per day, the recorded book by hour (spread, depth, imbalance); what each implies for features and strategies | ctx: scripts/market_analysis.py, research/market_analysis.md | gate: python scripts/market_analysis.py --check
      2026-09-17T12:52:03Z exit 0 in 1s
      2026-09-17T12:55:27Z exit 0 in 1s
- [x] 2. Event dataset (ch. 2.5, 3, 4): CUSUM-filtered decision events on per-second features for every complete window since 2026-03-17 (per-window resolution rule), one row per event and primary rule (q_taker side, fair_maker quote side, flow side), simple calculable features (time: seconds left, hour and weekday sin/cos, session; movement; volatility; volume and flow; microstructure; regime; book features where recorded), label = the side’s settlement P&L after fees > 0, span = event → window end, uniqueness and time-decay weights; day-partitioned parquet; leak tests | ctx: src/pmlab/afml/events.py, scripts/afml_events.py | gate: python -m pytest tests/test_afml_events.py -p no:warnings -q && python scripts/afml_events.py --check
      2026-09-17T13:04:37Z exit 0 in 7s
      2026-09-17T13:06:20Z exit 0 in 7s
- [x] 3. Feature importance before any backtest (ch. 7–8): purged k-fold with embargo, MDI, MDA and SFI per primary rule, clustered MDA for correlated features, PCA-rank weighted Kendall τ check, stability across time folds and hours | needs: 2 | ctx: scripts/feature_importance.py, research/feature_importance.md | gate: python scripts/feature_importance.py --check
      2026-09-18T01:10:23Z exit 0 in 9s
- [x] 4. Meta-label models (ch. 3, 6, 9): the book's bagged trees / random forest (max_samples = mean uniqueness, fit weights = average uniqueness × time decay, no class balancing: balanced weights pull P(win) to ½ and bet sizing needs calibrated probabilities, research/code_review/afml.md models 1–2) tuned by purged CV on log loss, against logistic regression (the floor) and XGBoost as a challenger (installed only after the user approves the download); refit schedule (static, daily rolling, per settled window) tested as a hyper-parameter; every trial in the registry; label-permutation and future-perturbation leak tests | needs: 3,11 | ctx: src/pmlab/afml/models.py, scripts/meta_label.py | gate: python -m pytest tests/test_meta_label.py tests/test_meta_label_leaks.py -p no:warnings -q && python scripts/meta_label.py --check
      2026-09-19T11:15:14Z exit 0 in 301s
      2026-09-19T11:52:00Z exit 0 in 297s
- [x] 5. Bet sizing (ch. 10): probability → size by the book (z statistic → CDF, averaging of active bets, discretisation, limit price), against fixed stake and Kelly on the same purged folds, in dollars | needs: 4,12 | ctx: scripts/afml_sizing.py | gate: python scripts/afml_sizing.py --check
      2026-09-19T12:51:17Z exit 0 in 16s
      2026-09-19T14:32:03Z exit 0 in 15s
      2026-09-19T14:36:59Z exit 0 in 15s
- [x] 6. Backtests by the book (ch. 11–15): combinatorial purged CV paths for each primary rule with and without its meta-label filter and sizing; PBO (CSCV) over every configuration tried; DSR with the registry's trial count; backtest statistics (ch. 14: HHI, drawdowns, time under water, implementation shortfall); strategy risk (ch. 15: the precision needed at the observed frequency and payout) | needs: 5,13 | ctx: scripts/afml_backtest.py, research/afml_backtest.md | gate: python scripts/afml_backtest.py --check
      2026-09-19T13:45:58Z exit 0 in 226s
      2026-09-19T14:01:23Z exit 0 in 207s
      2026-09-19T14:35:58Z exit 0 in 218s
      2026-09-19T14:40:33Z exit 0 in 214s
- [x] 7. Allocation (ch. 16): hierarchical risk parity across the strategies that survive stage 6, against inverse-variance and equal weights, out of sample on CPCV paths | needs: 6 | ctx: src/pmlab/afml/hrp.py, scripts/afml_allocation.py | gate: python -m pytest tests/test_afml_hrp.py -p no:warnings -q && python scripts/afml_allocation.py --check
      2026-09-19T13:57:53Z exit 0 in 10s
      2026-09-19T14:32:13Z exit 0 in 10s
      2026-09-19T14:40:43Z exit 0 in 10s
- [x] 8. Deployment lifecycle (ch. 1): graduation and decommission rules written before anything runs; each survivor of stage 6 runs live as an exploratory paper strategy after an embargo, with the registered confirmation legs shown unchanged by replay (outcome-neutral record) | needs: 6 | ctx: research/lifecycle.md, src/pmlab/live/app.py | gate: python scripts/lifecycle.py --check
      2026-09-19T14:15:36Z exit 0 in 2s
      2026-09-19T14:32:16Z exit 0 in 2s
      2026-09-19T14:40:46Z exit 0 in 2s
- [ ] 9. (Deferred by date: needs ≥ 4 weeks of recorded books, from about 2026-10-14.) Order-book models once ≥ 4 weeks of books are recorded: the same pipeline on recorded-book events (maker adverse selection first), a 1-D CNN over the last T seconds of microstructure frames as a challenger to the forest | needs: 4 | ctx: src/pmlab/afml/cnn.py, scripts/meta_label_book.py | gate: python scripts/meta_label_book.py --check
- [x] 10. Pipeline report in the book's order linking every stage, with what worked, what did not and why | needs: 1,2,3,4,5,6,7,8 | ctx: scripts/afml_pipeline.py, research/afml_pipeline.md | gate: python scripts/afml_pipeline.py --check
      2026-09-20T07:54:12Z exit 0 in 0s
- [x] 11. Model infrastructure by the book (ch. 6, 7, 9), built before stage 3's results: bagged trees / random forest with max_samples = mean uniqueness and balanced subsample weights, logistic floor and HistGradientBoosting challenger in one interface; purged k-fold search (grid and randomised with a log-uniform distribution) scored by weighted log loss inside training folds; a pipeline that passes sample weights; refit schedules (static, daily rolling, per settled window) as a search dimension; trials written to the registry; leak tests (future perturbation, label permutation at chance, scalers inside folds, no confirmation windows) | needs: 2 | ctx: src/pmlab/afml/models.py | gate: python -m pytest tests/test_meta_label.py tests/test_meta_label_leaks.py -p no:warnings -q
      2026-09-17T16:02:50Z exit 0 in 23s
      2026-09-17T16:03:51Z exit 0 in 23s
- [x] 12. Bet-sizing infrastructure by the book (ch. 10): probability to size through the z statistic and normal CDF, averaging of active bets over their spans, size discretisation, dynamic position size and limit price, all in dollars for binary payoffs held to expiry, against fixed stake and Kelly on the same folds | needs: 2 | ctx: src/pmlab/afml/sizing_pipeline.py | gate: python -m pytest tests/test_afml_sizing_pipeline.py -p no:warnings -q
      2026-09-17T15:35:47Z exit 0 in 1s
      2026-09-17T15:36:24Z exit 0 in 1s
- [x] 13. Backtest infrastructure by the book (ch. 11–15): CPCV paths over whole Taiwan days with purging and embargo for a primary rule with and without a meta-label filter and sizing; PBO by CSCV over every configuration; DSR with the registry's trial count; ch. 14 statistics (HHI of returns, drawdowns, time under water, bet timing, implementation shortfall); ch. 15 precision needed at the observed frequency and payout | needs: 2 | ctx: src/pmlab/afml/backtest_pipeline.py | gate: python -m pytest tests/test_afml_backtest_pipeline.py -p no:warnings -q
      2026-09-17T15:37:01Z exit 0 in 2s
      2026-09-17T15:41:22Z exit 0 in 2s
      2026-09-17T15:41:52Z exit 0 in 2s
- [x] 14. The book walked through snippet by snippet: pm5m node 66 (docs/afml_fidelity.md, scripts/afml_fidelity.py) | ctx: docs/afml_fidelity.md | gate: python scripts/afml_fidelity.py --check
      2026-09-17T16:04:13Z exit 0 in 17s
- [x] 15. Chapter 19 explored in full (user, 2026-09-17: "ch 19 has microstructure feature... maybe it is worth the exploration"): beyond the measures already built (tick rule, Roll, high-low volatility, Corwin–Schultz, Kyle, Amihud, Hasbrouck, VPIN with bulk volume classification), the chapter's additional microstructural features computed per event without look-ahead: distribution of order sizes and the frequency of round sizes, serial correlation of signed order flow, time-weighted execution algorithm footprints (trades clustering at regular intervals, per wallet where the tape carries one), cancellation and replacement rates from the recorded book, and options-implied information where a source exists (Deribit DVOL); a data-quality section per feature (history vs recorded-book period) | needs: 2 | ctx: src/pmlab/afml/micro_features.py, scripts/micro_features.py, research/micro_features.md | gate: python -m pytest tests/test_afml_micro_features.py -p no:warnings -q && python scripts/micro_features.py --check
      2026-09-17T16:15:59Z exit 0 in 5s
      2026-09-17T16:16:42Z exit 0 in 5s
- [x] 16. Chapter 19 features judged by the book's feature importance (stage 3's purged MDA, SFI and clustered MDA, over the price + time baseline) per primary rule and period, and added to stage 4's feature set only where they carry out-of-sample information | needs: 3,15 | ctx: research/micro_features.md | gate: python scripts/micro_features.py --check --importance
      2026-09-19T15:16:40Z exit 0 in 11s
      2026-09-19T15:18:38Z exit 0 in 11s
- [x] 17. Section-by-section checker (user, 2026-09-17: "make sure the dev is correctly implemented chapter sub chapter sub chapter by one by one"): every numbered heading of chapters 1–22 (276, numbers only in tests/afml_sections.json, compared with the private book text), one row each in docs/afml_sections/chNN.md with status implemented / applied / deviates / not applicable / deferred, code, tests or reports as evidence, and the commit it was verified at; findings table per chapter | ctx: scripts/afml_sections.py, tests/test_afml_sections.py | gate: python -m pytest -p no:warnings tests/test_afml_sections.py
      2026-09-17T16:58:07Z exit 0 in 1s
- [x] 18. Chapters 1–5 verified section by section against the code (every heading read in the book, then the code and tests; discrepancies recorded as findings, not silently fixed) | needs: 17 | ctx: docs/afml_sections, docs/afml_fidelity.md, research/book_check_part1.md | gate: python scripts/afml_sections.py --check --chapters 1-5
      2026-09-17T17:23:33Z exit 0 in 64s
- [x] 19. Chapters 6–12 verified section by section | needs: 17 | ctx: docs/afml_sections, docs/afml_fidelity.md, research/book_check_part2.md | gate: python scripts/afml_sections.py --check --chapters 6-12
      2026-09-17T18:04:29Z exit 0 in 23s
- [x] 20. Chapters 13–17 verified section by section | needs: 17 | ctx: docs/afml_sections, docs/afml_fidelity.md, research/book_check_part3.md | gate: python scripts/afml_sections.py --check --chapters 13-17
      2026-09-17T17:25:14Z exit 0 in 92s
- [x] 21. Chapters 18–22 verified section by section | needs: 17 | ctx: docs/afml_sections, docs/afml_fidelity.md, research/book_check_part3.md | gate: python scripts/afml_sections.py --check --chapters 18-22
      2026-09-17T17:21:00Z exit 0 in 11s
- [x] 22. Every section finding fixed with a failing-then-passing test or resolved with its reason; the whole book passes | needs: 18,19,20,21 | ctx: docs/afml_sections | gate: python scripts/afml_sections.py --check --closed
      2026-09-19T03:09:46Z exit 0 in 97s
      2026-09-19T03:15:00Z exit 0 in 100s
- [x] 23. Structural-break features judged by feature importance (docs/afml_sections/ch17.md F17.1): per-event causal statistics from ch. 17 on token and BTC prices known at the event second (SADF on log prices, Chu–Stinchcombe–White CUSUM, the exponential trend that beat its null inside 13–15% of windows), added to the event dataset with leak tests, then purged MDA / SFI over the price + time baseline as in node 16 | needs: 3 | ctx: src/pmlab/afml/breaks.py, src/pmlab/afml/events.py, research/afml_extensions.md | gate: python -m pytest -p no:warnings tests/test_afml_break_features.py -q && python scripts/feature_importance.py --check --family breaks
      2026-09-19T15:16:51Z exit 0 in 7s
      2026-09-19T15:18:46Z exit 0 in 7s
