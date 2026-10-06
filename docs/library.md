# The library: AFML chapter to module to test

`src/pmlab/afml/` holds one module per chapter or method. The bars and labels it builds on sit one level up, in
`src/pmlab/`. The modules were written for one market, Polymarket's BTC 5-minute Up/Down tokens: binary bets held to
expiry, sampled from per-trade data. Some names reflect that, such as `WINDOW_SECONDS = 300` and the `events` module's
per-window decision events.

| AFML concept (chapter) | module | key names | tests |
|---|---|---|---|
| Standard bars: time, tick, volume, dollar (2) | `pmlab/bars/standard.py` | `TimeBars`, `TickBars`, `VolumeBars`, `DollarBars` | `test_bars_standard.py` |
| Imbalance and run bars, tick rule (2) | `pmlab/bars/imbalance.py`, `runs.py`, `tickrule.py` | `ImbalanceBars`, `RunBars` | `test_bars_imbalance.py`, `test_bars_runs.py` |
| CUSUM event sampling (2, 17) | `pmlab/labels.py`, `afml/events.py` | `cusum_filter` | `test_labels.py`, `test_afml_events.py` |
| Triple barrier, events, side and size labels (3) | `afml/labeling.py`, `pmlab/labels.py` | `get_events`, `get_bins`, `triple_barrier` | `test_afml_labeling.py`, `test_labels.py` |
| Meta-labelling (3) | `pmlab/labels.py`, `afml/models.py` | `meta_label`, `MetaModel` | `test_meta_label.py`, `test_meta_label_leaks.py` |
| Uniqueness, sample weights, sequential bootstrap (4) | `afml/sampling.py` | `indicator_matrix`, `seq_bootstrap` | `test_afml_sampling.py` |
| Fractional differencing, fixed-width window (5) | `afml/fracdiff.py` | `frac_diff_ffd`, `min_ffd` | `test_afml_fracdiff.py` |
| Ensembles and bagging (6) | `afml/ensemble.py` | | `test_afml_ensemble.py` |
| Purged k-fold with embargo (7) | `afml/cv.py` | `PurgedKFold`, `train_times`, `cv_score` | `test_afml_cv.py`, `test_afml_embargo.py` |
| MDI, MDA, SFI, clustered importance (8) | `afml/importance.py`, `afml/event_importance.py` | `mdi`, `pca_fit`, `weighted_tau` | `test_afml_importance.py`, `test_feature_importance.py` |
| Hyper-parameter search with purged CV (9) | `afml/tuning.py` | | `test_afml_tuning.py` |
| Bet sizing (10) | `afml/betsizing.py`, `afml/sizing_pipeline.py` | `bet_size` | `test_afml_betsizing.py`, `test_afml_sizing_pipeline.py` |
| Combinatorial purged CV, PBO by CSCV (11, 12) | `afml/cpcv.py`, `afml/backtest_pipeline.py` | `cpcv_splits`, `paths`, `pbo` | `test_afml_cpcv.py`, `test_afml_backtest_pipeline.py` |
| Synthetic data, optimal trading rules (13) | `afml/synthetic.py` | `otr_mesh` | `test_afml_synthetic.py` |
| Backtest statistics, PSR, deflated Sharpe (14) | `afml/backtest_stats.py`, `pmlab/metrics.py` | `deflated_sharpe_ratio` | `test_afml_backtest_stats.py`, `test_metrics.py` |
| Strategy risk (15) | `afml/strategy_risk.py` | | `test_afml_strategy_risk.py` |
| Hierarchical risk parity (16) | `afml/hrp.py` | `hrp`, `quasi_diagonal` | `test_afml_hrp.py` |
| Structural breaks: CUSUM, SADF, Chow (17) | `afml/breaks.py`, `afml/break_features.py` | `bsadf`, `chow_dfc` | `test_afml_breaks.py`, `test_afml_break_features.py` |
| Entropy features (18) | `afml/entropy.py` | `lempel_ziv`, `kontoyiannis` | `test_afml_entropy.py` |
| Microstructure: Roll, Kyle, Amihud, Corwin-Schultz, VPIN (19) | `afml/microstructure.py`, `pmlab/micro.py`, `afml/micro_features.py` | `vpin`, `corwin_schultz_spread` | `test_afml_microstructure.py`, `test_micro.py`, `test_afml_micro_features.py` |
| Multiprocessing and vectorisation (20) | `afml/parallel.py`, `threadcaps.py`, `threadio.py` | | `test_afml_parallel.py` |
| Brute force: trading trajectory as integer search (21) | `afml/trajectory.py` | | `test_afml_trajectory.py` |

The Sharpe ratio, the probabilistic Sharpe ratio and the deflated Sharpe ratio follow their standard published forms
(Bailey and López de Prado, 2012 and 2014). Every other formula is referred to by its AFML equation or snippet number in
the claim notes, not reproduced.

## What is here and what is not

Published: the library above and the research modules it imports (`pmlab/fair.py`, `features.py`, `strategies.py`,
`backtest/`, `evaluation.py` and the market-data clients `binance.py` and `polymarket.py`), because the event-level
modules depend on them.

Not published: the live paper-trading engine and dashboard, the order-execution layer, the data recorder and its
cache, the study scripts and their outputs, a trained model file, and the platform rewrite (`v2`). Tests that need any
of these were either left out or are listed in `tests/unpublished.txt` with the missing input.
