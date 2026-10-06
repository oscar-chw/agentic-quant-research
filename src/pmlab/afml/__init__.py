"""Chapters of López de Prado, "Advances in Financial Machine Learning" (Wiley, 2018), one module each.

Earlier chapters live elsewhere: ch.2 bars (pmlab.bars), ch.2.5 CUSUM, ch.3 labels, ch.4 average
uniqueness and ch.7 purged k-fold (pmlab.labels), ch.14 PSR/DSR (pmlab.metrics), first ch.19 measures
(pmlab.micro).

- events          ch.2.5, 3, 4, 7  the event dataset: CUSUM events, primary-rule sides, settlement labels and
                  meta-labels, label spans, training-fold weights, purged day folds (scripts/afml_events.py builds it)
- seasonality     ch.2   clock profiles (Taiwan hour, weekday) with day-block intervals, a heterogeneity test and
                  activity adjustment (scripts/market_analysis.py)
- labeling        ch.3   the book's general labelling: volatility target, triple barrier, side and meta-labels, rare labels
- sampling        ch.4   indicator matrix, sequential bootstrap, return-attribution and time-decay weights
- fracdiff        ch.5   fixed-width fractional differentiation, ADF, minimum d
- ensemble        ch.6   accuracy of a bagging classifier
- cv              ch.7   purging, embargo, PurgedKFold (an sklearn splitter), CV score weighted in fit and score
- importance      ch.8   MDI, MDA, SFI, clustered MDA
- tuning          ch.9   purged grid and randomised search, a pipeline that takes sample weights, log-uniform
- betsizing       ch.10  size from probability, active-bet averaging, discretisation, dynamic size and limit price
- cpcv            ch.11-12  combinatorial purged CV and paths; PBO by CSCV (Bailey et al. 2017)
- synthetic       ch.13  Ornstein-Uhlenbeck calibration and the optimal trading rule mesh
- backtest_stats  ch.14  bet timing, holding period, HHI concentration, drawdown and time under water
- strategy_risk   ch.15  Sharpe of binary bets, implied precision and frequency, probability of failure
- hrp             ch.16  hierarchical risk parity and the appendix's Monte Carlo against IVP and minimum variance
- breaks          ch.17  Brown-Durbin-Evans and Chu-Stinchcombe-White CUSUM, Chow-type DF, SADF, QADF, CADF, SMT
- entropy         ch.18  plug-in, Lempel-Ziv, Kontoyiannis estimators and encodings
- microstructure  ch.19  Hasbrouck lambda, Beckers-Parkinson and Corwin-Schultz volatility, the Corwin-Schultz spread
                  series, VPIN with bulk volume classification
- parallel        ch.20  linear and two-nested-loops partitions, molecule dispatch (the book's mpPandasObj), the serial
                  debugging path, asynchronous calls with progress, on-the-fly output reduction and the column-block
                  principal components, sized for the 4 workers of docs/hardware.md
- trajectory      ch.21  the optimal trading trajectory as an integer search: ordered partitions of capital, signed
                  weight vectors, exhaustive evaluation over trajectories and the static local-optimum benchmark

Docstrings cite section numbers of the book; they do not reproduce its text. docs/afml_fidelity.md maps every numbered
snippet of the book to its function and test (scripts/afml_fidelity.py --check).
"""
