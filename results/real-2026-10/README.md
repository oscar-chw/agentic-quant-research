# Binance daily study, protocol v1 (REAL data)

Does an LLM proposer choose better lagged-return hypotheses than enumerating the
grid or drawing K at random? In v1 the LLM arm did not run: no model call was
made, and none was faked. The grid and random arms did, and nothing survived
the test split. [Protocol v2](../forward-2026-09/README.md) re-runs the comparison on a
forward window, with one promotion rule for every arm.

| file | what it is |
|---|---|
| [protocol.json](protocol.json), [campaign.json](campaign.json), [experiment.json](experiment.json) | pre-registered 2026-10-03 18:10 (UTC+08:00), before the first run on the data at 18:16; SHA-256 of each in the [publication history](../../docs/design-history.md#publication-history), since git order is not public |
| [ablation.json](ablation.json), [hypotheses.json](hypotheses.json), [REPORT.md](REPORT.md) | the run, 2026-10-03 18:16 (UTC+08:00); never rewritten (SHA-256 of each in the [publication history](../../docs/design-history.md#publication-history)) |
| [posthoc.json](posthoc.json) | POST-HOC checks written after review ([posthoc_real.py](../../scripts/posthoc_real.py)); not pre-registered |

## Disclosures

- **The test window was not untouched.** The author's alpha-gp-lab repository
  evaluated the same 34 pairs, splits, seed 20261003,
  10 bps and the momentum-20 and reversal-1 baselines at commit `2eeb94a` of that
  repository (2026-10-03 17:56:41 +08:00; now `74c2f81` after that repository's history rewrites
  of 2026-10-04 and 2026-10-06, which kept every tree and date). That was 14 minutes before this protocol
  was registered (18:10:31). Its momentum-20 test IC of −0.00616 is this study's
  number. The protocol was registered before this repository ran anything on the
  data (protocol.json SHA-256
  `6bc1e31d934c657ede178f86b439f3b894351ce3fb8680f6d7c2e02cf903cb16`; this
  repository was published with fresh history, so git order is not available to
  check it), but 2025-01-01 → 2026-08-31 had already been looked at. v2 declares it USED.
- **The arms faced different gates.** v1 promoted an LLM selection only if it had
  also passed the loop's card rule. That rule needs mean rank IC ≥ 0.01 and
  positive sleeve net, where the sleeve cost opens and closes every position daily:
  about 20 bps a day of cost, a hurdle no slow daily signal clears. No hypothesis
  passes that rule on validation or on test (posthoc.json, `loop_card_rule_passes`),
  so the v1 LLM arm could never have been promoted. The grid and random arms faced
  only the shared gate. v2 removes the asymmetry.
- **At v1 the loop also used the test split as a filter** before the critic, across
  K cards and with no correction. Since 2026-10-03 18:40 (UTC+08:00) the loop filters on
  validation only and first shows test at the human gate.

## Results (test 2025-01-01 → 2026-08-31, 607 intervals; net per day per unit gross)

| arm | tested on validation | selected | validation IC (t; Newey-West t) | test IC (t) | test net bps/day | test turnover | promoted |
|---|---:|---|---|---|---:|---:|---|
| LLM | not run (no model call was made) | | | | | | |
| grid | 120 | reversal-40 | +0.0297 (2.01; 2.13) | −0.0023 (−0.20) | −6.89 | 0.238 | no |
| random-K | 8 | reversal-48 | +0.0248 (1.63; 1.72) | −0.0030 (−0.26) | −7.38 | 0.218 | no |

The baselines are fixed, not selected: 20-day momentum +0.59 bps/day (test IC −0.0062),
1-day reversal −15.32, equal-weight hold −3.92 (−23.6% over the split). Turnover
is traded notional per day per unit gross. Newey-West t-stats use lag 5
(posthoc.json); the rest is ablation.json.

**What carries the verdict.** The test split carries it: the grid's pick has test
t = −0.20 (Newey-West −0.21), and over 10,000 further random-K draws the selection's
test net was positive in 0.12% of them. The multiple-testing statistics are weaker
evidence:

- *Bonferroni:* the validation p of the grid pick, 0.022 one-sided × 120, is 1.000.
  Because reversal is exactly −momentum, that equals two-sided × 60.
- *Deflated Sharpe* (here applied to the daily IC series, not to returns): ablation.json
  reports 0.032. That figure uses the variance of all 120 signed trial Sharpes, which
  mostly measures the ± mirror, and counts 120 trials.
  - *Standard specification:* the null sampling variance 1/T, with the effective
    number of trials taken from the eigenvalues of the 60 lookbacks' IC correlations
    (Li–Ji 12, Nyholt 32.5). This gives **0.46 to 0.63**
    ([posthoc-dsr.json](posthoc-dsr.json)).
  - *Earlier value:* the 0.97 reported first is reached only by pairing the variance
    of the correlated trials' Sharpes, which already reflects their correlation, with
    a participation-ratio N of 2.1. That counts the correlation twice. Each correction
    alone gives 0.91 or 0.88 ([posthoc.json](posthoc.json)).
  - *Verdict:* none of these settles the result; the test split does.
- *Autocorrelation:* labels are non-overlapping one-day returns, so the daily IC
  series is close to uncorrelated even for long lookbacks. The grid pick's lag-1
  autocorrelation is −0.08. The Newey-West t is 2.13 at lag 5 and 2.14 at lag 20,
  slightly above the plain 2.01.
- *Holm across arms:* 1.000 for both arms. With two arms, both long-lookback
  reversal picks, it adds nothing; it will be reported once all three arms exist.

**Selection metric versus gate metric.** Selection uses validation IC; the gate
uses net return. The two disagree in sign for 36 of the 60 momentum lookbacks,
which have negative validation IC and positive validation net. Momentum-20, for
example, has validation IC −0.0175 and net +6.67 bps/day. Rank IC weights every
asset's rank equally, while P&L follows the largest moves.

- *If the arms had selected on validation net:* the grid would pick momentum-14
  (validation +7.69 bps/day, test −0.15 bps/day), still below the 20-day momentum
  baseline and not promoted. Random-K would pick momentum-19 (test −0.51 bps/day).
- *In hindsight:* 39 momentum lookbacks had positive test net (lookback 20 and
  23–60, best +3.79 bps/day), and 17 were positive on both validation and test.
  No selection on validation IC could reach them, and the validation-net pick
  sits just outside them.

**The pending LLM arm's possible outcomes.** Every hypothesis is already scored, so
what any 8 cards could produce is known. The selection must have at least 7 cards
ranked below it on validation IC, which leaves 113 possible picks, with test net from
−15.32 to +3.79 bps/day. 30 of those 113 would pass the shared gate; under v1's card
rule, none would.

**Why believe the numbers.** All 120 trials were prepared, run and recomputed from
frozen inputs by the factor lab. [crosscheck_real.py](../../scripts/crosscheck_real.py)
recomputes all 962 reported numbers from the raw CSVs with pandas, without importing
the factor lab; the largest difference is 8.9e-16. `ASOF_DATA_DIR=<dir> bash scripts/demo.sh`
runs it. `python scripts/posthoc_real.py --data <dir> --results <copy of this directory without posthoc.json>`
rebuilds posthoc.json.
