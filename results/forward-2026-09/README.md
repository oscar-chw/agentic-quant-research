# Binance daily study, protocol v2 (REAL data, PENDING)

[protocol.json](protocol.json) was registered on 2026-10-03 at 18:52 (UTC+08:00) and
amended four times by 20:13 the same evening (amendments 1-4, recorded in the file; 4
is wording only), and a fifth time on 2026-10-05 (amendment 5: the model; its text was
completed the same day, before any LLM call, with sourced dates, the key source and a
reasoning effort the model offers). As
last amended its SHA-256 is `778ab5b408722cd8cdd6d1a596b3f394dce968b93b0fe60f585ad735bb5d4c97`; every registered
file's digest is in the [publication history](../../docs/design-history.md#publication-history).
All seven versions came before any LLM call and before this repository fetched any bar
after 2026-08-31. This repository was published with fresh history, so git order is
not available to check that; the evidence is each file's SHA-256 and its date. v2 re-runs the
[v1 comparison](../real-2026-10/README.md) with these changes:

- **One rule for every arm, and a matching selection metric.** Every arm selects
  the hypothesis with the highest validation **rebalanced net**, the metric the
  gate uses, and the same gate promotes it. Selecting on rank IC instead is
  reported as secondary. On this window the two disagree: IC selection would pick
  reversal-02 for the grid, at −11.63 bps/day of validation net, so "the LLM beats
  the grid" would be nearly free.
- **The non-LLM selections are frozen now.** [selections.json](selections.json)
  holds the 120-row validation table and the seeded random-K draw. On net, the grid
  picks momentum-25 (+3.21 bps/day) and random-K picks momentum-19 (+2.21). The
  scoring run refuses to proceed unless it reproduces these numbers. A pandas
  recomputation from the raw CSVs matches the table to 1.1e-16.
- **A test window no selection can use.** 2025-01-01 → 2026-08-31 was v1's test
  split, and alpha-gp-lab had already evaluated it, so it is now validation. Test is
  2026-09-01 → 2027-08-31, scored once. Bars from 2026-09-01 to 2026-10-03 were
  public at registration but not fetched here. The frozen selections cannot depend
  on them. The LLM's proposals could depend on them only through the model's
  training data, which the contamination clause covers.
- **A pinned open-weight model.** Amendment 5 (2026-10-05) replaced `claude-opus-5-5`,
  which was never called, with `qwen/qwen3.8-27b:free` on OpenRouter: the open weights
  `Qwen/Qwen3.8-27B` at Hugging Face revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
  Apache-2.0. The arm uses no Anthropic or OpenAI model. The pinned revision is dated
  2026-08-14, the Hugging Face repository was created on 2026-08-05 and OpenRouter listed
  the model on 2026-08-14 ([model-evidence](model-evidence/README.md): the API responses,
  committed). Those weights were published before any test-window bar existed, so they
  cannot have been trained on one. OpenRouter does not say which revision or quantisation
  it serves, and the model check compares only the id string, so this rests on OpenRouter
  serving that release. Requests are fixed in
  advance: temperature 0, `max_tokens` 8192, reasoning effort low (the smallest the
  [listing](model-evidence/openrouter-qwen38-free.json) offers; it has no "none") with the
  reasoning excluded from the response; it still counts against `max_tokens`. An answer whose
  response `model` is not the pinned id (with or without `:free`), that was cut off, that
  carries an error, or that is empty is refused and not recorded. Each accepted response's
  model, provider and id go into the replay's provenance. If the free endpoint is retired
  before the critic runs, the same weights run on another host, logged as a deviation;
  never a different model.
- **A verdict in code, on the metric that selects.** `ablation.verdict` tests each
  arm's daily test net with a one-sided Newey-West test, and requires promotion.
  The LLM helps only if its arm helps and paired one-sided Newey-West tests on the
  daily LLM-minus-grid and LLM-minus-random-K net differences both give p < 0.05.
  Rank IC is reported, but only as a secondary metric. A verdict on IC was predicted
  to fail: 0 of 120 hypotheses have both positive validation net and positive IC, and
  every net-selected pick has negative validation IC.
- **Delisting in code.** `ablation fill` extends every pair to the test end at its
  last close and lists the filled dates. `DELISTED=drop` is the robustness run on
  the unfilled data: a pair is absent only on the dates it lacks, so the frozen picks
  still reproduce. Both paths are tested with a pair delisted in the test window,
  through `ablation.run` and through `scripts/real_run.sh` itself (`tests/test_real_run.py`).
  `scripts/fetch_binance_daily.py` treats a 404 after a pair's listed months as the
  end of its listing and records it in its manifest, so a delisted pair reaches the
  fill.

| step | command | needs | status |
|---|---|---|---|
| 1. live smoke on the SYNTHETIC panel (≤ 3 calls), then the real proposals alone (1 call, no prices read) | `bash scripts/forward_propose.sh` | `OPENROUTER_API_KEY`, or `~/.config/openrouter/api_key` when it is unset | pending: not run yet |
| 2. commit both REAL LLM OUTPUT replays | `git add apps/quantos/examples/loop/replay.*.json` | step 1 | pending |
| 3. fetch bars to 2027-08-31; commit `fixtures/binance_universe_forward.json` | `scripts/fetch_binance_daily.py` | 2027-09 | pending |
| 4. score all arms; the critic is called live (≤ 8 calls) | `STUDY=results/forward-2026-09 ASOF_DATA_DIR=<dir> bash scripts/real_run.sh --live` | steps 2 and 3 | pending |
| 5. delisting robustness run on the unfilled data | `DELISTED=drop STUDY=results/forward-2026-09 ASOF_DATA_DIR=<dir> WORK_DIR=work/forward-drop bash scripts/real_run.sh` | step 4 | pending |

To rebuild selections.json from data through 2026-08-31, run `factor-research
from-ohlcv` with [selection-experiment.json](selection-experiment.json), then
`python -m quantos_showcase.ablation select`. That experiment puts the validation
window in the factor lab's last split, the only one that may end at the last
available bar.

Power is low, and that was stated before any LLM call, from validation data only
([power_forward.py](../../scripts/power_forward.py) writes [power.json](power.json)).
The figures below are break-even means: at such a mean, a one-sided 5% test over one
year (364 intervals; computed with 365) rejects half the time. The formula is
1.645 × (HAC sd per day) / √365.

- *One arm's daily net:* about 8.7 bps/day. If momentum-25's true mean equals its
  validation 3.21, the test rejects with probability about 15%. 80% power needs
  about 13.2 bps/day.
- *The paired difference:*
  - 1.4–1.5 bps/day for an adjacent lookback;
  - 3.6–4.1 for momentum-20 or momentum-19 against momentum-25;
  - 13.7 for reversal-02.

  For a neighbouring pick, the arm-level test is the one that binds.

A null is the likely outcome, and it will be reported as one.

Departures from the protocol are logged, dated, in [DEVIATIONS.md](DEVIATIONS.md).
