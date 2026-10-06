# Protocol v2: departures

[protocol.json](protocol.json) (`deviations`) asks that any departure from it be
written here before the study is scored. Entries are dated; none changes a rule, a
threshold, a frozen selection or any name the protocol cites
(`quantos_showcase.ablation.select`, `.fill_missing`, `.verdict`).

## 2026-10-04: the research loop seals the test split until its gate (code only)

- **What changed.** A code change on 2026-10-04 at 03:21 (UTC+08:00), after registration
  (2026-10-03 18:52 to 20:13; protocol.json SHA-256
  `ecc385788da9391177b178ccb47f06b309b8b6833b5c43317729bf041f162cf2`). Git order is not
  public ([publication history](../../docs/design-history.md#publication-history)):
  - `quantos_showcase.loop` runs every pre-gate trial on the frozen prices without
    the test split's rows, and computes test only for a card that passes the
    validation rule and the critic;
  - `quantos_showcase.ablation.llm_arm`'s cross-check skips a sealed test split.

  A change at 03:43 the same morning then makes `loop.verify` also refuse anything
  under `run/gate/` for a card that did not reach the gate, an outcome file that
  differs from its recomputed trial, and a `REPORT.md` that is not the rendering of
  the verified ledger.
- **Why.** Before, the loop computed test for every frozen card and its report showed
  the numbers for losers too, so "test is first shown at the gate" held only for the
  filter and the critic.
- **Effect on v2's score: none.**
  - Every arm, the LLM's included, is still scored from the grid trials, which this
    change does not touch.
  - The LLM arm's cards are still cross-checked against the grid trial of the same
    hypothesis on validation. A sealed trial reproduces the full trial's validation
    metrics exactly, because the factor lab never labels across a split.
  - The critic saw validation evidence only before and after, and the replay's
    prompt hashes did not change.
- **Evidence.**
  - `apps/quantos/tests/test_ablation.py`: the LLM arm runs through the sealed loop and
    its cross-check against the grid.
  - `apps/quantos/tests/test_loop.py`: sealed trials hold no test row, and losers have
    no test number in the ledger, outcome files or report. `verify` refuses each tamper
    (`test_test_metrics_exist_only_for_gate_survivors`, the forced-gate,
    stray-outcome and report tests).
