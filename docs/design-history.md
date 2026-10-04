# Design history

There are no releases, so there is no changelog. What this page keeps instead is a
decision trail: the decisions that shaped the study, in the order they were made, each
with its date and why. Every time on this page is the author's local time, UTC+08:00,
read from the development history, which is not public (see the next section).

## Publication history

**This repository's history starts fresh:** its first commit (2026-10-04) holds the
tree as it stood that day. Its first public push comes later; GitHub records that date,
not this page. The earlier development history is not public,
because it contained code that is not Oscar's to publish: modules copied from a private
codebase, replaced on 2026-10-04 (see "After the study"). Removing them from the current
tree did not remove them from the commits that held them, so those commits stay private.

**What this costs the study.** For a pre-registered study the order of events is part of
the evidence, and a public git log would have shown it. That log is not available here:
git order cannot be used to check any pre-registration claim in this repository. What
stands in for it is the SHA-256 of each registered file, recorded from the development
history at publication, with the time it was registered. A reader can check that the
files published here are byte-identical to the listed digests
(`shasum -a 256 <file>`; [tests/test_docs.py](../tests/test_docs.py) checks every row).
The times are the author's record; nothing in this repository proves them independently.

**What the public record can prove, for v2.** For v1 the order of events rests on the
author's record alone. v2 can do better: its protocol, its frozen grid and random-K
selections ([selections.json](../results/forward-2026-09/selections.json)) and its power
analysis become public with this repository's first push to GitHub, and GitHub records
that push on its own servers, independently of the author. The first public push is the
timestamp that matters; check it on GitHub. Whatever happens after it can be checked
against a timestamp the author does not control: the LLM arm's proposal call (pending;
[forward_propose.sh](../scripts/forward_propose.sh) makes it before this repository
fetches any bar after 2026-08-31, and its replay files are committed after it) and the
rest of v2's test window, 2026-09-01 → 2027-08-31. Bars from 2026-09-01 up to the push
were already public, but the frozen selections were fixed on validation and cannot use
them.

**Protocol v1** ([results/real-2026-10](../results/real-2026-10/README.md)):

| File | Registered | SHA-256 |
|---|---|---|
| [fixtures/binance_universe.json](../fixtures/binance_universe.json) | 2026-10-03 18:09:21, the universe checksums, one minute before registration | `6275a74031f3bc0a9ffc42683d9c94ad33479e1ba402e8a9a8d84d5612416d1e` |
| [results/real-2026-10/protocol.json](../results/real-2026-10/protocol.json) | 2026-10-03 18:10:31 | `6bc1e31d934c657ede178f86b439f3b894351ce3fb8680f6d7c2e02cf903cb16` |
| [results/real-2026-10/campaign.json](../results/real-2026-10/campaign.json) | 2026-10-03 18:10:31 | `57726010bd0d897fd3919ba5cb187f343542cec0b7939c6eebdaf441823c1c89` |
| [results/real-2026-10/experiment.json](../results/real-2026-10/experiment.json) | 2026-10-03 18:10:31 | `60cb991e15680c1e0e0f390a52ac8cbf8b7803581f0dad962a05093f95555c7d` |

The v1 run's outputs, written at 2026-10-03 18:16:43 and never rewritten:

| File | Written | SHA-256 |
|---|---|---|
| [results/real-2026-10/ablation.json](../results/real-2026-10/ablation.json) | 2026-10-03 18:16:43 | `cb7414b8dba2fae1b6c28c71bccf40ae2348a56f1daa974352a8d76fae15f7a2` |
| [results/real-2026-10/hypotheses.json](../results/real-2026-10/hypotheses.json) | 2026-10-03 18:16:43 | `79b6022f77a8f036344d3c9f40a8d60c87de566d7759133dd9fd29e6c38b281d` |
| [results/real-2026-10/REPORT.md](../results/real-2026-10/REPORT.md) | 2026-10-03 18:16:43 | `0dc2f3675ea3c8753fe0fd6714588f0b4c7a580c9c21ba99e037c39f4fe90145` |

**Protocol v2** ([results/forward-2026-09](../results/forward-2026-09/README.md)),
registered at 2026-10-03 18:52:14 and amended at 19:13:42, 19:28:07, 19:50:08 and
20:13:14 (amendment 4, wording only), then on 2026-10-05 (amendment 5, the model; committed 2026-10-05 with an author-set
commit date, its text completed the same day, and no LLM replay file exists in the
repository). Each file is listed as it stood after the last
change to it; earlier versions are not published:

| File | Registered, last changed | SHA-256 |
|---|---|---|
| [results/forward-2026-09/protocol.json](../results/forward-2026-09/protocol.json) | 18:52:14, amendment 4 at 20:13:14, amendment 5 on 2026-10-05 (text completed the same day) | `778ab5b408722cd8cdd6d1a596b3f394dce968b93b0fe60f585ad735bb5d4c97` |
| [results/forward-2026-09/campaign.json](../results/forward-2026-09/campaign.json) | 18:52:14 | `3cc3c622ecb36d5a17cfe52db9389c30f44a9d87c6fdc75e96f3202d0ab0062e` |
| [results/forward-2026-09/experiment.json](../results/forward-2026-09/experiment.json) | 18:52:14 | `f992086c2cf7119013cea013e577512b30ad394cddb968e16d6d715cbcb816c0` |
| [results/forward-2026-09/selection-experiment.json](../results/forward-2026-09/selection-experiment.json) | amendment 1 at 19:13:42 | `8cd956dc80bc8f004ce76603106dbe534ebb390bb50d325b743d637de260fa2c` |
| [results/forward-2026-09/selections.json](../results/forward-2026-09/selections.json) | amendment 1 at 19:13:42 (the frozen grid and random-K picks) | `32b628a9d6818e9a4b922258f3bafcaa43b94e66e2fa7c129b8b979a97f6f23f` |
| [results/forward-2026-09/power.json](../results/forward-2026-09/power.json) | amendment 2 at 19:28:07, amendment 3 at 19:50:08 | `1fef56d66a3668c56b813cee659649100eaa829deb77abd4ebe6b7d9b393ff06` |

**The protocol files still say "committed", and one cites commit hashes.** They are
registered, so they are published unchanged: editing one would change its SHA-256.
Read them this way:

- "committed" in `registered`, `amendments` and `supersedes` means added to the
  non-public development history at the time listed above.
- In v2's protocol.json, `used_windows` cites "commit 2ec30c8": that is the v1 run of
  2026-10-03 18:16:43, the outputs in the second table. It also cites "its commit
  2eeb94a": that hash belongs to the separate alpha-gp-lab repository, not this one.
  "The v1 protocol commit" is v1's registration at 18:10:31.

**Before October 2026** the code that became `packages/research` was `qrae-rd`
(July 2026): a web dashboard with scheduled research endpoints, then a local research
kernel and the bounded broker. Its version numbers (for example `qrae-rd 0.16.0`) come
from that work; its history is not part of this repository either. On 2026-10-03 at
16:29 it was replaced by a monorepo holding the packages: the research kernel (`qrae`),
market data, the vault, the QuantOS app, the factor lab and the IMC simulator. The
study's work, from the research loop to the fourth amendment of protocol v2, all fell
between 16:29 and 20:26 that afternoon and evening, done with AI coding agents under
the author's design and review.

## Decisions, in order

| When | What changed | Why |
|---|---|---|
| 2026-10-03 16:35 | Daily OHLCV import and a checksum-verified Binance fetch script | Real data enters only through declared clocks and verified checksums |
| 2026-10-03 16:45 | The research loop: propose, freeze, test, critique, human gate | The LLM drafts; deterministic code computes every number; a human decides |
| 2026-10-03 16:47 | A card must also clear a positive net-of-cost sleeve mean, not rank IC alone | A hypothesis could pass on ranking while losing after costs. Later this sleeve cost (about 20 bps a day) proved a hurdle no slow daily signal clears; see 18:16 |
| 2026-10-03 17:50 | The C++20 order-book replay port and its pybind11 binding, opt-in | Byte-identical parity first; Python stays the default because the port refuses fractional and NaN query times |
| 2026-10-03 18:03 | The critic's output schema goes into the prompt; the model is pinned; live calls are capped | The schema reached Codex as a file argument but never reached `claude -p`, so every live critic answer would have been quarantined |
| 2026-10-03 18:09 | The ablation: the LLM's cards, the enumerate-all grid and seeded random-K choose from the same scored trials | To ask whether the LLM beats a dumb search, not only whether it finds something |
| 2026-10-03 18:10 | Protocol v1 pre-registered (SHA-256 above) | Universe, splits, cost, grid, arms, selection on validation IC and the gate, fixed before any number exists |
| 2026-10-03 18:16 | v1 run on 34 Binance pairs: neither control's validation pick passes the gate on the test split | The LLM arm did not run: the `claude` CLI was not signed in at run time, so no call was made and none was faked. Under v1 the LLM's pick also had to pass the loop's card rule, which no hypothesis passes |
| 2026-10-03 18:40 | The loop filters on validation only; the critic sees validation only; test is first shown at the gate (still computed for every card until 2026-10-04 03:21) | Until this change the filter also used test, across K cards and with no correction. v1's LLM arm was registered under that rule |
| 2026-10-03 18:44 | Newey-West t, deflated-Sharpe inputs and the pending arm's possible outcomes, written to `posthoc.json` labelled POST-HOC | An independent review of v1 questioned these; they are measured without rewriting v1's outputs |
| 2026-10-03 18:52 | Protocol v2 pre-registered: one gate for every arm, on the forward window 2026-09-01 → 2027-08-31 | v1's test window had been evaluated by the author's alpha-gp-lab repository, and v1's arms faced different gates |
| 2026-10-03 19:13 | Amendment 1: every arm selects on validation net (the gate's metric); the grid and random-K picks are frozen; the model id is pinned | IC selection would pick a hypothesis predicted to fail the gate, which would make "the LLM beats the grid" nearly free |
| 2026-10-03 19:23 | `REJECTED_BY_TEST` renamed `FAILED_ON_VALIDATION`; ledger schema v2 refuses v1 ledgers | The name described the filter before 18:40 |
| 2026-10-03 19:28 | Amendment 2: the verdict is a one-sided Newey-West test on daily net, plus paired tests against both controls; power stated | Decide on the metric that selects and gates; say in advance how large an effect the window can detect |
| 2026-10-03 19:50 | Amendment 3: the delisting robustness run uses the unfilled data | Amendment 2's "drop the pair entirely" could not reproduce the frozen selections |
| 2026-10-03 20:13 | The fetcher records a delisting (a 404 after a pair's listed months) instead of stopping | A pair delisted in the forward window must still reach the fill |
| 2026-10-03 20:13 | Amendment 4, wording only; the drop run is listed as step 5 | No rule, threshold or number changed |
| 2026-10-05 | Amendment 5: the LLM arm's model becomes the open weights Qwen3.8-27B (`qwen/qwen3.8-27b:free` on OpenRouter), replacing `claude-opus-5-5`; the request settings and a rule for a retired endpoint are pre-registered. Before it, protocol.json's SHA-256 was `ecc385788da9391177b178ccb47f06b309b8b6833b5c43317729bf041f162cf2`; as first committed, `665b7c5c3c80b932fe4e878bce2265492723415bbc9d69d96460b63274cb67e7` | The author decided the arm uses no Anthropic or OpenAI model. `claude-opus-5-5` was never called (the CLI was logged out), so nothing was discarded |
| 2026-10-05 | Amendment 5's text completed, before any LLM call (`text_completed` in protocol.json): the release dates sourced to committed API responses ([model-evidence](../results/forward-2026-09/model-evidence/README.md)), the pinned revision's own date (2026-08-14) added, the contamination claim limited to the pinned weights with OpenRouter's limitation stated, and the key source written as the scripts implement it | A review found the dates unsourced, the revision's date missing, "cannot contain" too strong, and the protocol's key source at odds with the scripts. No rule, threshold, model or request setting changed |

Each amendment was made before any LLM call, before any v2 score and before this
repository fetched any bar after 2026-08-31; the amendments are recorded inside
[protocol.json](../results/forward-2026-09/protocol.json). As above, these times are
the author's record, evidenced here by the SHA-256 of each file, not by git order.

## After the study

Nothing below changed a protocol, a registered result or a frozen selection.

| When | What changed | Why |
|---|---|---|
| 2026-10-04 02:50 | The POST-HOC selection figure, drawn only from the v1 results | It shows why selection failed: the best lookback moved |
| 2026-10-04 02:52 | [How a hypothesis dies](how-a-hypothesis-dies.md); the hand-written critic no longer says "The test passed" | The critic sees validation only |
| 2026-10-04 02:56 | This page and the [docs index](README.md); stale docs renamed or fixed | They contradicted the README |
| 2026-10-04 02:59 | The README rewritten around its question | The first screen led with what had not run |
| 2026-10-04 03:21 | The loop seals the test split until the gate: pre-gate trials run without the test rows, and only a card that passes the rule and the critic gets a test number | Before, test was computed for every frozen card and REPORT.md showed it for losers too, so a reader learned which loser would have won. Demo REPORT.md files now show `sealed` for those rows; no registered result was produced by the loop, so none changed ([logged against v2](../results/forward-2026-09/DEVIATIONS.md)) |
| 2026-10-04 03:21 | The figure says "largely reversed" for Spearman −0.59, and ranks raw ICs | "Nearly reverse order" overclaimed |
| 2026-10-04 03:22 | This page's dates were generated from the git history, and CI checked them | The page had said every study change was dated 2026-10-03; the later ones were not. The generator was removed at publication, with the history it read |
| 2026-10-04 03:24 | Documentation cleanup: package docs trimmed to what the code does; `posthoc-round2.json` renamed `posthoc-dsr.json`, content unchanged | It is POST-HOC and no protocol names it; the new name says what it holds |
| 2026-10-04 03:25 | The demo runs the loop and the real-data table; the quote trial needs `--quotes` | The first thing a reader runs should be the research question |
| 2026-10-04 03:43 | `quantos-loop verify` also refuses gate files for a card that did not reach the gate, an outcome file that differs from its trial, and a REPORT.md that is not the rendering of the verified ledger | Before, deleting only a forced loser's gate trial left its test numbers unchecked, and test numbers written into a loser's outcome file or report row passed |
| 2026-10-04 22:57 | Replaced platform-derived collector code with a fresh, generic replay module: the feed handler, Parquet writer, cold tier, age-out, their tests and the archive-integrity doc are gone; [`packages/marketdata/replay`](../packages/marketdata/replay/book.py) rebuilds books from keyframes and deltas, and the connected demo stores its history as hashed JSONL | That code had been copied from a private codebase. The new module was written from the C++ port's public contract and golden cases; the parity digest (2,782 bytes, sha256 279cc869…3849), all 246 golden cases and the demo's prices are unchanged. Re-measured at 23:02, C++ batch is 25.10× Python batch (was 60.01×) because the new Python batch path is faster ([results](../packages/marketdata/native/bench/results-2026-10-04.json)) |
| 2026-10-04 | First commit of the fresh history; the first public push is dated by GitHub | See "Publication history" above |
