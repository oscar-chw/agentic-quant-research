# Docs index

Start at the [root README](../README.md): the question, the result, the figure and the demo.
This page lists every Markdown page in the repository, grouped by what you want to do
([tests/test_docs.py](../tests/test_docs.py) checks that none is missing).

Three short paths: **researchers** read *Understand it* top to bottom, then the two study READMEs;
**LLM-safety reviewers** start at "Can the model move a number?" under *Check the evidence*;
**quant developers** go to *Packages*.

## Understand it

| Page | What it answers |
|---|---|
| [how-a-hypothesis-dies.md](how-a-hypothesis-dies.md) | How does the loop treat a hypothesis? Five demo hypotheses, four ways to die, test sealed until the gate. |
| [diagrams.md](diagrams.md) | What does each part look like? Eight numbered diagrams: the harness, the pre-registration timeline, v1 against v2 windows, the loop's states, the LLM transport, the replay parity check, the factor lab and imc-sim. |
| [architecture.md](architecture.md) | Which package owns what, what evidence each has, what is missing, and their distribution and import names. |
| [design-decisions.md](design-decisions.md) | What does each design decision buy and cost, and which seal applies where? |
| [engineering-case-study.md](engineering-case-study.md) | Which engineering decisions shaped the packages, and when should they be revisited? |
| [other-experiments.md](other-experiments.md) | What did the quote-imbalance and inventory-aware quoting experiments show? Both SYNTHETIC, both negative. |
| [quote-imbalance-walkthrough.md](quote-imbalance-walkthrough.md) | What does the optional `demo.sh --quotes` run do, step by step? |

## Check the evidence

| Page | What it answers |
|---|---|
| [results/real-2026-10/README.md](../results/real-2026-10/README.md) | What did protocol v1 find on REAL data, what was disclosed, and what carries the verdict? |
| [results/real-2026-10/REPORT.md](../results/real-2026-10/REPORT.md) | The v1 run's generated report, never rewritten. |
| [results/real-2026-10/protocol.json](../results/real-2026-10/protocol.json) | What was fixed before v1's first run? |
| [scripts/crosscheck_real.py](../scripts/crosscheck_real.py) | Do the reported numbers hold without the factor lab? A separate pandas recomputation of all 962. |
| [results/forward-2026-09/README.md](../results/forward-2026-09/README.md) | What does protocol v2 fix, which controls are frozen, and how much power does it have? |
| [results/forward-2026-09/protocol.json](../results/forward-2026-09/protocol.json) | What v2 registered, including its five amendments. |
| [results/forward-2026-09/DEVIATIONS.md](../results/forward-2026-09/DEVIATIONS.md) | Has anything departed from protocol v2? Dated entries; none changes a rule, a threshold or a frozen selection. |
| [results/forward-2026-09/model-evidence/README.md](../results/forward-2026-09/model-evidence/README.md) | Where do the LLM arm's model release dates, licence and listing come from? |
| [design-history.md](design-history.md) | In what order were the rules fixed, and why? Why the public history starts at publication, and the SHA-256 of every registered file. |
| [evidence/selection.json](evidence/selection.json) | The numbers behind the README figure (POST-HOC). |
| [evidence/replay-benchmark.json](evidence/replay-benchmark.json) | The replay benchmark's workload and the two digests every implementation reproduces. |
| Can the model move a number? [loop.py](../apps/quantos/src/quantos_showcase/loop.py) (lines 1–15), [codex_broker.py](../packages/research/src/qrae/codex_broker.py), [llm.py](../packages/research/src/qrae/llm.py), [validate_envelope.py](../tools/agent-review/validate_envelope.py) | What each agent may do; the critic's fixed schema and signed receipt; the replay cache bound to each prompt's SHA-256 and the pinned open-weight model; the validator that refuses an overclaiming run summary. |

## Packages

Each package's README first, then its own `docs/` pages. The package docs keep their original
UPPERCASE names because links and tests point at them.

| Page | What it answers |
|---|---|
| [apps/quantos/README.md](../apps/quantos/README.md) | The application: the loop, the ablation and the other workflows built on the packages. |
| [apps/quantos/docs/execution-workflow.md](../apps/quantos/docs/execution-workflow.md) | How to run and inspect the native quote strategy. |
| [apps/quantos/docs/persisted-execution.md](../apps/quantos/docs/persisted-execution.md) | How to pause, resume and inspect the same quote study. |
| [apps/quantos/docs/feed-workflow.md](../apps/quantos/docs/feed-workflow.md) | How an archived feed becomes reproducible research input. |
| [apps/quantos/docs/quote-workflow.md](../apps/quantos/docs/quote-workflow.md) | How quote features keep their event, received and available clocks. |
| [apps/quantos/docs/method-workflow.md](../apps/quantos/docs/method-workflow.md) | How a reading note becomes a testable, source-backed method. |
| [apps/quantos/docs/run-index.md](../apps/quantos/docs/run-index.md) | How one index replays each workflow's numbers with its owning package and links the reports. |
| [packages/factor/README.md](../packages/factor/README.md) | The factor lab: momentum against reversal from your own price CSV. |
| [packages/factor/docs/IMPORT_CONTRACT.md](../packages/factor/docs/IMPORT_CONTRACT.md) | What a price CSV and its contract must declare. |
| [packages/factor/docs/ARCHITECTURE.md](../packages/factor/docs/ARCHITECTURE.md) | Which module does what inside the factor lab. |
| [packages/factor/docs/INDEPENDENT_CALCULATIONS.md](../packages/factor/docs/INDEPENDENT_CALCULATIONS.md) | Hand-checkable arithmetic and timing cases for ranks, costs and clocks. |
| [packages/factor/docs/ALLOCATION.md](../packages/factor/docs/ALLOCATION.md) | How factor scores become holdings and trades. |
| [packages/factor/docs/MEASUREMENTS.md](../packages/factor/docs/MEASUREMENTS.md) | Historical v0.1.0 throughput on SYNTHETIC panels. |
| [packages/vault/README.md](../packages/vault/README.md) | Paper notes: which count as evidence, retrieval and method cards. |
| [packages/research/README.md](../packages/research/README.md) | `qrae`: work orders, the point-in-time catalog, the critic broker and the LLM transport. |
| [packages/research/docs/portfolio-architecture.md](../packages/research/docs/portfolio-architecture.md) | A walkthrough of the `qrae` source, its quant decisions and missing interfaces. |
| [packages/research/docs/ARCHITECTURE.md](../packages/research/docs/ARCHITECTURE.md) | Who owns which decision: data, validators or humans. |
| [packages/research/docs/LOCAL_KERNEL.md](../packages/research/docs/LOCAL_KERNEL.md) | The first executable slice: one baseline, one immutable run contract. |
| [packages/research/docs/HISTORICAL_DATA.md](../packages/research/docs/HISTORICAL_DATA.md) | The intake contract for historical price bars. |
| [packages/research/examples/historical_source/README.md](../packages/research/examples/historical_source/README.md) | A template for that intake manifest. |
| [packages/marketdata/native/README.md](../packages/marketdata/native/README.md) | The Python order-book replay module and its opt-in C++20 port: parity first, then timing (SYNTHETIC). |
| [packages/imc-sim/README.md](../packages/imc-sim/README.md) | The Market-Making Lab: IMC Prosperity 4 post-competition quote simulator and diagnostics. |
| [packages/imc-sim/CONTRACT.md](../packages/imc-sim/CONTRACT.md) | The normalized event contract the analyzer reads. |
| [packages/imc-sim/docs/QUOTE_POLICY.md](../packages/imc-sim/docs/QUOTE_POLICY.md) | The inventory-aware quote policy and its controlled execution study. |
| [packages/imc-sim/docs/QUOTE_STUDY_RESULTS.md](../packages/imc-sim/docs/QUOTE_STUDY_RESULTS.md) | What the frozen SYNTHETIC quote study shows. |
| [packages/imc-sim/docs/BOUNDARY_STATE.md](../packages/imc-sim/docs/BOUNDARY_STATE.md) | The complete-step state and in-memory restoration. |
| [packages/imc-sim/docs/PERSISTED_EXECUTION.md](../packages/imc-sim/docs/PERSISTED_EXECUTION.md) | The persisted SYNTHETIC execution workflow. |
| [packages/imc-sim/docs/PERSISTENCE_FILESYSTEM_REPAIR.md](../packages/imc-sim/docs/PERSISTENCE_FILESYSTEM_REPAIR.md) | How filesystem admission was repaired after two ownership failures. |
| [packages/imc-sim/docs/COMMUNITY_IMPORT.md](../packages/imc-sim/docs/COMMUNITY_IMPORT.md) | How to import one saved community-backtester log. |
| [packages/imc-sim/docs/UPSTREAM_FORMAT.md](../packages/imc-sim/docs/UPSTREAM_FORMAT.md) | Which upstream files were inspected, at which revision, and what was not reused. |
| [packages/imc-sim/examples/community-log/README.md](../packages/imc-sim/examples/community-log/README.md) | An original SYNTHETIC import case. |

## Reference

Hand-written example inputs, not research findings.

| Page | What it answers |
|---|---|
| [Jegadeesh–Titman 1993](../apps/quantos/examples/loop/vault/jegadeesh-titman-1993.md), [Jegadeesh 1990](../apps/quantos/examples/loop/vault/jegadeesh-1990.md), [Lehmann 1990](../apps/quantos/examples/loop/vault/lehmann-1990.md), [Moskowitz–Ooi–Pedersen 2012](../apps/quantos/examples/loop/vault/moskowitz-ooi-pedersen-2012.md), [Cont–Kukanov–Stoikov 2014](../apps/quantos/examples/loop/vault/cont-kukanov-stoikov-2014.md) | The demo vault's five paraphrase notes the proposer retrieves. |
| [derivation.md](../apps/quantos/examples/methods/derivation.md) | The quote method's original local derivation. |
| [source.md](../apps/quantos/examples/execution/source.md) | The execution study's reading note. |
| [evidence/demo.json](evidence/demo.json) | The quote-imbalance method trial's native result (SYNTHETIC), cited by [other-experiments.md](other-experiments.md). |
