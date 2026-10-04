# Docs index

Start at the [root README](../README.md): the question, the result, the figure and
the demo. Then pick the path that matches why you are here. Every Markdown file in
the repository outside `packages/*/docs` is listed on this page
([tests/test_docs.py](../tests/test_docs.py) checks that).

## Three reading paths

**1. The research question: can an LLM beat brute force under one honest gate?** (about an hour)
1. [how-a-hypothesis-dies.md](how-a-hypothesis-dies.md): the loop, run on the demo; five hypotheses, four ways to die, test sealed until the gate.
2. [results/real-2026-10/README.md](../results/real-2026-10/README.md): the v1 study on real data, its disclosures and what carries the verdict.
3. [protocol.json](../results/real-2026-10/protocol.json): what was fixed before the first run (registered 2026-10-03 18:10 UTC+08:00, six minutes before the run; SHA-256 `6bc1e31d934c657ede178f86b439f3b894351ce3fb8680f6d7c2e02cf903cb16`). Git order is not public: see the [publication history](design-history.md#publication-history).
4. [crosscheck_real.py](../scripts/crosscheck_real.py): the independent pandas recomputation of all 962 reported numbers.
5. [results/forward-2026-09/README.md](../results/forward-2026-09/README.md): protocol v2, its five amendments, its frozen controls and its stated power.
6. [design-history.md](design-history.md): the order in which the rules were fixed, and why; why this repository's history starts at publication, and the SHA-256 of every registered file.

**2. LLM-safety reviewer: can the model move a number?**
1. [loop.py](../apps/quantos/src/quantos_showcase/loop.py), lines 1–15: what each agent may and may not do.
2. [codex_broker.py](../packages/research/src/qrae/codex_broker.py): the bounded critic broker (named for its first backend; here it runs through the LLM transport). Fixed schema, no state transition, a signed receipt.
3. [llm.py](../packages/research/src/qrae/llm.py): the transport, a replay cache bound to each prompt's SHA-256, and the pinned live model: open weights on OpenRouter, no Anthropic or OpenAI model.
4. [validate_envelope.py](../tools/agent-review/validate_envelope.py): refuses a run summary that overclaims or asks for authority.

**3. Quant developer: the supporting packages** (separate from the study)
1. [packages/marketdata/native/README.md](../packages/marketdata/native/README.md): the Python replay module ([replay/book.py](../packages/marketdata/replay/book.py)) and its C++20 port, ~2.3× for a Python caller and 25.10× C++ to C++, with byte-identical parity on a SYNTHETIC workload ([results-2026-10-04.json](../packages/marketdata/native/bench/results-2026-10-04.json)).
2. [evidence/replay-benchmark.json](evidence/replay-benchmark.json): the benchmark workload's parameters and the two digests every implementation reproduces, with a note on the timing it held until 2026-10-04.
3. [engineering-case-study.md](engineering-case-study.md): the decisions above, and when to revisit them.

## Everything else

**Top-level docs**
- [architecture.md](architecture.md): every package, what it owns, its evidence and what is missing.
- [other-experiments.md](other-experiments.md): the quote-imbalance and inventory-aware quoting experiments, both SYNTHETIC, both negative.
- [quote-imbalance-walkthrough.md](quote-imbalance-walkthrough.md): the optional `demo.sh --quotes` run, step by step.

**Study records**
- [results/real-2026-10/REPORT.md](../results/real-2026-10/REPORT.md): the v1 run's generated report, never rewritten.
- [results/forward-2026-09/DEVIATIONS.md](../results/forward-2026-09/DEVIATIONS.md): dated departures from protocol v2; none changes a rule, a threshold or a frozen selection.

**Package references** (each package's own README, then its `docs/` folder)
- [packages/research](../packages/research/README.md): `qrae`, work orders, the point-in-time catalog, the broker and the LLM transport. Template: [historical source](../packages/research/examples/historical_source/README.md).
- [packages/factor](../packages/factor/README.md): the factor lab, point-in-time panels, rank IC and costs.
- [packages/vault](../packages/vault/README.md): note eligibility, retrieval and method cards.
- [packages/marketdata/native](../packages/marketdata/native/README.md): the Python replay module (`replay`) and the opt-in C++ replay path.
- [packages/imc-sim](../packages/imc-sim/README.md): IMC Prosperity 4 post-competition analysis; its [event contract](../packages/imc-sim/CONTRACT.md) and [import example](../packages/imc-sim/examples/community-log/README.md).
- [apps/quantos](../apps/quantos/README.md): the application. Its workflow guides: [execution](../apps/quantos/docs/execution-workflow.md), [persisted execution](../apps/quantos/docs/persisted-execution.md), [feed](../apps/quantos/docs/feed-workflow.md), [quotes](../apps/quantos/docs/quote-workflow.md), [methods](../apps/quantos/docs/method-workflow.md) and [run index](../apps/quantos/docs/run-index.md).

**Example inputs** (hand-written fixtures, not research findings)
- The demo vault's five paraphrase notes: [Jegadeesh–Titman 1993](../apps/quantos/examples/loop/vault/jegadeesh-titman-1993.md), [Jegadeesh 1990](../apps/quantos/examples/loop/vault/jegadeesh-1990.md), [Lehmann 1990](../apps/quantos/examples/loop/vault/lehmann-1990.md), [Moskowitz–Ooi–Pedersen 2012](../apps/quantos/examples/loop/vault/moskowitz-ooi-pedersen-2012.md), [Cont–Kukanov–Stoikov 2014](../apps/quantos/examples/loop/vault/cont-kukanov-stoikov-2014.md).
- The quote method's [derivation](../apps/quantos/examples/methods/derivation.md) and the execution study's [source note](../apps/quantos/examples/execution/source.md).
