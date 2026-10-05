# Design decisions: what each one buys and what it costs

The decisions that keep the LLM away from the numbers, each with its trade-off. The
[root README](../README.md) lists them in one line each; this page holds the full reasoning.
When each decision was made, and why, is in the [design history](design-history.md).

- **A critic that cannot promote.** A fixed schema and a signed receipt: it can only object, and a broken critic blocks. Trade-off: false negatives, never false positives.
- **Frozen method cards**, hashed with the data before evaluation. Trade-off: any tweak is a new card and another trial.
- **Test sealed until the gate, in code.** Pre-gate trials run without the test rows, so a card that stops earlier never gets a test number; `verify` refuses a pre-gate trial that saw them ([how this changed](design-history.md)). Trade-off: survivors run twice.
- **Which seal applies where.** The loop seals test until its gate; the race scores all 120 trials on test, but only after every arm's pick is frozen; and in v2 the seal is the frozen selection plus order: bars from 2026-09-01 to 2026-10-03 were public at registration but not fetched here, the frozen grid and random-K picks cannot use them, the rest of the window did not exist yet, and the LLM arm's proposal call is made before this repository fetches any forward bar ([protocol.json](../results/forward-2026-09/protocol.json), `splits.test`). This repository's first public push is the timestamp that matters: GitHub records it, independently of the author, so check it on GitHub; whatever follows it, such as the commit of the LLM replays made after that call, can be checked against it ([publication history](design-history.md#publication-history)). The frozen card holds the full prices, so the loop's seal guards the pipeline, not the operator.
- **Replay bound to the prompt hash.** Offline answers; a changed prompt fails loudly. Trade-off: every prompt change re-binds the cache.
- **Point-in-time contracts.** Every price carries `available_at`. Trade-off: declared, not audited.
- **Controls frozen before the LLM runs.** v2's grid and random-K picks are frozen in [selections.json](../results/forward-2026-09/selections.json), its SHA-256 computed at publication and equal to the file as registered in the private development history ([publication history](design-history.md#publication-history)). Trade-off: none can adapt later, by design.

The engineering decisions behind the packages (owned libraries, immutable artifacts, exact
arithmetic, as-of availability), with when to revisit each, are in the
[engineering case study](engineering-case-study.md#decisions-and-when-to-revisit-them).
