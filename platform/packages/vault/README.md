> Package reference. Start at the [root README](../../README.md); the [docs index](../../docs/README.md) lists every page.

# Research evidence core

The research loop's paper notes and method cards: it decides which notes count as evidence, retrieves them offline for the proposer, and validates the method cards that fix a rule before it is tested. Installable, with SQLite metadata; the QuantOS application in `apps/quantos` consumes it directly.

- `source_access.py` decides whether a note is eligible evidence (digest-matched, current, inside the source root) and keeps backend errors distinct from empty results.
- `note_index.py` is a standard-library term-overlap backend for `query_note_sources`; it reads `id:`/`title:`-headed Markdown notes. Retrieval quality is not evaluated.
- `method_contract.py` validates method cards: `quote-weighted-midpoint/v1` (a fixed exact-rational forecast comparison, see [the workflow](../../apps/quantos/docs/method-workflow.md)) and `factor-rank-ic/v1` (signal, lookback, cost, splits and the rank-IC and net-mean bars, used by the research loop). A card never executes code and does not establish statistical significance.

An optional MCP adapter (`search_mcp.py`) is preserved for stubbed boundary tests only; its external stacks (chromadb, mcp, yaml) are not installed or qualified here. It returns vault context to its caller and calls no model.
