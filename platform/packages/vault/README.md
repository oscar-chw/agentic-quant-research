> Package reference. Start at the [root README](../../README.md); the [docs index](../../docs/README.md) lists every page.

# Research evidence core

The research loop's paper notes and method cards: it decides which notes count as evidence, retrieves them offline for the proposer, and validates the method cards that fix a rule before it is tested. Installable, with SQLite metadata; the QuantOS application in `apps/quantos` consumes it directly.

- `source_access.py` decides whether a note is eligible evidence (digest-matched, current, inside the source root) and keeps backend errors distinct from empty results.
- `note_index.py` is a standard-library term-overlap backend for `query_note_sources`; it reads `id:`/`title:`-headed Markdown notes. Retrieval quality is not evaluated.
- `method_contract.py` validates method cards: `quote-weighted-midpoint/v1` (a fixed exact-rational forecast comparison, see [the workflow](../../apps/quantos/docs/method-workflow.md)) and `factor-rank-ic/v1` (signal, lookback, cost, splits and the rank-IC and net-mean bars, used by the research loop). A card never executes code and does not establish statistical significance.

An optional MCP adapter (`search_mcp.py`) is preserved for stubbed boundary tests only; its external stacks (chromadb, mcp, yaml) are not installed or qualified here. It returns vault context to its caller and calls no model.

## Shared method-note consumer

`shared_manifest.load_shared_note(root, document_id, expected_manifest_sha256=...)`
reads one explicitly selected `quant-shared-notes/v1` entry. It checks the
pinned manifest digest, per-document digest, root-contained regular path, and
the manifest's `MIT` / `original_summary` / `public` rights tuple. It returns
`MATCHES` with a complete envelope (`document_id`, `document_type`, `title`,
`document`, `source_sha256`, `manifest_sha256`, `source_refs`, `rights`) or
`NO_MATCHES`; invalid, stale, escaped, and refused evidence raises a coded
`SharedManifestError`. Method notes remain `method_note`, never paper notes.

`method_contract.shared_note_citation` adapts that envelope to the
`vault-method-citation/v1` citation fields while preserving method-note type and both
digests. Existing paper-note selection and cards continue using their current
interfaces.
