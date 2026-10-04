# Architecture: packages, the research loop and what each part owns

The repository is one Python monorepo. Each package owns one kind of
calculation; the `apps/quantos` application composes them and copies no
numerical engine. LLMs sit at the edges of the research loop: they draft
hypotheses and critiques, and deterministic code computes every number that
decides what survives.

```mermaid
flowchart TB
  subgraph Agents[LLM agents: draft only]
    P[Proposer: K hypotheses, each citing a vault note]
    C[Adversarial critic: may object, cannot set metrics]
  end
  subgraph Loop[apps/quantos: research loop]
    F[Freeze: method card fixes signal, splits, cost, rule]
    T[Test: factor lab prepare, run, verify --recompute]
    G[Human gate: promote or reject]
  end
  V[packages/vault: notes, eligibility, method cards] --> P
  P --> F --> T --> C --> G
  D[packages/factor: point-in-time panels, rank IC, cost sleeves] --> T
  O[Daily OHLCV directory or synthetic fixture] --> D
  B[packages/research qrae: bounded broker, HMAC receipts, replay cache] -. transport .-> P
  B -. transport .-> C
  M[packages/marketdata: coverage-aware book replay] --> Q[apps/quantos: quote feature and method trials]
  B --> Q
  I[packages/imc-sim: inventory-aware quoting, fills, persistence] --> X[apps/quantos: execution and run index]
  D --> X
  Q --> X
```

Solid arrows are implemented calls. Dotted arrows mark the LLM transport: a
replay cache by default, or the local `claude -p` when `--live` is passed.

| Package | Owns | Current evidence | Missing |
|---|---|---|---|
| `packages/research` (qrae) | Work orders, point-in-time catalog, chronological price baseline, the bounded critic broker (`codex_broker.py`, named for its first backend; here it runs `claude -p`), the LLM transport (`llm.py`) | Unit and tamper tests; the broker's schema forbids state transitions and signs results with an HMAC receipt | No recorded live model run |
| `packages/vault` | Note eligibility (`source_access.py`), offline lexical retrieval (`note_index.py`), method cards for quote and factor methods (`method_contract.py`) | Digest-checked notes; cards that fix rules before evaluation | Retrieval quality is not evaluated; the demo vault is five hand-written notes |
| `packages/factor` | CSV import under declared clocks, momentum and reversal features, rank IC, cost-adjusted interval sleeves, OHLCV directory import | Reference checks, missing-data and tamper tests, synthetic positive and regime-switch fixtures; 120 recomputed trials on 34 Binance daily pairs ([results/real-2026-10](../results/real-2026-10/REPORT.md)), matched by an independent pandas recomputation | A survivorship-free universe; a second test period |
| `packages/marketdata` | Point-in-time book replay (`replay/`: scalar and batch reconstruction, a message feed, archived-feed admission); an opt-in C++20 replay port with a pybind11 module (`native/`) | Tests on 246 recorded golden cases and generated histories; a 41.66x batch-over-scalar replay measurement on 100,000 synthetic messages; byte-identical C++ output and Python-side parity tests for the module ([results-2026-10-04.json](../packages/marketdata/native/bench/results-2026-10-04.json)) | Live capture, a storage layer and representative workloads; a Python caller gets ~2.3x, not 25.10x, because the Python side of the call (by subtraction) dominates ([native/README.md](../packages/marketdata/native/README.md)) |
| `packages/imc-sim` | Post-competition IMC Prosperity 4 analysis: inventory-aware quoting, matching, accounting, persisted runs | Synthetic quote studies and recovery tests | Official engine parity; observed inputs |
| `apps/quantos` | The research loop, the LLM-vs-grid-vs-random ablation (`ablation.py`), quote method trials, execution wrapper and run index | End-to-end demo and verify commands; the pre-registered real-data ablation's grid and random arms | The ablation's LLM arm: not run yet (protocol v2 scores it on 2026-09-01 → 2027-08-31) |

## Design rules

- **Freeze before evaluating.** A method card and its data are hashed and
  committed before the trial runs; `quantos-loop verify` recomputes every trial
  from the frozen card and refuses any drift.
- **The LLM never sees prices.** The proposer sees the campaign question, the
  split dates and retrieved notes. The same drafts therefore face both demo
  panels, and only the deterministic test separates them.
- **Agents have no write path to numbers.** The critic answers through the
  broker's fixed schema (`summary`, `claims`, `artifact_refs`,
  `requested_transition` = `NONE`); a response with any other field is
  quarantined, and the loop fails closed.
- **One owner per calculation.** Applications compose packages; adapters keep
  each package's units and failure states instead of a combined score.
- **Measure before adding machinery.** Processes, columnar engines or native
  kernels belong where a representative workload shows the need and a
  reference result verifies equivalence.
