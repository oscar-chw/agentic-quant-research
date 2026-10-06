# Diagrams

Numbered index of every diagram in this repository. The README embeds 1 and 2; [system.md](system.md) embeds 2 to 5.
If a diagram and the code disagree, the code wins.

1. [System overview](#1-system-overview)
2. [How one claim moves](#2-how-one-claim-moves)
3. [A node's life in the gate graph](#3-a-nodes-life-in-the-gate-graph)
4. [One node, from goal to gate](#4-one-node-from-goal-to-gate)
5. [Where this build sits](#5-where-this-build-sits)

Colours: blue = stores and inputs, grey = processing, amber = checks, green = results, dashed = external or
private, purple = the path the diagram is about.

## 1. System overview

The three parts and the loop between them: the book goes into the vault as claims, the agents build from the claims under the harness, the gate decides when a node is done, and what the tests and studies measure is written back next to the claims. The paper layer was not queried in this build.

```mermaid
flowchart TB
    OSCAR["Oscar<br/>goals and review"]:::ext
    BOOK["AFML, 22 chapters<br/>private copy"]:::ext
    subgraph VAULT["Vault: the second brain"]
        direction LR
        CLAIMS[("Book layer<br/>1,805 claims")]:::key
        PAPERS[("Paper layer<br/>quant-research-vault")]:::ext
        RESULTS[("Results layer<br/>measured notes")]:::data
    end
    subgraph HARNESS["Harness: plan gate graph"]
        direction LR
        ORCH["Claude Code<br/>orchestrator"]:::step
        GATE{"gate<br/>exit 0?"}:::gate
    end
    AGENTS["Subagents: builders,<br/>reviewers, auditors"]:::step
    LIB["pmlab library<br/>and tests"]:::out
    OSCAR -->|"goals, constraints"| ORCH
    BOOK -->|"read section by section"| AGENTS
    AGENTS ==>|"paraphrase, cite section"| CLAIMS
    CLAIMS ==>|"each claim names a test"| ORCH
    ORCH ==>|"briefs nodes"| AGENTS
    AGENTS ==>|"code and tests"| LIB
    LIB ==>|"pytest"| GATE
    GATE -->|"non-zero: back to pending"| ORCH
    GATE ==>|"exit 0: node done"| ORCH
    AGENTS -->|"record what was measured"| RESULTS
    RESULTS -.->|"written back"| CLAIMS
    PAPERS -.->|"not queried in this build"| ORCH
    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `docs/claims/`, `plan/`, `src/pmlab/afml/`, `tests/`.

## 2. How one claim moves

A claim is paraphrased from a section, gets a status, and if done names a test; a second agent checks the test could fail; what was measured is written back into the claim's note.

```mermaid
flowchart LR
    SEC["Book section"]:::ext
    ROW["Claim row<br/>paraphrased"]:::data
    ST{"status"}:::gate
    TEST["Named test"]:::step
    AUD["Second reading:<br/>can it fail?"]:::gate
    NOTE["Note: what was<br/>measured here"]:::out
    SEC -->|"reader agent"| ROW
    ROW --> ST
    ST -->|"done"| TEST
    ST -->|"deferred or n/a:<br/>reason"| NOTE
    TEST -->|"other agent audits"| AUD
    AUD -->|"cannot fail: fix"| TEST
    AUD -->|"sound"| NOTE
    NOTE -.->|"written back"| ROW
    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
```

Where in the code: `docs/claims/chNN.md`, `docs/claims/audit/chNN.md`, `scripts/claims.py`.

## 3. A node's life in the gate graph

Exit 0 is the only way to done; a failed gate appends evidence and sends the node back to pending, so there is no failed state.

```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Leased: a session takes it
    Leased --> Pending: released unfinished
    Leased --> GateRun: plan gate N
    GateRun --> Done: exit 0
    GateRun --> Pending: exit non-zero, evidence appended
    Done --> [*]
```

Where in the code: `plan/*.md`, `scripts/plan_stats.py`, `docs/harness-rules.md`.

## 4. One node, from goal to gate

Oscar sets the goal; the orchestrator delegates nodes to subagents (and writes some code itself), runs the gate, and sends selected diffs to a reviewer that did not write them.

```mermaid
sequenceDiagram
    participant O as Oscar
    participant C as Orchestrator
    participant P as plan graph
    participant B as Builder subagent
    participant R as Reviewer subagent
    O->>C: goal and constraints
    C->>P: nodes with gates
    C->>B: a node, its claims, its files
    B->>B: code and its test
    B-->>C: reports done
    C->>P: plan gate N runs pytest
    P-->>C: exit code, evidence appended
    C->>R: selected diffs it did not write
    R-->>C: findings to fix
```

Where in the code: private briefs and transcripts; their output is `src/pmlab/afml/`, `tests/` and the gate runs in `plan/*.md`.

## 5. Where this build sits

What came before and after it among Oscar's repositories.

```mermaid
timeline
    title What came before and after
    April 2026 : quant-research-vault, the paper layer
    July 2026 : Polymarket-Crypto-5min, an earlier separate experiment, not built from the book
    17 to 21 Sep 2026 : this build, AFML to a tested library under the gated harness
    Late Sep 2026 : agent-harness, the harness practices packaged
```

Where in the code: links in [system.md](system.md).
