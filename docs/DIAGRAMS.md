# Diagrams

Numbered index of every diagram in this repository. The README embeds 1 and 3; [system.md](system.md) embeds 3 to 6.
If a diagram and the code disagree, the code wins.

1. [The system and its components](#1-the-system-and-its-components)
2. [Inside the book build](#2-inside-the-book-build)
3. [How one claim moves](#3-how-one-claim-moves)
4. [A node's life in the gate graph](#4-a-nodes-life-in-the-gate-graph)
5. [One node, from goal to gate](#5-one-node-from-goal-to-gate)
6. [Where this build sits](#6-where-this-build-sits)

Colours: blue = stores and inputs, grey = processing, amber = checks, green = results, dashed = external or
private, purple = the path the diagram is about.

## 1. The system and its components

The book build in this repository at the centre, and the separate repositories around it: the paper layer of the
vault (not used in this build), the agent harness that later packaged the build's practices, the research harness,
and an earlier experiment that was not built from the book. The component table is in the
[README](../README.md#components-of-the-system).

```mermaid
flowchart TB
    OSCAR["Oscar<br/>goals and review"]:::ext
    GRAPH{"plan gate graph<br/>exit 0 = done"}:::gate
    AGENTS["AI coding agents<br/>Claude Code, subagents"]:::step
    subgraph VAULT["Vault: the second brain"]
        direction LR
        BOOK[("Book notes and<br/>measured results<br/>this repo")]:::key
        PAPERS[("Paper layer<br/>quant-research-vault")]:::data
    end
    LIB["pmlab library<br/>and tests, this repo"]:::key
    AH["agent-harness<br/>rules, memory,<br/>guard, work graph"]:::step
    ASOF["asof-research<br/>LLM proposes, code<br/>scores, human decides"]:::step
    PM["Polymarket-Crypto-5min<br/>July 2026, separate,<br/>not from the book"]:::ext
    OSCAR -->|"goals, constraints"| GRAPH
    OSCAR -.->|"earlier experiment"| PM
    GRAPH ==>|"briefs nodes"| AGENTS
    GRAPH -.->|"practices<br/>packaged later"| AH
    AGENTS ==>|"paraphrase, cite"| BOOK
    PAPERS -.->|"MCP search,<br/>not used here"| AGENTS
    BOOK ==>|"each claim<br/>names a test"| LIB
    AGENTS ==>|"code and tests"| LIB
    LIB ==>|"pytest exit code"| GRAPH
    AGENTS -.->|"also implemented"| ASOF
    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `docs/claims/`, `plan/`, `src/pmlab/afml/`, `tests/`; the other boxes are separate repositories.

## 2. Inside the book build

The three parts and the loop between them: the book goes into the vault as claims, the agents build from the claims under the harness, the gate decides when a node is done, and what the tests and studies measure is written back next to the claims. The paper layer was not queried in this build.

```mermaid
flowchart TB
    OSCAR["Oscar<br/>goals and review"]:::ext
    BOOK["AFML, 22 chapters<br/>private copy"]:::ext
    subgraph VAULT["Vault: the second brain"]
        CLAIMS[("Book layer<br/>1,805 claims")]:::key
        RESULTS[("Results layer<br/>measured notes")]:::data
        PAPERS[("Paper layer<br/>not queried here")]:::ext
    end
    subgraph HARNESS["Harness: plan gate graph"]
        ORCH["Claude Code<br/>orchestrator"]:::step
        GATE{"gate<br/>exit 0?"}:::gate
    end
    AGENTS["Subagents: builders,<br/>reviewers, auditors"]:::step
    LIB["pmlab library<br/>and tests"]:::out
    OSCAR -->|"goals, constraints"| ORCH
    BOOK -->|"read by section"| AGENTS
    AGENTS ==>|"paraphrase, cite"| CLAIMS
    CLAIMS ==>|"each names a test"| ORCH
    ORCH ==>|"briefs nodes"| AGENTS
    AGENTS ==>|"code and tests"| LIB
    LIB ==>|"pytest"| GATE
    GATE -->|"non-zero:<br/>back to pending"| ORCH
    GATE ==>|"exit 0:<br/>node done"| ORCH
    AGENTS -->|"what was measured"| RESULTS
    RESULTS -.->|"written back"| CLAIMS
    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `docs/claims/`, `plan/`, `src/pmlab/afml/`, `tests/`.

## 3. How one claim moves

A claim is paraphrased from a section, gets a status, and if done names a test; a second agent checks the test could fail; what was measured is written back into the claim's note.

```mermaid
flowchart TB
    SEC["Book section"]:::ext
    ROW["Claim row<br/>paraphrased"]:::data
    ST{"status"}:::gate
    TEST["Named test"]:::step
    AUD["Second reading:<br/>can it fail?"]:::gate
    NOTE["Note: what was<br/>measured here"]:::out
    SEC -->|"reader agent"| ROW
    ROW -->|"assigned"| ST
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

## 4. A node's life in the gate graph

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

## 5. One node, from goal to gate

Oscar sets the goal; the orchestrator delegates nodes to subagents (and writes some code itself), runs the gate, and sends selected diffs to a reviewer that did not write them.

```mermaid
sequenceDiagram
    participant O as Oscar
    participant C as Orchestrator
    participant P as plan graph
    participant B as Builder
    participant R as Reviewer
    O->>C: goal and<br/>constraints
    C->>P: nodes with gates
    C->>B: node, claims,<br/>files
    B->>B: code and<br/>its test
    B-->>C: reports done
    C->>P: plan gate N<br/>runs pytest
    P-->>C: exit code,<br/>evidence appended
    C->>R: diffs it<br/>did not write
    R-->>C: findings to fix
```

Where in the code: private briefs and transcripts; their output is `src/pmlab/afml/`, `tests/` and the gate runs in `plan/*.md`.

## 6. Where this build sits

What came before and after it among Oscar's repositories.

```mermaid
flowchart TB
    V["April 2026<br/>quant-research-vault<br/>the paper layer"]:::data
    P["July 2026<br/>Polymarket-Crypto-5min<br/>separate, not from the book"]:::ext
    B["17 to 21 Sep 2026<br/>this build: AFML to<br/>a tested library"]:::key
    H["Late Sep 2026<br/>agent-harness<br/>practices packaged"]:::step
    V -->|"then"| P
    P -->|"then"| B
    B -->|"then"| H
    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef ext fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: links in [system.md](system.md).
