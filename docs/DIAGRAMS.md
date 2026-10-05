# asof-research in diagrams

The visual companion to the [root README](../README.md). Every box is a module, function, file,
command or registered step in this repository, and every number is one the README or the cited
study files already state. If a diagram and the code disagree, the code wins.

1. System overview: the harness
2. Pre-registration timeline: v1, v2 and its amendments
3. Split windows: v1 against v2
4. A hypothesis in the research loop
5. The LLM transport: replay first, then the pinned live model
6. Order-book replay: the C++ port against the Python path
7. The factor lab package
8. The imc-sim package

Colours: blue = input or store, grey = processing, amber = a check that can refuse,
green = result, dashed = optional, external or pending, purple = the path the study is about.
All times are the author's local time, UTC+08:00, as recorded in [design-history.md](design-history.md).

## 1. System overview: the harness

Prices enter only through a checksummed, point-in-time contract; hypotheses come from the LLM's
frozen method cards or from the enumerate-all grid; every hypothesis is scored by the same factor
lab; each arm selects on validation only, and its pick gets one look at test through a gate fixed
before any run, after which a human or the pre-registered rule decides.

```mermaid
flowchart TB
    subgraph PIT["Point-in-time data contract"]
        UNI[("binance_universe.json<br/>SHA-256 per file")]
        CSV[("Binance daily CSVs<br/>34 USDT pairs")]
        IMPORT["from-ohlcv<br/>each price has<br/>available_at"]
    end
    PROTO[("protocol.json<br/>registered<br/>before any score")]
    SEL[("v2: selections.json<br/>frozen picks")]
    subgraph HYP["Hypotheses"]
        VAULT[("packages/vault<br/>notes, method cards")]
        LLM["LLM proposer, pending<br/>K = 8, never sees prices"]
        CHECK{"loop.check_proposals"}
        CARD["loop.freeze<br/>card hashed<br/>with the data"]
        GRID["ablation.grid<br/>momentum or reversal<br/>x lookback 1-60"]
    end
    LAB["factor lab<br/>prepare, run,<br/>verify --recompute"]
    subgraph ARMS["Arms: select on validation only"]
        ALLM["LLM arm"]
        AGRID["grid arm<br/>all 120"]
        ARND["random-K arm<br/>8 drawn"]
    end
    GATE{"ablation.gate_passes<br/>test net > 0 and<br/>above best baseline"}
    HUMAN["loop.decide<br/>human or<br/>pre-registered rule"]
    RES[("ablation.json, REPORT.md<br/>never overwritten")]
    XC["crosscheck_real.py<br/>pandas, no factor lab"]

    UNI -->|"verify-data refuses<br/>a changed file"| CSV
    CSV ==>|"daily bars"| IMPORT
    PROTO -->|"universe,<br/>experiment"| IMPORT
    VAULT -->|"retrieved notes"| LLM
    LLM -.->|"draft JSON"| CHECK
    CHECK -->|"schema, citation,<br/>no duplicate"| CARD
    CARD -->|"frozen cards"| LAB
    GRID ==>|"120 hypotheses"| LAB
    IMPORT ==>|"prices.csv,<br/>contract"| LAB
    LAB -.->|"cards, checked<br/>against grid trial"| ALLM
    LAB ==>|"scored trials"| AGRID
    LAB -->|"same trials"| ARND
    SEL -.->|"v2 refuses unless<br/>reproduced"| AGRID
    ALLM -.->|"its pick"| GATE
    AGRID ==>|"pick: v1 by IC,<br/>v2 by net"| GATE
    ARND -->|"its pick"| GATE
    GATE ==>|"promoted or not"| RES
    GATE -->|"card awaiting<br/>the human"| HUMAN
    HUMAN -->|"bound to ledger<br/>SHA-256"| RES
    XC -.->|"recomputes from<br/>raw CSVs: 962<br/>numbers match"| RES

    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
    class CSV,UNI,PROTO,SEL,VAULT data
    class CARD,ARND,HUMAN,XC step
    class CHECK,GATE gate
    class RES out
    class LLM,ALLM ext
    class IMPORT,GRID,LAB,AGRID key
```

Where in the code: `scripts/fetch_binance_daily.py`, `packages/factor/src/factor_research/{ohlcv,prepare,runs}.py`,
`apps/quantos/src/quantos_showcase/ablation.py` (`verify_data`, `grid`, `select`, `random_arm`, `llm_arm`,
`check_selections`, `gate_passes`, `run`), `apps/quantos/src/quantos_showcase/loop.py` (`check_proposals`,
`freeze`, `decide`), `packages/vault/method_contract.py`, `scripts/crosscheck_real.py`, `scripts/real_run.sh`.

## 2. Pre-registration timeline: v1, v2 and its amendments

The order in which the rules were fixed: v1 was registered six minutes before its run; v2 was
registered after v1's test window turned out to be used, and every amendment came before any LLM
call, any v2 score and any bar after 2026-08-31. The times are the author's record, evidenced by
each registered file's SHA-256 in the [publication history](design-history.md#publication-history),
not by git order. What is left runs in the order the protocol fixes.

```mermaid
flowchart TB
    subgraph V1["Protocol v1, 2026-10-03"]
        U["18:09:21<br/>universe checksums<br/>binance_universe.json"]
        P1["18:10:31<br/>v1 registered: protocol,<br/>campaign, experiment"]
        R1["18:16:43<br/>v1 run: ablation.json,<br/>hypotheses.json, REPORT.md"]
        PH["18:44<br/>posthoc.json<br/>labelled POST-HOC"]
    end
    subgraph V2["Protocol v2"]
        P2["18:52:14<br/>v2 registered: one gate for<br/>every arm, forward test window"]
        A1["19:13:42 amendment 1<br/>select on validation net,<br/>grid and random-K frozen"]
        A2["19:28:07 amendment 2<br/>Newey-West verdict,<br/>power stated"]
        A3["19:50:08 amendment 3<br/>delisting run on<br/>the unfilled data"]
        A4["20:13:14 amendment 4<br/>wording only"]
        PUB["2026-10-04<br/>fresh public history"]
        A5["2026-10-05 amendment 5<br/>open weights Qwen3.8-27B<br/>replace claude-opus-5-5,<br/>no Anthropic or OpenAI model"]
    end
    subgraph PEND["Pending, in protocol order"]
        S1["scripts/forward_propose.sh<br/>LLM proposals, 1 call,<br/>no prices read"]
        S2["commit the REAL LLM<br/>OUTPUT replays"]
        S3["fetch bars to 2027-08-31<br/>in 2027-09"]
        S4["real_run.sh --live<br/>score every arm once"]
    end

    U -->|"one minute later"| P1
    P1 ==>|"six minutes later,<br/>before any number"| R1
    R1 -->|"a review questioned v1"| PH
    R1 ==>|"test window already used<br/>by alpha-gp-lab, unequal gates"| P2
    P2 ==>|"before any LLM call"| A1
    A1 -->|"verdict on the<br/>selection metric"| A2
    A2 -->|"drop rule could not<br/>reproduce the picks"| A3
    A3 -->|"no rule changed"| A4
    A4 -->|"next day"| PUB
    PUB -->|"next day, before<br/>any LLM call"| A5
    A5 -.->|"before any<br/>forward bar is fetched"| S1
    S1 -.->|"replay files"| S2
    S2 -.->|"then"| S3
    S3 -.->|"after the last test bar"| S4

    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
    class U data
    class PH,A2,A3,A4,A5 step
    class A1 gate
    class R1,PUB out
    class S1,S2,S3,S4 ext
    class P1,P2 key
```

Where in the code: [design-history.md](design-history.md) (the dated decisions and the SHA-256 table,
checked by `tests/test_docs.py`), `results/real-2026-10/protocol.json`, `results/forward-2026-09/protocol.json`
(`amendments`), [results/forward-2026-09/README.md](../results/forward-2026-09/README.md) (the pending steps),
`scripts/forward_propose.sh`, `scripts/real_run.sh`.

## 3. Split windows: v1 against v2

The windows each protocol registered. v1's test window, already evaluated by the sibling
repository alpha-gp-lab, becomes part of v2's validation window, and v2's test is a forward year
that did not exist when it was registered.

```mermaid
gantt
    title Split windows registered by each protocol
    dateFormat YYYY-MM-DD
    axisFormat %Y
    section v1, registered 2026-10-03
    train, warm-up and lookbacks only     :done, 2020-01-01, 2023-12-31
    validation, selection                 :active, 2024-01-01, 2024-12-31
    test, one look                        :crit, 2025-01-01, 2026-08-31
    section alpha-gp-lab, earlier
    test window it had already scored     :crit, 2025-01-01, 2026-08-31
    section v2, registered 2026-10-03
    train, warm-up and lookbacks only     :done, 2020-01-01, 2023-12-31
    validation, selection                 :active, 2024-01-01, 2026-08-31
    test, scored once after 2027-08-31    :crit, 2026-09-01, 2027-08-31
```

Where in the code: `results/real-2026-10/protocol.json` and `results/forward-2026-09/protocol.json`
(`splits`, `used_windows`), `results/*/experiment.json` (`splits`).

## 4. A hypothesis in the research loop

Every status a proposed hypothesis can end in. Pre-gate trials run on the frozen prices without
the test rows, so a test number exists only for a card that reaches the human gate; the critic
may only object, and a broken critic blocks.

```mermaid
stateDiagram-v2
    [*] --> INVALID: check_proposals refuses<br/>SCHEMA, CITATION_NOT_RETRIEVED,<br/>DUPLICATE, ...
    [*] --> PROPOSED: valid draft,<br/>frozen as a method card
    PROPOSED --> FAILED_ON_VALIDATION: sealed trial fails<br/>the card rule, or is refused
    PROPOSED --> TESTED: sealed trial supports<br/>the rule on validation
    TESTED --> REJECTED_BY_CRITIC: one REJECTED claim
    TESTED --> CRITIC_UNUSABLE: broker result<br/>not DRAFT_READY
    TESTED --> AWAITING_HUMAN: no objection,<br/>open_test computes test
    AWAITING_HUMAN --> PROMOTED: decide
    AWAITING_HUMAN --> REJECTED_BY_HUMAN: decide
```

Where in the code: `apps/quantos/src/quantos_showcase/loop.py` (`check_proposals`, `freeze`, `sealed_prices`,
`test_card`, `passes_validation`, `critique`, `critic_verdict`, `VERDICT_STATUS`, `open_test`, `decide`, `verify`);
walked through on the demo in [how-a-hypothesis-dies.md](how-a-hypothesis-dies.md).

## 5. The LLM transport: replay first, then the pinned live model

How the loop gets an LLM answer. Recorded keys are answered from the replay file and bound to
the prompt's SHA-256; only unrecorded keys reach the pinned open-weight model, within a hard call
budget, and any doubtful answer is refused and never recorded. No model has run the loop yet:
every response in the repository is hand-written.

```mermaid
sequenceDiagram
    participant L as loop.py
    participant T as ReplayThenLive
    participant R as ReplayProvider
    participant O as OpenRouterProvider
    participant M as OpenRouter
    Note over L,T: provider_for: --replay,<br/>--live, or both<br/>(replayed keys first)
    Note over L,T: --live, no OPENROUTER_API_KEY:<br/>MissingApiKey, exit 4, no call
    L->>T: complete(key,<br/>prompt)
    alt key recorded in the replay
        T->>R: complete
        R-->>T: recorded<br/>response
        Note over R: prompt SHA-256<br/>changed: ReplayMiss,<br/>never a silent<br/>live call
    else key not recorded
        T->>O: complete
        Note over O: at --max-calls:<br/>refused before<br/>the call
        O->>M: POST, pinned<br/>qwen3.8-27b:free,<br/>temperature 0
        M-->>O: JSON body
        alt non-200, error field,<br/>finish_reason not stop,<br/>no content, over 256 KiB,<br/>another model
            O--xL: RuntimeError: refused,<br/>not recorded, exit 2
        else accepted
            O-->>T: text, model,<br/>provider, id
        end
    end
    T-->>L: response<br/>text
    L->>L: check_proposals,<br/>or the critic's<br/>fixed schema
    Note over L,T: --record writes a new<br/>replay file, never<br/>overwrites one
    Note over L,T: real_run.sh --live: a<br/>failed session is not<br/>committed, arm pending
```

Where in the code: `packages/research/src/qrae/llm.py` (`ReplayProvider`, `OpenRouterProvider`, `ReplayThenLive`,
`broker_runner`), `apps/quantos/src/quantos_showcase/loop.py` (`provider_for`, `main`), `scripts/forward_propose.sh`,
`scripts/real_run.sh`; the request settings are pre-registered in `results/forward-2026-09/protocol.json` (`arms.llm`).

## 6. Order-book replay: the C++ port against the Python path

A supporting package, not part of the study. The Python module is the reference; the opt-in
C++20 port must write the same bytes on the SYNTHETIC workload before any timing counts, and
from Python it is reached only on request. The edge that differs between the two paths is the
query-time type: the port refuses fractional and NaN times, so pure Python stays the default.

```mermaid
flowchart TB
    EVID[("evidence/replay-benchmark.json<br/>pinned digests")]
    GEN["export_workload.py<br/>SYNTHETIC, 100,000<br/>messages, 64 queries"]
    WL[("build/workload.txt<br/>sha256 67d36ed8…aebc")]
    subgraph PY["Python, the reference"]
        PYB["replay.book<br/>reconstruct_many"]
        EXP[("build/expected.json<br/>2,782 bytes<br/>sha256 279cc869…3849")]
    end
    subgraph CPP["C++20 port, opt-in"]
        CLI["tools/replay_cli.cpp<br/>batch and --scalar"]
        OUTC[("build/cpp-batch.json<br/>build/cpp-scalar.json")]
    end
    GOLD[("tests/golden/<br/>246 recorded cases")]
    PAR{"python/parity.py<br/>byte for byte,<br/>no tolerance"}

    EVID -->|"pins both digests"| GEN
    GEN ==>|"runs the reference"| PYB
    GEN -->|"writes"| WL
    PYB ==>|"reference answer"| EXP
    WL -->|"same input"| CLI
    CLI -->|"writes"| OUTC
    PYB -->|"make_golden.py<br/>re-records"| GOLD
    EXP ==>|"reference bytes"| PAR
    OUTC -->|"C++ bytes"| PAR
    GOLD -->|"must equal a<br/>fresh recording"| PAR

    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
    class EVID,WL,GOLD data
    class GEN,CLI step
    class PAR gate
    class OUTC out
    class PYB,EXP key
```

From Python, the port is reached only on request; `tests/test_binding.py` holds both paths to the same answers.

```mermaid
flowchart TB
    CALL["replay.book<br/>reconstruct_many"]
    PYB["pure-Python sweep<br/>the default"]
    EXT["_native_many:<br/>asof_replay_native<br/>bindings/py_replay.cpp"]
    TB{"tests/test_binding.py<br/>books and refusals<br/>must match"}

    CALL ==>|"native left unset"| PYB
    CALL -.->|"native=True or<br/>ASOF_REPLAY_NATIVE=1"| EXT
    EXT -->|"books or NotCovered;<br/>int64 ms only: float<br/>or NaN raises"| TB
    PYB ==>|"pure-Python answers"| TB

    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
    class CALL step
    class TB gate
    class EXT ext
    class PYB key
```

Where in the code: `packages/marketdata/replay/book.py`, `packages/marketdata/native/` (`python/parity.py`,
`python/export_workload.py`, `python/make_golden.py`, `tools/replay_cli.cpp`, `bindings/py_replay.cpp`,
`tests/test_binding.py`, `Makefile` targets `parity` and `pytest`); details and timings in
[native/README.md](../packages/marketdata/native/README.md).

## 7. The factor lab package

`packages/factor`, which computes every number the study uses: prices are refused unless their
clocks and mapping are declared, every run is registered as a hashed trial before it is parsed,
and `verify --recompute` rebuilds the numbers with the original code.

```mermaid
flowchart TB
    OHLCV[("directory of daily<br/>OHLCV CSVs")]
    FROM["factor-research<br/>from-ohlcv"]
    CSV[("long-form price CSV<br/>one row per asset, time")]
    CON[("import contract JSON<br/>columns, clocks, universe,<br/>calendar, splits, cost")]
    PREP{"factor-research prepare<br/>refuses unmapped columns,<br/>duplicates, or broken<br/>time ≤ observed_at ≤ available_at"}
    BUNDLE[("prepared bundle<br/>raw CSV, contract,<br/>panel, hash manifest")]
    RUN["factor-research run<br/>momentum P(t-1)/P(t-1-L) - 1,<br/>reversal its negative"]
    TRIAL[("trial directory<br/>inputs, code hashes,<br/>report.json, manifest")]
    VER{"verify --recompute<br/>needs the original<br/>code hash"}
    OUT["report.md<br/>rank IC, coverage, costs"]

    OHLCV -->|"daily bars"| FROM
    FROM -->|"writes"| CSV
    FROM -->|"writes"| CON
    CSV ==>|"raw rows"| PREP
    CON ==>|"declared mapping"| PREP
    PREP ==>|"availability label:<br/>ASSUMED_DELAY or<br/>PROVIDED_CLOCKS_UNAUDITED"| BUNDLE
    BUNDLE ==>|"run --prepared"| RUN
    RUN ==>|"registered before<br/>parsing, failures kept"| TRIAL
    TRIAL ==>|"artifact hashes"| VER
    VER -->|"recreated report<br/>matches"| OUT

    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
    class OHLCV,CSV,CON data
    class FROM step
    class PREP,VER gate
    class OUT out
    class BUNDLE,RUN,TRIAL key
```

Where in the code: `packages/factor/src/factor_research/` (`ohlcv.py`, `prepare.py`, `evaluator.py`, `runs.py`,
`cli.py`); the study calls `prepare_prices`, `run_trial` and `verify_run` from `loop.py` and `ablation.py`.
Contract and refusal cases: [packages/factor/README.md](../packages/factor/README.md).

## 8. The imc-sim package

`packages/imc-sim`, post-competition tooling for IMC Prosperity 4 and separate from the study:
a saved community-backtester log or a SYNTHETIC quote study becomes normalized events, which one
analyzer turns into cash, inventory and markout diagnostics. It contains none of the team's
competition code, and every example is SYNTHETIC.

```mermaid
flowchart TB
    LOG[("saved community-<br/>backtester log")]
    ICFG[("import.json<br/>format, identity, fees,<br/>opening state")]
    SCEN[("quoting_scenarios.json<br/>SYNTHETIC")]
    IMP{"community_import<br/>one pinned format,<br/>multi-day refused"}
    QS["quote_study.run_study<br/>inventory-aware quotes,<br/>simulated fills"]
    JSONL[("normalized events<br/>imc4-analysis/v1 JSONL")]
    VAL{"contracts.py<br/>strict event order,<br/>unique fill ids,<br/>decimal money"}
    AN["analyzer.analyze<br/>cash, positions, marks,<br/>markouts after causal pass"]
    REP["report.render<br/>HTML and SVG,<br/>no accounting in the UI"]
    OUT[("new output directory<br/>report.html, report.json,<br/>complete.json last")]

    LOG -->|"raw log, kept"| IMP
    ICFG -->|"declared mapping"| IMP
    IMP ==>|"own fills and<br/>market trades apart"| JSONL
    SCEN -->|"queue and delay<br/>assumptions"| QS
    QS -->|"orders and fills"| JSONL
    JSONL ==>|"one record per line"| VAL
    VAL ==>|"admitted events"| AN
    AN ==>|"exact totals,<br/>sampled snapshots"| REP
    REP ==>|"existing directory<br/>refused"| OUT

    classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
    classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
    classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
    classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
    classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
    classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
    class LOG,ICFG,SCEN data
    class QS,REP step
    class IMP,VAL gate
    class OUT out
    class JSONL,AN key
```

Where in the code: `packages/imc-sim/src/imc4_analysis/` (`community_import.py`, `quote_study.py`, `quoting.py`,
`contracts.py`, `analyzer.py`, `report.py`, `cli.py`); details in [packages/imc-sim/README.md](../packages/imc-sim/README.md).
