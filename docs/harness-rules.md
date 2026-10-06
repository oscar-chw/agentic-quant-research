# The rules the build ran under

The build repository's `CLAUDE.md`, which Claude Code loads into every session there, copied below the line with its
heading levels shifted and its title annotated. The
`plan` tool it names is not in this repository; [agent-harness](https://github.com/oscar-chw/agent-harness) is its later,
packaged form. The sanitized graph the build left behind is in [`plan/`](../plan/).

---

## polymarket research (the build repository)

Learn AFML information-driven bars by applying all nine (tick/volume/dollar x standard/imbalance/run) to Polymarket BTC 5-minute markets, find the fair price and candidate signals (GP + Bayesian optimisation), watch it live, then test hypotheses.

### How work runs here

Plan state lives in `plan/<slug>.md`, git-tracked. Marks are `[ ]` pending,
`[>]` running (leased), `[x]` done. **There is no failed state** — a failed gate
appends evidence and returns the node to pending, which is the debug loop.

    plan ready <slug>     what can start now, or exactly why nothing can
    plan run <slug> --workers 3    drain it autonomously
    plan gate <slug> <n>  exit 0 is the only definition of done

Every node needs a gate: a shell command whose exit code settles it. A node
with an irreversible effect gets `risk: irreversible` and is held for a human.

### Rules

- The gate is the evidence. A report of success is a claim; run the gate.
- Never weaken a gate to make it pass. Fix the gate and say so in a note.
- Every test assertion must be able to fail.
- Don't add a subsystem without a recorded failure demanding it.
