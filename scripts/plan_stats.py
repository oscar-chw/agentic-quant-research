#!/usr/bin/env python3
"""Statistics of the build's plan graph, read only from the committed sanitized copy in plan/.

    python scripts/plan_stats.py                          # print the tables
    python scripts/plan_stats.py --write docs/evidence.md # rewrite the generated block
    python scripts/plan_stats.py --check docs/evidence.md # exit 1 if the block is out of date

A node is `- [mark] N. title | needs: ... | gate: ...`; mark x = done, space = pending, > = leased. A gate run is an
indented `<UTC time> exit <code> in <s>s` line under its node: exit 0 is the only way a node is marked done, and a
non-zero exit is a failed attempt that sent the node back to pending (there is no failed state).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plan"
NODE = re.compile(r"^- \[([ x>])\] (\d+)\. (.*)$")
RUN = re.compile(r"^\s+(\d{4}-\d\d-\d\d)T\S+ exit (\d+) in \d+s$")
# The plans whose goal is the book itself; the others built the data pipeline, the paper-trading engine, the dashboard,
# the deployment and a platform rewrite. Kept separate so a book-build number is never a whole-project number.
BOOK_PLANS = ("afml", "afml-pipeline", "book-v2", "bookfix-bars", "philosophy")
BEGIN, END = "<!-- plan:begin (scripts/plan_stats.py --write) -->", "<!-- plan:end -->"


def load() -> list[dict]:
    nodes = []
    for path in sorted(PLAN.glob("*.md")):
        current = None
        for line in path.read_text().splitlines():
            m = NODE.match(line)
            if m:
                current = {"plan": path.stem, "mark": m.group(1), "gated": " | gate: " in m.group(3), "runs": []}
                nodes.append(current)
            elif (r := RUN.match(line)) and current is not None:
                current["runs"].append((r.group(1), int(r.group(2))))
    return nodes


def book_line(nodes: list[dict]) -> str:
    ns = [n for n in nodes if n["plan"] in BOOK_PLANS]
    rs = [c for n in ns for _, c in n["runs"]]
    return (f"- Book plans only ({', '.join(BOOK_PLANS)}): {len(ns)} nodes, {sum(n['mark'] == 'x' for n in ns)} done; "
            f"{len(rs)} gate runs, {sum(1 for c in rs if c)} failed.")


def table(nodes: list[dict]) -> str:
    runs = [code for n in nodes for _, code in n["runs"]]
    days = sorted({d for n in nodes for d, _ in n["runs"]})
    done = [n for n in nodes if n["mark"] == "x"]
    marks = Counter(n["mark"] for n in nodes)
    lines = [
        f"- Plan files: {len({n['plan'] for n in nodes})}; nodes: {len(nodes)}, every one with a gate: "
        f"{'yes' if all(n['gated'] for n in nodes) else 'NO'}.",
        f"- Done (`[x]`): **{marks['x']}**; pending: {marks[' ']}; leased: {marks['>']}.",
        f"- Done nodes with a recorded exit-0 gate run: {sum(any(c == 0 for _, c in n['runs']) for n in done)}.",
        f"- Gate runs recorded: {len(runs)}, of which exit 0: {runs.count(0)}, non-zero (failed attempts): "
        f"{len(runs) - runs.count(0)}; nodes that failed a gate at least once: "
        f"{sum(any(c for _, c in n['runs']) for n in nodes)}.",
        f"- Gate runs dated {days[0]} to {days[-1]} (UTC)." if days else "- No gate runs recorded.",
        book_line(nodes),
        "", "| plan | nodes | done | pending | gate runs | failed runs |", "|---|---|---|---|---|---|"]
    for plan in sorted({n["plan"] for n in nodes}):
        ns = [n for n in nodes if n["plan"] == plan]
        rs = [c for n in ns for _, c in n["runs"]]
        lines.append(f"| {plan} | {len(ns)} | {sum(n['mark'] == 'x' for n in ns)} | {sum(n['mark'] != 'x' for n in ns)}"
                     f" | {len(rs)} | {sum(1 for c in rs if c)} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", type=Path)
    ap.add_argument("--check", type=Path)
    args = ap.parse_args()
    nodes = load()
    if not nodes:
        print("no plan nodes found under plan/")       # an absent graph must not read as a pass
        return 1
    block = table(nodes)
    print(block)
    target = args.write or args.check
    if target:
        text = target.read_text()
        if BEGIN not in text or END not in text:
            print(f"{target}: no generated block ({BEGIN} ... {END})")
            return 1
        head, rest = text.split(BEGIN, 1)
        new = head + BEGIN + "\n" + block + "\n" + END + rest.split(END, 1)[1]
        if args.write:
            target.write_text(new)
        elif new != text:
            print(f"\n{target} is out of date: run scripts/plan_stats.py --write {target}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
