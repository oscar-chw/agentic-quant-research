#!/usr/bin/env python3
"""Write the sanitized copy of the build's plan graph that this repo publishes in plan/.

    python scripts/export_plan.py <private-repo>/plan plan/

Kept per node: its mark, number, title, needs, ctx, risk and gate. Kept per gate run: the timestamp, the exit code
and the duration. Dropped: the owner field (machine and session names), gate output, free-text notes, and the plan
files' prose sections. Interpreter paths in gates become `python`; home-directory prefixes and the file name of the
publisher's companion PDF are removed; four-level section numbers are written 2.3.2(1) as in docs/claims/, and the
claims folder's old name is updated. Run once by the author against the private repository; plan/ is its committed
output and scripts/plan_stats.py reads only that.
"""
import re
import sys
from pathlib import Path

NODE = re.compile(r"^- \[([ x>])\] (\d+)\. (.*)$")
RUN = re.compile(r"^\s+(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) exit (\d+) in (\d+)s\b")
KEEP = ("needs", "ctx", "risk", "gate")
VENV = re.compile(r"\.venv[\w-]*/bin/")
HOME = re.compile(r"/(?:Users|home)/[^/\s|]+/(?:Desktop/)?")          # a sibling repo keeps its name, not the path
COMPANION = re.compile(r"references/\S+\.pdf")                         # the publisher's companion PDF is not published
QUAD = re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{1,2})\.(\d{1,2})\b")     # a dotted four-level number reads as an IPv4 address to a leak scan


def node_line(mark: str, n: str, rest: str) -> str:
    title, *fields = [p.strip() for p in rest.split(" | ")]
    kept = [f for f in fields if f.split(":", 1)[0] in KEEP]
    line = f"- [{mark}] {n}. " + " | ".join([title, *(VENV.sub("", f) for f in kept)])
    line = COMPANION.sub("the publisher's companion tables", HOME.sub("", line))
    return QUAD.sub(r"\1(\2)", line).replace("docs/book_v2/", "docs/claims/")


def export(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(src.glob("*.md")):
        lines = path.read_text().splitlines()
        goal = next((ln for ln in lines if ln.startswith("## goal:")), None)
        out, in_node = [], False
        for line in lines:
            m = NODE.match(line)
            if m:
                out.append(node_line(*m.groups()))
                in_node = True
                continue
            r = RUN.match(line)
            if r and in_node:
                out.append(f"      {r.group(1)} exit {r.group(2)} in {r.group(3)}s")
        if not any(NODE.match(x) for x in out):
            continue                                  # a handoff note, not a work graph
        head = [f"# plan: {path.stem} (sanitized copy)", ""] + ([goal, ""] if goal else [])
        (dst / path.name).write_text("\n".join(head + out) + "\n")
        written += 1
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())
    print(f"{export(Path(sys.argv[1]), Path(sys.argv[2]))} plan files written")
