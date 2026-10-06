#!/usr/bin/env python3
"""Count the paraphrased AFML claims in docs/claims/ and say, claim by claim, whether a test that ran here passed.

    python scripts/claims.py                          # totals by status, kind and chapter
    python scripts/claims.py --junit results.xml      # ... and each claim's test outcome in that pytest run
    python scripts/claims.py --junit results.xml --write docs/evidence.md   # rewrite the generated block
    python scripts/claims.py --junit results.xml --check docs/evidence.md   # exit 1 if the block is out of date
    python scripts/claims.py --junit results.xml --csv claims.csv           # one row per claim

The second reading is docs/claims/audit/chNN.md: one row per numbered heading, `| section | claims | items | missing |
test mismatches | verdict | note |`, verdict ok (right as written) or fixed (something had to be added or changed).

A claim is one table row `| I<chapter>.<n> | section | kind | statement | where | evidence | status | note |` in
docs/claims/chNN.md. Its tests are every `tests/<file>.py::<test>` named in `where` or `evidence`.

Outcome of a claim, given one pytest run (a parametrised test passes only if every case passed):
  passing         status done, and every test it names ran here and passed
  partly run      status done, at least one named test passed here, the others did not run here
  reference only  status done, it names tests, but none of them ran here: each is in tests/unpublished.txt, or is
                  published and was skipped
  not in repo     status done, none of its tests ran here, and at least one names a test file that is not in tests/
                  and not in tests/unpublished.txt. The test may exist in the private build; this repo cannot show it
  failing         a named test ran here and failed or errored
  no test         status deferred or not applicable: the note gives the reason instead
A cited test whose file IS in tests/ but which that file does not define (a typo or a rename) is "unknown": it is
neither of the above, it is a broken citation, and it makes the run exit 1.
Exit 1 with --check when the generated block differs, or whenever any claim is failing or cites an unknown test.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "docs" / "claims"
ROW = re.compile(r"^\|\s*(I(\d{1,2})\.\d+)\s*\|")
AUDIT_ROW = re.compile(r"^\|\s*\d{1,2}\.\d+[\d.()]*\s*\|")
TEST = re.compile(r"tests/(\w+)\.py::(\w+)")
TESTS = ROOT / "tests"
OUTCOMES = ("passing", "partly run", "reference only", "not in repo", "failing", "no test")
BEGIN, END = "<!-- claims:begin (scripts/claims.py --write) -->", "<!-- claims:end -->"


def load() -> list[dict]:
    rows = []
    for path in sorted(CLAIMS.glob("ch[0-9][0-9].md")):
        for line in path.read_text().splitlines():
            m = ROW.match(line)
            if not m:
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) != 8:
                raise SystemExit(f"{path.name}: {m.group(1)} has {len(c)} columns, not 8")
            rows.append({"id": c[0], "chapter": int(m.group(2)), "section": c[1], "kind": c[2], "status": c[6],
                         "tests": sorted(set(TEST.findall(c[4] + " " + c[5])))})
    return rows


def audit() -> dict:
    out = Counter()
    for path in sorted((CLAIMS / "audit").glob("ch[0-9][0-9].md")):
        for line in path.read_text().splitlines():
            if AUDIT_ROW.match(line):
                c = [x.strip() for x in line.strip().strip("|").split("|")]
                out["headings"] += 1
                out[c[5]] += 1
                out["claims counted"] += int(c[1])
                out["items"] += int(c[2])
    return out


def junit_results(path: Path) -> dict[tuple[str, str], str]:
    """(module, test function) -> passed | failed | skipped, folding parametrised cases into one outcome."""
    out: dict[tuple[str, str], str] = {}
    rank = {"passed": 0, "skipped": 1, "failed": 2}
    for case in ET.parse(path).getroot().iter("testcase"):
        module = case.get("classname", "").split(".")[-1]
        name = case.get("name", "").split("[")[0]
        if case.find("failure") is not None or case.find("error") is not None:
            got = "failed"
        elif case.find("skipped") is not None:
            got = "skipped"
        else:
            got = "passed"
        key = (module, name)
        if key not in out or rank[got] > rank[out[key]]:
            out[key] = got
    return out


def published() -> dict[str, set[str]]:
    """test module -> the test functions it defines, read from the source of every tests/test_*.py in this repo."""
    return {f.stem: {n.name for n in ast.parse(f.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for f in sorted(TESTS.glob("*.py"))}


def unpublished() -> set[tuple[str, str]]:
    """The (module, function) pairs tests/unpublished.txt lists: published tests deselected for want of an input."""
    out = set()
    for line in (TESTS / "unpublished.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            m = TEST.search(line.split("#")[0])
            if m:
                out.add(m.groups())
    return out


def classify(test: tuple[str, str], results: dict, defined: dict[str, set[str]], listed: set) -> str:
    """passed | failed | skipped | listed | absent (file not in tests/) | unknown (file here, no such function) | unrun."""
    if test in results:
        return results[test]
    if test in listed:
        return "listed"
    if test[0] not in defined:
        return "absent"
    return "unrun" if test[1] in defined[test[0]] else "unknown"


def outcome(claim: dict, results: dict, defined: dict | None = None, listed: set | None = None) -> str:
    if claim["status"] != "done":
        return "no test"
    defined = published() if defined is None else defined
    listed = unpublished() if listed is None else listed
    got = [classify(t, results, defined, listed) for t in claim["tests"]]
    if "failed" in got:
        return "failing"
    if got and all(g == "passed" for g in got):
        return "passing"
    if "passed" in got:
        return "partly run"
    return "not in repo" if "absent" in got else "reference only"


def unknown_tests(rows: list[dict], results: dict) -> list[str]:
    """Cited tests that sit in a published file which defines no such function: a typo or a rename, never a reference."""
    defined, listed = published(), unpublished()
    return sorted({f"tests/{m}.py::{n}" for r in rows for m, n in r["tests"]
                   if classify((m, n), results, defined, listed) == "unknown"})


def table(rows: list[dict], results: dict | None) -> str:
    total = Counter(r["status"] for r in rows)
    lines = [f"**{len(rows):,} claims** in {len({r['chapter'] for r in rows})} chapter files: "
             f"{total['done']:,} done, {total['deferred']:,} deferred, {total['not applicable']:,} not applicable.", ""]
    a = audit()
    lines += [f"Second reading: {a['headings']} headings audited, {a['ok']} right as written, {a['fixed']} fixed; "
              f"{a['claims counted']:,} claims counted in the text against {a['items']:,} items.", ""]
    if results is None:
        return "\n".join(lines)
    tests_named = {t for r in rows for t in r["tests"]}
    defined, listed = published(), unpublished()
    ran = {t for t in tests_named if results.get(t) in ("passed", "failed")}
    absent = {t for t in tests_named if classify(t, results, defined, listed) == "absent"}
    lines += [f"Distinct tests the claims name: {len(tests_named):,}. Ran in this repo: {len(ran):,} "
              f"(passed {sum(results[t] == 'passed' for t in ran):,}, failed {sum(results[t] == 'failed' for t in ran):,}). "
              f"Named but in a test file that is not in this repo: {len(absent):,} in {len({m for m, _ in absent}):,} files.",
              "", "| chapter | claims | passing | partly run | reference only | not in repo | failing | no test |",
              "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(Counter)
    for r in rows:
        by[r["chapter"]][outcome(r, results, defined, listed)] += 1
    for ch in sorted(by):
        lines.append(f"| {ch} | {sum(by[ch].values())} | " + " | ".join(str(by[ch][o]) for o in OUTCOMES) + " |")
    allc = Counter(outcome(r, results, defined, listed) for r in rows)
    lines.append(f"| **all** | **{len(rows):,}** | " + " | ".join(f"**{allc[o]:,}**" for o in OUTCOMES) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--junit", type=Path)
    ap.add_argument("--write", type=Path)
    ap.add_argument("--check", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    rows = load()
    if not rows:
        print("no claims found under docs/claims/")       # an empty ledger must not read as a pass
        return 1
    results = junit_results(args.junit) if args.junit else None
    defined, listed = published(), unpublished()
    block = table(rows, results)
    print(block)
    if args.csv and results is not None:
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "chapter", "section", "kind", "status", "outcome", "tests"])
            for r in rows:
                w.writerow([r["id"], r["chapter"], r["section"], r["kind"], r["status"], outcome(r, results, defined, listed),
                            " ".join(f"tests/{m}.py::{n}" for m, n in r["tests"])])
    failing = results is not None and any(outcome(r, results, defined, listed) == "failing" for r in rows)
    unknown = unknown_tests(rows, results) if results is not None else []
    if unknown:                                        # a citation that points at nothing must not pass as a reference
        print(f"\n{len(unknown)} cited test(s) are in a published test file that does not define them:")
        print("\n".join("  " + u for u in unknown))
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
            print(f"\n{target} is out of date: run scripts/claims.py --junit <run> --write {target}")
            return 1
    return 1 if failing or unknown else 0


if __name__ == "__main__":
    sys.exit(main())
