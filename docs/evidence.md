# Evidence: every number, and the command that produces it

The two generated blocks below are rewritten by their scripts and checked by `scripts/check.sh`, which fails if either
block no longer matches what the scripts compute from the repository.

## (a) Claims

**How many.** One claim is one row of a chapter table in `docs/claims/`. Two ways to count them:

```bash
python scripts/claims.py                                              # totals by status, and the second reading
cat docs/claims/ch*.md | grep -cE '^\| I[0-9]{1,2}\.[0-9]+ \|'        # the same rows, counted by grep: 1805
```

**Which ones have a passing test.** A claim names its tests as `tests/<file>.py::<test>`. One pytest run is matched
against those names, claim by claim:

```bash
python -m pytest -p no:cacheprovider --junitxml=/tmp/junit.xml tests
python scripts/claims.py --junit /tmp/junit.xml                        # the table below
python scripts/claims.py --junit /tmp/junit.xml --csv claims.csv       # one row per claim, with its outcome
```

- **passing**: status done, and every test the claim names ran here and passed.
- **partly run**: at least one named test passed here; the others did not run here.
- **reference only**: the claim names tests, but none of them ran here.
- **failing**: a named test ran here and failed. A failing claim makes `claims.py` exit 1.
- **no test**: deferred or not applicable; the claim's note gives the reason.

<!-- claims:begin (scripts/claims.py --write) -->
**1,805 claims** in 22 chapter files: 1,615 done, 31 deferred, 159 not applicable.

Second reading: 276 headings audited, 111 right as written, 165 fixed; 1,826 claims counted in the text against 1,805 items.

Distinct tests the claims name: 535. Ran in this repo: 449 (passed 449, failed 0).

| chapter | claims | passing | partly run | reference only | failing | no test |
|---|---|---|---|---|---|---|
| 1 | 122 | 56 | 1 | 46 | 0 | 19 |
| 2 | 161 | 97 | 1 | 33 | 0 | 30 |
| 3 | 117 | 109 | 4 | 3 | 0 | 1 |
| 4 | 125 | 122 | 2 | 1 | 0 | 0 |
| 5 | 108 | 102 | 0 | 0 | 0 | 6 |
| 6 | 82 | 59 | 7 | 6 | 0 | 10 |
| 7 | 58 | 50 | 1 | 6 | 0 | 1 |
| 8 | 102 | 92 | 0 | 9 | 0 | 1 |
| 9 | 47 | 43 | 0 | 4 | 0 | 0 |
| 10 | 44 | 40 | 3 | 0 | 0 | 1 |
| 11 | 55 | 39 | 2 | 3 | 0 | 11 |
| 12 | 60 | 54 | 3 | 3 | 0 | 0 |
| 13 | 75 | 72 | 0 | 0 | 0 | 3 |
| 14 | 93 | 81 | 2 | 3 | 0 | 7 |
| 15 | 49 | 42 | 0 | 5 | 0 | 2 |
| 16 | 79 | 30 | 6 | 37 | 0 | 6 |
| 17 | 74 | 68 | 0 | 0 | 0 | 6 |
| 18 | 79 | 70 | 0 | 0 | 0 | 9 |
| 19 | 92 | 73 | 0 | 2 | 0 | 17 |
| 20 | 68 | 59 | 0 | 2 | 0 | 7 |
| 21 | 51 | 42 | 0 | 1 | 0 | 8 |
| 22 | 64 | 17 | 0 | 2 | 0 | 45 |
| **all** | **1,805** | **1,417** | **32** | **166** | **0** | **190** |
<!-- claims:end -->

**Why a named test may not run here.** The original build named 535 distinct tests; 449 of them ran here. The rest
fall into four groups:

1. **Test files not published**, because they import the live engine, the data recorder, the confirmation study or
   study scripts. Chapter 16 is the largest case: its HRP and strategy-lifecycle claims cite `test_lifecycle.py`, which
   imports the live performance module.
2. **Deselected by name** in `tests/unpublished.txt`: 21 tests (23 cases, since two are parametrised) whose inputs are
   not published (a study script, a study
   output, the recorded data cache or the heading index built from the book). `tests/conftest.py` fails the run if a
   line names a test that does not exist.
3. **Skipped**: 15 meta-label tests need `xgboost`, which the original build also kept out of its main environment.
4. **Not a test**: some evidence is a study report in the private repository. Such a claim counts as reference only.

Chapter 1 and chapter 22 are mostly about organisation and hardware. Most of their claims cite the design pages,
plan files and hardware page of the private repository, or are not applicable to one machine, so they are rarely
"passing" here.

**The test run.** `bash scripts/check.sh` in a fresh virtual environment built from `requirements.txt`
(Python 3.11): 758 passed, 15 skipped, 23 deselected, 0 failed.

## (b) Orchestration

**From the committed plan files.** `plan/` is a sanitized copy of the build's 18 plan graphs, written by
`scripts/export_plan.py`: node titles, dependencies and gates, and the time and exit code of each recorded gate run.
Owners, gate output and notes were dropped.

```bash
python scripts/plan_stats.py
cat plan/*.md | grep -cE '^- \[x\]'                                     # done nodes: 266
```

<!-- plan:begin (scripts/plan_stats.py --write) -->
- Plan files: 18; nodes: 285, every one with a gate: yes.
- Done (`[x]`): **266**; pending: 19; leased: 0.
- Done nodes with a recorded exit-0 gate run: 266.
- Gate runs recorded: 399, of which exit 0: 386, non-zero (failed attempts): 13; nodes that failed a gate at least once: 12.
- Gate runs dated 2026-09-16 to 2026-09-20 (UTC).
- Book plans only (afml, afml-pipeline, book-v2, bookfix-bars, philosophy): 73 nodes, 72 done; 116 gate runs, 5 failed.

| plan | nodes | done | pending | gate runs | failed runs |
|---|---|---|---|---|---|
| afml | 11 | 11 | 0 | 11 | 0 |
| afml-pipeline | 23 | 22 | 1 | 42 | 0 |
| bindings | 11 | 11 | 0 | 14 | 0 |
| book-v2 | 18 | 18 | 0 | 31 | 1 |
| bookfix-bars | 11 | 11 | 0 | 15 | 4 |
| code-eval | 15 | 15 | 0 | 16 | 0 |
| confirm-v2 | 9 | 9 | 0 | 11 | 0 |
| dashboard-book | 9 | 9 | 0 | 10 | 1 |
| dashboard-v2 | 12 | 12 | 0 | 19 | 0 |
| dashboard-v4 | 10 | 9 | 1 | 17 | 0 |
| deploy-ready | 15 | 13 | 2 | 21 | 0 |
| fair-band | 8 | 8 | 0 | 8 | 0 |
| markets | 13 | 12 | 1 | 16 | 1 |
| performance | 11 | 11 | 0 | 23 | 1 |
| philosophy | 10 | 10 | 0 | 17 | 0 |
| pm5m | 68 | 56 | 12 | 81 | 5 |
| pricing | 13 | 12 | 1 | 15 | 0 |
| v2-architecture | 18 | 17 | 1 | 32 | 0 |
<!-- plan:end -->

The plans cover the whole private project, not only the AFML library. The "book plans" line counts the five plans whose
goal is the book itself. The others built the market-data pipeline, the paper-trading engine, the dashboard, the
deployment, a line-by-line code review and a platform rewrite.

**Subagents: approximate, from private session logs.** The plan files do not record which agent ran a node. The
counts below come from the orchestrating session's private logs, which are not published, so they cannot be
reproduced from this repository: **about 140 subagent runs across the whole project** (138 subagent transcripts: 107
general-purpose agents that built, audited or re-read, 26 code reviewers, 3 security auditors and 2 code-search
agents). The orchestrator started 111 of them and subagents started the other 27. By task title, 58 of the 138 are
about the book; the 3 security audits were of the execution layer, which is not published. A plain text search of
the orchestrator's log finds about 220 matches, because it records each of its 111 calls twice.

**Commits, from the private repository's history.** 270 commits between 17 and 21 September 2026 (UTC+8), 269
of them with a `Co-Authored-By: Claude` trailer. This repository's history is new; the original history stays private.

## Copyright: the copy check

The book is copyrighted, so every published file is compared with a plain-text copy of it, which you must supply
yourself:

```bash
python scripts/copy_check.py --book <your-copy-of-AFML>.txt --report 12
```

The check fails on any run of 16 or more consecutive words shared with the book (exit 1), and exits 2 if the book
file is missing. It prints file names and run lengths only. Results on 6 October 2026:

| book text | files checked | longest shared run | where |
|---|---|---|---|
| PDF text layer | 192 | 14 words: formula symbols in a code comment | `src/pmlab/afml/breaks.py` |
| EPUB text | 192 | 12 words: a list of names from one of the book's tables | `docs/claims/ch01.md` |

Before publication the check also found three quoted sentences in test docstrings (15 to 21 words) and a run of
figure captions. These were reworded, as were 12 claim statements that shared 11 to 14 words with the book.
AFML-specific equations are referred to by section ("the book's equation in this section") rather than spelled out.
Formulas that are standard and published elsewhere, such as the Sharpe ratio, the deflated Sharpe ratio, Shannon
entropy and the Roll and Kyle models, are described in words.

## What was left out

The books and any text extracted from them; the publisher's companion PDF; a trained model file; the trading
execution layer and the live engine; the recorded market data and the test fixtures holding third-party trade
records; secrets, hosts and machine paths; private platform material; session transcripts. The history scan and the
private-code overlap check that guard these are run by the portfolio's publishing tools before any push.
