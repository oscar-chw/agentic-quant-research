# plan: book-v2 (sanitized copy)

## goal: Implement AFML v2: every chapter walked sub-subsection by sub-subsection into to-do lists, every logic, formula, structure, algorithm and pitfall implemented and evaluated

- [x] 1. Analyse: what is actually being asked, and what would prove it done | gate: grep -q '^## analysis' plan/book-v2.md
      2026-09-19T00:47:48Z exit 0 in 0s
- [x] 2. Research: prior art, constraints, unknowns resolved or explicitly deferred | needs: 1 | gate: grep -q '^## research' plan/book-v2.md
      2026-09-19T00:47:48Z exit 0 in 0s
- [x] 3. Plan, with acceptance tests written before any implementation | needs: 2 | gate: grep -q '^## plan' plan/book-v2.md
      2026-09-19T00:47:48Z exit 0 in 0s
- [x] 4. Finalise: scope, gates and done-condition agreed | needs: 3 | gate: grep -q '^APPROVED' plan/book-v2.md
      2026-09-19T00:47:48Z exit 0 in 0s
- [x] 5. The v2 tool and schema: scripts/book_v2.py parses docs/claims/chNN.md (one item table per heading: id, section, kind logic/math/structure/algo/pitfall/constraint, statement, where, evidence, status done/deferred/not applicable, note), checks every heading of tests/afml_sections.json is present, every item has a valid status, done needs a `tests/...py::test_...` that exists, deferred and not applicable need a reason of at least 40 characters, ids are unique; --check also runs every named test; --todo prints open items; tests/test_book_v2.py covers each failure mode | gate: python -m pytest -p no:warnings tests/test_book_v2.py -q
      2026-09-19T00:48:56Z exit 0 in 0s
- [x] 6. Chapters 1–5 as v2 to-do files, every item resolved | needs: 5 | ctx: docs/book_v2, docs/afml_sections | gate: python scripts/book_v2.py --check --chapters 1-5
      2026-09-19T01:28:52Z exit 0 in 72s
      2026-09-19T01:35:03Z exit 0 in 72s
- [x] 7. Chapters 6–9 | needs: 5 | ctx: docs/book_v2, src/pmlab/afml | gate: python scripts/book_v2.py --check --chapters 6-9
      2026-09-19T01:29:16Z exit 0 in 52s
      2026-09-19T01:35:41Z exit 0 in 52s
      2026-09-19T01:35:56Z exit 0 in 52s
- [x] 8. Chapters 10–12 | needs: 5 | ctx: docs/book_v2, src/pmlab/afml | gate: python scripts/book_v2.py --check --chapters 10-12
      2026-09-19T01:23:29Z exit 0 in 19s
      2026-09-19T01:25:06Z exit 0 in 22s
- [x] 9. Chapters 13–16 | needs: 5 | ctx: docs/book_v2, src/pmlab/afml | gate: python scripts/book_v2.py --check --chapters 13-16
      2026-09-19T01:15:08Z exit 0 in 18s
      2026-09-19T01:24:30Z exit 0 in 19s
- [x] 10. Chapters 17–19 | needs: 5 | ctx: docs/book_v2, src/pmlab/afml | gate: python scripts/book_v2.py --check --chapters 17-19
      2026-09-19T01:29:54Z exit 0 in 66s
      2026-09-19T01:33:41Z exit 0 in 65s
- [x] 11. Chapters 20–22, judged against this Mac Studio (12 cores, 32 GB): every item adapted, not applicable or deferred with the number that decides it | needs: 5 | ctx: docs/book_v2, docs/hardware.md | gate: python scripts/book_v2.py --check --chapters 20-22
      2026-09-19T01:20:02Z exit 0 in 18s
      2026-09-19T01:21:06Z exit 0 in 19s
- [x] 12. Hardware budget written once: docs/hardware.md gives this machine's cores, memory, disk and measured throughput (events/s in replay, rows/s in the event build, RSS per worker), the ceilings they impose on every stage, and what each chapter-20–22 item may assume | needs: 5 | ctx: docs/hardware.md | gate: python scripts/book_v2.py --check --hardware
      2026-09-19T00:49:28Z exit 1 in 0s
      2026-09-19T00:49:41Z exit 0 in 0s
- [x] 13. Every open item closed and the whole book passes: docs/claims/README.md indexes the chapters with counts per kind and status, lists every deferred item with its reason and date, and ends CODE V2 COMPLETE | needs: 6,7,8,9,10,11,12 | gate: python scripts/book_v2.py --check
      2026-09-19T01:43:08Z exit 0 in 193s
- [x] 14. The audit schema and the element rule are themselves tested: scripts/book_v2.py --audit requires docs/claims/audit/chNN.md with a row of seven columns (section, claims, items, missing, test mismatches, verdict, note) for every heading, an unfixed omission cannot be 'ok', claims > items forces 'fixed', and every Snippet, Figure, Table and Example starting inside a heading's span is named by one of its items; tests/test_book_v2.py covers each of those failure modes | needs: 13 | ctx: scripts/book_v2.py | gate: python -m pytest -p no:warnings tests/test_book_v2.py -q
      2026-09-19T02:48:46Z exit 0 in 0s
      2026-09-19T03:39:23Z exit 0 in 0s
- [x] 15. Second reading, chapters 1-5: an auditor who did not write them re-reads every line of all 82 heading spans, counts the claims, names what no item covers, opens every test a done item names and replaces the ones that could not fail, and records it in docs/claims/audit/ch01.md..ch05.md | needs: 14 | ctx: docs/book_v2, docs/claims/audit | gate: python scripts/book_v2.py --check --audit --chapters 1-5
      2026-09-19T03:35:12Z exit 0 in 89s
      2026-09-19T03:39:23Z exit 0 in 89s
- [x] 16. Second reading, chapters 6-12: an auditor who did not write them re-reads every line of all 54 heading spans, counts the claims, names what no item covers, opens every test a done item names and replaces the ones that could not fail, and records it in docs/claims/audit/ch06.md..ch12.md | needs: 14 | ctx: docs/book_v2, docs/claims/audit | gate: python scripts/book_v2.py --check --audit --chapters 6-12
      2026-09-19T03:21:53Z exit 0 in 72s
      2026-09-19T03:37:44Z exit 0 in 78s
- [x] 17. Second reading, chapters 13-17: an auditor who did not write them re-reads every line of all 61 heading spans, counts the claims, names what no item covers, opens every test a done item names and replaces the ones that could not fail, and records it in docs/claims/audit/ch13.md..ch17.md | needs: 14 | ctx: docs/book_v2, docs/claims/audit | gate: python scripts/book_v2.py --check --audit --chapters 13-17
      2026-09-19T03:56:17Z exit 0 in 91s
      2026-09-19T04:15:12Z exit 0 in 92s
      2026-09-19T04:22:09Z exit 0 in 88s
- [x] 18. Second reading, chapters 18-22: an auditor who did not write them re-reads every line of all 79 heading spans, counts the claims, names what no item covers, opens every test a done item names and replaces the ones that could not fail, re-derives every chapter-20-22 hardware judgement from docs/hardware.md, and records it in docs/claims/audit/ch18.md..ch22.md | needs: 14 | ctx: docs/book_v2, docs/claims/audit, docs/hardware.md | gate: python scripts/book_v2.py --check --audit --chapters 18-22
      2026-09-19T03:49:46Z exit 0 in 24s
