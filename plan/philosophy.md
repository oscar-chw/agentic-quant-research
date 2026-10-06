# plan: philosophy (sanitized copy)

## goal: The book's design philosophy, extracted from where it is argued and enforced by machinery — not remembered

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/philosophy.md
      2026-09-20T05:56:54Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/philosophy.md
      2026-09-20T05:56:54Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/philosophy.md
      2026-09-20T05:56:54Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/philosophy.md
      2026-09-20T05:56:54Z exit 0 in 0s
- [x] 5. Extract the philosophy: read every chapter for arguments about how to build and run the research, not for its methods; write docs/philosophy.md with one entry per principle — our own words, the sections where it is argued, and what it forbids as well as what it asks for | needs: 4 | ctx: docs/philosophy.md | gate: python scripts/philosophy_check.py --check --extracted
      2026-09-20T06:43:09Z exit 0 in 0s
      2026-09-20T07:03:53Z exit 0 in 0s
- [x] 6. Map each principle to its mechanism: the refusal, required argument, gate or artefact that enforces it, named by file and symbol; a principle with no mechanism is marked unenforced with the plan node that would fix it | needs: 5 | ctx: docs/philosophy.md | gate: python scripts/philosophy_check.py --check --mechanisms
      2026-09-20T06:43:12Z exit 0 in 0s
      2026-09-20T07:03:53Z exit 0 in 0s
- [x] 7. Each mechanism proven able to fail: a mutation harness breaks each named mechanism in a scratch copy, runs that principle's test, and requires it to fail; the evidence recorded per principle. The harness does not own the tests it judges — it reads the principles and reports | needs: 6 | ctx: docs/philosophy.md, src/pmlab | gate: python scripts/philosophy_check.py --check --mutations
      2026-09-20T06:43:12Z exit 0 in 0s
      2026-09-20T07:03:53Z exit 0 in 0s
      2026-09-20T07:12:08Z exit 0 in 0s
- [x] 8. The unenforced ones closed or recorded: for each principle with no mechanism, either build the smallest one that catches a violation, or record why it cannot be mechanised here and what a human must check instead | needs: 7 | ctx: docs/philosophy.md | gate: python scripts/philosophy_check.py --check
      2026-09-20T06:43:12Z exit 0 in 0s
      2026-09-20T07:03:53Z exit 0 in 0s
      2026-09-20T07:12:09Z exit 0 in 0s
- [x] 9. Where the book is wrong for this market: principles our own evidence contradicts, with the measurement and what we do instead — kept beside the principle, never silently dropped | needs: 5 | ctx: docs/philosophy.md, quant-harness/docs/book_errors.md | gate: python scripts/philosophy_check.py --check --contradictions
      2026-09-20T06:43:12Z exit 0 in 0s
      2026-09-20T07:03:54Z exit 0 in 0s
- [x] 10. On the page: the philosophy as a view — each principle, its mechanism, whether its test is green, and the contradictions — so the console answers "is this system still built the way the book means" | needs: 8 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_contract.py -q && python scripts/philosophy_check.py --check
      2026-09-20T07:46:44Z exit 0 in 3s
