# plan: dashboard-book (sanitized copy)

## goal: The dashboard shows the book, in the book's order, with nothing it computes left off the page

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/dashboard-book.md
      2026-09-19T02:39:13Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/dashboard-book.md
      2026-09-19T02:39:13Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/dashboard-book.md
      2026-09-19T02:39:13Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/dashboard-book.md
      2026-09-19T02:39:13Z exit 0 in 0s
- [x] 5. The inventory: scripts/book_display.py builds research/book_display.json — every computed quantity (feature, estimator, statistic, allocation, size) with the book section that asks for it, where it is computed, whether it is live or a study result, and where it appears on the page; tests/test_book_display.py fails if a feature exists in code but not in the inventory | needs: 4 | ctx: src/pmlab/afml, src/pmlab/bars, docs/book_v2 | gate: python -m pytest -p no:warnings tests/test_book_display.py -q
      2026-09-19T02:51:13Z exit 0 in 2s
- [x] 6. The Book tab: a view whose top level is the book's four parts and their chapters, each chapter listing its quantities with live values or the latest study value, the status from docs/book_v2 (done, deferred with its date, not applicable with its reason) and a link to the report; chapters with nothing implemented say so | needs: 5 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_contract.py tests/test_book_display.py -q && python scripts/dashboard_visual.py --check --no-screenshots
      2026-09-19T03:24:37Z exit 1 in 3s
      2026-09-19T03:27:23Z exit 0 in 112s
- [x] 7. Existing views re-ordered to the book and completed: the bars panel shows all nine kinds with their calibration and what each is for (2.3), not a bare count; the live features panel groups by chapter with the definition of each; microstructure shows every chapter-19 estimator, not a subset; labels, weights and uniqueness (3, 4) become visible; the fair price keeps its band | needs: 5 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_contract.py -q && python scripts/dashboard_visual.py --check --no-screenshots
      2026-09-19T03:29:17Z exit 0 in 110s
- [x] 8. Nothing computed is left off the page: `--check` fails if a quantity in the inventory has no place on the page and no recorded reason; docs/dashboard_v4.md updated to describe the book ordering | needs: 6,7 | gate: python scripts/book_display.py --check
      2026-09-19T03:29:23Z exit 0 in 2s
- [x] 9. Deployed (user, 2026-09-19: "i approve restarts" — standing approval; deploy when node 8 gates, report afterwards), with an outcome-neutral clarification for any registered dashboard file that changed | needs: 8 | risk: irreversible | gate: python scripts/deploy.py --check && python scripts/confirm.py --check
      2026-09-19T07:52:37Z exit 0 in 1s
