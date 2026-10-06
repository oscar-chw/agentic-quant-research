# plan: dashboard-v2 (sanitized copy)

## goal: The console reads v2's seams, not one engine's internals — so it serves many markets, many instruments and many models without a rewrite

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/dashboard-v2.md
      2026-09-20T05:28:20Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/dashboard-v2.md
      2026-09-20T05:28:20Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/dashboard-v2.md
      2026-09-20T05:28:20Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/dashboard-v2.md
      2026-09-20T05:28:20Z exit 0 in 0s
- [x] 5. Every panel declares its seam: a map from each view to the v2 protocol or market/pricing/bindings object it reads, checked by `scripts/dashboard_v2_check.py --check --seams`; a panel reading an engine internal with no declared seam fails | needs: 4 | ctx: src/pmlab/dashboard | gate: python scripts/dashboard_v2_check.py --check --seams
      2026-09-20T06:02:03Z exit 0 in 1s
      2026-09-20T06:40:21Z exit 0 in 1s
      2026-09-20T06:41:25Z exit 0 in 1s
- [x] 6. Market and instrument navigation: a market selector, the sector tree with an overview at every level from the hierarchy's own aggregates, and an instrument view; with one market the tree is one node deep and the page looks as it does today | needs: 5 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_v2.py -k navigation -q
      2026-09-20T06:02:08Z exit 0 in 1s
      2026-09-20T06:40:24Z exit 0 in 1s
- [x] 7. Prices through the pricing layer: every price shown with its band and, when refused, the reason in words; champion and challenger side by side, the challenger marked as shadow | needs: 5 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_v2.py -k pricing -q
      2026-09-20T06:02:10Z exit 0 in 1s
      2026-09-20T06:40:25Z exit 0 in 1s
      2026-09-20T06:41:27Z exit 0 in 1s
- [x] 8. What runs where, per node: the bindings card becomes navigable — pick a node, see the effective set with provenance and every toggle's who/when/why; a toggle shows what it would change before it is applied | needs: 6 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_v2.py -k bindings -q
      2026-09-20T06:02:13Z exit 0 in 1s
      2026-09-20T06:40:27Z exit 0 in 1s
- [x] 9. Jobs and lineage: the research loop's jobs with their declared inputs, outputs, budgets, checkpoints and last run; any published number traceable to code version, data vintage and parameter hash from the page | needs: 5 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_v2.py -k jobs -q
      2026-09-20T06:02:11Z exit 0 in 1s
      2026-09-20T06:40:28Z exit 0 in 1s
- [x] 10. Engine against v2, visibly: a difference panel showing the live engine's numbers beside v2's for the same window, with the worst difference and any refusal mismatch — the human half of plan/v2-architecture.md node 14 | needs: 7 | ctx: src/pmlab/dashboard | gate: python scripts/dashboard_v2_check.py --check --parity
      2026-09-20T07:31:37Z exit 0 in 2s
- [x] 11. Nothing lost: every view that worked before still works, the contract test covers the new panels, and the headless visual check passes at three widths and both themes | needs: 6,7,8,9,10 | gate: python -m pytest -p no:warnings tests/test_dashboard_contract.py tests/test_dashboard_v2.py -q && python scripts/dashboard_visual.py --check --no-screenshots
      2026-09-20T07:34:02Z exit 0 in 114s
- [x] 12. Written down and deployed: docs/dashboard_v2.md explains which seam each view reads and how a new market appears without a page change; deployed and reported | needs: 11 | gate: python scripts/dashboard_v2_check.py --check && python scripts/deploy.py --check
      2026-09-20T07:40:27Z exit 0 in 3s
