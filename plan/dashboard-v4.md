# plan: dashboard-v4 (sanitized copy)

## goal: Revamp the whole dashboard without losing any function: every feature the user asked for, everything an auto-trading and analysis system needs, more information, designed and verified visually

- [x] 1. Analyse: what is actually being asked, and what would prove it done | gate: grep -q '^## analysis' plan/dashboard-v4.md
      2026-09-17T12:36:32Z exit 0 in 0s
- [x] 2. Research: prior art, constraints, unknowns resolved or explicitly deferred | needs: 1 | gate: grep -q '^## research' plan/dashboard-v4.md
      2026-09-17T12:36:32Z exit 0 in 0s
- [x] 3. Plan, with acceptance tests written before any implementation | needs: 2 | gate: grep -q '^## plan' plan/dashboard-v4.md
      2026-09-17T12:36:32Z exit 0 in 0s
- [x] 4. Finalise: scope, gates and done-condition agreed | needs: 3 | gate: grep -q '^APPROVED' plan/dashboard-v4.md
      2026-09-17T12:39:41Z exit 0 in 0s
- [x] 5. Functional contract: tests/test_dashboard_contract.py records every route, tab, panel, control, table (with its sort), chart and ⓘ key of today's site (from index.html, the JS modules and server.py) and fails when any disappears; passes on the current site before any redesign | needs: 4 | ctx: src/pmlab/dashboard | gate: python -m pytest tests/test_dashboard_contract.py -p no:warnings -q
      2026-09-17T12:59:18Z exit 0 in 1s
      2026-09-17T13:15:47Z exit 0 in 1s
- [x] 6. Design system and information architecture: tokens (colour, type, spacing) with a dark theme kept as default and a light theme added, components (stat tile, card, sortable table, chart frame, badge, alert, stage pill), tabs Overview · Live · Market · Strategies · Performance · Research · Data · Ops, every current feature mapped to its new place in docs/dashboard_v4.md; design critique and accessibility review recorded | needs: 5 | ctx: docs/dashboard_v4.md, src/pmlab/dashboard/static/style.css | gate: python -m pytest tests/test_dashboard_contract.py -p no:warnings -q && grep -q '^## feature map' docs/dashboard_v4.md
      2026-09-17T13:15:09Z exit 0 in 1s
      2026-09-17T13:15:49Z exit 0 in 2s
- [x] 7. Read-only server endpoints for the new views: ops (service, watchdog state and alerts, deploy history, integrity reports, disk, memory, feed lag), lifecycle status, execution shadow and refusal chain, pipeline stages; each tested and bounded in memory | needs: 5 | ctx: src/pmlab/dashboard/server.py | gate: python -m pytest tests/test_dashboard_ops.py -p no:warnings -q
      2026-09-17T13:29:22Z exit 0 in 0s
      2026-09-17T14:01:19Z exit 0 in 0s
      2026-09-17T14:13:38Z exit 0 in 0s
      2026-09-17T14:14:57Z exit 0 in 0s
- [x] 8. Rebuild the page on the design system: every tab implemented, every contract item present, the new Overview and Ops views live, ⓘ texts explicit for every new item | needs: 6,7 | ctx: src/pmlab/dashboard/static | gate: python -m pytest tests/test_dashboard.py tests/test_dashboard_contract.py tests/test_confirmation_panel.py tests/test_replay_server.py -p no:warnings -q
      2026-09-17T14:08:16Z exit 0 in 88s
      2026-09-17T14:13:37Z exit 0 in 88s
      2026-09-17T14:16:25Z exit 0 in 87s
- [x] 9. Visual verification: scripts/dashboard_visual.py drives the built-in browser against a simulated dashboard at 375, 768 and 1440 px in both themes, saves screenshots under research/figures/dashboard_v4/, fails on console errors, horizontal scroll, missing ⓘ texts or empty views | needs: 8 | ctx: scripts/dashboard_visual.py | gate: python scripts/dashboard_visual.py --check
      2026-09-17T14:12:01Z exit 0 in 100s
      2026-09-17T14:18:05Z exit 0 in 99s
- [ ] 10. Deploy after the confirmation v2 registration (plan/confirm-v2.md node 8 done) through scripts/deploy.py, health verified, rollback ready | needs: 9 | risk: irreversible | ctx: scripts/deploy.py | gate: test -s research/preregistration.json && python scripts/deploy.py --check
