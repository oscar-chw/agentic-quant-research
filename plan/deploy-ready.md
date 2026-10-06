# plan: deploy-ready (sanitized copy)

## goal: Make the whole project deploy and trade ready: paper trading production-ready and always on (this Mac, login service, watchdog, alerts), a real-order path built and tested against a simulated exchange but locked behind a kill switch and limits until a strategy graduates and the user adds keys, data integrity checks and disk alarms

- [x] 1. Analyse: what is actually being asked, and what would prove it done | gate: grep -q '^## analysis' plan/deploy-ready.md
      2026-09-17T12:11:25Z exit 0 in 0s
- [x] 2. Research: prior art, constraints, unknowns resolved or explicitly deferred | needs: 1 | gate: grep -q '^## research' plan/deploy-ready.md
      2026-09-17T12:11:25Z exit 0 in 0s
- [x] 3. Plan, with acceptance tests written before any implementation | needs: 2 | gate: grep -q '^## plan' plan/deploy-ready.md
      2026-09-17T12:11:25Z exit 0 in 0s
- [x] 4. Finalise: scope, gates and done-condition agreed | needs: 3 | gate: grep -q '^APPROVED' plan/deploy-ready.md
      2026-09-17T12:12:36Z exit 0 in 0s
- [x] 5. Release worktree, deploy and rollback: scripts/deploy.py deploys a commit to .releases/<sha> (git worktree, gitignored; data/ linked to the main tree's data; run with that release's src first on PYTHONPATH), runs the quick gates, restarts after a window settles, verifies warm start and /api/snapshot health, and rolls back to the previous release automatically on failure; `--rollback` and `--check` | needs: 4 | ctx: scripts/deploy.py, .gitignore | gate: python -m pytest tests/test_deploy.py -p no:warnings -q
      2026-09-17T12:41:41Z exit 0 in 36s
      2026-09-17T12:46:11Z exit 0 in 36s
- [ ] 6. Supervision (user, 2026-09-17: skip auto-start, after launchd could not run the app from ~/Desktop, exit 78 EX_CONFIG): the watchdog loop deploy/watchdog_loop.sh restarts the pid-file app on a crash or stalled feeds, proven by a crash test (scripts/service.py crash-test: the app killed, the watchdog brings it back healthy); deploys use the pid-file backend; the failed com.pmlab.live job is removed by the user (launchd unloading is blocked for Claude); no reboot recovery | needs: 5 | ctx: scripts/service.py, scripts/watchdog.py, deploy/watchdog_loop.sh | gate: python -m pytest tests/test_service.py tests/test_watchdog.py -p no:warnings -q && python scripts/watchdog.py --check && python -c "import json; r=json.load(open('data/service_crash_test.json')); assert r['ok'] and r['supervisor'] == 'ProcessBackend'" && ! launchctl print gui/$(id -u)/com.pmlab.live >/dev/null 2>&1
- [x] 7. Watchdog: scripts/watchdog.py run every 60 s by launchd: health endpoint timeout, feed ages beyond thresholds for minutes, queue growth, memory cap, disk free (warn < 100 GB, critical < 20 GB); restarts through launchctl kickstart at most 3 times an hour, then notifies only; macOS notification and data/alerts.log; decisions tested on injected faults | needs: 4 | ctx: scripts/watchdog.py | gate: python -m pytest tests/test_watchdog.py -p no:warnings -q
      2026-09-17T12:41:58Z exit 0 in 0s
      2026-09-17T12:43:55Z exit 0 in 0s
      2026-09-17T12:46:11Z exit 0 in 0s
- [x] 8. Nightly data integrity: scripts/integrity.py (launchd, 03:30 Taiwan time): every recorded hour readable, rows per kind per hour against minimums, gaps listed, compaction backlog, db views resolve; report in data/integrity/<date>.json; notification on failure | needs: 4 | ctx: scripts/integrity.py, src/pmlab/live/warehouse.py | gate: python -m pytest tests/test_integrity.py -p no:warnings -q && python scripts/integrity.py --run --check
      2026-09-17T12:42:39Z exit 0 in 37s
      2026-09-17T12:44:35Z exit 0 in 40s
      2026-09-17T12:46:52Z exit 0 in 41s
- [x] 9. Dependencies pinned: requirements.lock from the working venv (no downloads) and scripts/deps.py --check comparing installed versions | needs: 4 | ctx: pyproject.toml | gate: python scripts/deps.py --check
      2026-09-17T12:13:03Z exit 0 in 0s
- [x] 10. Strategy lifecycle by the book (AFML ch. 1, 15, 16): research/lifecycle.md rules (stages research, embargo, paper, graduated, decommissioned; graduation needs a confirmed registered leg, a minimum paper record, strategy-risk failure probability below its bound, DSR above its bound with the registry's trials; decommission triggers; HRP allocation among graduated strategies) and a checker writing research/lifecycle_status.json with each strategy's stage and the reason | needs: 4 | ctx: src/pmlab/afml/lifecycle.py, scripts/lifecycle.py | gate: python -m pytest tests/test_lifecycle.py -p no:warnings -q && python scripts/lifecycle.py --check
      2026-09-17T13:00:56Z exit 0 in 18s
      2026-09-17T13:11:54Z exit 0 in 16s
- [x] 11. Execution layer, locked (package src/pmexec, outside the registered code): order router with a stop switch, per-order/window/day notional limits, price bands, reconciliation that halts on mismatch, a simulated exchange implementing the documented order, cancel, open-orders and trades interface, shadow mode reading the engine's recorded signals; real trading refused unless a strategy is graduated (node 10), the user enabled it in a read-only config, Keychain credentials exist and the geoblock check passes; each refusal and each limit is a test; no real Polymarket adapter while the geoblock blocks this location | needs: 4 | ctx: src/pmexec | gate: python -m pytest tests/test_execution.py -p no:warnings -q && python -m pmexec --check
      2026-09-17T13:00:33Z exit 0 in 2s
      2026-09-17T13:11:57Z exit 0 in 2s
- [x] 12. Security review of the execution layer and credential handling (security-auditor agent, fresh context); every finding fixed with a test or recorded with its reason | needs: 11 | ctx: src/pmexec, research/security_review_execution.md | gate: grep -q '^REVIEW CLOSED' research/security_review_execution.md
      2026-09-17T14:56:04Z exit 0 in 0s
- [x] 13. Operations runbook: docs/operations.md (start, stop, deploy, rollback, service, watchdog and alerts, integrity reports, stop switch, lifecycle stages, what enabling real trading requires and why it is refused from this location) | needs: 5,6,7,8,11 | ctx: docs/operations.md | gate: python -m pytest tests/test_runbook.py -p no:warnings -q
      2026-09-17T13:15:22Z exit 0 in 4s
- [ ] 14. Ready check: scripts/ready.py --check runs deploy --check, service --check, watchdog last run, integrity --check, deps --check, preflight --check, confirm.py --check, lifecycle --check, pmexec --check and the runbook test, printing each result | needs: 5,6,7,8,9,10,11,12,13 | ctx: scripts/ready.py | gate: python scripts/ready.py --check
- [x] 15. Execution layer hardening from the round-3 security re-review (research/security_review_execution.md L1–L8): send lock and thread check on submit, submit_batch and cancel; a non-simulated router refuses to start without a Heartbeat on the same exchange and wires chain freshness itself; a wrongly sized batch answer marks every intent refused and records sent orders as open; the geoblock request connects to a checked global address and compares the peer; interface allowlist from hardware ports; is_simulated checks helpers and instance callables; only documented rejection codes recorded; exchange side validated; each with a failing-then-passing test | needs: 12 | ctx: src/pmexec | gate: python -m pytest tests/test_execution.py tests/test_execution_hardening.py -p no:warnings -q && python -m pmexec --check
      2026-09-17T15:25:25Z exit 0 in 6s
      2026-09-17T15:26:03Z exit 0 in 4s
