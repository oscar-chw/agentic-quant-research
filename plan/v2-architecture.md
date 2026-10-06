# plan: v2-architecture (sanitized copy)

## goal: v2 — a quant platform for research, trading and analysis that a firm could actually run, with the current system as its first tenant

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/v2-architecture.md
      2026-09-19T07:20:06Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/v2-architecture.md
      2026-09-19T07:20:06Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/v2-architecture.md
      2026-09-19T07:20:06Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/v2-architecture.md
      2026-09-19T07:20:06Z exit 0 in 0s
- [x] 5. The skeleton: `src/pmlab/v2/` with the layer boundaries above as protocols only — no logic — plus a registry per plugin kind and a conformance test suite that any implementation must pass | needs: 4 | ctx: src/pmlab/v2 | gate: python -m pytest -p no:warnings tests/v2/test_protocols.py -q
      2026-09-19T07:34:24Z exit 0 in 3s
      2026-09-19T08:00:18Z exit 0 in 7s
      2026-09-19T08:14:50Z exit 0 in 7s
- [x] 6. Time, instruments and the sector tree: calendars, sessions, horizons, point-in-time rules, `Instrument`/`Contract`, market → sector → industry → instrument with aggregates at each level | needs: 5 | ctx: src/pmlab/v2 | gate: python -m pytest -p no:warnings tests/v2/test_time_instruments.py -q
      2026-09-19T07:40:28Z exit 0 in 1s
      2026-09-19T08:00:19Z exit 0 in 1s
      2026-09-19T08:14:51Z exit 0 in 1s
- [x] 7. Record and catalogue: one reader and writer over the existing warehouse, with lineage (source, vintage, code version) and a point-in-time read that fails a test if it could see the future | needs: 5 | ctx: src/pmlab/v2, src/pmlab/live/warehouse.py | gate: python -m pytest -p no:warnings tests/v2/test_record.py -q
      2026-09-19T07:46:30Z exit 0 in 2s
      2026-09-19T08:00:22Z exit 0 in 2s
      2026-09-19T08:14:54Z exit 0 in 2s
- [x] 8. Sampling, features, labels and weights over the record, market-agnostic, with a versioned cache keyed by input vintage and code version | needs: 6,7 | ctx: src/pmlab/v2 | gate: python -m pytest -p no:warnings tests/v2/test_features.py -q
      2026-09-19T13:09:30Z exit 0 in 1s
      2026-09-19T13:55:36Z exit 0 in 2s
- [x] 9. Models and sizing: purged cross-validation and embargo built into the fitting path so a model cannot be fitted without them, plus chapter 10 sizing | needs: 8 | ctx: src/pmlab/v2 | gate: python -m pytest -p no:warnings tests/v2/test_models.py -q
      2026-09-19T13:09:34Z exit 0 in 1s
      2026-09-19T13:55:38Z exit 0 in 2s
- [x] 10. Portfolio, risk and accounting: limits, allocation, drawdown and kill switches as policy objects; one ledger reconciled every window against fills | needs: 6 | ctx: src/pmlab/v2 | gate: python -m pytest -p no:warnings tests/v2/test_portfolio.py -q
      2026-09-19T13:09:36Z exit 0 in 1s
      2026-09-19T13:55:41Z exit 0 in 2s
- [x] 11. Execution: `Broker` with the paper and simulated implementations behind it, queue and cost models per venue, and the live path still refused by the existing enablement chain | needs: 10 | ctx: src/pmlab/v2, src/pmexec | gate: python -m pytest -p no:warnings tests/v2/test_execution_layer.py -q
      2026-09-20T05:31:34Z exit 0 in 1s
      2026-09-20T06:38:53Z exit 0 in 1s
- [x] 12. Runtime: the live loop and the research loop as jobs with declared inputs, outputs, budgets and checkpoints; hot swap at window boundaries (plan/markets.md node 11) | needs: 9,11 | ctx: src/pmlab/v2 | gate: python -m pytest -p no:warnings tests/v2/test_runtime.py -q
      2026-09-20T05:36:54Z exit 0 in 1s
      2026-09-20T06:38:55Z exit 0 in 1s
- [x] 13. Reproducibility as a property: every published number carries (code version, data vintage, parameter hash); `scripts/v2_check.py --trace` fails on any that does not, and a replay reproduces it | needs: 7,9,12 | ctx: src/pmlab/v2 | gate: python scripts/v2_check.py --trace
      2026-09-20T06:16:03Z exit 0 in 1s
      2026-09-20T06:38:57Z exit 0 in 2s
      2026-09-20T06:45:31Z exit 0 in 2s
- [x] 14. Parity: the Polymarket BTC 5-minute system runs end to end through v2 and reproduces today's engine to the last digit on recorded windows, refusing where it refuses | needs: 12 | ctx: scripts/v2_check.py | gate: python scripts/v2_check.py --parity
      2026-09-20T06:16:25Z exit 0 in 22s
      2026-09-20T06:39:26Z exit 0 in 25s
      2026-09-20T06:45:28Z exit 0 in 24s
- [x] 15. Second tenant: the Binance spot market and a linear contract run through v2 unchanged, proving no special case was needed | needs: 14 | gate: python scripts/v2_check.py --second-market
      2026-09-20T07:58:33Z exit 0 in 11s
- [x] 16. Firm operations, the parts the book leaves out: capital and treasury (cash, margin, funding, capacity per strategy), cost accounting (fees, data, compute, so P&L is net of running the thing), compliance and permissions (jurisdiction gating, restricted instruments, who may change what), reporting generated from the trading ledger rather than re-keyed, incident response with an audit trail and a written post-mortem, and roles (researcher, trader, operator) | needs: 10,12 | ctx: src/pmlab/v2 | gate: python -m pytest -p no:warnings tests/v2/test_firm_operations.py -q
      2026-09-20T07:58:37Z exit 0 in 1s
- [x] 17. Written down: docs/v2/ — the layer map, what each seam guarantees, the budgets, and worked walk-throughs for adding a market, a feature, a strategy, a model and a venue | needs: 14,15,16 | gate: python scripts/v2_check.py --check
      2026-09-20T07:59:11Z exit 0 in 33s
- [ ] 18. (Held until 2026-10-24, the confirmation's analysis.) Cutover: the live app runs on v2, with registered leg behaviour unchanged, an amendment entry recording the swap, and the old path kept for one week as the rollback | needs: 13,14,17 | risk: irreversible | gate: python scripts/confirm.py --check && python scripts/v2_check.py --check && python scripts/preflight.py --check
