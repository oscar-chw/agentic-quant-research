# plan: bindings (sanitized copy)

## goal: Strategies, features, pricing models and risk bound at any level of the tree — market, sector, industry or one instrument — each with its own toggle

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/bindings.md
      2026-09-19T13:54:16Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/bindings.md
      2026-09-19T13:54:16Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/bindings.md
      2026-09-19T13:54:16Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/bindings.md
      2026-09-19T13:54:16Z exit 0 in 0s
- [x] 5. The binding model: `src/pmlab/bindings.py` — `Binding` (kind, name, node path, enabled, params, reason, since), a `BindingSet` loaded from versioned data, and `effective(instrument)` returning each entry with the level that supplied it; refusal at load for an unknown plugin, an unknown node, or a duplicate binding at one node | needs: 4 | ctx: src/pmlab/bindings.py | gate: python -m pytest -p no:warnings tests/test_bindings.py -q
      2026-09-19T14:44:55Z exit 0 in 9s
- [x] 6. Precedence and toggles, proven: most specific wins for parameters and for the switch; off at a level is off beneath it; a lower node may turn it back on; every entry carries its provenance; an instrument left with no pricing model or no risk policy is an error naming the instrument | needs: 5 | ctx: src/pmlab/bindings.py | gate: python -m pytest -p no:warnings tests/test_bindings.py -k precedence -q
      2026-09-19T14:45:00Z exit 0 in 1s
- [x] 7. Today's arrangement expressed as bindings: the 19 strategies, their managed and micro variants, the feature sets, the pricing models and the risk limits, all bound at the market level of `polymarket-btc-5m`, reproducing exactly what runs today — proven by comparing the effective set against the live engine's own strategy set | needs: 6 | ctx: config/bindings.json | gate: python scripts/bindings_check.py --check --parity
      2026-09-19T14:45:02Z exit 0 in 1s
      2026-09-19T14:57:47Z exit 0 in 1s
- [x] 8. A second market with real structure: the Binance spot market with a sector tree and bindings that differ by sector and by instrument, showing a strategy on one sector only, a pricing model overridden for one instrument, and a feature set switched off for another | needs: 6 | ctx: config/bindings.json | gate: python scripts/bindings_check.py --check --second-market
      2026-09-19T14:45:04Z exit 0 in 1s
      2026-09-19T14:57:48Z exit 0 in 1s
- [x] 9. Changes ride the hot-swap path: a binding added, changed or toggled takes effect at the next window boundary, is validated before admission, is recorded in the run log with its reason, and is reversible; registered legs refused until 2026-10-24 | needs: 7 | ctx: src/pmlab/hotswap.py | gate: python -m pytest -p no:warnings tests/test_bindings.py -k swap -q
      2026-09-19T14:45:08Z exit 0 in 4s
- [x] 10. On the page: every level of the tree shows its effective set with provenance, and every toggle shows who turned it off, when and why; the inventory records it | needs: 7 | ctx: src/pmlab/dashboard, scripts/book_display.py | gate: python -m pytest -p no:warnings tests/test_dashboard_contract.py -q && python scripts/book_display.py --check
      2026-09-19T14:45:16Z exit 0 in 4s
- [x] 11. Written down: docs/bindings.md — the tree, the precedence rule, what a toggle means at each level, how a change is applied and reversed, and a worked example per market | needs: 8,9,10 | gate: test -s docs/bindings.md && python scripts/bindings_check.py --check
      2026-09-20T05:53:06Z exit 0 in 1s
      2026-09-20T06:21:15Z exit 0 in 1s
