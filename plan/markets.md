# plan: markets (sanitized copy)

## goal: One underlying pipeline, many markets: A-shares, crypto or anything else added as a description rather than a rewrite

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/markets.md
      2026-09-19T07:08:03Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/markets.md
      2026-09-19T07:08:03Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/markets.md
      2026-09-19T07:08:04Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/markets.md
      2026-09-19T07:08:04Z exit 0 in 0s
- [x] 5. The description: `src/pmlab/markets/` with a `Market` dataclass (id, sector, horizon, session calendar, instruments, valuation, settlement, costs, record shapes, hierarchy) and a registry (`register`, `get`, `all`), documented so a new market is one file and nothing else; nothing registered imports it yet | needs: 4 | ctx: src/pmlab/markets | gate: python -m pytest -p no:warnings tests/test_markets.py -q
      2026-09-19T07:45:40Z exit 0 in 7s
- [x] 6. This market, described: `markets/polymarket_btc_5m.py` states today's behaviour exactly — 300 s windows, two outcome books, the TWAP-digital fair price, the oracle settlement rule, the taker fee and queue | needs: 5 | ctx: src/pmlab/markets | gate: python -m pytest -p no:warnings tests/test_markets.py::test_the_described_market_matches_the_engine -q
      2026-09-19T07:45:45Z exit 0 in 4s
- [x] 7. Parity: the described market reproduces the live engine's own fair price, costs and settlement on recorded windows, to the last digit where the engine is deterministic | needs: 6 | ctx: scripts/parity.py | gate: python scripts/market_parity.py --check
      2026-09-19T07:49:52Z exit 0 in 5s
      2026-09-19T08:07:05Z exit 1 in 6s
      2026-09-19T08:08:48Z exit 0 in 6s
- [x] 8. A second market, to prove the interface is real: `markets/binance_spot.py` over the klines already on disk — its own clock, one instrument, mid valuation, no settlement, commission costs — run end to end through bars, labels, weights, a feature table and a backtest | needs: 5 | ctx: scripts/market_demo.py | gate: python scripts/market_demo.py --check
      2026-09-19T07:50:00Z exit 0 in 7s
      2026-09-19T08:07:13Z exit 0 in 7s
- [x] 9. Registries for the other three kinds, in the same shape: strategy, model family and feature, so adding one is a file and a decorator rather than edits in several places; the existing sets register themselves and behave identically | needs: 5 | ctx: src/pmlab/registry.py | gate: python -m pytest -p no:warnings tests/test_registry.py -q
      2026-09-19T07:50:01Z exit 0 in 1s
- [x] 10. The hierarchy: instruments carry market → sector → industry → instrument, each level with its own aggregate; the dashboard's inventory and views read it so a level has an overview | needs: 5 | ctx: src/pmlab/markets, src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_markets.py -k hierarchy -q && python scripts/book_display.py --check
      2026-09-19T07:50:04Z exit 0 in 2s
- [x] 11. Hot swap, within what a trading process can safely do: a plugin (feature, strategy, model, market view) is added, replaced or removed while the app runs, taking effect at the next window boundary and never inside a window; the swap is validated before it is admitted (imports, registers, produces a value on a recorded window), refused with its reason if not, recorded in the run log, and reversible by swapping back; registered legs are refused outright until the confirmation's analysis | needs: 9 | ctx: src/pmlab/registry.py, src/pmlab/live | gate: python -m pytest -p no:warnings tests/test_hot_swap.py -q
      2026-09-19T13:45:23Z exit 0 in 11s
      2026-09-19T13:48:34Z exit 0 in 7s
- [x] 12. Written down: docs/markets.md — what every market must supply, what it may choose, the seven differences above, and a worked walk-through of adding one | needs: 8,9,10,11 | gate: test -s docs/markets.md && python -m pytest -p no:warnings tests/test_markets.py -q
      2026-09-19T13:45:34Z exit 0 in 8s
- [ ] 13. (Held until the confirmation's analysis on 2026-10-24.) The live engine reads its market from the registry instead of its constants, with the registered leg behaviour unchanged and an amendment entry recording the swap | needs: 7,12 | risk: irreversible | gate: python scripts/confirm.py --check && python -m pytest -p no:warnings -m 'not integration' tests -q
