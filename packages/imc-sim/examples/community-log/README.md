# Original synthetic import case

These files were authored for this project. Instruments and numerical records are invented; no upstream strategy or market data is included. The section/header layout follows the [pinned community writer](../../docs/UPSTREAM_FORMAT.md). This is not a historical IMC4 result.

From the monorepo root, after `make install` (or use `$PY -m imc4_analysis` after `source scripts/env.sh`):

```sh
imc4-analyze import-log packages/imc-sim/examples/community-log/sample.log --config packages/imc-sim/examples/community-log/import.json --out imported-run
```

Expected own fills: buy ALPHA 2 at 101; sell BETA 1 at 50; sell ALPHA 1 at 105; buy BETA 1 at 48. Each has the explicitly configured fee 1. The two trades whose buyer and seller are other participants are market-only and excluded.

Cash ledger: 1000 → 797 → 846 → 950 → 901. Final inventory is ALPHA 1 and BETA 0; mark ALPHA 102 gives equity 1003 and marked P&L 3. The first buy breaches ALPHA limit 1. At a horizon of 100 source-timestamp units, gross markouts are 6, 2, 3 and unavailable future. Source `profit_and_loss` cells deliberately contain zero to demonstrate that they are not used as the accounting oracle.

Review every setting before replacing this fixture with your own log. In particular, do not inherit the synthetic fee or opening cash as observed facts.
