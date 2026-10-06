# What the frozen quote study shows

This is one deterministic synthetic study, not market evidence. Reproduce all 18 policy/scenario/execution runs with `imc4-analyze quote-study --out quote-study`; see [the derivation and execution contract](QUOTE_POLICY.md). Every result remains in the generated comparison, including zero-fill cases and losses.

Version 0.3.0; fixture SHA-256 `b5ae49724a60025a1b2db3220d8cc1d930be24a764ab6f5814d2320979f31ea1`. The measured installed Python-source fingerprint is `10b56edc29bae37065f5af8e12e48cb4168510542bebcd0c2fcebe9bda89d725`. Code fingerprints exclude documentation, metadata, interpreter and fixture; the fixture has its own commitment.

## Inventory control gives up opportunities

| Path / execution | Symmetric P&L | Adjusted P&L | Symmetric / adjusted peak inventory | Symmetric / adjusted inventory-square time |
|---|---:|---:|---:|---:|
| Rising / immediate | 34.6 | −0.6 | 6 / 2 | 116 / 28 |
| Falling / immediate | −9.4 | −4.6 | 6 / 2 | 116 / 28 |
| Rising / queue 2 | 23.2 | 1.7 | 4 / 1 | 44 / 7 |
| Falling / queue 2 | −8.8 | −0.3 | 4 / 1 | 44 / 7 |
| Rising / one-tick cancellation | 26.4 | −14.6 | 6 / 2 | 121 / 29 |
| Falling / one-tick cancellation | −9.4 | −10.6 | 6 / 2 | 116 / 29 |

P&L includes the declared .1 fee per executed unit and values residual inventory at the terminal mark. Peak inventory is absolute units; inventory-square time is units² × ticks. These are exact path results, with no sampling-based confidence interval or aggregate ranking.

In immediate execution the inventory-adjusted policy has less exposure. That helps on the falling path but forgoes directional gains on the rising path. The strategy can still become short after it sells its earlier long inventory, and its inventory penalty shrinks near the horizon. A reservation-price shift is not a target-flat controller or liquidation instruction.

The hand case verifies the accounting independently: symmetric ends at cash900.9 plus one unit marked98, giving equity998.9 and P&L−1.1; adjusted ends flat with cash/equity1000.8 and P&L.8. With a queue haircut of two units, its one-unit demand creates no fills for either policy. A price satisfying a quote is therefore not sufficient execution evidence even in this deliberately simple model.

## Cancellation latency changes the economics

The rising adjusted trace explains its −14.6 result. Both policies initially buy; the adjusted policy buys two units at 99. At tick4, an old ask at 99 remains pending cancellation and sells two units, while a newer ask at101 sells one more. Inventory changes from +2 to −1. At tick5, the remaining pending ask at101 sells one additional unit, taking inventory to −2. The terminal mark is108. Those stale offers remove the potential directional gain and create an unfavorable short exposure.

All orders still obey the hard ±6 bound because the policy reserves pending quantities. **Feasibility did not guarantee desirable execution.** Eliminating that risk would require another control choice—such as waiting for cancellation acknowledgement before placing a replacement or sizing toward a target inventory—and a new frozen comparison. No such change was tuned into these results after seeing the loss.

## Engineering evidence and limits

The actual installed command completed 18 runs, 126 policy calls (including 18 terminal no-order decisions), 198 normalized events and 94 output files in approximately .641 seconds on CPython3.9.6/macOS arm64. The output occupied 993,126 serialized bytes. This timing includes interpreter startup, validation, source fingerprints, execution, analysis, HTML rendering, serialization and file writes; it is not per-decision latency, a cold-disk guarantee or a speedup claim.

The center/capacity calculation is fixed arithmetic per call; execution additionally sorts the bounded set of live orders by price/time priority. We retain full traces at this small admitted workload rather than compress away partial or delayed fills. The existing analyzer supplies exact Decimal accounting and retrospective markouts, so there is one monetary implementation to check. All 64 installed tests passed, including the earlier 44 analyzer/import tests; the old normalized demo stays at −32 and the imported hand example stays at +3.

No implementation test failed in this study. Research tooling initially could not screenshot the primary PDF through the web renderer; a read-only local PDF render confirmed the displayed equations. Earlier browser URL restrictions on local HTML remain a visual-review limitation, not a reason to infer that rendering was checked. Observed-market execution data, real queue calibration and a restartable service remain separate gates. The missing original IMC4 team source is not reconstructed by this study.
