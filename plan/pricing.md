# plan: pricing (sanitized copy)

## goal: A pricing layer a quant firm could live with: contracts, models, calibrations and uncertainty as first-class objects that can be changed anywhere without breaking anything

- [x] 1. Analyse | gate: grep -q '^## analysis' plan/pricing.md
      2026-09-19T07:10:10Z exit 0 in 0s
- [x] 2. Research | needs: 1 | gate: grep -q '^## research' plan/pricing.md
      2026-09-19T07:10:10Z exit 0 in 0s
- [x] 3. Plan with acceptance tests | needs: 2 | gate: grep -q '^## plan' plan/pricing.md
      2026-09-19T07:10:10Z exit 0 in 0s
- [x] 4. Finalise | needs: 3 | gate: grep -q '^APPROVED' plan/pricing.md
      2026-09-19T07:10:11Z exit 0 in 0s
- [x] 5. The seam: `src/pmlab/pricing/` with `Contract`, `PricingModel` (declared inputs, validity domain, refusal on stale or out-of-domain inputs), `Calibration` (span, code version, hash) and a registry; no global state, nothing registered imports it yet | needs: 4 | ctx: src/pmlab/pricing | gate: python -m pytest -p no:warnings tests/test_pricing_layer.py -q
      2026-09-19T07:57:09Z exit 0 in 2s
- [x] 6. Today's pricing, expressed in it: the TWAP-digital fair price and the three adjustment models as registered models with their existing frozen calibrations, plus the digital's closed-form sensitivities to the underlying and to volatility | needs: 5 | ctx: src/pmlab/pricing, src/pmlab/fair.py | gate: python -m pytest -p no:warnings tests/test_pricing_layer.py -k digital -q
      2026-09-19T07:57:15Z exit 0 in 1s
- [x] 7. Parity: for every recorded window, the new layer reproduces the live engine's price to the last digit, and refuses in exactly the cases the engine refuses | needs: 6 | ctx: scripts/pricing_check.py | gate: python scripts/pricing_check.py --check --parity
      2026-09-19T07:57:27Z exit 0 in 11s
      2026-09-19T08:31:30Z exit 0 in 11s
- [x] 8. Uncertainty as part of the price: every model returns the band and its reasons through one interface, fed by the existing fair-band machinery; a price without a band fails the check | needs: 6 | ctx: src/pmlab/pricing, scripts/fair_band.py | gate: python scripts/pricing_check.py --check --bands
      2026-09-19T07:57:46Z exit 0 in 19s
      2026-09-19T08:31:51Z exit 0 in 21s
- [x] 9. A second contract, to prove the seam: a linear instrument (spot, from the klines already on disk) priced by a mid model and a drift model, through the same interface, with its own calibration and band | needs: 5 | ctx: src/pmlab/pricing | gate: python -m pytest -p no:warnings tests/test_pricing_layer.py -k spot -q
      2026-09-19T07:57:48Z exit 0 in 1s
- [x] 10. Champion and challenger: the scoreboard records model id and calibration hash per row, scores held-out data with the dependence-aware test, and names which model is live and since when; a challenger can run in shadow with no effect on decisions | needs: 6,9 | ctx: scripts/pricing_scoreboard.py | gate: python scripts/pricing_check.py --check --scoreboard
      2026-09-19T13:45:42Z exit 0 in 7s
- [x] 11. Reproducibility: every published price is traceable to (contract, model id, calibration hash, input vintage), and a replay of any window reproduces it exactly; a missing link fails the check | needs: 7,8 | ctx: src/pmlab/pricing | gate: python scripts/pricing_check.py --check --trace
      2026-09-19T13:46:06Z exit 0 in 21s
      2026-09-19T13:48:56Z exit 0 in 21s
- [x] 12. Written down: docs/pricing.md — the five objects, what a model must declare, how a calibration is recorded and rotated, how uncertainty is reported, and a worked walk-through of adding a model for a new contract | needs: 8,9,10,11 | gate: test -s docs/pricing.md && python scripts/pricing_check.py --check
      2026-09-19T13:46:59Z exit 0 in 52s
- [ ] 13. (Held until 2026-10-24, the confirmation's analysis.) The live engine prices through the registry, with registered leg behaviour unchanged and an amendment entry recording the swap | needs: 11,12 | risk: irreversible | gate: python scripts/confirm.py --check && python -m pytest -p no:warnings -m 'not integration' tests -q
