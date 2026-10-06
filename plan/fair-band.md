# plan: fair-band (sanitized copy)

## goal: Show the fair price's probability region: how uncertain Q is and when the model is mismatched or under-fitted

- [x] 1. Analyse: what is actually being asked, and what would prove it done | gate: grep -q '^## analysis' plan/fair-band.md
      2026-09-17T17:09:18Z exit 0 in 0s
- [x] 2. Research: prior art, constraints, unknowns resolved or explicitly deferred | needs: 1 | gate: grep -q '^## research' plan/fair-band.md
      2026-09-17T17:09:19Z exit 0 in 0s
- [x] 3. Plan, with acceptance tests written before any implementation | needs: 2 | gate: grep -q '^## plan' plan/fair-band.md
      2026-09-17T17:09:19Z exit 0 in 0s
- [x] 4. Finalise: scope, gates and done-condition agreed | needs: 3 | gate: grep -q '^APPROVED' plan/fair-band.md
      2026-09-17T17:10:34Z exit 0 in 0s
- [x] 5. Band research: scripts/fair_band.py fits the four layers on data before CONFIRM_START: day-block bootstrap of the live fit (vol_mult, basis; lag and half-life as the fit's grid neighbours), realised/forecast σ quantiles by time left, calibration table by Q bucket × time left with day-clustered intervals, mismatch thresholds; writes research/fair_band.{json,md}; tests on simulated windows with known σ and basis show the parameter band covers the truth at the nominal rate and the calibration band detects an injected bias; --check re-derives the numbers without writing | needs: 3 | ctx: src/pmlab/fair.py, scripts/fit_live_models.py, scripts/vol_study.py, scripts/pricing_scoreboard.py | gate: python -m pytest -p no:warnings tests/test_fair_band.py -q && python scripts/fair_band.py --check
      2026-09-17T17:43:25Z exit 0 in 84s
- [x] 6. Held-out coverage: on the last 3 complete Taiwan days before CONFIRM_START available at run time (node 5 refits without them for this test), how often the settled outcome's frequency sits inside each band, per layer and combined; reported in research/fair_band.md with the verdict per layer (keep, widen, drop) | needs: 5 | ctx: research/fair_band.md | gate: python scripts/fair_band.py --check --coverage
      2026-09-17T17:43:58Z exit 0 in 28s
- [x] 7. Dashboard: the fair price chart shades the parameter + volatility band (default) with toggles for model range and calibration band; the order ticket and window header show Q with its interval; mismatch badges with the reason and the number that tripped them; ⓘ texts from research/fair_band.json; contract test and headless visual check updated; nothing in registered code changes | needs: 4,6 | ctx: src/pmlab/dashboard | gate: python -m pytest -p no:warnings tests/test_dashboard_contract.py tests/test_fair_band.py tests/test_dashboard_fair_band.py -q && python scripts/dashboard_visual.py --check
      2026-09-17T18:26:31Z exit 0 in 136s
- [x] 8. Deployed after the user confirms the restart | needs: 7 | risk: irreversible | gate: python scripts/deploy.py --check
      2026-09-18T02:40:50Z exit 0 in 0s
