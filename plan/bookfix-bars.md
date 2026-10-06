# plan: bookfix-bars (sanitized copy)

## goal: Fix AFML book-check findings in bars/labels/alpha_bars/afml_extensions/bar_study (parts 1-3 assigned subset + robustness audit 10-12)

- [x] 1. Acceptance tests written first (labels, bars, afml library, script helpers) | gate: grep -q test_meta_label_keeps_vertical tests/test_labels.py && grep -q test_constructor_default_is_the_activity_anchor tests/test_bars_runs.py && test -s tests/test_bars_docs.py && test -s tests/test_afml_scripts.py && grep -q test_smt_matches tests/test_afml_breaks.py && grep -q test_a_half_with_no_trades tests/test_afml_cpcv.py && grep -q test_two_outcome_formulas tests/test_afml_strategy_risk.py && grep -q test_estimator_bias tests/test_afml_entropy.py && grep -q test_hold_to_expiry tests/test_afml_betsizing.py
      2026-09-17T07:33:06Z exit 2 in 1s
      2026-09-17T07:33:13Z exit 0 in 0s
- [x] 2. labels: barrier/horizon return, book meta-label, CUSUM flag per second | needs: 1 | ctx: src/pmlab/labels.py | gate: python -m pytest tests/test_labels.py -q -p no:warnings
      2026-09-17T07:33:41Z exit 0 in 0s
- [x] 3. bars: activity anchor default, Jensen wording, citations; glossary bar.* and reviews R1 note | needs: 1 | ctx: src/pmlab/bars/runs.py, src/pmlab/bars/imbalance.py, src/pmlab/bars/__init__.py | gate: python -m pytest tests/test_bars_runs.py tests/test_bars_imbalance.py tests/test_bars_standard.py tests/test_bars_docs.py -q -p no:warnings
      2026-09-17T07:42:46Z exit 1 in 2s
      2026-09-17T07:46:24Z exit 0 in 2s
- [x] 4. afml library: cpcv zero-variance half, strategy_risk two-point guard + bootstrap, entropy bias + order imbalance, breaks BDE/Chow/QADF/CADF/SMT, betsizing running mean, citations | needs: 1 | ctx: src/pmlab/afml/cpcv.py, src/pmlab/afml/strategy_risk.py, src/pmlab/afml/entropy.py, src/pmlab/afml/breaks.py, src/pmlab/afml/betsizing.py | gate: python -m pytest tests/test_afml_betsizing.py tests/test_afml_breaks.py tests/test_afml_cpcv.py tests/test_afml_entropy.py tests/test_afml_fracdiff.py tests/test_afml_importance.py tests/test_afml_microstructure.py tests/test_afml_sampling.py tests/test_afml_strategy_risk.py tests/test_afml_synthetic.py -q -p no:warnings
      2026-09-17T07:41:06Z exit 1 in 38s
      2026-09-17T07:42:44Z exit 0 in 41s
- [x] 5. alpha_bars.py: all-event labels + conditional table, CUSUM legacy switch, spans, weighted scoring, no kelly, explore fair params, pinned universe, v2 output | needs: 2 | ctx: scripts/alpha_bars.py | gate: python -m pytest tests/test_labels.py -q -p no:warnings && python scripts/confirm.py --check
      2026-09-17T07:46:54Z exit 0 in 1s
- [x] 6. afml_extensions.py: pinned universe and PBO grid, explore fair params, tb rows, spans, weighted scores, SFI base, bet sizing, strategy risk, entropy, breaks, live-book exits | needs: 4,5 | gate: python -m py_compile scripts/afml_extensions.py
      2026-09-17T07:52:26Z exit 0 in 0s
- [x] 7. shared inputs rebuilt (fair fit, events, seconds) | needs: 5,6 | gate: test -s research/afml_extensions_universe.json && test -s data/cache/afml/alpha_bars_events/summaries_final.json
      2026-09-17T08:00:08Z exit 0 in 0s
- [x] 8. alpha_bars v2 run | needs: 7 | gate: python scripts/alpha_bars.py --check
      2026-09-17T08:57:19Z exit 0 in 1s
- [x] 9. afml_extensions stage reruns | needs: 7 | gate: python scripts/afml_extensions.py --check
      2026-09-17T08:57:23Z exit 0 in 3s
- [x] 10. bar_study constant-size count stability | ctx: scripts/bar_study.py | gate: python -c "import json; d=json.load(open('research/bar_study.json')); assert d['count_stability']['constant_size']"
      2026-09-17T08:57:23Z exit 0 in 0s
- [x] 11. reports rewritten (alpha_bars.md, afml_extensions.md, bar_study.md) | needs: 8,9,10 | gate: python -m pytest tests/test_afml_*.py tests/test_labels.py tests/test_bars_*.py -q -p no:warnings && python scripts/alpha_bars.py --check && python scripts/afml_extensions.py --check && python scripts/confirm.py --check
      2026-09-17T09:13:18Z exit 1 in 45s
      2026-09-17T10:05:58Z exit 0 in 46s
