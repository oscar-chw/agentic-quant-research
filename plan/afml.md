# plan: afml (sanitized copy)

## goal: Implement AFML ch.4,5,8,10,11-13,15,17,18,19 extensions under src/pmlab/afml with tests and research/afml_extensions.md on pre-confirmation data

- [x] 5. Ch.5 fractional differentiation + ADF | ctx: src/pmlab/afml/fracdiff.py | gate: python -m pytest tests/test_afml_fracdiff.py -q -p no:warnings
      2026-09-17T04:41:08Z exit 0 in 0s
- [x] 6. Ch.4 indicator matrix, sequential bootstrap, return and time-decay weights | ctx: src/pmlab/afml/sampling.py | gate: python -m pytest tests/test_afml_sampling.py -q -p no:warnings
      2026-09-17T04:41:08Z exit 0 in 0s
- [x] 7. Ch.8 MDI, MDA, SFI, clustered MDA | ctx: src/pmlab/afml/importance.py | gate: python -m pytest tests/test_afml_importance.py -q -p no:warnings
      2026-09-17T04:41:11Z exit 0 in 2s
- [x] 8. Ch.10 bet sizing | ctx: src/pmlab/afml/betsizing.py | gate: python -m pytest tests/test_afml_betsizing.py -q -p no:warnings
      2026-09-17T04:41:12Z exit 0 in 1s
- [x] 9. Ch.11/12 CPCV and PBO via CSCV | ctx: src/pmlab/afml/cpcv.py | gate: python -m pytest tests/test_afml_cpcv.py -q -p no:warnings
      2026-09-17T04:41:12Z exit 0 in 0s
- [x] 10. Ch.13 OU calibration and optimal trading rules | ctx: src/pmlab/afml/synthetic.py | gate: python -m pytest tests/test_afml_synthetic.py -q -p no:warnings
      2026-09-17T04:41:13Z exit 0 in 0s
- [x] 11. Ch.15 strategy risk | ctx: src/pmlab/afml/strategy_risk.py | gate: python -m pytest tests/test_afml_strategy_risk.py -q -p no:warnings
      2026-09-17T04:41:14Z exit 0 in 0s
- [x] 12. Ch.17 structural breaks (CSW CUSUM, SADF) | ctx: src/pmlab/afml/breaks.py | gate: python -m pytest tests/test_afml_breaks.py -q -p no:warnings
      2026-09-17T04:41:20Z exit 0 in 6s
- [x] 13. Ch.18 entropy estimators and encodings | ctx: src/pmlab/afml/entropy.py | gate: python -m pytest tests/test_afml_entropy.py -q -p no:warnings
      2026-09-17T04:41:21Z exit 0 in 0s
- [x] 14. Ch.19 Hasbrouck λ, Beckers–Parkinson, VPIN with BVC | ctx: src/pmlab/afml/microstructure.py | gate: python -m pytest tests/test_afml_microstructure.py -q -p no:warnings
      2026-09-17T04:41:22Z exit 0 in 1s
- [x] 15. Empirical run on pre-confirmation data and research/afml_extensions.md | needs: 5,6,7,8,9,10,11,12,13,14 | ctx: scripts/afml_extensions.py, research/afml_extensions.md | gate: python scripts/afml_extensions.py --check && test -s research/afml_extensions.md
      2026-09-17T05:20:13Z exit 0 in 1s
