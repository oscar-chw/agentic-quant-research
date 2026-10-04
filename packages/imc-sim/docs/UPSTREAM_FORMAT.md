# Source identity and reuse boundary

Inspected 2026-09-14: community repository [nabayansaha/imc-prosperity-4-backtester](https://github.com/nabayansaha/imc-prosperity-4-backtester), revision `0094c681f8cd019889761e6431a1a47ea151aaa8`. The authors identify it as a Prosperity 4 backtester based on the earlier Prosperity 3 project; the repository/package selection is therefore explicit. It is not an IMC-published official parser. Its README uses the distribution/CLI name `prosperity4btest` while its Python directory is `prosperity4bt`.

Exactly five files were retrieved at that revision. No strategy, market-data archive or installed upstream package was used.

| Source | SHA-256 | What was inspected |
|---|---|---|
| [README.md](https://github.com/nabayansaha/imc-prosperity-4-backtester/blob/0094c681f8cd019889761e6431a1a47ea151aaa8/README.md) | `a2a7f52ccf78f053bd7add46917a141c3a39eecd0e80c6cac105094dc5a91706` | Producer identity and one-day/custom-output CLI usage; unsupported conversions and fee-assumption boundary |
| [LICENSE](https://github.com/nabayansaha/imc-prosperity-4-backtester/blob/0094c681f8cd019889761e6431a1a47ea151aaa8/LICENSE) | `7de97c04be5fd70fa8f9a51be190c972138ec7f95da8cd15f5a2319791b4d0c0` | Repository declares MIT, copyright 2024 Jasper van Merle. Its notice conditions apply to reused upstream code. |
| [models.py](https://github.com/nabayansaha/imc-prosperity-4-backtester/blob/0094c681f8cd019889761e6431a1a47ea151aaa8/prosperity4bt/models.py) | `2b54639bf7055ebbb8b33a5c164926820425034e1dbd88ff4c3435dbe6b7afdf` | Activity CSV and trade field serialization, XIREC currency spelling, trailing object comma; no exported unique fill ID or fee |
| [runner.py](https://github.com/nabayansaha/imc-prosperity-4-backtester/blob/0094c681f8cd019889761e6431a1a47ea151aaa8/prosperity4bt/runner.py) | `6b67f58b0ae731afe656f5bcb0900fe3024aad5a5c29f3493d9a60151c78e0ed` | Sampled activity construction, activity-before-matching flow, own buyer/seller markers and inclusion of residual market trades |
| [__main__.py](https://github.com/nabayansaha/imc-prosperity-4-backtester/blob/0094c681f8cd019889761e6431a1a47ea151aaa8/prosperity4bt/__main__.py) | `9275ed65454ffcd591a1ec088ae69d8391b1e611717b1cba104a99a69ae419a0` | Sampled output writer: section delimiters/17-column header/trade array; merged-day timestamp and P&L offsets |

Our parser is independently authored from these observed format facts. This application distributes no upstream strategy, implementation file or market dataset. The example contains invented instruments, prices and fills; it is original project material, not a redistributed team result. No new project-wide license grant is made here.

Verification establishes the observed serializer contract and tests original format-shaped fixtures against it. The upstream backtester was not executed, and no original user/team log or independently sourced real run was available. We therefore qualify the **pinned community-format import**, not complete engine parity or every similarly named package. Raw logs lack a version header; `producer_revision` is declared configuration, not authenticated provenance.
