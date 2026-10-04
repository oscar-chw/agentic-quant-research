# Historical source template

Copy `manifest.template.json`, replace every placeholder, and use the exact family-control names from `src/qrae/historical_data.py`. Paths are relative to the `--workspace` root.

Compute each file's SHA-256 and exact byte size without pasting file contents into Codex. On PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath data/prices.csv
(Get-Item -LiteralPath data/prices.csv).Length
```

The entitlement ID field stores only a SHA-256 of a non-secret local identifier. Never place credentials, signed URLs, tokens, raw private keys, or account secrets in the manifest or evidence files.

The template is intentionally invalid until all hashes and sizes are replaced. Passing import verifies mechanical integrity only; it does not establish legal permission or E1 research evidence.
