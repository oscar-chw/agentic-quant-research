#!/usr/bin/env bash
# The whole gate: every published test, the claim ledger against those results, the plan-graph statistics against
# docs/evidence.md, then the demo. Exits 0 only if all of it passes; a test run that collects nothing fails (exit 5).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== tests"
"$PY" -m pytest -p no:cacheprovider -rs --junitxml="$out/junit.xml" tests | tail -n 12
echo "== claims: every claim's tests, as run above, match docs/evidence.md"
"$PY" scripts/claims.py --junit "$out/junit.xml" --check docs/evidence.md
echo "== plan graph: statistics from plan/ match docs/evidence.md"
"$PY" scripts/plan_stats.py --check docs/evidence.md
echo "== demo"
bash scripts/demo.sh
