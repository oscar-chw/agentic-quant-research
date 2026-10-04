#!/usr/bin/env bash
# Every package's tests, the native C++ path when its toolchain is present (else
# SKIPPED with the reason), then the offline demo with --quotes, so both of its
# paths run. Exits 0 only if all that ran passed; a suite that collects no tests
# fails (pytest exit 5).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
unset ASOF_REPLAY_NATIVE  # the suites test the pure-Python default; the native stage opts in itself
for suite in packages/research/tests packages/marketdata/tests packages/vault packages/factor/tests \
             packages/imc-sim/tests apps/quantos/tests tests; do
  echo "== $suite"
  "$PY" -m pytest -q -p no:cacheprovider -rs "$suite" | tail -n 4
done
echo "== packages/marketdata/native (optional C++ hot path)"
if ! command -v clang++ >/dev/null; then
  echo "SKIPPED: clang++ not found"
elif ! "$PY" -c "import pybind11, hypothesis" 2>/dev/null; then
  echo "SKIPPED: $PY lacks pybind11 or hypothesis (pinned in requirements.txt)"
else
  make native-build native-test PYTHON="$PY"
fi
echo "== demo, with the separate quote-imbalance trial"
bash scripts/demo.sh --quotes
