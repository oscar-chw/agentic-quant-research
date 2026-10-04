# Sourced by check.sh and demo.sh: run every package from this source tree, no install.
# native/build/python holds the optional C++ module once `make native-build` has run.
# Interpreter: $PYTHON if set (an explicit override, as in CI), else
# $PORTFOLIO_VENV/bin/python, else ./.venv/bin/python. It needs the pins in
# requirements.txt; without an environment this says how to make one and stops.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -n "${PORTFOLIO_VENV:-}" ]; then PY="$PORTFOLIO_VENV/bin/python"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else
  echo "error: no Python environment. Either set PORTFOLIO_VENV to a venv that has requirements.txt," >&2
  echo "or create one here: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 3
fi
if ! "$PY" -c "import pytest, hypothesis" 2>/dev/null; then
  echo "error: $PY lacks pytest or hypothesis: $PY -m pip install -r requirements.txt" >&2
  exit 3
fi
export PYTHONPATH="packages/research/src:packages/marketdata:packages/marketdata/native/build/python:packages/vault:packages/factor/src:packages/imc-sim/src:apps/quantos/src"
export PYTHONDONTWRITEBYTECODE=1
# Critic receipt keys go to a throwaway directory unless the caller chose one,
# so checks and demos never write into the user profile.
export QRAE_RECEIPT_KEY_ROOT="${QRAE_RECEIPT_KEY_ROOT:-$(cd "$(mktemp -d)" && pwd -P)/receipt-keys}"
