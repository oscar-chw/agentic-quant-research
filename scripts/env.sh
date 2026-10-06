# Sourced by check.sh and demo.sh. Interpreter: $PYTHON if set (as in CI), else $PORTFOLIO_VENV/bin/python, else
# ./.venv/bin/python. Without one, say how to make it and stop: a missing environment must not read as a pass.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -n "${PORTFOLIO_VENV:-}" ]; then PY="$PORTFOLIO_VENV/bin/python"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else
  echo "error: no Python environment. Create one: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 3
fi
if ! "$PY" -c "import numpy, pandas, scipy, sklearn, pyarrow, pytest" 2>/dev/null; then
  echo "error: $PY lacks the pinned packages: $PY -m pip install -r requirements.txt" >&2
  exit 3
fi
export PYTHONPATH="src"
export PYTHONDONTWRITEBYTECODE=1
