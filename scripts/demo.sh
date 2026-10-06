#!/usr/bin/env bash
# Three of the book's warnings on SYNTHETIC data (scripts/demo.py). Exits non-zero if a known answer is missed.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
"$PY" scripts/demo.py
