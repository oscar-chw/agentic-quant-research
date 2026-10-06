#!/usr/bin/env bash
# Offline demo, a few seconds: the LLM research loop on two SYNTHETIC panels
# with the hand-written replay fixture, then the committed real-data table.
# With ASOF_DATA_DIR set to the Binance daily CSVs, it first verifies their
# checksums and recomputes every committed number independently.
# --quotes also runs the quote-imbalance method trial, a separate SYNTHETIC
# experiment outside the research loop (docs/quote-imbalance-walkthrough.md).
# Usage: scripts/demo.sh [--quotes] [new-output-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
QUOTES=0
if [ "${1:-}" = --quotes ]; then QUOTES=1; shift; fi
# Resolved, because the workflows refuse paths through symlinks (macOS /var is one).
OUT="${1:-$(cd "$(mktemp -d)" && pwd -P)/demo}"
if [ -e "$OUT" ]; then echo "error: $OUT exists; pass a new directory" >&2; exit 2; fi
LOOP=apps/quantos/examples/loop

echo "== SYNTHETIC panels: 8 assets x 60 days; 'control' keeps its trend, 'regime' flips it in the test split"
"$PY" -m factor_research fixture "$OUT/panels/control" --assets 8 --dates 60 --persistent >/dev/null
"$PY" -m factor_research fixture "$OUT/panels/regime" --assets 8 --dates 60 >/dev/null

for run in control regime; do
  echo "== research loop '$run' (LLM transport: HAND-WRITTEN FIXTURE replay)"
  "$PY" -m quantos_showcase.loop run --campaign "$LOOP/campaign.json" --prices "$OUT/panels/$run/panel.csv" \
    --contract "$LOOP/synthetic-contract.json" --vault "$LOOP/vault" --replay "$LOOP/replay.hand-written.json" \
    --store "$OUT/loop" --run-id "$run" --receipt-keys "$OUT/receipt-keys" >/dev/null
done

# The filter and critic saw validation only, so both panels send H1 to the gate. The gate is the
# first place the test split is shown: test rank IC 1.000 on 'control', -0.895 on 'regime'.
echo "== human gate (scripted for the demo): promote H1 on 'control', reject it on 'regime' after seeing test"
"$PY" -m quantos_showcase.loop gate --run "$OUT/loop/control" --promote H1 >/dev/null
"$PY" -m quantos_showcase.loop gate --run "$OUT/loop/regime" --reject H1 >/dev/null
for run in control regime; do
  "$PY" -m quantos_showcase.loop verify --run "$OUT/loop/$run" --receipt-keys "$OUT/receipt-keys"
  "$PY" tools/agent-review/validate_envelope.py "$OUT/loop/$run/envelope.json"
done

echo "== loop reports (test shows as 'sealed' for every card that did not reach the gate):"
echo "   $OUT/loop/control/REPORT.md $OUT/loop/regime/REPORT.md"

echo "== REAL data: the pre-registered Binance study (results/real-2026-10)"
if [ -z "${ASOF_DATA_DIR:-}" ]; then
  echo "(set ASOF_DATA_DIR to the 34 Binance daily CSVs to verify their checksums and recompute every number below)"
else
  "$PY" -m quantos_showcase.ablation verify-data --data "$ASOF_DATA_DIR" --universe fixtures/binance_universe.json
  if "$PY" -c "import pandas" 2>/dev/null; then
    "$PY" scripts/crosscheck_real.py --data "$ASOF_DATA_DIR" --results results/real-2026-10
  else
    echo "independent recomputation skipped: $PY lacks pandas (pinned in requirements.txt)"
  fi
fi
"$PY" -m quantos_showcase.ablation show --results results/real-2026-10
echo "This is v1's frozen output. Its deflated-Sharpe and Holm columns are superseded: see"
echo "results/real-2026-10/README.md and posthoc-dsr.json (Newey-West t; deflated Sharpe 0.46-0.63)."

if [ "$QUOTES" = 1 ]; then
  echo "== separate experiment: quote-imbalance method trial (SYNTHETIC archive; a retained negative result)"
  "$PY" -m quantos_showcase.methods run --card apps/quantos/examples/methods/card.json \
    --archive apps/quantos/examples/feed/archive.jsonl --spec apps/quantos/examples/feed/spec.json \
    --store "$OUT/quotes" --trial-id imbalance >/dev/null
  "$PY" -m quantos_showcase.methods verify --store "$OUT/quotes" --trial-id imbalance
  echo "== quote trial report: $OUT/quotes/methods/runs/imbalance/REPORT.md"
fi
