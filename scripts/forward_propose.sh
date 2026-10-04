#!/usr/bin/env bash
# Protocol v2, LLM arm, steps 1 and 2 (results/forward-2026-09/protocol.json). Needs a logged-in
# `claude` CLI and no market data. Uses at most 4 of the arm's 12 live calls:
#   1. smoke: the loop live on the SYNTHETIC control panel (at most 3 calls), to show the
#      proposer and critic return schema-valid output through the real transport;
#   2. the real proposals alone (1 call, no prices read), so they exist before this
#      repository fetches any bar of the forward window.
# Both sessions are recorded as REAL LLM OUTPUT replays under apps/quantos/examples/loop/.
# Commit them before fetching any bar after 2026-08-31.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
R=results/forward-2026-09
E=apps/quantos/examples/loop
SMOKE="$E/replay.live-synthetic.json"
PROPOSALS="$E/replay.forward-2026-09.json"
MODEL="$("$PY" -c 'import json; print(json.load(open("results/forward-2026-09/protocol.json"))["arms"]["llm"]["model"])')"
for f in "$SMOKE" "$PROPOSALS"; do
  if [ -e "$f" ]; then echo "error: $f exists; recorded sessions are never overwritten" >&2; exit 2; fi
done
STATUS="$({ claude auth status 2>/dev/null || true; } | "$PY" -c 'import json, sys; print(json.load(sys.stdin).get("loggedIn"))' 2>/dev/null || true)"
if [ "$STATUS" != True ]; then echo "error: the claude CLI is not logged in (claude auth status)" >&2; exit 4; fi
OUT="$(cd "$(mktemp -d)" && pwd -P)"
claude --version > "$OUT/cli-version.txt"

echo "== 1. smoke: live loop on the SYNTHETIC control panel (at most 3 calls)"
"$PY" -m factor_research fixture "$OUT/control" --assets 8 --dates 60 --persistent >/dev/null
"$PY" -m quantos_showcase.loop run --campaign "$E/campaign.json" --prices "$OUT/control/panel.csv" \
  --contract "$E/synthetic-contract.json" --vault "$E/vault" --store "$OUT/loop" --run-id smoke \
  --receipt-keys "$OUT/keys" --live --model "$MODEL" --max-calls 3 --record "$SMOKE"
"$PY" -m quantos_showcase.loop verify --run "$OUT/loop/smoke" --receipt-keys "$OUT/keys"

echo "== 2. the real proposals, before this repository fetches any forward bar (1 call)"
"$PY" -m quantos_showcase.loop propose --campaign "$R/campaign.json" --experiment "$R/experiment.json" \
  --vault "$E/vault" --live --model "$MODEL" --max-calls 1 --record "$PROPOSALS"
echo "CLI: $(cat "$OUT/cli-version.txt"). Now commit $SMOKE and $PROPOSALS."
