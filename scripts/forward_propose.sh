#!/usr/bin/env bash
# Protocol v2, LLM arm, steps 1 and 2 (results/forward-2026-09/protocol.json). Needs an OpenRouter
# key (OPENROUTER_API_KEY) and no market data; the model is the protocol's pinned open-weight id.
# Uses at most 4 of the arm's 12 live calls:
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
# The key comes from OPENROUTER_API_KEY, else from ~/.config/openrouter/api_key. Never printed.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -r "$HOME/.config/openrouter/api_key" ]; then
  OPENROUTER_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/openrouter/api_key")"
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then echo "error: OPENROUTER_API_KEY is not set (nor ~/.config/openrouter/api_key)" >&2; exit 4; fi
export OPENROUTER_API_KEY
OUT="$(cd "$(mktemp -d)" && pwd -P)"

echo "== 1. smoke: live loop on the SYNTHETIC control panel (at most 3 calls)"
"$PY" -m factor_research fixture "$OUT/control" --assets 8 --dates 60 --persistent >/dev/null
"$PY" -m quantos_showcase.loop run --campaign "$E/campaign.json" --prices "$OUT/control/panel.csv" \
  --contract "$E/synthetic-contract.json" --vault "$E/vault" --store "$OUT/loop" --run-id smoke \
  --receipt-keys "$OUT/keys" --live --model "$MODEL" --max-calls 3 --record "$SMOKE"
"$PY" -m quantos_showcase.loop verify --run "$OUT/loop/smoke" --receipt-keys "$OUT/keys"

echo "== 2. the real proposals, before this repository fetches any forward bar (1 call)"
"$PY" -m quantos_showcase.loop propose --campaign "$R/campaign.json" --experiment "$R/experiment.json" \
  --vault "$E/vault" --live --model "$MODEL" --max-calls 1 --record "$PROPOSALS"
echo "Model: $MODEL on OpenRouter (each response's model, provider and id are in the replay provenance)."
echo "Now commit $SMOKE and $PROPOSALS."
