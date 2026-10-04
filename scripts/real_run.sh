#!/usr/bin/env bash
# A pre-registered real-data study, end to end. STUDY names its directory (default
# results/real-2026-10, protocol v1; results/forward-2026-09 is v2). ASOF_DATA_DIR must hold
# the Binance daily CSVs named in the protocol's universe file (scripts/fetch_binance_daily.py
# writes that format). About 2.5 minutes with 8 worker processes.
#   scripts/real_run.sh          grid, random-K and baselines; the LLM arm replays the committed
#                                REAL LLM OUTPUT if it exists, else it is recorded as pending
#   scripts/real_run.sh --live   the LLM loop live: replayed keys first (protocol v2 commits the
#                                proposals in advance), claude -p for the rest, within the budget
# Writes $STUDY/{hypotheses,ablation}.json and REPORT.md, refusing to overwrite them
# (git rm them to rerun; history keeps the earlier ones). WORK_DIR holds the trials.
# A protocol with a delisting rule (v2) first extends every pair to the test end at its last
# close (the list goes to $STUDY/filled-pairs.json). DELISTED=drop is the robustness run
# (amendment 3): the unfilled data, so a pair is absent only on the dates it lacks and the frozen
# selections still reproduce; it writes to $STUDY/robustness-dropped/, with the LLM arm's pick read
# from that run's per-hypothesis table rather than re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
: "${ASOF_DATA_DIR:?set ASOF_DATA_DIR to the Binance daily CSV directory}"
R="${STUDY:-results/real-2026-10}"
W="${WORK_DIR:-work/$(basename "$R")}"
REPLAY="apps/quantos/examples/loop/replay.$(basename "$R").json"
field() { "$PY" -c 'import json, sys; d = json.load(open(sys.argv[1]))
for k in sys.argv[2:]: d = d[k]
print(d)' "$R/protocol.json" "$@"; }
UNIVERSE="$(field universe file)"
if [ -e "$W" ]; then echo "error: $W exists; trials are never overwritten (set WORK_DIR)" >&2; exit 2; fi
mkdir -p "$W"

"$PY" -m quantos_showcase.ablation verify-data --data "$ASOF_DATA_DIR" --universe "$UNIVERSE"
DATA="$ASOF_DATA_DIR"
OUT="$R"
if "$PY" -c 'import json, sys; sys.exit("delisting" not in json.load(open(sys.argv[1]))["universe"])' "$R/protocol.json"; then
  THROUGH="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["splits"]["test"]["end"][:10])' "$R/experiment.json")"
  if [ "${DELISTED:-fill}" = drop ]; then
    OUT="$R/robustness-dropped"
  else
    MANIFEST="$R/filled-pairs.json"
    if [ -e "$MANIFEST" ]; then echo "error: $MANIFEST exists" >&2; exit 2; fi
    "$PY" -m quantos_showcase.ablation fill --data "$DATA" --through "$THROUGH" --out "$W/data" > "$W/filled.json"
    cp "$W/filled.json" "$MANIFEST"
    DATA="$W/data"
  fi
fi
"$PY" -m factor_research from-ohlcv --dir "$DATA" --experiment "$R/experiment.json" --out "$W/import" >/dev/null
LOOP=(--campaign "$R/campaign.json" --prices "$W/import/prices.csv" --contract "$W/import/import-contract.json"
      --vault apps/quantos/examples/loop/vault --store "$W/loop" --run-id llm --receipt-keys "$W/receipt-keys")
LLM=(--llm-run "$W/loop/llm" --receipt-keys "$W/receipt-keys")

# `claude auth status` exits nonzero when logged out; its JSON still says why.
login() { { claude auth status 2>/dev/null || true; } \
          | "$PY" -c 'import json, sys; print(json.load(sys.stdin).get("loggedIn"))' 2>/dev/null || echo unavailable; }
if [ "$OUT" != "$R" ]; then
  LLM=(--llm-pending "robustness run: the LLM arm is scored in the primary run; its pick's row is in hypotheses.json here")
elif [ "${1:-}" = --live ]; then
  if [ "$(login)" != True ]; then echo "error: the claude CLI is not logged in (claude auth status)" >&2; exit 4; fi
  if [ "$(field schema)" = asof-ablation-protocol/v2 ]; then BUDGET="$(field arms llm live_call_budget critic)"
  else BUDGET="$(field arms llm max_live_calls)"; fi
  PRIOR=(); [ -f "$REPLAY" ] && PRIOR=(--replay "$REPLAY")
  # Recorded under WORK_DIR first: a failed session must not become the committed replay.
  if "$PY" -m quantos_showcase.loop run "${LOOP[@]}" ${PRIOR[@]+"${PRIOR[@]}"} --live --model "$(field arms llm model)" \
       --max-calls "$BUDGET" --record "$W/replay.json" 2>"$W/live.err"; then
    cp "$W/replay.json" "$REPLAY.new" && mv "$REPLAY.new" "$REPLAY"
  else
    tail -n 3 "$W/live.err" >&2
    LLM=(--llm-pending "live LLM run failed on $(date -u +%F), live calls stopped: $(tail -n 1 "$W/live.err")")
  fi
elif [ -f "$REPLAY" ] && "$PY" -m quantos_showcase.loop run "${LOOP[@]}" --replay "$REPLAY" >/dev/null 2>"$W/replay.err"; then
  :
else
  LLM=(--llm-pending "no complete LLM run: claude auth status reported loggedIn=$(login) on $(date -u +%F)")
fi
"$PY" -m quantos_showcase.ablation run --protocol "$R/protocol.json" --prices "$W/import/prices.csv" \
  --contract "$W/import/import-contract.json" --work "$W/trials" --out "$OUT" "${LLM[@]}"
"$PY" -m quantos_showcase.ablation show --results "$OUT"
