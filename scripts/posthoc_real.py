#!/usr/bin/env python3
"""POST-HOC checks on the v1 real-data results, written after an independent review of v1; not pre-registered.

    python scripts/posthoc_real.py --data "$ASOF_DATA_DIR" --results results/real-2026-10 [--out NAME]

posthoc.json was written with the first version of the deflated-Sharpe report;
posthoc-dsr.json with the second (null sampling variance with Li-Ji and Nyholt
effective N as the headline, double-counting specifications flagged).

Reads the committed v1 rows, rebuilds the daily rank-IC series from the raw CSVs
(as scripts/crosscheck_real.py does) and writes posthoc.json beside them: Newey-West
t-stats and autocorrelations of each pick's IC series, the deflated Sharpe under every
defensible input, selection on validation NET instead of IC, the sign disagreement
between validation IC and net, the loop card rule's pass counts, and the range of
outcomes the pending LLM arm could produce. It never rewrites a v1 file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscheck_real import load_closes  # noqa: E402

from quantos_showcase import ablation  # noqa: E402

SPLITS = ("validation", "test")


def ic_series(closes, forward, signal, lookback, lo, hi):
    sign = 1 if signal == "momentum" else -1
    feature = (closes.shift(1) / closes.shift(1 + lookback) - 1) * sign
    return feature.rank(axis=1).corrwith(forward.rank(axis=1), axis=1).loc[lo:hi].tolist()


def pick_stats(xs):
    m = ablation.moments(xs)
    return {"ic_mean": m["mean"], "t": m["t"], "t_newey_west_lag5": ablation.newey_west_t(xs, 5),
            "t_newey_west_lag20": ablation.newey_west_t(xs, 20),
            "autocorrelation": {str(k): ablation.autocorrelation(xs, k) for k in (1, 2, 5, 10)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="posthoc.json", help="file name inside --results (never overwritten)")
    args = parser.parse_args()
    out = Path(args.results) / args.out
    if out.exists():
        raise SystemExit(f"{out} exists; post-hoc outputs are never overwritten")
    closes, forward = load_closes(args.data)
    result = json.load(open(os.path.join(args.results, "ablation.json")))
    rows = {r["id"]: r for r in json.load(open(os.path.join(args.results, "hypotheses.json")))}
    spans = {s: (v["first_decision"][:10], (pd.Timestamp(v["last_label"][:10]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
             for s, v in result["splits"].items()}
    series = {s: {i: ic_series(closes, forward, r["signal"], r["lookback"], *spans[s]) for i, r in rows.items()}
              for s in SPLITS}
    for i, r in rows.items():  # the rebuilt series must be the one the study scored
        for s in SPLITS:
            if abs(ablation.moments(series[s][i])["mean"] - r[s]["ic_mean"]) > 1e-12:
                raise SystemExit(f"{i} {s}: rebuilt IC series differs from the committed row")
    ids = list(rows)
    best_net = max(b["test"]["net_mean"] for b in result["baselines"].values())
    arms = {}
    for arm in result["arms"]:
        if arm["status"] != "RUN":
            continue
        chosen = arm["selected"]
        by_net = max(arm["hypotheses"], key=lambda i: rows[i]["validation"]["net_mean"])
        arms[arm["arm"]] = {
            "selected": chosen,
            "validation_ic": pick_stats(series["validation"][chosen]),
            "test_ic": pick_stats(series["test"][chosen]),
            "deflated_sharpe": ablation.deflation(chosen, arm["hypotheses"], rows, series["validation"]),
            "if_selected_on_validation_net": {
                "selected": by_net, "validation_net_mean": rows[by_net]["validation"]["net_mean"],
                "test_net_mean": rows[by_net]["test"]["net_mean"], "test_ic_mean": rows[by_net]["test"]["ic_mean"],
                "promoted_by_v1_gate": ablation.gate_passes(rows[by_net]["test"], best_net)},
        }
    momentum = [r for r in rows.values() if r["signal"] == "momentum"]
    v = lambda r, k: r["validation"][k]
    card = lambda r, s, net: r[s]["ic_mean"] >= 0.01 and r[s][net] > 0
    body = {
        "label": "POST-HOC: computed after the v1 results and their review; not pre-registered",
        "newey_west": "Bartlett kernel; lag 5 and lag 20",
        "arms": arms,
        "sign_disagreement": {
            "momentum_lookbacks": len(momentum),
            "validation_ic_and_net_opposite_sign": sum((v(r, "ic_mean") > 0) != (v(r, "net_mean") > 0) for r in momentum),
            "validation_ic_negative_net_positive": sum(v(r, "ic_mean") < 0 < v(r, "net_mean") for r in momentum),
            "correlation_validation_ic_vs_net": pd.Series([v(r, "ic_mean") for r in momentum]).corr(
                pd.Series([v(r, "net_mean") for r in momentum])),
        },
        "positive_net": {
            "test": sorted(i for i, r in rows.items() if r["test"]["net_mean"] > 0),
            "validation_and_test": sorted(i for i, r in rows.items()
                                          if r["test"]["net_mean"] > 0 and r["validation"]["net_mean"] > 0),
            "test_sleeve": sorted(i for i, r in rows.items() if r["test"]["sleeve_net_mean"] > 0),
            "validation_sleeve": sorted(i for i, r in rows.items() if r["validation"]["sleeve_net_mean"] > 0),
        },
        "loop_card_rule_passes": {
            "rule": "mean rank IC >= 1/100 and net > 0 (the campaign rule), counted per split",
            "validation_sleeve_net": sum(card(r, "validation", "sleeve_net_mean") for r in rows.values()),
            "test_sleeve_net": sum(card(r, "test", "sleeve_net_mean") for r in rows.values()),
            "validation_rebalanced_net": sum(card(r, "validation", "net_mean") for r in rows.values()),
        },
        "llm_arm_v1_outcome_range": dict(
            ablation.outcome_range(ids, rows, 8, best_net),
            note="Under v1 the LLM selection also had to pass the loop's card rule and critic; no card passes the "
                 "card rule, so v1 could never promote it. Protocol v2 removes that asymmetry."),
    }
    out.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out), "grid_deflated_sharpe": arms["grid"]["deflated_sharpe"]["headline"],
                      "effective_trials": arms["grid"]["deflated_sharpe"]["effective_trials"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
