#!/usr/bin/env python3
"""Recompute the committed real-data results from the raw CSVs alone, with pandas.

    python scripts/crosscheck_real.py --data "$ASOF_DATA_DIR" --results results/real-2026-10

Independent of the factor lab: no factor_research or quantos_showcase import. For
every hypothesis and split it rebuilds the signal (delay 1: closes up to t-1), the
daily Spearman rank IC against the t -> t+1 return, the unit-gross rank weights,
the rebalanced book's traded notional and net; and the equal-weight hold. Exits 1
if any reported number differs by more than 1e-9.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

TOL = 1e-9


def book(signal, returns, cost_bps):
    ranks = signal.rank(axis=1)
    centred = ranks.sub(ranks.mean(axis=1), axis=0)
    weights = centred.div(centred.abs().sum(axis=1), axis=0)
    gross = (weights * returns).sum(axis=1)
    drifted = (weights.shift(1) * (1 + returns.shift(1))).fillna(0)
    traded = (weights - drifted).abs().sum(axis=1)
    traded.iloc[-1] += (weights.iloc[-1] * (1 + returns.iloc[-1])).abs().sum()
    return gross - traded * cost_bps / 10_000, traded


def load_closes(folder):
    """Daily closes (date x asset) and the t -> t+1 return that day t's position earns."""
    closes = pd.concat({os.path.basename(f)[:-4]: pd.read_csv(f, index_col="date")["close"]
                        for f in sorted(glob.glob(os.path.join(folder, "*.csv")))}, axis=1).sort_index()
    return closes, closes.pct_change(fill_method=None).shift(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    closes, forward = load_closes(args.data)
    result = json.load(open(os.path.join(args.results, "ablation.json")))
    rows = json.load(open(os.path.join(args.results, "hypotheses.json")))
    spans = {s: (v["first_decision"][:10], (pd.Timestamp(v["last_label"][:10]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
             for s, v in result["splits"].items()}
    worst, checked = 0.0, 0
    for row in rows:
        sign = 1 if row["signal"] == "momentum" else -1
        signal = (closes.shift(1) / closes.shift(1 + row["lookback"]) - 1) * sign
        for split, (lo, hi) in spans.items():
            ic = signal.rank(axis=1).corrwith(forward.rank(axis=1), axis=1).loc[lo:hi]
            net, traded = book(signal.loc[lo:hi], forward.loc[lo:hi], 10)
            mine = {"ic_mean": ic.mean(), "ic_t": ic.mean() / (ic.std() / np.sqrt(len(ic))),
                    "net_mean": net.mean(), "turnover_mean": traded.mean()}
            for key, value in mine.items():
                worst = max(worst, abs(value - row[split][key]))
                checked += 1
    for split, (lo, hi) in spans.items():
        start, end = closes.loc[lo], closes.shift(-1).loc[hi]
        cumulative = (end / start).mean() - 1
        worst = max(worst, abs(cumulative - result["baselines"]["equal_weight_hold"][split]["cumulative_return"]))
        checked += 1
    ok = worst <= TOL
    print(json.dumps({"status": "MATCH" if ok else "MISMATCH", "numbers_checked": checked,
                      "max_abs_difference": worst, "tolerance": TOL, "hypotheses": len(rows)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
