#!/usr/bin/env python3
"""Power of protocol v2's verdict, from validation data only (amendment 2).

    python scripts/power_forward.py --data "$ASOF_DATA_DIR" --out results/forward-2026-09/power.json

The forward test is one-sided at 5% with a Newey-West (Bartlett, lag 5) standard error
over 365 days. For a daily series x with HAC variance s^2 per day (estimated on the v2
validation window, 2024-01-01..2026-08-31), the smallest mean that reaches p < 0.05 is
    threshold = 1.645 * s / sqrt(365),
the BREAK-EVEN mean, at which the test rejects half the time (50% power). The power
at a true mean mu is Phi((mu - threshold) / se) with se = threshold / 1.645, and 80%
power needs mu = threshold + 0.8416 * se. This applies to one arm's daily net, and to
the daily LLM-minus-arm net difference for the paired test. The pairs are illustrative: the frozen picks against each other and
against neighbouring and distant hypotheses. Reads no data after 2026-08-31.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscheck_real import book, load_closes  # noqa: E402

from quantos_showcase import ablation  # noqa: E402

LO, HI, DAYS, Z = "2024-01-01", "2026-08-30", 365, 1.6448536269514722
NORMAL = ablation.NORMAL


def net_series(closes, forward, hid):
    signal, lookback = hid.rsplit("-", 1)
    sign = 1 if signal == "momentum" else -1
    feature = (closes.shift(1) / closes.shift(1 + int(lookback)) - 1) * sign
    return book(feature.loc[LO:HI], forward.loc[LO:HI], 10)[0].tolist()


def threshold(xs):
    """Smallest one-sided-5% mean over DAYS days, with the per-day HAC sd of xs."""
    n, mean = len(xs), sum(xs) / len(xs)
    se = mean / ablation.newey_west_t(xs)  # HAC standard error of the mean over n days
    return Z * abs(se) * math.sqrt(n / DAYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    closes, forward = load_closes(args.data)
    closes, forward = closes.loc[:"2026-08-31"], forward.loc[:"2026-08-31"]
    ids = ["momentum-25", "momentum-24", "momentum-26", "momentum-19", "momentum-20", "momentum-14", "momentum-05",
           "reversal-02"]
    nets = {i: net_series(closes, forward, i) for i in ids}
    single = {i: threshold(nets[i]) for i in ids}
    pairs = {f"{a} minus {b}": threshold([x - y for x, y in zip(nets[a], nets[b])])
             for a, b in (("momentum-24", "momentum-25"), ("momentum-26", "momentum-25"),
                          ("momentum-19", "momentum-25"), ("momentum-20", "momentum-25"),
                          ("momentum-14", "momentum-25"), ("momentum-05", "momentum-25"),
                          ("reversal-02", "momentum-25"))}
    frozen = json.loads(Path("results/forward-2026-09/selections.json").read_text())["validation"]
    both = sorted(i for i, r in frozen.items() if r["net_mean"] > 0 and r["ic_mean"] > 0)
    best = max(r["net_mean"] for r in frozen.values())
    body = {
        "label": "computed from validation data (through 2026-08-31) for protocol v2 amendments 2 and 3, before any LLM call",
        "method": ("break-even threshold (50% power) = 1.645 * HAC sd per day / sqrt(365); HAC = Newey-West, "
                   "Bartlett kernel, lag 5; power(mu) = Phi((mu - threshold) / (threshold / 1.645))"),
        "validation_intervals": len(nets["momentum-25"]),
        "single_arm_break_even_net_bps_per_day": {i: v * 1e4 for i, v in single.items()},
        "paired_difference_break_even_bps_per_day": {k: v * 1e4 for k, v in pairs.items()},
        "best_validation_net_bps_per_day": best * 1e4,
        "momentum-25_power_if_true_mean_is_its_validation_net": NORMAL.cdf((best - single["momentum-25"])
                                                                          / (single["momentum-25"] / Z)),
        "momentum-25_mean_for_80pct_power_bps_per_day": (single["momentum-25"] * (1 + NORMAL.inv_cdf(.8) / Z)) * 1e4,
        "binding_test": ("For a neighbouring pick the paired threshold is below the single-arm one, so the arm-level "
                         "test on the LLM pick's own net binds, not the paired test."),
        "positive_validation_net_and_ic": {"count": len(both), "of": len(frozen), "ids": both},
    }
    Path(args.out).write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
