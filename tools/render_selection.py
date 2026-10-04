"""POST-HOC: why selection failed in the v1 Binance study, drawn from committed results only.

    python tools/render_selection.py            # writes docs/evidence/selection.json and docs/assets/selection.png
    python tools/render_selection.py --check    # exit 1 if selection.json is not what hypotheses.json gives

Reads results/real-2026-10/hypotheses.json (the pre-registered run of 2026-10-03 18:16, UTC+08:00) and
nothing else: no data, no network, no LLM. Everything here was computed after that run and
is not part of any protocol. The extract is standard library only; the figure needs
matplotlib, which is optional and not in requirements.txt.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "results/real-2026-10/hypotheses.json"
EXTRACT = "docs/evidence/selection.json"
FIGURE = "docs/assets/selection.png"
BASELINE = "momentum-20"  # the 20-day momentum baseline is this grid row (ablation.json: baselines.momentum_20.hypothesis)


def bps(x):
    return round(x * 1e4, 4)


def ranks(values):
    """Average ranks, 1-based, ties sharing the mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5


def spearman(x, y):
    return round(pearson(ranks(x), ranks(y)), 4)


def extract(raw):
    rows = json.loads(raw)
    momentum = sorted((r for r in rows if r["signal"] == "momentum"), key=lambda r: r["lookback"])
    series = [{"lookback": r["lookback"],
               "validation_net_bps": bps(r["validation"]["net_mean"]), "test_net_bps": bps(r["test"]["net_mean"]),
               "validation_ic": round(r["validation"]["ic_mean"], 6), "test_ic": round(r["test"]["ic_mean"], 6)}
              for r in momentum]
    # Rank the raw values: rounding first would create ties (two test ICs agree to 6 dp).
    col = {"lookback": [r["lookback"] for r in momentum],
           **{f"{split}_{key}": [r[split][field] for r in momentum] for split in ("validation", "test")
              for key, field in (("net_bps", "net_mean"), ("ic", "ic_mean"))}}
    by_id = {r["id"]: r for r in rows}
    ic_pick = max(rows, key=lambda r: r["validation"]["ic_mean"])
    top5 = sorted(series, key=lambda s: -s["validation_net_bps"])[:5]
    best_test = max(series, key=lambda s: s["test_net_bps"])

    def pos(split):
        return {r["id"] for r in rows if r[split]["net_mean"] > 0}

    return {
        "label": "POST-HOC: computed from the committed v1 results after the pre-registered run; not part of any protocol",
        "data_label": "REAL",
        "source": {"path": SOURCE, "sha256": hashlib.sha256(raw).hexdigest()},
        "units": "net: rebalanced-book net return, bps per day per unit gross, 10 bps per side; "
                 "validation = 2024, test = 2025-01-01 to 2026-08-31",
        "momentum": series,
        "spearman_momentum": {
            "validation_net_vs_test_net": spearman(col["validation_net_bps"], col["test_net_bps"]),
            "validation_ic_vs_test_ic": spearman(col["validation_ic"], col["test_ic"]),
            "lookback_vs_validation_net": spearman(col["lookback"], col["validation_net_bps"]),
            "lookback_vs_test_net": spearman(col["lookback"], col["test_net_bps"]),
        },
        "top5_by_validation_net": [s["lookback"] for s in top5],
        "best_test_lookback": best_test["lookback"],
        "baseline_momentum_20_test_net_bps": bps(by_id[BASELINE]["test"]["net_mean"]),
        "v1_grid_pick_on_validation_ic": {"id": ic_pick["id"], "test_net_bps": bps(ic_pick["test"]["net_mean"])},
        "counts_all_120": {"tested": len(rows), "positive_validation_net": len(pos("validation")),
                           "positive_test_net": len(pos("test")),
                           "positive_both": len(pos("validation") & pos("test"))},
    }


def render(ex, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#bdbcb5", "text.color": ink,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "savefig.facecolor": "white"})
    s = ex["momentum"]
    lb = [r["lookback"] for r in s]
    val = [r["validation_net_bps"] for r in s]
    test = [r["test_net_bps"] for r in s]
    row = {r["lookback"]: r for r in s}
    rho = ex["spearman_momentum"]["validation_net_vs_test_net"]
    hurdle = ex["baseline_momentum_20_test_net_bps"]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(left=0.07, right=0.84, top=0.80, bottom=0.18)
    fig.text(0.07, 0.94, "The best momentum lookback moved between 2024 and 2025–26",
             fontsize=17, weight="bold")
    fig.text(0.07, 0.885, f"The 2024 ranking of the 60 lookbacks by net is largely reversed on the "
             f"test window (Spearman {sgn(rho)}).", fontsize=11, color=muted)
    fig.text(0.07, 0.845, "REAL: 34 Binance USDT pairs, daily, delay 1, 10 bps per side. "
             "Net = rebalanced book, bps per day per unit gross.", fontsize=10, color=muted)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#9a998f", lw=1)
    ax.axhline(hurdle, color=muted, lw=1, ls=(0, (4, 3)))
    ax.text(60.8, hurdle, f"gate hurdle on test:\nabove 20-day momentum,\n{sgn(hurdle)}", va="center",
            fontsize=9, color=muted)
    ax.plot(lb, val, color=blue, lw=2, label="validation, 2024")
    ax.plot(lb, test, color=orange, lw=2, label="test, 2025-01 → 2026-08")
    ax.text(60.8, val[-1] - 0.6, "validation\n2024", color=ink, fontsize=10, va="center")
    ax.text(60.8, test[-1] + 0.9, "test\n2025–26", color=ink, fontsize=10, va="center")
    top = ex["top5_by_validation_net"]
    ax.scatter(top, [row[k]["validation_net_bps"] for k in top], s=46, color=blue, edgecolor="white", lw=1.5, zorder=3)
    ax.scatter(top, [row[k]["test_net_bps"] for k in top], s=46, color=orange, edgecolor="white", lw=1.5, zorder=3)
    for k in top:
        ax.plot([k, k], [row[k]["test_net_bps"], row[k]["validation_net_bps"]], color="#bdbcb5", lw=1, zorder=1)
    pick = row[top[0]]
    ax.annotate(f"best five on 2024 (lookbacks {', '.join(map(str, sorted(top)))}):\n"
                f"all negative on test. Lookback {top[0]}: "
                f"{sgn(pick['validation_net_bps'])} → {sgn(pick['test_net_bps'])}",
                xy=(top[0], pick["validation_net_bps"]), xytext=(21, 9.6), fontsize=9.5, color=ink,
                arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8})
    b = row[ex["best_test_lookback"]]
    ax.annotate(f"best on test: lookback {b['lookback']}, {sgn(b['test_net_bps'])}\n"
                f"(it was {sgn(b['validation_net_bps'])} on 2024)",
                xy=(b["lookback"], b["test_net_bps"]), xytext=(41, 6.4), fontsize=9.5, color=ink,
                arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8})
    ax.set_xlim(0, 60.5)
    ax.set_ylim(-12, 11)
    ax.set_xlabel("momentum lookback L (days): signal = P(t−1) / P(t−1−L) − 1")
    ax.set_ylabel("net return, bps per day")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    v1 = ex["v1_grid_pick_on_validation_ic"]
    fig.text(0.07, 0.03, f"POST-HOC view of committed results ({SOURCE}); not pre-registered.\n"
             f"v1's pre-registered grid pick, selected on validation IC, was {v1['id']}: "
             f"{sgn(v1['test_net_bps'])} bps/day on test. Extract and statistics: {EXTRACT}; "
             "script: tools/render_selection.py.", fontsize=8.5, color=muted)
    fig.text(0.97, 0.945, "POST-HOC", fontsize=11, weight="bold", color=muted, ha="right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, metadata={"Software": None})
    plt.close(fig)


def sgn(x):
    """A signed number with a typographic minus, for the figure's text."""
    return f"{x:+.2f}".replace("-", "\u2212")


def dumps(ex):
    return json.dumps(ex, indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only compare the committed extract")
    args = parser.parse_args(argv)
    ex = extract((ROOT / SOURCE).read_bytes())
    if args.check:
        same = (ROOT / EXTRACT).read_text(encoding="utf-8") == dumps(ex)
        print(f"{EXTRACT}: {'matches' if same else 'DIFFERS from'} {SOURCE}")
        return 0 if same else 1
    (ROOT / EXTRACT).write_text(dumps(ex), encoding="utf-8")
    render(ex, ROOT / FIGURE)
    print(f"wrote {EXTRACT} and {FIGURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
