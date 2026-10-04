"""Does the LLM proposer help? A pre-registered ablation against non-LLM hypothesis sources.

    verify-data  check an OHLCV directory against the committed universe file (SHA-256 per file)
    run          test every grid hypothesis through the factor lab (prepare -> run ->
                 verify --recompute), then score each arm of the frozen protocol
    show         print the report of a results directory

Arms: the LLM loop's frozen cards (once a live run exists), the enumerate-all grid
over the same signals and lookbacks, and random-K draws from that grid. Every arm
selects the hypothesis with the highest VALIDATION mean rank IC and reports that
one's TEST result; the test split never enters a selection. The protocol file
fixes the splits, costs, baselines, selection, gate and seeds before any run.
Protocol v1 also required the LLM selection to pass the loop's card rule and
critic; v2 promotes every arm's selection by the same rule.

Statistics: plain and Newey-West t of the daily rank IC, Bonferroni on the
validation p, and the deflated Sharpe of the validation IC series under six input
specifications (it ranges widely; the verdict rests on the test split instead).

Net returns here hold one continuous book, rebalanced daily to the factor lab's
weights: cost = bps per side x traded notional, with drift between closes. The
factor lab's own sleeve net (every position opened and closed every day) is kept
beside it as a conservative upper bound on cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from factor_research.contracts import iso, read_config, read_panel
from factor_research.prepare import prepare_prices
from factor_research.runs import run_trial, verify_run

PROTOCOL_SCHEMAS = {"asof-ablation-protocol/v1", "asof-ablation-protocol/v2"}
UNIVERSE_SCHEMA = "asof-universe/v1"
RESULT_SCHEMA = "asof-ablation-result/v1"
SPLITS = ("validation", "test")
EULER_GAMMA = 0.5772156649015329
NW_LAGS = 5
NORMAL = statistics.NormalDist()


def log(stage, **counts):
    print(f"[{stage}] " + " ".join(f"{k}={v}" for k, v in counts.items()), file=sys.stderr)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- data identity

def verify_data(data_dir, universe_path):
    """Every file the universe names, byte for byte, and nothing else."""
    universe = read_json(universe_path)
    if universe.get("schema") != UNIVERSE_SCHEMA:
        raise ValueError("universe file must be " + UNIVERSE_SCHEMA)
    expected = universe["sha256"]
    present = {p.name for p in Path(data_dir).glob("*.csv")}
    missing, extra = sorted(set(expected) - present), sorted(present - set(expected))
    changed = sorted(n for n in set(expected) & present
                     if sha256_bytes((Path(data_dir) / n).read_bytes()) != expected[n])
    if missing or extra or changed:
        raise ValueError(f"data differs from {Path(universe_path).name}: missing {missing}, "
                         f"unexpected {extra}, checksum mismatch {changed}")
    return {"status": "VERIFIED", "files": len(expected), "universe": universe["id"],
            "universe_sha256": sha256_bytes(Path(universe_path).read_bytes())}


def fill_missing(data_dir, through, out_dir):
    """Protocol v2's delisting rule, as code. Every pair is extended to `through`: a
    date with no bar (a gap, or every date after a delisting) gets a bar at the last
    available close with zero volume, so its return is zero while it is still ranked.
    The robustness run (amendment 3) is the unfilled data itself: a pair is absent only
    on the dates it lacks. Returns the filled dates per pair as inclusive ranges."""
    from datetime import date, timedelta
    end = date.fromisoformat(through)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    filled = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        lines = path.read_text(encoding="utf-8").splitlines()
        header, rows = lines[0], [line.split(",") for line in lines[1:] if line]
        by_day = {r[0]: r for r in rows}
        day, last, out_rows, ranges = date.fromisoformat(rows[0][0]), None, [], []
        while day <= end:
            key = day.isoformat()
            if key in by_day:
                last = by_day[key]
                out_rows.append(",".join(last))
            else:
                close = last[4]
                out_rows.append(",".join([key, close, close, close, close, "0"]))
                if ranges and ranges[-1][1] == (day - timedelta(days=1)).isoformat():
                    ranges[-1][1] = key
                else:
                    ranges.append([key, key])
            day += timedelta(days=1)
        if ranges:
            filled[path.stem] = ranges
        (out / path.name).write_text("\n".join([header, *out_rows]) + "\n", encoding="utf-8")
    return {"through": through, "rule": "fill with last close", "filled": filled,
            "pairs_written": len(list(out.glob("*.csv")))}


# ---------------------------------------------------------------- statistics

def newey_west_t(xs, lags=NW_LAGS):
    """t of the mean with a Bartlett-kernel HAC (Newey-West) standard error."""
    n = len(xs)
    mean = math.fsum(xs) / n
    c = [x - mean for x in xs]
    gamma = lambda k: math.fsum(c[i] * c[i - k] for i in range(k, n)) / n
    variance = gamma(0) + 2 * math.fsum((1 - k / (lags + 1)) * gamma(k) for k in range(1, lags + 1))
    return mean / math.sqrt(variance / n) if variance > 0 else None


def autocorrelation(xs, lag):
    n, mean = len(xs), math.fsum(xs) / len(xs)
    c = [x - mean for x in xs]
    return math.fsum(c[i] * c[i - lag] for i in range(lag, n)) / math.fsum(v * v for v in c)


def effective_trials(series):
    """Participation ratio of the trials' correlation matrix, n^2 / sum(C_ij^2): n for
    independent trials, 1 for identical ones. Signs do not matter, so a mirrored pair
    (reversal = -momentum) counts once."""
    n = len(series)
    if n < 2:
        return float(n)
    def corr(a, b):
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        ma, mb = (math.fsum(p[i] for p in pairs) / len(pairs) for i in (0, 1))
        sab = math.fsum((x - ma) * (y - mb) for x, y in pairs)
        saa, sbb = (math.fsum((p[i] - m) ** 2 for p in pairs) for i, m in ((0, ma), (1, mb)))
        return sab / math.sqrt(saa * sbb) if saa and sbb else 0.0
    total = n + 2 * math.fsum(corr(series[i], series[j]) ** 2 for i in range(n) for j in range(i))
    return n * n / total


def eigen_effective_trials(series):
    """Effective number of independent trials from the eigenvalues of the trials'
    correlation matrix: Nyholt (2004), 1 + (n - 1)(1 - var(eigenvalues) / n), and
    Li and Ji (2005), the sum over eigenvalues of [l >= 1] + (l - floor(l)); with the
    participation ratio, the smallest of the three, for comparison."""
    n = len(series)
    if n < 2:
        return {"participation": float(n), "li_ji": float(n), "nyholt": float(n)}
    matrix = np.nan_to_num(np.corrcoef(np.array([[np.nan if x is None else x for x in s] for s in series])))
    np.fill_diagonal(matrix, 1.0)
    eig = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
    return {"participation": effective_trials(series),
            "li_ji": float(sum((e >= 1) + (e - math.floor(e)) for e in eig)),
            "nyholt": float(1 + (n - 1) * (1 - np.var(eig, ddof=1) / n))}


def moments(xs):
    """n, mean, sample sd, t of the mean (plain and Newey-West), one-sided p (H1: mean > 0),
    skew, raw kurtosis."""
    n = len(xs)
    mean = math.fsum(xs) / n
    sd = statistics.stdev(xs) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd else None
    central = [x - mean for x in xs]
    m2 = math.fsum(c * c for c in central) / n
    skew = math.fsum(c ** 3 for c in central) / n / m2 ** 1.5 if m2 else 0.0
    kurt = math.fsum(c ** 4 for c in central) / n / m2 ** 2 if m2 else 3.0
    t_nw = newey_west_t(xs) if sd else None
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": None if t is None else 1 - NORMAL.cdf(t),
            "t_nw": t_nw, "p_nw": None if t_nw is None else 1 - NORMAL.cdf(t_nw),
            "sharpe": mean / sd if sd else None, "skew": skew, "kurt": kurt}


def deflated_sharpe(sharpe, n_obs, skew, kurt, trial_sharpes, n_trials=None, sharpe_variance=None):
    """Bailey and Lopez de Prado (2014): the probability that the selected per-period
    Sharpe beats the maximum expected from n_trials unskilled trials (default: one per
    trial Sharpe). The trials' Sharpe variance is sharpe_variance if given (e.g. the
    null sampling variance 1/T), else the variance of trial_sharpes."""
    n = len(trial_sharpes) if n_trials is None else n_trials
    if n < 2:
        threshold = 0.0
    else:
        spread = math.sqrt(statistics.variance(trial_sharpes) if sharpe_variance is None else sharpe_variance)
        threshold = spread * ((1 - EULER_GAMMA) * NORMAL.inv_cdf(1 - 1 / n)
                              + EULER_GAMMA * NORMAL.inv_cdf(1 - 1 / (n * math.e)))
    scale = math.sqrt(max(1e-12, 1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2))
    return NORMAL.cdf((sharpe - threshold) * math.sqrt(n_obs - 1) / scale), threshold


def holm(pvalues):
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


# ---------------------------------------------------------------- one trial

def rebalanced_book(daily, prices, cost_bps):
    """Gross, traded notional and net per interval for one book held across days.

    Weights are the factor lab's; between closes each position drifts with its
    return, and the next close trades back to the new weights. The first interval
    pays the entry and the last also pays the exit. The gross must equal the
    factor lab's own sleeve gross, which pins the price alignment.

    A held pair with no exit bar (the DELISTED=drop robustness run, on unfilled data)
    closes at its last close: zero return that interval. The factor lab marks such a
    day MISSING_EXIT_MARK with no gross, so that day is not cross-checked. The pair has
    no entry price afterwards, so the lab gives it no weight and the next close sells it.
    """
    rows, held = [], {}
    for i, day in enumerate(daily):
        weights = day["weights"]
        returns = {}
        for asset, weight in weights.items():
            if weight:
                entry, exit_ = prices.get((day["time"], asset)), prices.get((day["label_time"], asset))
                if entry is None:
                    raise ValueError(f"no entry price for held {asset} at {day['time']}")
                returns[asset] = 0.0 if exit_ is None else exit_ / entry - 1
        gross = math.fsum(weights[a] * r for a, r in returns.items())
        lab = day["portfolio"]["gross"]
        if (lab is None) != (day["portfolio"].get("status") == "MISSING_EXIT_MARK") or \
                (lab is not None and abs(gross - lab) > 1e-12):
            raise ValueError(f"gross at {day['time']} differs from the factor lab's")
        traded = math.fsum(abs(weights.get(a, 0.0) - held.get(a, 0.0)) for a in set(weights) | set(held))
        held = {a: weights[a] * (1 + r) for a, r in returns.items()}
        if i == len(daily) - 1:
            traded += math.fsum(abs(v) for v in held.values())
        rows.append((gross, traded, gross - traded * cost_bps / 10_000))
    return rows


def split_metrics(part, prices, cost_bps):
    daily = part["daily"]
    book = rebalanced_book(daily, prices, cost_bps)
    ic = moments([d["rank_ic"] for d in daily if d["rank_ic"] is not None])
    net = moments([r[2] for r in book])
    summary = part["summary"]
    return {
        "intervals": len(daily), "ic_dates": ic["n"],
        "ic_mean": ic["mean"], "ic_t": ic["t"], "ic_t_nw": ic["t_nw"], "ic_p": ic["p"], "ic_p_nw": ic["p_nw"],
        "ic_sharpe": ic["sharpe"], "ic_skew": ic["skew"], "ic_kurt": ic["kurt"],
        "gross_mean": math.fsum(r[0] for r in book) / len(book),
        "net_mean": net["mean"], "net_t": net["t"], "net_p": net["p"], "net_t_nw": net["t_nw"], "net_p_nw": net["p_nw"],
        "net_sharpe": net["sharpe"], "net_skew": net["skew"], "net_kurt": net["kurt"],
        "net_sharpe_annual": None if net["sharpe"] is None else net["sharpe"] * math.sqrt(365),
        "turnover_mean": math.fsum(r[1] for r in book) / len(book),
        "sleeve_net_mean": summary["net_mean"], "sleeve_turnover_mean": summary["turnover_mean"],
        "first": daily[0]["time"], "last_label": daily[-1]["label_time"],
    }


def panel_prices(trial_dir):
    """The exact prices the factor lab evaluated, keyed by (ISO time, asset)."""
    config = read_config((trial_dir / "config.json").read_bytes())
    panel = read_panel((trial_dir / "panel.csv").read_bytes(), config)
    return {(iso(t), a): row.price for (t, a), row in panel.items()}


def run_hypothesis(job):
    """prepare -> run -> verify --recompute for one (signal, lookback); returns small metrics."""
    hid, signal, lookback, prices_path, contract_raw, work = job
    work = Path(work)
    contract = json.loads(contract_raw)
    contract["experiment"] = dict(contract["experiment"], lookback=lookback, candidates=[signal])
    contract_path = work / "contracts" / f"{hid}.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    prepare_prices(prices_path, contract_path, work / "prepared" / hid)
    run_trial(None, None, work / "trials", hid, prepared=work / "prepared" / hid)
    trial_dir = work / "trials" / hid
    checked = verify_run(trial_dir, recompute=True)
    if checked["status"] != "SUCCESS" or not checked["recomputed"]:
        raise ValueError(f"{hid}: trial did not complete and recompute")
    report = json.loads((trial_dir / "report.json").read_bytes())
    prices = panel_prices(trial_dir)
    parts = {"validation": report["development"][signal]["validation"], "test": report["test"]}
    row = {"id": hid, "signal": signal, "lookback": lookback,
           "trial": {k: checked["identity"][k] for k in ("input_sha256", "config_sha256", "import_sha256",
                                                         "code_sha256")}}
    cost = contract["experiment"]["cost_bps"]
    row.update({s: split_metrics(parts[s], prices, cost) for s in SPLITS})
    series = {s: {"ic_mean": [d["rank_ic"] for d in parts[s]["daily"]],
                  "net_mean": [r[2] for r in rebalanced_book(parts[s]["daily"], prices, cost)]} for s in SPLITS}
    return row, series


def hold_baseline(trial_dir, cost_bps):
    """The equal-weight hold on the same decision dates and prices as the grid trials."""
    report = json.loads((trial_dir / "report.json").read_bytes())
    prices = panel_prices(trial_dir)
    signal = report["selection"]["selected"]
    parts = {"validation": report["development"][signal]["validation"], "test": report["test"]}
    return {s: equal_weight_hold(parts[s]["daily"], prices, cost_bps) for s in SPLITS}


def equal_weight_hold(daily, prices, cost_bps):
    """Long-only equal weights bought at the first decision close and held to the
    split's last label close; cost on the entry and on the exit value."""
    assets = sorted(a for (t, a) in prices if t == daily[0]["time"])
    start = {a: prices[(daily[0]["time"], a)] for a in assets}
    last = dict(start)  # a pair with no bar is valued at its last close (the DELISTED=drop run)
    value = [1.0]
    for day in daily:
        last.update({a: prices[(day["label_time"], a)] for a in assets if (day["label_time"], a) in prices})
        value.append(math.fsum(last[a] / start[a] for a in assets) / len(assets))
    pnl = [value[i + 1] - value[i] for i in range(len(daily))]
    traded = [0.0] * len(daily)
    traded[0] += 1.0
    traded[-1] += value[-1]
    net = moments([p - t * cost_bps / 10_000 for p, t in zip(pnl, traded)])
    return {"intervals": len(daily), "ic_mean": None, "ic_t": None, "ic_p": None,
            "gross_mean": math.fsum(pnl) / len(pnl), "net_mean": net["mean"], "net_t": net["t"],
            "net_p": net["p"], "net_sharpe_annual": None if net["sharpe"] is None else net["sharpe"] * math.sqrt(365),
            "turnover_mean": math.fsum(traded) / len(traded), "cumulative_return": value[-1] - 1,
            "assets": len(assets), "first": daily[0]["time"], "last_label": daily[-1]["label_time"]}


# ---------------------------------------------------------------- arms

def hypothesis_id(signal, lookback):
    return f"{signal}-{lookback:02d}"


def grid(protocol):
    g = protocol["grid"]
    return [(hypothesis_id(s, lb), s, lb) for s in g["signals"]
            for lb in range(g["min_lookback"], g["max_lookback"] + 1)]


def select(ids, rows, metric="ic_mean"):
    """Highest validation value of the protocol's selection metric (v1: mean rank IC;
    v2 after amendment 1: mean rebalanced net, the gate's own metric); the first in
    grid order wins a tie."""
    return max(ids, key=lambda i: rows[i]["validation"][metric])


def gate_passes(test, best_baseline_net):
    return test["net_mean"] > 0 and test["net_mean"] > best_baseline_net


SHAPE = {"ic_mean": ("ic_sharpe", "ic_skew", "ic_kurt", "ic_dates"),
         "net_mean": ("net_sharpe", "net_skew", "net_kurt", "intervals")}


def deflation(chosen, ids, rows, series, metric="ic_mean"):
    """The deflated Sharpe of the selection's daily validation series (rank IC or net,
    whichever selected it).

    Headline: the null sampling variance of a Sharpe estimate (1/T) with the effective
    number of trials from the eigenvalues of the trials' correlation matrix (Li-Ji and
    Nyholt), reported as a range. The variance of trial Sharpes measured across
    correlated trials already reflects their correlation, so pairing it with a reduced N
    counts the correlation twice; those cells stay in the sensitivity table, flagged.
    """
    sharpe_key, skew_key, kurt_key, n_key = SHAPE[metric]
    val = rows[chosen]["validation"]
    if val.get(sharpe_key) is None:  # a constant series (a synthetic control) has no Sharpe
        return None
    sharpe = lambda i: rows[i]["validation"][sharpe_key]
    signed = [sharpe(i) for i in ids if sharpe(i) is not None]
    distinct = {}
    for i in ids:  # reversal is exactly -momentum in IC: one series per lookback, oriented like the pick
        if sharpe(i) is not None and (rows[i]["lookback"] not in distinct or rows[i]["signal"] == rows[chosen]["signal"]):
            same = rows[i]["signal"] == rows[chosen]["signal"]
            distinct[rows[i]["lookback"]] = (sharpe(i) if same else -sharpe(i), series[i])
    oriented = [v[0] for v in distinct.values()]
    n_eff = eigen_effective_trials([v[1] for v in distinct.values()])
    t_obs = val[n_key]
    variances = {"null_1_over_T": 1 / t_obs, "trials_signed": None, "trials_distinct": None}
    trial_sets = {"null_1_over_T": signed, "trials_signed": signed, "trials_distinct": oriented}
    counts = {"all": len(signed), "distinct": len(oriented), **n_eff}
    table, doubled = {}, []
    for var, v in variances.items():
        for label, n in counts.items():
            name = f"{var}__n_{label}"
            table[name] = deflated_sharpe(val[sharpe_key], t_obs, val[skew_key], val[kurt_key], trial_sets[var],
                                          max(1, round(n)), sharpe_variance=v)[0]
            if var != "null_1_over_T" and label in ("participation", "li_ji", "nyholt"):
                doubled.append(name)
    head = [table["null_1_over_T__n_li_ji"], table["null_1_over_T__n_nyholt"]]
    return {"applied_to": f"the daily validation {metric.split('_')[0]} series of the selection",
            "effective_trials": n_eff, "distinct_lookbacks": len(oriented),
            "headline": {"spec": "null sampling variance 1/T; N = Li-Ji and Nyholt effective trials",
                         "li_ji": head[0], "nyholt": head[1], "range": [min(head), max(head)]},
            "sensitivity": table, "double_counted": doubled}


def score_arm(name, ids, rows, best_baseline_net, series, metric="ic_mean", extra_gate=None):
    chosen = select(ids, rows, metric)
    other = "ic_mean" if metric == "net_mean" else "net_mean"
    alt = select(ids, rows, other)
    val, test = rows[chosen]["validation"], rows[chosen]["test"]
    passes = gate_passes(test, best_baseline_net) and (extra_gate is None or extra_gate(chosen))
    return {
        "arm": name, "status": "RUN", "hypotheses_tested_on_validation": len(ids), "hypotheses": list(ids),
        "selection_metric": metric, "selected": chosen,
        "validation": {"ic_mean": val["ic_mean"], "ic_t": val["ic_t"], "ic_t_nw": val["ic_t_nw"],
                       "ic_p": val["ic_p"],
                       "ic_p_bonferroni": None if val["ic_p"] is None else min(1.0, len(ids) * val["ic_p"]),
                       "deflated_sharpe": deflation(chosen, ids, rows, {i: series[i][metric] for i in ids}, metric),
                       "net_mean": val["net_mean"]},
        "test": {k: test.get(k) for k in ("ic_mean", "ic_t", "ic_t_nw", "ic_p", "ic_p_nw", "net_mean", "net_t",
                                          "net_t_nw", "net_p_nw", "net_sharpe_annual", "turnover_mean",
                                          "sleeve_net_mean")},
        "promoted": passes,
        "secondary_selection": {"metric": other, "selected": alt, "test_net_mean": rows[alt]["test"]["net_mean"],
                                "test_ic_mean": rows[alt]["test"]["ic_mean"],
                                "promoted": gate_passes(rows[alt]["test"], best_baseline_net)},
    }


def outcome_range(ids, rows, k, best_baseline_net, metric="ic_mean"):
    """Every result a K-card arm could report from these trials: a hypothesis can be
    selected only if at least K-1 others rank below it on the validation metric."""
    order = sorted(ids, key=lambda i: rows[i]["validation"][metric])
    reachable = order[k - 1:]
    nets = [rows[i]["test"]["net_mean"] for i in reachable]
    ics = [rows[i]["test"]["ic_mean"] for i in reachable]
    return {"k": k, "selection_metric": metric, "selectable": len(reachable), "test_net_mean": [min(nets), max(nets)],
            "test_ic_mean": [min(ics), max(ics)],
            "promotable": sorted(i for i in reachable if gate_passes(rows[i]["test"], best_baseline_net))}


def random_draw(protocol, ids):
    spec = protocol["arms"]["random"]
    rng = random.Random(spec["seed"])
    return rng, rng.sample(ids, spec["k"])


def random_arm(protocol, ids, rows, best_baseline_net, series, metric="ic_mean"):
    spec = protocol["arms"]["random"]
    rng, draw = random_draw(protocol, ids)
    arm = score_arm("random-K", draw, rows, best_baseline_net, series, metric)
    picks = [select(rng.sample(ids, spec["k"]), rows, metric) for _ in range(spec["distribution_draws"])]
    ics = sorted(rows[p]["test"]["ic_mean"] for p in picks)
    nets = sorted(rows[p]["test"]["net_mean"] for p in picks)
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
    arm["distribution"] = {
        "draws": len(picks), "seed": spec["seed"], "note": "draws after the reported one, same generator",
        "test_ic_mean": {"p05": q(ics, .05), "median": q(ics, .5), "p95": q(ics, .95)},
        "test_net_mean": {"p05": q(nets, .05), "median": q(nets, .5), "p95": q(nets, .95)},
        "share_test_net_positive": sum(n > 0 for n in nets) / len(nets),
        "share_promoted": sum(gate_passes(rows[p]["test"], best_baseline_net) for p in picks) / len(picks),
    }
    return arm


P_NW = {"net_mean": "net_p_nw", "ic_mean": "ic_p_nw"}


def paired(a, b, lags=NW_LAGS):
    """One-sided Newey-West test that the daily series a beats b on the same dates."""
    if len(a) != len(b):
        raise ValueError("paired series must cover the same dates")
    diff = [x - y for x, y in zip(a, b)]
    mean = math.fsum(diff) / len(diff)
    t = newey_west_t(diff, lags) if any(d != mean for d in diff) else None
    return {"mean_difference": mean, "t_nw": t, "p_nw": None if t is None else 1 - NORMAL.cdf(t)}


def verdict(arms, spec, test_series):
    """Protocol v2's verdict, in code (amendment 2).

    An arm helps only if its selection is promoted and its daily test series of
    spec["metric"] (net, the metric that selects and gates) has one-sided Newey-West
    p < alpha. The LLM helps only if its arm helps and, for every arm named in
    spec["paired_against"], a paired one-sided Newey-West test on the daily
    LLM-minus-arm difference of that metric has p < alpha. None while the LLM arm is
    pending. test_series maps hypothesis id to its daily test series of the metric."""
    alpha, key = spec["alpha"], P_NW[spec["metric"]]
    # The arms' own p-values are computed per trial with NW_LAGS; a protocol naming
    # another lag is refused rather than silently tested at the wrong one.
    if spec["newey_west_lags"] != NW_LAGS:
        raise ValueError(f"protocol Newey-West lag {spec['newey_west_lags']} differs from the {NW_LAGS} the arms use")
    by = {a["arm"]: a for a in arms}
    helps = {name: a["status"] == "RUN" and a["promoted"] and a["test"].get(key) is not None
             and a["test"][key] < alpha for name, a in by.items()}
    llm = by.get("LLM")
    comparisons, llm_helps = {}, None
    if llm is not None and llm["status"] == "RUN":
        for other in spec["paired_against"]:
            test = paired(test_series[llm["selected"]], test_series[by[other]["selected"]], spec["newey_west_lags"])
            comparisons[other] = dict(test, significant=test["p_nw"] is not None and test["p_nw"] < alpha)
        llm_helps = helps["LLM"] and all(c["significant"] for c in comparisons.values())
    elif llm is not None and llm["status"] != "PENDING":
        llm_helps = False
    name = {"net_mean": "net return", "ic_mean": "rank IC"}[spec["metric"]]
    rule = (f"An arm helps if its pick is promoted and its daily test {name} has a one-sided Newey-West "
            f"(lag {spec['newey_west_lags']}) p below {alpha}. The LLM helps if, in addition, paired one-sided "
            f"Newey-West tests of its daily test {name} minus that of the "
            f"{' and the '.join(spec['paired_against'])} pick each give p below {alpha}.")
    return {"alpha": alpha, "metric": spec["metric"], "rule": rule,
            "arm_helps": helps, "llm_vs": comparisons, "llm_helps": llm_helps}


def llm_arm(llm_run, rows, best_baseline_net, receipt_keys, series, symmetric, metric="ic_mean"):
    """The loop's frozen cards, cross-checked against the grid trial of the same
    (signal, lookback); the deterministic human gate is recorded in the loop run.

    symmetric (protocol v2): the arm's selection is promoted by the same rule as the
    other arms. Otherwise (v1) it must also have passed the loop's card rule and critic."""
    from quantos_showcase import loop

    loop.verify(llm_run, receipt_keys=receipt_keys)
    ledger = loop.read_ledger(llm_run)
    frozen = [r for r in ledger["hypotheses"] if "card_sha256" in r]
    by_card = {}
    for r in frozen:
        hid = hypothesis_id(r["signal"], r["lookback"])
        if hid not in rows:
            raise ValueError(f"{r['id']}: {hid} is outside the protocol grid")
        for split in SPLITS:
            got = ((r.get("decision") or {}).get("metrics") or {}).get(split)
            if got is None and split == "test" and r["status"] != "AWAITING_HUMAN":
                continue  # the loop seals test for every card that did not reach its gate
            got = got or {}
            want = rows[hid][split]
            if (got.get("rank_ic_mean"), got.get("net_mean")) != (want["ic_mean"], want["sleeve_net_mean"]):
                raise ValueError(f"{r['id']}: loop metrics differ from the grid trial {hid}")
        by_card[hid] = r
    if not by_card:
        return {"arm": "LLM", "status": "NO_VALID_PROPOSALS", "counts": ledger["counts"]}
    decided = loop.read_decisions(llm_run)
    for hid, r in by_card.items():
        if r["status"] == "AWAITING_HUMAN" and r["id"] not in decided:
            passes = gate_passes(rows[hid]["test"], best_baseline_net)
            loop.decide(llm_run, r["id"], "PROMOTED" if passes else "REJECTED_BY_HUMAN",
                        note="pre-registered rule: test rebalanced net > 0 and > best baseline")
    arm = score_arm("LLM", list(by_card), rows, best_baseline_net, series, metric,
                    extra_gate=None if symmetric else lambda hid: by_card[hid]["status"] == "AWAITING_HUMAN")
    arm["loop_counts"] = loop.status(llm_run)["counts"]
    arm["card_ids"] = {hid: r["id"] for hid, r in by_card.items()}
    return arm


# ---------------------------------------------------------------- run

def run_grid(protocol, prices_path, contract_raw, work, workers=None):
    """Every grid hypothesis through prepare -> run -> verify --recompute, in parallel."""
    hypotheses = grid(protocol)
    jobs = [(hid, s, lb, str(prices_path), contract_raw, str(work)) for hid, s, lb in hypotheses]
    with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        done = list(pool.map(run_hypothesis, jobs))
    return hypotheses, {row["id"]: row for row, _ in done}, {row["id"]: s for row, s in done}


def freeze_selections(protocol_path, prices_path, contract_path, work, out, workers=None):
    """Write down, before the forward window is scored, every selection the data in hand
    already determines: the grid and seeded random-K picks under both metrics and the
    full validation table. The selection-time import puts the study's validation window
    in the factor lab's last split, the only one that may end at the last available bar;
    a split's metrics do not depend on which slot it occupies, and the scoring run checks
    that its validation numbers equal these."""
    protocol = json.loads(Path(protocol_path).read_bytes())
    contract_raw = Path(contract_path).read_bytes()
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen selections are never overwritten")
    (Path(work) / "contracts").mkdir(parents=True, exist_ok=False)
    hypotheses, rows, _ = run_grid(protocol, prices_path, contract_raw, work, workers)
    ids = [h[0] for h in hypotheses]
    table = {i: {k: rows[i]["test"][k] for k in ("ic_mean", "ic_t", "ic_t_nw", "net_mean", "turnover_mean",
                                                  "sleeve_net_mean", "first", "last_label", "intervals")}
             for i in ids}
    view = {i: {"validation": table[i]} for i in ids}
    _, draw = random_draw(protocol, ids)
    picks = {arm: {m: {"selected": (pick := select(members, view, m)), "validation_net_mean": table[pick]["net_mean"],
                       "validation_ic_mean": table[pick]["ic_mean"]} for m in ("net_mean", "ic_mean")}
             for arm, members in (("grid", ids), ("random-K", draw))}
    body = {"schema": "asof-ablation-selections/v1", "protocol": protocol["id"],
            "inputs": {"prices_sha256": sha256_bytes(Path(prices_path).read_bytes()),
                       "contract_sha256": sha256_bytes(contract_raw)},
            "validation_window": {"first_decision": table[ids[0]]["first"], "last_label": table[ids[0]]["last_label"],
                                  "intervals": table[ids[0]]["intervals"]},
            "primary_metric": protocol.get("selection_metric", "ic_mean"),
            "random_k_draw": draw, "selections": picks, "validation": table}
    out.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return body


def check_selections(protocol, rows, ids, tolerance=1e-12):
    """The scoring run must reproduce the frozen selections and their validation numbers."""
    frozen = read_json(protocol["selections"])
    _, draw = random_draw(protocol, ids)
    if draw != frozen["random_k_draw"]:
        raise ValueError("random-K draw differs from the frozen selections")
    for i, want in frozen["validation"].items():
        got = rows[i]["validation"]
        if any(abs(got[k] - want[k]) > tolerance for k in ("ic_mean", "net_mean", "turnover_mean")):
            raise ValueError(f"{i}: validation metrics differ from the frozen selections")


def run(protocol_path, prices_path, contract_path, work, out, llm_run=None, llm_pending=None,
        receipt_keys=None, workers=None):
    protocol_raw = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_raw)
    if protocol.get("schema") not in PROTOCOL_SCHEMAS:
        raise ValueError("protocol must be one of " + ", ".join(sorted(PROTOCOL_SCHEMAS)))
    symmetric = protocol["schema"].endswith("/v2")
    if (llm_run is None) == (llm_pending is None):
        raise ValueError("give exactly one of a live LLM loop run or the reason it is pending")
    contract_raw = Path(contract_path).read_bytes()
    if json.loads(contract_raw)["experiment"]["cost_bps"] != protocol["costs"]["bps_per_side"]:
        raise ValueError("import contract cost differs from the protocol")
    out = Path(out)
    targets = [out / n for n in ("hypotheses.json", "ablation.json", "REPORT.md")]
    if any(t.exists() for t in targets):
        raise FileExistsError("results already written; results are never overwritten")
    work = Path(work)
    (work / "contracts").mkdir(parents=True, exist_ok=False)

    hypotheses, rows, all_series = run_grid(protocol, prices_path, contract_raw, work, workers)
    series = {i: v["validation"] for i, v in all_series.items()}
    metric = protocol.get("selection_metric", "ic_mean")
    ids = [h[0] for h in hypotheses]
    if "selections" in protocol:
        check_selections(protocol, rows, ids)
    hold = hold_baseline(work / "trials" / hypotheses[0][0], protocol["costs"]["bps_per_side"])
    log("test", trials=len(rows), recomputed=len(rows))

    baselines = {}
    for name, spec in protocol["baselines"].items():
        if spec:
            row = rows[hypothesis_id(spec["signal"], spec["lookback"])]
            baselines[name] = {"hypothesis": row["id"], **{s: row[s] for s in SPLITS}}
        else:
            baselines[name] = {"hypothesis": None, **hold}
    best_name = max(baselines, key=lambda n: baselines[n]["test"]["net_mean"])
    best_net = baselines[best_name]["test"]["net_mean"]

    arms = [score_arm("grid", ids, rows, best_net, series, metric),
            random_arm(protocol, ids, rows, best_net, series, metric)]
    if llm_run is not None:
        arms.insert(0, llm_arm(llm_run, rows, best_net, receipt_keys, series, symmetric, metric))
    else:
        arms.insert(0, {"arm": "LLM", "status": "PENDING", "reason": llm_pending, "live_calls_used": 0,
                        "outcome_range": outcome_range(ids, rows, protocol["arms"]["llm"]["k"], best_net, metric)})
    scored = [a for a in arms if a["status"] == "RUN"]
    # Holm across the arms' test looks, on the verdict's p (v2: net, Newey-West; v1: IC), only
    # once all three arms exist; with two it adds nothing.
    holm_key = P_NW[protocol["verdict"]["metric"]] if symmetric else "ic_p"
    with_p = [a for a in scored if a["test"][holm_key] is not None]
    for arm in scored:
        arm["test"]["p_holm_across_arms"] = None
    if len(with_p) >= 3:
        for arm, p in zip(with_p, holm([a["test"][holm_key] for a in with_p])):
            arm["test"]["p_holm_across_arms"] = p
    tested = sorted({h for a in scored for h in a["hypotheses"]} | {b["hypothesis"] for b in baselines.values()
                                                                     if b["hypothesis"]})
    for arm in scored:
        log("arm", arm=arm["arm"], tested=arm["hypotheses_tested_on_validation"], selected=arm["selected"],
            promoted=arm["promoted"])

    result = {
        "schema": RESULT_SCHEMA, "protocol": protocol["id"], "data_label": protocol["data_label"],
        "inputs": {"protocol_sha256": sha256_bytes(protocol_raw),
                   "prices_sha256": sha256_bytes(Path(prices_path).read_bytes()),
                   "contract_sha256": sha256_bytes(contract_raw),
                   "factor_code_sha256": sorted({r["trial"]["code_sha256"] for r in rows.values()})},
        "splits": {s: {"first_decision": hold[s]["first"], "last_label": hold[s]["last_label"],
                       "intervals": hold[s]["intervals"]} for s in SPLITS},
        "assets": hold["test"]["assets"],
        "counts": {"trials_run_and_recomputed": len(rows), "distinct_hypotheses_tested": len(tested),
                   "baselines": len(baselines),
                   "hypotheses_tested_on_validation": {a["arm"]: a["hypotheses_tested_on_validation"]
                                                       for a in scored}},
        "baselines": baselines, "best_baseline_on_test": best_name, "arms": arms,
        "selection_metric": metric,
    }
    if symmetric:
        spec = protocol["verdict"]
        result["verdict"] = verdict(arms, spec, {i: v["test"][spec["metric"]] for i, v in all_series.items()})
    out.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps([rows[i] for i in ids], indent=1, sort_keys=True) + "\n", encoding="utf-8")
    targets[1].write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    targets[2].write_text(render(result, [rows[i] for i in ids]), encoding="utf-8")
    return result


# ---------------------------------------------------------------- report

def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def bps(value):
    return fmt(None if value is None else value * 10_000, "+.2f")


def render(result, rows):
    lines = [f"# Ablation `{result['protocol']}` ({result['data_label']} data, {result['assets']} assets)", "",
             f"Selection: highest validation {'mean rebalanced net' if result.get('selection_metric') == 'net_mean' else 'mean rank IC'}. "
             "Net: one book rebalanced daily, 10 bps per side "
             "on traded notional; bps are per day per unit gross. Sleeve net: every position opened and "
             "closed daily (conservative).", "",
             "| split | first decision | last label | intervals |", "|---|---|---|---|"]
    lines += [f"| {s} | {v['first_decision'][:10]} | {v['last_label'][:10]} | {v['intervals']} |"
              for s, v in result["splits"].items()]
    lines += ["", "## Arms: best-of-validation selection, then its test result", "",
              "| arm | tested on validation | selected | val IC (t, Newey-West t) | val net bps/day | val p Bonferroni "
              "| deflated Sharpe: Li-Ji to Nyholt N | test IC (t, NW p) | test net bps/day (NW p) | test turnover "
              "| test Sharpe (ann.) | promoted |",
              "|---|---:|---|---|---:|---:|---|---|---:|---:|---:|---|"]
    for a in result["arms"]:
        if a["status"] != "RUN":
            lines.append(f"| {a['arm']} | {a['status']}: {a.get('reason', '')} | | | | | | | | | | |")
            r = a.get("outcome_range")
            if r:
                lines += ["", f"Pending LLM arm, every outcome its {r['k']} cards could produce on these trials: "
                          f"test net {bps(r['test_net_mean'][0])} to {bps(r['test_net_mean'][1])} bps/day, test IC "
                          f"{r['test_ic_mean'][0]:+.4f} to {r['test_ic_mean'][1]:+.4f}; promotable selections: "
                          f"{', '.join(r['promotable']) or 'none'}.", ""]
            continue
        v, t, d = a["validation"], a["test"], a["validation"]["deflated_sharpe"]
        dsr = "n/a" if d is None else f"{d['headline']['range'][0]:.2f}-{d['headline']['range'][1]:.2f}"
        lines.append(
            f"| {a['arm']} | {a['hypotheses_tested_on_validation']} | {a['selected']} "
            f"| {fmt(v['ic_mean'], '+.4f')} ({fmt(v['ic_t'], '+.2f')}, {fmt(v['ic_t_nw'], '+.2f')}) | {bps(v['net_mean'])} "
            f"| {fmt(v['ic_p_bonferroni'], '.3f')} | {dsr} | {fmt(t['ic_mean'], '+.4f')} ({fmt(t['ic_t'], '+.2f')}, "
            f"{fmt(t.get('ic_p_nw'), '.3f')}) "
            f"| {bps(t['net_mean'])} ({fmt(t.get('net_p_nw'), '.3f')}) | {fmt(t['turnover_mean'], '.3f')} "
            f"| {fmt(t['net_sharpe_annual'], '+.2f')} | {'yes' if a['promoted'] else 'no'} |")
    if "verdict" in result:
        v = result["verdict"]
        lines += ["", f"Verdict (protocol v2, in code): {v['rule']} Arm helps: "
                  + ", ".join(f"{k} {'yes' if h else 'no'}" for k, h in v["arm_helps"].items())
                  + "".join(f"; LLM minus {k}: {bps(c['mean_difference'])} bps/day, Newey-West p "
                            f"{fmt(c['p_nw'], '.3f')}" for k, c in v["llm_vs"].items())
                  + f". LLM helps: {'pending' if v['llm_helps'] is None else 'yes' if v['llm_helps'] else 'no'}."]
    holm_p = {a["arm"]: a["test"].get("p_holm_across_arms") for a in result["arms"]
              if a["status"] == "RUN" and a["test"].get("p_holm_across_arms") is not None}
    if holm_p:
        lines += ["", "Holm-adjusted test p across arms: " + ", ".join(f"{k} {v:.3f}" for k, v in holm_p.items()) + "."]
    for a in result["arms"]:
        if "distribution" in a:
            d = a["distribution"]
            lines += ["", f"Random-K over {d['draws']} further draws: selection's test IC median "
                      f"{d['test_ic_mean']['median']:+.4f} (5-95%: {d['test_ic_mean']['p05']:+.4f} to "
                      f"{d['test_ic_mean']['p95']:+.4f}); test net median {bps(d['test_net_mean']['median'])} bps/day; "
                      f"test net > 0 in {d['share_test_net_positive']:.1%}; promoted in {d['share_promoted']:.1%}."]
    lines += ["", "## Baselines (fixed, not selected)", "",
              "| baseline | val IC | val net bps/day | test IC (t) | test net bps/day | test turnover | test Sharpe (ann.) |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for name, b in result["baselines"].items():
        v, t = b["validation"], b["test"]
        lines.append(f"| {name} | {fmt(v['ic_mean'], '+.4f')} | {bps(v['net_mean'])} | {fmt(t['ic_mean'], '+.4f')} "
                     f"({fmt(t['ic_t'], '+.2f')}) | {bps(t['net_mean'])} | {fmt(t['turnover_mean'], '.3f')} "
                     f"| {fmt(t['net_sharpe_annual'], '+.2f')} |")
    c = result["counts"]
    lines += ["", f"Best baseline on test: {result['best_baseline_on_test']}. Distinct hypotheses tested: "
              f"{c['distinct_hypotheses_tested']}; trials run and recomputed: {c['trials_run_and_recomputed']}.",
              "", "## Every hypothesis", "",
              "| id | val IC (t) | val net bps/day | test IC (t) | test net bps/day | test turnover | test sleeve net bps/day |",
              "|---|---|---:|---|---:|---:|---:|"]
    for r in rows:
        v, t = r["validation"], r["test"]
        lines.append(f"| {r['id']} | {fmt(v['ic_mean'], '+.4f')} ({fmt(v['ic_t'], '+.2f')}) | {bps(v['net_mean'])} "
                     f"| {fmt(t['ic_mean'], '+.4f')} ({fmt(t['ic_t'], '+.2f')}) | {bps(t['net_mean'])} "
                     f"| {fmt(t['turnover_mean'], '.3f')} | {bps(t['sleeve_net_mean'])} |")
    return "\n".join(lines) + "\n"


def show(results_dir):
    """The summary tables of a results directory (the per-hypothesis table stays in REPORT.md)."""
    report = (Path(results_dir) / "REPORT.md").read_text(encoding="utf-8")
    return report.split("\n## Every hypothesis")[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify-data")
    check.add_argument("--data", required=True)
    check.add_argument("--universe", required=True)
    go = commands.add_parser("run")
    for flag in ("--protocol", "--prices", "--contract", "--work", "--out"):
        go.add_argument(flag, required=True)
    arm = go.add_mutually_exclusive_group(required=True)
    arm.add_argument("--llm-run", help="a quantos-loop run made with --live on the same prices and contract")
    arm.add_argument("--llm-pending", help="why there is no live LLM run yet (recorded as the LLM arm)")
    go.add_argument("--receipt-keys")
    go.add_argument("--workers", type=int)
    freeze = commands.add_parser("select", help="freeze the selections the data in hand already determines")
    for flag in ("--protocol", "--prices", "--contract", "--work", "--out"):
        freeze.add_argument(flag, required=True)
    freeze.add_argument("--workers", type=int)
    fill = commands.add_parser("fill", help="protocol v2 delisting rule: extend every pair to --through")
    fill.add_argument("--data", required=True)
    fill.add_argument("--through", required=True, help="last date, YYYY-MM-DD")
    fill.add_argument("--out", required=True)
    view = commands.add_parser("show")
    view.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-data":
            print(json.dumps(verify_data(args.data, args.universe), indent=2, sort_keys=True))
        elif args.command == "run":
            result = run(args.protocol, args.prices, args.contract, args.work, args.out, llm_run=args.llm_run,
                         llm_pending=args.llm_pending, receipt_keys=args.receipt_keys, workers=args.workers)
            print(json.dumps({"out": args.out, "counts": result["counts"]}, indent=2, sort_keys=True))
        elif args.command == "fill":
            print(json.dumps(fill_missing(args.data, args.through, args.out), indent=2, sort_keys=True))
        elif args.command == "select":
            body = freeze_selections(args.protocol, args.prices, args.contract, args.work, args.out, args.workers)
            print(json.dumps(body["selections"], indent=2, sort_keys=True))
        else:
            print(show(args.results))
        return 0
    except (ValueError, OSError, LookupError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
