"""Deterministic baseline features, rank IC and independent one-period sleeves."""
import math
from .contracts import iso


def average_ranks(values):
    """Ascending 1-based average ranks for exact ties; O(N log N)."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for pos in range(start, end):
            ranks[order[pos]] = rank
        start = end
    return ranks


def rank_ic(features, labels):
    if len(features) != len(labels):
        raise ValueError("rank IC vectors must have equal lengths")
    pairs = [(x, y) for x, y in zip(features, labels) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x, y = (average_ranks([p[i] for p in pairs]) for i in (0, 1))
    center = (len(pairs) + 1) / 2
    x, y = ([v - center for v in values] for values in (x, y))
    xx, yy = math.fsum(v * v for v in x), math.fsum(v * v for v in y)
    if xx == 0 or yy == 0:
        return None
    return max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(x, y)) / math.sqrt(xx*yy)))


def known_price(panel, at, asset, decision):
    row = panel.get((at, asset))
    return row.price if row is not None and row.available_at <= decision else None


def change(end, start):
    value = end / start - 1
    if not math.isfinite(value):
        raise ValueError("derived return is not finite")
    return value


def build_features(panel, config, factor):
    rows = []
    for i, decision in enumerate(config.calendar):
        values = {}
        for asset in config.assets:
            value = None
            if i >= config.lookback + 1:
                end = known_price(panel, config.calendar[i-1], asset, decision)
                start = known_price(panel, config.calendar[i-1-config.lookback], asset, decision)
                if end is not None and start is not None:
                    value = change(end, start) * (1 if factor == "momentum" else -1)
            values[asset] = value
        rows.append(values)
    return rows


def target_weights(signals, entry_prices):
    """Weights use signals and current marks only, never label availability."""
    eligible = [a for a in signals if signals[a] is not None and entry_prices[a] is not None]
    weights = {a: 0.0 for a in signals}
    ranks = average_ranks([signals[a] for a in eligible])
    centered = [r - (len(eligible)+1)/2 for r in ranks]
    scale = math.fsum(abs(x) for x in centered)
    if scale:
        weights.update({a: value/scale for a, value in zip(eligible, centered)})
    return weights


def sleeve(weights, returns, cost_bps):
    """Fixed unit notional, full entry/exit each interval, no reinvestment."""
    held = [a for a, w in weights.items() if w != 0]
    if any(returns[a] is None for a in held):
        return dict(status="MISSING_EXIT_MARK", gross=None, turnover=None, cost=None, net=None)
    gross = math.fsum(weights[a]*returns[a] for a in held)
    turnover = math.fsum(abs(weights[a])*(2 + returns[a]) for a in held)
    cost = turnover * cost_bps / 10000
    if not all(math.isfinite(x) for x in (gross, turnover, cost)):
        raise ValueError("portfolio arithmetic is not finite")
    return dict(status="ACTIVE" if held else "ABSTAIN", gross=gross,
                turnover=turnover, cost=cost, net=gross-cost)


def evaluate_split(panel, config, features, split):
    start, end = config.splits[split]
    daily = []
    for i in range(start, end):  # last decision excluded: label would cross split end
        now, later = config.calendar[i], config.calendar[i+1]
        signals = features[i]
        entry = {a: known_price(panel, now, a, now) for a in config.assets}
        exits = {a: known_price(panel, later, a, later) for a in config.assets}
        returns = {a: change(exits[a], entry[a]) if exits[a] is not None and entry[a] is not None
                   else None for a in config.assets}
        weights = target_weights(signals, entry)
        feature_count = sum(v is not None for v in signals.values())
        pair_count = sum(signals[a] is not None and returns[a] is not None for a in config.assets)
        daily.append(dict(time=iso(now), label_time=iso(later),
                          rank_ic=rank_ic(list(signals.values()), list(returns.values())),
                          feature_count=feature_count, ic_pair_count=pair_count,
                          feature_coverage=feature_count/len(config.assets),
                          ic_coverage=pair_count/len(config.assets),
                          entry_eligible_count=sum(signals[a] is not None and entry[a] is not None for a in config.assets),
                          weights=weights, portfolio=sleeve(weights, returns, config.cost_bps)))
    ics = [d["rank_ic"] for d in daily if d["rank_ic"] is not None]
    complete = all(d["portfolio"]["net"] is not None for d in daily)
    mean = lambda xs: math.fsum(xs)/len(xs) if xs else None
    return dict(daily=daily, excluded_label_boundary_dates=[iso(config.calendar[end])], summary=dict(
        intervals=len(daily), valid_ic_dates=len(ics), rank_ic_mean=mean(ics),
        feature_coverage_mean=mean([d["feature_coverage"] for d in daily]),
        ic_coverage_mean=mean([d["ic_coverage"] for d in daily]),
        active_intervals=sum(d["portfolio"]["status"] == "ACTIVE" for d in daily),
        missing_exit_intervals=sum(d["portfolio"]["status"] == "MISSING_EXIT_MARK" for d in daily),
        portfolio_complete=complete,
        gross_mean=mean([d["portfolio"]["gross"] for d in daily]) if complete else None,
        net_mean=mean([d["portfolio"]["net"] for d in daily]) if complete else None,
        turnover_mean=mean([d["portfolio"]["turnover"] for d in daily]) if complete else None,
        cost_mean=mean([d["portfolio"]["cost"] for d in daily]) if complete else None))


def evaluate(panel, config):
    features = {f: build_features(panel, config, f) for f in config.candidates}
    development = {f: {s: evaluate_split(panel, config, features[f], s) for s in ("train", "validation")}
                   for f in config.candidates}
    scored = [f for f in config.candidates if development[f]["validation"]["summary"]["rank_ic_mean"] is not None]
    selected = max(scored, key=lambda f: development[f]["validation"]["summary"]["rank_ic_mean"]) if scored else None
    test = evaluate_split(panel, config, features[selected], "test") if selected else None
    return dict(schema_version="factor-report/v1", universe=list(config.assets),
                observation_rows=len(panel), expected_panel_rows=len(config.assets)*len(config.calendar),
                missing_panel_rows=len(config.assets)*len(config.calendar)-len(panel),
                null_prices=sum(r.price is None for r in panel.values()),
                delayed_rows=sum(r.available_at > t for (t, _), r in panel.items()),
                feature_rows={f: [dict(time=iso(t), values=v) for t, v in zip(config.calendar, values)]
                              for f, values in features.items()},
                development=development,
                selection=dict(method="highest validation mean rank IC; configured order breaks ties",
                               selected=selected, uses_test=False,
                               status="SELECTED" if selected else "NO_VALIDATION_IC"),
                test=test,
                units=dict(rank_ic="unitless Spearman correlation with average ties",
                           coverage="fraction of configured static universe",
                           gross="P&L per unit fixed initial gross notional per independent interval",
                           turnover="entry plus price-adjusted exit traded notional per unit",
                           cost="turnover * cost_bps / 10000", net="gross minus modeled roundtrip cost"),
                limitations=["New synthetic/offline research software; no platform equivalence or historical-work claim.",
                             "Idealized close-to-close marks, fractional positions, no cross-date netting or compounding.",
                             "No borrow, financing, market impact, execution availability or significance qualification.",
                             "Static configured universe; no survivorship-free external-data claim.",
                             "Test is exposed once reported; reruns must retain trial history, not call it untouched."])
