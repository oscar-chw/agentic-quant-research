"""How a strategy's results are judged. One procedure for every study (research/reviews.md R4).

- Splits are frozen here, never derived from whatever the cache holds today (R4.10). The days
  2026-09-13…09-16 were looked at by several analyses before any rule was fixed, so they are an
  exploratory holdout, not a confirmation set (R4.3). Confirmation starts 2026-09-19 00:00 UTC (confirmation v1 from 2026-09-17 was retired before its analysis,
  research/confirmation_v1/SUPERSEDED.md).
- One return series (R4.6): every traded window stakes the same dollars; P&L per $, Sharpe, PSR
  and intervals are all computed on that series.
- Selection vs confirmation (R4.1): the deflated Sharpe applies to the Sharpe of the variant that
  was selected, on the data it was selected on, with that selection's trial count. On held-out
  data nothing is selected, so each pre-registered trading leg gets a one-sided p-value from its own
  return shape (shape_pvalue: outcome null) and the p-values of all legs evaluated on the same days
  are Holm-corrected together (holm_confirm; R4.2, R4.9, audit R6 findings 9 and 11).
- Intervals: outcome_ci (inversion of the outcome test) for trading legs, bootstrap_t_ci for plain
  returns (audit R6 finding 11); both resample or redraw single windows, which is exact when outcomes
  are independent given the prices paid.
- Dependent series (the confirmation's prediction legs, audit R6 finding 1): dependent_mean_test, the
  series' own low-frequency long-run variance with degrees of freedom from the dependence length
  measured beforehand (dependence_block); the 2 h block bootstrap and batch means are sensitivity only.
- Power: windows_needed (the PSR formula) and windows_needed_sim (the leg's own test, simulated).
"""
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from pmlab import metrics, stats

EXPLORE_END = int(datetime(2026, 9, 13, tzinfo=timezone.utc).timestamp())       # train < this
CONFIRM_START = int(datetime(2026, 9, 19, tzinfo=timezone.utc).timestamp())     # untouched from here
USED_HOLDOUT = (EXPLORE_END, CONFIRM_START)


def split(starts) -> dict[str, list[int]]:
    starts = sorted(starts)
    return {"train": [s for s in starts if s < EXPLORE_END],
            "used_holdout": [s for s in starts if EXPLORE_END <= s < CONFIRM_START],
            "confirm": [s for s in starts if s >= CONFIRM_START]}


def fixed_stake_returns(pnl_per_dollar) -> np.ndarray:
    """Per traded window: P&L of a constant $1 stake (scale-free)."""
    return np.asarray(pnl_per_dollar, dtype=float)


def psr_pvalue(returns) -> float:
    """One-sided p-value of H0: per-window Sharpe ≤ 0 (1 − PSR), non-normal moments included."""
    s = metrics.summary(returns)
    if s["n"] < 3 or not np.isfinite(s["sharpe"]):
        return np.nan
    return float(1 - s["psr"])


def holm_confirm(legs: dict, alpha: float = 0.05, missing_as_one: bool = False, n_boot: int = 9999,
                 seed: int = 0) -> dict[str, dict]:
    """One-sided p-value per pre-registered leg from its own return shape (shape_pvalue: outcome null for a leg given
    as per-window up, down, cost, fees, pnl; recentred bootstrap-t for plain returns), Holm-adjusted across all legs
    tested on the same data. The PSR p-value it replaced (psr_p, kept for reference) put the Holm family-wise false
    positive rate over the 16 walk-forward legs at 7.8% (4.4% with this test) and read a maker's week without a loss as
    proof (audit R6 findings 9, 11).

    missing_as_one: a registered leg with no p-value (too few returns, never traded) counts in the family
    size with p = 1 instead of being left out, so an untestable leg cannot make the others easier to confirm
    (research/robustness_audit.md finding 8; research/preregistration_clarifications.md)."""
    names = list(legs)
    tests = [shape_pvalue(legs[n], n_boot=n_boot, seed=seed) for n in names]
    ps = [p for p, _ in tests]
    if missing_as_one:
        ps = [1.0 if p is None or not np.isfinite(p) else p for p in ps]
    reject, adjusted = stats.holm(ps, alpha)
    return {n: {"n": int(len(leg_returns(legs[n]))), "p": p, "p_holm": a, "significant": bool(r), "test": kind,
                "psr_p": psr_pvalue(leg_returns(legs[n]))}
            for n, p, a, r, (_, kind) in zip(names, ps, adjusted, reject, tests)}


def selection_dsr(selected_returns, trial_sharpes, trial_n=None, min_n: int = 30) -> float:
    """Deflated Sharpe of the variant selected on this data, against its own trials (AFML 14.7.3).

    With `trial_n` (returns behind each trial Sharpe), trials of different sizes are put on one scale:
    z_k = SR_k·√(n_k − 1) has variance 1 under no skill whatever n_k, so V[SR] at the selection's n is
    var(z_k)/(n − 1). N is every trial tried (len(trial_sharpes), short trials and trials without a Sharpe
    included): the book and Bailey & López de Prado count the trials of the search, and the selection may pick a
    short one (docs/afml_sections/ch14.md F14.6). Only V leaves out trials with fewer than `min_n` returns: their
    Sharpes are mostly noise and would otherwise set the benchmark (research/book_check_part3.md finding 1).
    Without `trial_n`, the raw trial Sharpes are used as before."""
    s = metrics.summary(selected_returns)
    if s["n"] < 3 or not np.isfinite(s["sharpe"]):
        return np.nan
    if trial_n is None:
        trials = [x for x in trial_sharpes if np.isfinite(x)]
        if len(trials) < 2:
            return np.nan
        return float(metrics.deflated_sharpe_ratio(s["sharpe"], s["n"], s["skew"], s["kurtosis"], trials))
    z = [sr * np.sqrt(n - 1) for sr, n in zip(trial_sharpes, trial_n)
         if sr is not None and np.isfinite(sr) and n is not None and n >= min_n]
    if len(z) < 2:
        return np.nan
    sr_star = metrics.expected_max_sharpe(len(trial_sharpes), np.var(z, ddof=1) / (s["n"] - 1))
    return float(metrics.probabilistic_sharpe_ratio(s["sharpe"], s["n"], s["skew"], s["kurtosis"], sr_star))


def block_bootstrap_ci(times, values, stat=np.mean, block_seconds: int = 7200, n_boot: int = 2000,
                       level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile interval of stat(values) resampling contiguous time blocks with replacement."""
    times, values = np.asarray(times), np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    times, values = times[ok], values[ok]
    if len(values) < 3:
        return (np.nan, np.nan)
    blocks = times // block_seconds
    groups = [values[blocks == b] for b in np.unique(blocks)]
    if len(groups) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        draws.append(stat(np.concatenate([groups[i] for i in pick])))
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def block_bootstrap_p(times, values, block_seconds: int = 7200, n_boot: int = 2000, seed: int = 0) -> float:
    """One-sided p-value that mean(values) < 0 is not chance: share of centred block-bootstrap means
    at or below the observed mean (blocks as in block_bootstrap_ci)."""
    times, values = np.asarray(times), np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    times, values = times[ok], values[ok]
    blocks = times // block_seconds
    groups = [values[blocks == b] for b in np.unique(blocks)]
    if len(values) < 3 or len(groups) < 2:
        return np.nan
    observed, centre = float(np.mean(values)), float(np.mean(values))
    rng = np.random.default_rng(seed)
    draws = np.array([np.mean(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) - centre
                      for _ in range(n_boot)])
    return float((1 + np.sum(draws <= observed)) / (n_boot + 1))


def windows_needed(sharpe: float, skew: float = 0.0, kurtosis: float = 3.0, alpha: float = 0.05,
                   power: float = 0.8) -> float:
    """Traded windows for the PSR test (AFML 14.7.2) to detect a per-window Sharpe with this power.

    PSR rejects when SR̂·√(n − 1)/σ ≥ z_{1−α}, σ² = 1 − γ₃·SR + (γ₄ − 1)/4·SR² (γ₄ non-excess), so
    n = 1 + σ²·(z_{1−α} + z_power)²/SR². Skewed, fat-tailed returns (a maker's rare total loss) need far
    more windows than the normal formula says (research/book_check_part3.md finding 3)."""
    if not sharpe or sharpe <= 0 or not np.isfinite(sharpe):
        return np.inf
    var = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2
    return float(1 + max(var, 0.0) * ((norm.ppf(1 - alpha) + norm.ppf(power)) / sharpe) ** 2)


def verdict(train_dsr: float, confirm: dict | None) -> str:
    """Candidate only when selection survives deflation AND held-out confirmation survives Holm."""
    if confirm is None:
        return "untested on held-out data"
    if np.isfinite(train_dsr) and train_dsr >= 0.95 and confirm["significant"]:
        return "confirmed"
    return "not confirmed"


# ----------------------------------------------------------------------------- audit R6: dependence and return shape
# Findings 1, 9, 11, 12 (research/audit_r6_findings.md). A 2 h block bootstrap of a per-window difference whose
# dependence lasts days rejects a true zero mean several times too often; the PSR's normal approximation and
# Var(SR̂) = 1/(n − 1) are wrong for two-point lottery returns. The replacements below take the dependence length
# from earlier data and the null distribution from each leg's own return shape.

VIF_HOURS = (0.5, 2, 6, 12, 24, 48, 96)


def variance_inflation(times, values, hours=VIF_HOURS) -> dict:
    """Per block length L hours: blocks are the contiguous spans [k·L, (k + 1)·L) of `times` holding data and
    vif = Var(block means, ddof 1) × mean values per block ÷ Var(values, ddof 1): ≈ 1 for independent values, and it
    grows with L while neighbouring blocks stay correlated. NaN with fewer than 3 blocks."""
    t, x = np.asarray(times), np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    base = np.var(x, ddof=1) if len(x) > 1 else np.nan
    out = {}
    for h in hours:
        _, inv, counts = np.unique(t // int(round(h * 3600)), return_inverse=True, return_counts=True)
        means = np.bincount(inv, x) / counts
        vif = float(np.var(means, ddof=1) * counts.mean() / base) if len(means) >= 3 and base > 0 else np.nan
        out[h] = {"vif": vif, "blocks": int(len(means))}
    return out


def dependence_block(times, values, frac: float = 0.95, hours=VIF_HOURS) -> dict:
    """Where the dependence of `values` runs out, measured on earlier data: the shortest block length L (hours) whose
    variance inflation reaches `frac` × the largest over `hours`. Returns {"hours": L, "vif": vif(L), "blocks": number
    of L-blocks in this data, "curve": {L: vif}}. dependent_mean_test takes L; on 18 days of earlier data L is noisy
    and usually long (96 h in 42% of runs of a 4 h latent half-life), which that test only turns into lost power."""
    curve = variance_inflation(times, values, hours)
    finite = {h: c["vif"] for h, c in curve.items() if np.isfinite(c["vif"])}
    if not finite:
        return {"hours": np.nan, "vif": np.nan, "blocks": 0, "curve": {}}
    top = max(finite.values())
    L = next(h for h in hours if h in finite and finite[h] >= frac * top)
    return {"hours": L, "vif": finite[L], "blocks": curve[L]["blocks"], "curve": {str(h): v for h, v in finite.items()}}


EWC_BLOCKS = 8                 # blocks of the measured dependence length per cosine term (degree of freedom)
EWC_MIN_DF = 4                 # the fewest terms: the most power once the long-dependence correction below applies
DEPENDENCE_MAX_HOURS = 60      # the longest latent half-life the correction assumes (the size holds to 48 h with margin)


def dependent_mean_test(times, values, hours: float, level: float = 0.95,
                        max_hours: float = DEPENDENCE_MAX_HOURS) -> dict:
    """Test and interval for mean(values) of a dependent series, `hours` its dependence length measured beforehand
    (dependence_block on earlier data). The variance of the mean comes from these values: the equal-weighted cosine
    (EWC) long-run variance (Lazarus, Lewis, Stock & Watson 2018), lrv = mean_j Λ_j² / b, Λ_j = √(2/n)·Σ_i
    cos(πj·u_i)·(x_i − x̄) with u_i = (t_i − t_1 + Δ/2)/S (S the span, Δ the median time step), for j = 1…ν,
    ν = max(EWC_MIN_DF, ⌊S / (EWC_BLOCKS·hours)⌋) (at most n − 1); b = mean_j 1/(1 + (πj·τ/S)²) corrects the terms'
    shortfall below the zero-frequency variance for a latent AR(1) with time constant τ = min(hours, max_hours)/ln 2.
    se = √(lrv/n), t = mean/se against Student t with ν df. p: one-sided p-value of H0 mean ≥ 0 (small for a negative mean); p_positive: of
    H0 mean ≤ 0; ci: two-sided.

    Size over 35 analysed days, dependence measured on 18 earlier days (the audit's slow-latent null series; 5,000 runs
    each, Monte Carlo sd 0.3 pt): 3.5%, 3.2%, 3.3%, 3.7%, 4.9% at latent half-lives 4, 8, 12, 24, 48 h, where
    vif_mean_test (the variance inflation and df of the earlier span) rejected 3.6%, 4.3%, 5.4%, 8.3% and 12.9%
    (tests/test_evaluation.py; research/code_review/data_eval.md, Resolution). Power at the effect an oracle one-sided
    test detects 80% of the time: 53–63% (vif_mean_test's 64–87% came with that excess size)."""
    t, x = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(t)
    order = np.argsort(t[ok], kind="stable")
    t, x = t[ok][order], x[ok][order]
    n = len(x)
    nan = {"n": n, "mean": float(np.mean(x)) if n else np.nan, "se": np.nan, "t": np.nan, "df": None, "hours": hours,
           "bias": np.nan, "p": np.nan, "p_positive": np.nan, "ci": (np.nan, np.nan)}
    if n < 3 or hours is None or not np.isfinite(hours) or hours <= 0:
        return nan
    steps = np.diff(np.unique(t))
    step = float(np.median(steps)) if len(steps) else 1.0
    span = float(t[-1] - t[0] + step)
    nu = int(min(max(EWC_MIN_DF, span // (EWC_BLOCKS * hours * 3600)), n - 1))
    u, xc = (t - t[0] + step / 2) / span, x - x.mean()
    j = np.arange(1, nu + 1)
    lam = np.concatenate([np.sqrt(2 / n) * (np.cos(np.pi * np.outer(j[k:k + 16], u)) @ xc) for k in range(0, nu, 16)])
    tau = min(hours, max_hours) * 3600 / np.log(2)
    bias = float(np.mean(1 / (1 + (np.pi * j * tau / span) ** 2)))
    m, se = float(np.mean(x)), float(np.sqrt(np.mean(lam ** 2) / bias / n))
    if not se > 0:
        return {**nan, "se": se, "df": nu, "bias": bias}
    stat, q = m / se, float(student_t.ppf(0.5 + level / 2, nu))
    return {"n": n, "mean": m, "se": se, "t": float(stat), "df": nu, "hours": float(hours), "bias": bias,
            "p": float(student_t.cdf(stat, nu)), "p_positive": float(student_t.sf(stat, nu)), "ci": (m - q * se, m + q * se)}


def vif_mean_test(values, vif: float, df: int, level: float = 0.95) -> dict:
    """Sensitivity only (the primary test until the data_eval review): se = √(max(vif, 1)·s²/n) with the variance
    inflation measured on earlier data and s² the variance of these values, t = mean/se against Student t with df =
    that block length's blocks in the earlier data − 1. Over 35 analysed days it rejects a true zero mean in 8.3% and
    12.9% of runs at 24 h and 48 h latent half-lives (dependent_mean_test).
    p: one-sided p-value of H0 mean ≥ 0 (small for a negative mean); p_positive: of H0 mean ≤ 0; ci: two-sided."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    nan = {"n": n, "mean": float(np.mean(x)) if n else np.nan, "se": np.nan, "t": np.nan, "df": df, "vif": vif,
           "p": np.nan, "p_positive": np.nan, "ci": (np.nan, np.nan)}
    if n < 3 or not df or df < 1 or not np.isfinite(vif):
        return nan
    m, s2 = float(np.mean(x)), float(np.var(x, ddof=1))
    se = float(np.sqrt(max(vif, 1.0) * s2 / n))
    if not se > 0:
        return {**nan, "se": se}
    t, q = m / se, float(student_t.ppf(0.5 + level / 2, df))
    return {"n": n, "mean": m, "se": se, "t": float(t), "df": int(df), "vif": float(vif), "p": float(student_t.cdf(t, df)),
            "p_positive": float(student_t.sf(t, df)), "ci": (m - q * se, m + q * se)}


def batch_means_p(times, values, hours: float, origin: int = 0) -> float:
    """Sensitivity only: one-sided p-value (H0 mean ≥ 0) from the weighted means of blocks [origin + k·L, origin +
    (k + 1)·L), cluster variance with G/(G − 1), Student t with G − 1 df."""
    t, x = np.asarray(times), np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    _, inv, w = np.unique((t - origin) // int(round(hours * 3600)), return_inverse=True, return_counts=True)
    G, N = len(w), len(x)
    if G < 2:
        return np.nan
    s = np.bincount(inv, x)
    m = s.sum() / N
    var = G / (G - 1) * np.sum((s - w * m) ** 2) / N ** 2
    return float(student_t.cdf(m / np.sqrt(var), G - 1)) if var > 0 else np.nan


TIE = 1e-9        # relative tolerance for "at least as extreme" (float noise only; see _observed_returns)


def _t_rows(r: np.ndarray) -> np.ndarray:
    """√n·mean/sd (ddof 1) per row; ±inf when a row's sd is 0 and its mean is not, 0 when both are."""
    n = r.shape[-1]
    m, sd = r.mean(axis=-1), r.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, m / sd * np.sqrt(n), np.sign(m) * np.inf)
    return np.nan_to_num(t, nan=0.0, posinf=np.inf, neginf=-np.inf)


def zero_edge_up_probability(up, down, cost, fees) -> np.ndarray:
    """Per traded window, the P(Up) at which its expected P&L is zero: up·p + down·(1 − p) = cost + fees, clipped to
    [0, 1] (a window that loses whatever happens stays negative: conservative). NaN when up = down (no outcome risk)."""
    up, down, cost, fees = (np.asarray(v, dtype=float) for v in (up, down, cost, fees))
    d = up - down
    p = np.divide(cost + fees - down, d, out=np.full(d.shape, np.nan), where=np.abs(d) > 1e-9)
    return np.clip(p, 0.0, 1.0)


def _outcome_draws(leg, n_draws: int, rng, windows=None) -> np.ndarray:
    """Zero-edge per-window returns, one row per draw: each window keeps its shares, stake and fees; its outcome is
    redrawn with zero_edge_up_probability (a window without outcome risk returns 0). `windows` (draws × n) picks
    windows by index, for resampled window sets."""
    up, down, cost, fees = (np.asarray(leg[k], dtype=float) for k in ("up", "down", "cost", "fees"))
    p = zero_edge_up_probability(up, down, cost, fees)
    if windows is None:
        windows = np.broadcast_to(np.arange(len(up)), (n_draws, len(up)))
    y = rng.random(windows.shape) < np.nan_to_num(p)[windows]
    ret = (np.where(y, up[windows], down[windows]) - cost[windows] - fees[windows]) / cost[windows]
    return np.where(np.isnan(p)[windows], 0.0, ret)


def _observed_returns(leg) -> np.ndarray:
    """pnl / cost per window, with a P&L that equals the win or the loss payoff to within recording rounding (shares
    4 dp, fees 5 dp) replaced by that payoff exactly, so a zero-edge draw of the same outcomes ties with it."""
    up, down, cost, fees, pnl = (np.asarray(leg[k], dtype=float) for k in ("up", "down", "cost", "fees", "pnl"))
    won, lost = up - cost - fees, down - cost - fees
    tol = 1e-3 + 2e-4 * (up + down)
    exact = np.where(np.abs(pnl - won) <= tol, won, np.where(np.abs(pnl - lost) <= tol, lost, pnl))
    return exact / cost


def is_outcome_leg(leg) -> bool:
    return hasattr(leg, "keys") and {"up", "down", "cost", "fees", "pnl"} <= set(leg.keys())


def leg_returns(leg) -> np.ndarray:
    """Per traded window P&L per $ staked: pnl / cost for an outcome leg (up, down, cost, fees, pnl), else the values."""
    if is_outcome_leg(leg):
        cost = np.asarray(leg["cost"], dtype=float)
        r = np.divide(np.asarray(leg["pnl"], dtype=float), cost, out=np.full(cost.shape, np.nan), where=cost > 0)
    else:
        r = np.asarray(leg, dtype=float)
    return r[np.isfinite(r)]


def shape_pvalue(leg, n_boot: int = 9999, seed: int = 0, chunk: int = 2_000_000) -> tuple[float, str]:
    """One-sided p-value of H0 "no edge" (mean per-window return ≤ 0) from the leg's own return shape, with the
    statistic t = √n·mean/sd of its per-window returns (findings 9, 11):
    - outcome leg (per window up and down shares, cost, fees, pnl): "outcome" null — every traded window keeps its
      shares, stake and fees and its outcome is redrawn at the zero-edge probability; exact when outcomes are
      independent given the prices paid, so a maker's week without a loss is judged against how often zero-skill
      makers have one;
    - plain returns: "recentred bootstrap-t" null — windows resampled with replacement from returns − their mean.
    p = (1 + #{t* ≥ t}) / (n_boot + 1). NaN with fewer than 3 windows."""
    rng = np.random.default_rng(seed)
    if is_outcome_leg(leg):
        cost = np.asarray(leg["cost"], dtype=float)
        keep = np.isfinite(np.asarray(leg["pnl"], dtype=float)) & (cost > 0)
        leg = {k: np.asarray(leg[k], dtype=float)[keep] for k in ("up", "down", "cost", "fees", "pnl")}
        r, kind = _observed_returns(leg), "outcome"
    else:
        r, kind = leg_returns(leg), "recentred bootstrap-t"
    n = len(r)
    if n < 3:
        return np.nan, kind
    t_obs = _t_rows(r[None, :])[0]
    t_obs = t_obs - TIE * max(1.0, abs(t_obs)) if np.isfinite(t_obs) else t_obs    # recorded P&L is rounded: ties count
    hits, done, step = 0, 0, max(1, chunk // n)
    centred = r - r.mean()
    while done < n_boot:
        b = min(step, n_boot - done)
        if kind == "outcome":
            draws = _outcome_draws(leg, b, rng)
        else:
            draws = centred[rng.integers(0, n, (b, n))]
        hits += int(np.sum(_t_rows(draws) >= t_obs))
        done += b
    return float((1 + hits) / (n_boot + 1)), kind


def bootstrap_t_ci(returns, level: float = 0.95, n_boot: int = 9999, seed: int = 0) -> tuple[float, float]:
    """Percentile-t interval for the mean per-window return (finding 11): windows resampled with replacement,
    t* = (mean* − mean)/se*, interval [mean − q(1 − a/2)·se, mean − q(a/2)·se]. It follows skew, where the percentile
    interval of a 2 h block bootstrap covered a true zero mean only 69–94% of the time on the walk-forward legs."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m, se = float(r.mean()), float(r.std(ddof=1) / np.sqrt(n))
    if not se > 0:
        return (np.nan, np.nan)
    tb = np.empty(0)
    step = max(1, 2_000_000 // n)
    while len(tb) < n_boot:
        s = r[rng.integers(0, n, (min(step, n_boot - len(tb)), n))]
        mb, sb = s.mean(axis=1), s.std(axis=1, ddof=1) / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = np.concatenate([tb, np.where(sb > 0, (mb - m) / sb, np.where(mb == m, 0.0, np.sign(mb - m) * np.inf))])
    lo_q, hi_q = np.quantile(tb, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(m - hi_q * se), float(m - lo_q * se)


def outcome_ci(leg, level: float = 0.95, n_boot: int = 4000, seed: int = 0, tol: float = 1e-4) -> tuple[float, float]:
    """Interval for the mean per-window return of an outcome leg by inverting its outcome test (finding 11): under
    edge δ every window's chance of its side winning is the zero-edge probability moved by δ (clipped to [0, 1]); the
    interval holds the expected mean return of every δ whose simulated mean return (common random draws) is not beyond
    the observed one in either (1 − level)/2 tail. Exact up to simulation error when outcomes are independent given
    the prices paid and the edge is common to the leg's windows; percentile intervals of the same legs covered a true
    zero mean 71–97%, bootstrap-t 80–100%."""
    cost = np.asarray(leg["cost"], dtype=float)
    keep = np.isfinite(np.asarray(leg["pnl"], dtype=float)) & (cost > 0)
    up, down, cost, fees, pnl = (np.asarray(leg[k], dtype=float)[keep] for k in ("up", "down", "cost", "fees", "pnl"))
    n = len(up)
    if n < 3:
        return (np.nan, np.nan)
    p0 = zero_edge_up_probability(up, down, cost, fees)
    risky = np.isfinite(p0)
    side = np.sign(up - down)                                  # +1 holds more Up, −1 more Down
    observed = float(np.mean(_observed_returns({"up": up, "down": down, "cost": cost, "fees": fees, "pnl": pnl})))
    tie = TIE * max(1.0, abs(observed))
    u = np.random.default_rng(seed).random((n_boot, n))
    a = (1 - level) / 2

    def probs(delta):
        return np.clip(np.nan_to_num(p0) + side * delta, 0.0, 1.0)

    def expected(delta):
        p = probs(delta)
        return float(np.mean(np.where(risky, (up * p + down * (1 - p) - cost - fees) / cost, 0.0)))

    def tails(delta):
        p = probs(delta)
        m = np.mean(np.where(risky, (np.where(u < p, up, down) - cost - fees) / cost, 0.0), axis=1)
        return float(np.mean(m >= observed - tie)), float(np.mean(m <= observed + tie))

    def solve(ok_side):                                        # boundary δ between rejected and not rejected
        lo, hi = -1.0, 1.0
        while hi - lo > tol:
            mid = (lo + hi) / 2
            lo, hi = ok_side(mid, lo, hi)
        return (lo + hi) / 2
    d_lo = solve(lambda d, lo, hi: (d, hi) if tails(d)[0] < a else (lo, d))     # smallest δ still able to reach it
    d_hi = solve(lambda d, lo, hi: (d, hi) if tails(d)[1] >= a else (lo, d))    # largest δ not beyond it
    return expected(d_lo), expected(d_hi)


def luck_hurdle(leg, n_trials: int, n_sims: int = 20000, seed: int = 0) -> dict:
    """The best Sharpe `n_trials` zero-skill strategies with this leg's own return shape reach by luck, and the
    deflated Sharpe against it (finding 9). Zero-skill Sharpes come from outcome draws (outcome leg) or recentred
    resamples (plain returns); SR* = their mean + their sd × E[max of n_trials standard normals]; dsr = share of
    zero-skill Sharpes, moved to centre on SR*, below the observed Sharpe (the PSR against SR* with the simulated
    sampling distribution in place of Var(SR̂) = 1/(n − 1) and the normal approximation)."""
    r = leg_returns(leg)
    n = len(r)
    out = {"n": n, "sharpe": np.nan, "null_sd": np.nan, "null_mean": np.nan, "sr_star": np.nan, "dsr": np.nan}
    if n < 3:
        return out
    rng = np.random.default_rng(seed)
    step, sims = max(1, 2_000_000 // n), []
    while sum(len(s) for s in sims) < n_sims:
        b = min(step, n_sims - sum(len(s) for s in sims))
        draws = _outcome_draws(leg, b, rng) if is_outcome_leg(leg) else (r - r.mean())[rng.integers(0, n, (b, n))]
        m, sd = draws.mean(axis=1), draws.std(axis=1, ddof=1)
        sims.append(np.divide(m, sd, out=np.full(b, np.nan), where=sd > 0))
    null = np.concatenate(sims)
    null = null[np.isfinite(null)]
    sr = metrics.sharpe(r)
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
    sr_star = mu + metrics.expected_max_sharpe(n_trials, sd ** 2)
    dsr = float(np.mean(null - mu + sr_star < sr)) if np.isfinite(sr) else np.nan
    return {"n": n, "sharpe": sr, "null_sd": sd, "null_mean": mu, "sr_star": float(sr_star), "dsr": dsr,
            "naive_sd": float(1 / np.sqrt(n - 1))}


def windows_needed_sim(leg, alpha: float = 0.05, power: float = 0.8, n_sims: int = 2000, max_windows: int = 50_000,
                       seed: int = 0) -> float:
    """Traded windows the leg's own test (shape_pvalue) needs to reject H0 at `alpha` with `power` if future traded
    windows look like these (finding 12): windows resampled with replacement; the critical t is the 1 − alpha quantile
    of t from zero-edge draws of resampled windows (outcome leg) or recentred resamples (plain returns), the power the
    share of resamples of the observed returns above it. Smallest such n by doubling and bisection; inf when the
    observed mean return is ≤ 0 or more than max_windows are needed."""
    r = leg_returns(leg)
    if len(r) < 3 or not r.mean() > 0:
        return np.inf
    outcome = is_outcome_leg(leg)
    if outcome:
        cost = np.asarray(leg["cost"], dtype=float)
        keep = np.isfinite(np.asarray(leg["pnl"], dtype=float)) & (cost > 0)
        leg = {k: np.asarray(leg[k], dtype=float)[keep] for k in ("up", "down", "cost", "fees", "pnl")}
    centred = r - r.mean()

    def achieved(n: int) -> float:
        rng = np.random.default_rng(seed)
        null, alt, done, step = [], [], 0, max(1, 4_000_000 // n)
        while done < n_sims:
            b = min(step, n_sims - done)
            idx = rng.integers(0, len(r), (b, n))
            null.append(_t_rows(_outcome_draws(leg, b, rng, idx) if outcome else centred[idx]))
            alt.append(_t_rows(r[rng.integers(0, len(r), (b, n))]))
            done += b
        crit = np.quantile(np.concatenate(null), 1 - alpha)
        return float(np.mean(np.concatenate(alt) > crit))

    lo, hi = 2, 4
    while achieved(hi) < power:
        lo, hi = hi, hi * 2
        if hi > max_windows:
            return np.inf
    while hi - lo > max(1, lo // 50):
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if achieved(mid) >= power else (mid, hi)
    return float(hi)
