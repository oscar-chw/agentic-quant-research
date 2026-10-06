"""The strategy lifecycle as code: AFML ch. 1 (section 1.3.1(6): embargo, paper trading, graduation, re-allocation,
decommission), ch. 15 (strategy risk) and ch. 16 (hierarchical risk parity). research/lifecycle.md states every rule,
its bound, why, and whether the bound is the book's or ours; scripts/lifecycle.py gathers the evidence and writes
research/lifecycle_status.json.

Stages: research -> embargo -> paper -> graduated -> decommissioned.
- research: every strategy starts here; a registered leg whose registration is missing or broken, and a leg the
  confirmation did not confirm, are here too.
- embargo: a leg of a registration written before its confirmation starts, while scripts/confirm.py --check passes and
  until the one registered analysis. Its paper results on confirmation windows are sealed: never read here.
- paper: a leg the analysis confirmed (scripts/confirm.py --check --final passes). Graduation needs every
  graduation_checks rule; decommission_triggers apply from the analysis on.
- graduated: stays graduated while every graduation rule still holds, no decommission trigger fires and the final
  check passes: a "graduated" in the previous status file (writable by the user's uid) is never taken over, only its
  time, when it lies between the analysis and now; the Sharpe at graduation is recomputed from the paper windows that
  settled by that time. Its allocation is allocate() (HRP across graduated strategies, a
  concave ramp from a small start, shrunk when performance decays). Graduation is a correctness gate for real trading,
  not a security control: pmexec also requires the graduation time pinned in the root-owned enablement config.
- decommissioned: terminal for that strategy version; a new variant starts in research.

Paper evidence (`record`): one row per traded window of the strategy's paper account since CONFIRM_START with start,
pnl, cost, up, down, fees (pmlab.performance.live_windows). assess() refuses it before the analysis exists.
"""
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pmlab import evaluation, metrics
from pmlab.afml import hrp, strategy_risk

STAGES = ("research", "embargo", "paper", "graduated", "decommissioned")
DAY = 86400
TW = timezone(timedelta(hours=8))

# graduation (research/lifecycle.md, "Graduation")
MIN_PAPER_WINDOWS = 1000          # ours: floor on settled traded paper windows since CONFIRM_START
PAPER_POWER = 0.8                 # ours (project convention): windows the leg's own test needs for this power
MIN_PAPER_DAYS = 7                # ours: days in the paper stage after the analysis
TARGET_SHARPE = 2.0               # book (exercise 15.1): annualised Sharpe that separates failure from success
STRATEGY_RISK_YEARS = 2.0         # book (15.4.1): k, the years investors judge a strategy over
MAX_FAILURE_PROBABILITY = 0.05    # book (15.4.2): strategies with P[p < p*] above this are too risky
MIN_DSR = 0.95                    # book (ch. 14) confidence, the trial registry's convention
MAX_DRAWDOWN = 0.30               # ours: of the paper account's running peak
BANKROLL0 = 1000.0                # the paper engine's starting bankroll per strategy (pmlab.live.engine)

# decommission (research/lifecycle.md, "Decommission")
LOSS_MIN_WINDOWS = 100            # ours
LOSS_LEVEL = 0.90                 # two-sided interval: its upper end < 0 is one-sided 5% evidence of a loss
DECAY_DAYS = 7                    # ours: consecutive daily assessments with failure probability above the bound
DECAY_LOOKBACK_DAYS = 30          # ours
DECAY_MIN_WINDOWS = 200           # ours: fewer traded windows in a lookback is no evidence either way

# re-allocation (research/lifecycle.md, "Re-allocation")
INITIAL_ALLOCATION = 0.10         # ours: share of the HRP weight at graduation (the book: small)
RAMP_DAYS = 90                    # ours: days to the full HRP weight along sqrt (the book: concave)
HEALTH_LOOKBACK_DAYS = 30         # ours
HEALTH_MIN_WINDOWS = 200          # ours


@dataclass
class Confirmation:
    """What the checker knows about the pre-registered confirmation. registration: None until written, else
    {"version", "registered_at", "confirm_start", "legs"}; registration_check / analysis_check: (exit code, last
    output line) of scripts/confirm.py --check / --check --final; analysis: None until analysed, else
    {"run_at", "legs": {strategy: {"confirmed": bool, "p_holm": float | None}}}."""
    version: int
    design_legs: tuple
    confirm_start: int
    analysis_at: int
    registration: dict | None = None
    registration_check: tuple | None = None
    analysis: dict | None = None
    analysis_check: tuple | None = None

    @property
    def analysed(self) -> bool:
        return self.analysis is not None


def when(ts: float) -> str:
    u = datetime.fromtimestamp(ts, timezone.utc)
    return f"{u:%Y-%m-%d %H:%M} UTC = {datetime.fromtimestamp(ts, TW):%Y-%m-%d %H:%M} Taiwan"


def check(rule: str, ok: bool, value, bound, source: str) -> dict:
    return {"rule": rule, "ok": bool(ok), "value": value, "bound": bound, "source": source}


def _num(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def _leg(rows: pd.DataFrame) -> dict:
    return {k: rows[k].to_numpy(float) for k in ("up", "down", "cost", "fees", "pnl")}


def _returns(rows: pd.DataFrame) -> np.ndarray:
    return evaluation.leg_returns(_leg(rows))


# ----------------------------------------------------------------------------------------------- ch. 15 strategy risk

def failure_probability(returns, span_years: float, target_sharpe: float = TARGET_SHARPE,
                        years: float = STRATEGY_RISK_YEARS, n_boot: int = 10_000, seed: int = 0) -> dict:
    """P[p < p*] (15.4) for per-bet returns: pi_plus and pi_minus the mean win and loss, n = bets / span_years
    (15.4.1), p* the precision reaching target_sharpe at n (15.3). Two distributions of p, the larger kept: the 15.4.1
    bootstrap (floor(n k) draws) and the standard error of the precision estimated from the T observed bets (which the
    bootstrap ignores when n k > T). Ours: the two-outcome variance (pi+ - pi-)^2 p (1 - p) understates the risk of bets
    whose payouts spread (a taker's win depends on the price paid), so the target is raised by
    ratio = max(1, sd of returns / two-outcome sd): a Sharpe of target_sharpe * ratio in the two-outcome model is
    target_sharpe for the observed variance. For two-point payouts ratio = 1, the book's computation."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    pos, neg = r[r > 0], r[r <= 0]
    out = {"bets": int(len(r)), "span_years": float(span_years), "target_sharpe": target_sharpe, "years": years}
    if not len(r) or not span_years > 0:
        return out | {"prob_failure": float("nan")}
    n = len(r) / span_years
    out["bets_per_year"] = float(n)
    if not len(pos):
        return out | {"precision": 0.0, "prob_failure": 1.0}
    if not len(neg):
        return out | {"precision": 1.0, "prob_failure": float("nan")}        # pi_minus undefined: no verdict
    p = len(pos) / len(r)
    pi_plus, pi_minus = float(pos.mean()), float(neg.mean())
    binary_sd = (pi_plus - pi_minus) * np.sqrt(p * (1 - p))
    ratio = max(1.0, float(r.std() / binary_sd))
    theta = target_sharpe * ratio
    boot = strategy_risk.prob_failure(r, n, theta, "bootstrap", years, n_boot, seed)
    se = strategy_risk.prob_failure(r, n, theta, "estimation_se")
    return out | {"precision": p, "pi_plus": pi_plus, "pi_minus": pi_minus, "variance_ratio": ratio,
                  "implied_precision": strategy_risk.implied_precision(pi_minus, pi_plus, n, theta),
                  "prob_failure_bootstrap": boot, "prob_failure_estimation_se": se, "prob_failure": max(boot, se)}


def span_years(rows: pd.DataFrame) -> float:
    """Years between the first and the last bet (15.4.1), windows counted whole."""
    if rows.empty:
        return 0.0
    return float((rows["start"].max() + 300 - rows["start"].min()) / (365.25 * DAY))


def account_drawdown(rows: pd.DataFrame, since: float) -> float:
    """Largest fall from a running peak of the paper account (BANKROLL0 + P&L of every earlier traded window), counted
    from its equity at `since` on."""
    rows = rows.sort_values("start")
    before = float(rows.loc[rows["start"] < since, "pnl"].sum())
    after = rows.loc[rows["start"] >= since, "pnl"].to_numpy(float)
    equity = np.concatenate([[BANKROLL0 + before], BANKROLL0 + before + np.cumsum(after)])
    return metrics.max_drawdown(equity) if equity.max() > 0 else float("inf")


# ------------------------------------------------------------------------------------------------ graduation rules

def graduation_checks(record: pd.DataFrame, confirmed: bool, analysed_at: float, n_trials: int | None, now: float,
                      seed: int = 0) -> list[dict]:
    """Every graduation rule on the paper record since CONFIRM_START; graduation needs all of them."""
    rows = record[record["cost"] > 0].sort_values("start")
    r = _returns(rows)
    out = [check("confirmed registered leg (scripts/confirm.py, analysis and --check --final)", confirmed, confirmed,
                 True, "book: embargoed performance consistent with the backtest")]
    needed = evaluation.windows_needed_sim(_leg(rows), power=PAPER_POWER, seed=seed) if len(r) > 2 else float("inf")
    bound = max(MIN_PAPER_WINDOWS, needed)
    out.append(check("settled traded paper windows since the confirmation start", len(r) >= bound, int(len(r)),
                     _num(bound), f"ours: max({MIN_PAPER_WINDOWS}, windows the leg's own test needs for "
                                  f"{PAPER_POWER:.0%} power at its observed returns)"))
    days = (now - analysed_at) / DAY
    out.append(check("days in the paper stage after the analysis", days >= MIN_PAPER_DAYS, round(days, 3),
                     MIN_PAPER_DAYS, "ours"))
    fp = failure_probability(r, span_years(rows), seed=seed)
    ok = fp["prob_failure"] < MAX_FAILURE_PROBABILITY                     # NaN (no verdict) fails
    out.append(check(f"strategy-risk failure probability P[p < p*] at annual Sharpe {TARGET_SHARPE:g}",
                     ok, _num(fp["prob_failure"]), MAX_FAILURE_PROBABILITY,
                     "book: 15.4 (bound 0.05, 15.4.2); ours: max of bootstrap and estimation error, variance ratio")
               | {"detail": {k: _num(v) if isinstance(v, float) else v for k, v in fp.items()}})
    if n_trials and len(r) >= 3:
        h = evaluation.luck_hurdle(_leg(rows), int(n_trials))
        dsr = h["dsr"]
    else:
        dsr = float("nan")
    out.append(check("deflated Sharpe ratio against the trial registry's selection family",
                     np.isfinite(dsr) and dsr >= MIN_DSR, _num(dsr), MIN_DSR,
                     f"book: ch. 14 at 95%; trials = research/trial_registry.json total ({n_trials})"))
    dd = account_drawdown(rows, -np.inf) if len(rows) else float("nan")
    out.append(check("max drawdown of the paper account since the confirmation start", np.isfinite(dd) and
                     dd <= MAX_DRAWDOWN, _num(dd), MAX_DRAWDOWN, "ours"))
    return out


# ----------------------------------------------------------------------------------------------- decommission rules

def day_ends(since: float, now: float) -> list[int]:
    """Taiwan midnights in (since, now]."""
    first = int((since + 8 * 3600) // DAY + 1) * DAY - 8 * 3600
    return list(range(first, int(now) + 1, DAY))


def decommission_triggers(record: pd.DataFrame, since: float, now: float, seed: int = 0) -> list[dict]:
    """Every decommission rule on the paper record from `since` (the stage's start) to `now`; any fired one
    decommissions. D2's trailing lookback is cut at `since` too, so it judges only the stage's own record."""
    rows = record[record["cost"] > 0].sort_values("start")
    after = rows[rows["start"] >= since]
    out = []
    ci = (float("nan"), float("nan"))
    if len(after) >= LOSS_MIN_WINDOWS:
        ci = evaluation.outcome_ci(_leg(after), level=LOSS_LEVEL, seed=seed)
    out.append(check("loss: upper end of the 90% interval of the mean return per $ below 0", np.isfinite(ci[1]) and
                     ci[1] < 0, _num(ci[1]), 0.0, f"ours (book: performance below expectations); needs "
                                                  f">= {LOSS_MIN_WINDOWS} traded windows, has {len(after)}"))
    failing = []
    for end in day_ends(since, now)[-DECAY_DAYS:]:
        span = after[(after["start"] >= end - DECAY_LOOKBACK_DAYS * DAY) & (after["start"] + 300 <= end)]
        if len(span) < DECAY_MIN_WINDOWS:
            continue
        fp = failure_probability(_returns(span), span_years(span), seed=seed)["prob_failure"]
        if not (np.isfinite(fp) and fp < MAX_FAILURE_PROBABILITY):
            failing.append(end)
    fired = len(failing) == DECAY_DAYS
    out.append(check(f"strategy risk: failure probability >= {MAX_FAILURE_PROBABILITY} on the trailing "
                     f"{DECAY_LOOKBACK_DAYS} days at each of the last {DECAY_DAYS} daily assessments", fired,
                     len(failing), DECAY_DAYS, "book: 15.4.2 bound, sustained (ch. 1: for a sufficiently extended "
                                               "period); ours: 7 days, 30-day lookback"))
    dd = account_drawdown(rows, since) if len(after) else 0.0
    out.append(check("drawdown of the paper account since the stage started above the bound", dd > MAX_DRAWDOWN,
                     _num(dd), MAX_DRAWDOWN, "ours"))
    return out


# ------------------------------------------------------------------------------------------------------ the stages

def _carry(previous: dict | None, stage: str, now: float) -> float:
    return previous["since"] if previous and previous.get("stage") == stage and previous.get("since") else now


def assess(name: str, conf: Confirmation, record: pd.DataFrame | None = None, n_trials: int | None = None,
           previous: dict | None = None, now: float | None = None, seed: int = 0) -> dict:
    """{"stage", "since", "reason", "registered_leg", "confirmed", "checks", "triggers"[, "graduation"]} for one
    strategy. `record` (paper evidence) is refused before the confirmation is analysed."""
    now = time.time() if now is None else now
    if record is not None and not conf.analysed:
        raise ValueError("paper results of confirmation windows are sealed until the registered analysis")
    passes = lambda result: "not run" if result is None else ("passes" if result[0] == 0 else f"fails: {result[1]}")
    evidence = {"design_leg": name in conf.design_legs,
                "registration": "not written" if conf.registration is None else
                f"written, scripts/confirm.py --check {passes(conf.registration_check)}",
                "analysis": f"not analysed (at {when(conf.analysis_at)})" if not conf.analysed else
                f"run at {when(conf.analysis['run_at'])}, scripts/confirm.py --check --final {passes(conf.analysis_check)}",
                "paper_record": f"sealed until the analysis at {when(conf.analysis_at)}: not read" if not conf.analysed else
                "not read (no confirmed leg)" if record is None else
                f"{int((record['cost'] > 0).sum())} traded windows since {when(conf.confirm_start)}"}
    base = {"registered_leg": False, "confirmed": None, "evidence": evidence, "checks": [], "triggers": []}

    def out(stage: str, reason: str, **extra) -> dict:
        return base | {"stage": stage, "since": _carry(previous, stage, now), "reason": reason} | extra

    if previous and previous.get("stage") == "decommissioned":
        return base | {k: previous[k] for k in ("stage", "since", "reason") if k in previous} | \
            {"triggers": previous.get("triggers", [])}
    analysis_at = when(conf.analysis_at)
    unanalysed = (f"confirmation v{conf.version} is not analysed yet (its one analysis is at {analysis_at}), and "
                  f"graduation needs a registered leg that analysis confirmed")
    reg = conf.registration
    legs = tuple(reg["legs"]) if reg else tuple(conf.design_legs)
    if name not in legs:
        return out("research", f"not a leg of confirmation v{conf.version} (legs: {', '.join(legs)}): exploratory. "
                               f"Cannot graduate: {unanalysed}; a strategy enters the embargo only as a leg of a "
                               f"registration written before its confirmation starts.")
    base["registered_leg"] = True
    if reg is None:
        return out("research", f"leg of the approved confirmation v{conf.version} design, but "
                               f"research/preregistration.json is not written yet: the embargo starts once it is "
                               f"written before {when(conf.confirm_start)} and scripts/confirm.py --check passes. "
                               f"Cannot graduate: {unanalysed}.")
    if not (reg.get("version") == conf.version and reg.get("confirm_start") == conf.confirm_start
            and reg.get("registered_at", np.inf) < conf.confirm_start):
        return out("research", f"research/preregistration.json does not register confirmation v{conf.version} before "
                               f"{when(conf.confirm_start)}: no embargo. Cannot graduate: {unanalysed}.")
    code, line = conf.registration_check or (1, "not run")
    if code != 0:
        return out("research", f"scripts/confirm.py --check fails ({line}): the registration no longer describes the "
                               f"running code, so there is no embargo. Cannot graduate: {unanalysed}.")
    if not conf.analysed:
        return out("embargo", f"leg of confirmation v{conf.version}, embargoed on windows from "
                              f"{when(conf.confirm_start)} to its one analysis at {analysis_at}; its paper results on "
                              f"those windows are sealed and not read. Cannot graduate: {unanalysed}.")
    code, line = conf.analysis_check or (1, "not run")
    if code != 0:
        return out("research", f"scripts/confirm.py --check --final fails ({line}): the analysis does not stand for "
                               f"the running code. Cannot graduate until it passes.")
    leg = conf.analysis["legs"].get(name, {})
    base["confirmed"] = bool(leg.get("confirmed"))
    if not base["confirmed"]:
        return out("research", f"not confirmed by confirmation v{conf.version} (Holm p {leg.get('p_holm')}): back to "
                               f"research; it can return only as a leg of a new registration.")
    if record is None:
        raise ValueError(f"{name}: a confirmed leg needs its paper record")
    run_at = float(conf.analysis["run_at"])
    marked = bool(previous and previous.get("stage") == "graduated")
    since = previous.get("since") if marked else None
    # the status file is writable by the user's uid: its "graduated" is re-judged below, and its time is carried only
    # when it lies between the analysis and now (for real trading it must also match the root-owned config, pmexec (a))
    carried = isinstance(since, (int, float)) and not isinstance(since, bool) and run_at <= since <= now
    entry = since if carried else run_at
    triggers = decommission_triggers(record, entry, now, seed)
    fired = [t["rule"] for t in triggers if t["ok"]]
    if fired:
        return base | {"stage": "decommissioned", "since": now, "triggers": triggers,
                       "reason": f"decommissioned at {when(now)}: {'; '.join(fired)}."}
    checks = graduation_checks(record, True, run_at, n_trials, now, seed)
    failed = [f"{c['rule']} ({c['value']} vs {c['bound']})" for c in checks if not c["ok"]]
    if failed:
        was = " (it was marked graduated; graduation rules are re-checked on every run)" if marked else ""
        return out("paper", f"confirmed leg paper-trading since the analysis at {when(run_at)}; not graduated{was}: "
                            f"{'; '.join(failed)}.", checks=checks, triggers=triggers)
    rows = record[record["cost"] > 0]
    if carried:
        # the graduation record is rebuilt from the paper record as it stood then, never read from the status file
        # (its Sharpe sets health(); code review afml.md, lifecycle finding 2)
        then = rows[rows["start"] + 300 <= since]
        graduation = {"at": since, "sharpe": _num(metrics.sharpe(_returns(then))), "windows": int(len(then))}
        return base | {"stage": "graduated", "since": since, "checks": checks, "triggers": triggers,
                       "graduation": graduation,
                       "reason": f"graduated since {when(since)}; every graduation rule still holds and no decommission "
                                 f"trigger has fired."}
    return base | {"stage": "graduated", "since": now, "checks": checks, "triggers": triggers,
                   "graduation": {"at": now, "sharpe": _num(metrics.sharpe(_returns(rows))), "windows": int(len(rows))},
                   "reason": f"graduated at {when(now)}: every graduation rule holds."}


# --------------------------------------------------------------------------------------------------- re-allocation

def ramp(days: float) -> float:
    """Share of the HRP weight a strategy gets `days` after graduation: INITIAL_ALLOCATION at 0, rising along a
    square root (concave) to 1 at RAMP_DAYS."""
    return float(min(1.0, INITIAL_ALLOCATION + (1 - INITIAL_ALLOCATION) * np.sqrt(max(days, 0.0) / RAMP_DAYS)))


def health(record: pd.DataFrame, graduation_sharpe: float | None, since: float, now: float) -> float:
    """Recent per-window Sharpe (traded windows since graduation, of the last HEALTH_LOOKBACK_DAYS) over the Sharpe at
    graduation, in [0, 1]; 1 with fewer than HEALTH_MIN_WINDOWS such windows (no evidence of decay yet)."""
    rows = record[(record["cost"] > 0) & (record["start"] >= max(since, now - HEALTH_LOOKBACK_DAYS * DAY))]
    if len(rows) < HEALTH_MIN_WINDOWS or not graduation_sharpe or graduation_sharpe <= 0:
        return 1.0
    sr = metrics.sharpe(_returns(rows))
    return float(np.clip(sr / graduation_sharpe, 0.0, 1.0)) if np.isfinite(sr) else 0.0


def allocate(graduated: dict[str, dict], records: dict[str, pd.DataFrame], now: float) -> dict:
    """Capital shares among graduated strategies (name -> {"since", "graduation": {"sharpe"}}): HRP weights from the
    covariance of each paper account's P&L per window ÷ BANKROLL0 (0 where it did not trade) on every window from the
    earliest record, times ramp(days since graduation) times health(); what is left stays unallocated."""
    names = sorted(graduated)
    if not names:
        return {"strategies": {}, "unallocated": 1.0}
    starts = [records[n]["start"] for n in names if not records[n].empty]
    lo, hi = (int(min(s.min() for s in starts)), int(max(s.max() for s in starts))) if starts else (0, 0)
    grid = np.arange(lo, hi + 300, 300)
    R = np.column_stack([records[n].groupby("start")["pnl"].sum().reindex(grid, fill_value=0.0).to_numpy() / BANKROLL0
                         for n in names])
    var = R.var(axis=0, ddof=1) if len(grid) > 1 else np.zeros(len(names))
    live = [i for i in range(len(names)) if var[i] > 0]
    w = np.zeros(len(names))
    if live:
        w[live] = hrp.hrp(np.cov(R[:, live], rowvar=False).reshape(len(live), len(live)))["weights"]
    out = {}
    for i, n in enumerate(names):
        g = graduated[n]
        days = (now - g["since"]) / DAY
        h = health(records[n], (g.get("graduation") or {}).get("sharpe"), g["since"], now)
        out[n] = {"hrp_weight": float(w[i]), "ramp": ramp(days), "health": h, "fraction": float(w[i] * ramp(days) * h)}
    return {"strategies": out, "unallocated": float(1.0 - sum(v["fraction"] for v in out.values()))}


# ------------------------------------------------------------------------------------------------------ invariants

def validate(status: dict) -> list[str]:
    """What must hold in a status file; each broken rule is one line."""
    bad = []
    analysed = bool(status.get("confirmation", {}).get("analysed"))
    for name, s in status.get("strategies", {}).items():
        if s.get("stage") not in STAGES:
            bad.append(f"{name}: unknown stage {s.get('stage')!r}")
        if not str(s.get("reason", "")).strip():
            bad.append(f"{name}: no reason")
        if s.get("stage") in ("paper", "graduated") and not (analysed and s.get("confirmed") is True):
            bad.append(f"{name}: {s.get('stage')} without a leg confirmed by the analysis")
        if s.get("stage") == "embargo" and analysed:
            bad.append(f"{name}: embargo after the analysis")
        if not analysed and (s.get("checks") or s.get("graduation")
                             or not str(s.get("evidence", {}).get("paper_record", "")).startswith("sealed")):
            bad.append(f"{name}: paper evidence before the analysis")
        if s.get("stage") == "graduated" and not s.get("graduation"):
            bad.append(f"{name}: graduated without its graduation record")
    alloc = status.get("allocation", {}).get("strategies", {})
    for name in alloc:
        if status.get("strategies", {}).get(name, {}).get("stage") != "graduated":
            bad.append(f"{name}: allocation without graduation")
    if sum(v.get("fraction", 0.0) for v in alloc.values()) > 1 + 1e-9:
        bad.append("allocations sum above 1")
    return bad
