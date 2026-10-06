"""ch.20, the one process that is not research: the trading loop's own per-second budget, and the alert it raises.

Chapters 20–22 size *research* — molecules across processes, threads where the work waits, native pools capped so
they do not take the machine. The live loop is the opposite problem. It is single-threaded and event-driven on
purpose, it has exactly one deadline, and that deadline is physical rather than chosen:

    the work attributable to engine second c must finish inside one second

because the feeds push into an asyncio queue whatever the loop is doing. A second that costs 1.2 s does not merely
run late: the queue grows, the next second starts later still, and the loop walks away from the exchange clock. Every
number below is therefore expressed against that one second, in milliseconds.

**The three stages, end to end.** What the live programme (`pmlab.live.app.main`) does with a second is:

    feed      the events stamped in that second, decoded and applied     engine.handle(item)
    decision  the second itself: prices, rows, strategies, fills, tap    engine._second(c)
    record    the same events to parquet, and the engine's own rows      recorder.add(item) / TableWriter.flush()

`scripts/perf.py::live_second_end_to_end` measures all three on **recorded** data through a fresh engine — never
against the running app, which is a process this project may watch and must not touch. Measured over 900 consecutive
recorded seconds (965,986 events, 2026-09-16 20:15 UTC onwards), with the app running and threads pinned:

    mean 18.6 ms   median 15.3 ms   p90 34.7 ms   p99 68.4 ms   worst 95.5 ms
    feed mean 6.3 (worst 36.4) · decision mean 8.4 (worst 70.4) · record mean 4.0 (worst 54.6)

So the loop spends about 2% of each second on its work and at worst a tenth of it. `MEASURED` keeps those numbers
next to the budget they set, because a budget whose measurement is not written down is a wish. Repeats of the same
stretch move: the worst second has come back anywhere from 90 to 112 ms depending on what else the machine was
doing, which is the spread the ceiling is chosen against and the reason it is not set at the worst observed.

**The budget, and why it is not the worst measurement.** `CEILING_MS` is a quarter of `DEADLINE_MS`: 250 ms, which is
2.6x the worst second observed. Setting it at the observed worst would alert on the next parquet flush that lands
next to a garbage collection; setting it at the deadline would alert only once the loop had already fallen behind,
which is too late to be an alert at all. A quarter leaves the loop four times its worst second before anyone is told,
and still tells them while three quarters of every second is unused.

**An alert is about a window, not a second.** One 250 ms second is a flush, a page fault or a collection, and a
system that pages a human for one of those gets ignored. A window is 300 seconds of the same work, so the rule is:

    OVER_IN_WINDOW = 3 seconds over CEILING_MS inside one window  ->  alert
    any single second over DEADLINE_MS                            ->  alert at once, whatever the count

The first is 1% of a window, and the measurement above put *zero* of 900 seconds over the ceiling, so on recorded
data this rule is silent — which `scripts/perf.py --check --live` re-checks on every run, because a rule that fires
on a healthy loop is a rule that will be muted. The second needs no count: the loop is behind the clock already.

**Where this runs, and where it does not.** Nothing here is wired into the live engine. `src/pmlab/live/*` is hashed
by the confirmation registered 2026-09-18 and may not change until its analysis on **2026-10-24**, so this module is
the rule and the arithmetic, driven offline by the benchmark and by the tests, and
`tests/test_live_budget.py::test_the_engine_is_still_untimed_and_says_why` carries the wiring specification and fails
the day `pmlab/live/engine.py` leaves the registered set. It lives under `pmlab.afml` for the same reason
`threadcaps` and `threadio` do: scripts/confirm.py registers every module of pmlab outside afml and dashboard.

The alert dict has exactly the keys `pmlab.live.detect` gives its own, so the day the wiring lands the dashboard,
the `alerts` table and the websocket tick carry it with no change of shape.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from pmlab import WINDOW_SECONDS as T

DEADLINE_MS = 1000.0
"""The physical deadline: one engine second's work has one second of wall clock. Past it the queue grows."""

CEILING_MS = 250.0
"""The budget: a quarter of the deadline, 2.6x the worst second measured. See MEASURED."""

OVER_IN_WINDOW = 3
"""Seconds over CEILING_MS inside one window before it is the loop rather than one flush. 1% of a window."""

COOLDOWN_S = float(T)
"""One alert per window per kind: the condition is a property of the whole window, so repeating it says nothing new."""

MAX_ALERTS = 50
"""Alerts kept for the snapshot, as `pmlab.live.detect` keeps its own. A list that only grows is a leak in a
process meant to run for months."""

STALE_AFTER = 2 * T + 120
"""A window this far behind the clock is dropped whether or not anyone called `forget`. The engine does call it for
every window it settles; a caller that does not — a benchmark, a test, a script — must still not grow without
bound. The margin is the detector's, and for the same reason: a settled window keeps arriving for a while."""

MEASURED = {
    "stretch": "1789589700..1789590600 (900 s, 2026-09-16 20:15 UTC onwards, recorded)",
    "events": 965986,
    "mean_ms": 18.62, "median_ms": 15.26, "p90_ms": 34.69, "p99_ms": 68.40, "worst_ms": 95.53,
    "feed_mean_ms": 6.29, "decide_mean_ms": 8.35, "record_mean_ms": 3.97,
    "feed_worst_ms": 36.37, "decide_worst_ms": 70.43, "record_worst_ms": 54.58,
    "over_ceiling": 0,
    "machine": "Apple M2 Max, 12 cores, live app running, threads pinned to 2",
}
"""The measurement CEILING_MS was set from. docs/performance.md quotes it and a test ties the two together."""

STAGES = ("feed", "decide", "record")
"""The three stages of a second, in the order the loop does them."""


@dataclass(frozen=True)
class SecondCost:
    """What one engine second cost, split by stage, in milliseconds of wall clock.

    `events` is how many events that second fed: a second that fed none costs nothing and proves nothing, and the
    watcher below keeps it out of the alert's evidence so an idle stretch cannot dilute a slow one.
    """

    second: int
    feed_ms: float
    decide_ms: float
    record_ms: float
    events: int = 0

    @property
    def total_ms(self) -> float:
        return self.feed_ms + self.decide_ms + self.record_ms

    @property
    def stage(self) -> str:
        """The stage that dominated this second — the first thing anyone woken by the alert wants to know."""
        return max(STAGES, key=lambda s: getattr(self, f"{s}_ms"))

    def as_dict(self) -> dict:
        return {"second": int(self.second), "feed_ms": round(self.feed_ms, 3),
                "decide_ms": round(self.decide_ms, 3), "record_ms": round(self.record_ms, 3),
                "total_ms": round(self.total_ms, 3), "events": int(self.events), "stage": self.stage}


def window_of(second: float, window: int = T) -> int:
    """The window a second belongs to. Same arithmetic as `pmlab.polymarket.window_start`, without the import.

    `pmlab.polymarket` is registered code; this module deliberately imports nothing frozen but the window length, so
    a change there can never silently move a budget.
    """
    return int(math.floor(float(second) / window) * window)


class WindowBudget:
    """Per-second costs in, alerts out: the loop's budget applied window by window.

    Feed it one `SecondCost` per engine second, in order. It returns an alert dict the moment a window breaks the
    rule, and `None` every other second. `drain()` hands over the alerts raised since the last call, exactly as
    `pmlab.live.detect.Detector.drain` does, so a caller can poll it once a second and publish whatever it got.

    Memory: per-window state is dropped by `forget()` and, as a safety net, for any window more than `STALE_AFTER`
    behind the clock; at most `MAX_ALERTS` alerts are kept. `report()`'s counts are running totals and are not
    touched by either, because forgetting a window's books must not unmeasure its seconds.
    """

    def __init__(self, ceiling_ms: float = CEILING_MS, deadline_ms: float = DEADLINE_MS,
                 over: int = OVER_IN_WINDOW, window: int = T, cooldown_s: float = COOLDOWN_S):
        if not 0 < ceiling_ms <= deadline_ms:
            raise ValueError(f"ceiling_ms must be in (0, deadline_ms]; got {ceiling_ms} against {deadline_ms}")
        if over < 1:
            raise ValueError(f"over must be at least 1 second; got {over}")
        self.ceiling_ms, self.deadline_ms, self.over, self.window = float(ceiling_ms), float(deadline_ms), int(over), int(window)
        self.cooldown_s = float(cooldown_s)
        self.seconds: dict[int, int] = {}                 # window -> seconds seen
        self.breaches: dict[int, list[SecondCost]] = {}   # window -> the seconds over the ceiling, in order
        self.totals: dict[int, dict[str, float]] = {}     # window -> summed stage costs
        self.last_alert: dict[tuple, float] = {}
        self.serial = 0
        self.alerts: deque = deque(maxlen=MAX_ALERTS)
        self.new: list[dict] = []
        self.worst: SecondCost | None = None              # the costliest second seen, over the ceiling or not
        self.seen_total = 0                               # running counts, unaffected by forget(): they were measured
        self.over_total = 0
        self.alerts_total = 0
        self.windows_total = 0
        self.last_window: int | None = None               # seconds arrive in order, so a change of window is a new one

    # ------------------------------------------------------------------ input

    def on_second(self, cost: SecondCost) -> dict | None:
        s = window_of(cost.second, self.window)
        for old in [w for w in self.seconds if w + STALE_AFTER < cost.second]:
            self.forget(old)
        self.seconds[s] = self.seconds.get(s, 0) + 1
        self.seen_total += 1
        if s != self.last_window:
            self.windows_total += 1
            self.last_window = s
        totals = self.totals.setdefault(s, {f"{k}_ms": 0.0 for k in STAGES})
        for stage in STAGES:
            totals[f"{stage}_ms"] += getattr(cost, f"{stage}_ms")
        if self.worst is None or cost.total_ms > self.worst.total_ms:
            self.worst = cost
        if cost.total_ms <= self.ceiling_ms:
            return None
        self.breaches.setdefault(s, []).append(cost)
        self.over_total += 1
        if cost.total_ms > self.deadline_ms:
            return self._alert(cost, s, "deadline")
        if len(self.breaches[s]) >= self.over:
            return self._alert(cost, s, "ceiling")
        return None

    def forget(self, window: int) -> None:
        """Drop a settled window's state, the way the engine drops a settled window's books."""
        for d in (self.seconds, self.breaches, self.totals):
            d.pop(int(window), None)
        for key in [k for k in self.last_alert if k[1] == int(window)]:
            del self.last_alert[key]

    # ------------------------------------------------------------------ output

    def _alert(self, cost: SecondCost, window: int, rule: str) -> dict | None:
        key = ("slow_second", window, rule)
        if cost.second - self.last_alert.get(key, -math.inf) < self.cooldown_s:
            return None
        self.last_alert[key] = float(cost.second)
        self.serial += 1
        self.alerts_total += 1
        breached = self.breaches.get(window, [])
        worst = max(breached, key=lambda x: x.total_ms)
        n = max(self.seconds.get(window, 1), 1)
        totals = self.totals.get(window, {})
        if rule == "deadline":
            text = (f"the trading loop missed the clock: engine second {cost.second} (t={cost.second - window} s) "
                    f"took {cost.total_ms:.0f} ms, over the one-second deadline, mostly in {cost.stage}")
        else:
            text = (f"the trading loop is over its budget: {len(breached)} of the window's {n} seconds cost more "
                    f"than {self.ceiling_ms:.0f} ms, the worst {worst.total_ms:.0f} ms at t={worst.second - window} s, "
                    f"mostly in {worst.stage}")
        alert = {
            "id": self.serial, "kind": "slow_second", "time": float(cost.second), "window": window,
            "t": int(cost.second - window), "side": None, "price": None, "token": None, "token_price": None,
            "evidence": {"rule": rule, "seconds_over": len(breached), "seconds_seen": n,
                         "worst": worst.as_dict(), "this": cost.as_dict(),
                         **{f"mean_{k}": round(v / n, 3) for k, v in totals.items()}},
            "thresholds": {"ceiling_ms": self.ceiling_ms, "deadline_ms": self.deadline_ms,
                           "over_in_window": self.over, "cooldown_s": self.cooldown_s},
            "text": text,
        }
        self.alerts.append(alert)
        self.new.append(alert)
        if len(self.new) > MAX_ALERTS:                    # a caller that stops draining must not grow the buffer
            del self.new[0]
        return alert

    def drain(self) -> list[dict]:
        out, self.new = self.new, []
        return out

    def report(self) -> dict:
        """Everything seen so far, for a benchmark's parity line: the counts a budget is judged by.

        `worst` is the costliest second of all, not the costliest breach: on a healthy loop there are no breaches,
        and a report whose worst second is `None` whenever the budget holds would say nothing on exactly the runs
        that matter. Every count here is a running total, unaffected by `forget` and by the stale sweep and
        uncapped by MAX_ALERTS: dropping a settled window's books must never unmeasure its seconds.
        """
        return {"seconds": self.seen_total, "windows": self.windows_total, "over_ceiling": self.over_total,
                "alerts": self.alerts_total, "ceiling_ms": self.ceiling_ms, "deadline_ms": self.deadline_ms,
                "worst": self.worst.as_dict() if self.worst is not None else None}


def judge(report: dict) -> tuple[bool, str]:
    """A `report()` in, (pass, one line) out: the verdict `scripts/perf.py --check --live` fails on.

    Three ways this says no, and the third is the one that catches a benchmark rather than a loop:

    * an alert came out — the rule fired on the recorded data it was measured from, so it is a rule that will be
      muted rather than heeded;
    * the worst second is over the ceiling — the loop is slower than the budget says, alert or no alert;
    * nothing was measured — no seconds, or seconds with no cost, which is how a check reports success over
      something it never compared.

    It lives here rather than in the benchmark so a test can hand it a report and watch it refuse, which a
    judgement written inline in a parity function cannot be made to do.
    """
    worst = report.get("worst")
    seconds, ceiling = report.get("seconds") or 0, report.get("ceiling_ms")
    if not seconds or worst is None:
        return False, f"{seconds} second(s) measured, no cost recorded — the check compared nothing"
    detail = (f"{seconds} seconds over {report['windows']} window(s); worst {worst['total_ms']:.1f} ms "
              f"(feed {worst['feed_ms']:.1f} / decide {worst['decide_ms']:.1f} / record {worst['record_ms']:.1f}) "
              f"against a ceiling of {ceiling:.0f} ms; {report['over_ceiling']} second(s) over it, "
              f"{report['alerts']} alert(s)")
    if report.get("alerts"):
        return False, detail + " — the budget alerted on the data it was measured from"
    if worst["total_ms"] > ceiling:
        return False, detail + " — the worst second is over the ceiling"
    return True, detail
