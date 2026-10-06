"""Hold-to-expiry strategies. Each is a pure function of a causal View plus its own position.

- ProbTaker: buy the side a probability model says is cheap after fees, Kelly-sized. With
  prob="fair" this is the Q-measure (risk-neutral) price; with a P-model column it is the
  change-of-measure version. Kelly = maximising E[ln wealth], the log utility of slide 3.
- FlowTaker: follow the order flow of the last completed bar of one of the nine bar kinds,
  as long as the price is not far above the Q fair value.
- FairMaker: rest bids for both tokens `half_spread` below the Q fair value; pays no fees.

candidates() is a vectorised prefilter: a superset of the seconds where decide() can act,
so the backtest skips the rest. It may use each second's own column values only (the backtest
passes the columns without the outcome or the tape and asks the seconds in time order); decide()
alone sees the causal View and makes the decision. The live engine calls decide() every second,
so decide() enforces its own time rules too.
"""
from dataclasses import dataclass

import numpy as np

from pmlab.backtest import Order, Position, View
from pmlab.fees import fee_per_share

PRICE_MIN, PRICE_MAX = 0.02, 0.98


def kelly_fraction(p: float, price: float) -> float:
    """Kelly stake / bankroll for a $1 binary bought at `price` (fees included) with belief p."""
    return max((p - price) / (1.0 - price), 0.0) if price < 1 else 0.0


def _cooldown(t_rng, cooldown):
    return np.arange(t_rng[0], t_rng[1] + 1, max(int(cooldown), 1))


@dataclass
class ProbTaker:
    prob: str = "fair"
    edge: float = 0.03           # minimum expected profit per share after fees
    kelly: float = 0.25          # fraction of full Kelly
    max_stake: float = 50.0
    max_window_cost: float = 150.0
    max_window_frac: float = 0.15   # never commit more than this share of the bankroll in one window
    t_min: int = 5
    t_max: int = 285
    max_age: float = 5.0         # ignore quotes older than this (seconds)
    slip: float = 0.01           # limit above the quote
    step: int = 1
    name: str = "prob_taker"

    def _edges(self, p, ask, bid):
        e_up = p - (ask + self.slip) - fee_per_share(np.clip(ask + self.slip, 0, 1))
        e_dn = (1 - p) - (1 - bid + self.slip) - fee_per_share(np.clip(1 - bid + self.slip, 0, 1))
        return e_up, e_dn

    def candidates(self, f):
        t = _cooldown((self.t_min, self.t_max), self.step)
        p, ask, bid = f[self.prob][t], f["ask"][t], f["bid"][t]
        e_up, e_dn = self._edges(p, ask, bid)
        return t[(np.nan_to_num(e_up, nan=-1) > self.edge) | (np.nan_to_num(e_dn, nan=-1) > self.edge)]

    def decide(self, v: View, pos: Position, bankroll: float) -> list[Order]:
        if not self.t_min <= v.t <= self.t_max or (v.t - self.t_min) % max(int(self.step), 1):
            return []
        p = v.now(self.prob)
        if not np.isfinite(p):
            return []
        exp = pos.exposure(v.t)
        room = min(self.max_window_cost, self.max_window_frac * bankroll) - exp["cost"]
        orders = []
        for side, quote, age in (("up", v.now("ask"), v.now("ask_age")), ("down", v.now("bid"), v.now("bid_age"))):
            if not np.isfinite(quote) or age > self.max_age:
                continue
            price = quote + self.slip if side == "up" else 1 - quote + self.slip
            belief = p if side == "up" else 1 - p
            if not PRICE_MIN <= price <= PRICE_MAX:
                continue
            all_in = price + fee_per_share(price)
            if belief - all_in <= self.edge or exp["down" if side == "up" else "up"] > 0:
                continue
            # Kelly sizes the whole window position, not each order (R2.3)
            target = self.kelly * kelly_fraction(belief, all_in) * bankroll
            stake = min(target - exp["cost"], self.max_stake, room)
            if stake > 1:
                orders.append(Order(side, stake / all_in, price, tag=f"{self.name}:{side}"))
                room -= stake
                exp[side] += stake / all_in                    # the other side is now off limits (R2.7)
                exp["cost"] += stake
        return orders


@dataclass
class FlowTaker:
    kind: str = "tick_imbalance"
    imb: float = 0.5             # |risk imbalance / risk| of the last bar needed to act
    max_age: float = 3.0         # the bar must have closed within this many seconds
    slack: float = 0.02          # pay at most fair + slack for the flow side
    stake: float = 20.0
    max_window_cost: float = 100.0
    t_min: int = 5
    t_max: int = 280
    slip: float = 0.01
    name: str = "flow_taker"

    def candidates(self, f):
        t = np.arange(self.t_min, self.t_max + 1)
        imb, age = f[f"{self.kind}_imb"][t], f[f"{self.kind}_age"][t]
        return t[(np.abs(np.nan_to_num(imb)) >= self.imb) & (np.nan_to_num(age, nan=1e9) <= self.max_age)]

    def decide(self, v: View, pos: Position, bankroll: float) -> list[Order]:
        if not self.t_min <= v.t <= self.t_max:
            return []
        imb, age = v.now(f"{self.kind}_imb"), v.now(f"{self.kind}_age")
        if not (np.isfinite(imb) and abs(imb) >= self.imb and age <= self.max_age):
            return []
        side = "up" if imb > 0 else "down"
        quote, qage = (v.now("ask"), v.now("ask_age")) if side == "up" else (v.now("bid"), v.now("bid_age"))
        if not np.isfinite(quote) or qage > 5:
            return []
        price = quote + self.slip if side == "up" else 1 - quote + self.slip
        fair = v.now("fair") if side == "up" else 1 - v.now("fair")
        if not np.isfinite(fair):                  # no fair value yet (e.g. start reference unknown)
            return []
        exp = pos.exposure(v.t)
        room = self.max_window_cost - exp["cost"]
        if not PRICE_MIN <= price <= PRICE_MAX or price > fair + self.slack or room < self.stake:
            return []
        if exp["down" if side == "up" else "up"] > 0:
            return []
        return [Order(side, self.stake / (price + fee_per_share(price)), price, tag=f"{self.name}:{self.kind}")]


@dataclass
class FairMaker:
    prob: str = "fair"
    half_spread: float = 0.03
    stake: float = 20.0
    requote: int = 5             # seconds between quotes; each quote rests this long
    max_window_cost: float = 150.0
    t_min: int = 5
    t_max: int = 240
    name: str = "fair_maker"

    def candidates(self, f):
        return np.arange(self.t_min, self.t_max + 1, max(int(self.requote), 1))

    def decide(self, v: View, pos: Position, bankroll: float) -> list[Order]:
        if not self.t_min <= v.t <= self.t_max or (v.t - self.t_min) % max(int(self.requote), 1):
            return []
        p = v.now(self.prob)
        if not np.isfinite(p):
            return []
        exp = pos.exposure(v.t)
        orders = []
        for side, fair_side, quote in (("up", p, v.now("ask")), ("down", 1 - p, 1 - v.now("bid"))):
            bid = round(fair_side - self.half_spread, 2)
            # never post a marketable bid: that would be a taker trade in disguise
            if not PRICE_MIN <= bid <= PRICE_MAX or (np.isfinite(quote) and bid >= quote):
                continue
            if exp["cost"] + self.stake > self.max_window_cost:
                break
            orders.append(Order(side, self.stake / bid, bid, maker=True, ttl=int(self.requote), tag=f"{self.name}:{side}"))
            exp["cost"] += self.stake
        return orders


@dataclass
class OpenFairTaker:
    """research/alpha.md H5a: at the open the fair price already leans (spot vs the prior-minute
    average), while the market often still sits near 0.5. Buy the leaning side once, early."""
    lean: float = 0.15           # |fair − 0.5| needed
    gap: float = 0.10            # fair − mid, in the leaning direction
    t_min: int = 2
    t_max: int = 30
    stake: float = 20.0
    slip: float = 0.01
    max_age: float = 5.0
    prob: str = "fair"
    name: str = "open_fair"

    def candidates(self, f):
        t = np.arange(self.t_min, self.t_max + 1)
        p, mid = f[self.prob][t], (f["ask"][t] + f["bid"][t]) / 2
        s = np.sign(p - 0.5)
        return t[(np.abs(np.nan_to_num(p, nan=0.5) - 0.5) >= self.lean) & (np.nan_to_num((p - mid) * s) >= self.gap)]

    def decide(self, v: View, pos: Position, bankroll: float) -> list[Order]:
        if not self.t_min <= v.t <= self.t_max or pos.fills or pos.pending:
            return []                                  # one entry per window
        p, ask, bid = v.now(self.prob), v.now("ask"), v.now("bid")
        if not all(np.isfinite([p, ask, bid])) or max(v.now("ask_age"), v.now("bid_age")) > self.max_age:
            return []
        s = 1 if p > 0.5 else -1
        if abs(p - 0.5) < self.lean or (p - (ask + bid) / 2) * s < self.gap:
            return []
        side, price = ("up", ask + self.slip) if s > 0 else ("down", 1 - bid + self.slip)
        if not PRICE_MIN <= price <= PRICE_MAX:
            return []
        return [Order(side, self.stake / (price + fee_per_share(price)), round(price, 3), tag=self.name)]


@dataclass
class EndgameMaker:
    """research/alpha.md H3 (maker): in the last seconds the outcome is usually decided while sellers
    still hit bids just under 1. Rest a bid on the decided favourite; no fee as a maker."""
    conf: float = 0.99           # max(p, 1 − p) needed
    bid: float = 0.99
    t_min: int = 270
    t_max: int = 299
    stake: float = 50.0
    requote: int = 5
    prob: str = "fair"
    name: str = "endgame_maker"

    def candidates(self, f):
        return np.arange(self.t_min, self.t_max + 1, max(int(self.requote), 1))

    def decide(self, v: View, pos: Position, bankroll: float) -> list[Order]:
        if not self.t_min <= v.t <= self.t_max or (v.t - self.t_min) % max(int(self.requote), 1):
            return []
        p = v.now(self.prob)
        if not np.isfinite(p) or max(p, 1 - p) < self.conf:
            return []
        side = "up" if p > 0.5 else "down"
        if pos.exposure(v.t)["down" if side == "up" else "up"] > 0:
            return []
        return [Order(side, self.stake / self.bid, self.bid, maker=True, ttl=int(self.requote), tag=self.name)]


PROB_COLUMNS = {"q_taker": "fair", "q_taker_cl": "fair_cl", "tilt_taker": "p_tilt", "drift_taker": "p_drift", "gp_taker": "p_gp"}


def build(name: str, params: dict):
    """A strategy from its family name and tuned parameters (research/strategy_params.json)."""
    if name in PROB_COLUMNS:
        return ProbTaker(prob=PROB_COLUMNS[name], name=name, **params)
    if name == "fair_maker":
        return FairMaker(name=name, **params)
    if name.startswith("flow_"):
        return FlowTaker(kind=name[len("flow_"):], name=name, **params)
    if name.startswith("open_fair"):
        return OpenFairTaker(name=name, **params)
    if name.startswith("endgame_maker"):
        return EndgameMaker(name=name, **params)
    raise ValueError(f"unknown strategy family {name!r}")
