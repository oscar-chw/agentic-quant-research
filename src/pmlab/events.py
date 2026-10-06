"""The one event schema every consumer (bars, fair price, strategies, live engine) reads.

A Trade is a taker trade already in the Up frame (see pmlab.polymarket): historical
tapes and the live websocket both convert into this before anything else sees them.
"""
import math
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True, frozen=True)
class Trade:
    ts: float        # unix seconds (whole seconds from the REST tape, ms precision live)
    window: int      # window start, unix seconds
    p_up: float      # price in the Up frame
    sign: int        # +1 aggressor buys Up, -1 aggressor sells Up
    shares: float
    usdc: float      # cash the taker actually paid or received, in the token traded
    risk_usd: float  # shares * sqrt(p_up * (1 - p_up)): $ std-dev of the payoff transferred

    @property
    def t(self) -> float:
        return self.ts - self.window


def risk_usd(shares: float, p_up: float) -> float:
    return shares * math.sqrt(max(p_up * (1.0 - p_up), 0.0))


def make_trade(ts: float, window: int, p_up: float, sign: int, shares: float, usdc: float) -> Trade:
    return Trade(ts, window, p_up, sign, shares, usdc, risk_usd(shares, p_up))


def trades_from_tape(tape: pd.DataFrame, window: int) -> list[Trade]:
    """Chronological Trade list from a normalised tape (pmlab.polymarket.normalise_trades)."""
    return [make_trade(float(ts), window, float(p), int(s), float(q), float(u))
            for ts, p, s, q, u in zip(tape["ts"], tape["p_up"], tape["sign"], tape["shares"], tape["usdc"])]
