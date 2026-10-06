"""Polymarket taker fees (docs.polymarket.com, verified 2026-09-17).

fee = C × feeRate × p × (1 − p) USDC for C shares at price p, with feeRate 0.07 on these
markets, rounded to 5 decimals and never below 0.00001 once non-zero. Makers pay nothing.
p(1 − p) is the same for Up at p and Down at 1 − p, so the Up frame does not change it.

Makers may also earn a rebate: the markets' feeSchedule lists rebateRate 0.2 (a maker rebates
programme paying a share of the taker fees their fills generate). Its eligibility terms are not
verified here, so backtests and paper trading leave it off by default (Config.maker_rebate = 0)
and report it as a sensitivity (research/microstructure.md).
"""
FEE_RATE = 0.07
MIN_FEE = 1e-5


def taker_fee(shares: float, price: float, rate: float = FEE_RATE) -> float:
    raw = shares * rate * price * (1.0 - price)
    return 0.0 if raw <= 0 else max(round(raw, 5), MIN_FEE)


def fee_per_share(price: float, rate: float = FEE_RATE) -> float:
    """Unrounded fee per share: the edge a taker must beat at price p."""
    return rate * price * (1.0 - price)


REBATE_RATE = 0.2


def maker_rebate(shares: float, price: float, rebate_rate: float = REBATE_RATE, rate: float = FEE_RATE) -> float:
    """Rebate a maker would receive on a fill: rebate_rate × the taker fee that fill generated."""
    return rebate_rate * shares * rate * price * (1.0 - price)
