"""The native fill balance update, shared by reporting and state projection."""
from decimal import localcontext


def apply_fill(cash, position, total_fees, *, side, units, price, fee):
    """Update validated Decimal/integer balances under accounting precision 50.

    This primitive adds no admission, financing, liquidation or fee policy.
    Derived cash is not constrained by the normalized input's monetary cap.
    """
    with localcontext() as context:
        context.prec = 50
        signed_units = units * (1 if side == "buy" else -1)
        delta_cash = -signed_units * price - fee
        cash += delta_cash
        position += signed_units
        total_fees += fee
        return cash, position, total_fees, delta_cash
