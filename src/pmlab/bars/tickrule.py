"""The tick rule (AFML 2.3.2(1)): infer the aggressor from price changes.

b_t = sign(Δp_t) when the price moved, else b_{t-1}. The Polymarket tape tells us the true
aggressor, so here the rule is something to measure, not something to depend on.
"""


class TickRule:
    def __init__(self, initial: int = 1):
        self.initial = initial
        self.reset()

    def reset(self):
        self.last_p: float | None = None
        self.last_b = self.initial

    def __call__(self, p: float) -> int:
        if self.last_p is not None and p != self.last_p:
            self.last_b = 1 if p > self.last_p else -1
        self.last_p = p
        return self.last_b


def accuracy(prices, true_signs, initial: int = 1) -> float:
    rule = TickRule(initial)
    inferred = [rule(p) for p in prices]
    return sum(a == b for a, b in zip(inferred, true_signs)) / len(inferred) if inferred else float("nan")
