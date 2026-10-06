"""Single-period diagonal-risk convex allocation with an auditable dual certificate.

All weights use fixed reference capital. Utility coefficients are scores, not
calibrated expected returns. Correlated covariance is intentionally unsupported.
"""
import math

FEAS_TOL = 1e-9
GAP_TOL = 1e-8
ITERATIONS = 80


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name+" must be finite numeric")
    return float(value)


def diagonal_covariance(matrix, n):
    if not isinstance(matrix, list) or len(matrix) != n or any(not isinstance(r, list) or len(r) != n for r in matrix):
        raise ValueError("covariance shape must match assets")
    diagonal = []
    for i, row in enumerate(matrix):
        for j, raw in enumerate(row):
            value = finite(raw, "covariance")
            if i != j and value != 0:
                raise ValueError("only diagonal PSD covariance is supported; correlations are not discarded")
            if i == j:
                if value < 0:
                    raise ValueError("covariance must be PSD (nonnegative diagonal)")
                diagonal.append(value)
    return diagonal


def terms(weights, previous, utility, variances, risk_aversion, ridge, costs, exit_costs):
    risk = math.fsum(v*w*w for v, w in zip(variances, weights))
    score = math.fsum(m*w for m, w in zip(utility, weights))
    risk_penalty = risk_aversion*risk/2
    regularization = ridge*math.fsum(w*w for w in weights)/2
    turnover = math.fsum(abs(w-p) for w, p in zip(weights, previous))
    entry = math.fsum(c*abs(w-p) for c, w, p in zip(costs, weights, previous))
    exit_proxy = math.fsum(c*abs(w) for c, w in zip(exit_costs, weights))
    result = dict(score_utility=score, variance=risk, modeled_volatility=math.sqrt(risk),
                  risk_penalty=risk_penalty, ridge_penalty=regularization,
                  rebalance_turnover=turnover, rebalance_cost_fraction=entry,
                  exit_cost_proxy_fraction=exit_proxy,
                  objective_utility=score-risk_penalty-regularization-entry-exit_proxy)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite allocation arithmetic")
    return result


def residuals(weights, lower, upper, gross_limit, target_net):
    return dict(box=max([0.0]+[max(lo-w, w-hi) for w, lo, hi in zip(weights, lower, upper)]),
                net=abs(math.fsum(weights)-target_net),
                gross=max(0.0, math.fsum(abs(w) for w in weights)-gross_limit))


def _coordinate(a, mu, previous, entry_cost, holding_cost, lower, upper):
    # Strongly convex, piecewise quadratic. Candidate roots plus all feasible
    # kink/endpoints exhaust possible global minima of this one-dimensional term.
    points = sorted({lower, upper, max(lower, min(upper, 0.0)), max(lower, min(upper, previous))})
    candidates = list(points)
    for lo, hi in zip(points, points[1:]):
        mid = (lo+hi)/2
        sign_trade = 1 if mid > previous else -1
        sign_hold = 1 if mid > 0 else -1
        root = (mu-entry_cost*sign_trade-holding_cost*sign_hold)/a
        if lo < root < hi:
            candidates.append(root)
    def objective(w):
        return .5*a*w*w-mu*w+entry_cost*abs(w-previous)+holding_cost*abs(w)
    return min(candidates, key=lambda w: (objective(w), w))


def solve_allocation(utility, covariance, previous, costs, exit_costs, lower, upper,
                     gross_limit, target_net=0.0, risk_aversion=1.0, ridge=0.5):
    n = len(utility)
    if not 1 <= n <= 64:
        raise ValueError("allocator supports 1..64 assets")
    vectors = [utility, previous, costs, exit_costs, lower, upper]
    if any(len(v) != n for v in vectors):
        raise ValueError("allocation vectors must have equal lengths")
    utility, previous, costs, exit_costs, lower, upper = [
        [finite(x, "allocation input") for x in v] for v in vectors]
    variances = diagonal_covariance(covariance, n)
    gross_limit, target_net, risk_aversion, ridge = [finite(v, "allocation scalar") for v in
                                                   (gross_limit, target_net, risk_aversion, ridge)]
    if gross_limit < 0 or risk_aversion < 0 or ridge <= 0 or any(c < 0 for c in costs+exit_costs):
        raise ValueError("nonnegative gross/risk/costs and strictly positive ridge required")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("infeasible position bounds")
    if not math.fsum(lower)-FEAS_TOL <= target_net <= math.fsum(upper)+FEAS_TOL:
        raise ValueError("infeasible net exposure")
    closest = [max(lo, min(hi, 0.0)) for lo, hi in zip(lower, upper)]
    minimum_gross = math.fsum(abs(w) for w in closest)+abs(target_net-math.fsum(closest))
    if gross_limit < minimum_gross-FEAS_TOL:
        raise ValueError("infeasible gross exposure for the net/position constraints")
    quadratic = [risk_aversion*v+ridge for v in variances]
    if not all(math.isfinite(a) and a > 0 for a in quadratic):
        raise ValueError("invalid quadratic coefficients")
    counts = dict(coordinate_evaluations=0, net_bisections=0, gross_bisections=0)

    def at_multipliers(net_dual, gross_dual):
        counts['coordinate_evaluations'] += n
        return [_coordinate(a, mu-net_dual, p, c, h+gross_dual, lo, hi)
                for a, mu, p, c, h, lo, hi in zip(quadratic, utility, previous, costs, exit_costs, lower, upper)]

    def net_solution(gross_dual):
        lo, hi = -1.0, 1.0
        for _ in range(ITERATIONS):
            if math.fsum(at_multipliers(lo, gross_dual)) >= target_net-FEAS_TOL/16:
                break
            lo *= 2
        else:
            raise ValueError("net multiplier lower bracket unavailable")
        for _ in range(ITERATIONS):
            if math.fsum(at_multipliers(hi, gross_dual)) <= target_net+FEAS_TOL/16:
                break
            hi *= 2
        else:
            raise ValueError("net multiplier upper bracket unavailable")
        for _ in range(ITERATIONS):
            counts['net_bisections'] += 1
            mid = (lo+hi)/2
            weights = at_multipliers(mid, gross_dual)
            gap = math.fsum(weights)-target_net
            if abs(gap) <= FEAS_TOL/16:
                return weights, mid
            if gap > 0:
                lo = mid
            else:
                hi = mid
        raise ValueError("net constraint did not converge")

    weights, net_dual = net_solution(0.0)
    gross_dual = 0.0
    if math.fsum(abs(w) for w in weights) > gross_limit+FEAS_TOL/8:
        lo, hi = 0.0, 1.0
        for _ in range(ITERATIONS):
            candidate, candidate_net = net_solution(hi)
            if math.fsum(abs(w) for w in candidate) <= gross_limit:
                break
            hi *= 2
        else:
            raise ValueError("gross multiplier bracket unavailable")
        weights, net_dual, gross_dual = candidate, candidate_net, hi
        for _ in range(ITERATIONS):
            counts['gross_bisections'] += 1
            mid = (lo+hi)/2
            candidate, candidate_net = net_solution(mid)
            gross = math.fsum(abs(w) for w in candidate)
            if gross <= gross_limit:
                hi = mid
                weights, net_dual, gross_dual = candidate, candidate_net, mid
            else:
                lo = mid
            if gross_limit-math.fsum(abs(w) for w in weights) <= FEAS_TOL/8:
                break
    metrics = terms(weights, previous, utility, variances, risk_aversion, ridge, costs, exit_costs)
    constraint_residuals = residuals(weights, lower, upper, gross_limit, target_net)
    primal = -metrics['objective_utility']
    net_error = math.fsum(weights)-target_net
    gross_error = math.fsum(abs(w) for w in weights)-gross_limit
    dual = primal+net_dual*net_error+gross_dual*gross_error
    gap = primal-dual
    complementarity = abs(gross_dual*gross_error)
    tolerance = GAP_TOL*(1+abs(primal)+abs(dual))
    if max(constraint_residuals.values()) > FEAS_TOL or abs(gap) > tolerance or complementarity > tolerance:
        raise ValueError("allocation certificate failed; no feasible/optimal target released")
    return dict(weights=weights, trades=[w-p for w, p in zip(weights, previous)], metrics=metrics,
                certificate=dict(status="PASS_EPSILON_FEASIBLE_OPTIMAL", feasibility_tolerance=FEAS_TOL,
                                 gap_tolerance=tolerance, residuals=constraint_residuals, primal=primal,
                                 dual_lower_bound=dual, duality_gap=gap, complementarity=complementarity,
                                 net_multiplier=net_dual, gross_multiplier=gross_dual, iterations=counts))
