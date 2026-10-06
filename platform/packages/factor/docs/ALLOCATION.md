# From factor scores to holdings and trades

The `allocate` command makes a bounded portfolio decision using a verified research experiment, a prior portfolio and a fixed scenario contract. It emits target holdings, share changes, fees and cash, then values scheduled liquidation at the next observation. It compares these decisions with the evaluator's existing rank weights under the same information, initial state and costs.

From the monorepo root (see the [package README](../README.md) for installation):

```sh
factor-research allocate --research-run work/runs/csv-demo --scenario packages/factor/examples/allocation/scenario.json --out work/allocation-demo
factor-research verify-allocation work/allocation-demo
python3 packages/factor/tools/allocation_reference_check.py work/allocation-demo
```

The research run must have been generated with the same installed package code. Keep earlier wheels to recompute earlier immutable runs; create a new run ID when upgrading. The allocation bundle copies the research record and scenario and commits hashes after writing the report. An identical request is verified and reused without rewriting bytes; conflicts, tampering and incomplete destinations are refused. These are local integrity checks, not signed protection against a malicious writer replacing the entire record. `VERIFIED_ALLOCATION` describes bundle integrity; individual scenarios can still be refused or have unavailable outcomes.

## Mathematical decision

For assets i, let p be the marked prior holdings and w the target holdings, both divided by fixed reference capital C. The solver minimizes

```
F(w) = gamma/2 * w' Sigma w + ridge/2 * ||w||² - mu'w
       + sum_i c_i |w_i-p_i| + sum_i h_i |w_i|

lower_i <= w_i <= upper_i
sum_i w_i = target_net
sum_i |w_i| <= gross_limit
```

| Input or result | Meaning and units |
|---|---|
| C, cash, P&L | Declared currency units; no currency conversion |
| p, w, net, gross, position bounds | Marked dollar exposure divided by fixed C |
| mu | Fixed-scale score utility per unit weight; **not expected return** |
| Sigma | Diagonal covariance of one-grid-interval simple returns, in squared return units |
| gamma | User-declared conversion from variance to objective utility |
| ridge > 0 | User-declared quadratic regularization in objective utility units |
| c, h | Proportional fee rate; the scenario converts basis points by dividing by 10,000 |
| w' Sigma w, square root | Modeled variance and volatility for one grid interval, without annualization |
| F and its terms | Dimensionless scenario objective; its scale is a modeling assumption |

The scenario sets c=h to the same fee rate. Rebalance cost is exact at current marks. The h term estimates scheduled liquidation cost using current marks; realized closing fees use next-period prices. Those future prices never determine w. No cost-free terminal inventory is implicitly rewarded.

This narrower adaptation follows the separation between forecasts, risk/cost models and portfolio optimization described on the primary [Boyd et al. paper page, Multi-Period Trading via Convex Optimization (2017)](https://stanford.edu/~boyd/papers/cvx_portfolio.html). The page's overview was consulted; this implementation does not reproduce the full paper, a general covariance solver or a multi-period controller. Diagonal risk preserves separability and removes an external numerical dependency. It also ignores diversification and concentration caused by correlations, so it is a substantial modeling limit.

The adapter obtains the validation-selected factor's strictly lagged signals. Existing centered ranks are divided by their maximum absolute value and multiplied by `score_scale`. All-constant or insufficient signals yield zero utility. This rescaling is fixed arithmetic, not return calibration. The risk matrix contains per-asset sample variances using only consecutive training-period returns whose endpoint marks were available at those endpoints. Each asset needs at least two such returns. Nonzero off-diagonal matrix entries are explicitly refused by the solver; supplied correlations are never silently discarded. Nonnegative diagonal entries establish PSD; positive ridge establishes strong convexity even when a variance is zero.

The scenario's `policy_fixed_by` must be no later than training end. This is a declaration, not proof of historical preregistration or honest calibration. The example's scale, risk aversion, ridge and four cost/constraint cases are hand-authored assumptions. The test decision and its next mark must both lie inside the test split. A test label mutation is checked to preserve selection, risk and target weights. Reusing exposed test outcomes to adjust a later scenario would consume that holdout; the tool cannot police a human's experiment history.

## Why the small solver works

Write a_i=gamma*Sigma_ii+ridge>0. With net multiplier nu and gross multiplier lambda>=0, the box-constrained Lagrangian is separable:

```
L(w,nu,lambda) = sum_i [a_i*w_i²/2 - (mu_i-nu)*w_i
                         + c_i*|w_i-p_i| + (h_i+lambda)*|w_i|]
                 - nu*target_net - lambda*gross_limit
```

Each coordinate is a strongly convex piecewise quadratic. Candidate minimizers are the box endpoints, feasible kinks at 0 and p_i, and a stationary root inside each intervening interval. On an interval with signs s_trade and s_hold, that root is `(mu_i-nu-c_i*s_trade-(h_i+lambda)*s_hold)/a_i`. Evaluating this finite list obtains the coordinate minimum without a gradient step size.

For fixed lambda, total w is continuous and nonincreasing in nu. Bounded bracket expansion followed by bisection enforces the net equality. At lambda=0, an already feasible gross exposure is optimal. Otherwise increasing lambda penalizes gross exposure; the optimal gross exposure with the net equality is nonincreasing. A second bounded bracket and bisection finds a feasible gross constraint. All loops have an 80-iteration cap and failures release no accepted target. The algorithm is intentionally bounded to 64 assets and eight scenarios, with no scale or speed claim.

Before solving, the net target must lie between the sums of the lower and upper bounds. Let b be zero clipped into each box. The minimum possible gross at the required net is `sum(abs(b)) + abs(target_net-sum(b))`: shifting the net from the minimum-gross box point adds that much absolute exposure, and box/net feasibility guarantees sufficient directional capacity. A lower gross cap is infeasible.

For the returned coordinate minimizer, d=L(w,nu,lambda) is the computed dual lower bound. The report checks box, absolute net and positive gross residuals <=1e-9, and checks the absolute primal-minus-dual gap and gross complementarity <=`1e-8*(1+abs(primal)+abs(dual))`. Exact feasibility and a zero duality gap certify the convex optimum; the implemented result is an **epsilon feasibility/optimality check in floating-point arithmetic**, not an exact rational or interval-arithmetic proof. The report preserves multipliers, residuals and iteration counts. Ill-scaled finite inputs can fail to converge and are refused. A near-zero signed numerical gap is retained, not clamped into better-looking evidence.

## Independent tiny oracle and accounting

With mu=(.1,-.1), Sigma=diag(.5,.5), gamma=1, ridge=.5, c=h=(.01,.01), p=0 and net=0, write w=(x,-x). For x>=0, maximized utility is `.16*x-x²`; its derivative gives x=.08, gross=.16 and utility=.0064. Gross cap .12 gives x=.06 and utility=.006; per-position cap .04 gives x=.04. Tests compare the solver with this hand result and a separate exact Fraction grid over x. A second three-asset grid compares objective values over feasible points. Grid agreement supports those finite examples, not arbitrary-dimensional correctness.

The initial state must reconcile: `cash_before+C*sum(p)=C`. At current prices P, prior shares are `C*p/P`, target shares `C*w/P`, and trades are their difference. Cash becomes `cash_before-sum(trade*P)-fee*sum(abs(trade)*P)`. Negative cash beyond tolerance refuses the state transition. Current equity must equal C minus rebalance fees. This cash gate follows the optimizer: it may reject a mathematically valid allocation; there is no second optimization or claim that cash is inside the feasible set. Constraints are measured against C, not fee-reduced equity. The model does not reserve short collateral or enforce broker margin.

At next prices Q, liquidate all target shares. Gross P&L is `sum(shares*(Q-P))`; terminal fee is `fee*sum(abs(shares)*Q)`; net P&L subtracts both fees. Terminal cash less C must reconcile to net P&L. For C=1000, prior weights (.03,-.03), target (.08,-.08), current prices (100,100), next prices (110,90) and 1% fees: trades=(.5,-.5) shares, cash after rebalance=999, gross P&L=16, terminal fee=1.6, net P&L=13.4 and terminal cash=1013.4. `packages/factor/tools/allocation_reference_check.py` reconstructs state, fees and objective from saved inputs without importing package code.

Missing current marks on nonzero prior holdings refuse the decision. Missing signals force that asset's target to zero when its current mark exists. A missing future mark on any nonzero target leaves realized P&L and terminal state unavailable; targets are preserved and no survivor reweighting occurs. The comparison is suppressed until both alternatives have complete outcomes.

## Demonstration and useful decision

The hand-authored example starts with weights (.03,0,-.03), C=1000 and cash=1000. It uses three training returns per asset and one test-period decision. Four fixed cases expose low cost, high cost, tight gross and tight position behavior. The raw rank comparator always uses the unchanged evaluator weighting. Its constraint violations remain visible; objective-superiority comparisons are suppressed when it is infeasible. JSON includes both portfolios' variance, turnover, costs, objectives and cash transitions; Markdown is a readable view.

The installed 0.3.0 demonstration produced the following values. Capital is 1000 currency units, raw rank weights are (.5,0,-.5), and raw rank rebalance turnover is .94 of capital in every case. Its modeled variance is approximately 8.0395791e-8.

| Case | A/C target weights | Policy turnover / C | Policy variance | Policy net P&L | Raw rank net P&L | Rank feasible |
|---|---|---:|---:|---:|---:|---|
| 10 bps | +.17971104 / -.17971104 | .29942208 | 1.0385869e-8 | 6.6354986 | 18.354643 | Yes |
| 500 bps | +.03 / -.03 | 0 | 2.8942485e-10 | -1.7751623 | -76.586039 | Yes |
| Gross cap .1, 10 bps | +.05 / -.05 | .04 | 8.0395791e-10 | 1.8894643 | 18.354643 | No |
| Position cap .04, 10 bps | +.04 / -.04 | .02 | 5.1453307e-10 | 1.5235714 | 18.354643 | No |

The maximum observed net residual was 5.01e-11; box and positive gross residuals were zero. The largest absolute computed duality gap was 3.93e-13. These are numerical checks on the four synthetic examples. At low cost, the lower-risk allocation earns less than rank weighting in the observed interval despite a higher scenario objective. At high cost, retaining prior holdings avoids rebalance fees but still incurs terminal fees and loses money. Lower modeled risk or a better score objective is not evidence of higher realized returns. Tiny synthetic training variances, assumed availability clocks, fractional close fills, absent borrow/funding/impact and diagonal risk prevent a market-performance claim.

The practical lesson is that a factor rank does not itself specify portfolio risk, turnover or a self-financing trade. Separate the signal scale, risk estimate, constraints and cash state, then expose the tradeoff against a baseline. A future empirical study would need licensed point-in-time data, development-only scale/risk calibration, realistic execution and untouched evaluation periods; none of those gates is completed here.
