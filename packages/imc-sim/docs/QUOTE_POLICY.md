# Inventory-aware quotes and a controlled execution study

Run the installed strategy, simulated execution and existing diagnostics together:

```sh
imc4-analyze quote-study --out quote-study
```

Open `quote-study/comparison.md` and follow each run's report link. Every run also has `trace.json`, normalized input and a code-bound report. `scenario.json` preserves the exact input and the root `complete.json` commits all traces, normalized inputs and nested reports. A partial output without that root receipt is incomplete; use a new directory to retry. This is a finite local simulation, not a live or restartable trading service.

The [packaged original fixture](../src/imc4_analysis/data/quoting_scenarios.json) contains a hand ledger, rising/falling paths and three execution assumptions. To run another bounded synthetic case, pass its JSON path before `--out`. Only the explicit synthetic schema is admitted; observed-market replay needs another input qualification.

The [frozen result interpretation](QUOTE_STUDY_RESULTS.md) follows the inventory/execution tradeoff and the loss caused by pending cancellations. It retains the adjusted policy's unfavorable outcomes.

## From utility to a quote center

The starting reference is [Avellaneda and Stoikov, *High-frequency trading in a limit order book*](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf), sections 2.1–2.2, equations 1 and 3–8; sections 2.4 and 3.2 distinguish reservation values from execution-dependent quote placement. Their driftless normal price model and exponential utility motivate the inventory adjustment. The implementation below is a fixed-spread approximation, not their full optimal quoting solution.

For the baseline derivation, write the terminal price as `S = s + sigma*sqrt(tau)*Z`, with standard normal `Z`, cash `x`, fixed inventory `q` and positive risk aversion `gamma`. Applying the normal moment-generating identity gives:

```text
E[-exp(-gamma*(x+q*S))]
  = -exp(-gamma*(x+q*s) + gamma²*q²*sigma²*tau/2)

certainty equivalent CE(q) = x + q*s - gamma*q²*sigma²*tau/2
```

The one-unit indifference buy value is `s - gamma*(q+1/2)*sigma²*tau`; the sell value is `s - gamma*(q-1/2)*sigma²*tau`. Their midpoint is

```text
r = s - q*gamma*sigma²*tau
```

The same expression is the marginal inventory value `dCE/dq`. Positive inventory moves the center downward, discouraging more buying and encouraging selling. Zero penalty uses the risk-neutral limit. This derivation concerns a fixed-inventory terminal valuation; it does not prove that our discrete constrained policy is globally optimal.

Our implementation chooses a separately declared positive half-spread `h` and quotes around `r`. Holding `h` fixed isolates the effect of the inventory shift in the comparison. We do not fit arrival intensities or infer an optimal spread from missing queue data. The deliberate comparison is `inventory_penalty = configured gamma` versus the same configuration with `gamma = 0`.

## Units, feasible orders and adaptation

| Quantity | Required interpretation |
|---|---|
| `fair_price`, `half_spread`, `tick_size` | Currency per instrument unit; integer or decimal string |
| `inventory` | Signed integer instrument units, in whole lots |
| `volatility` | Absolute price per square root of a synthetic tick; not a return percentage |
| `remaining_horizon` | Integer synthetic ticks, between zero and declared `max_horizon` |
| `inventory_penalty` | Inverse currency; nonnegative |
| `outstanding_buy`, `outstanding_sell` | Remaining live own units, including cancellation requests without acknowledgement |

The adjustment has units `(units)*(1/currency)*(currency²/units²/tick)*(ticks) = currency/units`. For example, `q=3`, `gamma=1`, `sigma=2`, `tau=2` shifts the center downward by 24. Doubling sigma quadruples the adjustment. Changing the clock requires transforming sigma so that `sigma²*tau` remains the same; labels alone cannot convert it.

The pure function [quote_orders](../src/imc4_analysis/quoting.py) takes only settings and the observed state. It has no client, random generator, file or clock access. Bid uses floor rounding and ask uses ceiling rounding:

```text
bid = tick * floor((r-h)/tick)
ask = tick * ceil ((r+h)/tick)

buy_capacity  = upper - inventory - outstanding_buy
sell_capacity = inventory - lower - outstanding_sell
size(side) = min(order_size, lot * floor(capacity(side)/lot))
```

This capacity rule adapts the unconstrained center to executable discrete orders. Opposite-side orders do not release capacity: a buy can fill while a sell does not. For `q=4`, upper bound 6 and one outstanding buy unit, a new buy may contain only one unit. With two outstanding buy units it is suppressed. Pending cancellations remain in that calculation until acknowledged. Infeasible existing inventory/outstanding orders refuse; there is no silent clipping of state. An off-tick, off-lot or oversized order also refuses, and the execution adapter rejects a new order that would cross an existing own order.

Nonpositive quote-price sides are suppressed. Zero remaining horizon emits no new orders. These choices do not liquidate residual inventory: the terminal result marks it at the last price. The capacity model covers inventory bounds; it is not a cash, credit or margin constraint. Invalid/missing values, percentage volatility labels and unsupported clocks refuse before an output directory is created.

## Execution mechanism and information timing

[quote_study.py](../src/imc4_analysis/quote_study.py) runs the same policy function through this sequence at every tick:

1. Acknowledge due cancellations, or expire all orders at the declared terminal tick.
2. Request cancellation of surviving old quotes. Zero delay acknowledges immediately; delayed requests remain executable.
3. Observe current fair price and volatility, own inventory and remaining live quantities. Produce and validate new orders. The policy is not given current demand or any future price.
4. Allocate external sellers to eligible own bids, then buyers to eligible own asks. Update own inventory from actual simulated fills.
5. Emit the observed market mark and own fills to the existing normalized analyzer. It alone computes cash, fees, equity and markouts.

Each external seller declares a floor price and finite units; each buyer declares a ceiling. Eligible own orders share one residual demand budget per side. That budget is `max(0, external_units - queue_ahead_units)`, allocated by own price priority then creation time. Fills occur at the resting own limit price, can be partial, and incur the declared fee per executed unit. The queue haircut is a sensitivity assumption, not measured priority. Price eligibility with zero residual demand produces no fill. There is no second budget for each own order, estimated intensity, market impact or reconstructed external order book.

The two policies receive exactly the same exogenous rows and execution settings. Their inventories, quotes, accepted orders and fills remain separate and endogenous. A shared external-input hash proves the input comparison; it does not imply identical fills. The deterministic paths are stress examples, not samples from a fitted stochastic process.

## Frozen comparisons and the independent hand ledger

The rising and falling cases use marks `100..108` and `100..92`. Sellers submit three units at `mark-1` during ticks 0–3; buyers submit three at `mark+1` during ticks 4–7. Both policies use half-spread 1, tick .5, lot 1, requested size 2, bounds ±6, sigma 1 and a .1 fee per executed unit. Adjusted gamma is .5. Three variants change one assumption at a time: immediate cancellation with queue 0; queue 2 with immediate cancellation; one-tick cancellation delay with queue 0. No parameter search or winner selection occurs.

The smaller hand case uses gamma 1, size 1, bounds ±2 and tick 1. Both policies buy one at 99 when the initial fair mark is 100. At tick 1 the fair mark remains 100: the adjusted policy quotes around 99 and sells at 100 to a buyer with ceiling 100; the symmetric ask at 101 does not fill. The terminal mark is 98. Starting from cash 1000, with .1 per filled unit:

| Immediate, queue 0 | Final cash | Inventory | Equity | Marked P&L | Fees | Inventory-square time |
|---|---:|---:|---:|---:|---:|---:|
| Symmetric | 900.9 | 1 | 998.9 | −1.1 | .1 | 2 |
| Inventory adjusted | 1000.8 | 0 | 1000.8 | .8 | .2 | 1 |

Inventory-square time is the integral of post-fill `q²` over the following one-tick interval. The comparison also reports peak absolute inventory, partial-fill events, units filled during pending cancellation and terminal inventory. Lower exposure can give up spread capture or directional gains; all negative and underperforming cases remain in the result. A reduced time integral is a path-specific exposure observation, not proof of improved risk-adjusted returns.

## Why use the existing diagnostics

Compare inventory paths and equity changes, then inspect fills around a loss or pending cancellation. The report distinguishes fees, current marked inventory and retrospective signed markouts. A bad markout can expose adverse selection in this simulation; it does not identify a causal source of market alpha. Use the adjacent trace to find the quote center, capacity, live-order state, cancellation timing and finite demand behind that fill. No new accounting or chart framework is required.

The tests link the mathematics to code through hand values, sigma/horizon scaling, ticks/lots, live-order bounds, independent cash/position conservation, demand-volume budgets, delayed/partial fills, deterministic replay and future-input exclusion. Old analyzer and importer cases remain in the same suite. Input is capped at 256 KiB, eight scenarios, four execution variants and 64 decision ticks plus a terminal mark. Existing normalized caps remain in force. Larger/observed studies, calibration, queue realism and operational restart are separate work. This is newly authored post-competition tooling; it does not recover the missing IMC4 team submission or make IMC3 history into IMC4 evidence.
