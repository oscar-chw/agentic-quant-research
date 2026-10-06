# Local review note: inventory-adjusted reservation quotes

Method background: Avellaneda and Stoikov, High-frequency trading in a limit order book, sections 2.1–2.2. Primary source: https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf . This is a newly authored local reading note, not copied paper text or an original competition submission.

The accepted IMC implementation values fixed inventory with certainty equivalent x + q*s - gamma*q*q*sigma*sigma*tau/2 and shifts the reservation center to s - gamma*q*sigma*sigma*tau. Gamma has inverse-currency units, sigma is price per square-root synthetic tick, and tau is remaining ticks. Its independently fixed half-spread isolates inventory adjustment; it does not implement the paper's full optimal spread or fitted arrival process.

The native package applies outward tick rounding, whole lots and worst-case outstanding-order capacity. Pending cancellations still reserve capacity and may fill. A finite synthetic demand path supplies side-specific fill budgets; the existing analyzer accounts for cash, fees, positions and marked terminal equity. These assumptions can make lower inventory exposure coincide with worse marked P&L.

The frozen study compares both policies on the same exogenous paths and retains delayed-cancellation losses. The QuantOS adapter binds this note to those exact inputs before execution, invokes the native method and exposes its trace. This local note has no source-authenticity, untouched-holdout, market-quality or publication-rights qualification.
