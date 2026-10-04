# Fixed weighted-midpoint hypothesis — original local derivation

This note was authored for the 14 September 2026 software integration example.
It is not a summary of a paper or a claim about a competition result.

For best bid b, best ask a, displayed bid size B and ask size A, the installed
feature is w = (a B + b A) / (A + B). The ordinary midpoint is m = (a + b) / 2.
Algebra gives w - m = ((a - b) / 2) ((B - A) / (A + B)). Thus larger displayed
bid size moves the forecast toward the ask. This identity describes the feature;
it does not prove that book imbalance predicts a later price.

Testable hypothesis: w predicts the next decision-time as-of midpoint with
lower mean absolute error than m on the declared split. The output price grid
is 0.0001 and reported errors are ticks. Compare both forecasts on the same
eligible pairs. Exclude stale/missing quotes and continuity breaks according
to the existing feed/feature evaluator.

Assumptions to investigate with actual data include informative displayed
liquidity, usable timestamps and queue behavior. Cancellations, hidden liquidity,
spoofing, latency and regime changes can defeat the hypothesis. No fill model,
fees or execution costs enter this forecast-error comparison.

The provided archive was already inspected. Its validation set has one eligible
pair out of three, with candidate MAE 50 and midpoint MAE 0. The example's minimum
count of one and coverage of one third exercise the software decision only;
they are not sensible evidence thresholds for a market research conclusion.
A larger frozen campaign requires a separate sample-size, uncertainty,
dependence and multiple-testing design. This one-step weighted midpoint is not
a fitted microprice model.
