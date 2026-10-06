import pytest

from pmlab.fees import fee_per_share, taker_fee


@pytest.mark.case
def test_fee_formula_matches_docs():
    assert taker_fee(100, 0.5) == pytest.approx(1.75)              # 100 * 0.07 * .25
    assert taker_fee(100, 0.3) == taker_fee(100, 0.7) == pytest.approx(1.47)
    assert fee_per_share(0.9) == pytest.approx(0.07 * 0.09)


@pytest.mark.case
def test_fee_rounding_minimum_and_certain_prices():
    assert taker_fee(1, 0.123456) == round(0.07 * 0.123456 * 0.876544, 5)
    assert taker_fee(0.001, 0.999) == 1e-5                          # tiny but non-zero -> minimum
    assert taker_fee(10, 1.0) == 0.0 and taker_fee(10, 0.0) == 0.0


@pytest.mark.case
def test_maker_rebate_is_a_share_of_the_generated_taker_fee_and_off_by_default():
    from pmlab.backtest import Config, Order, simulate_fill
    import pandas as pd
    from pmlab.fees import maker_rebate
    assert maker_rebate(100, 0.5) == pytest.approx(0.2 * 1.75)
    tp = pd.DataFrame({"ts": [5.0], "sign": [-1], "p_up": [0.49], "shares": [100.0]})
    [off] = simulate_fill(tp, 0, 3, Order("up", 10, 0.50, maker=True), Config(print_delay=0))
    [on] = simulate_fill(tp, 0, 3, Order("up", 10, 0.50, maker=True), Config(print_delay=0, maker_rebate=0.2))
    assert off.fee == 0.0 and on.fee == pytest.approx(-maker_rebate(10, 0.5))
