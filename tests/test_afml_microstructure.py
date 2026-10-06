import math

import numpy as np
import pytest
from scipy import stats

from pmlab import micro
from pmlab.afml import microstructure as ms


@pytest.mark.eval
def test_beckers_parkinson_recovers_brownian_volatility():
    """Threshold: 2000 bars of 2000 Brownian steps with sigma 0.01 per bar; both moment estimators within
    4% (the discrete path's range is biased low by about 1.3% at this step count).

    19.3.3's actual claim is a COMPARISON -- a high-low estimator is more accurate than the standard
    close-to-close one -- so the two are run on the same bars: per bar, sqrt(log(H/L)^2 / (4 log 2)) against
    |log(C/O)| sqrt(pi/2), both unbiased-ish for sigma under this process. Threshold: the close-to-close
    estimate's root mean squared error is more than twice the range estimate's."""
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.01 / math.sqrt(2000), (2000, 2000))
    path = np.concatenate([np.zeros((2000, 1)), np.cumsum(steps, axis=1)], axis=1)
    est = ms.beckers_parkinson(np.exp(path.max(axis=1)), np.exp(path.min(axis=1)))
    assert est["sigma_sq"] == pytest.approx(0.01, rel=0.04)
    assert est["sigma_abs"] == pytest.approx(0.01, rel=0.04)
    per_bar_range = np.sqrt(np.log(np.exp(path.max(axis=1)) / np.exp(path.min(axis=1))) ** 2 / ms.K1)
    per_bar_close = np.abs(path[:, -1]) * math.sqrt(math.pi / 2)
    rmse = lambda e: float(np.sqrt(np.mean((e - 0.01) ** 2)))
    assert rmse(per_bar_close) > 2 * rmse(per_bar_range)
    assert per_bar_close.std() > 2 * per_bar_range.std()


@pytest.mark.case
def test_beckers_parkinson_constants():
    hl = np.array([1.02, 1.05])
    est = ms.beckers_parkinson(hl, [1.0, 1.0])
    logs = np.log(hl)
    assert est["sigma_sq"] == pytest.approx(math.sqrt(np.mean(logs ** 2) / (4 * math.log(2))))
    assert est["sigma_abs"] == pytest.approx(np.mean(logs) / math.sqrt(8 / math.pi))


@pytest.mark.eval
def test_hasbrouck_lambda_recovers_the_impact_slope():
    """Threshold: 5000 bars, true lambda 0.002, estimate within 4 standard errors and |t| > 10."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, 5000)
    r = 0.002 * x + rng.normal(0, 0.01, 5000)
    est = ms.hasbrouck_lambda(r, x)
    se = est["lambda"] / est["t"]
    assert abs(est["lambda"] - 0.002) < 4 * se and est["t"] > 10
    assert est["lambda"] == pytest.approx(x @ r / (x @ x))


@pytest.mark.case
def test_bulk_volume_classification():
    assert ms.bulk_volume_buy([0.0], [10.0], 0.01).tolist() == [5.0]
    assert ms.bulk_volume_buy([0.05], [10.0], 0.01)[0] == pytest.approx(10 * stats.norm.cdf(5))
    assert ms.bulk_volume_buy([-0.01], [10.0], 0.01, df=3)[0] == pytest.approx(10 * stats.t.cdf(-1, 3))


@pytest.mark.case
def test_volume_buckets_split_bars_and_conserve_volume():
    b = ms.volume_buckets([3, 5, 4], [3, 0, 2], 4)
    assert b["buy"].tolist() == pytest.approx([3, 0, 2]) and b["sell"].tolist() == pytest.approx([1, 4, 2])
    assert b["end_bar"].tolist() == [1, 1, 2]
    rng = np.random.default_rng(2)
    v = rng.uniform(0, 10, 500)
    bb = ms.volume_buckets(v, v * rng.uniform(0, 1, 500), 37.0)
    assert (bb["buy"] + bb["sell"]) == pytest.approx(np.full(len(bb["buy"]), 37.0))
    assert len(bb["buy"]) == int(v.sum() // 37.0)


@pytest.mark.case
def test_vpin_hand_case_and_extremes():
    out = ms.vpin([3, 0, 2], [1, 4, 2], n=2)
    assert np.isnan(out[0]) and out[1:].tolist() == pytest.approx([0.75, 0.5])
    assert ms.vpin(np.ones(60), np.zeros(60), 50)[-1] == 1.0
    assert ms.vpin(np.ones(60), np.ones(60), 50)[-1] == 0.0


@pytest.mark.eval
def test_bvc_vpin_tracks_true_vpin_when_flow_moves_prices():
    """Threshold: simulated bars whose price change is 0.5 x the signed flow plus noise; the correlation of
    BVC-VPIN with true-side VPIN over 20-bucket windows exceeds 0.7, and with price-independent flow it is
    below 0.3."""
    rng = np.random.default_rng(3)
    n = 20000
    vol = rng.uniform(5, 15, n)
    tox = np.repeat(rng.uniform(0.05, 0.95, n // 200), 200)          # regimes of one-sidedness
    side = np.where(rng.random(n) < 0.5, 1, -1)
    buy_share = np.where(side > 0, tox, 1 - tox)
    true_buy = vol * buy_share
    signed = true_buy - (vol - true_buy)
    for coupling, bound in ((0.5, 0.7), (0.0, 0.3)):
        dp = coupling * signed + rng.normal(0, 1.0, n)
        bvc = ms.bulk_volume_buy(dp, vol, dp.std())
        t = ms.volume_buckets(vol, true_buy, 200.0)
        e = ms.volume_buckets(vol, bvc, 200.0)
        vt, ve = ms.vpin(t["buy"], t["sell"], 20), ms.vpin(e["buy"], e["sell"], 20)
        ok = np.isfinite(vt) & np.isfinite(ve)
        c = np.corrcoef(vt[ok], ve[ok])[0, 1]
        assert (c > bound) if coupling else (abs(c) < bound)


@pytest.mark.case
def test_bar_order_imbalance_is_the_old_micro_measure():
    buy, sell = [5, 0, 2], [1, 3, 2]
    assert ms.bar_order_imbalance(buy, sell) == micro.vpin(buy, sell) == pytest.approx((4 + 3 + 0) / 13)


@pytest.mark.case
def test_corwin_schultz_series_matches_the_one_pair_live_form_and_averages_pairs():
    from pmlab.micro import corwin_schultz
    h = np.array([0.52, 0.55, 0.54, 0.60, 0.58])
    lo = np.array([0.50, 0.51, 0.50, 0.53, 0.55])
    s = ms.corwin_schultz_spread(h, lo)
    assert np.isnan(s[0])
    for t in range(1, 5):
        live = corwin_schultz(h[t - 1], lo[t - 1], h[t], lo[t]) / (h[t - 1] * lo[t - 1] * h[t] * lo[t]) ** 0.25
        assert s[t] == pytest.approx(live, abs=1e-12)
    b2 = ms.corwin_schultz_beta(h, lo, sl=2)
    b1 = ms.corwin_schultz_beta(h, lo, sl=1)
    assert np.isnan(b2[:2]).all() and b2[2:] == pytest.approx((b1[1:-1] + b1[2:]) / 2)
    beta, gamma = b1[1:], ms.corwin_schultz_gamma(h, lo)[1:]
    paper = (np.sqrt(beta / 2) - np.sqrt(beta)) / (ms.K2 * (3 - 2 * np.sqrt(2))) + np.sqrt(gamma / (ms.K2 ** 2 * (3 - 2 * np.sqrt(2))))
    assert ms.high_low_sigma(beta, gamma) == pytest.approx(np.maximum(paper, 0.0))


@pytest.mark.eval
def test_high_low_sigma_ignores_the_spread_that_inflates_the_plain_range_estimator():
    """Threshold: 4,000 bars of 500-step Brownian log prices (sigma 0.01 per bar), prints bouncing between bid and ask
    with relative spread 0, 0.005, 0.01: the Snippet 19.2 sigma stays within 6% of 0.01 at every spread while
    Beckers-Parkinson overstates it by more than 40% at 0.01; the Corwin-Schultz spread rises with the true spread and
    is within 30% of it at 0.01."""
    rng = np.random.default_rng(0)
    means = []
    for spread in (0.0, 0.005, 0.01):
        z = np.cumsum(rng.normal(0, 0.01 / np.sqrt(500), 4000 * 500)).reshape(4000, 500)
        px = np.exp(z) * (1 + rng.choice([-1, 1], size=z.shape) * spread / 2)
        h, lo = px.max(axis=1), px.min(axis=1)
        sigma = ms.high_low_sigma(ms.corwin_schultz_beta(h, lo), ms.corwin_schultz_gamma(h, lo))
        assert np.nanmean(sigma) == pytest.approx(0.01, rel=0.06)
        means.append(np.nanmean(ms.corwin_schultz_spread(h, lo)))
    assert ms.beckers_parkinson(h, lo)["sigma_sq"] > 1.4 * 0.01
    assert means[0] < means[1] < means[2] and means[2] == pytest.approx(0.01, rel=0.3)


@pytest.mark.eval
def test_kyles_equilibrium_is_the_lambda_the_signed_volume_regression_recovers():
    """19.4.1: in Kyle's equilibrium mu = p0, lambda = (1/2) sqrt(Sigma0 / sigma_u^2), beta = 1 / (2 lambda) and
    alpha = -mu / (2 lambda), so the informed trader's optimum x = (v - mu) / (2 lambda) is exactly the linear
    demand alpha + beta v the market maker conjectured, and the maker's rule p = lambda y + mu is E[v | y] (the
    market-efficiency half of the equilibrium). Illiquidity rises with uncertainty about v and falls with noise.
    As a feature lambda is the slope of dp on signed volume, which the origin regression recovers.
    Thresholds (200,000 draws): E[v | y] slope and intercept within 2%, mean profit within 3%."""
    p0, sigma0, sigma_u = 30.0, 4.0, 9.0
    lam = 0.5 * math.sqrt(sigma0 / sigma_u ** 2)
    beta, alpha, mu = 1 / (2 * lam), -p0 / (2 * lam), p0
    assert beta == pytest.approx(math.sqrt(sigma_u ** 2 / sigma0)) and lam > 0
    assert alpha == pytest.approx(-p0 * math.sqrt(sigma_u ** 2 / sigma0))
    rng = np.random.default_rng(0)
    v = p0 + math.sqrt(sigma0) * rng.standard_normal(200_000)
    u = sigma_u * rng.standard_normal(200_000)
    x = (v - mu) / (2 * lam)
    assert x == pytest.approx(alpha + beta * v)                     # the two conjectures agree
    y = x + u
    p = lam * y + mu
    # the maker's linear rule is the conditional expectation of v given the order flow it sees
    fit = np.polyfit(y, v, 1)
    assert fit[0] == pytest.approx(lam, rel=0.02) and fit[1] == pytest.approx(p0, rel=0.02)
    # the informed trader's three sources of profit: mispricing, noise variance, and 1 / terminal variance
    v_fixed = p0 + 3.0
    x_fixed = (v_fixed - mu) / (2 * lam)
    profit = (v_fixed - (lam * (x_fixed + u) + mu)) * x_fixed
    assert profit.mean() == pytest.approx((v_fixed - p0) ** 2 / (4 * lam), rel=0.03)
    assert 0.5 * math.sqrt(9.0 / sigma_u ** 2) > lam                # more uncertainty about v: less liquidity
    assert 0.5 * math.sqrt(sigma0 / 20.0 ** 2) < lam                # more noise to hide in: more liquidity
    # as a feature: dp = lambda (b V) + e, fitted through the origin as the section writes it
    assert micro.origin_slope(p - p0, y)[0] == pytest.approx(lam, rel=1e-9)
    noisy = p - p0 + 0.05 * rng.standard_normal(len(y))
    assert micro.origin_slope(noisy, y)[0] == pytest.approx(lam, rel=0.02)


@pytest.mark.eval
def test_amihuds_price_response_per_dollar_is_recovered_by_its_regression():
    """19.4.2: dlog[p] of a bar regressed on the bar's dollar volume gives lambda, the price response per dollar
    traded. The ratio form the live features use (mean |dlog p| / dollars) is not that slope: on the same bars
    it reads the mean of a ratio, which is pulled up by small bars, so it is a different statistic that only
    ranks with it (docs/afml_sections/ch19.md F19.4). Thresholds (40,000 bars, lambda 2.5e-7, bar noise 0.02):
    both regressions within 3% of the truth, while the ratio form reads more than twice it."""
    rng = np.random.default_rng(1)
    lam, n = 2.5e-7, 40_000
    dollars = np.exp(rng.normal(np.log(40_000), 0.9, n))
    ret = lam * dollars + 0.02 * rng.standard_normal(n)
    assert micro.ols_slope(ret, dollars) == pytest.approx(lam, rel=0.03)
    assert micro.origin_slope(ret, dollars)[0] == pytest.approx(lam, rel=0.03)
    # the ratio form is the mean of |return| / dollars, which the smallest bars dominate: a different statistic
    assert micro.amihud(np.abs(ret), dollars) > 2 * lam
    quiet = lam * dollars + 1e-5 * rng.standard_normal(n)           # with little noise the two nearly agree
    assert micro.amihud(np.abs(quiet), dollars) == pytest.approx(lam, rel=0.02)


@pytest.mark.eval
def test_the_roll_model_returns_both_the_spread_and_the_efficient_price_variance():
    """19.3.2: with p_t = m_t + b_t c, m a driftless random walk and b a fair independent coin, the observed price
    changes satisfy sigma^2[dp] = 2c^2 + sigma_u^2 and sigma[dp_t, dp_(t-1)] = -c^2, so c = sqrt(max(0, -cov))
    and sigma_u^2 = sigma^2[dp] + 2 cov. Thresholds (200,000 prints): both moments within 3%, the recovered half
    spread and efficient-price variance within 4%. Only the spread is computed in code (F19.1)."""
    rng = np.random.default_rng(2)
    c, sigma_u, n = 0.004, 0.01, 200_000
    m = np.cumsum(sigma_u * rng.standard_normal(n))
    b = rng.choice([-1.0, 1.0], n)
    p = m + b * c
    dp = np.diff(p)
    cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
    assert float(np.var(dp, ddof=1)) == pytest.approx(2 * c ** 2 + sigma_u ** 2, rel=0.03)
    assert cov == pytest.approx(-c ** 2, rel=0.03)
    assert math.sqrt(max(-cov, 0.0)) == pytest.approx(c, rel=0.02)
    assert micro.roll_spread(p) == pytest.approx(2 * c, rel=0.02)
    assert float(np.var(dp, ddof=1)) + 2 * cov == pytest.approx(sigma_u ** 2, rel=0.04)
    # a series with no bid-ask bounce has a non-negative covariance, where the estimator floors the spread at 0
    assert micro.roll_spread(m) == 0.0 or micro.roll_spread(m) < 0.1 * c


@pytest.mark.case
def test_the_breakeven_bid_ask_spread_is_pin_times_the_range_between_the_two_news():
    """19.5.1: the breakeven ask and bid of a market maker facing informed arrivals at rate mu and uninformed at
    rate epsilon give E[A - B] = k (S_G - E[S]) + k (E[S] - S_B) with k = alpha mu / (alpha mu + 2 epsilon) when
    the news is equally likely to be good or bad, and that telescopes to PIN (S_G - S_B) whatever the starting
    price. PIN rises with the informed rate and falls with the uninformed one."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        alpha, delta = rng.uniform(0.05, 0.95), 0.5
        mu, eps = rng.uniform(1, 400), rng.uniform(1, 400)
        s0, s_b = rng.uniform(5, 100), rng.uniform(1, 4)
        s_g = s_b + rng.uniform(1, 30)
        e_s = (1 - alpha) * s0 + alpha * (delta * s_b + (1 - delta) * s_g)
        ask = e_s + mu * alpha * (1 - delta) / (eps + mu * alpha * (1 - delta)) * (s_g - e_s)
        bid = e_s - mu * alpha * delta / (eps + mu * alpha * delta) * (e_s - s_b)
        pin = alpha * mu / (alpha * mu + 2 * eps)
        assert ask - bid == pytest.approx(pin * (s_g - s_b))
        assert alpha * (2 * mu) / (alpha * 2 * mu + 2 * eps) > pin       # more informed flow: wider spread
        assert alpha * mu / (alpha * mu + 2 * 2 * eps) < pin             # more uninformed flow: tighter


@pytest.mark.eval
def test_vpin_recovers_the_pin_of_a_simulated_poisson_mixture_once_mu_is_large():
    """19.5.2 and 18.8.4: draw buy and sell volumes from 19.5.1's three-Poisson mixture with known
    {alpha, delta, mu, epsilon}. Then E[V_B - V_S] = alpha mu (1 - 2 delta) exactly and E[V_B + V_S] = alpha mu +
    2 epsilon exactly, while E[|V_B - V_S|] = alpha mu only "for a sufficiently large mu": the absolute value also
    picks up the uninformed imbalance of the quiet buckets, about (1 - alpha) sqrt(4 epsilon / pi), and VPIN
    inherits that same upward bias. Thresholds (20,000 buckets, alpha 0.3, epsilon 700): the two exact
    expectations within 4 standard errors; at mu = 900 both |.| and VPIN read 8-10% high and match the noise
    floor within 2%; at mu = 9000 the bias falls under 2%."""
    def mixture(mu, seed):
        rng = np.random.default_rng(seed)
        alpha, delta, eps, n = 0.3, 0.4, 700.0, 20_000
        event = rng.random(n) < alpha
        bad = event & (rng.random(n) < delta)
        buy, sell = rng.poisson(eps + mu * (event & ~bad)), rng.poisson(eps + mu * bad)
        return alpha, delta, eps, n, buy, sell

    alpha, delta, eps, n, buy, sell = mixture(900.0, 4)
    mu = 900.0
    diff, total = buy - sell, buy + sell
    assert abs(diff.mean() - alpha * mu * (1 - 2 * delta)) < 4 * diff.std(ddof=1) / math.sqrt(n)
    assert abs(total.mean() - (alpha * mu + 2 * eps)) < 4 * total.std(ddof=1) / math.sqrt(n)
    floor = (1 - alpha) * math.sqrt(4 * eps / math.pi)          # mean |Poisson(eps) - Poisson(eps)| in quiet buckets
    assert abs(np.abs(diff).mean() - (alpha * mu + floor)) < 4 * np.abs(diff).std(ddof=1) / math.sqrt(n)
    assert 1.05 < np.abs(diff).mean() / (alpha * mu) < 1.12
    pin = alpha * mu / (alpha * mu + 2 * eps)
    vpin_all = np.abs(diff).sum() / total.sum()
    assert vpin_all == pytest.approx((alpha * mu + floor) / (alpha * mu + 2 * eps), rel=0.02)
    assert 1.08 < vpin_all / pin < 1.10
    series = ms.vpin(buy, sell, n=1000)
    assert np.isnan(series[:999]).all() and np.isfinite(series[999:]).all()
    assert np.nanmean(series) == pytest.approx(vpin_all, rel=0.02)

    alpha, delta, eps, n, buy, sell = mixture(9000.0, 5)        # "sufficiently large mu": the bias washes out
    mu, big = 9000.0, np.abs(buy - sell)
    pin, floor = alpha * mu / (alpha * mu + 2 * eps), (1 - alpha) * math.sqrt(4 * eps / math.pi)
    assert abs(big.mean() - (alpha * mu + floor)) < 4 * big.std(ddof=1) / math.sqrt(n)
    assert floor / (alpha * mu) < 0.01                          # ten times the mu, a tenth of the relative bias
    assert (alpha * mu + floor) / (alpha * mu + 2 * eps) == pytest.approx(pin, rel=0.01)
    assert np.nanmean(ms.vpin(buy, sell, n=1000)) == pytest.approx(pin, rel=0.03)
