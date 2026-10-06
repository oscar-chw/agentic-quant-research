import numpy as np
import pytest

from pmlab.afml import synthetic as sy


@pytest.mark.case
def test_half_life_and_phi_are_inverse():
    assert sy.phi_from_half_life(5) == pytest.approx(0.5 ** 0.2)
    assert sy.half_life(sy.phi_from_half_life(25)) == pytest.approx(25)
    assert sy.half_life(1.0) == np.inf


@pytest.mark.eval
def test_ou_fits_recover_simulated_parameters():
    """Thresholds: phi within 4 standard errors, sigma within 2%, the constant mean within 0.1 (n = 20000)."""
    rng = np.random.default_rng(0)
    phi, sigma, mean, n = 0.9, 0.5, 2.0, 20000
    x = np.empty(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = (1 - phi) * mean + phi * x[t - 1] + sigma * rng.standard_normal()
    a = sy.ou_fit(x[:-1], x[1:], forecast=mean)
    assert abs(a["phi"] - phi) < 4 * a["phi_se"] and a["sigma"] == pytest.approx(sigma, rel=0.02)
    b = sy.ou_fit_constant(x[:-1], x[1:])
    assert abs(b["phi"] - phi) < 4 * b["phi_se"] and abs(b["mean"] - mean) < 0.1
    # the wrong forecast biases the through-origin fit towards 1
    assert sy.ou_fit(x[:-1], x[1:], forecast=0.0)["phi"] > phi + 0.05


@pytest.mark.property
def test_vectorised_mesh_matches_the_path_by_path_rule():
    rng = np.random.default_rng(1)
    eps = rng.standard_normal((400, 21))
    for forecast, phi in ((3.0, 0.8), (0.0, 1.0), (-2.0, 0.95)):
        gains = sy.simulate_gains(forecast, phi, 1.3, eps=eps)
        pts, sls = [0.0, 0.7, 2.5, 50.0], [0.0, 1.1, 3.0, 50.0]
        res = sy.otr_from_gains(gains, pts, sls)
        for i, pt in enumerate(pts):
            for j, sl in enumerate(sls):
                loop = sy.otr_mesh_loop(eps, forecast, phi, 1.3, pt, sl, max_hp=20)
                assert res["mean"][i, j] == pytest.approx(loop.mean(), abs=1e-12)
                assert res["std"][i, j] == pytest.approx(loop.std(ddof=1), abs=1e-12)


@pytest.mark.eval
def test_random_walk_has_no_optimal_rule():
    """Threshold: under phi = 1 every node's mean closing gain is within 4.5 standard errors of 0
    (optional stopping of a martingale with a bounded holding period)."""
    r = sy.otr_mesh(0.0, 1.0, n_iter=20000, rng=2)
    t = r["mean"] / (r["std"] / np.sqrt(20000))
    assert np.nanmax(np.abs(t)) < 4.5


@pytest.mark.eval
def test_mean_reversion_regimes_of_the_book():
    """Thresholds (20000 paths, half-life 5): with the forecast at the seed the best node takes profit
    at <= 3 sigma with a stop >= 6 sigma and Sharpe > 1, and the mirrored rule has the mirrored Sharpe
    sign; forecast +5 gives Sharpe > 1 at the widest barriers, forecast -5 gives no positive node."""
    phi = sy.phi_from_half_life(5)
    zero = sy.otr_mesh(0.0, phi, n_iter=20000, rng=3)
    assert zero["best"]["pt"] <= 3 and zero["best"]["sl"] >= 6 and zero["best"]["sharpe"] > 1
    assert zero["sharpe"][0, 20] > 0.5 > -0.5 > zero["sharpe"][20, 0]
    up = sy.otr_mesh(5.0, phi, n_iter=20000, rng=3)
    assert up["sharpe"][20, 20] > 1
    down = sy.otr_mesh(-5.0, phi, n_iter=20000, rng=3)
    assert np.nanmax(down["sharpe"]) < 0 and down["sharpe"][20, 20] < -1


@pytest.mark.case
def test_ou_fit_hand_case():
    # forecast E = 1: x = prev - E = [1, 2, -1], y = next - E = [2, 2, 0], phi = x.y / x.x = 6 / 6
    prev, nxt = np.array([2.0, 3.0, 0.0]), np.array([3.0, 3.0, 1.0])
    f = sy.ou_fit(prev, nxt, forecast=1.0)
    x, y = prev - 1, nxt - 1
    phi = (x @ y) / (x @ x)
    assert f["phi"] == pytest.approx(phi) and phi == pytest.approx(6 / 6)
    f = sy.ou_fit(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert f["phi"] == pytest.approx(6 / 5)                  # x.y = 6, x.x = 5 (y.y = 8 would give 0.75)
    resid = np.array([2.0, 2.0]) - 1.2 * np.array([1.0, 2.0])
    assert f["sigma"] == pytest.approx(np.sqrt(resid @ resid / 1))


@pytest.mark.eval
def test_positive_equilibrium_optima_follow_the_books_half_life_progression():
    """13.6.2's reported optima (docs/afml_sections/ch13.md F13.2), forecast 5, sigma 1, 20,000 paths, max holding
    100: half-life 5 peaks at profit-taking near 6 with a wide stop-loss range and a Sharpe near 12; half-life 10
    peaks near profit-taking 5 with stops 7-10 and a Sharpe near 9; and the peak falls to about 2.7, 0.8 and 0.3 as
    the half-life grows to 25, 50 and 100. Pinning only the sign and one Sharpe would let a change move the optimum."""
    def mesh(tau):
        return sy.otr_mesh(5.0, sy.phi_from_half_life(tau), n_iter=20000, rng=3)

    five = mesh(5)
    assert five["best"]["pt"] == pytest.approx(6.0, abs=0.5) and five["best"]["sharpe"] == pytest.approx(12.0, abs=1.5)
    wide = [five["sl"][j] for i, j in np.argwhere(five["sharpe"] > 0.98 * np.nanmax(five["sharpe"]))]
    assert min(wide) <= 6.0 and max(wide) >= 9.0                  # the book's wide stop-loss range at tau = 5

    ten = mesh(10)
    assert ten["best"]["pt"] == pytest.approx(5.0, abs=0.5) and ten["best"]["sharpe"] == pytest.approx(9.0, abs=1.0)
    stops = [ten["sl"][j] for i, j in np.argwhere(ten["sharpe"] > 0.98 * np.nanmax(ten["sharpe"]))]
    assert min(stops) >= 7.0                                       # narrower, higher stops than at tau = 5

    peaks = [float(np.nanmax(mesh(t)["sharpe"])) for t in (25, 50, 100)]
    assert peaks == pytest.approx([2.7, 0.8, 0.32], abs=0.4)
    assert five["best"]["sharpe"] > ten["best"]["sharpe"] > peaks[0] > peaks[1] > peaks[2]


@pytest.mark.case
def test_ou_fit_is_a_through_origin_regression_not_eq_13_7s_covariance_form():
    """docs/afml_sections/ch13.md F13.1. Eq. 13.7 writes phi = cov(Y, X) / cov(X, X); the covariance operator
    subtracts both sample means, which is the regression WITH an intercept. ou_fit takes the forecast at face value
    and regresses through the origin. The two agree on a long path around its equilibrium and differ on a short one
    entered away from it, which is the case the module is used for."""
    rng = np.random.default_rng(0)
    phi, e, sigma = 0.8, 5.0, 1.0

    def path(n, start):
        p = np.empty(n)
        p[0] = start
        for t in range(1, n):
            p[t] = (1 - phi) * e + phi * p[t - 1] + sigma * rng.normal()
        return p

    def eq_13_7(prev, nxt, forecast):
        x, y = prev - forecast, nxt - forecast
        return float(np.cov(y, x, ddof=1)[0, 1] / np.cov(x, x, ddof=1)[0, 1])

    long_p = path(20000, e)
    prev, nxt = long_p[:-1], long_p[1:]
    assert sy.ou_fit(prev, nxt, forecast=e)["phi"] == pytest.approx(eq_13_7(prev, nxt, e), abs=0.01)
    assert sy.ou_fit(prev, nxt, forecast=e)["phi"] == pytest.approx(phi, abs=0.02)

    # 300 short paths (20 steps) entered well above equilibrium, as a position is
    through, with_intercept = [], []
    for _ in range(300):
        short = path(20, e + 8.0)
        through.append(sy.ou_fit(short[:-1], short[1:], forecast=e)["phi"])
        with_intercept.append(eq_13_7(short[:-1], short[1:], e))
    through, with_intercept = np.array(through), np.array(with_intercept)
    assert np.mean(np.abs(through - with_intercept)) > 0.05      # not the same estimator on this sample
    assert abs(through.mean() - phi) < abs(with_intercept.mean() - phi)   # the through-origin one is less biased
    assert through.mean() == pytest.approx(phi, abs=0.03)


@pytest.mark.eval
def test_zero_equilibrium_heat_maps_are_the_market_makers_rule_and_flatten_towards_the_random_walk():
    """13.6.1's reported shape, forecast 0, sigma 1, 20,000 paths, max holding 100. Thresholds: at half-life 5 the
    best rule takes a small profit (<= 2 sigma) with a wide stop (>= 8 sigma) and its Sharpe is within 0.8 of the
    book's 3.2 (3.77 here: the book draws 100,000 fresh paths per node and divides by n, this uses 20,000 common
    paths and ddof 1); the diagonal, where profit-taking and stop-loss are symmetric, is within 0.05 of neutral; the
    worst rule is the mirror of the best, the whole mesh being antisymmetric to within 5% of its peak; and the peak
    falls monotonically as the half-life grows to 10, 25, 50 and 100, reaching under 0.2 as phi approaches 1."""
    def mesh(tau):
        return sy.otr_mesh(0.0, sy.phi_from_half_life(tau), n_iter=20000, rng=3)

    five = mesh(5)
    s = five["sharpe"]
    assert five["best"]["pt"] <= 2.0 and five["best"]["sl"] >= 8.0
    assert five["best"]["sharpe"] == pytest.approx(3.2, abs=0.8)

    diagonal = np.array([s[i, i] for i in range(1, len(five["pt"]))])      # symmetric barriers: neutral
    assert np.nanmax(np.abs(diagonal)) < 0.05
    assert np.nanmax(np.abs(s[1:, 1:] + s[1:, 1:].T)) < 0.05 * np.nanmax(s)   # the worst rule mirrors the best
    worst = np.unravel_index(np.nanargmin(s), s.shape)
    assert five["pt"][worst[0]] >= 8.0 and five["sl"][worst[1]] <= 2.0        # large target, short stop

    peaks = [float(np.nanmax(mesh(t)["sharpe"])) for t in (5, 10, 25, 50, 100)]
    assert peaks == sorted(peaks, reverse=True)
    assert peaks[1] == pytest.approx(2.1, abs=0.5) and peaks[-1] < 0.2


@pytest.mark.eval
def test_negative_equilibrium_is_the_rotated_negative_of_the_positive_case():
    """13.6.3: with the forecast below the entry every rule loses, and the heat-map is the positive case's rotated
    photographic negative. Thresholds (20,000 paths, half-life 5 and 10): no node is positive; the worst region sits
    at a stop-loss of 6 sigma with profit-taking from 4 to 10 sigma and a Sharpe within 1.5 of the book's -12, and
    -9.8 (book -9) at half-life 10; and sharpe[-5][pt, sl] equals -sharpe[+5][sl, pt] to within 10% of the peak."""
    for tau, want in ((5, -12.0), (10, -9.0)):
        phi = sy.phi_from_half_life(tau)
        down = sy.otr_mesh(-5.0, phi, n_iter=20000, rng=3)
        up = sy.otr_mesh(5.0, phi, n_iter=20000, rng=3)
        s = down["sharpe"]
        assert np.nanmax(s) < 0                                            # no rule makes money
        assert np.nanmin(s) == pytest.approx(want, abs=1.5)
        i, j = np.unravel_index(np.nanargmin(s), s.shape)
        assert 4.0 <= down["pt"][i] <= 10.0 and down["sl"][j] == pytest.approx(6.0, abs=1.5)
        assert np.nanmax(np.abs(s[1:, 1:] + up["sharpe"][1:, 1:].T)) < 0.10 * abs(np.nanmin(s))


@pytest.mark.case
def test_the_mesh_is_the_books_half_sigma_grid_and_the_sharpe_ignores_the_position_size():
    """13.5.1 step 2, Table 13.1 and 13.6: the mesh is the Cartesian product of profit-taking and stop-loss in
    half-sigma steps out to 10 sigma (the book's 20 x 20 plus the zero row and column this implementation adds),
    the holding cap is the table's 100, and performance is computed per unit held -- scaling the forecast, sigma
    and therefore the barriers together leaves every Sharpe ratio alone, which is why the book fixes m = 1 and
    sigma = 1 without loss of generality."""
    r = sy.otr_mesh(5.0, sy.phi_from_half_life(5), n_iter=2000, rng=0)
    assert r["pt"] == pytest.approx([0.5 * k for k in range(21)]) and r["sl"] == pytest.approx(r["pt"])
    assert r["sharpe"].shape == (21, 21)
    assert r["holding"].max() <= 101                       # maxHP = 100 steps, the closing observation included
    explicit = sy.otr_mesh(5.0, sy.phi_from_half_life(5), n_iter=2000, max_hp=100, rng=0)
    assert r["sharpe"][1:, 1:] == pytest.approx(explicit["sharpe"][1:, 1:])           # 100 is the default

    # the optimum in the book's own regimes is a node of its 20 x 20 grid, not the zero row or column added here
    for forecast in (0.0, 5.0):
        best = sy.otr_mesh(forecast, sy.phi_from_half_life(5), n_iter=20000, rng=3)["best"]
        assert best["pt"] >= 0.5 and best["sl"] >= 0.5, forecast

    # m_i = 1: doubling the scale of the process doubles every profit and every barrier, so the ratio is untouched
    one = sy.otr_mesh(5.0, 0.87, sigma=1.0, n_iter=4000, rng=1)
    two = sy.otr_mesh(10.0, 0.87, sigma=2.0, n_iter=4000, rng=1)
    assert two["mean"] == pytest.approx(2.0 * one["mean"], rel=1e-12)
    assert two["sharpe"][1:, 1:] == pytest.approx(one["sharpe"][1:, 1:], rel=1e-12)


@pytest.mark.eval
def test_the_ou_moments_of_equation_13_4_hold_at_every_step():
    """Eq. 13.4: from the seed P_0 = 0 the process is Gaussian with mean (1 - phi) E sum_{j<t} phi^j - P_0 and
    variance sigma^2 sum_{j<t} phi^(2j), which close to E (1 - phi^t) and sigma^2 (1 - phi^(2t)) / (1 - phi^2).
    Thresholds (40,000 paths, phi = 0.9, sigma = 1.3, E = 4): the mean is within 4 standard errors of the closed
    form and the variance within 3% of it at every step, the stationary limit sigma / sqrt(1 - phi^2) is reached by
    step 20 to within 3%, and the share of paths inside +-1.96 standard deviations is 95% +- 1 point (Gaussian)."""
    phi, sigma, e, n = 0.9, 1.3, 4.0, 40_000
    g = sy.simulate_gains(e, phi, sigma, n_iter=n, max_hp=39, rng=7)
    for h in range(g.shape[1]):
        t = h + 1
        mean = e * (1 - phi ** t)
        var = sigma ** 2 * (1 - phi ** (2 * t)) / (1 - phi ** 2)
        col = g[:, h]
        assert abs(col.mean() - mean) < 4 * np.sqrt(var / n), t
        assert col.var(ddof=1) == pytest.approx(var, rel=0.03), t
    last = g[:, -1]
    assert last.std(ddof=1) == pytest.approx(sigma / np.sqrt(1 - phi ** 2), rel=0.03)      # the stationary limit
    inside = np.abs(last - last.mean()) < 1.96 * last.std(ddof=1)
    assert inside.mean() == pytest.approx(0.95, abs=0.01)


@pytest.mark.eval
def test_the_longer_half_lives_and_the_forecast_of_ten_repeat_the_books_progression():
    """13.6.2's Figures 13.8-13.15 and 13.6.3's Figures 13.18-13.25, which the shorter half-lives alone do not
    reach. Thresholds (20,000 paths, sigma 1, max holding 100): at forecast 5 and half-life 25 the optimum is the
    book's small target with a wide stop (profit-taking 3 +- 0.5, stop-loss at least 9); at forecast 10 the peak
    falls over half-lives 5, 10, 25, 50, 100 and stands above the forecast-5 peak at every one of them, the same
    pattern at a larger scale; and the negative cases mirror them -- no node of the forecast -10 mesh is positive,
    it is minus the transpose of the forecast +10 mesh to within 10% of its peak, and at forecast -5 the worst
    Sharpe flattens from -2.5 towards zero as the half-life grows to 25, 50 and 100."""
    def mesh(forecast, tau):
        return sy.otr_mesh(forecast, sy.phi_from_half_life(tau), n_iter=20000, rng=3)

    quarter = mesh(5.0, 25)                                        # Figure 13.8's reported region
    assert quarter["best"]["pt"] == pytest.approx(3.0, abs=0.5) and quarter["best"]["sl"] >= 9.0

    up5 = [float(np.nanmax(mesh(5.0, t)["sharpe"])) for t in (5, 10, 25, 50, 100)]
    up10 = [float(np.nanmax(mesh(10.0, t)["sharpe"])) for t in (5, 10, 25, 50, 100)]
    assert up10 == sorted(up10, reverse=True)                      # the same progression at a forecast of ten
    assert all(b > a for a, b in zip(up5, up10))                   # at a larger scale, as the figures show

    down10, top10 = mesh(-10.0, 5), mesh(10.0, 5)
    assert np.nanmax(down10["sharpe"]) < 0                         # a rotated photographic negative
    assert np.nanmax(np.abs(down10["sharpe"][1:, 1:] + top10["sharpe"][1:, 1:].T)) < 0.10 * abs(
        np.nanmin(down10["sharpe"]))

    worst = [float(np.nanmin(mesh(-5.0, t)["sharpe"])) for t in (25, 50, 100)]
    assert worst == sorted(worst) and worst[0] == pytest.approx(-2.5, abs=0.5) and worst[-1] > -1.0
    assert all(np.nanmax(mesh(-5.0, t)["sharpe"]) < 0 for t in (25, 50, 100))
