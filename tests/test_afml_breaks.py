import math

import numpy as np
import pytest
from scipy.stats import kstest

from pmlab.afml import breaks as br
from pmlab.afml import synthetic
from pmlab.afml.fracdiff import adf


@pytest.mark.case
def test_csw_hand_case():
    y = np.array([0.0, 1.0, 1.0, 3.0])
    r = br.csw_cusum(y)
    sd = math.sqrt(5 / 3)                               # mean squared difference (1 + 0 + 4) / 3
    s = [(3 - 0) / (sd * math.sqrt(3)), (3 - 1) / (sd * math.sqrt(2)), (3 - 1) / sd]
    assert r["s_first"][3] == pytest.approx(s[0])
    assert r["s_sup"][3] == pytest.approx(max(s)) and r["n_sup"][3] == 2
    crit = [math.sqrt(4.6 + math.log(g)) for g in (3, 2, 1)]
    assert r["ratio_sup"][3] == pytest.approx(max(a / c for a, c in zip(s, crit)))
    assert not r["flag_first"][3]
    assert br.csw_cusum(-y, two_sided=True)["s_sup"][3] == pytest.approx(max(s))


@pytest.mark.eval
def test_csw_size_and_power():
    """Thresholds (300 walks of 300 steps): the n = 1 test flags <= 6% of random walks and >= 70% of walks
    whose drift turns to +0.5 sd halfway; the book's pointwise value flags > 90% of walks for the
    backward-shifting supremum, while the simulated 95% quantile flags 1-10%."""
    rng = np.random.default_rng(0)
    crit = br.csw_sup_critical(300, 300, rng=1)["q95"]
    first = sup_book = sup_mc = brk = 0
    for _ in range(300):
        c = br.csw_cusum(np.cumsum(rng.standard_normal(300)))
        first += c["flag_first"].any()
        sup_book += c["flag_sup"].any()
        sup_mc += c["max_ratio"] > crit
        e = rng.standard_normal(300)
        e[150:] += 0.5
        brk += br.csw_cusum(np.cumsum(e))["flag_first"].any()
    assert first / 300 <= 0.06 and brk / 300 >= 0.7
    assert sup_book / 300 > 0.9
    assert 0.01 <= sup_mc / 300 <= 0.10


def _naive_bsadf(y, min_len, lags):
    dy = np.diff(y)
    m = len(dy) - lags
    X = np.column_stack([y[lags:-1]] + [dy[lags - i:len(dy) - i] for i in range(1, lags + 1)] + [np.ones(m)])
    t = dy[lags:]
    out = np.full(m, np.nan)
    for e in range(min_len - 1, m):
        best = -np.inf
        for a in range(0, e - min_len + 2):
            Xa, ta = X[a:e + 1], t[a:e + 1]
            beta, *_ = np.linalg.lstsq(Xa, ta, rcond=None)
            res = ta - Xa @ beta
            s2 = res @ res / (len(ta) - X.shape[1])
            best = max(best, beta[0] / math.sqrt(s2 * np.linalg.inv(Xa.T @ Xa)[0, 0]))
        out[e] = best
    return out


@pytest.mark.property
def test_bsadf_matches_window_by_window_regressions():
    rng = np.random.default_rng(2)
    for lags in (0, 1, 2):
        y = 100 + np.cumsum(rng.standard_normal(70)) * 0.01
        got = br.bsadf(y, min_len=12, lags=lags)
        want = _naive_bsadf(y, 12, lags)
        ok = np.isfinite(want)
        assert np.array_equal(ok, np.isfinite(got["bsadf"]))
        assert got["bsadf"][ok] == pytest.approx(want[ok], rel=1e-6)
        assert got["sadf"] == pytest.approx(np.nanmax(want), rel=1e-6)


@pytest.mark.case
def test_single_window_sadf_is_the_adf_statistic():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(150))
    for lags in (0, 1, 3):
        rows = len(y) - 1 - lags
        assert br.bsadf(y, min_len=rows, lags=lags)["sadf"] == pytest.approx(adf(y, lags)["stat"], rel=1e-8)


@pytest.mark.eval
def test_sadf_detects_a_bubble_and_holds_size_on_random_walks():
    """Thresholds: 95% critical value from 300 walks of 200 steps; >= 90% of walks ending in a 60-step
    bubble (growth 8% per step on a +2 deviation) exceed it, and 1-11% of fresh random walks do."""
    rng = np.random.default_rng(4)
    crit = br.sadf_critical(200, 20, 0, n_sim=300, rng=5)["q95"]
    det = false = 0
    for _ in range(100):
        e = rng.standard_normal(200)
        x = list(np.cumsum(e[:140]))
        dev = 2.0
        for j in range(60):
            dev = 1.08 * dev + e[140 + j]
            x.append(x[139] + dev)
        det += br.bsadf(np.array(x), 20, 0)["sadf"] > crit
        false += br.bsadf(np.cumsum(rng.standard_normal(200)), 20, 0)["sadf"] > crit
    assert det / 100 >= 0.9
    assert 0.01 <= false / 100 <= 0.11


@pytest.mark.case
def test_daily_flags_shape():
    rng = np.random.default_rng(6)
    days = {"a": np.cumsum(rng.standard_normal(100)), "b": np.cumsum(rng.standard_normal(100) + 0.8)}
    out = br.daily_flags(days, min_len=15, n_sim=60, rng=7)
    assert set(out) == {"a", "b"} and out["a"]["sadf_crit95"] == out["b"]["sadf_crit95"]
    assert out["b"]["csw_first_flag"] and out["b"]["csw_sup_flag"]
    assert out["a"]["sadf_wild_crit95"] != out["b"]["sadf_wild_crit95"]          # each day gets its own null


@pytest.mark.eval
def test_wild_bootstrap_keeps_size_under_changing_variance_and_still_finds_bubbles():
    """Thresholds (T = 100, min window 15): on 40 random walks whose volatility quadruples halfway, the
    Gaussian random-walk 95% value flags > 25% and the wild bootstrap (79 draws) < 15%; on 20 walks ending in
    a 40-step, 8%-per-step bubble the wild bootstrap flags >= 70%."""
    rng = np.random.default_rng(8)
    gauss = br.sadf_critical(100, 15, 0, n_sim=300, rng=9)["q95"]
    stat = lambda z: br.bsadf(z, 15)["sadf"]
    fg = fw = 0
    for _ in range(40):
        e = rng.standard_normal(99)
        e[50:] *= 4
        y = np.concatenate([[0.0], np.cumsum(e)])
        s = stat(y)
        fg += s > gauss
        fw += s > br.wild_bootstrap_critical(y, stat, 79, rng=rng)["q95"]
    assert fg / 40 > 0.25 and fw / 40 < 0.15
    det = 0
    for _ in range(20):
        e = rng.standard_normal(100)
        x = list(np.cumsum(e[:60]))
        dev = 2.0
        for j in range(40):
            dev = 1.08 * dev + e[60 + j]
            x.append(x[59] + dev)
        y = np.array(x)
        det += stat(y) > br.wild_bootstrap_critical(y, stat, 79, rng=rng)["q95"]
    assert det / 20 >= 0.7


@pytest.mark.case
def test_bsadf_skips_flat_stretches_instead_of_failing():
    y = np.concatenate([np.zeros(30), np.cumsum(np.random.default_rng(10).standard_normal(40))])
    r = br.bsadf(y, min_len=10)
    assert np.isfinite(r["sadf"])
    assert np.isnan(r["bsadf"][:9]).all() and r["bsadf"][9] == -np.inf       # windows inside the flat start


# ---------------------------------------------------------------- 17.3.1, 17.4.1, 17.4.2(4)-5, 17.4.3

@pytest.mark.case
def test_brown_durbin_evans_recursive_residuals_hand_case():
    # constant model: beta_(t-1) is the mean of the first t - 1 points, f_t = 1 + 1 / (t - 1)
    y = np.array([1.0, 2.0, 3.0, 5.0])
    r = br.brown_durbin_evans(y)
    w = [(2 - 1) / math.sqrt(2), (3 - 1.5) / math.sqrt(1.5), (5 - 2) / math.sqrt(4 / 3)]
    assert r["residuals"][1:].tolist() == pytest.approx(w) and np.isnan(r["residuals"][0])
    s = np.cumsum(w) / np.std(w)
    assert r["cusum"][1:].tolist() == pytest.approx(s.tolist())
    bound = 0.948 * (math.sqrt(3) + 2 * np.arange(1, 4) / math.sqrt(3))
    assert r["bound"][1:].tolist() == pytest.approx(bound.tolist())
    assert r["max_ratio"] == pytest.approx(np.max(np.abs(s) / bound))


@pytest.mark.property
def test_brown_durbin_evans_with_regressors_matches_refitting():
    rng = np.random.default_rng(11)
    X = np.column_stack([np.ones(40), rng.standard_normal(40)])
    y = X @ [0.3, 1.2] + rng.standard_normal(40)
    r = br.brown_durbin_evans(y, X)
    for t in range(2, 40):
        beta = np.linalg.lstsq(X[:t], y[:t], rcond=None)[0]
        f = 1 + X[t] @ np.linalg.inv(X[:t].T @ X[:t]) @ X[t]
        assert r["residuals"][t] == pytest.approx((y[t] - X[t] @ beta) / math.sqrt(f), rel=1e-8)


@pytest.mark.property
def test_chow_dfc_matches_a_regression_at_every_break_date():
    rng = np.random.default_rng(12)
    y = 3 + np.cumsum(rng.standard_normal(60))
    r = br.chow_dfc(y, tau0=0.2)
    z = y - y[0]
    dy, lag = np.diff(z), z[:-1]
    for j, tau in enumerate(r["breaks"]):
        x = np.where(np.arange(len(dy)) >= tau, lag, 0.0)
        d = (x @ dy) / (x @ x)
        s2 = np.sum((dy - d * x) ** 2) / (len(dy) - 1)
        assert r["dfc"][j] == pytest.approx(d / math.sqrt(s2 / (x @ x)), rel=1e-9)
    assert r["sdfc"] == pytest.approx(np.max(r["dfc"]))
    assert r["breaks"][0] == math.floor(0.2 * len(dy)) and r["breaks"][-1] == math.ceil(0.8 * len(dy)) - 1


@pytest.mark.property
def test_quantile_and_conditional_adf_bound_sadf():
    rng = np.random.default_rng(13)
    y = np.cumsum(rng.standard_normal(80))
    s = br.bsadf(y, min_len=15)
    q1 = br.adf_quantiles(y, min_len=15, q=1.0, v=0.0)
    ok = np.isfinite(s["bsadf"])
    assert q1["q"][ok] == pytest.approx(s["bsadf"][ok])                   # SADF_t = Q_(t, 1)
    r = br.adf_quantiles(y, min_len=15, q=0.9, v=0.05)
    fin = np.isfinite(r["c"])
    assert np.all(r["q"][fin] <= s["bsadf"][fin] + 1e-12) and np.all(r["c"][fin] <= s["bsadf"][fin] + 1e-12)
    assert np.all(r["c"][fin] >= r["q"][fin] - 1e-12) and np.all(r["qdot"][fin] >= 0) and np.all(r["cdot"][fin] >= 0)
    # brute force at the last end point
    e = len(r["q"]) - 1
    stats_ = []
    dy = np.diff((y - y[0]) / np.std(np.diff(y)))
    lvl = ((y - y[0]) / np.std(np.diff(y)))[:-1]
    for a in range(0, e - 15 + 2):
        X = np.column_stack([lvl[a:e + 1], np.ones(e + 1 - a)])
        beta, *_ = np.linalg.lstsq(X, dy[a:e + 1], rcond=None)
        res = dy[a:e + 1] - X @ beta
        stats_.append(beta[0] / math.sqrt(res @ res / (len(res) - 2) * np.linalg.inv(X.T @ X)[0, 0]))
    stats_ = np.array(stats_)
    qv = np.quantile(stats_, 0.9)
    tail = stats_[stats_ >= qv]
    assert r["q"][e] == pytest.approx(qv) and r["c"][e] == pytest.approx(tail.mean())
    assert r["qdot"][e] == pytest.approx(np.quantile(stats_, 0.95) - np.quantile(stats_, 0.85))
    assert r["cdot"][e] == pytest.approx(tail.std())
    assert r["qsadf"] == pytest.approx(np.nanmax(r["q"])) and r["csadf"] == pytest.approx(np.nanmax(r["c"]))
    assert r["sadf"] == pytest.approx(s["sadf"])


def _naive_smt(y, spec, min_len, phi):
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    for e in range(min_len - 1, len(y)):
        best = -np.inf
        for a in range(0, e - min_len + 2):
            n = e + 1 - a
            t = np.arange(1, n + 1, dtype=float)
            target = y[a:e + 1] if spec == "poly1" else np.log(y[a:e + 1])
            X = {"poly1": np.column_stack([np.ones(n), t, t * t]), "poly2": np.column_stack([np.ones(n), t, t * t]),
                 "exp": np.column_stack([np.ones(n), t]), "power": np.column_stack([np.ones(n), np.log(t)])}[spec]
            beta, *_ = np.linalg.lstsq(X, target, rcond=None)
            res = target - X @ beta
            se = math.sqrt(res @ res / (n - X.shape[1]) * np.linalg.inv(X.T @ X)[-1, -1])
            best = max(best, abs(beta[-1]) / (se * n ** phi))
        out[e] = best
    return out


@pytest.mark.property
def test_smt_matches_window_by_window_regressions():
    rng = np.random.default_rng(14)
    y = np.exp(0.05 * np.cumsum(rng.standard_normal(45)))
    for spec in ("poly1", "poly2", "exp", "power"):
        for phi in (0.0, 0.5):
            got = br.smt(y, spec=spec, min_len=10, phi=phi)
            want = _naive_smt(y, spec, 10, phi)
            ok = np.isfinite(want)
            assert np.array_equal(ok, np.isfinite(got["smt"])), spec
            assert got["smt"][ok] == pytest.approx(want[ok], rel=1e-6), (spec, phi)
            assert got["sup"] == pytest.approx(np.nanmax(want), rel=1e-6)
    with pytest.raises(ValueError):
        br.smt(-y, spec="exp", min_len=10)


@pytest.mark.eval
def test_brown_durbin_evans_size_and_power():
    """Thresholds (200 series of 200 i.i.d. normals): the 5% boundaries flag 1-8% (they are conservative), and
    >= 85% of series whose mean shifts by +1 sd halfway."""
    rng = np.random.default_rng(20)
    size = sum(br.brown_durbin_evans(rng.standard_normal(200))["flag"] for _ in range(200)) / 200
    shifted = []
    for _ in range(200):
        e = rng.standard_normal(200)
        e[100:] += 1.0
        shifted.append(br.brown_durbin_evans(e)["flag"])
    assert 0.01 <= size <= 0.08 and np.mean(shifted) >= 0.85


@pytest.mark.eval
def test_chow_dfc_size_and_power():
    """Thresholds: 95% value from 300 random walks of 150; 1-10% of 200 fresh walks exceed it, and >= 80% of
    walks that turn explosive (rho = 1.05) for their last 60 steps."""
    rng = np.random.default_rng(21)
    crit = br.chow_dfc_critical(150, 0.15, n_sim=300, rng=22)["q95"]
    size = power = 0
    for _ in range(200):
        size += br.chow_dfc(np.cumsum(rng.standard_normal(150)))["sdfc"] > crit
        e = rng.standard_normal(150)
        x = list(np.cumsum(e[:90]))
        for j in range(60):
            x.append(1.05 * x[-1] + e[90 + j])
        power += br.chow_dfc(np.array(x))["sdfc"] > crit
    assert 0.01 <= size / 200 <= 0.10 and power / 200 >= 0.8


@pytest.mark.eval
def test_qadf_and_cadf_hold_size_and_find_bubbles():
    """Thresholds: 95% values of QSADF and CSADF (q = 0.95) from 200 random walks of 120; 1-11% of 100 fresh
    walks exceed each, and >= 60% of 100 walks ending in a 40-step bubble growing 8% per step."""
    rng = np.random.default_rng(23)
    null = [br.adf_quantiles(np.cumsum(rng.standard_normal(120)), 15) for _ in range(200)]
    cq, cc = (np.quantile([r[k] for r in null], 0.95) for k in ("qsadf", "csadf"))
    size = {"q": 0, "c": 0}
    power = {"q": 0, "c": 0}
    for _ in range(100):
        r = br.adf_quantiles(np.cumsum(rng.standard_normal(120)), 15)
        size["q"] += r["qsadf"] > cq
        size["c"] += r["csadf"] > cc
        e = rng.standard_normal(120)
        x, dev = list(np.cumsum(e[:80])), 2.0
        for j in range(40):
            dev = 1.08 * dev + e[80 + j]
            x.append(x[79] + dev)
        r = br.adf_quantiles(np.array(x), 15)
        power["q"] += r["qsadf"] > cq
        power["c"] += r["csadf"] > cc
    assert all(0.01 <= v / 100 <= 0.11 for v in size.values()), size
    assert all(v / 100 >= 0.6 for v in power.values()), power


@pytest.mark.eval
def test_smt_holds_size_and_finds_explosive_trends():
    """Thresholds: 95% values from 200 random walks of 100 (geometric for exp); 1-12% of 100 fresh walks exceed
    them, and >= 80% of walks whose last 40 steps add a quadratic (poly1: 0.05 (t - 60)^2) or exponential (exp: log
    level + 0.1 per step) trend."""
    rng = np.random.default_rng(24)
    tt = np.arange(100)
    for spec, bend in (("poly1", lambda w: w + 0.05 * np.maximum(tt - 60, 0) ** 2),
                       ("exp", lambda w: np.exp(0.01 * w + 0.1 * np.maximum(tt - 60, 0)))):
        crit = br.smt_critical(100, spec, 15, 0.0, n_sim=200, rng=25)["q95"]
        size = power = 0
        for _ in range(100):
            w = np.cumsum(rng.standard_normal(100))
            size += br.smt(w if spec == "poly1" else np.exp(0.01 * w), spec, 15)["sup"] > crit
            power += br.smt(bend(w), spec, 15)["sup"] > crit
        assert 0.01 <= size / 100 <= 0.12 and power / 100 >= 0.8, (spec, size, power)


@pytest.mark.case
def test_smt_length_penalty_favours_short_strong_trends():
    rng = np.random.default_rng(26)
    tt = np.arange(120, dtype=float)
    y = 0.02 * tt ** 2 / 120 + 0.05 * rng.standard_normal(120)
    y[90:] += 0.02 * (tt[90:] - 90) ** 2                            # a sharp late acceleration
    flat = br.smt(y, "poly1", 15, phi=0.0)
    penal = br.smt(y, "poly1", 15, phi=1.0)
    e = len(y) - 1
    assert penal["start"][e] > flat["start"][e]


@pytest.mark.eval
def test_the_two_sided_first_flag_has_about_twice_the_one_sided_size():
    """17.3.2 (docs/afml_sections/ch17.md F17.2): two_sided compares |S_(1,t)| with c_alpha, which is the ONE-sided
    critical value, so flag_first rejects on either tail at roughly twice the size. Threshold: 1,000 driftless
    Gaussian random walks of 288 steps (a Taiwan day of 5-minute windows); the share of paths flagged is about 2% one
    sided and about 4% two sided, and the two-sided share is between 1.5 and 2.5 times the one-sided one."""
    rng = np.random.default_rng(0)
    one = two = 0
    n_sim = 1000
    for _ in range(n_sim):
        y = np.cumsum(rng.standard_normal(288))
        one += bool(br.csw_cusum(y, two_sided=False)["flag_first"].any())
        two += bool(br.csw_cusum(y, two_sided=True)["flag_first"].any())
    one, two = one / n_sim, two / n_sim
    assert 0.01 <= one <= 0.035 and 0.02 <= two <= 0.06
    assert 1.5 <= two / one <= 2.5
    assert "one-sided critical value" in br.__doc__


@pytest.mark.case
def test_bsadf_windows_hold_min_len_regression_rows_whatever_the_lag_count():
    """17.4.2(6) (docs/afml_sections/ch17.md F17.3): min_len counts regression rows here, so a window always fits
    min_len of them and needs min_len + lags + 1 points; the snippet's start points leave minSL - lags rows once the
    lagged differences are consumed. The two window sets agree at lags 0 and differ above it, which is why the
    statistics differ there."""
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal(120))

    def shortest_window(lags):
        r = br.bsadf(y, min_len=20, lags=lags)
        ends = np.flatnonzero(np.isfinite(r["bsadf"]))
        return int((ends - r["start"][ends] + 1).min()), r

    for lags in (0, 1, 2):
        rows, r = shortest_window(lags)
        assert rows == 20                                  # always 20 REGRESSION rows, whatever the lag count
        assert r["rows"] == len(y) - lags - 1              # each lag costs one regression row overall
    # 20 regression rows at lags = 2 span 20 + 2 + 1 original points, where the snippet would span 20
    r0, r2 = br.bsadf(y, min_len=20, lags=0), br.bsadf(y, min_len=20, lags=2)
    assert np.isfinite(r2["bsadf"]).sum() < np.isfinite(r0["bsadf"]).sum()
    assert r2["sadf"] != pytest.approx(r0["sadf"])
    assert "min_len counts REGRESSION ROWS" in br.__doc__


def _trend_t(y, i, regressor):
    X = np.column_stack([np.ones(len(i)), regressor(i)])
    logy = np.log(y)
    beta, *_ = np.linalg.lstsq(X, logy, rcond=None)
    resid = logy - X @ beta
    se = np.sqrt(resid @ resid / (len(i) - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return abs(beta[1]) / se


@pytest.mark.case
def test_only_the_power_trend_depends_on_where_the_windows_clock_starts():
    """17.4.3 (docs/afml_sections/ch17.md F17.4): smt restarts the clock at each window (i = 1..n). `exp` regresses on
    a regressor affine in i, so its t-statistic is unchanged by the shift; SM-Power regresses on log i, and log of
    local time is not log of series time plus a constant, so a late window differs from the same rows read on the
    series' own time index."""
    rng = np.random.default_rng(5)
    x = np.exp(0.01 * np.cumsum(rng.standard_normal(60)))
    k = 25
    tail = x[k:]
    i_local = np.arange(1, len(tail) + 1, dtype=float)
    i_series = np.arange(k + 1, len(x) + 1, dtype=float)

    assert _trend_t(tail, i_local, lambda i: i) == pytest.approx(_trend_t(tail, i_series, lambda i: i), rel=1e-9)
    assert _trend_t(tail, i_local, np.log) != pytest.approx(_trend_t(tail, i_series, np.log), rel=1e-3)

    smt_tail = br.smt(tail, spec="power", min_len=len(tail))["smt"][-1]
    assert smt_tail == pytest.approx(_trend_t(tail, i_local, np.log), rel=1e-6)      # smt uses the local clock
    assert smt_tail != pytest.approx(_trend_t(tail, i_series, np.log), rel=1e-3)
    assert "SM-Power is NOT" in br.__doc__


@pytest.mark.case
def test_the_flop_model_reproduces_the_books_worked_sadf_cost():
    """17.4.2(2): Table 17.1's operations summed give f(N, T) = N^3 + N^2(2T + 3) + N(4T - 1) + 2T + 2 per ADF
    estimate, and the double loop costs sum_(t=tau)^T f(N, t) + (T - tau) per SADF update. On the book's dollar-bar
    case (N = 3, T = 356631) it quotes 11,412,245 operations per estimate, 2.035 TFLOPs per update and 242 PFLOPs
    for the series; all three come out exactly, which also pins the minimum sample length behind them at tau = 1000."""
    assert br.adf_flops(3, 356631) == 11_412_245
    cost = br.sadf_flops(3, 356631, 1000)
    assert cost["per_adf"] == 11_412_245
    assert cost["update"] == 2_034_979_648_799                       # roughly 2.035 TFLOPs
    assert cost["series"] == 241_910_974_617_448_672                 # roughly 242 PFLOPs
    assert cost["tests"] == br.sadf_test_count(356631, 1000) == 63_237_237_528
    # the count is the book's closed form, (T - tau + 2)(T - tau + 1) / 2, and matches a direct count
    assert br.sadf_test_count(30, 20) == sum(t - 20 + 1 for t in range(20, 31)) == 66
    assert br.sadf_test_count(10, 20) == 0
    # and it is the number of fits bsadf actually performs over its admissible windows
    y = np.cumsum(np.random.default_rng(0).standard_normal(40))
    fits = sum(len(t) for _, _, t in list(br._adf_by_end(y, 20, 0))[1:])
    assert fits == br.sadf_test_count(len(y) - 1, 20)


@pytest.mark.case
def test_the_adf_specification_on_raw_prices_is_structurally_heteroscedastic():
    """17.4.2(1): the change of variable x = k y adds log k to a log-price level, which the regression's constant
    absorbs, so the statistic is untouched (and rescaling leaves the raw-price statistic alone too, both sides
    scaling by k: that is not the reason to prefer logs). The reason is where the level enters. If returns have
    constant variance, then dy_t = y_(t-1)(exp(r_t) - 1) has an error variance proportional to y_(t-1)^2, so the
    raw specification is heteroscedastic in the level, while dlog[y_t] = r_t is not: on logs the level conditions
    the MEAN of returns, on raw prices it would have to condition their volatility."""
    rng = np.random.default_rng(3)
    r = 0.02 * rng.standard_normal(4000)
    y = 100 * np.exp(np.cumsum(r))
    for k in (0.01, 7.5, 1000.0):
        assert br.bsadf(np.log(k * y), min_len=20)["sadf"] == pytest.approx(br.bsadf(np.log(y), min_len=20)["sadf"])

    def level_vs_squared_error(level):
        lag, d = level[:-1], np.diff(level)
        X = np.column_stack([np.ones(len(lag)), lag])
        resid = d - X @ np.linalg.lstsq(X, d, rcond=None)[0]
        return float(np.corrcoef(lag, resid ** 2)[0, 1])

    assert level_vs_squared_error(y) > 0.3                      # raw: error variance grows with the price
    assert abs(level_vs_squared_error(np.log(y))) < 0.05        # log: it does not


@pytest.mark.case
def test_the_zero_lag_specification_gives_the_books_three_states():
    """17.4.2(3): for dlog[y_t] = alpha + beta log[y_(t-1)], E[log y_t] = -alpha/beta + (1 + beta)^t (log y_0 + alpha/beta).
    beta < 0 is steady and converges to -alpha/beta with half-life -log 2 / log(1 + beta); beta = 0 is a martingale;
    beta > 0 is explosive, upwards when log y_0 > -alpha/beta (the book writes the threshold as alpha/beta) and
    downwards below it. The half-life is the chapter 13 O-U half-life at phi = 1 + beta."""
    def path(alpha, beta, ly0, n):
        ly = [ly0]
        for _ in range(n):
            ly.append(ly[-1] + alpha + beta * ly[-1])
        return np.array(ly)

    alpha, beta, ly0 = 0.4, -0.2, 5.0
    closed = -alpha / beta + (1 + beta) ** np.arange(21) * (ly0 + alpha / beta)
    assert path(alpha, beta, ly0, 20) == pytest.approx(closed)
    assert path(alpha, beta, ly0, 400)[-1] == pytest.approx(-alpha / beta)          # steady: limit -alpha/beta
    hl = synthetic.half_life(1 + beta)
    assert hl == pytest.approx(-math.log(2) / math.log(1 + beta))
    gap = closed + alpha / beta                                                     # the disequilibrium log y~
    assert np.interp(hl, np.arange(21), gap) == pytest.approx(gap[0] / 2, rel=0.02)
    assert path(alpha, 0.0, ly0, 50)[-1] == pytest.approx(ly0 + 50 * alpha)         # unit root: a drifting martingale
    assert path(0.0, 0.05, ly0, 100)[-1] == pytest.approx(ly0 * 1.05 ** 100)        # explosive: alpha = 0 leaves (1 + beta)^t
    assert path(0.0, 0.05, ly0, 100)[-1] > 100 * ly0 and path(0.0, 0.05, -ly0, 100)[-1] < -100 * ly0    # both ways


@pytest.mark.case
def test_the_conditional_adf_outlier_ratio_widens_with_the_right_tail():
    """17.4.2(5): C_(t,q) <= SADF_t by construction (Figure 17.2's lower boundary), and (SADF_t - C_(t,q)) / Cdot_(t,q)
    is the outlier measure of Figure 17.3: it grows when a single start point's statistic is pushed out while the
    body of s_t is unchanged, which is the bias of SADF that the section warns about."""
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.standard_normal(150))
    r = br.adf_quantiles(y, min_len=20, q=0.95, v=0.025)
    ok = np.isfinite(r["c"]) & np.isfinite(r["bsadf"])
    assert ok.any() and np.all(r["c"][ok] <= r["bsadf"][ok] + 1e-9)
    ratio = lambda s: (s.max() - s[s >= np.quantile(s, 0.95)].mean()) / s[s >= np.quantile(s, 0.95)].std()
    s = rng.standard_normal(400)
    base = ratio(s)
    s_out = s.copy()
    s_out[int(np.argmax(s))] += 6.0
    assert ratio(s_out) > base + 1.0


@pytest.mark.eval
def test_a_collapsing_bubble_hides_from_the_plain_adf_but_not_from_sadf():
    """17.4.2: a periodically collapsing bubble looks like a unit root or a stationary autoregression to a
    full-sample right-tail ADF, which is why SADF expands the start point backwards instead. Thresholds (100
    series of 200 steps, a 40-step bubble at 8% per step that then collapses back): the plain ADF on the whole
    sample exceeds the SADF 95% critical value in under 10% of them, SADF in at least 80%."""
    rng = np.random.default_rng(30)
    crit = br.sadf_critical(200, 20, 0, n_sim=200, rng=31)["q95"]
    plain = sup = 0
    for _ in range(100):
        e = rng.standard_normal(200)
        x = list(np.cumsum(e[:100]))
        dev = 1.5
        for j in range(40):
            dev = 1.08 * dev + e[100 + j]
            x.append(x[99] + dev)
        for j in range(60):                                      # the burst: back to where it started
            x.append(x[139] + (x[99] - x[139]) * (j + 1) / 60 + e[140 + j])
        y = np.array(x)
        plain += adf(y, 0)["stat"] > crit
        sup += br.bsadf(y, 20, 0)["sadf"] > crit
    assert plain / 100 < 0.10 and sup / 100 >= 0.8


@pytest.mark.eval
def test_the_martingale_tests_flag_a_collapse_as_readily_as_a_bubble():
    """17.4.3: SMT takes the ABSOLUTE trend coefficient because explosive collapse matters as much as explosive
    growth. Thresholds (100 walks of 100 steps, a quadratic bend of +-0.05 (t - 60)^2 over the last 40): at least
    80% of both signs exceed the 95% critical value, and the two powers are within 10 points of each other."""
    rng = np.random.default_rng(32)
    tt = np.arange(100)
    crit = br.smt_critical(100, "poly1", 15, 0.0, n_sim=200, rng=33)["q95"]
    up = down = 0
    for _ in range(100):
        w = np.cumsum(rng.standard_normal(100))
        bend = 0.05 * np.maximum(tt - 60, 0) ** 2
        up += br.smt(w + bend, "poly1", 15)["sup"] > crit
        down += br.smt(w - bend, "poly1", 15)["sup"] > crit
    assert up >= 80 and down >= 80 and abs(up - down) <= 10


@pytest.mark.eval
def test_the_csw_statistic_is_standard_normal_under_the_null():
    """17.3.2: under H0 (no forecast change) the standardised departure of the log price from the reference level is
    N(0, 1), which is what makes the section's critical value readable at all. Thresholds (1,200 random walks, the
    n = 1 statistic at the end of walks of 120 and 240 steps): the mean is within 0.08 of zero, the standard
    deviation within 5% of one, the 95th percentile within 0.15 of 1.645, and a Kolmogorov-Smirnov test against the
    standard normal is not rejected at 1%."""
    rng = np.random.default_rng(3)
    for steps in (120, 240):
        s = np.array([br.csw_cusum(np.cumsum(rng.standard_normal(steps)))["s_first"][-1] for _ in range(1200)])
        assert abs(s.mean()) < 0.08, steps
        assert s.std(ddof=1) == pytest.approx(1.0, rel=0.05), steps
        assert np.quantile(s, 0.95) == pytest.approx(1.645, abs=0.15), steps
        assert kstest(s, "norm").pvalue > 0.01, steps


@pytest.mark.eval
def test_the_sadf_series_spikes_inside_the_bubble_and_falls_back_once_it_bursts():
    """Figure 17.1: the SADF line spikes while prices behave like a bubble and returns to low levels when the bubble
    bursts -- the whole reason it is read as a per-end-point feature rather than as one number for the sample.
    Thresholds (40 series of 200 steps: a random walk, a 40-step bubble at 8% per step, then a 60-step burst back to
    where it started, against the 95% critical value of a random walk): at least 60% peak above the critical value
    inside the bubble (0.70-0.85 over seeds 30, 34, 35 and 36), every one of them is back under it over the last
    30 end points (1.0 over those seeds), at least 70% of the series have a peak twice their own level after the
    burst, and the mean peak is at least three times the mean level after it (9-15 over those seeds)."""
    rng = np.random.default_rng(34)
    crit = br.sadf_critical(200, 20, 0, n_sim=200, rng=31)["q95"]
    peaks, ends = [], []
    for _ in range(40):
        e = rng.standard_normal(200)
        x = list(np.cumsum(e[:100]))
        dev = 1.5
        for j in range(40):
            dev = 1.08 * dev + e[100 + j]
            x.append(x[99] + dev)
        for j in range(60):                                          # the burst: back to where it started
            x.append(x[139] + (x[99] - x[139]) * (j + 1) / 60 + e[140 + j])
        s = np.asarray(br.bsadf(np.array(x), 20, 0)["bsadf"], dtype=float)
        peaks.append(float(np.nanmax(s[100:141])))                   # inside the bubble
        ends.append(float(np.nanmax(s[170:])))                       # after the burst
    peaks, ends = np.array(peaks), np.array(ends)
    assert (peaks > crit).mean() >= 0.6                              # it spikes inside the bubble
    assert (ends < crit).mean() >= 0.9                               # and is back under the value after the burst
    assert (peaks > 2 * np.maximum(ends, 0.5)).mean() >= 0.7         # series by series, not only on average
    assert peaks.mean() > 3 * ends.mean()
