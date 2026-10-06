import numpy as np
import pytest

from pmlab import evaluation as ev


@pytest.mark.case
def test_splits_are_frozen_dates_not_cache_fractions():
    s = [ev.EXPLORE_END - 300, ev.EXPLORE_END, ev.CONFIRM_START - 300, ev.CONFIRM_START]
    out = ev.split(s)
    assert out == {"train": [s[0]], "used_holdout": [s[1], s[2]], "confirm": [s[3]]}
    assert ev.split(s + [ev.CONFIRM_START + 86400 * 5])["train"] == [s[0]]      # growing cache: cut unchanged


@pytest.mark.property
def test_holm_confirm_controls_false_positives_across_many_noise_legs():
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(200):
        legs = {f"leg{i}": rng.normal(0, 1, 150) for i in range(20)}          # no edge anywhere
        hits += any(v["significant"] for v in ev.holm_confirm(legs).values())
    assert hits / 200 <= 0.08                                                  # family-wise ≈ 5%
    legs = {"real": rng.normal(0.35, 1, 300), **{f"noise{i}": rng.normal(0, 1, 300) for i in range(9)}}
    assert ev.holm_confirm(legs)["real"]["significant"]


@pytest.mark.property
def test_block_bootstrap_is_wider_than_iid_under_block_dependence():
    rng = np.random.default_rng(1)
    times = np.arange(0, 86400 * 4, 300)
    shocks = np.repeat(rng.normal(0, 1, len(times) // 24 + 1), 24)[: len(times)]  # 2 h common shocks
    values = shocks + rng.normal(0, 0.3, len(times))
    lo_b, hi_b = ev.block_bootstrap_ci(times, values, block_seconds=7200)
    lo_i, hi_i = ev.block_bootstrap_ci(times, values, block_seconds=300)
    assert hi_b - lo_b > 2 * (hi_i - lo_i)


@pytest.mark.case
def test_windows_needed_is_the_psr_power_formula_with_skew_and_kurtosis():
    z = 1.6448536 + 0.8416212
    assert ev.windows_needed(0.2) == pytest.approx(1 + (1 + 0.5 * 0.04) * (z / 0.2) ** 2, rel=1e-6)
    heavy = ev.windows_needed(0.2, skew=-30, kurtosis=900)
    assert heavy == pytest.approx(1 + (1 + 6 + 899 / 4 * 0.04) * (z / 0.2) ** 2, rel=1e-6) and heavy > 10 * ev.windows_needed(0.2)
    assert ev.windows_needed(0) == np.inf and ev.windows_needed(-0.1) == np.inf


@pytest.mark.property
def test_windows_needed_gives_the_stated_power_in_simulation():
    rng = np.random.default_rng(5)
    n = int(np.ceil(ev.windows_needed(0.15)))
    hits = sum(ev.psr_pvalue(rng.normal(0.15, 1, n)) < 0.05 for _ in range(400))
    assert 0.72 <= hits / 400 <= 0.88


@pytest.mark.property
def test_selection_dsr_is_not_ruined_by_tiny_noisy_trials_when_trial_sizes_are_given():
    rng = np.random.default_rng(3)
    skilled = rng.normal(0.25, 1, 500)
    sizes = [3, 4, 5, 6] * 5 + [500] * 20
    trials = [rng.normal(0, 1, k) for k in sizes] + [skilled]
    sharpes = [t.mean() / t.std(ddof=1) for t in trials]
    ns = [len(t) for t in trials]
    old = ev.selection_dsr(skilled, sharpes)
    new = ev.selection_dsr(skilled, sharpes, trial_n=ns)
    assert old < 0.5 < 0.95 <= new
    noise = [rng.normal(0, 1, 500) for _ in range(40)]
    sh = [t.mean() / t.std(ddof=1) for t in noise]
    best = noise[int(np.argmax(sh))]
    assert ev.selection_dsr(best, sh, trial_n=[500] * 40) < 0.95        # still deflates the best of many noise trials


@pytest.mark.case
def test_selection_dsr_counts_every_trial_tried_and_takes_the_sharpe_variance_from_the_long_ones_only():
    """docs/afml_sections/ch14.md F14.6: trials with fewer than min_n returns left N as well as V, so a search with many
    short trials was deflated as if it had tried fewer. The same long trials plus short ones must deflate more."""
    rng = np.random.default_rng(146)
    selected = rng.normal(0.12, 1, 400)
    long_trials = [rng.normal(0, 1, 400) for _ in range(20)]
    short_trials = [rng.normal(0, 1, k) for k in [5, 8, 12, 20, 29] * 6]
    sharpe = lambda r: r.mean() / r.std(ddof=1)
    long_sr, long_n = [sharpe(t) for t in long_trials], [len(t) for t in long_trials]
    without = ev.selection_dsr(selected, long_sr, trial_n=long_n)
    with_short = ev.selection_dsr(selected, long_sr + [sharpe(t) for t in short_trials] + [np.nan],
                                  trial_n=long_n + [len(t) for t in short_trials] + [1])
    assert 0.05 < with_short < without < 0.999
    s = ev.metrics.summary(selected)
    z = [sr * np.sqrt(n - 1) for sr, n in zip(long_sr, long_n)]                 # V from the long trials only
    star = ev.metrics.expected_max_sharpe(20 + 30 + 1, np.var(z, ddof=1) / (s["n"] - 1))   # N: all 51 tried
    assert with_short == pytest.approx(ev.metrics.probabilistic_sharpe_ratio(s["sharpe"], s["n"], s["skew"], s["kurtosis"], star))


@pytest.mark.case
def test_verdict_requires_both_deflated_selection_and_held_out_confirmation():
    good = {"significant": True}
    bad = {"significant": False}
    assert ev.verdict(0.97, good) == "confirmed"
    assert ev.verdict(0.90, good) == "not confirmed"
    assert ev.verdict(0.97, bad) == "not confirmed"
    assert ev.verdict(0.97, None) == "untested on held-out data"


@pytest.mark.property
def test_selection_dsr_penalises_the_best_of_many_noise_trials():
    rng = np.random.default_rng(2)
    trials = [rng.normal(0, 1, 100) for _ in range(40)]
    sharpes = [t.mean() / t.std(ddof=1) for t in trials]
    best = trials[int(np.argmax(sharpes))]
    assert ev.selection_dsr(best, sharpes) < 0.95
    assert 1 - ev.psr_pvalue(best) > ev.selection_dsr(best, sharpes)          # PSR alone is fooled more


@pytest.mark.property
def test_block_bootstrap_p_is_small_for_a_real_negative_mean_and_large_for_noise():
    from pmlab.evaluation import block_bootstrap_p
    rng = np.random.default_rng(11)
    times = np.arange(2000) * 300
    assert block_bootstrap_p(times, rng.normal(-0.3, 1, 2000)) < 0.01
    assert block_bootstrap_p(times, rng.normal(0.0, 1, 2000)) > 0.05
    assert block_bootstrap_p(times, rng.normal(0.3, 1, 2000)) > 0.95


@pytest.mark.case
def test_holm_can_count_untestable_registered_legs_as_p_one():
    rng = np.random.default_rng(12)
    legs = {"a": rng.normal(0.4, 1, 200), "b": rng.normal(0.35, 1, 200), "c": np.array([0.1])}   # c has no p-value
    loose, strict = ev.holm_confirm(legs), ev.holm_confirm(legs, missing_as_one=True)
    assert np.isnan(loose["c"]["p"]) and strict["c"]["p"] == 1.0 and not strict["c"]["significant"]
    assert strict["a"]["p_holm"] > loose["a"]["p_holm"]                 # family of 3, not 2


# ----------------------------------------------------------------------------- audit R6 (research/audit_r6_findings.md)

def _slow_latent(rng, n, half_life_h, noise):
    """The audit's null series (e_block_size.py): zero mean, a latent AR(1) with this half-life, t4 noise."""
    from scipy import signal
    phi = 0.5 ** (1 / (half_life_h * 12))
    a = signal.lfilter([1.0], [1.0, -phi], rng.standard_normal(n + 2000))[2000:]
    return a * np.sqrt(1 - phi ** 2) + noise * rng.standard_t(4, n) / np.sqrt(2)


@pytest.mark.property
def test_dependence_measured_on_earlier_days_keeps_the_size_that_2h_blocks_lose():
    """Finding 1: with a latent half-life of 12 h the 2 h block bootstrap rejects a true zero mean far too often;
    the block length and variance inflation measured on 18 earlier days keep it near 5%."""
    rng = np.random.default_rng(21)
    nh, n = 18 * 288, 7 * 288
    th, t = np.arange(nh) * 300, np.arange(nh, nh + n) * 300
    new = old = 0
    reps = 150
    for _ in range(reps):
        z = _slow_latent(rng, nh + n, 12, 8.0)
        dep = ev.dependence_block(th, z[:nh])
        new += ev.dependent_mean_test(t, z[nh:], dep["hours"])["p"] < 0.05
        old += ev.block_bootstrap_p(t, z[nh:], n_boot=300, seed=int(rng.integers(1e9))) < 0.05
    assert new / reps <= 0.09 and old / reps >= 0.15


@pytest.mark.case
def test_dependence_block_is_where_the_variance_inflation_stops_growing():
    rng = np.random.default_rng(22)
    t = np.arange(20 * 288) * 300
    x = np.repeat(rng.normal(0, 1, 20 * 2), 144) + rng.normal(0, 3, len(t))        # 12 h common shocks
    curve = ev.variance_inflation(t, x)
    assert curve[0.5]["vif"] < 2.5 and curve[12]["vif"] > 8                          # theory: 1.5 and 15.3
    dep = ev.dependence_block(t, x)
    assert dep["hours"] >= 12 and dep["vif"] >= 0.95 * max(c["vif"] for c in curve.values())
    assert ev.dependence_block(t, rng.normal(0, 1, len(t)))["vif"] < 2
    x = rng.normal(-0.5, 2, 100)
    r = ev.vif_mean_test(x, vif=4.0, df=9)
    assert r["se"] == pytest.approx(np.sqrt(4 * np.var(x, ddof=1) / 100)) and r["t"] == pytest.approx(x.mean() / r["se"])
    assert r["p"] == pytest.approx(ev.student_t.cdf(r["t"], 9)) and r["ci"][0] < r["mean"] < r["ci"][1]
    assert ev.vif_mean_test(x, vif=0.2, df=9)["se"] == pytest.approx(np.sqrt(np.var(x, ddof=1) / 100))   # never below iid


@pytest.mark.case
def test_dependent_mean_test_is_the_equal_weighted_cosine_variance_with_df_from_the_measured_block_corrected_to_60h():
    """The primary test by hand: Λ_j = √(2/n)·Σ cos(πj(t − t₁ + Δ/2)/S)(x − x̄), ν = max(4, ⌊S / (8·hours)⌋),
    b = mean 1/(1 + (πj·τ/S)²) with τ = min(hours, 60 h)/ln 2, se = √(mean Λ² / b / n), Student t with ν df."""
    rng = np.random.default_rng(23)
    t = np.arange(35 * 288) * 300 + 1_789_776_000
    x = rng.normal(-0.05, 1, len(t))
    S, n = 35 * 86400, len(t)

    def by_hand(hours, nu):
        u = (t - t[0] + 150) / S
        lam = np.array([np.sqrt(2 / n) * np.sum(np.cos(np.pi * j * u) * (x - x.mean())) for j in range(1, nu + 1)])
        tau = min(hours, 60) * 3600 / np.log(2)
        b = np.mean([1 / (1 + (np.pi * j * tau / S) ** 2) for j in range(1, nu + 1)])
        se = np.sqrt(np.mean(lam ** 2) / b / n)
        return se, ev.student_t.cdf(x.mean() / se, nu), b

    for hours, nu in ((2, 52), (12, 8), (48, 4), (96, 4)):
        r = ev.dependent_mean_test(t, x, hours)
        se, p, b = by_hand(hours, nu)
        assert r["df"] == nu and r["se"] == pytest.approx(se) and r["p"] == pytest.approx(p) and r["bias"] == pytest.approx(b)
    assert ev.dependent_mean_test(t, x, 96)["bias"] == ev.dependent_mean_test(t, x, 60)["bias"] < 0.75   # capped at 60 h
    shuffled = rng.permutation(len(t))
    assert ev.dependent_mean_test(t[shuffled], x[shuffled], 12)["p"] == pytest.approx(ev.dependent_mean_test(t, x, 12)["p"])
    assert np.isnan(ev.dependent_mean_test(t, x, np.nan)["p"]) and np.isnan(ev.dependent_mean_test(t[:2], x[:2], 2)["p"])


@pytest.mark.property
def test_the_primary_test_holds_its_size_over_35_days_with_latent_dependence_from_4_to_48_hours():
    """data_eval review (evaluation 2): dependence measured on 18 earlier days, 35 analysed days, the audit's slow-latent
    null series with latent half-lives 4–48 h. Here the variance-inflation test of the earlier span (vif_mean_test, the
    primary test before) rejects a true zero mean in 9.1% of runs at 24 h and 14.4% at 48 h; dependent_mean_test stays
    within Monte Carlo error of 5% at every half-life (at most 5.6% here, 4.9% over 5,000 other runs each)."""
    rng = np.random.default_rng(3535)
    nh, n = 18 * 288, 35 * 288
    th, t = np.arange(nh) * 300, np.arange(nh, nh + n) * 300
    reps = 2000
    bound = 0.05 + 2 * np.sqrt(0.05 * 0.95 / reps)
    sizes = {}
    for half_life, noise in ((4, 6.0), (8, 6.0), (12, 8.0), (24, 8.0), (48, 8.0)):
        new = old = 0
        for _ in range(reps):
            z = _slow_latent(rng, nh + n, half_life, noise)
            dep = ev.dependence_block(th, z[:nh])
            new += ev.dependent_mean_test(t, z[nh:], dep["hours"])["p"] < 0.05
            old += ev.vif_mean_test(z[nh:], dep["vif"], dep["blocks"] - 1)["p"] < 0.05
        sizes[half_life] = (new / reps, old / reps)
    assert all(new <= bound for new, _ in sizes.values()), sizes
    assert np.mean([new for new, _ in sizes.values()]) <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / reps / len(sizes)), sizes
    assert sizes[48][1] > bound and sizes[24][1] > bound, sizes                        # the check catches the old test


def _maker_leg(rng, n, price=(0.96, 0.995)):
    p = rng.uniform(*price, n)
    shares = 50 / p
    return {"up": shares, "down": np.zeros(n), "cost": shares * p, "fees": np.zeros(n)}


@pytest.mark.case
def test_a_maker_week_without_a_loss_is_not_evidence_of_edge():
    """Finding 9/11: 100 endgame bids at 0.96–0.995 all winning happens to a zero-skill maker with probability
    Π p ≈ 7%; the PSR reads it as a huge Sharpe."""
    rng = np.random.default_rng(23)
    leg = _maker_leg(rng, 100)
    leg["pnl"] = leg["up"] - leg["cost"]                                               # every bid won
    zero_skill = float(np.prod(zero := ev.zero_edge_up_probability(leg["up"], leg["down"], leg["cost"], leg["fees"])))
    p, kind = ev.shape_pvalue(leg, n_boot=4999)
    assert kind == "outcome" and p == pytest.approx(zero_skill, abs=0.02) and p > 0.05
    assert ev.psr_pvalue(leg["pnl"] / leg["cost"]) < 0.001
    legs = {"maker": leg, **{f"noise{i}": rng.normal(0, 1, 200) for i in range(3)}}
    assert not ev.holm_confirm(legs)["maker"]["significant"]
    three = {k: v[:3] for k, v in leg.items()}                                         # recorded P&L is rounded (fills: 5 dp)
    three["pnl"] = np.round(three["pnl"], 5)
    p3, _ = ev.shape_pvalue(three, n_boot=4999)
    assert p3 == pytest.approx(float(np.prod(zero[:3])), abs=0.02)                       # a tie with the observed t counts
    lo, hi = ev.outcome_ci(three)
    assert lo < three["pnl"].sum() / three["cost"].sum() - 0.05 and hi >= np.mean(three["pnl"] / three["cost"]) - 1e-6
    assert zero.min() >= 0.96 - 1e-9


@pytest.mark.property
def test_shape_pvalues_keep_the_family_wise_rate_on_skewed_zero_edge_legs():
    """Finding 11: PSR p-values over walk-forward-like legs gave a Holm family-wise rate of 8.2%."""
    rng = np.random.default_rng(24)
    shapes = [_maker_leg(rng, 300), _maker_leg(rng, 60, (0.05, 0.2)), _maker_leg(rng, 40, (0.02, 0.1)),
              _maker_leg(rng, 150, (0.3, 0.7))]
    reps, family = 200, 0
    for _ in range(reps):
        legs = {}
        for i, s in enumerate(shapes):
            p = ev.zero_edge_up_probability(s["up"], s["down"], s["cost"], s["fees"])
            won = rng.random(len(p)) < p
            legs[f"leg{i}"] = {**s, "pnl": np.where(won, s["up"], s["down"]) - s["cost"] - s["fees"]}
        family += any(r["significant"] for r in ev.holm_confirm(legs, n_boot=999).values())
    assert family / reps <= 0.08


@pytest.mark.property
def test_intervals_from_the_outcome_test_cover_a_zero_mean_on_small_longshot_legs():
    """Finding 11: the 2 h block percentile interval covered 71–97% on the walk-forward legs' shapes."""
    rng = np.random.default_rng(25)
    shape = _maker_leg(rng, 60, (0.05, 0.2))
    p = ev.zero_edge_up_probability(shape["up"], shape["down"], shape["cost"], shape["fees"])
    times = np.arange(60) * 900
    cover = old = 0
    for _ in range(200):
        pnl = np.where(rng.random(60) < p, shape["up"], 0.0) - shape["cost"]
        lo, hi = ev.outcome_ci({**shape, "pnl": pnl}, n_boot=1000, seed=int(rng.integers(1e9)))
        cover += lo - 1e-9 <= 0 <= hi + 1e-9
        lo, hi = ev.block_bootstrap_ci(times, pnl / shape["cost"], n_boot=500, seed=int(rng.integers(1e9)))
        old += lo <= 0 <= hi
    assert cover / 200 >= 0.91 and old / 200 < cover / 200


@pytest.mark.property
def test_luck_hurdle_uses_the_legs_own_sharpe_dispersion():
    """Finding 9: Var(SR̂) = 1/(n − 1) understates a maker leg's sampling dispersion several-fold."""
    rng = np.random.default_rng(26)
    leg = _maker_leg(rng, 400, (0.99, 0.999))
    p = ev.zero_edge_up_probability(leg["up"], leg["down"], leg["cost"], leg["fees"])
    leg["pnl"] = np.where(rng.random(400) < p, leg["up"], 0.0) - leg["cost"]
    h = ev.luck_hurdle(leg, n_trials=10_000, n_sims=4000)
    assert h["null_sd"] > 2 * h["naive_sd"] and h["sr_star"] > ev.metrics.expected_max_sharpe(10_000, 1 / 399)
    assert h["dsr"] < 0.95


@pytest.mark.property
def test_windows_needed_by_simulation_gives_the_stated_power_for_a_skewed_leg():
    """Finding 12: for a maker-like leg the analytic count gave power far below 80%."""
    rng = np.random.default_rng(27)
    leg = _maker_leg(rng, 500, (0.97, 0.99))
    p = ev.zero_edge_up_probability(leg["up"], leg["down"], leg["cost"], leg["fees"]) + 0.006    # a real edge
    leg["pnl"] = np.where(rng.random(500) < p, leg["up"], 0.0) - leg["cost"]
    n = ev.windows_needed_sim(leg, n_sims=1000)
    assert np.isfinite(n)
    hits = 0
    for _ in range(200):                                                                # future windows like these ones
        idx = rng.integers(0, 500, int(n))
        hits += ev.shape_pvalue({k: np.asarray(v)[idx] for k, v in leg.items()}, n_boot=499, seed=int(rng.integers(1e9)))[0] < 0.05
    assert 0.7 <= hits / 200 <= 0.9
    assert n > ev.windows_needed(ev.metrics.sharpe(leg["pnl"] / leg["cost"]))                    # the normal formula
