from math import comb

import numpy as np
import pytest

from pmlab.afml import cpcv


@pytest.mark.case
def test_split_and_path_counts():
    assert (cpcv.n_splits(6, 2), cpcv.n_paths(6, 2)) == (15, 5)        # the book's N = 6, k = 2 figure
    assert (cpcv.n_splits(10, 2), cpcv.n_paths(10, 2)) == (45, 9)
    assert cpcv.n_paths(5, 1) == 1 and cpcv.n_paths(8, 4) == comb(7, 3)


@pytest.mark.property
def test_paths_use_every_split_group_pair_once():
    for N, k in ((6, 2), (7, 3), (5, 1), (8, 4)):
        P = cpcv.paths(N, k)
        splits = cpcv.split_groups(N, k)
        assert P.shape == (cpcv.n_paths(N, k), N) and (P >= 0).all()
        pairs = {(int(P[j, g]), g) for j in range(P.shape[0]) for g in range(N)}
        assert pairs == {(s, g) for s, gs in enumerate(splits) for g in gs}
        for g in range(N):
            assert all(g in splits[s] for s in P[:, g])


def _spans(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t0 = np.sort(rng.integers(0, 10_000, n))
    return t0, t0 + rng.integers(1, 120, n)


@pytest.mark.property
def test_cpcv_purges_embargoes_and_tests_each_observation_phi_times():
    t0, t1 = _spans()
    N, k, emb = 6, 2, 50
    splits = cpcv.cpcv_splits(t0, t1, N, k, embargo=emb)
    gid = cpcv.group_index(t0, N)
    counts = np.zeros(len(t0), dtype=int)
    for sp in splits:
        tr, te = sp["train"], sp["test"]
        assert not set(tr) & set(te)
        counts[te] += 1
        for g in sp["groups"]:
            lo, hi = t0[gid == g].min(), t1[gid == g].max()
            assert not np.any((t0[tr] < hi) & (t1[tr] > lo))
            assert not np.any((t0[tr] >= hi) & (t0[tr] < hi + emb))
        # nothing is dropped beyond what purging and embargo require
        dropped = np.setdiff1d(np.arange(len(t0)), np.concatenate([tr, te]))
        for i in dropped:
            assert any((t0[i] < t1[gid == g].max() + emb) and (t1[i] > t0[gid == g].min()) for g in sp["groups"])
    assert (counts == cpcv.n_paths(N, k)).all()


@pytest.mark.case
def test_assemble_paths_takes_each_group_from_its_assigned_split():
    t0, _ = _spans(120)
    N, k = 6, 2
    gid = cpcv.group_index(t0, N)
    preds = []
    for s, groups in enumerate(cpcv.split_groups(N, k)):
        p = np.full(len(t0), np.nan)
        p[np.isin(gid, groups)] = s
        preds.append(p)
    out = cpcv.assemble_paths(preds, t0, N, k)
    P = cpcv.paths(N, k)
    assert np.isfinite(out).all()
    for j in range(P.shape[0]):
        assert out[j].tolist() == [P[j, g] for g in gid]


@pytest.mark.case
def test_pbo_hand_case():
    mean = lambda R: R.mean(axis=0)
    flip = np.array([[3.0, 1.0], [3.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])    # the in-sample winner always loses
    r = cpcv.pbo(flip, n_blocks=2, metric=mean)
    assert r["combinations"] == 2 and r["pbo"] == 1.0
    assert r["lambda"] == pytest.approx([np.log(0.5), np.log(0.5)])
    dominant = np.array([[3.0, 1.0], [3.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    r = cpcv.pbo(dominant, n_blocks=2, metric=mean)
    assert r["pbo"] == 0.0 and r["lambda"] == pytest.approx([np.log(2.0), np.log(2.0)])
    assert r["prob_oos_loss"] == 0.0


@pytest.mark.eval
def test_pbo_near_half_for_noise_and_near_zero_for_real_skill():
    """Thresholds: 20 iid N(0,1) strategies, T = 1000, S = 10, mean PBO over 6 seeds in [0.35, 0.65];
    with one strategy at mean 0.25 sd PBO < 0.05 and a negative degradation slope for noise."""
    noise, skill, slopes = [], [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        M = rng.normal(size=(1000, 20))
        r = cpcv.pbo(M, n_blocks=10)
        noise.append(r["pbo"])
        slopes.append(r["degradation_slope"])
        M[:, 3] += 0.25
        skill.append(cpcv.pbo(M, n_blocks=10)["pbo"])
    assert 0.35 <= np.mean(noise) <= 0.65
    assert max(skill) < 0.05
    assert np.mean(slopes) < 0


@pytest.mark.case
def test_a_half_with_no_trades_scores_zero_not_minus_infinity():
    # untraded windows are 0 returns: a strategy with no trade in a half has earned nothing there, which ranks
    # above a strategy that lost money, not below it
    flat = np.zeros(8)
    loser = np.array([-1.0, -2.0, -1.0, -3.0, -1.0, -2.0, -1.0, -3.0])
    assert cpcv.sharpe_columns(np.column_stack([flat, loser])).tolist() == pytest.approx([0.0, loser.mean() / loser.std(ddof=1)])
    # blocks 0-1 in-sample: the flat strategy is best in-sample only if 0 > the others; make it the out-of-sample check
    M = np.column_stack([np.r_[np.ones(4) * [1.0, 2.0, 1.0, 2.0], np.zeros(4)],        # wins in-sample, idle out of sample
                         np.r_[np.ones(4) * [0.1, 0.4, 0.1, 0.2], -np.array([1.0, 2.0, 1.0, 3.0])],
                         np.r_[np.ones(4) * [0.0, 0.1, 0.0, 0.1], -np.array([2.0, 1.0, 2.0, 5.0])]])
    r = cpcv.pbo(M, n_blocks=2)
    first = [i for i, c in enumerate(cpcv.combinations(range(2), 1)) if c == (0,)][0]
    assert r["winner"][first] == 0 and r["oos_best"][first] == 0.0
    assert r["lambda"][first] == pytest.approx(np.log(3 / 1))           # idle is the best of three out of sample


@pytest.mark.case
def test_paths_reproduce_the_books_assignment_for_six_groups_and_two_tested():
    """12.4.1's assignment figure (docs/afml_sections/ch12.md F12.1): with N = 6 and k = 2 the book's path 1 takes
    G1 and G2 from S1, G3 from S2, G4 from S3, G5 from S4 and G6 from S5; its path 2 takes G1 from S2, G2 and G3 from
    S6, G4 from S7, G5 from S8 and G6 from S9. Checking only that every (split, group) pair appears once leaves many
    other assignments passing, so the whole matrix is pinned here (0-based splits and groups)."""
    P = cpcv.paths(6, 2)
    assert cpcv.split_groups(6, 2)[:9] == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5)]
    assert P.tolist() == [[0, 0, 1, 2, 3, 4],
                          [1, 5, 5, 6, 7, 8],
                          [2, 6, 9, 9, 10, 11],
                          [3, 7, 10, 12, 12, 13],
                          [4, 8, 11, 13, 14, 14]]
    # every path is a complete pass over the groups, and no two paths share a (split, group) pair
    for j in range(P.shape[0]):
        for g in range(6):
            assert g in cpcv.split_groups(6, 2)[P[j, g]]
    assert len({(int(P[j, g]), g) for j in range(P.shape[0]) for g in range(6)}) == 30


@pytest.mark.case
def test_the_walk_forward_decides_on_a_quarter_of_the_sample_plus_three_quarters_of_the_warm_up():
    """12.2.1's third pitfall as a closed form, checked against enumerating the decisions: with a warm-up of t0 out
    of T, the first half of the T - t0 decisions sees (T + 3 t0) / 4 observations on average."""
    r = cpcv.walk_forward_average_sample(1000, 200)
    assert (r["decisions"], r["half_decisions"]) == (800, 400)
    assert r["average_sample"] == pytest.approx(400.0) and r["fraction"] == pytest.approx(0.4)
    assert r["average_sample"] == pytest.approx(np.arange(200, 600).mean(), abs=1.0)   # one decision per observation
    for T, t0 in ((1000, 0), (1000, 500), (400, 100), (10_000, 2_000)):
        s = cpcv.walk_forward_average_sample(T, t0)
        assert s["fraction"] == pytest.approx(0.25 + 0.75 * t0 / T)
        assert s["average_sample"] == pytest.approx(np.arange(t0, (T + t0) / 2).mean(), abs=1.0)
    # a longer warm-up buys a larger fraction at the cost of a shorter backtest
    a, b = cpcv.walk_forward_average_sample(1000, 100), cpcv.walk_forward_average_sample(1000, 400)
    assert b["fraction"] > a["fraction"] and b["decisions"] < a["decisions"]
    assert cpcv.walk_forward_average_sample(1000, 1000)["fraction"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cpcv.walk_forward_average_sample(100, 200)


def _equicorrelated(rng, phi, rho, n):
    """(phi, n) rows of unit-variance normals with average off-diagonal correlation rho."""
    common = rng.normal(size=(1, n))
    return np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(size=(phi, n))


@pytest.mark.case
def test_the_variance_of_the_path_mean_is_the_books_formula_and_lies_between_its_two_bounds():
    """12.5: sigma^2[mu] = phi^-1 sigma^2 (1 + (phi - 1) rho), equal to the double sum of the covariances, and in
    [sigma^2 / phi, sigma^2) for 0 <= rho < 1 (docs/afml_sections/ch12.md F12.2)."""
    y, rho = np.array([1.0, 2.0, 3.0, 4.0]), 0.5
    r = cpcv.path_mean_variance(y, rho)
    phi, s2 = 4, y.var(ddof=1)
    cov = np.full((phi, phi), rho * s2)
    np.fill_diagonal(cov, s2)
    assert r["variance"] == pytest.approx(s2)
    assert r["variance_of_mean"] == pytest.approx(cov.sum() / phi**2)          # phi^-2 sum_j sum_k Cov(y_j, y_k)
    assert r["variance_of_mean"] == pytest.approx(s2 * (1 + (phi - 1) * rho) / phi)
    assert r["variance_if_independent"] <= r["variance_of_mean"] < r["variance_if_identical"]
    assert r["effective_paths"] == pytest.approx(s2 / r["variance_of_mean"]) and r["effective_paths"] < phi
    assert r["sd_of_mean"] == pytest.approx(np.sqrt(r["variance_of_mean"]))
    # the bounds are attained: independent paths give sigma^2 / phi, identical paths give sigma^2
    assert cpcv.path_mean_variance(y, 0.0)["variance_of_mean"] == pytest.approx(s2 / phi)
    assert cpcv.path_mean_variance(y, 1.0)["variance_of_mean"] == pytest.approx(s2)
    assert cpcv.path_mean_variance(y, 1.0)["effective_paths"] == pytest.approx(1.0)
    rising = [cpcv.path_mean_variance(y, r_)["variance_of_mean"] for r_ in np.linspace(0, 0.99, 20)]
    assert np.all(np.diff(rising) > 0)
    assert np.isnan(cpcv.path_mean_variance([1.0], 0.5)["variance_of_mean"])   # one path has no variance


@pytest.mark.case
def test_the_average_off_diagonal_correlation_is_read_from_the_paths_own_series():
    rng = np.random.default_rng(3)
    Y = _equicorrelated(rng, 8, 0.7, 4000)
    assert cpcv.mean_correlation(Y) == pytest.approx(0.7, abs=0.05)
    assert cpcv.mean_correlation(rng.normal(size=(8, 4000))) == pytest.approx(0.0, abs=0.05)
    same = np.tile(rng.normal(size=(1, 200)), (5, 1))
    assert cpcv.mean_correlation(same) == pytest.approx(1.0)
    assert np.isnan(cpcv.mean_correlation(np.ones((4, 50))))                  # no variation: undefined, not 1
    assert np.isnan(cpcv.mean_correlation(rng.normal(size=(1, 50))))


@pytest.mark.eval
def test_the_path_mean_variance_matches_a_simulation_of_correlated_paths():
    """Threshold: phi = 10 unit-variance paths with average correlation rho, 20,000 replications; the measured
    variance of the mean is within 5% of 12.5's formula, which at rho = 0.6 is 6.4x the independent bound."""
    rng = np.random.default_rng(11)
    unit = rng.normal(size=10)
    unit /= unit.std(ddof=1)                                   # sigma^2 = 1, the paths' marginal variance
    for rho in (0.0, 0.3, 0.6):
        draws = np.sqrt(rho) * rng.normal(size=(20_000, 1)) + np.sqrt(1 - rho) * rng.normal(size=(20_000, 10))
        measured = draws.mean(axis=1).var(ddof=1)
        r = cpcv.path_mean_variance(unit, rho)
        assert r["variance"] == pytest.approx(1.0)
        assert r["variance_if_independent"] == pytest.approx(0.1)
        assert measured == pytest.approx(r["variance_of_mean"], rel=0.05)
        assert measured == pytest.approx((1 + 9 * rho) / 10, rel=0.05)
    assert r["variance_of_mean"] / r["variance_if_independent"] == pytest.approx(6.4)


@pytest.mark.eval
def test_the_spread_across_correlated_paths_understates_their_marginal_variance():
    """The caveat 12.5 does not state: sigma^2 is the marginal variance of a path statistic, but the spread
    observed across phi equicorrelated paths estimates sigma^2 (1 - rho), so feeding path_mean_variance the sample
    spread understates the variance of the mean by about that factor. Threshold: 20,000 replications, within 5%."""
    rng, phi = np.random.default_rng(5), 10
    for rho in (0.3, 0.6):
        draws = np.sqrt(rho) * rng.normal(size=(20_000, 1)) + np.sqrt(1 - rho) * rng.normal(size=(20_000, phi))
        assert draws.var(axis=1, ddof=1).mean() == pytest.approx(1 - rho, rel=0.05)
        from_spread = np.mean([cpcv.path_mean_variance(row, rho)["variance_of_mean"] for row in draws[:2000]])
        assert from_spread == pytest.approx((1 - rho) * (1 + (phi - 1) * rho) / phi, rel=0.1)
        assert from_spread < draws.mean(axis=1).var(ddof=1)     # so the reported precision is optimistic, never pessimistic


@pytest.mark.case
def test_the_cscv_combination_count_is_s_choose_half_which_the_book_prints_transposed():
    """11.6 prints 12,780 combinations for S = 16; C(16, 8) = 12,870 (docs/afml_sections/ch11.md 11.6), and that is
    what the code enumerates."""
    assert comb(16, 8) == 12870 and comb(16, 8) != 12780
    M = np.random.default_rng(0).normal(size=(32, 3))
    assert cpcv.pbo(M, n_blocks=16)["combinations"] == 12870
    assert cpcv.pbo(M, n_blocks=16, max_combinations=500, rng=0)["combinations"] == 500
    with pytest.raises(ValueError):
        cpcv.pbo(M, n_blocks=15)


# ------------------------------------------------------------------ the second reading (plan book-v2 node 16)


@pytest.mark.property
def test_the_groups_are_contiguous_and_near_equal_and_every_split_trains_on_the_books_share():
    """12.4.1's partition and 12.3's second advantage, neither of which had a test of its own. The observations are
    cut into N contiguous groups in time order with no shuffling, every observation belongs to exactly one group, and
    the sizes differ by at most one row -- the book leaves the remainder in the last group, this spreads it over the
    first, which is the more equal of the two. Every split then tests exactly k groups and trains on the other
    N - k, so before purging the training share is the book's theta = 1 - k / N whichever split it is."""
    for n, N, k in ((100, 6, 2), (97, 5, 1), (53, 8, 4)):
        t0 = np.sort(np.random.default_rng(n).integers(0, 10_000, n))
        gid = cpcv.group_index(t0, N)
        sizes = np.bincount(gid, minlength=N)
        assert sizes.sum() == n and sizes.max() - sizes.min() <= 1
        assert sorted(set(gid)) == list(range(N))
        assert np.all(np.diff(gid[np.argsort(t0, kind="stable")]) >= 0)       # contiguous, in order, never shuffled
        for split in cpcv.cpcv_splits(t0, t0 + 5, N, k, embargo=0):
            assert len(set(split["groups"])) == k
            assert set(gid[split["test"]]) <= set(split["groups"])
            assert not set(gid[split["train"]]) & set(split["groups"])        # never a tested group

    t0 = np.arange(600)                                                       # spans of one unit: nothing to purge
    for N, k in ((6, 1), (6, 2), (6, 3), (10, 2)):
        shares = [len(s["train"]) / len(t0) for s in cpcv.cpcv_splits(t0, t0 + 1, N, k, embargo=0)]
        assert np.allclose(shares, 1 - k / N)                                 # theta, equal across every split
    rising = [len(cpcv.cpcv_splits(t0, t0 + 1, N, 2, embargo=0)[0]["train"]) for N in (4, 6, 10, 20)]
    assert all(a < b for a, b in zip(rising, rising[1:]))                     # theta rises with N at fixed k
    falling = [len(cpcv.cpcv_splits(t0, t0 + 1, 10, k, embargo=0)[0]["train"]) for k in (1, 2, 3, 5)]
    assert all(a > b for a, b in zip(falling, falling[1:]))                   # and falls as k approaches N / 2


@pytest.mark.eval
def test_neither_ordering_of_the_sample_beats_the_other_and_a_selection_made_on_one_does_not_carry_over():
    """12.2.1's second disadvantage, which the chapter file left deferred to a run of this project's own walk-forward.
    The general claim can be settled on a known answer: 150 candidate rules whose trades depend on a trailing filter,
    so their backtest depends on the order of the sample. On pure returns the forward and backward orderings never
    select the same rule and the winner of one ordering keeps on average under 60% of its score in the other -- so a
    walk-forward selection is exploiting the sequence -- while the best backward backtest is as large as the best
    forward one, which is the book's point that walk-backward backtests do not systematically beat walk-forward ones.
    With one rule that genuinely pays, both orderings select it in every seed. Thresholds over six seeds."""
    periods, rules, look = 600, 150, 20

    def backtest(features, returns):
        trailing = np.concatenate([np.zeros(look), np.convolve(returns, np.ones(look) / look, "valid")[:-1]])
        traded = (features > 0) & (trailing[:, None] > 0)          # a rule trades only after a positive trailing mean
        return (traded * returns[:, None]).sum(axis=0)

    carried, ratios = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(periods, rules))
        noise = rng.normal(0, 1, periods)
        forward, backward = backtest(features, noise), backtest(features[::-1], noise[::-1])
        best_forward, best_backward = int(np.argmax(forward)), int(np.argmax(backward))
        assert best_forward != best_backward                        # the ordering chooses the strategy
        ratios.append(float(forward.max() / backward.max()))
        carried.append(float(forward[best_backward] / backward[best_backward]))

        paying = 0.5 * np.sign(features[:, 7]) + rng.normal(0, 1, periods)
        assert int(np.argmax(backtest(features, paying))) == 7      # a real edge is found whichever way time runs
        assert int(np.argmax(backtest(features[::-1], paying[::-1]))) == 7

    assert np.mean(carried) < 0.6                                   # the selection does not survive the reversal
    assert 0.6 < np.mean(ratios) < 1.7                              # neither ordering is systematically the easier one
