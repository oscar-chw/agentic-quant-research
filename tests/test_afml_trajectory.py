"""AFML ch.21 on this machine: the trading trajectory as an integer search, checked against an independent enumeration.

The small case runs; the sizes that do not are recorded as counts, so the reason is a number and not an opinion
(docs/claims/ch21.md, docs/hardware.md).
"""
import itertools
from math import comb

import numpy as np
import pytest

from pmlab.afml import trajectory as T

pytestmark = pytest.mark.case


def _params(seed=7, size=2, horizons=2):
    rng = np.random.default_rng(seed)
    return T.problem_params(size, horizons, n_obs=200, sigma=1.0, rng=rng)


def test_the_count_of_ordered_allocations_matches_the_pigeonhole_formula():
    for k, n in [(6, 3), (10, 4), (3, 2), (5, 1)]:
        p = T.partitions(k, n)
        assert p.shape == (comb(k + n - 1, n - 1), n)
        assert T.partition_count(k, n) == len(p)
        assert (p.sum(axis=1) == k).all() and (p >= 0).all()
        assert len({tuple(row) for row in p}) == len(p)


def test_order_matters_when_capital_is_allocated():
    rows = {tuple(r) for r in T.partitions(6, 3)}
    assert (1, 2, 3) in rows and (3, 2, 1) in rows          # the same partition, two different allocations
    assert (2, 2, 2) in rows


def test_every_feasible_weight_vector_is_fully_invested_and_every_sign_pattern_is_present():
    k, n = 4, 3
    omega = T.feasible_weights(k, n)
    assert omega.shape == (T.partition_count(k, n) * 2 ** n, n)
    assert np.allclose(np.abs(omega).sum(axis=1), 1.0)
    assert (omega < 0).any()                                 # 21.5.2: a vector may be short, not only long
    assert {tuple(-r) for r in omega} == {tuple(r) for r in omega}     # and the set is closed under sign flips
    unique = T.feasible_weights(k, n, unique=True)
    assert len(unique) < len(omega)                          # a zero weight has no sign, so the book's set repeats
    assert len({tuple(r) for r in omega}) == len(unique)
    assert (np.lexsort(T.partitions(k, n).T[::-1]) == np.arange(T.partition_count(k, n))).all()   # lexicographic


def test_degenerate_sizes_are_refused_rather_than_answered_with_nan():
    with pytest.raises(ValueError):
        T.partitions(3, 0)                                   # used to recurse on n = -1 until RecursionError
    with pytest.raises(ValueError):
        T.feasible_weights(0, 3)                             # used to divide by zero and return all-NaN weights
    mu, cov, cost = T.problem_params(1, 2, n_obs=50, rng=np.random.default_rng(1))
    assert cov.shape == (2, 1, 1)                            # np.cov of one column is 0-d without atleast_2d
    assert np.isfinite(T.trajectory_sharpe(np.array([[1.0], [1.0]]), mu, cov, cost, np.zeros(1)))


def test_transaction_costs_follow_the_square_root_of_the_change():
    traj = np.array([[1.0, 0.0], [0.0, 1.0]])
    cost = np.array([[0.1, 0.2], [0.1, 0.2]])
    tau = T.transaction_costs(traj, cost, w0=np.array([0.0, 0.0]))
    assert tau[0] == pytest.approx(0.1 * 1.0 + 0.2 * 0.0)
    assert tau[1] == pytest.approx(0.1 * 1.0 + 0.2 * 1.0)


def test_the_sharpe_ratio_of_a_trajectory_matches_a_hand_computation():
    traj = np.array([[1.0]])
    sr = T.trajectory_sharpe(traj, mu=np.array([[0.02]]), cov=np.array([[[0.04]]]),
                             cost=np.array([[0.1]]), w0=np.array([0.0]))
    assert sr == pytest.approx((0.02 - 0.1) / 0.2)


def test_a_trajectory_that_never_trades_pays_only_the_entry_cost():
    traj = np.array([[0.5, -0.5], [0.5, -0.5]])
    cost = np.full((2, 2), 0.3)
    tau = T.transaction_costs(traj, cost, w0=np.array([0.5, -0.5]))
    assert tau.tolist() == [0.0, 0.0]


def test_the_exhaustive_search_finds_what_an_independent_enumeration_finds():
    mu, cov, cost = _params()
    omega, w0 = T.feasible_weights(3, 2, unique=True), np.zeros(2)
    best_sr, best = T.evaluate_trajectories(mu, cov, cost, w0, omega)
    hand_sr, hand = -np.inf, None
    for combo in itertools.product(range(len(omega)), repeat=len(mu)):
        traj = omega[list(combo)]
        gain = sum(float(mu[h] @ traj[h]) for h in range(len(mu)))
        previous = w0
        for h in range(len(mu)):
            gain -= float(sum(cost[h] * np.sqrt(np.abs(traj[h] - previous))))
            previous = traj[h]
        var = sum(float(traj[h] @ cov[h] @ traj[h]) for h in range(len(mu)))
        sr = gain / np.sqrt(var)
        if sr > hand_sr:
            hand_sr, hand = sr, traj
    assert best_sr == pytest.approx(hand_sr)
    assert np.allclose(best, hand)


def test_a_singular_covariance_makes_the_riskless_trajectory_the_answer_not_the_worst_one():
    """21.5.3: no analytical property of the objective is used, so an ill-conditioned V is no obstacle.

    Rank 1 with mu = (0.01, 0.02): w = (-1/2, +1/2) gains +0.005 at variance exactly 0. Ranking a riskless point
    -inf (what a `var <= 0` guard does) deletes the best point of the feasible set and answers (-1/2, +1/2)'s
    opposite instead.
    """
    mu = np.array([[0.01, 0.02]])
    cov = np.array([[[1.0, 1.0], [1.0, 1.0]]])               # rank 1: no inverse, no convex optimiser
    omega = T.feasible_weights(2, 2, unique=True)
    sr, traj = T.evaluate_trajectories(mu, cov, np.zeros((1, 2)), np.zeros(2), omega)
    assert np.allclose(traj, [[-0.5, 0.5]]) and sr == float("inf")
    assert np.allclose(np.abs(traj).sum(axis=1), 1.0)
    assert float((mu * traj).sum()) == pytest.approx(0.005)
    assert float(np.einsum("hi,hij,hj->", traj, cov, traj)) == 0.0
    assert T.trajectory_sharpe(-traj, mu, cov, np.zeros((1, 2)), np.zeros(2)) == float("-inf")   # riskless loss
    assert T.trajectory_sharpe(np.array([[0.5, 0.5]]), np.zeros((1, 2)), cov,
                               np.zeros((1, 2)), np.zeros(2)) == 0.0                             # riskless, no gain
    static_sr, static_traj = T.static_solution(mu, cov, np.zeros((1, 2)), np.zeros(2), omega)
    assert np.allclose(static_traj, traj) and static_sr == sr   # the benchmark ranks it the same way


def test_the_dynamic_optimum_is_never_worse_than_the_chain_of_local_optima():
    omega, w0 = T.feasible_weights(3, 2, unique=True), np.zeros(2)
    beaten = 0
    for seed in range(6):
        mu, cov, cost = _params(seed=seed)
        dynamic, _ = T.evaluate_trajectories(mu, cov, cost, w0, omega)
        static, traj = T.static_solution(mu, cov, cost, w0, omega)
        assert dynamic >= static - 1e-12
        assert np.allclose(np.abs(traj).sum(axis=1), 1.0)
        # 21.6.2: each row is that horizon's own optimum, ignoring what the move costs. A benchmark that picks any
        # other vector still satisfies the two assertions above, so it is checked row by row here.
        for h in range(len(mu)):
            score = (omega @ mu[h]) / np.sqrt(np.einsum("oi,ij,oj->o", omega, cov[h], omega))
            assert np.allclose(traj[h], omega[int(np.argmax(score))])
        beaten += dynamic > static + 1e-9
    assert beaten > 0                                        # costs make the local optima suboptimal, as 21.4 says


def test_the_pool_and_the_single_process_search_agree():
    mu, cov, cost = _params()
    omega, w0 = T.feasible_weights(3, 2, unique=True), np.zeros(2)
    one = T.evaluate_trajectories(mu, cov, cost, w0, omega)
    many = T.evaluate_trajectories(mu, cov, cost, w0, omega, num_threads=2, mp_batches=2)
    assert one[0] == pytest.approx(many[0]) and np.allclose(one[1], many[1])


def test_a_random_matrix_has_the_rank_it_was_asked_for():
    rng = np.random.default_rng(3)
    clean = T.random_matrix(200, 10, sigma=0.0, rank=4, rng=rng)
    assert np.linalg.matrix_rank(clean) == 4
    noisy = T.random_matrix(200, 10, sigma=0.2, rank=4, rng=rng)
    values = np.sort(np.linalg.eigvalsh(np.cov(noisy, rowvar=False)))[::-1]
    assert values[3] > 10 * values[4]                         # four signal eigenvalues, the rest noise


def test_the_problem_parameters_give_one_set_per_horizon():
    mu, cov, cost = T.problem_params(3, 4, n_obs=200, rng=np.random.default_rng(11))
    assert mu.shape == (4, 3) and cov.shape == (4, 3, 3) and cost.shape == (4, 3)
    # 21.6.1 draws mu, V and C per horizon: the chapter's premise is that they are not identically distributed
    assert not np.allclose(mu[0], mu[1]) and not np.allclose(cov[0], cov[1]) and not np.allclose(cost[0], cost[1])
    for h in range(4):
        assert np.allclose(cov[h], cov[h].T)
        assert np.linalg.eigvalsh(cov[h]).min() > -1e-9
    assert (cost >= 0).all()


def test_the_whole_search_space_is_walked_by_index_so_the_parent_holds_no_list_of_trajectories():
    """The chapter's own target case is 2.43e9 trajectories; listing their index tuples is 163 GiB on a 32 GB box."""
    omega = T.feasible_weights(3, 2, unique=True)
    every = [T.trajectory_at(i, omega, 2) for i in range(len(omega) ** 2)]
    assert len({tuple(t.ravel()) for t in every}) == len(omega) ** 2      # a bijection onto Phi
    assert np.allclose(every[0], [omega[0], omega[0]]) and np.allclose(every[-1], [omega[-1], omega[-1]])
    assert np.allclose(every[1], [omega[0], omega[1]])                    # lexicographic, last horizon fastest
    mu, cov, cost = _params(size=3, horizons=3)
    big = T.feasible_weights(3, 3, unique=True)                           # 38^3 = 54,872 trajectories
    assert len(big) ** 3 == 54_872
    peak = _peak_bytes(lambda: T.evaluate_trajectories(mu, cov, cost, np.zeros(3), big))
    assert peak < 1_000_000, peak      # listing the product alone is ~4 MB of index tuples at this size


def _peak_bytes(call):
    import tracemalloc
    tracemalloc.start()
    call()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def test_the_search_explodes_with_the_number_of_horizons():
    assert T.trajectory_count(3, 2, 2) == (T.partition_count(3, 2) * 2 ** 2) ** 2
    assert T.trajectory_count(10, 5, 1) == 32_032
    assert T.trajectory_count(10, 5, 4) == 32_032 ** 4
    assert T.trajectory_count(10, 5, 4) > 1e18               # out of reach here: see docs/claims/ch21.md


def test_the_hours_this_machine_would_need_are_the_count_over_the_measured_rate():
    """The hardware judgements of docs/claims/ch21.md are arithmetic on two numbers, so they can be wrong: the
    count, and the measured 313,000 trajectories a second on the 4 workers of docs/hardware.md (119,000 on one
    core, a 2.6x speed-up). This fixes both the counts and the conclusions drawn from them.

    A rate that is not the measured one, or a count read off the nominal set where the search walks the distinct
    one, moves these by more than the bounds allow -- which is how the first pass came to 64,000 years and to a
    'largest case that fits' ten times too small.
    """
    per_core, workers = 119_000, 4
    rate = 313_000                                            # measured, not per_core * workers
    assert rate < 0.8 * per_core * workers                    # the pool does not scale linearly
    hours = lambda count: count / rate / 3600

    assert T.trajectory_count(10, 5, 1) == 32_032 and T.trajectory_count(10, 5, 1, unique=True) == 14_002
    assert T.trajectory_count(10, 5, 4) == pytest.approx(1.053e18, rel=0.01)
    assert hours(T.trajectory_count(10, 5, 4)) / 24 / 365.25 == pytest.approx(107_000, rel=0.05)
    # even without the duplicates the construction repeats, the section's own size is out of reach
    assert hours(T.trajectory_count(10, 5, 4, unique=True)) / 24 / 365.25 == pytest.approx(3_890, rel=0.05)

    # the largest case inside the hours-not-days ceiling, counted on the set the search actually walks
    assert T.trajectory_count(8, 4, 1, unique=True) == 1_408
    assert hours(T.trajectory_count(8, 4, 3, unique=True)) == pytest.approx(2.48, rel=0.05)
    assert hours(T.trajectory_count(9, 4, 3, unique=True)) > 4.0        # the next step up leaves the ceiling
    assert hours(T.trajectory_count(6, 4, 3, unique=True)) < 0.25       # what the first pass called 1.3 hours
