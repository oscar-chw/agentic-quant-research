"""Book claims of AFML chapters 1-5 that the section-level walkthrough left unchecked.

docs/claims/ch01.md - ch05.md break each heading into its individual logic, formula, structure,
algorithm and pitfall items; the items that no existing test could settle are checked here, each
against the book itself: a worked example, a closed form, or a simulation with a known answer.
"""
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

from pmlab import labels
from pmlab.afml import fracdiff as fd
from pmlab.afml import hrp
from pmlab.afml import labeling
from pmlab.afml import lifecycle as lc
from pmlab.afml import sampling as sp
from pmlab import bars as bars_mod
from pmlab.bars.imbalance import ImbalanceBars
from pmlab.bars.runs import RunBars
from pmlab.events import make_trade

ROOT = Path(__file__).resolve().parents[1]
BOOK_V2 = ROOT / "docs" / "claims"

W = 1_000_000


def tr(t, sign=1, p=0.5, shares=1.0, window=W):
    return make_trade(window + t, window, p, sign, shares, p * shares)


# ---------------------------------------------------------------- chapter 2


@pytest.mark.case
def test_the_tick_rules_boundary_sign_is_carried_from_the_previous_bar_not_reset():
    """2.3.2(1): b_0 is set to match b_T of the immediately preceding bar (it resets only per window)."""
    b = ImbalanceBars("tick", init_T=2, init_bv=1.0, init_bv_sd=0.1, floor_sd=0, sign="tick_rule", anchor="bars")
    b.new_window(W)
    # theta = +1, 0, -1, -2: the bar closes on a down tick, so b_T = -1
    first = [b.update(tr(i, 1, p=p)) for i, p in enumerate([.50, .49, .48, .47], start=1)]
    assert [x for x in first if x is not None][0].n_ticks == 4
    assert b.tick_rule.last_b == -1

    # unchanged price carries -1 into the new bar, then two up ticks bring theta back to 0.
    # Had b_0 reset to the rule's initial +1, theta would have hit the threshold of 2 on the second tick.
    after = [b.update(tr(i, 1, p=p)) for i, p in enumerate([.47, .48, .49], start=5)]
    assert all(x is None for x in after)
    assert b.tick_rule.last_b == 1

    b.new_window(W + 300)                                   # a new token is a new price series
    assert b.tick_rule.last_p is None and b.tick_rule.last_b == 1


@pytest.mark.property
def test_the_imbalance_expectation_splits_into_a_buy_and_a_sell_component():
    """2.3.2(2): with v+ = P[b=1] E[v|b=1] and v- = P[b=-1] E[v|b=-1], the book's algebra says
    v+ + v- = E[v] and E_0[theta_T] = E_0[T] (v+ - v-) = E_0[T] (2 v+ - E[v])."""
    rng = np.random.default_rng(11)
    for p_buy in (0.2, 0.5, 0.75):
        b = np.where(rng.random(200_000) < p_buy, 1, -1)
        v = rng.lognormal(0.0, 0.6, 200_000) * np.where(b > 0, 1.3, 0.8)     # buys trade bigger
        v_plus = np.mean(b > 0) * v[b > 0].mean()
        v_minus = np.mean(b < 0) * v[b < 0].mean()
        assert v_plus + v_minus == pytest.approx(v.mean(), rel=1e-12)
        assert v_plus - v_minus == pytest.approx((b * v).mean(), rel=1e-12)
        assert 2 * v_plus - v.mean() == pytest.approx((b * v).mean(), rel=1e-12)
        expected_T = 40.0
        assert expected_T * abs(2 * v_plus - v.mean()) == pytest.approx(expected_T * abs((b * v).mean()), rel=1e-12)


def _book_cusum(y, h):
    """Snippet 2.4 literally: E_{t-1}[y_t] = y_{t-1}, so the increments are the returns themselves."""
    out, s_pos, s_neg = [], 0.0, 0.0
    for r in y:
        s_pos, s_neg = max(0.0, s_pos + r), min(0.0, s_neg + r)
        if s_neg < -h:
            s_neg = 0.0
            out.append(-1)
        elif s_pos > h:
            s_pos = 0.0
            out.append(1)
        else:
            out.append(0)
    return np.array(out)


@pytest.mark.property
def test_the_symmetric_cusum_filter_follows_the_books_recursion_step_by_step():
    """2.5.2(1): S+ = max(0, S+ + y - E[y]), S- = min(0, S- + y - E[y]), S = max(S+, -S-), and the
    side that fires resets to zero."""
    rng = np.random.default_rng(12)
    for h in (0.5, 1.0, 2.0):
        y = rng.normal(0, 1, 5000)
        assert labels.cusum_filter(y, h).tolist() == _book_cusum(y, h).tolist()


@pytest.mark.case
def test_the_cusum_filter_needs_a_full_run_of_h_and_does_not_retrigger_while_hovering():
    """2.5.2(1): unlike a band, hovering around the threshold does not fire repeatedly - after an
    event the fired side resets to zero, so a further h of run-up is needed."""
    # +1.0 then a long drift around the level: one event, then nothing until another full run of 1.
    y = np.array([0.6, 0.6, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05])
    assert labels.cusum_filter(y, 1.0).tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    # the same total move delivered in small steps still fires exactly once, at the step that
    # completes the run of h
    small = np.full(12, 0.1)
    fired = labels.cusum_filter(small, 1.0)
    assert fired.tolist() == [0] * 10 + [1, 0]
    # a run shorter than h never fires, however long it hovers
    assert not labels.cusum_filter(np.full(50, 0.01), 1.0).any()


# ---------------------------------------------------------------- chapter 3


@pytest.mark.case
def test_disabling_both_horizontal_barriers_reproduces_the_fixed_time_horizon_label():
    """3.4: the configuration [0, 0, 1] - both horizontal barriers switched off by a zero factor -
    is equivalent to the fixed-time horizon method of 3.2, whose label is the sign of
    r = p[t0 + h] / p[t0] - 1."""
    rng = np.random.default_rng(13)
    n, horizon = 400, 12
    path = 100 + np.cumsum(rng.normal(0, 1, n))
    times = np.arange(n, dtype=float)
    entries = np.arange(10, 380, 7)
    target = np.full(n, 0.01)                                    # 1% barriers, well inside the path

    off = labeling.get_events(times, path, times[entries], target, (0.0, 0.0),
                              t1=times[entries + horizon])
    assert off["t1"].to_numpy().tolist() == times[entries + horizon].tolist()   # only the vertical one
    bins = labeling.get_bins(times, path, off["t0"].to_numpy(), off["t1"].to_numpy())
    ret = path[entries + horizon] / path[entries] - 1
    assert bins["ret"].to_numpy() == pytest.approx(ret)
    assert bins["bin"].to_numpy().tolist() == np.sign(ret).tolist()
    fh, end = labels.fixed_horizon(path, entries + 1, horizon=horizon)
    assert end.tolist() == (entries + horizon).tolist()
    assert bins["bin"].to_numpy().tolist() == fh.tolist()

    # switching the horizontal barriers on decides some events earlier and changes their labels,
    # so the equality above is not vacuous
    on = labeling.get_events(times, path, times[entries], target, (1.0, 1.0), t1=times[entries + horizon])
    assert (on["t1"].to_numpy() < off["t1"].to_numpy()).any()
    on_bins = labeling.get_bins(times, path, on["t0"].to_numpy(), on["t1"].to_numpy())
    assert on_bins["bin"].to_numpy().tolist() != bins["bin"].to_numpy().tolist()


@pytest.mark.case
def test_precision_recall_and_f1_are_the_areas_of_the_books_confusion_matrix():
    """3.7 (Figure 3.2): precision = TP / (TP + FP), recall = TP / (TP + FN), accuracy =
    (TP + TN) / all, F1 = the harmonic average of precision and recall."""
    tp, fp, fn, tn = 30, 20, 10, 40
    y = np.array([1] * (tp + fn) + [0] * (fp + tn))
    p = np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    assert precision_score(y, p) == pytest.approx(precision)
    assert recall_score(y, p) == pytest.approx(recall)
    assert np.mean(y == p) == pytest.approx((tp + tn) / (tp + fp + fn + tn))
    assert f1_score(y, p) == pytest.approx(stats.hmean([precision, recall]))
    assert f1_score(y, p) == pytest.approx(2 * precision * recall / (precision + recall))


@pytest.mark.eval
def test_meta_labeling_raises_f1_by_filtering_the_false_positives_of_a_high_recall_primary():
    """3.7: build a primary with high recall and poor precision, then correct the precision with a
    secondary model fitted on its positives. Thresholds on 4,000 held-out rows: the primary's recall
    is above 0.95 and its precision below 0.60; meta-labelling raises precision by more than 0.20
    and F1 by more than 0.10, while recall falls (the trade the book describes)."""
    rng = np.random.default_rng(14)
    n = 12_000
    signal = rng.normal(size=n)                    # what the primary sees
    quality = rng.normal(size=n)                   # what only the secondary sees
    # the bet wins when the primary is right about the side AND the regime is favourable
    win = (signal > 0) & (quality + rng.normal(0, 0.5, n) > 0)
    side = signal > 0                              # the primary acts on every positive signal
    train, test = slice(0, 8000), slice(8000, n)

    y_true = win[test][side[test]]
    primary = np.ones(y_true.shape, dtype=bool)    # the primary acts on all of its positives
    assert recall_score(y_true, primary) > 0.95 and precision_score(y_true, primary) < 0.60

    fit_rows = side[train]
    model = LogisticRegression().fit(quality[train][fit_rows].reshape(-1, 1), win[train][fit_rows])
    act = model.predict(quality[test][side[test]].reshape(-1, 1)).astype(bool)

    assert precision_score(y_true, act) - precision_score(y_true, primary) > 0.20
    assert f1_score(y_true, act) - f1_score(y_true, primary) > 0.10
    assert recall_score(y_true, act) < recall_score(y_true, primary)


# ---------------------------------------------------------------- chapter 4


@pytest.mark.property
def test_average_uniqueness_is_the_reciprocal_of_the_harmonic_mean_of_concurrency():
    """4.4: u_bar_i is the mean of 1 / c_t over the label's span, i.e. the reciprocal of the
    harmonic average of c_t over that span."""
    rng = np.random.default_rng(15)
    t0, t1 = sp.random_spans(40, 120, 15, rng)
    units = np.arange(t0.min(), t1.max())
    c = sp.concurrency(t0, t1, units)
    u = labels.average_uniqueness(t0, t1)
    for i, (a, b) in enumerate(zip(t0, t1)):
        span = c[(units >= a) & (units < b)]
        assert u[i] == pytest.approx(np.mean(1 / span))
        assert u[i] == pytest.approx(1 / stats.hmean(span))


@pytest.mark.eval
def test_a_standard_bootstrap_leaves_about_one_in_e_of_the_observations_out_of_bag():
    """4.5: P[i never drawn in I draws] = (1 - 1/I)^I -> e^-1, so the unique share of an in-bag
    sample tends to 1 - e^-1 ~ 2/3. Threshold: 4,000 draws of I = 250 land within 0.005 of the
    closed form, which itself is within 0.002 of 1 - e^-1."""
    for I in (10, 100, 10_000):
        assert (1 - 1 / I) ** I == pytest.approx(math.exp(-1), abs=0.06 if I == 10 else 0.006)
    I = 250
    closed = 1 - (1 - 1 / I) ** I
    assert closed == pytest.approx(1 - math.exp(-1), abs=0.002)
    rng = np.random.default_rng(16)
    share = np.mean([len(np.unique(rng.integers(0, I, I))) / I for _ in range(4000)])
    assert share == pytest.approx(closed, abs=0.005)


@pytest.mark.case
def test_draws_of_a_non_overlapping_observation_are_poisson_with_mean_i_over_k():
    """4.5: with K <= I non-overlapping outcomes, the count of draws of one of them over I draws is
    approximately Poisson with mean I/K, and P[count = 0] = (1 - 1/K)^I ~ e^(-I/K). Assuming IID
    draws therefore oversamples: the mean count is I/K >= 1, not 1."""
    I, K = 1000, 100
    exact = stats.binom(I, 1 / K)
    approx = stats.poisson(I / K)
    ks = np.arange(0, 40)
    assert np.max(np.abs(exact.pmf(ks) - approx.pmf(ks))) < 0.005
    assert exact.pmf(0) == pytest.approx((1 - 1 / K) ** I, rel=1e-12)
    assert exact.pmf(0) == pytest.approx(math.exp(-I / K), abs=5e-4)
    assert exact.mean() == pytest.approx(I / K) and I / K >= 1
    # the IID reading of the same draws expects each of the I items once
    assert stats.binom(I, 1 / I).mean() == pytest.approx(1.0)


@pytest.mark.case
def test_time_decay_is_the_books_piecewise_line_with_its_two_boundary_conditions():
    """4.7: d[x] = max(0, a + b x) on cumulative uniqueness, with a = 1 - b sum(u) so that the
    newest weight is 1, b = (1 - c) / sum(u) for c in [0, 1] (d[0] = c), and b = 1 / ((c + 1)
    sum(u)) for c in (-1, 0) (d vanishes at -c sum(u)). Figure 4.3's values of c."""
    u = np.array([0.5, 0.5, 1.0, 1.0, 1.0])
    total, x = u.sum(), np.cumsum(u)
    for c in (1.0, 0.75, 0.5, 0.0, -0.25, -0.5):
        b = (1 - c) / total if c >= 0 else 1 / ((c + 1) * total)
        a = 1 - b * total
        assert sp.time_decay(u, c=c).tolist() == pytest.approx(np.maximum(0.0, a + b * x).tolist())
        assert a + b * total == pytest.approx(1.0)                      # the final weight has no decay
        if c >= 0:
            assert a + b * 0 == pytest.approx(c)                        # d[0] = c
            assert np.all(sp.time_decay(u, c=c) > 0) if c > 0 else True
        else:
            assert a + b * (-c * total) == pytest.approx(0.0)           # the oldest cT is erased
            assert sp.time_decay(u, c=c)[0] == 0.0
    assert sp.time_decay(u, c=1.0).tolist() == pytest.approx([1.0] * 5)  # c = 1 is no decay


@pytest.mark.case
def test_class_weights_that_do_not_sum_to_the_number_of_classes_act_as_regularisation():
    """4.8: class weights that do not sum to the number of classes act like a change of the
    regularisation parameter. sklearn minimises 0.5 w'w + C sum_i s_i loss_i, so scaling every
    class weight by k is exactly scaling C by k - and balancing, which does not scale uniformly,
    is not."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(400, 3))
    y = (X @ np.array([1.5, -1.0, 0.5]) + rng.normal(0, 0.5, 400) > 0).astype(int)
    k, C = 4.0, 0.3
    plain = LogisticRegression(C=C * k).fit(X, y)
    scaled = LogisticRegression(C=C, class_weight={0: k, 1: k}).fit(X, y)
    assert scaled.coef_ == pytest.approx(plain.coef_, rel=1e-4, abs=1e-6)
    assert scaled.intercept_ == pytest.approx(plain.intercept_, rel=1e-4, abs=1e-6)
    skewed = LogisticRegression(C=C, class_weight={0: 1.0, 1: k}).fit(X, y)
    assert not np.allclose(skewed.coef_, plain.coef_, rtol=1e-3)


# ---------------------------------------------------------------- chapter 5


@pytest.mark.property
def test_the_fracdiff_weights_are_the_binomial_coefficients_of_one_minus_b_to_the_d():
    """5.4: (1 - B)^d = sum_k (-B)^k prod_{i<k} (d - i) / k!, so w_k = (-1)^k prod_{i<k} (d - i) / k!
    computed straight from the product, not from the recursion the code uses."""
    for d in (0.0, 0.25, 0.4, 0.5, 1.0, 1.4, 2.0, 2.5):
        got = fd.expanding_weights(d, 12)
        want = []
        for k in range(12):
            prod = 1.0
            for i in range(k):
                prod *= d - i
            want.append((-1) ** k * prod / math.factorial(k))
        assert got.tolist() == pytest.approx(want, rel=1e-12, abs=1e-15)
    # the book's own example: (1 - B)^2 = 1 - 2B + B^2
    assert fd.expanding_weights(2.0, 4).tolist() == pytest.approx([1.0, -2.0, 1.0, 0.0])
    x = np.array([3.0, 5.0, 11.0, 12.0, 20.0])
    assert fd.frac_diff_ffd(x, 2.0, thres=1e-12)[-1] == pytest.approx(20 - 2 * 12 + 11)


@pytest.mark.case
def test_the_weight_signs_and_convergence_follow_the_parity_of_the_integer_part_of_d():
    """5.4.3: |w_k / w_{k-1}| = |(d - k + 1) / k| < 1 for k > d, so the weights converge to zero;
    once k >= d + 1 they are negative when int[d] is even and positive when it is odd."""
    # int[d] of 0, 1, 2 and 3: with only 0 and 1 the rule cannot be told apart from "negative below 1, positive
    # above it", which is why the even int[d] = 2 case (negative again) and the odd int[d] = 3 case are here
    for d, k0, positive in ((0.4, 1, False), (0.75, 1, False), (1.4, 2, True), (1.9, 2, True),
                            (2.5, 3, False), (2.2, 3, False), (3.5, 4, True), (3.1, 4, True)):
        w = fd.expanding_weights(d, 30)
        assert w[0] == 1.0
        tail = w[k0:]
        assert np.all(tail > 0) if positive else np.all(tail < 0)
        assert np.all(np.diff(np.abs(tail)) <= 0)                          # |w_k| decreasing once k > d
        ratios = np.abs(tail[1:] / tail[:-1])
        assert np.all(ratios < 1) and abs(w[-1]) < abs(w[k0]) and abs(w[-1]) < 0.05
        for k in range(1, 30):
            assert w[k] == pytest.approx(-w[k - 1] * (d - k + 1) / k, rel=1e-12, abs=1e-18)
    # d in (0, 1) keeps every later weight inside (-1, 0)
    w = fd.expanding_weights(0.4, 50)[1:]
    assert np.all((w > -1) & (w < 0))
    # d > 1 starts below -1, as Figure 5.2 shows
    assert fd.expanding_weights(1.4, 4)[1] < -1


@pytest.mark.case
def test_the_fixed_width_window_stops_at_the_first_weight_below_the_threshold():
    """5.5.2: w~_k = w_k for k <= l* and 0 after it, where |w_{l*}| >= tau > |w_{l*+1}|."""
    for d, tau in ((0.4, 1e-2), (0.4, 1e-4), (0.9, 1e-3), (1.5, 1e-5)):
        w = fd.ffd_weights(d, tau)
        assert abs(w[-1]) >= tau                                  # the last kept weight clears tau
        nxt = -w[-1] * (d - len(w) + 1) / len(w)                  # the one the window drops
        assert abs(nxt) < tau
        assert w.tolist() == pytest.approx(fd.expanding_weights(d, len(w)).tolist(), rel=1e-12)
    # a tighter tolerance can only widen the window
    assert len(fd.ffd_weights(0.4, 1e-5)) > len(fd.ffd_weights(0.4, 1e-2))
    # and at d = 1 the window is exactly the first difference, whatever the tolerance
    assert fd.ffd_weights(1.0, 1e-9).tolist() == pytest.approx([1.0, -1.0])


@pytest.mark.case
def test_the_adf_critical_value_of_the_book_is_the_mackinnon_value_at_its_sample_size():
    """5.6: the book reports a 95% critical value of -2.8623 for the E-mini daily series. That is
    MacKinnon's no-trend constant-only value at a few thousand observations; the asymptote is
    -2.86154 and the finite-sample correction supplies the rest."""
    assert fd.adf_critical(10 ** 9) == pytest.approx(-2.86154, abs=1e-5)
    assert fd.adf_critical(3800) == pytest.approx(-2.8623, abs=5e-5)
    assert -2.8624 < fd.adf_critical(3500) < -2.8620 and -2.8624 < fd.adf_critical(4200) < -2.8620
    # the correction is monotone: a shorter sample needs a more negative statistic
    ns = np.array([100, 250, 500, 1000, 3800])
    crits = np.array([fd.adf_critical(int(n)) for n in ns])
    assert np.all(np.diff(crits) > 0) and crits[0] < -2.88


# ------------------------------------------------------- chapter 1 (the second reading, plan/book-v2.md node 15)


@pytest.mark.case
def test_the_five_parts_of_the_book_partition_the_chapters_in_the_order_they_are_read():
    """1.3: five parts, read in order, each chapter assuming the ones before it. The boundaries are the book's own
    (data structures, research, backtesting, features, HPC), so a chapter that slipped out of the walkthrough or a
    part boundary written down wrongly fails here."""
    parts = {"1 data structures": range(2, 6), "2 modelling": range(6, 10), "3 backtesting": range(10, 17),
             "4 features": range(17, 20), "5 hpc": range(20, 23)}
    # Loaded here, not at import: the heading index script is not published, and this test is deselected without it.
    spec = importlib.util.spec_from_file_location("afml_sections", ROOT / "scripts" / "afml_sections.py")
    sections = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sections)
    index, problems, _ = sections.expected_index()
    assert problems == []
    covered = [c for r in parts.values() for c in r]
    assert covered == sorted(covered) == list(range(2, 23))            # in order, no gap, no chapter twice
    for name, rng in parts.items():
        for c in rng:
            assert index[str(c)], f"{name}: chapter {c} has no headings"
            assert (BOOK_V2 / f"ch{c:02d}.md").is_file(), f"{name}: chapter {c} has no v2 file"
    assert index["1"][:3] == ["1.1", "1.2", "1.2.1"] and index["22"]


@pytest.mark.case
def test_a_longer_track_record_earns_a_larger_share_of_the_same_hrp_weight():
    """1.3.1(6): old strategies take smaller allocations for diversification, tempered by the extra confidence that a
    longer track record earns. Two strategies with identical P&L - so identical HRP weight
    and identical health - differ only in how long ago they graduated, and the older one must get strictly more."""
    rng = np.random.default_rng(5)
    now = 90 * 86400.0
    grid = np.arange(0, now, 300)
    records, graduated = {}, {}
    for i, (name, age_days) in enumerate((("young", 10.0), ("old", 80.0))):
        pnl = rng.normal(0.4, 1.0, len(grid)) if i == 0 else None
        records[name] = pd.DataFrame({"start": grid, "pnl": pnl if pnl is not None else records["young"]["pnl"],
                                      "cost": 1.0})
        graduated[name] = {"since": now - age_days * 86400, "graduation": {"sharpe": None}}
    out = lc.allocate(graduated, records, now)["strategies"]
    assert out["old"]["hrp_weight"] == pytest.approx(out["young"]["hrp_weight"])   # same returns, same diversification
    assert out["old"]["health"] == out["young"]["health"] == 1.0
    assert out["old"]["ramp"] > out["young"]["ramp"]
    assert out["old"]["fraction"] > out["young"]["fraction"] * 1.5


@pytest.mark.eval
def test_the_in_sample_optimum_degrades_out_of_sample_where_the_robust_allocation_does_not():
    """1.3.2(4): an in-sample optimum can do badly out of sample, and nothing proves an investment will
    succeed. The book's 16.4 experiment in miniature: minimum variance (w proportional to the
    inverse covariance) is by construction the in-sample optimum, and HRP is not. Thresholds over 40 seeds of a
    three-factor world with 20 assets and only 30 training rows: minimum variance wins in sample every time, its
    out-of-sample variance is at least 3x its in-sample variance, and it degrades worse than HRP on every seed."""
    rng = np.random.default_rng(0)
    n, t = 20, 30
    best_in_sample, mv_degradation, hrp_degradation = [], [], []
    for _ in range(40):
        loads, sd = rng.normal(0, 1, (n, 3)), rng.uniform(0.5, 2.0, n)

        def draw(m):
            return ((loads @ rng.normal(0, 1, (3, m))).T + rng.normal(0, 1, (m, n))) * sd

        train, test = draw(t), draw(10 * t)
        cov = np.cov(train, rowvar=False)
        inv = np.linalg.pinv(cov) @ np.ones(n)
        mv = inv / inv.sum()                                          # the in-sample optimum
        w = hrp.hrp(cov)["weights"]
        best_in_sample.append((train @ mv).var(ddof=1) < (train @ w).var(ddof=1))
        mv_degradation.append((test @ mv).var(ddof=1) / (train @ mv).var(ddof=1))
        hrp_degradation.append((test @ w).var(ddof=1) / (train @ w).var(ddof=1))
    mv_degradation, hrp_degradation = np.array(mv_degradation), np.array(hrp_degradation)
    assert all(best_in_sample)                                        # it really is optimal where it was fitted
    assert np.median(mv_degradation) > 3.0 and mv_degradation.min() > 1.0
    assert np.median(hrp_degradation) < 2.0
    assert np.all(mv_degradation > hrp_degradation)


# ------------------------------------------------ chapter 2 (the second reading, plan/book-v2.md node 15)


def _ewma(x0: float, xs: list[float], a: float) -> float:
    """The book's exponentially weighted moving average written as its geometric weights, not as the update line:
    (1-a)^k x0 + a sum_j (1-a)^(k-j) x_j."""
    k = len(xs)
    return (1 - a) ** k * x0 + a * sum((1 - a) ** (k - j) * x for j, x in enumerate(xs, start=1))


@pytest.mark.case
def test_the_bar_expectations_are_the_exponentially_weighted_averages_the_book_prescribes():
    """2.3.2(1)-2.3.2(4): E_0[T] is an EWMA of the previous bars' lengths, E_0[b v] an EWMA of signed flow,
    P[b=1] an EWMA of the buy indicator, and E_0[v|b] an EWMA of the sizes of that side's own trades. The existing
    tests only assert that each is above a bound; here each is pinned to the geometric-weight definition at a known
    alpha = 2/(span+1), which fails if the wrong quantity or the wrong span were averaged."""
    a = 2.0 / (9 + 1)
    b = ImbalanceBars("volume", init_T=1000.0, init_bv=1.0, init_bv_sd=0.0, span_bars=9, span_ticks=9,
                      floor_sd=0.0, anchor="bars", clip=100.0)
    b.new_window(W)
    signed = []
    for i, (sign, shares) in enumerate([(1, 2.0), (-1, 3.0), (1, 1.0), (1, 4.0), (-1, 0.5)], start=1):
        assert b.update(tr(i, sign, shares=shares)) is None          # threshold 1000 is far away
        signed.append(sign * shares)
    assert b.E_bv == pytest.approx(_ewma(1.0, signed, a))
    assert b.E_bv == pytest.approx(0.85232, abs=1e-5)                # hand value at alpha = 0.2
    assert b.E_bv2 == pytest.approx(_ewma(1.0, [x * x for x in signed], a))

    # E_0[T] is an EWMA over closed bars, so it moves only when a bar closes, and it is the bar's tick count
    c = ImbalanceBars("tick", init_T=4.0, init_bv=1.0, init_bv_sd=0.0, span_bars=9, span_ticks=1e9,
                      floor_sd=0.0, anchor="bars", clip=100.0)
    c.new_window(W)
    closed = [x for x in (c.update(tr(i, 1)) for i in range(1, 5)) if x is not None]
    assert len(closed) == 1 and closed[0].n_ticks == 4               # threshold 4*1 = 4 reached on the fourth tick
    assert c.E_T == pytest.approx(_ewma(4.0, [4.0], a)) == pytest.approx(4.0)
    assert c.thr == pytest.approx(4.0)                               # frozen at open, unchanged during the bar

    r = RunBars("volume", init_T=1000.0, init_p_buy=0.5, init_v_buy=1.0, init_v_sell=1.0, span_bars=9,
                span_ticks=9, anchor="bars", jensen=False, clip=100.0)
    r.new_window(W)
    trades = [(1, 2.0), (-1, 3.0), (1, 1.0), (1, 4.0), (-1, 0.5)]
    for i, (sign, shares) in enumerate(trades, start=1):
        assert r.update(tr(i, sign, shares=shares)) is None
    assert r.P == pytest.approx(_ewma(0.5, [1.0 if s > 0 else 0.0 for s, _ in trades], a))
    assert r.v_buy == pytest.approx(_ewma(1.0, [v for s, v in trades if s > 0], a))     # buys only
    assert r.v_sell == pytest.approx(_ewma(1.0, [v for s, v in trades if s < 0], a))    # sells only
    assert r.v_buy == pytest.approx(1.728) and r.v_sell == pytest.approx(1.22)
    assert r.P == pytest.approx(0.53376)
    assert r.expected_threshold() == pytest.approx(r.E_T * max(r.P * r.v_buy, (1 - r.P) * r.v_sell))


@pytest.mark.case
def test_the_cusum_filter_fires_strictly_above_h_where_the_book_fires_at_h():
    """2.5.2(1): the book's S_t >= h is inclusive. This implementation is strict, so a run that reaches exactly h
    does not sample. The deviation is recorded in docs/claims/ch02.md; this pins both sides of the boundary so it
    cannot drift unnoticed in either direction."""
    assert labels.cusum_filter([0.5, 0.5], 1.0).tolist() == [0, 0]            # exactly h: the book would fire here
    assert labels.cusum_filter([0.5, 0.5001], 1.0).tolist() == [0, 1]         # a hair above h: this one fires
    assert labels.cusum_filter([-0.5, -0.5], 1.0).tolist() == [0, 0]
    assert labels.cusum_filter([-0.5, -0.5001], 1.0).tolist() == [0, -1]
    # and the book's own recursion either way: the side that fires resets, the other keeps its running sum
    assert labels.cusum_filter([1.5, -0.4, -1.2], 1.0).tolist() == [1, 0, -1]


@pytest.mark.case
def test_information_driven_bars_sample_more_often_when_the_flow_is_one_sided():
    """2.3.2(1) and 2.3.2(3): imbalance bars and run bars are meant to appear more frequently under informed trading
    (a persistent side) and under order slicing (long same-side runs). Two tapes of identical length and identical
    trade sizes, one with balanced signs and one one-sided, must give strictly more bars on the one-sided tape."""
    rng = np.random.default_rng(7)
    n = 3000
    balanced = rng.choice([-1, 1], n)
    one_sided = np.where(rng.random(n) < 0.72, 1, -1)                  # informed: buys dominate
    sliced = np.repeat(rng.choice([-1, 1], n // 30), 30)               # a parent order sliced into 30 children
    counts = {}
    for name, signs in (("balanced", balanced), ("one_sided", one_sided), ("sliced", sliced)):
        for kind, build in (("imbalance", lambda: ImbalanceBars("tick", init_T=50.0, init_bv=0.1, init_bv_sd=1.0,
                                                                floor_sd=1.0, anchor="bars")),
                            ("run", lambda: RunBars("tick", init_T=50.0, init_p_buy=0.5, init_v_buy=1.0,
                                                    init_v_sell=1.0, anchor="bars"))):
            b = build()
            b.new_window(W)
            counts[name, kind] = sum(b.update(tr(i, int(s))) is not None for i, s in enumerate(signs, start=1))
    assert counts["one_sided", "imbalance"] > counts["balanced", "imbalance"]
    assert counts["sliced", "run"] > counts["balanced", "run"]


@pytest.mark.case
def test_the_bar_study_settles_the_books_comparative_claims_on_this_market():
    """2.3.1(1)-2.3.1(4): the book's comparisons between bar families are empirical claims, and research/bar_study.md
    measured them here. Three of them hold on this market and one does not; both are pinned, because an item whose
    only evidence is that a script's docstring mentions the exercise cannot fail."""
    study = json.loads((ROOT / "research" / "bar_study.json").read_text())
    by = {(c["kind"], c["size"]): c for c in study["configs"]}
    sizes = (10, 30, 100)
    # 2.3.1(2): tick bars' returns are closer to normal than time bars' - every size, a wide margin
    for s in sizes:
        assert by["tick", s]["excess_kurtosis"] < by["time", s]["excess_kurtosis"] * 0.8
    # 2.3.1(3): volume bars improve on tick bars - true at the two smaller sizes, not at 100
    assert by["volume", 10]["excess_kurtosis"] < by["tick", 10]["excess_kurtosis"]
    assert by["volume", 30]["excess_kurtosis"] < by["tick", 30]["excess_kurtosis"]
    assert by["volume", 100]["excess_kurtosis"] > by["tick", 100]["excess_kurtosis"]      # recorded deviation
    # every family's returns are non-normal at every size, which is the reason the chapter exists
    assert all(by[k, s]["jb_p"] < 0.01 and not by[k, s]["normal"] for k in ("time", "tick", "volume", "dollar")
               for s in sizes)
    # 2.3.1(4): dollar bars' count is steadier than tick bars' at the two smaller sizes, and steadier than volume
    # bars' only at the smallest; at 100 bars a window the book's ordering has gone - both are recorded deviations
    for s in (10, 30):
        assert by["dollar", s]["bars_per_window_cv"] < by["tick", s]["bars_per_window_cv"]
    assert by["dollar", 10]["bars_per_window_cv"] < by["volume", 10]["bars_per_window_cv"]
    assert by["dollar", 100]["bars_per_window_cv"] > by["tick", 100]["bars_per_window_cv"]     # recorded deviation
    assert by["dollar", 100]["bars_per_window_cv"] > by["volume", 100]["bars_per_window_cv"]   # recorded deviation
    # 2.3.1(1): the book's homoscedasticity argument against time bars is the one that fails here - time bars have
    # the most stable daily variance of any family, at every size
    for s in sizes:
        assert by["time", s]["var_of_daily_var_rel"] < min(by[k, s]["var_of_daily_var_rel"]
                                                           for k in ("tick", "volume", "dollar"))
    # 2.3.2(1): the tick rule is a classifier, not the truth - it recovers the venue's aggressor flag about 69% of
    # the time here, which is why the builders default to the flag and the rule has to be asked for
    assert 0.6 < study["tick_rule"]["accuracy"] < 0.75


@pytest.mark.case
def test_the_within_window_logit_return_is_the_projects_stand_in_for_the_books_rolled_series():
    """2.4.3: the book builds a non-negative rolled series by dividing each rolled price change by the previous raw
    price and compounding. There is no roll here - a token expires and a new one starts - so the analogue is the
    logit change within one window, which never crosses a window boundary. Pinned by hand, with the boundary case."""
    bars = pd.DataFrame({"window": [1, 1, 1, 2, 2], "close": [0.4, 0.5, 0.6, 0.9, 0.2]})
    got = bars_mod.logit_returns(bars, clip=0.005)
    lg = np.log(np.array([0.4, 0.5, 0.6, 0.9, 0.2]) / (1 - np.array([0.4, 0.5, 0.6, 0.9, 0.2])))
    assert got.index.tolist() == [1, 2, 4]                       # one row per window is lost, never a cross-window one
    assert got.to_numpy() == pytest.approx([lg[1] - lg[0], lg[2] - lg[1], lg[4] - lg[3]])
    assert got.iloc[0] == pytest.approx(0.405465, abs=1e-6)
    # the clip is what keeps the logit finite at a settled token, where the book's price ratio would not be
    settled = pd.DataFrame({"window": [1, 1], "close": [0.5, 0.0]})
    assert np.isfinite(bars_mod.logit_returns(settled, clip=0.005).to_numpy()).all()


# ------------------------------------------------ chapter 3 (the second reading, plan/book-v2.md node 15)


def _book_fixed_horizon(ret, tau):
    """3.2: y = -1 for r < -tau, 0 for |r| <= tau, +1 for r > tau - the neutral band includes its own boundary."""
    r = np.asarray(ret, dtype=float)
    return np.where(r > tau, 1, np.where(r < -tau, -1, 0))


@pytest.mark.case
def test_the_fixed_horizon_label_has_three_states_and_the_neutral_band_includes_its_boundary():
    """3.2: the chapter's central rule, and the illustration that condemns it. A return of exactly +-tau is labelled
    0, not +-1; and a constant tau of 1e-2 against bar returns of order 1e-4 labels almost everything 0, which is
    the failure the triple-barrier method exists to fix."""
    tau = 0.01
    assert _book_fixed_horizon([0.01, -0.01, 0.0], tau).tolist() == [0, 0, 0]        # the boundary is neutral
    assert _book_fixed_horizon([0.0100001, -0.0100001], tau).tolist() == [1, -1]
    # at tau = 0 the rule is the sign, which is what pmlab.labels.fixed_horizon computes on a logit path
    path = np.array([0.0, 0.2, -0.3, 0.0, 0.5])
    got, end = labels.fixed_horizon(path, np.array([1, 2, 3, 4]), horizon=1)
    ret = labels.path_return(path, np.array([1, 2, 3, 4]), end)
    assert got.tolist() == _book_fixed_horizon(ret, 0.0).astype(float).tolist()
    # and the collapse: returns drawn at a bar's own scale against a threshold set for a day's
    rng = np.random.default_rng(3)
    small = rng.normal(0, 1e-4, 5000)
    assert np.mean(_book_fixed_horizon(small, 1e-2) == 0) == 1.0                     # every label neutral
    assert np.mean(_book_fixed_horizon(small, 2e-5) == 0) < 0.25                     # a tau at the series' own scale


@pytest.mark.case
def test_the_barrier_width_is_each_events_own_target_times_the_factor():
    """3.4: the horizontal barriers are the factor times *that event's* trgt, so two events on the same path with
    different targets touch at different times. Every other barrier test uses one constant target, which cannot
    tell a per-event width from a global one."""
    times = np.arange(6.0)
    price = np.array([100.0, 100.4, 102.5, 103.2, 105.6, 107.0])
    touch = labeling.barrier_touches(times, price, t0=[0.0, 0.0], t1=None, target=[0.01, 0.03], pt=1.0, sl=1.0)
    assert touch["pt"].tolist() == [2.0, 3.0]          # +1% is first exceeded at 102.5, +3% at 103.2
    assert touch["sl"].isna().all()
    # the factor multiplies that same per-event width, so it moves the two events by different amounts
    wider = labeling.barrier_touches(times, price, t0=[0.0, 0.0], t1=None, target=[0.01, 0.03], pt=3.0, sl=1.0)
    assert wider["pt"][0] == 3.0 and np.isnan(wider["pt"][1])         # +3% at 103.2; +9% never reached


@pytest.mark.case
def test_the_stop_loss_can_be_disabled_leaving_only_the_profit_target_and_the_vertical():
    """3.4, configurations [1,0,1] and [1,0,0]: a zero in ptSl switches that barrier off. No other test in the
    suite passes sl = 0, so the whole stop-loss-disabled family was unexercised."""
    times = np.arange(7.0)
    price = np.array([100.0, 97.0, 96.0, 99.0, 103.0, 104.0, 99.0])   # dips 4% then rallies past +3%
    kw = dict(times=times, price=price, t0=[0.0], target=[0.02])
    both = labeling.barrier_touches(**kw, t1=[6.0], pt=1.0, sl=1.0)
    assert both["sl"][0] == 1.0 and both["pt"][0] == 4.0              # [1,1,1]: stopped out first, at -3%
    off = labeling.barrier_touches(**kw, t1=[6.0], pt=1.0, sl=0.0)
    assert np.isnan(off["sl"][0]) and off["pt"][0] == 4.0             # [1,0,1]: the dip is ignored
    none_v = labeling.barrier_touches(**kw, t1=None, pt=1.0, sl=0.0)
    assert np.isnan(none_v["sl"][0]) and none_v["pt"][0] == 4.0       # [1,0,0]: no vertical either
    # and with the profit target off instead, only the stop loss and the vertical remain ([0,1,1])
    only_sl = labeling.barrier_touches(**kw, t1=[6.0], pt=0.0, sl=1.0)
    assert only_sl["sl"][0] == 1.0 and np.isnan(only_sl["pt"][0])
    # a path with no touch at all runs to the vertical, which is where the label comes from
    flat = labeling.barrier_touches(times, price=np.full(7, 100.0), t0=[0.0], t1=[6.0], target=[0.02], pt=1.0, sl=1.0)
    assert flat[["sl", "pt"]].isna().all().all()


@pytest.mark.case
def test_figure_3_1_one_path_stopped_out_and_one_reaching_the_vertical_barrier():
    """3.4, Figure 3.1: the left panel's path touches a horizontal barrier first and is labelled by its sign; the
    right panel's reaches the vertical barrier first and is labelled by the return over the holding period."""
    times = np.arange(6.0)
    stopped = np.array([100.0, 99.0, 97.0, 98.0, 99.0, 100.0])        # -3% at t=2, never +2%
    expiring = np.array([100.0, 100.5, 100.2, 100.6, 100.4, 100.9])   # inside both barriers the whole way
    for price, want_t1, want_bin in ((stopped, 2.0, -1.0), (expiring, 5.0, 1.0)):
        ev = labeling.get_events(times, price, t_events=[0.0], target=np.full(6, 0.02), pt_sl=[1.0, 1.0], t1=[5.0])
        assert ev["t1"][0] == want_t1
        bins = labeling.get_bins(times, price, ev["t0"], ev["t1"])
        assert bins["bin"][0] == want_bin
        assert bins["ret"][0] == pytest.approx(price[int(want_t1)] / price[0] - 1.0)


@pytest.mark.case
def test_the_roc_of_the_primary_rule_trades_recall_against_the_false_positive_rate():
    """3.7 and Figure 3.2: precision, recall and the ROC are areas of one partition of the sample. Raising the
    decision threshold moves along the curve - fewer positives, lower recall and a lower false positive rate - and
    the four cells of the confusion matrix always partition the sample, which is what the figure draws."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    score = np.clip(0.5 + 0.22 * (2 * y - 1) + rng.normal(0, 0.25, 4000), 0, 1)
    fpr, tpr, thresholds = roc_curve(y, score)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)      # the curve is traced by the threshold
    assert tpr[0] == 0.0 and fpr[0] == 0.0 and tpr[-1] == 1.0 and fpr[-1] == 1.0
    area = np.trapezoid(tpr, fpr)
    assert 0.6 < area < 0.95                                            # a real but imperfect primary
    last_tpr = last_fpr = 1.0
    for cut in (0.3, 0.5, 0.7):
        pred = score > cut
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        assert tp + fn == int(np.sum(y == 1)) and fp + tn == int(np.sum(y == 0))    # the rectangle of Figure 3.2
        assert tp + fp + fn + tn == len(y)
        this_tpr, this_fpr = tp / (tp + fn), fp / (fp + tn)
        assert this_tpr < last_tpr and this_fpr < last_fpr              # a higher bar costs recall and buys FPR
        assert recall_score(y, pred) == pytest.approx(this_tpr)
        assert precision_score(y, pred) == pytest.approx(tp / (tp + fp))
        last_tpr, last_fpr = this_tpr, this_fpr


# ------------------------------------------------ chapter 4 (the second reading, plan/book-v2.md node 15)

T0_BOOK, T1_BOOK = np.array([0, 2, 4]), np.array([3, 4, 6])       # the 4.5.3 example, as half-open spans


@pytest.mark.case
def test_snippet_4_6_draws_a_uniform_sample_and_a_sequential_one_on_the_books_matrix():
    """4.5.3, Snippet 4.6: the book's main() builds the three-label indicator matrix, draws a uniform sample of
    three columns and a sequential sample of three, and prints the mean average uniqueness of each. The uniform
    case has only 27 outcomes, so its expectation is exact: 52/81."""
    outcomes = [sp.bootstrap_uniqueness(T0_BOOK, T1_BOOK, list(s))
                for s in itertools.product(range(3), repeat=3)]
    assert len(outcomes) == 27
    assert float(np.mean(outcomes)) == pytest.approx(52 / 81)
    assert max(outcomes) == pytest.approx(31 / 36) and min(outcomes) == pytest.approx(1 / 3)
    seq = [sp.bootstrap_uniqueness(T0_BOOK, T1_BOOK, sp.seq_bootstrap(T0_BOOK, T1_BOOK, rng=np.random.default_rng(s)))
           for s in range(400)]
    assert np.mean(seq) > 52 / 81 + 0.03                       # the sequential draw is the more unique one
    assert np.mean(seq) == pytest.approx(0.706, abs=0.02)
    # and it is the repeats the sequential draw avoids: a uniform sample repeats far more often
    uniform_repeats = np.mean([len(set(s)) < 3 for s in itertools.product(range(3), repeat=3)])
    seq_repeats = np.mean([len(set(sp.seq_bootstrap(T0_BOOK, T1_BOOK, rng=np.random.default_rng(s)))) < 3
                           for s in range(400)])
    assert seq_repeats < uniform_repeats


@pytest.mark.case
def test_the_time_decay_domain_is_minus_one_to_one_and_outside_it_the_weights_stop_meaning_anything():
    """4.7: the book defines the decay only for c in (-1, 1]. Outside it the formula still returns numbers, and
    they are not decay weights: at c = 2 the oldest label gets the largest weight, c = -2 returns exactly the same
    thing, and c = -1 divides by zero. Pinned so the undefined region cannot be mistaken for a feature."""
    u = np.array([0.5, 0.5, 1.0, 1.0, 1.0])
    above = sp.time_decay(u, c=2.0)
    assert above.tolist() == pytest.approx([1.875, 1.75, 1.5, 1.25, 1.0])
    assert np.all(np.diff(above) < 0) and above[0] > above[-1]     # older labels weigh more: the rule inverted
    assert sp.time_decay(u, c=-2.0).tolist() == pytest.approx(above.tolist())   # and c and -c cannot be told apart
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.all(np.isnan(sp.time_decay(u, c=-1.0)))          # the boundary the domain excludes
    assert sp.time_decay(u, c=-0.999).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0], abs=2e-3)


@pytest.mark.case
def test_random_spans_are_the_books_uniform_draw_apart_from_a_recorded_floor():
    """4.5.4: the book draws t0 uniformly over the bars and the number of bars held uniformly up to maxH. This
    generator never produces a span of one unit, so the holding length is uniform on [2, max_h] rather than the
    book's [1, max_h]. It feeds the 0.6/0.7 Monte Carlo, so the deviation is pinned rather than left to drift."""
    t0, t1 = sp.random_spans(200_000, 100, 5, np.random.default_rng(1))
    assert t0.min() == 0 and t0.max() == 99                        # t0 uniform on [0, n_units)
    assert np.mean(t0) == pytest.approx(49.5, abs=0.3)
    length = t1 - t0
    assert length.min() == 2 and length.max() == 5                 # the book's would start at 1
    assert np.mean(length) == pytest.approx(3.5, abs=0.02)
    counts = np.bincount(length)[2:6] / len(length)
    assert counts == pytest.approx([0.25] * 4, abs=0.01)           # uniform over the four lengths it does draw


@pytest.mark.eval
def test_balanced_class_weights_lift_recall_on_the_rare_label():
    """4.8: without class weights the classifier maximises accuracy on the common label and treats the rare one as
    an outlier. Threshold: an 80/20 split with a weak feature - the unweighted forest's recall on the rare class is
    below 0.25 and 'balanced' lifts it above 0.5, while both keep the common class well above chance."""
    rng = np.random.default_rng(4)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)
    X = np.column_stack([y * 0.7 + rng.normal(0, 1.0, n), rng.normal(0, 1, n)])
    cut = int(0.6 * n)
    fit = dict(n_estimators=60, max_depth=4, min_samples_leaf=20, random_state=0)
    plain = RandomForestClassifier(**fit).fit(X[:cut], y[:cut])
    weighted = RandomForestClassifier(class_weight="balanced", **fit).fit(X[:cut], y[:cut])
    rare = y[cut:] == 1
    plain_recall = recall_score(y[cut:], plain.predict(X[cut:]))
    weighted_recall = recall_score(y[cut:], weighted.predict(X[cut:]))
    assert rare.mean() == pytest.approx(0.2, abs=0.04)
    assert plain_recall < 0.25 and weighted_recall > 0.5
    assert recall_score(y[cut:], plain.predict(X[cut:]), pos_label=0) > 0.9
    # the trade the book describes: the rare class is recovered at the cost of accuracy on the common one
    assert recall_score(y[cut:], weighted.predict(X[cut:]), pos_label=0) < \
        recall_score(y[cut:], plain.predict(X[cut:]), pos_label=0)


# ------------------------------------------------ chapter 5 (the second reading, plan/book-v2.md node 15)

WALK = np.cumsum(np.random.default_rng(4).normal(size=3000)) + 100.0


@pytest.mark.case
def test_the_minimum_d_keeps_almost_all_the_memory_it_pays_for():
    """5.6 and Figure 5.5: the chapter's whole point is that stationarity is bought at almost no cost in memory.
    On the book's own picture the ADF statistic crosses the 95% line at a small d while the correlation with the
    undifferenced series is still about 0.995, and full differencing destroys it. Asserting 0 < d* <= 1 admits
    d* = 1, which is exactly the over-differencing the chapter exists to avoid."""
    res = fd.min_ffd(WALK, ds=np.linspace(0, 1, 11), thres=1e-3)
    t = {r["d"]: r for r in res["table"]}
    assert res["min_d"] == pytest.approx(0.1) and res["min_d"] < 0.5
    assert res["corr_at_min_d"] > 0.99 and t[res["min_d"]]["corr"] == pytest.approx(res["corr_at_min_d"])
    assert t[1.0]["corr"] < 0.06                                  # the first difference keeps almost no memory
    assert t[res["min_d"]]["corr"] > 20 * t[1.0]["corr"]
    assert t[0.5]["corr"] == pytest.approx(0.723, abs=0.01)       # and the loss is monotone in d
    corrs = [t[d]["corr"] for d in sorted(t)]
    assert np.all(np.diff(corrs) < 0)
    # the statistic goes the other way: more differencing, more stationarity
    adfs = [t[d]["adf"] for d in sorted(t)]
    assert np.all(np.diff(adfs) < 0) and t[0.0]["adf"] > t[0.0]["crit5"]


@pytest.mark.case
def test_a_series_that_is_already_stationary_needs_no_differencing_and_an_explosive_one_needs_more_than_one():
    """5.6: "if the series is already stationary, d* = 0"; and a series can need d* > 1, which is why the search
    is a search. Neither case is reachable from the random walk every other fracdiff test uses."""
    white = np.random.default_rng(9).normal(size=3000)
    assert fd.min_ffd(white, thres=1e-3)["min_d"] == 0.0
    rng = np.random.default_rng(6)
    explosive = np.zeros(1000)
    for i in range(1, 1000):
        explosive[i] = 1.01 * explosive[i - 1] + rng.normal()     # an AR(1) root above one
    assert fd.min_ffd(explosive, ds=np.linspace(0, 1, 11), thres=1e-3)["min_d"] is None   # the default grid cannot
    assert fd.min_ffd(explosive, ds=np.linspace(0, 2, 21), thres=1e-3)["min_d"] == pytest.approx(1.1)


@pytest.mark.case
def test_the_expanding_window_drifts_downward_where_the_fixed_width_window_does_not():
    """5.5.1 and 5.5.2, Figures 5.3 and 5.4: the expanding window's weights keep growing in number, so the sum of
    the weights applied to a level keeps falling and the output drifts toward zero even when the series is flat.
    The fixed-width window applies the same weights to every point, so a flat series maps to a constant."""
    flat = np.full(400, 100.0)
    expanding = fd.frac_diff_expanding(flat, 0.4, thres=1.0)      # tau = 1: no weight-loss control at all
    assert expanding[:5].tolist() == pytest.approx([100.0, 60.0, 48.0, 41.6, 37.44])
    assert expanding.tolist() == pytest.approx((100.0 * np.cumsum(fd.expanding_weights(0.4, 400))).tolist())
    assert np.all(np.diff(expanding) < 0) and expanding[300] == pytest.approx(6.855, abs=1e-3)
    fixed = fd.frac_diff_ffd(flat, 0.4, thres=1e-4)
    finite = fixed[np.isfinite(fixed)]
    assert len(np.unique(np.round(finite, 9))) == 1 and finite[0] == pytest.approx(7.0369, abs=1e-4)
    # on a real walk the same contrast shows as a slope: the expanding series still trends, the FFD one does not
    ew = fd.frac_diff_expanding(WALK, 0.4, thres=1e-2)
    ff = fd.frac_diff_ffd(WALK, 0.4, thres=1e-2)
    slope = lambda v: float(np.polyfit(np.arange(len(v[np.isfinite(v)])), v[np.isfinite(v)], 1)[0])
    assert abs(slope(ew)) > 4 * abs(slope(ff))


@pytest.mark.case
def test_the_fixed_width_series_stays_correlated_and_non_normal_as_the_chapter_says():
    """5.5.2: the FFD series keeps memory, so it is strongly serially correlated, and it is not Gaussian - the two
    properties that separate it from the first difference the literature reaches for."""
    v = fd.frac_diff_ffd(WALK, 0.4, thres=1e-4)
    v = v[np.isfinite(v)]
    assert np.corrcoef(v[1:], v[:-1])[0, 1] > 0.8                 # 0.851 here
    assert stats.jarque_bera(v).pvalue < 0.05                     # normality rejected
    first = np.diff(WALK)
    assert abs(np.corrcoef(first[1:], first[:-1])[0, 1]) < 0.1    # the first difference has none of that memory


@pytest.mark.case
def test_a_missing_value_propagates_across_exactly_the_window_width():
    """5.5.2: the book's snippet forward-fills and skips NaNs; this implementation convolves instead, so one
    missing value blanks exactly the window's width of outputs. Recorded as a deviation, and pinned here so the
    reach of a gap cannot change unnoticed."""
    width = len(fd.ffd_weights(0.4, 1e-2))
    assert width == 11
    y = WALK.copy()
    y[100] = np.nan
    out = fd.frac_diff_ffd(y, 0.4, thres=1e-2)
    missing = np.flatnonzero(~np.isfinite(out))
    assert missing[:width - 1].tolist() == list(range(width - 1))          # the warm-up, as for a clean series
    assert missing[width - 1:].tolist() == list(range(100, 100 + width))   # and exactly `width` more
    assert len(missing) == (width - 1) + width


@pytest.mark.case
def test_the_p_value_rule_and_the_critical_value_rule_pick_the_same_minimum_d():
    """5.6 and Table 5.1: the book's step three is "the smallest d whose test rejects at 95%", which can be read
    off either the statistic against the critical value or the p-value against 0.05. min_ffd uses the first; this
    checks the second gives the same answer, so the item's wording and the code do not quietly disagree."""
    for series in (WALK, np.random.default_rng(9).normal(size=2000),
                   np.cumsum(np.random.default_rng(12).normal(size=1500)) * 0.3 + 10.0):
        res = fd.min_ffd(series, ds=np.linspace(0, 1, 11), thres=1e-3)
        rows = sorted(res["table"], key=lambda r: r["d"])
        by_stat = next((r["d"] for r in rows if r["adf"] < r["crit5"]), None)
        by_pvalue = next((r["d"] for r in rows if r["pvalue"] < 0.05), None)
        assert by_stat == by_pvalue == res["min_d"]
        assert all(0.0 <= r["pvalue"] <= 1.0 and r["nobs"] > 0 for r in rows)
