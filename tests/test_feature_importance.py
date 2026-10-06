"""Leak guards and hand cases for feature importance on the event dataset (pmlab.afml.event_importance, plan
afml-pipeline stage 3): fold construction, weights and tuning from training rows only, the PCA check, the weighted
Kendall tau and the label-permutation control.

Every guard here was mutation-checked: `scripts/feature_importance.py --mutations` applies each recorded mutation to a
copy of the library in a private temporary directory, runs this file against it and requires a failure
(research/feature_importance_mutations.json).
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, weightedtau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import event_importance as ei
from pmlab.afml import events as ev
from pmlab.evaluation import CONFIRM_START

DAY0 = (CONFIRM_START - 200 * 86400) // 86400 * 86400          # a UTC midnight well before the confirmation start
TAGGED = ei.ACTIVITY + ei.CLOCK + ("f_rule_L", "f_roll")


def make_frame(rng, n_days=12, windows=24, rows=4, informative=False, side=None):
    """Event-like rows: whole windows inside UTC days, spans to the window end, every stage-3 feature as noise. With
    `informative`, the window-constant f_fair sets the chance that Up wins (0.1 or 0.9) and the activity and clock
    features are window-constant, as they nearly are in the data, so a forest can memorise a window when its rows sit on
    both sides of a split."""
    recs, paths = [], {}
    for d in range(n_days):
        for w in range(windows):
            start = DAY0 + d * 86400 + w * 3600
            q0 = rng.choice([0.1, 0.9]) if informative else 0.5
            up = float(rng.random() < q0)
            ts = np.sort(rng.choice(np.arange(ev.T_MIN, ev.T_MAX + 1), rows, replace=False))
            q = np.clip(q0 + np.cumsum(rng.normal(0, 0.01, T)), 0.01, 0.99)
            paths[start] = np.concatenate([q, [up]])
            tag = rng.normal(size=len(TAGGED))
            for t in ts:
                s = int(side if side is not None else rng.choice([1, -1]))
                recs.append({"start": start, "t": int(t), "day": pd.Timestamp(start, unit="s").strftime("%Y-%m-%d"),
                             "side": s, "t0": start + int(t), "t1": start + T, "up_won": up,
                             "label": int(up if s > 0 else 1 - up), "_q0": q0, **dict(zip(TAGGED, tag))})
    df = pd.DataFrame(recs)
    X = pd.DataFrame(rng.normal(size=(len(df), len(ei.FEATURES))).astype(np.float32), columns=ei.FEATURES)
    if informative:
        X["f_fair"] = df["_q0"].astype(np.float32)
        for c in TAGGED:
            X[c] = df[c].astype(np.float32)
    return pd.concat([df.drop(columns=["_q0", *TAGGED]), X], axis=1), paths


@pytest.fixture(scope="module")
def frame():
    return make_frame(np.random.default_rng(3))


# ---------------------------------------------------------------------------------------------------- folds

@pytest.mark.case
def test_day_folds_test_whole_contiguous_days_and_embargo_the_days_after_each_test_block(frame):
    df, _ = frame
    days = sorted(df["day"].unique())
    e = ei.EMBARGO_S // 86400
    seen = []
    for tr, te in ei.day_folds(df, k=5):                     # 288 windows in 5 blocks: grouping by window would cut days
        test_days = sorted(df["day"].iloc[te].unique())
        assert set(np.flatnonzero(df["day"].isin(test_days))) == set(te)          # whole days
        idx = [days.index(d) for d in test_days]
        assert idx == list(range(idx[0], idx[-1] + 1))                              # one contiguous block
        train_days = set(df["day"].iloc[tr])
        assert not train_days & set(test_days)
        end = df["t1"].iloc[te].max()
        t0 = df["t0"].iloc[tr]
        assert not ((t0 >= end) & (t0 < end + ei.EMBARGO_S)).any()                  # embargo after the block
        after = [days[i] for i in range(idx[-1] + 1, min(idx[-1] + 1 + e, len(days)))]
        assert not train_days & set(after)
        if idx[-1] + 1 + e < len(days):
            assert days[idx[-1] + 1 + e] in train_days                              # and no more than that
        if idx[0] > 0:
            assert days[idx[0] - 1] in train_days                                   # no embargo before a block
        assert set(tr) | set(te) | set(np.flatnonzero(df["day"].isin(after))) == set(range(len(df)))
        seen += test_days
    assert sorted(seen) == days


@pytest.mark.case
def test_no_training_day_has_fair_parameters_fitted_on_its_test_block():
    """A day's fair parameters come from the settled windows of the FAIR_TRAIN_DAYS days before it
    (events.fair_training_starts). No training day of a fold may have one of those windows inside the fold's test block.
    A 1-day embargo breaks this for the two days after it; the derived embargo does not."""
    df, _ = make_frame(np.random.default_rng(12), n_days=30, windows=24, rows=2)
    inventory = df.drop_duplicates("start")[["start"]].assign(L=60).reset_index(drop=True)
    day0 = {d: int(pd.Timestamp(d, tz="UTC").timestamp()) for d in df["day"].unique()}

    def contaminated(folds):
        bad = []
        for tr, te in folds:
            test_days = df["day"].iloc[te].unique()
            lo, hi = min(day0[d] for d in test_days), max(day0[d] for d in test_days) + 86400
            for d in df["day"].iloc[tr].unique():
                fit = np.asarray(ev.fair_training_starts(inventory, day0[d], 60), dtype=np.int64)
                if ((fit + T > lo) & (fit < hi)).any():
                    bad.append(d)
        return bad

    assert ei.EMBARGO_S >= (ev.FAIR_TRAIN_DAYS + 1) * 86400
    assert contaminated(ei.day_folds(df, k=5)) == []
    assert len(contaminated(ei.day_folds(df, k=5, embargo=86400))) >= 2 * 4        # two days after each inner block


# ---------------------------------------------------------------------------------------------------- inputs

@pytest.mark.case
def test_the_design_matrix_refuses_outcome_span_weight_bet_and_book_columns(frame):
    df, _ = frame
    X = ei.design(df, ei.FEATURES)
    assert X.shape == (len(df), len(ei.FEATURES)) and X.dtype == np.float32
    for col in ["label", "meta_label", "up_won", "payoff", "pnl", "ret", "avg_uniqueness", "w_return_raw", "t0", "t1",
                "side", "price", "cost", "edge", "fair_side", "start"]:
        with pytest.raises(ValueError):
            ei.check_inputs(df, ei.FEATURES[:3] + [col])
    with pytest.raises(ValueError):
        ei.check_inputs(df, ["f_book_spread"])
    assert not any(c.startswith("f_book") for c in ei.FEATURES)
    assert set(ei.BASELINE) | set(ei.CLOCK) | set(ei.ACTIVITY) <= set(ei.FEATURES)


@pytest.mark.case
def test_rows_of_a_window_ending_after_the_confirmation_start_are_refused(frame):
    df, _ = frame
    ok = df.iloc[:2].copy()
    ok["start"] = CONFIRM_START - T                                                  # ends exactly at the start
    ei.check_inputs(ok, ei.FEATURES)
    late = ok.copy()
    late.loc[late.index[-1], "start"] = CONFIRM_START - T + 1
    with pytest.raises(ValueError):
        ei.check_inputs(late, ei.FEATURES)
    with pytest.raises(ValueError):
        ei.design(late, ei.FEATURES)


# ---------------------------------------------------------------------------------------------------- weights

@pytest.mark.case
def test_fold_weights_come_from_each_sides_own_rows_only():
    df, paths = make_frame(np.random.default_rng(11), n_days=3, windows=6, rows=8)
    rng = np.random.default_rng(0)
    train = np.sort(rng.choice(len(df), len(df) // 2, replace=False))                # rows of one window on both sides
    test = np.setdiff1d(np.arange(len(df)), train)
    w = ei.fold_weights(df, paths, train, test)
    own_tr = ev.sample_weights(df.iloc[train].reset_index(drop=True), paths)["avg_uniqueness"].to_numpy()
    own_te = ev.sample_weights(df.iloc[test].reset_index(drop=True), paths)["avg_uniqueness"].to_numpy()
    assert np.allclose(w["fit"], own_tr) and np.allclose(w["score"], own_te)
    assert w["max_samples"] == pytest.approx(own_tr.mean())

    moved = df.copy()                                                                # test spans change, train rows do not
    moved.loc[test, "t0"] = moved.loc[test, "start"] + ev.T_MIN
    moved.loc[test, "t"] = ev.T_MIN
    all_rows = ev.sample_weights(df, paths)["avg_uniqueness"].to_numpy()[train]
    all_moved = ev.sample_weights(moved, paths)["avg_uniqueness"].to_numpy()[train]
    assert not np.allclose(all_rows, all_moved)                                      # the perturbation is visible
    assert np.allclose(ei.fold_weights(moved, paths, train, test)["fit"], w["fit"])
    moved_tr = df.copy()
    moved_tr.loc[train, "t0"] = moved_tr.loc[train, "start"] + ev.T_MIN
    assert np.allclose(ei.fold_weights(moved_tr, paths, train, test)["score"], w["score"])


@pytest.mark.case
def test_a_fold_is_fitted_on_its_training_rows_only(frame):
    df, _ = frame
    rng = np.random.default_rng(1)
    X = rng.normal(size=(len(df), 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=len(df)) > 0).astype(int)
    tr, te = ei.day_folds(df, k=3)[1]
    w = np.ones(len(tr))
    a = ei.fold_losses(ei.forest(0.5, 1, 20, 0.01, 0), X, y, tr, te, w)
    flipped = y.copy()
    flipped[te] = 1 - flipped[te]
    b = ei.fold_losses(ei.forest(0.5, 1, 20, 0.01, 0), X, flipped, tr, te, w)
    assert np.allclose(a["model"].predict_proba(X[te]), b["model"].predict_proba(X[te]))
    assert not np.allclose(a["base"], b["base"])                                     # the test labels did reach the score


@pytest.mark.case
def test_each_tree_draws_max_samples_rows_whatever_the_scale_of_the_uniqueness_weights():
    """scikit-learn draws max_samples x sum(sample weights) rows per tree: with raw uniqueness 0.15 on 20,000 rows a
    tree would see about 450 rows instead of 3,000. fold_losses scales fit weights to mean 1."""
    rng = np.random.default_rng(13)
    n, u = 20000, 0.15
    X = rng.normal(size=(n, 2)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(int)
    train, test = np.arange(n), np.arange(100)
    expected = n * (1 - np.exp(-u))                                                   # distinct rows in u * n draws
    for scale in (u, 1.0, 7.0):
        w = np.full(n, scale) * rng.uniform(0.5, 1.5, n)
        r = ei.fold_losses(ei.forest(u, 1, 12, 0.01, 0), X, y, train, test, w)
        distinct = np.mean([len(np.unique(ix)) for ix in r["model"].estimators_samples_])
        assert distinct == pytest.approx(expected, rel=0.05)


@pytest.mark.case
def test_the_leaf_size_is_tuned_inside_the_training_rows_only():
    df, paths = make_frame(np.random.default_rng(5), n_days=12, windows=12, rows=3)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(len(df), 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=len(df)) > 0).astype(int)
    tr, te = ei.day_folds(df, k=4)[1]
    grid = (0.001, 0.05)
    a = ei.inner_leaf(df, paths, X, y, tr, grid, k=3, n_estimators=10, max_features=1, seed=0)
    flipped = y.copy()
    flipped[te] = 1 - flipped[te]
    Xm = X.copy()
    Xm[te] = 0.0
    b = ei.inner_leaf(df, paths, Xm, flipped, tr, grid, k=3, n_estimators=10, max_features=1, seed=0)
    assert a == b
    assert a["leaf"] in grid and set(a["scores"]) == set(grid)


# ---------------------------------------------------------------------------------------------------- PCA and tau

@pytest.mark.case
def test_pca_is_fitted_on_training_rows_only_and_gives_orthogonal_components():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(500, 3))
    X = np.column_stack([base, base[:, 0] + 0.1 * rng.normal(size=500), base[:, 1] * 2 + 1, rng.normal(size=500)])
    X[5, 2] = np.nan
    train, test = np.arange(400), np.arange(400, 500)
    fit, P_tr, P_te = ei.pca_fold(X, train, test)
    garbage = X.copy()
    garbage[test] = 1e3 * rng.normal(size=(100, X.shape[1]))
    fit2, P_tr2, _ = ei.pca_fold(garbage, train, test)
    assert np.allclose(fit["eigenvalues"], fit2["eigenvalues"]) and np.allclose(P_tr, P_tr2)
    assert fit["median"][2] == pytest.approx(np.nanmedian(X[train, 2]))
    cov = P_tr.astype(float).T @ P_tr.astype(float) / len(train)
    assert np.allclose(cov, np.diag(fit["eigenvalues"]), atol=1e-4)
    assert np.all(np.diff(fit["eigenvalues"]) <= 1e-9) and fit["explained"] >= 0.95
    assert P_te.shape == (100, len(fit["eigenvalues"]))


@pytest.mark.case
def test_weighted_tau_rewards_importance_that_follows_the_eigenvalue_rank():
    rank = np.arange(1, 9)
    assert ei.weighted_tau(9 - rank, rank) == pytest.approx(1.0)
    assert ei.weighted_tau(rank, rank) == pytest.approx(-1.0)
    top_right = np.array([8, 7, 6, 5, 1, 4, 2, 3], dtype=float)                   # top four in PCA order, tail scrambled
    top_wrong = np.array([5, 1, 4, 2, 8, 7, 6, 3], dtype=float)                   # tail in order, top scrambled
    assert ei.weighted_tau(top_right, rank) > kendalltau(top_right, -rank)[0]
    assert ei.weighted_tau(top_right, rank) > ei.weighted_tau(top_wrong, rank)
    assert ei.weighted_tau(top_right, rank) == pytest.approx(weightedtau(top_right, 1.0 / rank)[0])


# ---------------------------------------------------------------------------------------------------- control

@pytest.mark.case
def test_outcome_permutation_moves_whole_windows():
    df, _ = make_frame(np.random.default_rng(8), n_days=20, windows=24, rows=5)
    lab = ei.permute_outcomes(df, seed=1)
    up = np.where(df["side"] > 0, lab, 1 - lab)                                      # the shuffled outcome per row
    per_window = pd.Series(up).groupby(df["start"].to_numpy()).nunique()
    assert (per_window == 1).all()                                                   # one outcome per window
    real = df.groupby("start")["up_won"].first()
    shuffled = pd.Series(up).groupby(df["start"].to_numpy()).first()
    assert sorted(real) == sorted(shuffled)                                          # the same outcomes, moved
    assert abs(np.corrcoef(real, shuffled.loc[real.index])[0, 1]) < 0.15
    assert (ei.permute_outcomes(df, seed=1) == lab).all()


def _gain(df, paths, y, folds, X):
    """Out-of-fold weighted log loss against ln 2, millinats, pooled over folds."""
    loss, weight = [], []
    for tr, te in folds:
        w = ei.fold_weights(df, paths, tr, te)
        r = ei.fold_losses(ei.forest(w["max_samples"], "sqrt", 40, 0.0, 0), X, y, tr, te, w["fit"])
        loss.append(r["base"])
        weight.append(w["score"])
    return 1000 * (np.log(2) - ei.weighted_mean(np.concatenate(loss), np.concatenate(weight)))


@pytest.mark.eval
def test_label_permutation_control_scores_at_chance_under_day_folds_and_catches_a_leaky_split():
    """Thresholds: informative labels gain > 100 millinats over ln 2; the window-permuted control gains < 5 under purged
    day folds; under row-level folds (windows on both sides of the split) the same control gains > 10 millinats more.
    max_samples = mean uniqueness already limits how much a forest can memorise a window (7.3), so the leak shows as
    about +20 millinats here rather than as a large gain."""
    df, paths = make_frame(np.random.default_rng(9), n_days=40, windows=24, rows=6, informative=True, side=1)
    X = ei.design(df, ei.FEATURES)
    real = df["label"].to_numpy()
    control = ei.permute_outcomes(df, seed=4)
    day = ei.day_folds(df, k=5)
    assert _gain(df, paths, real, day, X) > 100
    at_chance = _gain(df, paths, control, day, X)
    assert at_chance < 5
    leaky = [(tr, te) for tr, te in KFold(5, shuffle=True, random_state=0).split(X)]
    assert _gain(df, paths, control, leaky, X) > at_chance + 10


# ---------------------------------------------------------------------------------------------------- features

@pytest.mark.case
def test_side_frame_hand_case():
    row = {c: 0.0 for c in ei.FEATURES}
    row.update(f_btc_ret_10=0.002, f_z_q=1.5, f_fair=0.8, f_buy_share_60=0.7, f_mid_minus_q=-0.03, f_tick_imb=0.4,
               f_cusum_sign=1.0, f_rv_60=3e-5, f_tau=120.0, f_tick_kyle=0.01, f_spread_proxy=0.02)
    df = pd.DataFrame([dict(row, side=1), dict(row, side=-1)])
    o = ei.orient(df)
    up, dn = o.iloc[0], o.iloc[1]
    for c in ("f_btc_ret_10", "f_z_q", "f_fair", "f_buy_share_60", "f_mid_minus_q", "f_tick_imb", "f_rv_60", "f_tau"):
        assert up[c] == pytest.approx(row[c])
    assert (dn["f_btc_ret_10"], dn["f_z_q"], dn["f_mid_minus_q"], dn["f_tick_imb"], dn["f_cusum_sign"]) == \
        pytest.approx((-0.002, -1.5, 0.03, -0.4, -1.0))
    assert (dn["f_fair"], dn["f_buy_share_60"]) == pytest.approx((0.2, 0.3))
    assert (dn["f_rv_60"], dn["f_tau"], dn["f_tick_kyle"], dn["f_spread_proxy"]) == pytest.approx((3e-5, 120.0, 0.01, 0.02))
    assert set(ei.SIGNED) | set(ei.SHARE) <= set(ei.FEATURES)
    with pytest.raises(ValueError):
        ei.orient(pd.DataFrame([dict(row, side=0)]))


@pytest.mark.case
def test_binance_trades_read_only_klines_closed_by_the_event():
    sec = np.arange(1000, 1400)
    n = np.random.default_rng(0).integers(0, 50, len(sec))
    e = np.array([1100, 1350])
    got = ei.btc_trades(sec, n, e, 60)
    assert got == pytest.approx(np.log1p([n[40:100].sum(), n[290:350].sum()]))     # opened 1040..1099, 1290..1349
    later = n.copy()
    later[100:] += 1000                                                              # klines opened at or after 1100
    assert ei.btc_trades(sec, later, e[:1], 60) == pytest.approx(got[:1])
    earlier = n.copy()
    earlier[99] += 1000                                                              # the kline that closed at 1100
    assert ei.btc_trades(sec, earlier, e[:1], 60)[0] > got[0]
    gap = np.delete(np.arange(len(sec)), 99)
    assert np.isnan(ei.btc_trades(sec[gap], n[gap], e[:1], 60)[0])


@pytest.mark.case
def test_subsample_caps_rows_and_ignores_their_content(frame):
    df, _ = frame
    a = ei.subsample(df, 50, "2026-06-01|q_taker|50")
    assert len(a) == 50 and a.index.is_monotonic_increasing
    changed = df.copy()
    changed["label"] = 0                                                              # every outcome column changed
    changed["up_won"] = 1.0
    changed["payoff"] = 0.5
    assert (ei.subsample(changed, 50, "2026-06-01|q_taker|50").index == a.index).all()
    assert not (ei.subsample(df, 50, "2026-06-02|q_taker|50").index == a.index).all()
    assert len(ei.subsample(df.iloc[:10], 50, "x")) == 10


@pytest.mark.case
def test_row_losses_are_sklearns_weighted_log_loss_and_permutation_moves_group_rows_together():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int)
    m = LogisticRegression().fit(X[:, :1], y)
    w = rng.random(400)
    r = ei.row_losses(m, X[:, :1], y)
    assert ei.weighted_mean(r, w) == pytest.approx(log_loss(y, m.predict_proba(X[:, :1]), sample_weight=w))

    class Probe:                                                                      # records what the model is shown
        classes_ = np.array([0, 1])
        seen = []

        def predict_proba(self, Z):
            Probe.seen.append(Z.copy())
            return np.column_stack([np.full(len(Z), 0.5), np.full(len(Z), 0.5)])

    out = ei.permuted_losses(Probe(), X, y, {"g": [1, 2], "h": [0]}, rng=0)
    Zg, Zh = Probe.seen
    assert np.allclose(Zg[:, 0], X[:, 0]) and not np.allclose(Zg[:, 1], X[:, 1])
    order = [np.flatnonzero((X[:, 1] == v))[0] for v in Zg[:, 1]]
    assert np.allclose(Zg[:, 2], X[order, 2])                                        # columns 1 and 2 moved together
    assert np.allclose(Zh[:, 1:], X[:, 1:])
    assert set(out) == {"g", "h"}


# ---------------------------------------------------------------------------------------------------- report rules

@pytest.fixture(scope="module")
def script():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import feature_importance_report
    return feature_importance_report


@pytest.mark.case
def test_every_folds_permutation_groups_must_agree_with_fold_zeros(script):
    """MDA pairs fold 0's group names with every fold's perm columns, so a fold in another order is a silent rename."""
    same = [{"meta": {"groups": ["c1", "c2", "c3"]}} for _ in range(3)]
    assert script.fold_groups(same, "full") == ["c1", "c2", "c3"]
    moved = [*same[:2], {"meta": {"groups": ["c2", "c1", "c3"]}}]
    with pytest.raises(RuntimeError, match="fold 2 has different permutation groups"):
        script.fold_groups(moved, "full")
    with pytest.raises(RuntimeError, match="control fold 1"):
        script.fold_groups([same[0], {"meta": {"groups": ["c1", "c2"]}}], "control")


@pytest.mark.case
def test_fold_statistics_holm_and_block_means_hand_cases(script):
    assert script.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adj = script.holm([0.001, float("nan"), 0.002])
    assert adj[0] == pytest.approx(0.002) and np.isnan(adj[1]) and adj[2] == pytest.approx(0.002)   # NaN not counted
    e = script.tci([1.0, 2.0, 3.0])
    assert (e["mean"], e["n"], e["pos"]) == (2.0, 3, 3)
    assert e["hi"] - e["mean"] == pytest.approx(4.302653 * 1.0 / np.sqrt(3), rel=1e-5)
    t = 2.0 / (1.0 / np.sqrt(3))
    assert e["p"] == pytest.approx(0.5 * (1 - t / np.sqrt(t * t + 2)))                # one-sided t, 2 df, closed form
    d = np.array([0.001, 0.003, -0.002, 0.0])
    w = np.array([1.0, 3.0, 1.0, 1.0])
    blocks = np.array([0, 0, 1, 1])
    assert script.block_values(d, w, blocks) == pytest.approx([2.5, -1.0])            # millinats, weighted per block


@pytest.mark.case
def test_keep_needs_holm_and_both_halves_and_the_k_check_flags_a_rising_loss(script):
    def entry(mean, p, early, late):
        return {"gain": {"mean": mean, "p": p}, "early": {"mean": early}, "late": {"mean": late}}
    gains = {"cluster:A": entry(5.0, 0.001, 4.0, 6.0), "cluster:B": entry(5.0, 0.001, 4.0, -1.0),
             "cluster:C": entry(2.0, 0.03, 1.0, 3.0), "cluster:D": entry(-1.0, 0.9, -1.0, -1.0),
             "set:clock": entry(1.0, 0.04, 1.0, 1.0)}
    script.decide(gains)
    assert gains["cluster:A"]["decision"] == "keep" and gains["cluster:A"]["holm_p"] == pytest.approx(0.004)
    assert gains["cluster:B"]["decision"] == "weak"                                    # one half negative
    assert gains["cluster:C"]["holm_p"] == pytest.approx(0.06)                          # third smallest of four: x 2
    assert gains["cluster:C"]["decision"] == "weak"
    assert gains["cluster:D"]["decision"] == "drop"
    assert gains["set:clock"]["decision"] == "keep"                                    # sets use the unadjusted p
    assert script.flat({"mean": -0.5, "hi": -0.1}) and script.flat({"mean": -3.0, "hi": 0.2})
    assert not script.flat({"mean": -3.0, "hi": -1.0})
