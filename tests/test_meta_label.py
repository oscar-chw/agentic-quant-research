"""Model infrastructure by the book (pmlab.afml.models; plan afml-pipeline node 11; AFML ch. 6, 7, 9): one interface for
three families, purged folds, training-only weights, the purged search with a trials ledger, the log-uniform
distribution and refit schedules. Leak guards are in tests/test_meta_label_leaks.py."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from meta_label_support import priced, synthetic
from pmlab.afml import events as ev
from pmlab.afml import models as md
from pmlab.afml import tuning

ROOT = Path(__file__).resolve().parents[1]
# XGBoost is deliberately absent from the registered environment (requirements.lock is a frozen artifact of the
# confirmation and an extra installed package fails its library check), so the boosted challenger and every family
# sweep that includes it run under .venv-research. Under the registered venv those parameters skip, loudly.
HAVE_XGBOOST = importlib.util.find_spec("xgboost") is not None
NO_XGB = pytest.mark.skipif(not HAVE_XGBOOST, reason="xgboost is not in the registered environment by design: run "
                                                     "this file with .venv-research/bin/python")
FAMILIES = [f for f in md.FAMILIES if f != "xgb" or HAVE_XGBOOST]
FAMILY_PARAMS = [pytest.param(f, marks=[NO_XGB]) if f == "xgb" else f for f in md.FAMILIES]
FAST = {"bagged_trees": {"n_estimators": 20, "min_weight_fraction_leaf": 0.02, "update_trees": 2},
        "logistic": {}, "hgb": {"max_iter": 30, "min_samples_leaf": 5, "val_splits": 3},
        "xgb": {"n_estimators": 30, "min_child_weight": 5.0, "val_splits": 3}}
SCHEDULES = ["static", "rolling_days=2", "per_window", "rolling_days=2+per_window"]


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def world():
    rows, paths = synthetic(days=8, windows=12, events=4, seed=3)
    return rows, paths, md.Dataset.from_rows(rows, paths)


def overlaps(data, a, b) -> bool:
    """Any span of positions a overlapping any span of positions b."""
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return False
    return bool(((data.t0[a][:, None] < data.t1[b][None, :]) & (data.t0[b][None, :] < data.t1[a][:, None])).any())


# ---------------------------------------------------------------------------------------------------- interface

@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_every_family_fits_its_training_rows_and_predicts_probabilities_of_the_label(world, family):
    _, _, data = world
    train, test = md.day_folds(data, 4)[1]
    m = md.MetaModel(family, FAST[family]).fit(data, train)
    p = m.predict_proba(data, test)
    assert p.shape == (len(test),) and np.all((p >= 0) & (p <= 1))     # unbalanced tree leaves can be pure
    assert m.info_["n_train"] == len(train)
    assert np.unique(np.round(p, 6)).size > 10                     # a fitted model, not a constant
    est = m.estimator_
    if family == "bagged_trees":
        u = ev.sample_weights(data.spans(train), data.paths)["avg_uniqueness"].mean()
        assert isinstance(est, BaggingClassifier) and est.max_samples == pytest.approx(u) == m.info_["max_samples"]
        assert 0.05 < u < 0.9
        assert len(est.estimators_) == 20 and all(type(t) is DecisionTreeClassifier for t in est.estimators_)
        assert all(t.criterion == "entropy" for t in est.estimators_)
    elif family == "logistic":
        assert [n for n, _ in est.steps] == ["impute", "scale", "logit"]
    else:
        if family == "hgb":
            assert est.early_stopping is False and est.max_iter == m.info_["n_iter"]
        else:                                                          # xgb: no early stopping, no class balancing
            assert est.n_estimators == m.info_["n_iter"] and est.scale_pos_weight == 1.0
            assert est.get_params().get("early_stopping_rounds") is None
        assert 1 <= m.info_["n_iter"] <= 30
        itr, iva = m.inner_
        assert set(itr) | set(iva) <= set(train) and not set(itr) & set(iva)
        train_days = np.unique(data.day[train])
        assert data.day[iva].min() > data.day[itr].max()               # the latest training days validate
        assert set(np.unique(data.day[iva])) == set(train_days[-len(np.unique(data.day[iva])):])
        assert not overlaps(data, itr, iva)
    with pytest.raises(ValueError):
        md.MetaModel(family, {"no_such_parameter": 1})
    with pytest.raises(ValueError):
        md.MetaModel("svm")


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_empty_and_constant_feature_columns_fit_in_every_family(world, family):
    rows, paths, _ = world
    rows = rows.assign(f_rule_L=np.float32(60.0), f_book_spread=np.float32(np.nan))
    last = rows["day_tw"] == rows["day_tw"].max()                      # recorded on the last day only, like the book
    rows.loc[last, "f_book_spread"] = np.random.default_rng(0).random(last.sum()).astype(np.float32)
    data = md.Dataset.from_rows(rows, paths)
    folds = md.day_folds(data, 4)
    for train, test in (folds[0], folds[-1]):                           # book present in training rows / absent
        p = md.predict_fold(data, family, FAST[family], "per_window", train, test)
        assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))


def test_bagged_trees_can_be_bagged_one_tree_forests_as_in_the_book(world):
    _, _, data = world
    train, test = md.day_folds(data, 4)[1]
    m = md.MetaModel("bagged_trees", {**FAST["bagged_trees"], "base": "forest"}).fit(data, train)
    assert all(type(t) is RandomForestClassifier and t.n_estimators == 1 and not t.bootstrap for t in m.estimator_.estimators_)
    assert np.all(np.isfinite(m.predict_proba(data, test)))
    m = md.MetaModel("bagged_trees", {**FAST["bagged_trees"], "base": "forest", "balanced": True}).fit(data, train)
    assert all(isinstance(t, md.BalancedForest) and t.n_estimators == 1 and not t.bootstrap for t in m.estimator_.estimators_)


def test_balanced_subsample_gives_both_classes_equal_weight_within_each_trees_own_draw():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    y = (rng.random(600) < 0.8).astype(int)
    counts = rng.poisson(1.0, 600).astype(float)
    counts[y == 0] *= rng.random((y == 0).sum()) < 0.5                  # the draw holds fewer 0s than the data
    root = lambda t: t.tree_.value[0, 0] * t.tree_.weighted_n_node_samples[0]
    total = root(md.BalancedTree(criterion="entropy", max_depth=2).fit(X, y, sample_weight=counts))
    assert total[0] == pytest.approx(total[1])
    plain = root(DecisionTreeClassifier(class_weight="balanced", max_depth=2).fit(X, y, sample_weight=counts))
    assert plain[0] != pytest.approx(plain[1], rel=0.05)                 # balancing on all rows is not on the draw
    bag = BaggingClassifier(md.BalancedTree(max_depth=2), n_estimators=8, max_samples=0.3, random_state=1)
    bag.fit(X, y, sample_weight=np.ones(600))
    for t in bag.estimators_:
        v = root(t)
        assert v[0] == pytest.approx(v[1])


def test_each_bagged_tree_draws_max_samples_of_the_training_rows_whatever_the_scale_of_the_weights():
    """6.3.3 / 4.5 (docs/afml_sections/ch06.md F6.2): sklearn 1.9 draws max_samples x the SUM of the sample weights, so
    raw average-uniqueness weights (mean well below 1) would give each tree n u^2 rows instead of the book's n u.
    MetaModel scales its fit weights to mean 1; asserting max_samples alone does not catch losing that scaling."""
    rng = np.random.default_rng(5)
    n, u = 4000, 0.15
    X, y = rng.normal(size=(n, 3)), (rng.random(n) < 0.5).astype(int)
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    raw = np.full(n, u)
    drawn = lambda w: np.mean([len(s) for s in BaggingClassifier(tree, n_estimators=8, max_samples=u, bootstrap=True,
                                                                random_state=0).fit(X, y, sample_weight=w).estimators_samples_])
    assert drawn(raw) == pytest.approx(n * u * u, rel=0.05)              # unscaled: the fault of F4.1 / F8.1
    assert drawn(raw / raw.mean()) == pytest.approx(n * u, rel=0.05)     # scaled to mean 1: the book's n x u

    rows, paths = synthetic(days=6, windows=10, events=4, seed=11)
    data = md.Dataset.from_rows(rows, paths)
    train = md.day_folds(data, 3)[1][0]
    m = md.MetaModel("bagged_trees", FAST["bagged_trees"]).fit(data, train)
    mean_u = m.info_["max_samples"]
    assert 0 < mean_u < 0.5                                              # overlapping spans: uniqueness well below 1
    sizes = [len(s) for s in m.estimator_.estimators_samples_]
    assert np.mean(sizes) == pytest.approx(len(train) * mean_u, rel=0.05)
    assert np.mean(sizes) == pytest.approx(len(train) * mean_u ** 2 / mean_u, rel=0.05)   # not the n u^2 draw
    assert np.mean(sizes) > 2 * len(train) * mean_u ** 2


def test_weighted_pipeline_hands_sample_weight_to_the_final_step_only():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3)) * [1, 10, 100]
    y = (X[:, 0] + rng.normal(size=400) > 0).astype(int)
    w = rng.exponential(size=400)
    pipe = md.WeightedPipeline([("scale", StandardScaler()), ("logit", LogisticRegression())]).fit(X, y, sample_weight=w)
    scaler = StandardScaler().fit(X)                                    # unweighted, on the rows given
    by_hand = LogisticRegression().fit(scaler.transform(X), y, sample_weight=w)
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, scaler.mean_)
    np.testing.assert_allclose(pipe.named_steps["logit"].coef_, by_hand.coef_, rtol=1e-6)
    unweighted = LogisticRegression().fit(scaler.transform(X), y)
    assert not np.allclose(pipe.named_steps["logit"].coef_, unweighted.coef_, rtol=1e-3)


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_per_window_updates_are_warm_starts_where_supported_and_refits_otherwise(world, family):
    _, _, data = world
    train, test = md.day_folds(data, 4)[2]
    extra = test[: len(test) // 3]
    union = np.concatenate([train, extra])
    m = md.MetaModel(family, FAST[family]).fit(data, train)
    fresh = md.MetaModel(family, FAST[family]).fit(data, union)
    m.update(data, union)
    if family == "bagged_trees":
        assert len(m.estimator_.estimators_) == 22                       # 2 trees added, the first 20 kept
        u = ev.sample_weights(data.spans(union), data.paths)["avg_uniqueness"].mean()
        assert m.estimator_.max_samples == pytest.approx(u)
    elif family == "logistic":                                           # warm start converges to the refit
        np.testing.assert_allclose(m.predict_proba(data, test), fresh.predict_proba(data, test), atol=2e-3)
    else:                                                                # refit with the base model's iterations
        X, y, w, _ = m._xyw(data, union)
        if family == "hgb":
            assert m.estimator_.max_iter == m.info_["n_iter"]
            again = md._hgb(md.MetaModel("hgb", FAST["hgb"]).params, m.info_["n_iter"], 0)
        else:
            assert m.estimator_.n_estimators == m.info_["n_iter"]
            again = md._xgb(md.MetaModel("xgb", FAST["xgb"]).params, m.info_["n_iter"], 0)
        np.testing.assert_allclose(m.predict_proba(data, test), again.fit(X, y, sample_weight=w).predict_proba(data.X[test])[:, 1])


# ---------------------------------------------------------------------------------------------------- folds, weights

def test_purged_folds_are_whole_taiwan_days_with_no_overlapping_training_span_and_the_default_embargo():
    rows, paths = synthetic(days=8, windows=12, events=4, seed=4)
    rows = rows.copy()
    late = rows["start"] == rows.groupby("day_tw")["start"].transform("max")   # each day's last spans cross midnight
    rows.loc[late, "t1"] = rows.loc[late, "t1"] + 12 * 3600
    data = md.Dataset.from_rows(rows, paths)
    folds = md.day_folds(data, 4)
    days = np.unique(data.day)
    assert len(folds) == 4
    seen = []
    for train, test in folds:
        tdays = np.unique(data.day[test])
        assert np.array_equal(tdays, np.arange(tdays.min(), tdays.max() + 1))           # contiguous whole days
        assert set(np.flatnonzero(np.isin(data.day, tdays))) == set(test)
        assert not np.isin(data.day[train], tdays).any()
        assert not overlaps(data, train, test)
        end = data.t1[test].max()
        assert not ((data.t0[train] >= end) & (data.t0[train] < end + md.EMBARGO)).any()  # embargo after the block
        before = np.setdiff1d(np.flatnonzero(data.day < tdays.min()), test)
        dropped = np.setdiff1d(before, train)
        assert all(overlaps(data, [i], test) for i in dropped)          # only purged rows are missing before it
        seen += list(tdays)
    assert sorted(seen) == list(days)
    assert any(overlaps(data, np.flatnonzero(data.day == d), np.flatnonzero(data.day == d + 1)) for d in days[:-1])
    for train, test in folds:
        for cv in md.CVS:
            for schedule in SCHEDULES:
                for st in md.plan_steps(data, train, test, schedule, cv):
                    assert not overlaps(data, st.train, st.test)


@pytest.mark.parametrize("family", FAMILY_PARAMS)
@pytest.mark.parametrize("schedule", ["static", "rolling_days=2+per_window"])
def test_sample_weights_are_computed_from_the_rows_each_model_is_fitted_on(world, monkeypatch, family, schedule):
    _, _, data = world
    train, test = md.day_folds(data, 4)[2]
    pos = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(data.start, data.t0))}
    calls, real = [], ev.span_weights

    def spy(events, paths):
        calls.append({pos[(int(s), int(t))] for s, t in zip(events["start"], events["t0"])})
        return real(events, paths)

    monkeypatch.setattr(md.ev, "span_weights", spy)
    md.predict_fold(data, family, FAST[family], schedule, train, test, "kfold")
    steps = md.plan_steps(data, train, test, schedule, "kfold")
    allowed = [set(st.base) for st in steps] + [set(st.train) for st in steps]
    assert calls
    for c in calls:
        assert any(c <= a for a in allowed)
        if schedule == "static":
            assert not c & set(test)
    if "per_window" in schedule:                                         # updates happen, on settled rows
        assert any(c and c <= set(st.settled) for st in steps if len(st.settled) for c in calls)


@pytest.mark.parametrize("family", ["logistic", "bagged_trees"])
@pytest.mark.parametrize("base", ["static", "rolling_days=2"])
def test_per_window_starts_each_day_from_the_base_model_and_learns_from_windows_settled_since(world, family, base):
    _, _, data = world
    train, test = md.day_folds(data, 4)[2]
    daily = md.predict_fold(data, family, FAST[family], base, train, test)
    windowed = md.predict_fold(data, family, FAST[family], "per_window" if base == "static" else base + "+per_window",
                               train, test)
    first = np.zeros(len(test), bool)
    for d in np.unique(data.day[test]):
        day = data.day[test] == d
        first |= day & (data.start[test] == data.start[test][day].min())
    assert first.sum() >= 2 and (~first).sum() > first.sum()
    np.testing.assert_allclose(windowed[first], daily[first], rtol=1e-9)
    assert not np.allclose(windowed[~first], daily[~first], rtol=1e-6)


# ---------------------------------------------------------------------------------------------------- search

def load_registry():
    spec = importlib.util.spec_from_file_location("trial_registry", ROOT / "scripts/trial_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_search_picks_the_planted_best_parameter_scores_weighted_log_loss_and_ledgers_every_candidate(
        tmp_path, monkeypatch):
    rows, paths = synthetic(days=8, windows=15, events=5, seed=5)
    data = md.Dataset.from_rows(rows, paths)
    ledger = tmp_path / "research/trials/meta_label_test/logistic.jsonl"
    s = md.PurgedSearch("logistic", {"C": [1e-6, 1.0, 1e-4]}, n_splits=4, ledger=ledger).fit(data)
    assert s.best_params_ == {"C": 1.0, "schedule": "static"}
    assert s.best_index_ == 1 and s.best_score_ == max(s.cv_results_["mean_score"])
    assert s.best_score_ > np.mean(s.baseline_fold_scores_) + 0.05
    assert all(len(f) == 4 for f in s.cv_results_["fold_scores"])
    assert s.best_fold_scores_ == s.cv_results_["fold_scores"][1]
    assert s.score_weight == "avg_uniqueness"
    for k, (_, test) in enumerate(s.folds_):                            # weighted log loss, test-row weights
        w = ev.sample_weights(data.spans(test), data.paths)["avg_uniqueness"].to_numpy()
        y, p = data.y[test], s.oof_[1, test].astype(float)
        ll = -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum()
        assert s.best_fold_scores_[k] == pytest.approx(-ll, rel=1e-5)
        assert s.best_fold_scores_[k] != pytest.approx(-np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))), rel=1e-4)
    lines = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [x["params"]["C"] for x in lines] == [1e-6, 1.0, 1e-4]
    assert all(x["params"]["family"] == "logistic" and x["params"]["schedule"] == "static" for x in lines)
    assert [x["value"] for x in lines] == pytest.approx(list(s.cv_results_["mean_score"]))
    reg = load_registry()
    monkeypatch.setattr(reg, "RESEARCH", tmp_path / "research")
    assert reg.ledger_counts() == {"meta_label_test": 3}

    par = md.PurgedSearch("logistic", {"C": [1e-6, 1.0, 1e-4]}, n_splits=4, n_jobs=2, ledger=tmp_path / "p.jsonl").fit(data)
    assert par.cv_results_["fold_scores"].tolist() == s.cv_results_["fold_scores"].tolist()


def test_randomised_search_draws_log_uniform_values_and_finds_the_planted_region(tmp_path):
    rows, paths = synthetic(days=8, windows=15, events=5, seed=5)
    data = md.Dataset.from_rows(rows, paths)
    s = md.PurgedSearch("logistic", {"C": md.log_uniform(1e-7, 1e1)}, n_iter=8, n_splits=4, seed=2,
                        ledger=tmp_path / "r.jsonl").fit(data)
    cs = [p["C"] for p in s.cv_results_["params"]]
    assert len(set(cs)) == 8 and all(1e-7 <= c <= 10 for c in cs)
    assert min(cs) < 1e-4 < s.best_params_["C"]                        # small C (no signal) was drawn and lost
    lines = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert len(lines) == 8 and {x["phase"] for x in lines} == {"random"}


def test_the_refit_schedule_is_a_search_dimension_that_finds_a_regime_change(tmp_path):
    rows, paths = synthetic(days=12, windows=15, events=5, flip_day=6, seed=6)
    data = md.Dataset.from_rows(rows, paths)
    space = {"C": [1.0], "schedule": ["static", "rolling_days=2", "rolling_days=8"]}
    s = md.PurgedSearch("logistic", space, n_splits=4, cv="walk_forward", ledger=tmp_path / "s.jsonl").fit(data)
    assert s.best_params_["schedule"] == "rolling_days=2"
    assert s.scored_.sum() == (data.day >= data.day.min() + 8).sum()    # rows every candidate predicts
    assert list(s.cv_results_["schedule"]) == space["schedule"]
    assert s.cv_results_["mean_score"][1] > s.cv_results_["mean_score"][0] + 0.1


def test_log_uniform_distribution_has_the_books_quantiles():
    a, b = 1e-3, 1e3
    dist = md.log_uniform(a, b)
    q = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    np.testing.assert_allclose(dist.ppf(q), a * (b / a) ** q, rtol=1e-9)
    x = np.array([2e-3, 0.5, 1.0, 40.0, 999.0])
    np.testing.assert_allclose(dist.cdf(x), np.log(x / a) / np.log(b / a), rtol=1e-9)
    np.testing.assert_allclose(dist.pdf(x), 1 / (x * np.log(b / a)), rtol=1e-9)
    draws = dist.rvs(size=20000, random_state=np.random.default_rng(0))
    assert draws.min() >= a and draws.max() <= b
    np.testing.assert_allclose(np.quantile(draws, [0.25, 0.5, 0.75]), a * (b / a) ** np.array([0.25, 0.5, 0.75]), rtol=0.15)
    assert stats.kstest(np.log(draws), stats.uniform(np.log(a), np.log(b) - np.log(a)).cdf).pvalue > 0.01
    for bad in ((0.0, 1.0), (2.0, 1.0)):
        with pytest.raises(ValueError):
            md.log_uniform(*bad)


# ---------------------------------------------------------------------------------------------------- schedules

@pytest.mark.parametrize("cv", md.CVS)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_refit_schedules_never_train_on_their_test_day(world, cv, schedule):
    rows, paths, _ = world
    rows = rows.copy()
    long = np.random.default_rng(0).random(len(rows)) < 0.15              # spans reaching into later windows and days
    rows.loc[long, "t1"] += np.random.default_rng(1).integers(300, 20 * 3600, long.sum())
    data = md.Dataset.from_rows(rows, paths)
    sch = md.Schedule.parse(schedule)
    assert str(sch) == schedule
    n_steps = 0
    for train, test in md.day_folds(data, 4):
        steps = md.plan_steps(data, train, test, schedule, cv)
        n_steps += len(steps)
        for st in steps:
            assert set(st.test) <= set(test) and np.all(data.day[st.test] == st.day)
            assert not np.isin(data.day[st.base], st.day).any()               # the base model never sees the day
            if sch.days is not None:
                assert data.day[st.base].min() >= st.day - sch.days and data.day[st.base].max() < st.day
            if cv == "walk_forward" or sch.days is not None:
                assert data.t1[st.base].max() <= data.t0[test].min() or sch.days is not None
                assert data.t1[st.base].max() <= data.t0[st.test].min()
            if sch.per_window:
                assert np.all(data.start[st.test] == st.window)
                assert np.all(data.day[st.settled] == st.day) and np.all(data.start[st.settled] < st.window)
                assert data.t1[st.settled].max(initial=0) <= data.t0[st.test].min()
                earlier = test[(data.day[test] == st.day) & (data.t1[test] <= data.t0[st.test].min())]
                assert set(st.settled) == set(earlier)                          # every settled window is used
            else:
                assert not len(st.settled) and set(st.test) == set(test[data.day[test] == st.day])
    assert n_steps > 0
    with pytest.raises(ValueError):
        md.Schedule.parse("rolling_days=0")
    with pytest.raises(ValueError):
        md.Schedule.parse("static+rolling_days=2")


def test_predictions_are_aligned_to_the_event_rows_with_their_spans(tmp_path):
    rows, paths = synthetic(days=6, windows=10, events=4, seed=8)
    rows = rows.sample(frac=1.0, random_state=0)
    rows.index = [f"row{i}" for i in range(len(rows))]
    data = md.Dataset.from_rows(rows, paths)
    s = md.PurgedSearch("logistic", {"schedule": ["static", "rolling_days=1"]}, n_splits=3, ledger=tmp_path / "a.jsonl").fit(data)
    for c in (0, 1):
        pr = s.predictions(c)
        assert list(pr.index) == list(rows.index)
        assert (pr["t0"] == rows["t0"]).all() and (pr["t1"] == rows["t1"]).all() and (pr["start"] == rows["start"]).all()
        assert (pr["t"] == rows["t"]).all() and (pr["day_tw"] == rows["day_tw"]).all()
        for k, (train, test) in enumerate(s.folds_):
            expect = md.predict_fold(data, "logistic", {}, s.cv_results_["schedule"][c], train, test)
            np.testing.assert_allclose(pr["p"].to_numpy()[test], expect, rtol=1e-6)
            assert (pr["fold"].to_numpy()[test] == k).all()
    first = rows["day_tw"] == rows["day_tw"].min()
    assert s.predictions(1)["p"][first].isna().all() and not s.predictions(0)["p"].isna().any()
    assert not s.predictions()["scored"][first].any() and s.predictions()["scored"][~first].all()


# ---------------------------------------------------------------------------------------------------- calibration

def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def test_default_weights_keep_the_probability_calibrated_when_the_price_is_the_probability(tmp_path):
    """Code review afml.md, models finding 1. The meta-label is drawn with the fair price as its true probability. With
    the default uniqueness weights the fitted slope on logit(price) is about 1 and scoring prefers the true
    probability to a constant 1/2. Return attribution (about |label - price|) stays an option, and shows why it is
    not the default: its fit flattens toward 1/2 and its score prefers the constant."""
    rows, paths = priced(days=10, windows=200, events=3, seed=4)
    data = md.Dataset.from_rows(rows, paths)
    train, test = md.day_folds(data, 5)[4]
    x = data.X[test][:, data.features.index("f_logit_q")].astype(float)
    y, q = data.y[test], 1 / (1 + np.exp(-x))
    const = np.full(len(test), 0.5)
    slope = lambda m: np.polyfit(x, logit(m.predict_proba(data, test)), 1)[0]

    m = md.MetaModel("logistic", {"C": 1e6}).fit(data, train)
    assert 0.85 < slope(m) < 1.15
    assert (m.fit_weight, m.score_weight) == ("uniqueness_decay", "avg_uniqueness")
    w = data.weights(test, m.score_weight)
    assert md.log_loss(y, q, w) < md.log_loss(y, const, w) - 0.03
    s = md.PurgedSearch("logistic", {"C": [1e-8, 1e6]}, n_splits=5, ledger=tmp_path / "c.jsonl").fit(data)
    assert s.best_params_["C"] == 1e6 and s.best_score_ > np.mean(s.baseline_fold_scores_) + 0.03

    attribution = md.MetaModel("logistic", {"C": 1e6}, fit_weight="weight", score_weight="w_return").fit(data, train)
    assert abs(slope(attribution)) < 0.3
    w = data.weights(test, "w_return")
    assert md.log_loss(y, q, w) > md.log_loss(y, const, w)
    with pytest.raises(ValueError):
        md.MetaModel("logistic", fit_weight="no_such_weight")
    with pytest.raises(ValueError):
        md.PurgedSearch("logistic", {}, score_weight="no_such_weight")


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_default_class_weights_keep_the_mean_probability_at_the_base_rate(family):
    """Code review afml.md, models finding 2: labels independent of the features with a base rate of 3/4. Every
    family's mean out-of-sample probability stays at the training base rate; class balancing (the bagged trees'
    option) pulls it to 1/2."""
    rows, paths = synthetic(days=8, windows=15, events=5, seed=13)
    label = (np.random.default_rng(13).random(len(rows)) < 0.75).astype(np.int8)
    data = md.Dataset.from_rows(rows.assign(label=label, meta_label=label), paths)
    train, test = md.day_folds(data, 4)[3]
    base = float(np.mean(data.y[train]))
    assert 0.7 < base < 0.8
    params = {"n_estimators": 60, "min_weight_fraction_leaf": 0.1} if family == "bagged_trees" else FAST[family]
    p = md.MetaModel(family, params).fit(data, train).predict_proba(data, test)
    assert abs(p.mean() - base) < 0.04
    if family == "bagged_trees":
        balanced = md.MetaModel(family, {**params, "balanced": True}).fit(data, train).predict_proba(data, test)
        assert balanced.mean() < p.mean() - 0.1


# ---------------------------------------------------------------------------------------------------- weight updates

def recomputed_xyw(self, data, idx, spans=None):
    """The fit weights as models.py computed them before code review afml.md, models finding 3: events.sample_weights
    on every fitted row at every fit and update, whatever the caller already has."""
    idx = np.asarray(idx, dtype=np.int64)
    sw = ev.sample_weights(data.spans(idx), data.paths)
    col = sw["avg_uniqueness"] * sw["w_decay"] if self.fit_weight == "uniqueness_decay" else sw[self.fit_weight]
    w = col.to_numpy(float)
    return data.X[idx], data.y[idx], w / w.mean(), float(sw["avg_uniqueness"].mean())


@pytest.mark.parametrize("family", FAMILY_PARAMS)
@pytest.mark.parametrize("fit_weight", ["uniqueness_decay", "weight"])
def test_per_window_weight_updates_give_the_predictions_of_recomputing_every_rows_weights(world, monkeypatch, family,
                                                                                       fit_weight):
    _, _, data = world
    rows = np.arange(len(data))
    u, raw = data.span_weights(rows)
    sw = ev.sample_weights(data.spans(rows), data.paths)
    for c in md.WEIGHTS:                                                  # events.sample_weights' formulas
        expect = sw["avg_uniqueness"] * sw["w_decay"] if c == "uniqueness_decay" else sw[c]
        np.testing.assert_array_equal(md.weights(u, raw, data.t0, c), expect.to_numpy(float))
    folds = md.day_folds(data, 4, embargo=0)
    for schedule, (train, test) in (("per_window", folds[1]), ("rolling_days=2+per_window", folds[3])):
        fast = md.predict_fold(data, family, FAST[family], schedule, train, test, fit_weight=fit_weight)
        with monkeypatch.context() as m:
            m.setattr(md.MetaModel, "_xyw", recomputed_xyw)
            slow = md.predict_fold(data, family, FAST[family], schedule, train, test, fit_weight=fit_weight)
        assert np.isfinite(fast).sum() > 50
        np.testing.assert_array_equal(fast, slow)


def test_split_windows_fall_back_to_weights_on_the_whole_training_set(world, monkeypatch):
    """Folds that split a window put settled rows beside base rows of the same window; their concurrency is shared."""
    _, _, data = world
    days = np.unique(data.day)
    test = np.flatnonzero(data.day == days[-1])
    train = np.setdiff1d(np.arange(len(data)), test)
    first = data.start[test].min()
    moved = test[(data.start[test] == first) & (data.t0[test] < np.median(data.t0[test][data.start[test] == first]))]
    train, test = np.union1d(train, moved), np.setdiff1d(test, moved)
    fast = md.predict_fold(data, "logistic", {}, "per_window", train, test)
    monkeypatch.setattr(md.MetaModel, "_xyw", recomputed_xyw)
    np.testing.assert_array_equal(fast, md.predict_fold(data, "logistic", {}, "per_window", train, test))


def test_per_window_schedules_weigh_each_row_once_and_refit_only_when_windows_settle(world, monkeypatch):
    """Counts, not wall-clock time: span weights are computed for each base once and for each settled row once
    (recomputing every training row at each update is quadratic in the windows of a test day), one fit per base and
    one update per window with newly settled rows."""
    _, _, data = world
    train, test = md.day_folds(data, 4)[2]
    schedule = "rolling_days=2+per_window"
    rows, fits, updates = [], [], []
    real_spans, real_fit, real_update = ev.span_weights, md.MetaModel.fit, md.MetaModel.update
    monkeypatch.setattr(md.ev, "span_weights", lambda events, paths: rows.append(len(events)) or real_spans(events, paths))
    monkeypatch.setattr(md.MetaModel, "fit", lambda self, *a, **k: fits.append(1) or real_fit(self, *a, **k))
    monkeypatch.setattr(md.MetaModel, "update", lambda self, *a, **k: updates.append(1) or real_update(self, *a, **k))
    md.predict_fold(data, "logistic", {}, schedule, train, test)
    steps = md.plan_steps(data, train, test, schedule)
    bases = {st.base_key: st.base for st in steps}
    refits, last = [], {}
    for st in steps:
        if len(st.settled) > last.get(st.day, 0):
            refits.append(st)
        last[st.day] = len(st.settled)
    settled = np.unique(np.concatenate([st.settled for st in steps]))
    assert len(bases) >= 2 and len(refits) > 10
    assert len(fits) == len(bases) and len(updates) == len(refits)
    assert sum(rows) == sum(len(b) for b in bases.values()) + len(settled)
    assert sum(rows) < (sum(len(b) for b in bases.values()) + sum(len(st.train) for st in refits)) / 5


def test_models_share_the_tuning_modules_weighted_pipeline_and_log_uniform():
    """Code review afml.md, models finding 4: one implementation of 9.2 and 9.4."""
    assert md.WeightedPipeline is tuning.WeightedPipeline and md.log_uniform is tuning.log_uniform


# ---------------------------------------------------------------------------------------------------- stage 4 script

def load_script(env: dict | None = None):
    """scripts/meta_label.py as a module. Its output paths are read at import time, so a test that writes anything
    passes ML_SMOKE_OUT through the environment before importing."""
    for k, v in (env or {}).items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location("meta_label_script", ROOT / "scripts/meta_label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_stage_4_feature_sets_are_the_baseline_plus_stage_3s_kept_clusters():
    """Stage 4 uses what stage 3 kept, per rule, and nothing else; `all` adds back exactly the rest."""
    ml = load_script()
    fi = {"baseline": ["f_fair", "f_tau"],
          "rules": {"q_taker": {"gains": {
              "cluster:C01": {"features": ["f_tau"], "decision": "keep"},
              "cluster:C02": {"features": ["f_btc_ret_1", "f_btc_ret_5"], "decision": "keep"},
              "cluster:C03": {"features": ["f_roll"], "decision": "weak"},
              "cluster:C04": {"features": ["f_spread_proxy", "f_last_age"], "decision": "drop"},
              "set:activity": {"features": ["f_sigma_har"], "decision": "keep"}}}}}     # sets are not clusters
    got = ml.kept_features("q_taker", fi)
    assert got["keep"] == ["f_fair", "f_tau", "f_btc_ret_1", "f_btc_ret_5"]              # baseline first, no repeats
    assert got["rest"] == ["f_roll", "f_spread_proxy", "f_last_age"]
    assert got["keep_clusters"] == ["cluster:C01", "cluster:C02"]
    assert "f_sigma_har" not in got["keep"]
    assert set(got["keep"]) | set(got["dropped"]) == set(ml.ei.FEATURES) and not set(got["keep"]) & set(got["dropped"])
    real = json.loads((ROOT / "research/feature_importance.json").read_text())
    for rule in ("q_taker", "fair_maker", "flow"):
        k = ml.kept_features(rule, real)
        assert k["keep"][:5] == real["baseline"] and len(k["keep"]) > 5
        assert all(c in real["rules"][rule]["gains"] for c in k["keep_clusters"])
        assert all(real["rules"][rule]["gains"][c]["decision"] == "keep" for c in k["keep_clusters"])
    with pytest.raises(ValueError):
        ml.feature_set("q_taker", "everything")


def test_reliability_buckets_weigh_each_independent_label_once_and_give_the_brier_score():
    ml = load_script()
    p = np.array([0.05, 0.15, 0.15, 0.95, 0.95])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    w = np.array([1.0, 1.0, 3.0, 2.0, 2.0])
    r = ml.reliability(y, p, w, bins=10)
    assert [b["lo"] for b in r["bins"]] == [0.0, 0.1, 0.9]
    mid = next(b for b in r["bins"] if b["lo"] == 0.1)
    assert mid["n"] == 2 and mid["weight"] == 4.0
    assert mid["p_mean"] == pytest.approx(0.15) and mid["y_rate"] == pytest.approx(3 / 4)
    hi = next(b for b in r["bins"] if b["lo"] == 0.9)
    assert hi["y_rate"] == pytest.approx(2 / 4)
    assert r["brier"] == pytest.approx(float(np.sum(w * (p - y) ** 2) / w.sum()))
    gap = (1 * 0.05 + 4 * abs(0.15 - 0.75) + 4 * abs(0.95 - 0.5)) / 9
    assert r["calibration_error"] == pytest.approx(gap)
    perfect = ml.reliability([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], [1, 1, 1, 1], bins=10)
    assert perfect["calibration_error"] == pytest.approx(0.1) and perfect["brier"] == pytest.approx(0.01)


def test_the_filter_check_takes_only_signals_whose_probability_clears_their_cost():
    """Acting on every primary signal against acting only where P(win) > cost: the P&L per dollar of each side, and a
    bootstrap over whole Taiwan days."""
    ml = load_script()
    df = pd.DataFrame({"cost": [0.5, 0.5, 0.8, 0.8], "payoff": [1.0, 0.0, 1.0, 0.0],
                       "day_tw": ["2026-05-01", "2026-05-01", "2026-05-02", "2026-05-02"]})
    p = np.array([0.9, 0.9, 0.1, 0.1])
    out = ml.filter_check(df, p, np.ones(4, bool), boot=200, seed=1)
    assert out["rows"] == 4 and out["taken"] == 2 and out["share_taken"] == 0.5 and out["days"] == 2
    assert out["all_pnl_per_dollar"] == pytest.approx((0.5 - 0.5 + 0.2 - 0.8) / 2.6)
    assert out["taken_pnl_per_dollar"] == pytest.approx(0.0)                       # the two rows priced at 0.5
    assert out["difference"] == pytest.approx(out["taken_pnl_per_dollar"] - out["all_pnl_per_dollar"])
    assert out["difference"] > 0
    assert out["lo"] <= out["difference"] <= out["hi"]
    half = ml.filter_check(df, p, np.array([True, True, False, False]), boot=50, seed=1)
    assert half["rows"] == 2 and half["taken"] == 2                                # unscored rows never count
    none = ml.filter_check(df, np.array([0.1, 0.1, 0.1, 0.1]), np.ones(4, bool), boot=50, seed=1)
    assert none["taken"] == 0 and np.isnan(none["taken_pnl_per_dollar"])


@NO_XGB
def test_xgboost_takes_its_rounds_from_the_purged_inner_validation_and_never_balances_classes(world):
    """6.4's challenger under the same discipline as hgb: no random early stopping, rounds from the latest training
    days, and scale_pos_weight left at 1 so the probability stays calibrated for ch. 10."""
    _, _, data = world
    train, test = md.day_folds(data, 4)[2]
    m = md.MetaModel("xgb", {"n_estimators": 40, "min_child_weight": 5.0, "val_splits": 3}).fit(data, train)
    itr, iva = m.inner_
    assert set(itr) | set(iva) <= set(train) and not set(itr) & set(iva)
    assert data.day[iva].min() > data.day[itr].max() and not overlaps(data, itr, iva)
    probe = md._xgb(m.params, 40, 0).fit(data.X[itr], data.y[itr],
                                         sample_weight=m._xyw(data, itr)[2],
                                         eval_set=[(data.X[iva], data.y[iva])],
                                         sample_weight_eval_set=[data.weights(iva, m.score_weight)], verbose=False)
    curve = probe.evals_result()["validation_0"]["logloss"]
    assert m.info_["n_iter"] == int(np.argmin(curve)) + 1 < 40                     # a real minimum, not the cap
    assert m.estimator_.n_estimators == m.info_["n_iter"] and m.estimator_.scale_pos_weight == 1.0
    p = m.predict_proba(data, test)
    assert np.all(np.isfinite(p)) and 0.0 < p.mean() < 1.0


def test_a_single_class_training_block_predicts_nothing_rather_than_failing():
    """A rolling window can hold one label. XGBoost refuses to infer classes from it and a scikit-learn model fitted
    on it has no class 1 to read, so the step is skipped and those rows stay NaN, unscored by every candidate."""
    rows, paths = synthetic(days=8, windows=12, events=4, seed=21)
    rows = rows.copy()
    data = md.Dataset.from_rows(rows, paths)
    train, test = md.day_folds(data, 4)[2]
    one = np.zeros(len(data), np.int8)
    one[test] = 1                                                    # every training row is class 0
    flat = md.Dataset(data.X, one, data.start, data.t0, data.t1, data.path_starts, data.path_q, data.features,
                      data.index, data.rule)
    for family in FAMILIES:
        p = md.predict_fold(flat, family, FAST[family], "static", train, test, "kfold")
        assert np.isnan(p).all(), family
        assert np.all(np.isfinite(md.predict_fold(data, family, FAST[family], "static", train, test, "kfold")))


def test_the_runner_refuses_to_start_in_the_registered_environment():
    """requirements.lock is a frozen artifact of confirmation v2 and must stay byte-identical, so xgboost may not be
    installed in the registered environment and stage 4 may not run there - whether or not someone has already
    installed it."""
    ml = load_script()
    assert ml.REGISTERED_VENV == ROOT / ".venv" and ml.RESEARCH_VENV == ROOT / ".venv-research"
    with pytest.raises(SystemExit, match=r"\.venv-research") as polluted:
        ml.check_environment(prefix=str(ROOT / ".venv"), xgboost_present=True)
    assert "requirements.lock" in str(polluted.value) and "byte-identical" in str(polluted.value)
    with pytest.raises(SystemExit, match=r"\.venv-research") as clean:
        ml.check_environment(prefix=str(ROOT / ".venv"), xgboost_present=False)
    assert "must not have" in str(clean.value)
    ml.check_environment(prefix=str(ROOT / ".venv-research"), xgboost_present=True)      # the research venv is fine
    ml.check_environment(prefix=str(tmp_prefix()), xgboost_present=False)                # so is anything else
    for action in (ml.run, ml.summarize, ml.check):
        assert "check_environment" in action.__code__.co_names, action.__name__


def tmp_prefix() -> Path:
    return ROOT / ".venv-somewhere-else"


@NO_XGB
@pytest.mark.integration
def test_the_stage_4_script_runs_end_to_end_and_check_re_derives_every_number(tmp_path):
    """The whole stage on a tiny configuration: sample, every study, the leak controls, the report and the gate. The
    gate must fail once a reported number is edited, so --check really re-derives rather than reading the file back."""
    env = {**os.environ, "ML_SMOKE_OUT": str(tmp_path)}
    run = lambda *args: subprocess.run([sys.executable, str(ROOT / "scripts/meta_label.py"), *args],
                                       cwd=ROOT, env=env, capture_output=True, text=True)
    done = run("--run", "--workers", "2", "--rules", "q_taker")
    assert done.returncode == 0, done.stdout + done.stderr
    assert run("--summarize").returncode != 0                       # fair_maker and flow have no checkpoints yet
    done = run("--run", "--workers", "2")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "meta_label.json").exists() and (tmp_path / "meta_label.md").exists()
    res = json.loads((tmp_path / "meta_label.json").read_text())
    assert set(res["rules"]) == {"q_taker", "fair_maker", "flow"} and res["ok"]
    assert res["checks"]["controls_at_chance"] and res["checks"]["perturbation_clean"]
    assert res["trials"]["ledgered"] == sum(len(f.read_text().splitlines())
                                            for f in (tmp_path / "trials").rglob("*.jsonl"))
    for rule, r in res["rules"].items():
        assert r["max_window_end"] <= res["confirm_start"]
        assert r["best"]["family"] in res["config"]["families"]
        assert len(r["feature_sets"]["kept"]) >= len(r["feature_sets"]["baseline"])
    assert run("--check").returncode == 0
    bad = json.loads((tmp_path / "meta_label.json").read_text())
    bad["rules"]["q_taker"]["best"]["log_loss"] += 1.0
    (tmp_path / "meta_label.json").write_text(json.dumps(bad))
    assert run("--check").returncode != 0


def test_the_xgb_family_says_where_xgboost_lives_when_it_is_not_installed(world, monkeypatch):
    """The registered environment has no xgboost on purpose (requirements.lock is a frozen artifact of confirmation
    v2 and an extra installed package fails its library check), so asking for the family there must name the
    environment that does, not raise a bare ImportError."""
    _, _, data = world
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ImportError, match=r"\.venv-research"):
        md._xgboost_class()
    train, _ = md.day_folds(data, 4)[1]
    with pytest.raises(ImportError, match=r"\.venv-research"):
        md.MetaModel("xgb", FAST["xgb"]).fit(data, train)
    assert "xgb" in md.FAMILIES                                     # the family is declared whatever is installed
