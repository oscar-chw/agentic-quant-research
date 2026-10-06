"""Leak guards of the meta-label models (pmlab.afml.models; plan afml-pipeline node 11): future perturbation, label
permutation at chance, transforms fitted outside the training rows, the confirmation start, feature columns.

Each guard was mutation-checked: a mutation of the library (the leak it guards against) applied to a copy, this file
run against the copy, and it must fail (results in the node 11 record of plan/afml-pipeline)."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from meta_label_support import permute_outcomes, synthetic
from pmlab import WINDOW_SECONDS as T
from pmlab.afml import models as md
from pmlab.evaluation import CONFIRM_START

FAST = {"bagged_trees": {"n_estimators": 20, "min_weight_fraction_leaf": 0.02, "update_trees": 2},
        "logistic": {}, "hgb": {"max_iter": 30, "min_samples_leaf": 5, "val_splits": 3},
        "xgb": {"n_estimators": 30, "min_child_weight": 5.0, "val_splits": 3}}
# see tests/test_meta_label.py: xgboost lives in .venv-research, not in the registered environment
HAVE_XGBOOST = importlib.util.find_spec("xgboost") is not None
NO_XGB = pytest.mark.skipif(not HAVE_XGBOOST, reason="xgboost is not in the registered environment by design: run "
                                                     "this file with .venv-research/bin/python")
FAMILIES = [f for f in md.FAMILIES if f != "xgb" or HAVE_XGBOOST]
FAMILY_PARAMS = [pytest.param(f, marks=[NO_XGB]) if f == "xgb" else f for f in md.FAMILIES]


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def world():
    return synthetic(days=8, windows=12, events=4, seed=11)


def with_X(data: md.Dataset, X: np.ndarray) -> md.Dataset:
    return md.Dataset(X, data.y, data.start, data.t0, data.t1, data.path_starts, data.path_q, data.features, data.index,
                      data.rule)


# ---------------------------------------------------------------------------------------------------- future

@pytest.mark.parametrize("family", FAMILY_PARAMS)
@pytest.mark.parametrize("schedule", ["static", "rolling_days=2", "per_window", "rolling_days=2+per_window"])
def test_perturbing_any_data_after_an_event_changes_no_prediction_for_that_event(world, tmp_path, family, schedule):
    """Walk-forward evaluation: every schedule only uses rows that settled before the rows it predicts. (The book's
    purged k-fold trains the static model on days after the test block too, so it is purged and embargoed instead.)"""
    rows, paths = world
    space = {**{k: [v] for k, v in FAST[family].items()}, "schedule": [schedule]}
    run = lambda r, p: md.PurgedSearch(family, space, n_splits=4, cv="walk_forward", ledger=tmp_path / "l.jsonl").fit(
        md.Dataset.from_rows(r, p)).predictions()["p"].to_numpy()
    before = run(rows, paths)
    cut = int(np.sort(rows["t0"].to_numpy())[int(0.7 * len(rows))]) + 7            # inside a late test day
    rng = np.random.default_rng(1)
    r2, p2 = rows.copy(), paths.copy()
    after = r2["t0"].to_numpy() > cut
    for c in [c for c in r2.columns if c.startswith("f_")]:
        r2.loc[after, c] = rng.normal(0, 3, after.sum()).astype(np.float32)
    unsettled = r2["t1"].to_numpy() > cut
    for c in ("label", "meta_label"):
        r2.loc[unsettled, c] = 1 - r2.loc[unsettled, c]
    r2.loc[unsettled, "up_won"] = 1 - r2.loc[unsettled, "up_won"]
    later = (p2["start"] + p2["t"]).to_numpy() > cut
    p2.loc[later, "q"] = rng.random(later.sum())
    moved = run(r2, p2)
    known = rows["t0"].to_numpy() <= cut
    assert np.isfinite(before[known]).sum() > 50
    np.testing.assert_array_equal(before[known], moved[known])
    changed = ~(np.isclose(before, moved) | (np.isnan(before) & np.isnan(moved)))
    assert changed[~known].any()                                                     # the perturbation does bite


# ---------------------------------------------------------------------------------------------------- permutation

CHANCE = {"bagged_trees": {"n_estimators": 30, "min_weight_fraction_leaf": 0.02}, "logistic": {},
          "hgb": {"max_iter": 50, "min_samples_leaf": 5}, "xgb": {"n_estimators": 50, "min_child_weight": 5.0}}
BETTER_TOL, WORSE_TOL = 0.02, 0.06          # nats: a leak scores better than chance, fitting noise a little worse


@pytest.mark.parametrize("family", FAMILY_PARAMS)
def test_label_permutation_scores_at_chance(tmp_path, family):
    rows, paths = synthetic(days=10, windows=20, events=5, seed=12)
    score = lambda r: md.PurgedSearch(family, {k: [v] for k, v in CHANCE[family].items()}, n_splits=5,
                                      ledger=tmp_path / "l.jsonl").fit(md.Dataset.from_rows(r, paths))
    real = score(rows)
    assert real.best_score_ > np.mean(real.baseline_fold_scores_) + 0.1                # the signal is there
    for seed in (1, 2):
        perm = score(permute_outcomes(rows, seed))
        base = float(np.mean(perm.baseline_fold_scores_))
        assert base - WORSE_TOL <= perm.best_score_ <= base + BETTER_TOL, (perm.best_score_, base)


# ---------------------------------------------------------------------------------------------------- transforms

def other_rows_moved(fit_predict, data, train, test) -> float:
    """Largest change of the predictions of a quarter of the test rows (the probe) when every other test row's features
    are rescaled: zero unless something was fitted on test rows (a scaler, an imputer, ...)."""
    probe, others = test[: len(test) // 4], test[len(test) // 4:]
    X = data.X.copy()
    X[others] = X[others] * 50 + 7
    a = fit_predict(data, train, test)[: len(probe)]
    b = fit_predict(with_X(data, X), train, test)[: len(probe)]
    return float(np.max(np.abs(a - b)))


def scaler_on_all_rows(data, train, test):
    """A logistic model whose imputer and scaler were fitted on training and test rows together."""
    both = np.concatenate([train, test])
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(data.X[both])
    sc = StandardScaler().fit(imp.transform(data.X[both]))
    m = LogisticRegression().fit(sc.transform(imp.transform(data.X[train])), data.y[train])
    return m.predict_proba(sc.transform(imp.transform(data.X[test])))[:, 1]


def test_a_scaler_fitted_outside_the_folds_is_detected(world):
    rows, paths = world
    data = md.Dataset.from_rows(rows, paths)
    train, test = md.day_folds(data, 4)[1]
    assert other_rows_moved(scaler_on_all_rows, data, train, test) > 1e-3                # the probe catches it
    for family in FAMILIES:
        fp = lambda d, tr, te: md.predict_fold(d, family, FAST[family], "static", tr, te, "kfold")
        assert other_rows_moved(fp, data, train, test) == 0.0, family
        m = md.MetaModel(family, FAST[family]).fit(data, train)
        X = data.X.copy()
        X[test] = X[test] * 50 + 7
        np.testing.assert_array_equal(m.predict_proba(data, train), md.MetaModel(family, FAST[family]).fit(
            with_X(data, X), train).predict_proba(data, train))


# ---------------------------------------------------------------------------------------------------- confirmation

def test_no_event_at_or_after_the_confirmation_start_is_ever_loaded(tmp_path, monkeypatch):
    def write_day(day: str, starts):
        rows, paths = synthetic(days=1, windows=len(starts), events=3, seed=len(starts))
        old = np.sort(rows["start"].unique())
        remap = dict(zip(old, starts))
        rows["start"] = rows["start"].map(remap)
        rows["t0"], rows["t1"] = rows["start"] + rows["t"], rows["start"] + T
        paths["start"] = paths["start"].map(remap)
        d = tmp_path / f"day={day}"
        d.mkdir()
        rows.to_parquet(d / "events.parquet")
        paths.to_parquet(d / "paths.parquet")

    utc = lambda s: int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    last = CONFIRM_START - T                                      # ends exactly at the confirmation start
    odd = CONFIRM_START - 100                                     # starts before it, ends after it
    write_day("2026-09-17", [utc("2026-09-17") + 3600, utc("2026-09-17") + 7200])
    write_day("2026-09-18", [utc("2026-09-18") + 3600, last, odd])
    write_day("2026-09-19", [CONFIRM_START, CONFIRM_START + 600])
    opened, real = [], pd.read_parquet
    monkeypatch.setattr(md.pd, "read_parquet", lambda p, *a, **k: opened.append(str(p)) or real(p, *a, **k))
    rows, paths = md.load_events(tmp_path)
    assert opened and not any("2026-09-19" in p for p in opened)
    assert any("2026-09-18" in p for p in opened)
    assert (rows["t1"] <= CONFIRM_START).all() and (rows["t0"] < CONFIRM_START).all()
    assert last in set(rows["start"]) and odd not in set(rows["start"]) and len(rows["start"].unique()) == 4
    assert set(paths["start"]) == set(rows["start"])
    only = md.load_events(tmp_path, days=["2026-09-18", "2026-09-19"], rules=["q_taker"])[0]
    assert set(only["start"]) == {utc("2026-09-18") + 3600, last}
    bad, bad_paths = synthetic(days=1, windows=2, events=3, seed=2)
    shift = CONFIRM_START - T + 60 - bad["start"].max()
    bad[["start", "t0", "t1"]] += shift
    bad_paths["start"] += shift
    with pytest.raises(ValueError, match="confirmation"):
        md.Dataset.from_rows(bad, bad_paths)


# ---------------------------------------------------------------------------------------------------- features

def test_feature_columns_are_only_f_columns_never_labels_spans_or_weights(world):
    rows, paths = world
    rows = rows.assign(weight=1.0, w_return=1.0, w_decay=1.0, edge=0.1, fee=0.0035)
    data = md.Dataset.from_rows(rows, paths)
    names = [c for c in rows.columns if c.startswith("f_")]
    assert data.features == names and data.X.shape == (len(rows), len(names))
    np.testing.assert_array_equal(data.X, rows[names].to_numpy(np.float32))
    for col in ("label", "meta_label", "up_won", "payoff", "pnl", "ret", "t0", "t1", "avg_uniqueness", "w_return_raw",
                "weight", "w_return", "w_decay", "side", "price", "cost", "edge", "fee", "start", "t", "rule"):
        with pytest.raises(ValueError):
            md.Dataset.from_rows(rows, paths, features=["f_signal", col])
    with pytest.raises(ValueError):
        md.Dataset.from_rows(rows, paths, features=["f_signal", "f_missing"])
    sub = md.Dataset.from_rows(rows, paths, features=["f_signal", "f_tau"])
    np.testing.assert_array_equal(sub.X, rows[["f_signal", "f_tau"]].to_numpy(np.float32))
    train, _ = md.day_folds(data, 4)[1]
    for family in FAMILIES:
        assert md.MetaModel(family, FAST[family]).fit(data, train).estimator_.n_features_in_ == len(names)


# ---------------------------------------------------------------------------------------------------- stage 4 script

def load_script():
    spec = importlib.util.spec_from_file_location("meta_label_script",
                                                  Path(__file__).resolve().parents[1] / "scripts/meta_label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_stage_4_sample_never_reads_a_day_that_reaches_the_confirmation_start(tmp_path, monkeypatch):
    """The confirmation set is never opened: a day partition is read only when the whole UTC day ends by
    CONFIRM_START, and only when stage 2 marked it complete."""
    ml = load_script()
    for day, complete in (("2026-09-15", False), ("2026-09-16", True), ("2026-09-17", True), ("2026-09-18", True),
                          ("2026-09-19", True), ("2026-09-20", True)):
        d = tmp_path / f"day={day}"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"day": day, "complete": complete, "rows": 9,
                                                 "rows_by_rule": {"q_taker": 9}}))
    monkeypatch.setattr(ml, "EVENTS", tmp_path)
    got = [m["day"] for m in ml.complete_days()]
    assert got == ["2026-09-16", "2026-09-17", "2026-09-18"]        # 09-18 ends exactly at the confirmation start
    assert datetime.strptime(got[-1], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() + 86400 == CONFIRM_START


def test_the_stage_4_label_permutation_keeps_one_shuffled_outcome_per_window(tmp_path):
    """The control has to break the feature-label link and keep the dependence: the rows of a window must still share
    one outcome, and each row's label must follow its own side."""
    ml = load_script()
    rows, _ = synthetic(days=6, windows=10, events=4, seed=31)
    out = ml.permuted_labels(rows, 11)
    assert list(out.columns) == list(rows.columns) and len(out) == len(rows)
    for start, g in out.groupby("start"):
        assert g["up_won"].nunique() == 1
        np.testing.assert_array_equal(g["label"].to_numpy(),
                                      (g["side"].to_numpy() == np.where(g["up_won"].to_numpy() > 0, 1, -1)).astype(np.int8))
    np.testing.assert_array_equal(out["label"].to_numpy(), out["meta_label"].to_numpy())
    np.testing.assert_array_equal(out["payoff"].to_numpy(), out["label"].to_numpy().astype(float))
    first = rows.groupby("start")["up_won"].first()
    assert sorted(out.groupby("start")["up_won"].first()) == sorted(first)      # a permutation, not a redraw
    assert not out["up_won"].equals(rows["up_won"])
    for c in [c for c in rows.columns if c.startswith("f_")] + ["side", "t0", "t1"]:
        pd.testing.assert_series_equal(out[c], rows[c])                         # only the outcome moves
