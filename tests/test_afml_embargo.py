"""The default embargo of every purged split over the event dataset outlasts the day models that read outcomes
(research/code_review/afml.md, events.py finding 1).

A UTC day's fair parameters are fitted on the settled windows of the events.FAIR_TRAIN_DAYS days before it
(events.fair_training_starts), and every price feature of that day depends on them. A training row whose day was
fitted on windows of its fold's test block therefore carries test outcomes. The splits checked here are the defaults of
events.purged_kfold (UTC days), models.day_folds and models.PurgedSearch (Taiwan days) and
backtest_pipeline.run_cpcv (Taiwan-day CPCV groups). Each check is also run with a one-day embargo, which it must
catch, so the guard can fail."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import backtest_pipeline as bp
from pmlab.afml import events as ev
from pmlab.afml import models as md

DAY = 86400
DAY0 = int(datetime(2026, 4, 1, tzinfo=timezone(timedelta(hours=8))).timestamp())      # a Taiwan midnight
L = 60


def frame(days: int = 24, windows: int = 10, per_window: int = 2, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Event-like rows over Taiwan days, each day holding its first (00:00) and last (23:55 Taiwan) window, so test
    blocks end at a Taiwan midnight (16:00 UTC), the hardest case for a UTC-day model."""
    rng = np.random.default_rng(seed)
    recs, paths = [], {}
    for d in range(days):
        slots = np.r_[0, np.sort(rng.choice(np.arange(1, 287), windows - 2, replace=False)), 287]
        for slot in slots:
            start = DAY0 + d * DAY + 300 * int(slot)
            up = float(rng.random() < 0.5)
            paths[start] = np.concatenate([np.clip(0.5 + np.cumsum(rng.normal(0, 0.01, T)), 0.01, 0.99), [up]])
            for t in np.sort(rng.choice(np.arange(ev.T_MIN, ev.T_MAX + 1), per_window, replace=False)):
                side = int(rng.choice([-1, 1]))
                payoff = up if side > 0 else 1.0 - up
                recs.append({"start": start, "t": int(t), "t0": start + int(t), "t1": start + T,
                             "day": pd.Timestamp(start, unit="s").strftime("%Y-%m-%d"), "rule": "q_taker",
                             "side": side, "price": 0.5, "fee": 0.0175, "cost": 0.5175, "up_won": up,
                             "payoff": payoff, "meta_label": int(payoff), "f_x": rng.normal()})
    return pd.DataFrame(recs), paths


def contaminated(starts: np.ndarray, train, test) -> list[int]:
    """UTC days of training rows whose fair parameters were fitted on a window of the test rows."""
    inventory = pd.DataFrame({"start": np.unique(starts), "L": L})
    test_windows = set(np.unique(starts[np.asarray(test)]).tolist())
    bad = []
    for day0 in np.unique(starts[np.asarray(train)] // DAY * DAY):
        if set(ev.fair_training_starts(inventory, int(day0), L)) & test_windows:
            bad.append(int(day0))
    return bad


def fold_contamination(starts, folds) -> int:
    return sum(len(contaminated(starts, train, test)) for train, test in folds)


@pytest.mark.case
def test_the_embargo_is_derived_from_the_fair_parameter_lookback_everywhere():
    assert ev.EMBARGO_S == (ev.FAIR_TRAIN_DAYS + 1) * DAY
    assert md.EMBARGO == ev.EMBARGO_S
    assert bp.EMBARGO == ev.EMBARGO_S


@pytest.mark.property
def test_no_training_row_of_a_default_utc_day_fold_has_fair_parameters_fitted_on_its_test_block():
    rows, _ = frame()
    starts = rows["start"].to_numpy(np.int64)
    assert fold_contamination(starts, ev.purged_kfold(rows, n_splits=6)) == 0
    assert fold_contamination(starts, ev.purged_kfold(rows, n_splits=6, embargo=DAY)) >= 5 * 2   # 2 days after each inner block


@pytest.mark.property
def test_no_training_row_of_the_meta_label_models_default_folds_has_fair_parameters_fitted_on_its_test_block(tmp_path):
    rows, paths = frame()
    data = md.Dataset.from_rows(rows, paths)
    search = md.PurgedSearch("logistic", {"C": [1.0]}, n_splits=6, ledger=tmp_path / "l.jsonl")
    assert fold_contamination(data.start, md.day_folds(data, 6)) == 0
    assert fold_contamination(data.start, md.day_folds(data, 6, search.embargo)) == 0
    assert fold_contamination(data.start, md.day_folds(data, 6, embargo=DAY)) >= 5 * 2


@pytest.mark.property
def test_no_training_row_of_a_default_cpcv_split_has_fair_parameters_fitted_on_its_test_groups():
    rows, _ = frame(days=30)
    seen = []

    def spy(train, test):
        seen.append((train["start"].to_numpy(np.int64), test["start"].to_numpy(np.int64)))
        return np.full(len(test), 0.6)

    def dirty(**kw) -> int:
        seen.clear()
        bp.run_cpcv(rows, bp.Strategy("spy", model=spy, threshold=0.5), n_groups=6, k_test=2, **kw)
        assert len(seen) == 15
        n = 0
        for tr, te in seen:
            starts = np.concatenate([tr, te])
            n += len(contaminated(starts, np.arange(len(tr)), np.arange(len(tr), len(starts))))
        return n

    assert dirty() == 0
    assert dirty(embargo=DAY) > 0
