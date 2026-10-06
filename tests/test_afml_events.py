"""Leak guards and hand cases for the event dataset (pmlab.afml.events, plan afml-pipeline stage 2).

Every leak test here was mutation-checked: `scripts/afml_events.py --mutations` applies each recorded mutation to a
copy of the library, runs this file against it and requires a failure (research/afml_events.json, "mutations").
"""
import copy
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab import vol
from pmlab.afml import events as ev
from pmlab.bars import NINE, calibrate
from pmlab.evaluation import CONFIRM_START
from pmlab.events import trades_from_tape
from pmlab.fair import FairParams
from pmlab.features import bar_features, run_bars
from pmlab.fees import fee_per_share
from pmlab.micro import bar_micro_features
from pmlab.polymarket import in_window

START = 1_788_000_000                     # a window start (multiple of 300) well before the confirmation start


def make_close(start, rng, before=ev.HISTORY + 1200, after=600, sigma=5e-5):
    opens = np.arange(start - before, start + T + after)
    r = rng.normal(0, sigma, len(opens)) * np.where(rng.random(len(opens)) < 0.02, 6.0, 1.0)
    return pd.Series(60_000 * np.exp(np.cumsum(r)), index=opens)


def make_tape(start, rng, lo=-120, extra=900):
    ts = np.sort(np.concatenate([np.arange(start + lo, start + T), rng.integers(start + lo, start + T, extra)]))
    n = len(ts)
    mid = np.clip(0.5 + np.cumsum(rng.normal(0, 0.006, n)), 0.06, 0.94)
    sign = rng.choice([1, -1], n)
    p_up = np.clip(np.round(mid + 0.01 * sign, 2), 0.01, 0.99)
    shares = rng.exponential(25.0, n)
    return pd.DataFrame({"ts": ts.astype(np.int64), "t": (ts - start).astype(np.int64), "p_up": p_up,
                         "sign": sign.astype(np.int8), "shares": shares,
                         "usdc": np.where(sign > 0, p_up, 1 - p_up) * shares})


def make_book(rng):
    t = np.arange(T)
    ask = np.round(rng.uniform(0.3, 0.7, T), 2)
    return pd.DataFrame({"t": t, "ask": ask, "bid": ask - 0.01, "spread": np.full(T, 0.01),
                         "depth_imb": rng.uniform(-1, 1, T), "microprice": ask - 0.005,
                         "ask_size": rng.exponential(100, T), "bid_size": rng.exponential(100, T)})


@dataclass
class World:
    close: pd.Series
    tape: pd.DataFrame
    book: pd.DataFrame
    ctx: ev.DayContext
    builders: dict
    up_won: float = 1.0

    def rows(self, close=None, tape=None, book=None, up_won=None, start=START):
        return ev.window_rows(start, self.tape if tape is None else tape, self.close if close is None else close, 60,
                              self.up_won if up_won is None else up_won, self.ctx, copy.deepcopy(self.builders),
                              self.book if book is None else book)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)
    cal_windows = {START - 300 * k: trades_from_tape(in_window(make_tape(START - 300 * k, rng)), START - 300 * k)
                   for k in (1, 2, 3)}
    builders = ev.make_builders(calibrate(cal_windows))
    ev.warm_builders(builders, cal_windows)
    har = vol.HAR(vol.HAR_SETS["seasonal"])
    har.coef, har.resid_var, har.n = np.array([0.0, 1.0, 0.0, 0.0, 0.0]), 0.0, 1
    ctx = ev.DayContext(FairParams(120.0, 1.2, 2e-5, 2, 1e-5), har, np.full(288, 2.5e-9))
    return World(make_close(START, rng), make_tape(START, rng), make_book(rng), ctx, builders)


def decision_columns(rows):
    """Everything in a row except what is decided at settlement: labels and the weights of later concurrency."""
    return [c for c in rows.columns if c not in ev.LABEL_COLUMNS + ev.WEIGHT_COLUMNS]


# ---------------------------------------------------------------------------------------------------- leak guards

@pytest.mark.property
@pytest.mark.parametrize("remove", [False, True], ids=["perturb", "remove"])
def test_perturbing_or_removing_anything_after_an_event_changes_nothing_known_at_it(world, remove):
    base, _, _ = world.rows()
    assert len(base) > 10 and set(base["rule"]) == set(ev.RULES)
    cols = decision_columns(base)
    assert set(ev.FEATURE_NAMES) <= set(cols)
    rng = np.random.default_rng(11)
    labels_changed = False
    for cut in sorted(set(base["t"].tolist())):
        after_c = world.close.index >= START + cut         # klines opened at >= start + cut: instants after the event
        late = world.tape["ts"] >= START + cut               # prints stamped >= start + cut: known after the event
        late_b = world.book["t"] > cut
        if remove:
            close, tape, book = world.close[~after_c], world.tape[~late], world.book[~late_b]
        else:
            close = world.close.copy()
            close[after_c] *= np.exp(rng.normal(0, 3e-4, int(after_c.sum())))
            tape = world.tape.copy()
            n = int(late.sum())
            tape.loc[late, "p_up"] = rng.uniform(0.02, 0.98, n).round(2)
            tape.loc[late, "sign"] = rng.choice([1, -1], n).astype(np.int8)
            tape.loc[late, "shares"] = rng.exponential(200.0, n)
            tape.loc[late, "usdc"] = tape.loc[late, "p_up"] * tape.loc[late, "shares"]
            extra = tape[late].sample(min(n, 50), random_state=1).assign(shares=999.0)
            tape = pd.concat([tape, extra]).sort_values("ts", kind="stable").reset_index(drop=True)
            book = world.book.copy()
            book.loc[late_b, ["ask", "bid", "spread", "depth_imb"]] = rng.uniform(0, 1, (int(late_b.sum()), 4))
        got, _, _ = world.rows(close=close, tape=tape, book=book, up_won=1.0 - world.up_won)
        a = base[base["t"] <= cut][cols].reset_index(drop=True)
        b = got[got["t"] <= cut][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        labels_changed |= bool((got[got["t"] <= cut]["label"].to_numpy() != base[base["t"] <= cut]["label"].to_numpy()).any())
    assert labels_changed                                  # the outcome did change: the comparison can see a change


@pytest.mark.case
def test_no_event_or_label_uses_a_window_at_or_after_the_confirmation_start(world):
    starts = [CONFIRM_START - 900, CONFIRM_START - 600, CONFIRM_START - 300, CONFIRM_START - 150, CONFIRM_START,
              CONFIRM_START + 300]                         # - 150 starts before the cut and ends after it
    assert ev.eligible_starts(starts) == [CONFIRM_START - 900, CONFIRM_START - 600, CONFIRM_START - 300]
    assert ev.eligible_starts(starts, confirm_start=CONFIRM_START - 300) == [CONFIRM_START - 900, CONFIRM_START - 600]
    for s in (CONFIRM_START, CONFIRM_START - 150):
        close = make_close(s, np.random.default_rng(1))
        with pytest.raises(ValueError, match="confirmation"):
            world.rows(close=close, tape=make_tape(s, np.random.default_rng(2)), start=s)


@pytest.mark.case
def test_every_span_starts_at_its_event_and_ends_at_the_window_end(world):
    rows, q, _ = world.rows()
    assert (rows["t0"] == START + rows["t"]).all()
    assert (rows["t1"] == START + T).all()
    assert ((rows["t"] >= ev.T_MIN) & (rows["t"] <= ev.T_MAX)).all()
    assert len(q) == T + 1 and q[-1] == world.up_won
    u, w = ev.window_span_weights(rows["t0"] - START, rows["t1"] - START, q)
    assert rows["avg_uniqueness"].to_numpy() == pytest.approx(u)
    assert rows["w_return_raw"].to_numpy() == pytest.approx(w)


def _events_table(rng, windows=6, per_window=12):
    parts, paths = [], {}
    for k in range(windows):
        s = START + 300 * k * 50                                          # windows on different days
        t = np.sort(rng.choice(np.arange(ev.T_MIN, ev.T_MAX + 1), per_window, replace=False))
        parts.append(pd.DataFrame({"start": s, "t": t, "t0": s + t, "t1": s + T, "day": str(s // 86400)}))
        paths[s] = np.concatenate([np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, T)), 0, 1), [float(rng.random() < 0.5)]])
    return pd.concat(parts, ignore_index=True), paths


@pytest.mark.property
def test_weights_are_computed_from_training_spans_only_when_a_split_is_given():
    rng = np.random.default_rng(3)
    events, paths = _events_table(rng)
    train = rng.random(len(events)) < 0.6
    w = ev.sample_weights(events, paths, train)
    assert w.loc[~train].isna().all().all() and w.loc[train].notna().all().all()
    # the test rows can be anything: moved, or gone
    moved = events.copy()
    moved.loc[~train, "t0"] = moved.loc[~train, "start"] + ev.T_MIN
    w_moved = ev.sample_weights(moved, paths, train)
    w_dropped = ev.sample_weights(events[train].reset_index(drop=True), paths)
    pd.testing.assert_frame_equal(w.loc[train], w_moved.loc[train])
    np.testing.assert_allclose(w.loc[train].to_numpy(), w_dropped.to_numpy())
    assert w.loc[train, "w_return"].sum() == pytest.approx(train.sum())
    full = ev.sample_weights(events, paths)
    assert not np.allclose(full.loc[train, "avg_uniqueness"], w.loc[train, "avg_uniqueness"])


@pytest.mark.property
def test_purged_kfold_drops_every_training_span_that_overlaps_a_test_span_and_embargoes_after_it():
    rng = np.random.default_rng(5)
    rows = []
    for d in range(8):                      # 8 days, a few windows each, spans to the window end
        for w in rng.choice(288, 8, replace=False):
            s = START + d * 86400 + int(w) * 300
            for t in rng.choice(np.arange(5, 281), 3, replace=False):
                rows.append((s, int(t), s + int(t), s + T, f"d{d}"))
    # a few long spans crossing days, so purging has something to do
    for d in range(1, 8):
        s = START + d * 86400 - 600
        rows.append((s, 0, s, s + 1200, f"d{d - 1}"))
    events = pd.DataFrame(rows, columns=["start", "t", "t0", "t1", "day"]).sort_values("t0").reset_index(drop=True)
    embargo = 20_000
    folds = ev.purged_kfold(events, n_splits=4, embargo=embargo)
    assert len(folds) == 4
    t0, t1 = events["t0"].to_numpy(), events["t1"].to_numpy()
    dropped_by_purge = dropped_by_embargo = 0
    for train, test in folds:
        is_test = np.zeros(len(events), bool)
        is_test[test] = True
        assert set(events.loc[is_test, "day"]).isdisjoint(set(events.loc[train, "day"]))
        for i in range(len(events)):
            if is_test[i]:
                continue
            overlap = bool(np.any((t0[i] < t1[is_test]) & (t1[i] > t0[is_test])))
            ends = t1[is_test]
            embargoed = bool(np.any((t0[i] >= ends) & (t0[i] < ends + embargo)))
            assert (i in set(train)) == (not overlap and not embargoed)
            dropped_by_purge += overlap
            dropped_by_embargo += embargoed and not overlap
    assert dropped_by_purge > 0 and dropped_by_embargo > 0


@pytest.mark.property
def test_day_models_read_nothing_from_the_day_they_serve():
    rng = np.random.default_rng(9)
    day0 = (START // 86400 + 10) * 86400
    inv = pd.DataFrame({"start": np.arange(day0 - 5 * 86400, day0 + 86400, 300)})
    inv["L"] = np.where(inv["start"] < day0 - 2 * 3600, 1, 60)
    rules = inv.set_index("start")["L"]
    for L, used in ((1, {1}), (60, {1, 60})):             # rule 60 has under half a day before the purge: every rule
        tr = ev.fair_training_starts(inv, day0, L)
        assert max(tr) < day0 - ev.FAIR_PURGE and min(tr) >= day0 - ev.FAIR_TRAIN_DAYS * 86400
        assert set(rules.loc[tr]) == used
    cal = ev.calibration_starts(inv["start"], day0)
    assert len(cal) == ev.CAL_WINDOWS and max(cal) + T <= day0
    opens = np.arange(day0 - ev.HAR_TRAIN_DAYS * 86400 - ev.HISTORY - 10, day0 + 3600)
    close = pd.Series(60_000 * np.exp(np.cumsum(rng.normal(0, 4e-5, len(opens)))), index=opens)
    har, profile, meta = ev.fit_har(close, day0, 120.0)
    later = close.copy()
    later[later.index >= day0 - 1] *= 1.05 * np.exp(rng.normal(0, 1e-3, int((later.index >= day0 - 1).sum())))
    har2, profile2, _ = ev.fit_har(later, day0, 120.0)
    assert np.array_equal(har.coef, har2.coef) and np.array_equal(profile, profile2)
    assert meta["last_instant"] <= day0 - 1


# ---------------------------------------------------------------------------------------------------- hand cases

@pytest.mark.case
def test_cusum_filter_hand_example():
    # S+ = 2, 4, 6 > 5 -> +1 at 3; S- = -1, -4, -7 < -5 -> -1 at 6; S+ = 4, 6 -> +1 at 9; S+ = 5 (not > 5), 5.1 -> +1 at 11
    z = np.array([np.nan, 2, 2, 2, -1, -3, -3, 0, 4, 2, 5, 0.1, 0])
    t, s = ev.cusum_events(z)
    assert t.tolist() == [3, 6, 9, 11] and s.tolist() == [1, -1, 1, 1]


@pytest.mark.case
def test_cusum_standardises_each_return_by_the_sigma_known_before_it():
    # a long run of returns of size a, then 3a, 3a: sigma before the first is a, before the second a*sqrt(1 + 8 alpha)
    a, n = 4e-5, 20_000
    start = 10_000_000_200
    r = np.concatenate([np.full(n, a), [3 * a, 3 * a], np.zeros(T)])
    opens = np.arange(start - n, start - n + len(r))           # instant start + 1 carries the first 3a return
    close = pd.Series(np.exp(np.cumsum(r)) * 50_000, index=opens)
    tl = vol.timeline(close)
    z = ev.cusum_z(tl, start)
    alpha = 1 - 0.5 ** (1 / ev.CUSUM_HALFLIFE)
    assert math.isnan(z[0]) and z[1] == pytest.approx(3.0, rel=1e-9)
    assert z[2] == pytest.approx(3.0 / math.sqrt(1 + 8 * alpha), rel=1e-6)
    t, s = ev.cusum_events(z)
    assert t.tolist() == [2] and s.tolist() == [1]                 # 3 + 2.93 > 5


@pytest.mark.case
def test_uniqueness_and_return_attribution_hand_example():
    q = np.arange(T + 1) / T                                       # r_u = 1/300 every second
    u, w = ev.window_span_weights([0, 100, 200], [300, 300, 300], q)
    assert u.tolist() == pytest.approx([(100 + 50 + 100 / 3) / 300, (50 + 100 / 3) / 200, (100 / 3) / 100])
    assert w.tolist() == pytest.approx([(1 + 1 / 2 + 1 / 3) / 3, (1 / 2 + 1 / 3) / 3, (1 / 3) / 3])


@pytest.mark.case
def test_primary_rule_sides_hand_cases():
    fair = np.array([0.60, 0.50, 0.90])
    ask_up, ask_age = np.array([0.55, 0.52, 0.50]), np.array([2.0, 2.0, 30.0])
    bid_up, bid_age = np.array([0.50, 0.49, np.nan]), np.array([2.0, 2.0, np.nan])
    flow = np.array([-0.3, np.nan, 0.2])
    s = ev.rule_sides(fair, ask_up, ask_age, bid_up, bid_age, flow)
    assert s["q_taker"].tolist() == [1, 0, 0]          # 0.60 - 0.55 > 0; no gap; stale ask
    assert s["fair_maker"].tolist() == [-1, -1, -1]    # Up bid 0.57 would cross 0.55; both quotable, Down closer; Up crosses
    assert s["flow"].tolist() == [-1, 0, 1]


@pytest.mark.case
def test_labels_are_the_sides_settlement_pnl_after_the_taker_fee(world):
    for up_won in (0.0, 1.0):
        rows, _, _ = world.rows(up_won=up_won)
        payoff = np.where(rows["side"] > 0, up_won, 1 - up_won)
        pnl = payoff - rows["price"] - fee_per_share(rows["price"].to_numpy())
        assert rows["pnl"].to_numpy() == pytest.approx(pnl)
        assert (rows["label"] == (pnl > 0)).all() and (rows["meta_label"] == rows["label"]).all()
        # with prices capped at 0.98 the fee never turns a winning side into a loss: the label is "the side won"
        assert (rows["label"] == payoff).all()
        up = rows["side"] > 0
        assert (rows.loc[up, "price"] == rows.loc[up, "ask_up"]).all()
        assert rows.loc[~up, "price"].to_numpy() == pytest.approx(1 - rows.loc[~up, "bid_up"].to_numpy())
        assert (rows["price_age"] <= ev.MAX_QUOTE_AGE).all()
        assert rows["label"].nunique() == 2


@pytest.mark.property
def test_bar_features_at_event_seconds_equal_the_pipelines(world):
    rng = np.random.default_rng(13)
    tape = make_tape(START, rng)
    trades = trades_from_tape(in_window(tape), START)
    b1, b2 = copy.deepcopy(world.builders), copy.deepcopy(world.builders)
    known = ev.run_window_bars(b1, START, trades)
    seconds = np.arange(T)
    for kind in NINE:
        ref = run_bars(b2[kind], START, tape)
        assert [(s, b.close, b.n_ticks) for s, b in known[kind]] == [(s, b.close, b.n_ticks) for s, b in ref]
        want = bar_micro_features(kind, ref, START)
        got = ev.micro_at(kind, known[kind], START, seconds)
        for k, v in want.items():
            np.testing.assert_array_equal(got[k], v)
        np.testing.assert_array_equal(bar_features(kind, known[kind], START)[f"{kind}_imb"],
                                      bar_features(kind, ref, START)[f"{kind}_imb"])


@pytest.mark.case
def test_a_build_survives_an_outcome_neutral_library_change_but_not_any_other(tmp_path, monkeypatch):
    """events.EMBARGO_S changed events.py's hash without moving a row: those builds stay valid, unknown hashes do not."""
    import scripts.afml_events as A
    old = next(iter(A.EQUIVALENT_CODE))
    inv = pd.DataFrame({"day": ["2026-09-01"], "start": [1]})
    monkeypatch.setattr(A, "day_dir", lambda d: tmp_path)
    (tmp_path / "events.parquet").write_bytes(b"")
    meta = {"version": A.ev.VERSION, "code_sha256": old, "complete": True, "windows": [1]}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert A.is_done("2026-09-01", inv, "new-hash")
    (tmp_path / "meta.json").write_text(json.dumps({**meta, "code_sha256": "something-else"}))
    assert not A.is_done("2026-09-01", inv, "new-hash")


@pytest.mark.case
def test_merged_test_stretches_are_bit_identical_to_the_sweep(rng=np.random.default_rng(20260919)):
    """`_merge` is a running maximum instead of a sweep over sorted spans (plan performance, node 7).

    purged_kfold purges training rows against these stretches, so a stretch that shifted by a second would change a
    fold and every number downstream of it. Equality with `_merge_loop`, on spans that overlap, nest, touch and
    repeat, is what makes the 21x free.
    """
    for _ in range(60):
        n = int(rng.integers(1, 200))
        t0 = rng.integers(0, 500, n).astype(np.int64)
        t1 = t0 + rng.integers(1, 60, n)
        lo, hi = ev._merge(t0, t1)
        lo0, hi0 = ev._merge_loop(t0, t1)
        assert np.array_equal(lo, lo0) and np.array_equal(hi, hi0)
        assert lo.dtype == lo0.dtype == np.int64 and hi.dtype == hi0.dtype == np.int64
    empty = ev._merge(np.empty(0, np.int64), np.empty(0, np.int64))
    assert len(empty[0]) == 0 and len(empty[1]) == 0 and empty[0].dtype == np.int64
