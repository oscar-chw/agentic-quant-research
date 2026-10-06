"""Per-event structural-break features (plan afml-pipeline node 23): the grid and reading point, the alignment of
each statistic with its mark, the side-frame pairs, availability, and the leak guards — nothing at or after the event
second is read, and nothing at or after the confirmation start.

Every assertion here is mutation-checked: `python scripts/break_features.py --mutations` applies each mutation of
scripts/break_features.MUTATIONS to a copy of pmlab.afml.break_features and requires this file to fail.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import break_features as bf
from pmlab.afml import breaks as bk
from pmlab.evaluation import CONFIRM_START

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("break_features_script", ROOT / "scripts/break_features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


START = 1_700_000_000 - 1_700_000_000 % T          # a window start well before the confirmation start


LEVEL = 100_000.0        # BTC's own order of magnitude: log 11.5 against a per-mark signal of 1e-3, which is what
                         # makes the conditioning of the trend solves worth testing (see the truncation test)


def klines(start: int, seed: int = 3, extra: int = 0, level: float = LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """1 s klines covering the grid: the kline opened at second s closes at s + 1."""
    lo, hi = start - bf.LOOKBACK - 5, start + T + extra
    sec = np.arange(lo, hi, dtype=np.int64)
    rng = np.random.default_rng(seed)
    close = level * np.exp(np.cumsum(rng.normal(0, 2e-4, len(sec))))
    return sec, close


def tape(start: int, seed: int = 5, step: int = 3, extra: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Prints every `step` seconds over the grid, in the Up frame."""
    ts = np.arange(start - bf.LOOKBACK - 5, start + T + extra, step, dtype=np.int64)
    rng = np.random.default_rng(seed)
    p = np.clip(0.5 + np.cumsum(rng.normal(0, 0.004, len(ts))), 0.02, 0.98)
    return ts, p


def grids(start: int, **kw):
    g = bf.marks(start)
    ksec, kclose = klines(start, **{k: v for k, v in kw.items() if k in ("seed", "extra", "level")})
    ts, p = tape(start, **{k: v for k, v in kw.items() if k in ("step", "extra")})
    tok, age = bf.token_marks(g, ts, p)
    return g, bf.btc_marks(g, ksec, kclose), tok, age


# ---------------------------------------------------------------------------------------------------- grid

def test_the_reading_mark_is_the_last_one_at_or_before_the_event_second():
    g = bf.marks(START)
    assert len(g) == bf.N_MARKS
    for t in range(0, T):
        j = int(bf.mark_index(START, t))
        assert 0 <= j < len(g)
        assert g[j] <= START + t, "a mark at or after the event second is read"
        assert j + 1 >= len(g) or g[j + 1] > START + t, "an earlier mark than the last usable one is read"
        assert START + t - g[j] < bf.MARK_S


def test_the_grid_starts_a_whole_lookback_before_the_window_and_ends_inside_it():
    g = bf.marks(START)
    assert g[0] == START - bf.LOOKBACK
    assert g[-1] < START + T


# ---------------------------------------------------------------------------------------------------- alignment

def test_each_statistic_is_the_one_its_own_library_gives_for_the_window_ending_at_that_mark():
    g, btc, tok, age = grids(START)
    t = 200
    j = int(bf.mark_index(START, t))
    row = bf.window_features(START, np.array([t]), btc, tok, age).iloc[0]

    log_btc = np.log(btc)
    direct = bk.bsadf(log_btc, min_len=bf.MIN_LEN, lags=bf.LAGS)["bsadf"]
    assert row["b17_btc_sadf"] == pytest.approx(direct[j - bf.LAGS - 1], rel=1e-6)
    csw = bk.csw_cusum(log_btc, b_alpha=bf.B_ALPHA)
    assert row["b17_btc_csw"] == pytest.approx(csw["s_first"][j], rel=1e-6)
    assert row["b17_btc_csw_pos"] == pytest.approx(csw["s_sup"][j], rel=1e-6)
    assert row["b17_btc_smt_exp"] == pytest.approx(
        bk.smt(btc, spec=bf.SMT_SPEC, min_len=bf.MIN_LEN, phi=bf.SMT_PHI)["smt"][j], rel=1e-6)

    lg = bf.logit(tok)
    assert row["b17_tok_sadf"] == pytest.approx(
        bk.bsadf(lg, min_len=bf.MIN_LEN, lags=bf.LAGS)["bsadf"][j - bf.LAGS - 1], rel=1e-6), \
        "the token's ADF test must run on the logit of its bounded price"
    assert row["b17_tok_csw"] == pytest.approx(bk.csw_cusum(lg, b_alpha=bf.B_ALPHA)["s_first"][j], rel=1e-6)
    assert row["b17_tok_smt_up"] == pytest.approx(
        bk.smt(tok, spec=bf.SMT_SPEC, min_len=bf.MIN_LEN, phi=bf.SMT_PHI)["smt"][j], rel=1e-6)
    assert row["b17_tok_smt_dn"] == pytest.approx(
        bk.smt(1.0 - tok, spec=bf.SMT_SPEC, min_len=bf.MIN_LEN, phi=bf.SMT_PHI)["smt"][j], rel=1e-6)


def test_the_two_cusum_directions_are_mirror_images_and_the_ratio_is_two_sided():
    _, btc, _, _ = grids(START)
    y = np.log(btc)
    up, dn = bf.csw_path(y), bf.csw_path(-y)
    fin = np.isfinite(up["csw_pos"]) & np.isfinite(dn["csw_neg"])
    assert fin.any()
    assert np.allclose(up["csw_pos"][fin], dn["csw_neg"][fin], rtol=1e-9, atol=1e-9)
    assert np.allclose(up["csw"][fin], -dn["csw"][fin], rtol=1e-9, atol=1e-9)
    two = bk.csw_cusum(y, b_alpha=bf.B_ALPHA, two_sided=True)["ratio_sup"]
    ok = np.isfinite(two) & np.isfinite(up["csw_ratio"])
    assert ok.any()
    assert np.allclose(up["csw_ratio"][ok], two[ok], rtol=1e-9, atol=1e-9), \
        "the ratio must be the supremum over both directions"


def test_a_rising_series_breaks_upward_and_a_falling_one_downward():
    ramp = np.linspace(0.0, 1.0, bf.N_MARKS) + 1e-6 * np.arange(bf.N_MARKS) % 3
    up = bf.csw_path(ramp)
    j = bf.N_MARKS - 1
    assert up["csw_pos"][j] > up["csw_neg"][j], "an upward ramp must break upward"
    dn = bf.csw_path(-ramp)
    assert dn["csw_neg"][j] > dn["csw_pos"][j]


def test_the_either_token_trend_is_the_larger_of_the_two_tokens():
    _, btc, tok, age = grids(START)
    f = bf.window_features(START, np.arange(20, 280, 7), btc, tok, age)
    a, b, m = (f[c].to_numpy(float) for c in ("b17_tok_smt_up", "b17_tok_smt_dn", "b17_tok_smt_exp"))
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.any()
    assert np.allclose(m[ok], np.maximum(a[ok], b[ok]), rtol=1e-6)
    assert (a[ok] != b[ok]).any(), "the two tokens must not give the same statistic on this sample"


def test_no_stored_value_is_an_infinity():
    """`breaks.bsadf` marks a mark whose every window was singular with -inf. That means "no statistic", and an
    infinity in a stored column is not a value a model may read (scikit-learn refuses one outright)."""
    g = bf.marks(START)
    ts = np.arange(g[0], g[-1] + 1, dtype=np.int64)
    flat_then_move = np.concatenate([np.full(len(ts) - 30, 0.4), np.linspace(0.4, 0.6, 30)])
    tok, age = bf.token_marks(g, ts, flat_then_move)
    btc = np.concatenate([np.full(len(g) - 30, LEVEL), np.linspace(LEVEL, 1.01 * LEVEL, 30)])
    f = bf.window_features(START, np.arange(10, 280, 11), btc, tok, age)
    for c in bf.FEATURE_NAMES:
        x = f[c].to_numpy(float)
        assert not np.isinf(x).any(), f"{c} stored an infinity"
    assert np.isnan(f["b17_tok_sadf"].to_numpy(float)).any(), "the flat stretch must leave marks without a statistic"


def test_the_token_staleness_column_counts_the_seconds_since_the_last_print():
    g = bf.marks(START)
    ts = np.array([g[0] - 1, g[0] + 97], dtype=np.int64)
    _, age = bf.token_marks(g, ts, np.array([0.4, 0.6]))
    assert age[0] == pytest.approx(np.log1p(1.0))
    j = 30                                                    # mark g[0] + 150, 53 s after the second print
    assert age[j] == pytest.approx(np.log1p(float(g[j] - ts[1])))


def test_a_series_that_begins_late_is_placed_from_its_own_first_mark():
    g = bf.marks(START)
    ts, p = tape(START)
    late = g[0] + 400
    tok, _ = bf.token_marks(g, ts[ts >= late], p[ts >= late])
    first = int(np.argmax(np.isfinite(tok)))
    assert first > 0
    trimmed = bf.sadf_path(bf.logit(tok))
    direct = bk.bsadf(bf.logit(tok[first:]), min_len=bf.MIN_LEN, lags=bf.LAGS)["bsadf"]
    fin = np.isfinite(trimmed)
    assert fin.any()
    assert np.allclose(trimmed[fin], direct[np.flatnonzero(fin) - first - bf.LAGS - 1], rtol=1e-9, atol=1e-9)
    assert not np.isfinite(trimmed[:first + bf.MIN_LEN]).any(), "no statistic before the series begins"


def test_a_flat_token_series_yields_no_break():
    """A token that never prints a new price has no statistic at all: the ADF and CUSUM tests divide by the movement
    that is not there, and the trend tests fit an exactly flat line, which is a singular solve once the series is
    scaled by its own first value."""
    g = bf.marks(START)
    ts = np.arange(g[0], g[-1] + 1, 3, dtype=np.int64)
    tok, age = bf.token_marks(g, ts, np.full(len(ts), 0.4))
    f = bf.window_features(START, np.array([250]), np.full(len(g), LEVEL), tok, age).iloc[0]
    for c in bf.SERIES["tok"]:
        if c == "b17_tok_age":
            assert np.isfinite(f[c]), "the age of the last print is known however flat it was"
            continue
        assert np.isnan(f[c]), f"{c} must be NaN, never an infinity or a spurious 0, on a series that never moves"
    assert np.isnan(f["b17_btc_smt_exp"]), "a flat BTC series has no trend statistic either"


# ---------------------------------------------------------------------------------------------------- availability

def test_token_statistics_need_the_minimum_number_of_marks():
    g = bf.marks(START)
    ts, p = tape(START)
    late = g[0] + bf.MARK_S * (bf.N_MARKS - 40)                 # the first print arrives 40 marks before the close
    tok, age = bf.token_marks(g, ts[ts >= late], p[ts >= late])
    first = int(np.argmax(np.isfinite(tok)))
    ksec, kclose = klines(START)
    btc = bf.btc_marks(g, ksec, kclose)
    ts_first = int(g[first])
    early = ts_first - START + bf.MARK_S * (bf.MIN_LEN - 1)     # fewer than MIN_LEN + 2 marks known
    late_t = ts_first - START + bf.MARK_S * (bf.MIN_LEN + 4)
    f = bf.window_features(START, np.array([early, late_t]), btc, tok, age)
    assert f["b17_tok_ok"].tolist() == [0.0, 1.0]
    assert not np.isfinite(f.loc[0, bf.SERIES["tok"]].to_numpy(float)).any()
    assert np.isfinite(f.loc[1, "b17_tok_csw"])
    assert np.isfinite(f.loc[0, "b17_btc_csw"]), "the BTC grid is complete and must not be blanked with the token's"


# ---------------------------------------------------------------------------------------------------- leaks

@pytest.mark.parametrize("how", ["perturb", "remove"])
def test_perturbing_or_removing_anything_after_an_event_changes_nothing_known_at_it(how):
    """An event at second e reads klines opened before its mark and prints stamped before it; everything at or after
    that mark may change without moving a single feature."""
    g = bf.marks(START)
    ksec, kclose = klines(START, extra=120)
    ts, p = tape(START, step=1, extra=120)                 # a print at every second, so a print sits on the mark too
    t = 173
    j = int(bf.mark_index(START, t))
    m = int(g[j])
    assert (ts == m).any() and (ksec == m).any(), "the fixture must offer data exactly at the mark to perturb"
    before = bf.window_features(START, np.array([t]), bf.btc_marks(g, ksec, kclose), *bf.token_marks(g, ts, p))

    rng = np.random.default_rng(99)
    if how == "remove":
        keep_k, keep_t = ksec < m, ts < m
        ksec2, kclose2, ts2, p2 = ksec[keep_k], kclose[keep_k], ts[keep_t], p[keep_t]
    else:
        ksec2, kclose2, ts2, p2 = ksec.copy(), kclose.copy(), ts.copy(), p.copy()
        hit_k, hit_t = ksec2 >= m, ts2 >= m
        kclose2[hit_k] = kclose2[hit_k] * np.exp(rng.normal(0, 0.05, int(hit_k.sum())))
        p2[hit_t] = np.clip(rng.uniform(0.02, 0.98, int(hit_t.sum())), 0.02, 0.98)
        assert hit_k.any() and hit_t.any()
    after = bf.window_features(START, np.array([t]), bf.btc_marks(g, ksec2, kclose2), *bf.token_marks(g, ts2, p2))
    for c in bf.FEATURE_NAMES:
        a, b = float(before[c].iloc[0]), float(after[c].iloc[0])
        assert np.isnan(a) == np.isnan(b), f"{c} appeared or vanished with data at or after {m}"
        if not np.isnan(a):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"{c} moved with data at or after the event's mark"


@pytest.mark.parametrize("t", [31, 77, 121, 199, 277])
def test_a_statistic_does_not_move_when_the_marks_after_it_are_dropped(t):
    """The whole grid is computed in one pass; the rescalings inside pmlab.afml.breaks leave every t-statistic
    invariant, so a mark's value must equal the one a series that stops there gives.

    Invariant is not the same as identical in floating point. `breaks.smt` scales local time by the series length, so
    a short window at the end of a long grid and the same window on a truncated series solve differently conditioned
    systems; on a series at BTC's own level that difference reached 4.3e-5 relative until `smt_path` divided the
    price by its first value. The fixture is therefore at LEVEL, not at 100, and several event seconds are swept:
    with a level of 100 the residue hid under the tolerance at every t."""
    g, btc, tok, age = grids(START)
    j = int(bf.mark_index(START, t))
    full = bf.window_features(START, np.array([t]), btc, tok, age)
    cut_btc, cut_tok, cut_age = btc.copy(), tok.copy(), age.copy()
    cut_btc[j + 1:], cut_tok[j + 1:], cut_age[j + 1:] = np.nan, np.nan, np.nan
    cut = bf.window_features(START, np.array([t]), cut_btc, cut_tok, cut_age)
    for c in bf.FEATURE_NAMES:
        a, b = float(full[c].iloc[0]), float(cut[c].iloc[0])
        assert np.isnan(a) == np.isnan(b), f"{c} appeared or vanished when later marks were dropped"
        if not np.isnan(a):
            assert a == pytest.approx(b, rel=1e-8, abs=1e-8), f"{c} depends on marks after it"


def test_no_kline_at_or_after_the_confirmation_start_is_read(monkeypatch):
    """The filter, not the cache: a day whose klines straddle the confirmation start keeps the ones before it and
    drops the rest, whatever the store happens to hold."""
    s = _script()
    day = pd.Timestamp(CONFIRM_START, unit="s", tz="UTC").strftime("%Y-%m-%d")
    sec = np.arange(CONFIRM_START - 5, CONFIRM_START + 5, dtype=np.int64)
    frame = pd.DataFrame({"sec": sec, "close": 1.0 + np.arange(len(sec), dtype=float)})
    monkeypatch.setattr(s.store, "days", lambda kind: [day])
    monkeypatch.setattr(s.store, "read_day", lambda kind, d: frame)
    ksec, kclose = s.klines_for(day)
    assert list(ksec) == list(range(CONFIRM_START - 5, CONFIRM_START))
    assert len(kclose) == 5


def test_the_build_refuses_a_window_that_ends_after_the_confirmation_start(monkeypatch):
    s = _script()
    bad = CONFIRM_START - T + 60
    monkeypatch.setattr(s, "event_keys", lambda day: pd.DataFrame({"start": [bad], "t": [10]}))
    monkeypatch.setattr(s, "day_inputs", lambda day: {})
    with pytest.raises(RuntimeError, match="confirmation start"):
        s.build_day("2026-09-18", "x")


def test_every_feature_is_named_and_grouped():
    assert len(bf.FEATURE_NAMES) == len(set(bf.FEATURE_NAMES))
    assert all(n.startswith("b17_") for n in bf.FEATURE_NAMES)
    assert set(bf.STAT_NAMES) | set(bf.FLAG_NAMES) == set(bf.FEATURE_NAMES)
    assert sorted(sum(bf.FAMILIES.values(), [])) == sorted(bf.STAT_NAMES)
    assert sorted(sum(bf.SERIES.values(), [])) == sorted(bf.STAT_NAMES)
    for a, b in bf.PAIRS:
        assert a in bf.FEATURE_NAMES and b in bf.FEATURE_NAMES
    assert set(bf.SIGNED) <= set(bf.FEATURE_NAMES)
