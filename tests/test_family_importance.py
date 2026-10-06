"""Judging a new family of features by stage 3's machinery (plan afml-pipeline nodes 16 and 23): the join, the side
frame, the input check, the coverage bar, the groups and the second baseline.

Every assertion here is mutation-checked: `python scripts/family_importance.py --mutations` applies each mutation of
scripts/family_importance.MUTATIONS to a copy of pmlab.afml.family_importance and requires this file to fail.
"""
import json

import numpy as np
import pandas as pd
import pytest

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import event_importance as ei
from pmlab.afml import events as ev
from pmlab.afml import family_importance as xf
from pmlab.evaluation import CONFIRM_START

ROOT_REAL = __import__("pathlib").Path(__file__).resolve().parents[1]
START = CONFIRM_START - 10 * 86400
DAY = "2026-09-08"


def rows_frame(n: int = 8, fam=None) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    d = {"start": np.full(n, START, np.int64), "t": np.arange(10, 10 + n, dtype=np.int64),
         "day": [DAY] * n, "rule": ["q_taker"] * n,
         "side": np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8),
         "label": (np.arange(n) % 3 == 0).astype(np.int8)}
    for c in ei.FEATURES:
        d[c] = rng.normal(size=n).astype(np.float32)
    if fam:
        for c in fam["names"]:
            d[c] = rng.normal(size=n).astype(np.float32)
    return pd.DataFrame(d)


def write_family_day(root, fam, rows: pd.DataFrame, drop_last: int = 0, code: str = "abc", version=None) -> None:
    out = root / fam["cache"] / f"day={DAY}"
    out.mkdir(parents=True, exist_ok=True)
    ft = rows[xf.KEY + fam["names"]].copy()
    if drop_last:
        ft = ft.iloc[:-drop_last]
    ft.to_parquet(out / "features.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"version": fam["module"].VERSION if version is None else version, "code_sha256": code}))


# ---------------------------------------------------------------------------------------------------- families

@pytest.mark.parametrize("name", ["micro", "breaks"])
def test_every_family_column_is_classified_for_the_side_frame(name):
    fam = xf.family(name)
    paired = [x for p in fam["pairs"] for x in p]
    assert len(paired) == len(set(paired)), "a column appears in two pairs"
    assert set(fam["signed"]).isdisjoint(paired)
    covered = set(fam["signed"]) | set(paired) | set(fam["unsigned"])
    assert covered == set(fam["features"]), "a family column is neither signed, paired nor unsigned"
    assert set(fam["flags"]).isdisjoint(fam["features"])
    for g, members in fam["groups"].items():
        assert set(members) <= set(fam["features"]), f"group {g} holds a flag or an unknown column"


# ---------------------------------------------------------------------------------------------------- side frame

@pytest.mark.parametrize("name", ["micro", "breaks"])
def test_the_side_frame_flips_signed_columns_and_swaps_pairs(name):
    fam = xf.family(name)
    rows = rows_frame(fam=fam)
    out = xf.orient(rows, fam)
    up, dn = rows["side"].to_numpy() > 0, rows["side"].to_numpy() < 0
    assert dn.any() and up.any()
    for c in fam["signed"]:
        assert np.allclose(out.loc[up, c], rows.loc[up, c], atol=1e-6)
        assert np.allclose(out.loc[dn, c], -rows.loc[dn, c], atol=1e-6), f"{c} is not put into the side's frame"
    for a, b in fam["pairs"]:
        assert np.allclose(out.loc[up, a], rows.loc[up, a], atol=1e-6)
        assert np.allclose(out.loc[dn, a], rows.loc[dn, b], atol=1e-6), f"{a} and {b} do not swap for a Down row"
        assert np.allclose(out.loc[dn, b], rows.loc[dn, a], atol=1e-6)
    for c in fam["unsigned"]:
        assert np.allclose(out[c], rows[c], atol=1e-6)


def test_the_side_frame_refuses_a_side_that_is_not_plus_or_minus_one():
    fam = xf.family("breaks")
    rows = rows_frame(fam=fam)
    rows.loc[0, "side"] = 0
    with pytest.raises(ValueError):
        xf.orient(rows, fam)


# ---------------------------------------------------------------------------------------------------- join

def test_the_join_is_on_the_event_and_a_missing_family_row_stops_the_run(tmp_path):
    fam = xf.family("breaks")
    rows = rows_frame(fam=fam)
    write_family_day(tmp_path, fam, rows)
    plain = rows.drop(columns=fam["names"])
    joined = xf.join_family(plain, fam, tmp_path, code_sha="abc")
    assert len(joined) == len(plain)
    for c in fam["names"]:
        assert np.allclose(joined[c], rows[c], atol=1e-6), f"{c} did not join on (start, t)"
    write_family_day(tmp_path, fam, rows, drop_last=2)
    with pytest.raises(ValueError, match="no family row"):
        xf.join_family(plain, fam, tmp_path, code_sha="abc")


def test_the_join_refuses_a_family_day_built_by_another_code_version(tmp_path):
    """The cache is resumable day by day, so a partial rebuild would otherwise mix two definitions of a column
    inside one judgement and nothing downstream could see it."""
    fam = xf.family("breaks")
    rows = rows_frame(fam=fam)
    plain = rows.drop(columns=fam["names"])
    write_family_day(tmp_path, fam, rows, code="abc")
    xf.join_family(plain, fam, tmp_path, code_sha="abc")
    with pytest.raises(ValueError, match="another version"):
        xf.join_family(plain, fam, tmp_path, code_sha="def")
    write_family_day(tmp_path, fam, rows, code="abc", version=999)
    with pytest.raises(ValueError, match="another version"):
        xf.join_family(plain, fam, tmp_path, code_sha="abc")


def test_the_join_refuses_two_family_rows_for_one_event(tmp_path):
    fam = xf.family("breaks")
    rows = rows_frame(fam=fam)
    out = tmp_path / fam["cache"] / f"day={DAY}"
    out.mkdir(parents=True)
    doubled = pd.concat([rows[xf.KEY + fam["names"]], rows[xf.KEY + fam["names"]].iloc[:1]], ignore_index=True)
    doubled.to_parquet(out / "features.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"version": fam["module"].VERSION, "code_sha256": "abc"}))
    with pytest.raises(ValueError, match="share an"):
        xf.join_family(rows.drop(columns=fam["names"]), fam, tmp_path, code_sha="abc")


def test_the_join_refuses_a_family_day_that_was_never_built(tmp_path):
    fam = xf.family("breaks")
    with pytest.raises(FileNotFoundError):
        xf.join_family(rows_frame(fam=fam).drop(columns=fam["names"]), fam, tmp_path, code_sha="abc")


# ---------------------------------------------------------------------------------------------------- inputs

@pytest.mark.parametrize("name", ["micro", "breaks"])
def test_only_the_baseline_and_the_family_may_enter_a_model(name):
    fam = xf.family(name)
    rows = rows_frame(fam=fam)
    xf.check_inputs(rows, list(xf.BASELINE) + fam["features"][:2], fam)
    for bad in (["up_won"], ["avg_uniqueness"], ["t0"], ["price"], ["weight"]):
        with pytest.raises(ValueError, match="not model inputs"):
            xf.check_inputs(rows, list(xf.BASELINE) + bad, fam)
    with pytest.raises(ValueError, match="not model inputs"):
        xf.check_inputs(rows, list(xf.BASELINE) + [fam["flags"][0]], fam), "an availability flag is not information"


@pytest.mark.parametrize("name", ["micro", "breaks"])
def test_a_stage_three_feature_outside_the_baseline_is_refused(name):
    fam = xf.family(name)
    rows = rows_frame(fam=fam)
    other = next(c for c in ei.FEATURES if c not in xf.BASELINE)
    with pytest.raises(ValueError, match="confound"):
        xf.check_inputs(rows, list(xf.BASELINE) + [other], fam)
    with pytest.raises(ValueError, match="confound"):
        xf.check_inputs(rows, list(xf.BASELINE) + fam["features"][:1] + [other], fam)
    keep = list(xf.BASELINE) + [other]
    xf.check_inputs(rows, keep + fam["features"][:1], fam, baseline=keep)       # allowed as the second baseline


@pytest.mark.parametrize("name", ["micro", "breaks"])
def test_no_row_of_a_window_ending_at_or_after_the_confirmation_start(name):
    fam = xf.family(name)
    rows = rows_frame(fam=fam)
    rows["start"] = np.int64(CONFIRM_START - T + 1)
    with pytest.raises(ValueError, match="confirmation start"):
        xf.check_inputs(rows, list(xf.BASELINE), fam)
    rows["start"] = np.int64(CONFIRM_START - T)
    xf.check_inputs(rows, list(xf.BASELINE), fam)


def test_design_returns_the_columns_asked_for_and_checks_them():
    fam = xf.family("breaks")
    rows = rows_frame(fam=fam)
    cols = list(xf.BASELINE) + fam["features"][:3]
    X = xf.design(rows, cols, fam)
    assert X.shape == (len(rows), len(cols))
    assert np.allclose(X[:, 0], rows[cols[0]].to_numpy(np.float32))
    with pytest.raises(ValueError):
        xf.design(rows, cols + ["label"], fam)


# ---------------------------------------------------------------------------------------------------- coverage

def test_a_column_almost_always_missing_is_not_judged():
    fam = xf.family("micro")
    rows = rows_frame(n=100, fam=fam)
    sparse, constant = fam["features"][0], fam["features"][1]
    x = rows[sparse].to_numpy(float)
    x[5:] = np.nan                                            # 5% coverage
    rows[sparse] = x
    rows[constant] = np.float32(1.0)
    usable, cover = xf.usable_columns(rows, fam, min_finite=0.2)
    assert sparse not in usable, "a column that is almost all NaN must not be judged"
    assert constant not in usable, "a column with one value has nothing to learn from"
    assert len(usable) >= len(fam["features"]) - 2
    assert cover[sparse]["all"] == pytest.approx(0.05)


def test_a_column_that_only_covers_part_of_the_sample_is_not_judged():
    """The global bar alone would admit a column whose values cover a contiguous fifth of the rows; it would then be
    all NaN in some folds' training or test rows and its per-fold gains would be noise."""
    fam = xf.family("breaks")
    rows = rows_frame(n=100, fam=fam)
    half = fam["features"][0]
    x = rows[half].to_numpy(float)
    x[:60] = np.nan                                           # 40% overall, 0% in the first block
    rows[half] = x
    blocks = [(np.arange(60, 100), np.arange(0, 60)), (np.arange(0, 60), np.arange(60, 100))]
    assert half in xf.usable_columns(rows, fam, min_finite=0.2)[0]
    usable, cover = xf.usable_columns(rows, fam, min_finite=0.2, blocks=blocks)
    assert half not in usable, "a column absent from a whole fold must not be judged"
    assert cover[half]["fold0.test"] == 0.0 and cover[half]["all"] == pytest.approx(0.4)


def test_groups_keep_every_usable_member_and_drop_the_rest():
    fam = xf.family("breaks")
    rows = rows_frame(n=60, fam=fam)
    usable, _ = xf.usable_columns(rows, fam)
    cl = {"K01": fam["groups"]["csw"], "K02": fam["groups"]["smt"]}
    g = xf.gain_groups(fam, cl, usable)
    assert g["cluster:K01"] == [c for c in fam["groups"]["csw"] if c in usable]
    assert len(g["cluster:K01"]) == len(fam["groups"]["csw"]), "a cluster lost members"
    for name, members in fam["groups"].items():
        assert g[f"family:{name}"] == [c for c in members if c in usable]
    assert g["set:breaks_all"] == [c for c in fam["features"] if c in usable]
    dropped = usable[:2]
    g2 = xf.gain_groups(fam, cl, [c for c in usable if c not in dropped])
    assert all(c not in sum(g2.values(), []) for c in dropped)


def test_a_group_identical_to_the_whole_family_is_not_measured_twice():
    fam = xf.family("micro")
    one = {"options": fam["groups"]["options"]}
    small = {**fam, "groups": one, "features": fam["groups"]["options"]}
    g = xf.gain_groups(small, {}, fam["groups"]["options"])
    assert list(g) == ["family:options"]


# ---------------------------------------------------------------------------------------------------- baselines

@pytest.mark.parametrize("rule", list(ev.RULES))
def test_the_second_baseline_is_what_stage_three_kept(rule):
    keep = xf.keep_baseline(ROOT_REAL, rule)
    assert set(xf.BASELINE) <= set(keep)
    assert len(keep) > len(xf.BASELINE), "stage 3 kept features beyond price + time for every rule"
    assert set(keep) <= set(ei.FEATURES)
    assert len(keep) == len(set(keep))


def test_the_embargo_is_the_library_one_and_is_never_shortened():
    import importlib.util
    spec = importlib.util.spec_from_file_location("xfi", ROOT_REAL / "scripts/family_importance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.EMBARGO_S == ev.EMBARGO_S == (ev.FAIR_TRAIN_DAYS + 1) * 86400
    assert all(r["k"] >= 3 for r in mod.RUNS.values())


def test_the_dispatch_that_reaches_this_runner_needs_nothing_of_stage_threes_run():
    """`scripts/feature_importance.py --check --family breaks` is node 23's gate. Stage 3's script freezes its own
    code and imports pmlab from that frozen copy, so the hand-over has to happen before any of that machinery runs -
    otherwise the gate breaks whenever stage 3's script or its snapshot is mid-change."""
    text = (ROOT_REAL / "scripts/feature_importance.py").read_text()
    i = text.find('if "--family" in sys.argv:')
    assert i > 0, "the hand-over is missing"
    before = text[:i]
    assert "import pmlab" not in before and "from pmlab" not in before, "pmlab loads before the hand-over"
    assert "SNAPSHOT" not in before, "the frozen-code machinery runs before the hand-over"
    assert "family_importance.py" in text[i:i + 600]
