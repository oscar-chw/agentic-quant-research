"""Judging a family of features that stage 3 never saw, by stage 3's own machinery (plan afml-pipeline nodes 16 and
23; AFML ch. 7-8). scripts/family_importance.py runs it; every leak guard here has a test in
tests/test_family_importance.py.

Stage 3 (research/feature_importance.md) measured what the event dataset's 88 `f_` features add to a price + time
baseline, per primary rule, on purged k-fold day folds with the library embargo. Two families were built afterwards
and have never been judged:

- `micro`: chapter 19.6's additional microstructural features (pmlab.afml.micro_features, 34 columns keyed by
  (start, t)) - order sizes, flow persistence, algorithm footprints, cancellations from the recorded book, wallets
  and Deribit DVOL.
- `breaks`: chapter 17's structural-break statistics as per-event features (pmlab.afml.break_features, 15 columns),
  the finding docs/afml_sections/ch17.md F17.1.

Nothing about the measurement changes: the rows, the folds, the weights, the forest, the inner leaf tuning, the
label-permutation control and the gain definition are pmlab.afml.event_importance's, unchanged and imported, and the
rows are stage 3's own sample files, so a gain here is on the same scale as a gain there. What this module adds is
only what a new family needs:

- The join. A family's day cache is keyed by (start, t) and the three primary rules of an event share it, so it is
  joined to stage 3's sampled rows on those two columns. A row whose event has no family row would be a silent hole,
  so the join is required to be complete (`join_family`).
- The side frame. Stage 3's features are put into the frame of the row's side with no fitted parameter
  (event_importance.orient); a new family needs the same treatment and it cannot be guessed from a column's name, so
  SIGNED (multiplied by the side) and PAIRS (the two members swap for a Down row) are declared per family below and a
  test requires every column to be classified.
- The input check. A model may see the baseline and its own family's columns and nothing else: never an outcome, a
  span, a weight, a bet column, a stage-3 feature outside the baseline (that would confound the gain with stage 3's
  own result), or a row of a window ending at or after pmlab.evaluation.CONFIRM_START.
- The groups. A gain is measured for each declared family, for each series or source, for the family as a whole, and
  for the correlation clusters of the family's own columns (pmlab.afml.importance.correlation_clusters, no label
  read), exactly as stage 3 measures cluster gains.
- The second baseline. "Over price + time" is the question stage 3 asked, but stage 4's question is whether a family
  adds anything to the features stage 3 already kept, so `stage3_keep` reads those per rule from
  research/feature_importance.json and `keep_baseline` builds the larger baseline from them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab import WINDOW_SECONDS as T
from pmlab.afml import event_importance as ei
from pmlab.afml import importance as fi
from pmlab.evaluation import CONFIRM_START

BASELINE = list(ei.BASELINE)
KEY = ["start", "t"]
STAGE3_JSON = "research/feature_importance.json"


# ---------------------------------------------------------------------------------------------------- families

def _micro() -> dict:
    from pmlab.afml import micro_features as mf
    names = list(mf.FEATURE_NAMES)
    groups = {}
    for f in mf.FEATURES:
        groups.setdefault(f["group"], []).append(f["name"])
    flags = [f"m19_wallet_known_{mf.N}", "m19_bk_ok"]                 # availability, not information
    return {
        "name": "micro", "prefix": "m19_", "module": mf, "cache": "data/cache/afml/micro_features",
        "names": names, "flags": flags, "features": [n for n in names if n not in flags],
        "groups": {k: [n for n in v if n not in flags] for k, v in groups.items()},
        # The Up frame of pmlab.afml.micro_features: aggressor signs and the book's two sides are Up-frame, so a Down
        # row's signed statistics change sign and its two book sides swap (the Down book is the Up book mirrored).
        "signed": ("m19_run_last", f"m19_minute_open_imb_{mf.MINUTE_LOOKBACK}", f"m19_wallet_top_flow_{mf.N}"),
        "pairs": tuple((f"m19_bk_{k}_bid_{mf.LB}", f"m19_bk_{k}_ask_{mf.LB}")
                       for k in ("cancel", "add", "cancel_add", "cancel_n")),
        "report": "research/micro_importance.md", "json": "research/micro_importance.json",
        "build_json": "research/micro_features.json",
    }


def _breaks() -> dict:
    from pmlab.afml import break_features as bf
    return {
        "name": "breaks", "prefix": "b17_", "module": bf, "cache": "data/cache/afml/break_features",
        "names": list(bf.FEATURE_NAMES), "flags": list(bf.FLAG_NAMES), "features": list(bf.STAT_NAMES),
        "groups": {**bf.FAMILIES, **{f"series_{k}": v for k, v in bf.SERIES.items()}},
        "signed": tuple(bf.SIGNED), "pairs": tuple(bf.PAIRS),
        "report": "research/break_features.md", "json": "research/break_importance.json",
        "build_json": "research/break_features_build.json",
    }


FAMILIES = {"micro": _micro, "breaks": _breaks}


def family(name: str) -> dict:
    if name not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}")
    f = FAMILIES[name]()
    paired = {x for p in f["pairs"] for x in p}
    unclassified = [n for n in f["features"] if n not in f["signed"] and n not in paired]
    f["unsigned"] = tuple(unclassified)
    return f


# ---------------------------------------------------------------------------------------------------- rows

def orient(rows: pd.DataFrame, fam: dict) -> pd.DataFrame:
    """The family's Up-frame columns in the frame of each row's side, with no fitted parameter: a signed column is
    multiplied by the side, the members of a pair swap for a Down row, everything else is the same either way."""
    out = rows.copy()
    s = out["side"].to_numpy(float)
    if not np.isin(s, (1.0, -1.0)).all():
        raise ValueError("side must be +1 or -1")
    down = s < 0
    for c in fam["signed"]:
        x = out[c].to_numpy(float)
        out[c] = (x * s).astype(np.float32)
    for a, b in fam["pairs"]:
        xa, xb = out[a].to_numpy(float).copy(), out[b].to_numpy(float).copy()
        out[a] = np.where(down, xb, xa).astype(np.float32)
        out[b] = np.where(down, xa, xb).astype(np.float32)
    return out


def join_family(rows: pd.DataFrame, fam: dict, root: Path, code_sha: str | None = None) -> pd.DataFrame:
    """The family's columns joined to `rows` on (start, t).

    Two things are refused rather than absorbed. A day built by another version of the family's library: the cache is
    resumable day by day, so rebuilding only the recent days would otherwise mix two definitions of a column inside
    one judgement, and nothing downstream could see it (`code_sha` is the sha256 of the library the run is frozen
    on). And a sampled event with no family row, which would become a column of NaN the forest reads as a value."""
    days = sorted({str(d) for d in rows["day"]})
    parts = []
    for d in days:
        folder = Path(root) / fam["cache"] / f"day={d}"
        f = folder / "features.parquet"
        if not f.exists():
            raise FileNotFoundError(f"{fam['name']}: {f} is missing; build the family first")
        meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
        if meta.get("version") != fam["module"].VERSION or (code_sha and meta.get("code_sha256") != code_sha):
            raise ValueError(f"{fam['name']}: {folder} was built by another version of the family's library "
                             f"({str(meta.get('code_sha256'))[:12]} against {str(code_sha)[:12]}); rebuild it")
        parts.append(pd.read_parquet(f, columns=KEY + fam["names"]))
    ft = pd.concat(parts, ignore_index=True)
    dup = ft.duplicated(KEY, keep=False)
    if dup.any():
        raise ValueError(f"{fam['name']}: {int(dup.sum())} rows share an (start, t) across the family's days")
    ft["_joined"] = np.int8(1)
    out = rows.merge(ft, on=KEY, how="left", validate="many_to_one")
    hole = int(out["_joined"].isna().sum())
    if hole:
        raise ValueError(f"{fam['name']}: {hole} of {len(rows)} sampled events have no family row")
    return out.drop(columns=["_joined"])


def check_inputs(rows: pd.DataFrame, columns, fam: dict, baseline=None) -> None:
    """Refuse any model input that is not the run's baseline or one of the family's own columns, and any row of a
    window ending at or after the confirmation start. `baseline` is price + time by default and stage 3's kept
    features for the second baseline; a stage-3 feature outside it would confound the family's gain with stage 3's
    own result."""
    base = list(BASELINE if baseline is None else baseline)
    allowed = set(base) | set(fam["features"])          # never an availability flag: it is not information
    bad = [c for c in columns if c in ei.INPUT_FORBIDDEN or c not in allowed]
    if bad:
        why = (" — a stage-3 feature outside the baseline would confound the family's gain with stage 3's own result"
               if any(c in ei.FEATURES for c in bad) else "")
        raise ValueError(f"not model inputs for family {fam['name']}: {bad}{why}")
    if len(rows) and (rows["start"].to_numpy(np.int64) + T > CONFIRM_START).any():
        raise ValueError("a row's window ends after the confirmation start")


def design(rows: pd.DataFrame, columns, fam: dict, baseline=None) -> np.ndarray:
    check_inputs(rows, columns, fam, baseline)
    return rows[list(columns)].to_numpy(np.float32)


# ---------------------------------------------------------------------------------------------------- groups

def clusters(X: np.ndarray, names: list[str]) -> dict:
    """Average-linkage correlation clusters of the family's own columns, k by mean silhouette; no label is read
    (pmlab.afml.importance.correlation_clusters, stage 3's function)."""
    res = fi.correlation_clusters(X, names)
    order = sorted(res["members"], key=lambda c: min(names.index(m) for m in res["members"][c]))
    return {"members": {f"K{i + 1:02d}": res["members"][c] for i, c in enumerate(order)},
            "silhouette": res["silhouette"]}


def gain_groups(fam: dict, cluster_members: dict, usable: list[str]) -> dict:
    """{group name: its columns}, each added to the baseline on its own: the declared families and series, the whole
    family, and its correlation clusters. Only columns with something to learn from on the sample (`usable`) enter."""
    out = {}
    for g, members in fam["groups"].items():
        cols = [m for m in members if m in usable]
        if cols:
            out[f"family:{g}"] = cols
    all_cols = [m for m in fam["features"] if m in usable]
    if len(all_cols) > 1 and not any(sorted(v) == sorted(all_cols) for v in out.values()):
        out[f"set:{fam['name']}_all"] = all_cols
    for c, members in cluster_members.items():
        cols = [m for m in members if m in usable]
        if cols:
            out[f"cluster:{c}"] = cols
    return out


def usable_columns(rows: pd.DataFrame, fam: dict, min_finite: float = 0.2, min_values: int = 2,
                   blocks=None) -> tuple[list[str], dict]:
    """(the family columns a fold can actually learn from, their coverage per block).

    A column is judged only where at least `min_finite` of the rows have a value and it takes at least `min_values`
    of them - and, when `blocks` is given (the fold split, as (train, test) index pairs), where that holds inside
    every block as well. The global bar alone would admit a column whose values cover a contiguous fifth of the
    sample, which is the shape of every feed that started mid-sample: it would then be all NaN in some folds'
    training or test rows and its per-fold gains would be noise. A column that is almost all NaN (the recorded book
    over six months of history) is not judged and is reported as not judgeable instead."""
    ok, coverage = [], {}
    parts = [("all", np.arange(len(rows)))]
    for i, (tr, te) in enumerate(blocks or []):
        parts += [(f"fold{i}.train", np.asarray(tr)), (f"fold{i}.test", np.asarray(te))]
    for c in fam["features"]:
        x = rows[c].to_numpy(float)
        fin = np.isfinite(x)
        coverage[c] = {k: float(fin[ix].mean()) if len(ix) else 0.0 for k, ix in parts}
        enough = all(v >= min_finite for v in coverage[c].values())
        if enough and len(np.unique(x[fin])) >= min_values:
            ok.append(c)
    return ok, coverage


# ---------------------------------------------------------------------------------------------------- baselines

def stage3_keep(root: Path, rule: str) -> list[str]:
    """The features stage 3 decided to keep for this rule (research/feature_importance.json), plus the baseline: the
    second baseline a family has to beat before stage 4 would use it."""
    s = json.loads((Path(root) / STAGE3_JSON).read_text())
    keep = list(BASELINE)
    for g, e in s["rules"][rule]["gains"].items():
        if g.startswith("cluster:") and e.get("decision") == "keep":
            keep += [m for m in e["members"] if m not in keep]
    return keep


def keep_baseline(root: Path, rule: str) -> list[str]:
    return [c for c in stage3_keep(root, rule) if c in ei.FEATURES]
