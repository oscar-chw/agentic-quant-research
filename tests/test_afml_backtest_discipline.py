"""AFML 11.4 and 11.5 as checks rather than as good intentions: the research spine must put feature importance
and every other research stage before any backtest, and no stage may be closed while a stage it depends on is
still open. The plan graph in plan/afml-pipeline.md is where that order lives, so that is what is checked."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plan" / "afml-pipeline.md"
NODE = re.compile(r"^- \[([ x>])\] (\d+)\. (.*)$")


def _nodes(path: Path = PLAN) -> dict[int, dict]:
    out = {}
    for line in path.read_text().splitlines():
        m = NODE.match(line)
        if not m:
            continue
        mark, n, rest = m.group(1), int(m.group(2)), m.group(3)
        needs = re.search(r"\|\s*needs:\s*([\d,\s]+)", rest)
        out[n] = {"mark": mark, "title": rest.split("|")[0].strip(),
                  "needs": [int(x) for x in needs.group(1).replace(" ", "").strip(",").split(",")] if needs else []}
    return out


def _ancestors(nodes: dict[int, dict], n: int) -> set[int]:
    seen, stack = set(), list(nodes[n]["needs"])
    while stack:
        m = stack.pop()
        if m in seen or m not in nodes:
            continue
        seen.add(m)
        stack += nodes[m]["needs"]
    return seen


@pytest.mark.case
def test_the_plan_runs_feature_importance_and_the_research_stages_before_any_backtest():
    """11.4: feature importance is the research tool and is derived ex ante; a backtest is not a research tool and
    is not started until the model is fully specified (11.5's third recommendation). The backtest stage therefore
    depends on the importance, model and sizing stages, and none of them depends on it."""
    nodes = _nodes()
    assert "feature importance" in nodes[3]["title"].lower() and "before any backtest" in nodes[3]["title"].lower()
    assert nodes[6]["title"].lower().startswith("backtests by the book")
    before = _ancestors(nodes, 6)
    for stage in (2, 3, 4, 5):                       # events, importance, meta-label models, bet sizing
        assert stage in before, nodes[stage]["title"]
    for stage in (1, 2, 3, 4, 5):
        assert 6 not in _ancestors(nodes, stage), stage
    assert 6 in _ancestors(nodes, 7) and 6 in _ancestors(nodes, 8)    # allocation and deployment come after it


@pytest.mark.case
def test_no_stage_is_closed_while_a_stage_it_depends_on_is_still_open():
    """11.5: do not backtest until all your research is complete. A stage marked done whose dependency is still
    pending is exactly that mistake, wherever it happens in the spine."""
    nodes = _nodes()
    broken = [(n, m) for n, v in nodes.items() if v["mark"] == "x"
              for m in _ancestors(nodes, n) if nodes[m]["mark"] != "x"]
    assert broken == []
    assert [n for n, v in nodes.items() if v["needs"] and n in v["needs"]] == []   # no stage needs itself


# ------------------------------------------------------------------ the second reading (plan book-v2 node 16)


@pytest.mark.eval
def test_bagging_the_selection_spoils_a_strategy_built_on_a_few_observations_and_leaves_a_real_one_alone():
    """11.5's second recommendation, the half that had no measurement: a strategy whose performance worsens when it is
    bagged was probably fitted to a handful of observations or outliers. The strategy here is the choice
    of one rule among 200 random ones. Bagging that choice means selecting on 50 bootstrap resamples and averaging
    what the selected rules earn on the original sample. Thresholds: on pure noise the bagged selection keeps under
    90% of the single best rule's backtest and lands on more than five different rules; with one rule that genuinely
    pays, every resample selects it and the bagged backtest is the single best one exactly."""
    import numpy as np

    rng = np.random.default_rng(3)
    periods, rules = 400, 200
    signals = rng.random((periods, rules)) < 0.3            # each candidate rule is on for about a third of the sample

    def selection(returns):
        backtest = np.array([float(np.sum(returns[signals[:, j]])) for j in range(rules)])
        picks = []
        for seed in range(50):
            draw = np.random.default_rng(100 + seed).choice(periods, periods, replace=True)
            resampled = np.array([float(np.sum(returns[draw][signals[draw, j]])) for j in range(rules)])
            picks.append(int(np.argmax(resampled)))
        return float(backtest.max()), float(np.mean(backtest[picks])), len(set(picks))

    best, bagged, distinct = selection(rng.normal(0, 1, periods))
    assert bagged < 0.9 * best and distinct > 5             # the edge was a handful of observations

    paying = rng.normal(0, 1, periods) + 0.8 * signals[:, 7]
    best, bagged, distinct = selection(paying)
    assert bagged == pytest.approx(best) and distinct == 1  # a real edge survives every resample
