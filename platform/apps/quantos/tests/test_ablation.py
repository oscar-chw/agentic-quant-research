"""The ablation's accounting, statistics, data identity and arm wiring, on SYNTHETIC data."""

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

import pytest
from factor_research.ohlcv import ohlcv_import
from factor_research.synthetic import make_fixture
from qrae.llm import ReplayProvider

from quantos_showcase import ablation, loop

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "apps/quantos/examples/loop"


def day(time, label, weights, gross):
    return {"time": time, "label_time": label, "weights": weights, "portfolio": {"gross": gross}}


def test_rebalanced_book_charges_drifted_trades_entry_and_exit():
    prices = {("d0", "A"): 100.0, ("d1", "A"): 110.0, ("d2", "A"): 110.0,
              ("d0", "B"): 100.0, ("d1", "B"): 80.0, ("d2", "B"): 80.0}
    daily = [day("d0", "d1", {"A": .5, "B": -.5}, .15), day("d1", "d2", {"A": .5, "B": -.5}, 0.0)]
    book = ablation.rebalanced_book(daily, prices, 10)
    # day 1: enter 1.0. Unchanged weights still trade on day 2: A drifted to .55 and
    # B to -.4, so trading back to .5 / -.5 costs .05 + .1; then the exit trades 1.0.
    assert [round(t, 12) for _, t, _ in book] == [1.0, 1.15]
    assert [round(n, 12) for _, _, n in book] == [.149, -.00115]


def test_a_held_pair_with_no_exit_bar_closes_at_its_last_close():
    # B has no bar at d1 (delisted): its short closes at zero return, and the lab has no gross that day
    prices = {("d0", "A"): 100.0, ("d1", "A"): 110.0, ("d0", "B"): 100.0}
    gone = {"time": "d0", "label_time": "d1", "weights": {"A": .5, "B": -.5},
            "portfolio": {"gross": None, "status": "MISSING_EXIT_MARK"}}
    [(gross, traded, net)] = ablation.rebalanced_book([gone], prices, 10)
    assert gross == pytest.approx(.05)          # A's +10% on 0.5; B contributes exactly 0
    assert traded == pytest.approx(1 + .55 + .5)  # entry 1.0, then the exit of A at 0.55 and of B at 0.5
    assert net == pytest.approx(.05 - 2.05 * 10 / 10_000)


def test_equal_weight_hold_values_a_missing_pair_at_its_last_close():
    prices = {("d0", "A"): 100.0, ("d1", "A"): 120.0, ("d2", "A"): 120.0,
              ("d0", "B"): 100.0, ("d1", "B"): 50.0}  # B has no bar at d2
    daily = [{"time": "d0", "label_time": "d1"}, {"time": "d1", "label_time": "d2"}]
    hold = ablation.equal_weight_hold(daily, prices, 0)
    assert hold["cumulative_return"] == pytest.approx((1.2 + .5) / 2 - 1)  # B held at 50, not 0 or 100
    assert hold["gross_mean"] == pytest.approx(((1.2 + .5) / 2 - 1) / 2)


def test_rebalanced_book_refuses_prices_the_factor_lab_did_not_use():
    prices = {("d0", "A"): 100.0, ("d1", "A"): 110.0, ("d0", "B"): 100.0, ("d1", "B"): 100.0}
    with pytest.raises(ValueError, match="differs from the factor lab"):
        ablation.rebalanced_book([day("d0", "d1", {"A": .5, "B": -.5}, .06)], prices, 10)


def test_deflated_sharpe_is_psr_for_one_trial_and_falls_with_more_trials():
    one, threshold = ablation.deflated_sharpe(.1, 366, 0.0, 3.0, [.1])
    # skew 0, kurtosis 3: the non-normality scale is sqrt(1 + 2/4 x SR^2)
    assert threshold == 0 and one == pytest.approx(ablation.NORMAL.cdf(.1 * math.sqrt(365) / math.sqrt(1.005)))
    trials = [.1, .02, -.03, .05, 0.0, .04, -.01, .06]
    many, threshold = ablation.deflated_sharpe(.1, 366, 0.0, 3.0, trials)
    assert threshold > 0 and many < one
    assert ablation.deflated_sharpe(.1, 366, 0.0, 3.0, trials * 8)[0] < many


def test_newey_west_matches_plain_t_at_lag_zero_and_grows_with_positive_autocorrelation():
    rng = random.Random(1)
    noise = [rng.gauss(.1, 1) for _ in range(400)]
    plain = ablation.moments(noise)["t"]
    assert ablation.newey_west_t(noise, 0) == pytest.approx(plain * math.sqrt(400 / 399))
    # by hand: gamma0 1.25, gamma1 0.3125, Bartlett weight 1/2 -> variance 1.5625, se 0.625, t 4
    assert ablation.newey_west_t([1, 2, 3, 4], 1) == pytest.approx(4.0)
    smooth = [sum(noise[i:i + 5]) / 5 for i in range(396)]  # overlapping sums: positive autocorrelation
    assert ablation.autocorrelation(smooth, 1) > .5
    assert abs(ablation.newey_west_t(smooth, 10)) < abs(ablation.moments(smooth)["t"]) / 1.5


def test_effective_trials_counts_mirrors_and_copies_once():
    rng = random.Random(2)
    a, b = ([rng.gauss(0, 1) for _ in range(500)] for _ in range(2))
    assert ablation.effective_trials([a, a, [-x for x in a]]) == pytest.approx(1)
    assert 1.9 < ablation.effective_trials([a, b]) <= 2


def test_eigen_effective_trials_span_one_to_n():
    rng = random.Random(3)
    a = [rng.gauss(0, 1) for _ in range(800)]
    copies = ablation.eigen_effective_trials([a, a, [-x for x in a]])
    assert copies["li_ji"] == pytest.approx(1) and copies["nyholt"] == pytest.approx(1)
    independent = ablation.eigen_effective_trials([[rng.gauss(0, 1) for _ in range(800)] for _ in range(4)])
    assert 3.5 < independent["nyholt"] <= 4 and 3 <= independent["li_ji"] <= 4


def test_null_variance_sets_the_deflated_threshold():
    _, threshold = ablation.deflated_sharpe(.1, 366, 0.0, 3.0, [0.0, 0.0], n_trials=10, sharpe_variance=1 / 366)
    expected = (1 / 366) ** .5 * ((1 - ablation.EULER_GAMMA) * ablation.NORMAL.inv_cdf(.9)
                                  + ablation.EULER_GAMMA * ablation.NORMAL.inv_cdf(1 - 1 / (10 * math.e)))
    assert threshold == pytest.approx(expected)


SPEC = {"metric": "net_mean", "alpha": 0.05, "newey_west_lags": 5, "paired_against": ["grid", "random-K"]}


def daily(mean, sd, seed, base=None):
    rng = random.Random(seed)
    return [(b if base else 0) + mean + rng.gauss(0, sd) for b in (base or [0] * 365)]


LLM_NET = daily(.004, .01, 1)                 # clearly positive on its own
SERIES = {
    "llm-pick": LLM_NET,
    "grid-pick": daily(-.003, .001, 2, LLM_NET),  # tracks the LLM pick, 30 bps/day worse: paired test is decisive
    "close-pick": daily(0, .01, 3, LLM_NET),      # same mean as the LLM pick: paired test is not significant
    "rand-pick": daily(-.003, .001, 4, LLM_NET),
}


def arm(name, selected, promoted=True, net_p=.001, ic_p=.5, status="RUN"):
    return {"arm": name, "status": status, "selected": selected, "promoted": promoted,
            "test": {"net_p_nw": net_p, "ic_p_nw": ic_p}}


@pytest.mark.parametrize(("llm", "grid", "llm_helps"), [
    (arm("LLM", "llm-pick"), arm("grid", "grid-pick"), True),
    (arm("LLM", "llm-pick", net_p=.2), arm("grid", "grid-pick"), False),     # arm not significant on net
    (arm("LLM", "llm-pick", promoted=False), arm("grid", "grid-pick"), False),
    (arm("LLM", "llm-pick"), arm("grid", "close-pick"), False),              # does not beat the grid, paired
    (arm("LLM", "llm-pick"), arm("grid", "llm-pick"), False),                # same pick: no difference to test
    ({"arm": "LLM", "status": "PENDING"}, arm("grid", "grid-pick"), None),
])
def test_v2_verdict_tests_daily_net_and_the_paired_difference(llm, grid, llm_helps):
    result = ablation.verdict([llm, grid, arm("random-K", "rand-pick")], SPEC, SERIES)
    assert result["llm_helps"] is llm_helps
    assert result["arm_helps"]["grid"] is True  # promoted, net p .001
    if llm_helps is not None:
        assert set(result["llm_vs"]) == {"grid", "random-K"}


def test_the_llm_must_also_beat_random_k_paired():
    arms = [arm("LLM", "llm-pick"), arm("grid", "grid-pick"), arm("random-K", "close-pick")]
    result = ablation.verdict(arms, SPEC, SERIES)
    assert result["llm_vs"]["grid"]["significant"] and not result["llm_vs"]["random-K"]["significant"]
    assert result["llm_helps"] is False


def test_verdict_rule_is_plain_prose_from_the_protocol():
    rule = ablation.verdict([arm("LLM", "llm-pick"), arm("grid", "grid-pick"), arm("random-K", "rand-pick")],
                            SPEC, SERIES)["rule"]
    assert rule == ("An arm helps if its pick is promoted and its daily test net return has a one-sided Newey-West "
                    "(lag 5) p below 0.05. The LLM helps if, in addition, paired one-sided Newey-West tests of its "
                    "daily test net return minus that of the grid and the random-K pick each give p below 0.05.")


def test_verdict_uses_the_metric_the_protocol_names():
    arms = [arm("LLM", "llm-pick", net_p=.001, ic_p=.4), arm("grid", "grid-pick"), arm("random-K", "rand-pick")]
    assert ablation.verdict(arms, SPEC, SERIES)["llm_helps"] is True
    assert ablation.verdict(arms, dict(SPEC, metric="ic_mean"), SERIES)["llm_helps"] is False


def test_paired_test_matches_a_hand_computed_difference():
    a, b = [3.0, 1.0, 4.0, 1.0, 5.0], [1.0, 1.0, 1.0, 1.0, 1.0]
    result = ablation.paired(a, b)
    assert result["mean_difference"] == pytest.approx(1.8)
    assert result["t_nw"] == pytest.approx(ablation.newey_west_t([2.0, 0.0, 3.0, 0.0, 4.0]))
    with pytest.raises(ValueError, match="same dates"):
        ablation.paired(a, b[:4])


def test_holm_steps_down_and_stays_monotone():
    assert ablation.holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])


@pytest.fixture
def ohlcv(tmp_path):
    folder = tmp_path / "ohlcv"
    folder.mkdir()
    rng = random.Random(7)
    for a in range(6):
        price, rows = 100.0, ["date,open,high,low,close,volume"]
        for i in range(120):
            price *= 1 + rng.gauss(0, .03)
            d = (date(2024, 1, 1) + timedelta(days=i)).isoformat()
            rows.append(f"{d},{price},{price},{price},{price},1")
        (folder / f"S{a}USDT.csv").write_text("\n".join(rows) + "\n")
    return folder


def universe_for(folder, path):
    files = {p.name: ablation.sha256_bytes(p.read_bytes()) for p in folder.glob("*.csv")}
    path.write_text(json.dumps({"schema": "asof-universe/v1", "id": "u", "sha256": files}))
    return path


def test_verify_data_names_every_mismatch(ohlcv, tmp_path):
    universe = universe_for(ohlcv, tmp_path / "u.json")
    assert ablation.verify_data(ohlcv, universe)["files"] == 6
    (ohlcv / "S0USDT.csv").write_text((ohlcv / "S0USDT.csv").read_text().replace("2024-01-01,", "2024-01-01,1"))
    (ohlcv / "S1USDT.csv").unlink()
    (ohlcv / "EXTRA.csv").write_text("x")
    with pytest.raises(ValueError, match=r"missing \['S1USDT.csv'\], unexpected \['EXTRA.csv'\], "
                                         r"checksum mismatch \['S0USDT.csv'\]"):
        ablation.verify_data(ohlcv, universe)


def test_delisted_pairs_keep_their_last_close_or_are_dropped(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    (data / "AUSDT.csv").write_text("date,open,high,low,close,volume\n2024-01-01,1,2,1,2,5\n2024-01-02,2,3,2,3,5\n"
                                    "2024-01-04,3,4,3,4,5\n")  # a one-day gap, then delisted after 01-04
    (data / "BUSDT.csv").write_text("date,open,high,low,close,volume\n" + "".join(
        f"2024-01-0{d},1,1,1,1,1\n" for d in range(1, 7)))
    manifest = ablation.fill_missing(data, "2024-01-06", tmp_path / "filled")
    assert manifest["filled"] == {"AUSDT": [["2024-01-03", "2024-01-03"], ["2024-01-05", "2024-01-06"]]}
    lines = (tmp_path / "filled/AUSDT.csv").read_text().splitlines()
    assert lines[3] == "2024-01-03,3,3,3,3,0" and lines[-1] == "2024-01-06,4,4,4,4,0" and len(lines) == 7
    assert (tmp_path / "filled/BUSDT.csv").read_text() == (data / "BUSDT.csv").read_text()
    ohlcv_import(tmp_path / "filled", {"available_delay_seconds": 0, "lookback": 1, "cost_bps": 10,
                                       "candidates": ["momentum"], "splits": {
                                           s: {"start": f"2024-01-0{a}T23:59:59Z", "end": f"2024-01-0{b}T23:59:59Z"}
                                           for s, (a, b) in zip(("train", "validation", "test"), ((1, 2), (3, 4), (5, 6)))}})


def protocol(tmp_path, max_lookback, seed=3, version=1, baselines=None, **extra):
    body = {"schema": f"asof-ablation-protocol/v{version}", "id": "synthetic-test", "data_label": "SYNTHETIC",
            "grid": {"signals": ["momentum", "reversal"], "min_lookback": 1, "max_lookback": max_lookback},
            "arms": {"llm": {"k": 3}, "random": {"k": 3, "seed": seed, "distribution_draws": 40}},
            "baselines": baselines or {"momentum_2": {"signal": "momentum", "lookback": 2},
                                       "reversal_1": {"signal": "reversal", "lookback": 1}, "equal_weight_hold": {}},
            "costs": {"bps_per_side": 10}, "multiple_testing": {"alpha": 0.05},
            **({"verdict": SPEC} if version == 2 else {}), **extra}
    path = tmp_path / f"protocol-v{version}.json"
    path.write_text(json.dumps(body))
    return path


SCORING = (("2024-01-01", "2024-02-09"), ("2024-02-10", "2024-03-20"), ("2024-03-21", "2024-04-29"))
# selection time: the scoring run's validation window sits in the last slot
SELECTION = (("2024-01-01", "2024-01-20"), ("2024-01-21", "2024-02-09"), ("2024-02-10", "2024-03-20"))


def imported(ohlcv, tmp_path, windows=SCORING, name=""):
    t = "T23:59:59Z"
    splits = {s: {"start": a + t, "end": b + t} for s, (a, b) in zip(("train", "validation", "test"), windows)}
    result = ohlcv_import(ohlcv, {"available_delay_seconds": 0, "lookback": 1, "cost_bps": 10,
                                  "candidates": ["momentum"], "splits": splits})
    (tmp_path / f"prices{name}.csv").write_bytes(result["prices"])
    (tmp_path / f"contract{name}.json").write_bytes(result["contract"])
    return tmp_path / f"prices{name}.csv", tmp_path / f"contract{name}.json"


def test_run_selects_on_validation_only_and_records_every_count(ohlcv, tmp_path):
    prices, contract = imported(ohlcv, tmp_path)
    result = ablation.run(protocol(tmp_path, 3), prices, contract, tmp_path / "work", tmp_path / "out",
                          llm_pending="no login", workers=2)
    rows = {r["id"]: r for r in json.loads((tmp_path / "out/hypotheses.json").read_text())}
    assert len(rows) == 6 and result["counts"]["trials_run_and_recomputed"] == 6
    llm, grid, rand = result["arms"]
    assert {k: llm[k] for k in ("status", "reason", "live_calls_used")} == {
        "status": "PENDING", "reason": "no login", "live_calls_used": 0}
    # whatever 3 cards an LLM proposes, its selection is one of these 4 (2 others must rank below it)
    reachable = sorted(rows, key=lambda i: rows[i]["validation"]["ic_mean"])[2:]
    assert llm["outcome_range"]["selectable"] == 4
    assert llm["outcome_range"]["test_net_mean"] == [min(rows[i]["test"]["net_mean"] for i in reachable),
                                                     max(rows[i]["test"]["net_mean"] for i in reachable)]
    assert grid["test"]["p_holm_across_arms"] is None  # two arms: no Holm
    dsr = grid["validation"]["deflated_sharpe"]
    assert dsr["distinct_lookbacks"] == 3 and all(1 <= n <= 3 for n in dsr["effective_trials"].values())
    head = dsr["headline"]
    assert head["range"] == [min(head["li_ji"], head["nyholt"]), max(head["li_ji"], head["nyholt"])]
    # the headline uses the null sampling variance only, never a trial-variance cell
    assert (head["li_ji"], head["nyholt"]) == (dsr["sensitivity"]["null_1_over_T__n_li_ji"],
                                               dsr["sensitivity"]["null_1_over_T__n_nyholt"])
    assert len(dsr["sensitivity"]) == 15
    # trial variance with a correlation-reduced N counts the correlation twice: flagged, never the headline
    assert sorted(dsr["double_counted"]) == sorted(f"trials_{v}__n_{n}" for v in ("signed", "distinct")
                                                   for n in ("participation", "li_ji", "nyholt"))
    assert "verdict" not in result  # v1 has no coded verdict
    assert grid["hypotheses_tested_on_validation"] == 6
    assert grid["selected"] == max(rows, key=lambda i: rows[i]["validation"]["ic_mean"])
    assert rand["hypotheses"] == random.Random(3).sample(list(rows), 3)
    assert rand["selected"] == max(rand["hypotheses"], key=lambda i: rows[i]["validation"]["ic_mean"])
    assert grid["validation"]["ic_p_bonferroni"] == pytest.approx(min(1, 6 * grid["validation"]["ic_p"]))
    best = max(b["test"]["net_mean"] for b in result["baselines"].values())
    assert grid["promoted"] == (grid["test"]["net_mean"] > max(0, best))
    hold = result["baselines"]["equal_weight_hold"]["test"]
    assert hold["assets"] == 6 and hold["ic_mean"] is None
    assert "| grid | 6 |" in (tmp_path / "out/REPORT.md").read_text()
    with pytest.raises(FileExistsError):
        ablation.run(protocol(tmp_path, 3), prices, contract, tmp_path / "work2", tmp_path / "out",
                     llm_pending="no login")


def test_v2_selects_on_validation_net_and_checks_frozen_selections(ohlcv, tmp_path):
    prices, contract = imported(ohlcv, tmp_path)
    early_prices, early_contract = imported(ohlcv, tmp_path, SELECTION, "-selection")
    frozen = ablation.freeze_selections(protocol(tmp_path, 3), early_prices, early_contract, tmp_path / "w0",
                                        tmp_path / "sel.json", workers=2)
    proto = protocol(tmp_path, 3, version=2, selection_metric="net_mean", selections=str(tmp_path / "sel.json"))
    result = ablation.run(proto, prices, contract, tmp_path / "w1", tmp_path / "out", llm_pending="no login", workers=2)
    rows = {r["id"]: r for r in json.loads((tmp_path / "out/hypotheses.json").read_text())}
    grid = result["arms"][1]
    assert grid["selection_metric"] == "net_mean"
    assert grid["selected"] == max(rows, key=lambda i: rows[i]["validation"]["net_mean"])
    assert grid["secondary_selection"]["selected"] == max(rows, key=lambda i: rows[i]["validation"]["ic_mean"])
    assert result["verdict"]["llm_helps"] is None
    # the slot does not matter: the frozen last-slot numbers are the scoring run's validation numbers
    for i in rows:
        for k in ("ic_mean", "net_mean", "turnover_mean"):
            assert frozen["validation"][i][k] == pytest.approx(rows[i]["validation"][k], abs=1e-15)
    assert frozen["selections"]["grid"]["net_mean"]["selected"] == grid["selected"]
    with pytest.raises(FileExistsError):
        ablation.freeze_selections(protocol(tmp_path, 3), prices, contract, tmp_path / "w2", tmp_path / "sel.json")


def test_scoring_refuses_selections_it_does_not_reproduce(ohlcv, tmp_path):
    prices, contract = imported(ohlcv, tmp_path)
    ablation.freeze_selections(protocol(tmp_path, 3), prices, contract, tmp_path / "w0", tmp_path / "sel.json", workers=2)
    # frozen from the scoring import itself, the table is its test split, not its validation
    proto = protocol(tmp_path, 3, version=2, selection_metric="net_mean", selections=str(tmp_path / "sel.json"))
    with pytest.raises(ValueError, match="differ from the frozen selections"):
        ablation.run(proto, prices, contract, tmp_path / "w1", tmp_path / "out", llm_pending="x", workers=2)


def test_a_pair_delisted_in_the_test_window_scores_with_fill_and_with_drop(ohlcv, tmp_path):
    path = ohlcv / "S5USDT.csv"
    path.write_text("".join(line + "\n" for line in path.read_text().splitlines() if line.startswith("date") or line < "2024-04-11"))
    early_prices, early_contract = imported(ohlcv, tmp_path, SELECTION, "-selection")
    ablation.freeze_selections(protocol(tmp_path, 3), early_prices, early_contract, tmp_path / "w0",
                               tmp_path / "sel.json", workers=2)
    proto = protocol(tmp_path, 3, version=2, selection_metric="net_mean", selections=str(tmp_path / "sel.json"))
    manifest = ablation.fill_missing(ohlcv, "2024-04-29", tmp_path / "filled")
    assert manifest["filled"] == {"S5USDT": [["2024-04-11", "2024-04-29"]]}
    runs = {}
    for name, folder in (("fill", tmp_path / "filled"), ("drop", ohlcv)):  # drop = the unfilled data
        prices, contract = imported(folder, tmp_path, SCORING, f"-{name}")
        runs[name] = ablation.run(proto, prices, contract, tmp_path / f"w-{name}", tmp_path / name,
                                  llm_pending="robustness test", workers=2)
    rows = {n: {r["id"]: r for r in json.loads((tmp_path / n / "hypotheses.json").read_text())} for n in runs}
    # both reproduce the frozen selections (the check runs inside run()) and agree on validation
    assert all(rows["fill"][i]["validation"] == rows["drop"][i]["validation"] for i in rows["fill"])
    assert runs["fill"]["arms"][1]["selected"] == runs["drop"]["arms"][1]["selected"]
    # after the delisting the filled pair is still ranked; the dropped one is not, so test differs
    assert any(rows["fill"][i]["test"]["net_mean"] != rows["drop"][i]["test"]["net_mean"] for i in rows["fill"])


def test_v2_verdict_inside_run_compares_the_picks_test_net(tmp_path):
    make_fixture(tmp_path / "regime", 8, 60)
    loop.run_loop(EXAMPLES / "campaign.json", tmp_path / "regime/panel.csv", EXAMPLES / "synthetic-contract.json",
                  EXAMPLES / "vault", tmp_path / "store", "regime",
                  ReplayProvider(EXAMPLES / "replay.hand-written.json"), receipt_keys=tmp_path / "keys")
    result = ablation.run(protocol(tmp_path, 5, version=2, selection_metric="net_mean"), tmp_path / "regime/panel.csv",
                          EXAMPLES / "synthetic-contract.json", tmp_path / "work", tmp_path / "out",
                          llm_run=tmp_path / "store/regime", receipt_keys=tmp_path / "keys", workers=2)
    rows = {r["id"]: r for r in json.loads((tmp_path / "out/hypotheses.json").read_text())}
    picks = {a["arm"]: a["selected"] for a in result["arms"]}
    llm_vs = result["verdict"]["llm_vs"]
    assert set(llm_vs) == {"grid", "random-K"}
    for other, comparison in llm_vs.items():
        want = rows[picks["LLM"]]["test"]["net_mean"] - rows[picks[other]]["test"]["net_mean"]
        assert comparison["mean_difference"] == pytest.approx(want, abs=1e-15)
    # the regime flips in test: validation and test differ, so the series fed in matters
    assert any(rows[picks["LLM"]]["validation"]["net_mean"] - rows[picks[o]]["validation"]["net_mean"]
               != pytest.approx(llm_vs[o]["mean_difference"], abs=1e-9) for o in llm_vs if picks[o] != picks["LLM"])


def test_verdict_refuses_a_lag_the_arms_were_not_computed_with():
    arms = [arm("LLM", "llm-pick"), arm("grid", "grid-pick"), arm("random-K", "rand-pick")]
    with pytest.raises(ValueError, match="Newey-West lag 10"):
        ablation.verdict(arms, dict(SPEC, newey_west_lags=10), SERIES)


def test_selection_ignores_the_test_split():
    rows = {"a": {"validation": {"ic_mean": .02}, "test": {"ic_mean": -.5}},
            "b": {"validation": {"ic_mean": .01}, "test": {"ic_mean": .9}}}
    assert ablation.select(["a", "b"], rows) == "a"
    rows["b"]["validation"]["ic_mean"] = .02
    assert ablation.select(["a", "b"], rows) == "a"  # ties go to grid order


@pytest.fixture
def loop_run(tmp_path):
    make_fixture(tmp_path / "control", 8, 60, persistent=True)
    make_fixture(tmp_path / "regime", 8, 60)
    loop.run_loop(EXAMPLES / "campaign.json", tmp_path / "control/panel.csv", EXAMPLES / "synthetic-contract.json",
                  EXAMPLES / "vault", tmp_path / "store", "control", ReplayProvider(EXAMPLES / "replay.hand-written.json"),
                  receipt_keys=tmp_path / "keys")
    return tmp_path


def test_llm_arm_uses_frozen_cards_and_applies_the_gate_rule(loop_run):
    t = loop_run
    result = ablation.run(protocol(t, 5), t / "control/panel.csv", EXAMPLES / "synthetic-contract.json",
                          t / "work", t / "out", llm_run=t / "store/control", receipt_keys=t / "keys", workers=2)
    llm = result["arms"][0]
    assert llm["status"] == "RUN" and llm["hypotheses_tested_on_validation"] == 4
    assert set(llm["card_ids"]) == {"momentum-01", "reversal-01", "momentum-03", "reversal-05"}
    decisions = loop.read_decisions(t / "store/control")
    assert set(decisions) == {"H1"}  # the only card that passed the validation rule and the critic
    best = max(b["test"]["net_mean"] for b in result["baselines"].values())
    h1_net = json.loads((t / "out/hypotheses.json").read_text())[0]["test"]["net_mean"]
    assert decisions["H1"] == ("PROMOTED" if h1_net > max(0, best) else "REJECTED_BY_HUMAN")
    assert loop.verify(t / "store/control", receipt_keys=t / "keys")["verified"]


class Objecting:
    """The hand-written proposals, and a critic that objects to every card."""

    name, provenance = "replay", "test double"

    def __init__(self):
        self.replay = ReplayProvider(EXAMPLES / "replay.hand-written.json")

    def complete(self, key, prompt):
        if key == "propose":
            return self.replay.entries["propose"]["response"]
        return json.dumps({"summary": "no", "requested_transition": "NONE", "artifact_refs": [], "claims": [
            {"text": "objection", "classification": "REJECTED", "artifact_refs": []}]})


def test_v2_promotes_the_llm_selection_by_the_same_rule_as_the_other_arms(tmp_path):
    make_fixture(tmp_path / "control", 8, 60, persistent=True)
    loop.run_loop(EXAMPLES / "campaign.json", tmp_path / "control/panel.csv", EXAMPLES / "synthetic-contract.json",
                  EXAMPLES / "vault", tmp_path / "store", "control", Objecting(), receipt_keys=tmp_path / "keys")
    baselines = {"reversal_1": {"signal": "reversal", "lookback": 1}, "equal_weight_hold": {}}
    promoted = {}
    for version in (1, 2):
        result = ablation.run(protocol(tmp_path, 5, version=version, baselines=baselines),
                              tmp_path / "control/panel.csv", EXAMPLES / "synthetic-contract.json",
                              tmp_path / f"work{version}", tmp_path / f"out{version}",
                              llm_run=tmp_path / "store/control", receipt_keys=tmp_path / "keys", workers=2)
        llm = result["arms"][0]
        assert llm["loop_counts"]["rejected_by_critic"] == 2 and llm["selected"] == "momentum-01"
        promoted[version] = llm["promoted"]
    # v1 also demanded the loop's card rule and critic; v2 applies the shared gate alone
    assert promoted == {1: False, 2: True}


def test_llm_arm_refuses_a_loop_run_on_other_prices(loop_run):
    t = loop_run
    with pytest.raises(ValueError, match="loop metrics differ from the grid trial"):
        ablation.run(protocol(t, 5), t / "regime/panel.csv", EXAMPLES / "synthetic-contract.json",
                     t / "work", t / "out", llm_run=t / "store/control", receipt_keys=t / "keys", workers=2)
