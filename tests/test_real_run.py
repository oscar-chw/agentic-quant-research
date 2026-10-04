"""scripts/real_run.sh itself, end to end, on a tiny SYNTHETIC protocol-v2 study: the fill
run and the DELISTED=drop robustness run, with one pair delisted inside the test window."""
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from factor_research.ohlcv import ohlcv_import
from quantos_showcase import ablation

ROOT = Path(__file__).resolve().parents[1]
T = "T23:59:59Z"
SCORING = (("2024-01-01", "2024-02-09"), ("2024-02-10", "2024-03-20"), ("2024-03-21", "2024-04-29"))
SELECTION = (("2024-01-01", "2024-01-20"), ("2024-01-21", "2024-02-09"), ("2024-02-10", "2024-03-20"))


def experiment(windows):
    return {"available_delay_seconds": 0, "lookback": 1, "cost_bps": 10, "candidates": ["momentum"],
            "splits": {s: {"start": a + T, "end": b + T} for s, (a, b) in zip(("train", "validation", "test"), windows)}}


def study(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = random.Random(11)
    for a in range(6):
        price, rows = 100.0, ["date,open,high,low,close,volume"]
        for i in range(120):  # 2024-01-01..2024-04-29; S5 stops trading after 2024-04-10
            day = date(2024, 1, 1) + timedelta(days=i)
            price *= 1 + rng.gauss(0, .03)
            if a == 5 and day > date(2024, 4, 10):
                continue
            rows.append(f"{day.isoformat()},{price},{price},{price},{price},1")
        (data / f"S{a}USDT.csv").write_text("\n".join(rows) + "\n")
    universe = tmp_path / "universe.json"
    universe.write_text(json.dumps({"schema": "asof-universe/v1", "id": "synthetic", "sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.csv"))}}))
    r = tmp_path / "study"
    r.mkdir()
    protocol = {"schema": "asof-ablation-protocol/v2", "id": "synthetic-forward", "data_label": "SYNTHETIC",
                "universe": {"file": str(universe), "delisting": "fill with last close"},
                "grid": {"signals": ["momentum", "reversal"], "min_lookback": 1, "max_lookback": 3},
                "selection_metric": "net_mean", "selections": str(r / "selections.json"),
                "arms": {"llm": {"k": 3, "model": "unused"}, "random": {"k": 3, "seed": 5, "distribution_draws": 20}},
                "baselines": {"reversal_1": {"signal": "reversal", "lookback": 1}, "equal_weight_hold": {}},
                "costs": {"bps_per_side": 10}, "multiple_testing": {"alpha": 0.05},
                "verdict": {"metric": "net_mean", "alpha": 0.05, "newey_west_lags": 5,
                            "paired_against": ["grid", "random-K"]}}
    (r / "protocol.json").write_text(json.dumps(protocol))
    (r / "experiment.json").write_text(json.dumps(experiment(SCORING)))
    (r / "campaign.json").write_text("{}")
    early = ohlcv_import(data, experiment(SELECTION))  # the frozen selections, from data up to validation end
    (tmp_path / "early-prices.csv").write_bytes(early["prices"])
    (tmp_path / "early-contract.json").write_bytes(early["contract"])
    ablation.freeze_selections(r / "protocol.json", tmp_path / "early-prices.csv", tmp_path / "early-contract.json",
                               tmp_path / "early-work", r / "selections.json", workers=2)
    return data, r


def real_run(tmp_path, data, r, work, mode):
    env = dict(os.environ, PYTHON=sys.executable, ASOF_DATA_DIR=str(data), STUDY=str(r),
               WORK_DIR=str(tmp_path / work), DELISTED=mode)
    return subprocess.run(["bash", "scripts/real_run.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=600)


def test_real_run_script_scores_a_v2_study_with_fill_and_with_drop(tmp_path):
    data, r = study(tmp_path)
    fill = real_run(tmp_path, data, r, "work-fill", "fill")
    assert fill.returncode == 0, fill.stderr[-2000:]
    assert json.loads((r / "filled-pairs.json").read_text())["filled"] == {"S5USDT": [["2024-04-11", "2024-04-29"]]}
    result = json.loads((r / "ablation.json").read_text())
    assert result["arms"][0]["status"] == "PENDING" and result["verdict"]["llm_helps"] is None
    drop = real_run(tmp_path, data, r, "work-drop", "drop")
    assert drop.returncode == 0, drop.stderr[-2000:]
    dropped = json.loads((r / "robustness-dropped/ablation.json").read_text())
    # the same frozen picks are compared; only the test split sees the delisting
    assert [a.get("selected") for a in dropped["arms"]] == [a.get("selected") for a in result["arms"]]


def test_forward_propose_without_a_key_stops_before_any_call(tmp_path):
    """No key in the environment or the key file: exit 4 before the smoke starts, never a silent skip."""
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env.update(PYTHON=sys.executable, HOME=str(tmp_path))
    result = subprocess.run(["bash", "scripts/forward_propose.sh"], cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 4
    assert "OPENROUTER_API_KEY is not set (nor ~/.config/openrouter/api_key)" in result.stderr
    assert "smoke" not in result.stdout
