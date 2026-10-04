"""The README's POST-HOC selection figure: its committed extract must be what hypotheses.json gives."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("render_selection", ROOT / "tools/render_selection.py")
render_selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_selection)

RAW = (ROOT / render_selection.SOURCE).read_bytes()
COMMITTED = json.loads((ROOT / render_selection.EXTRACT).read_text(encoding="utf-8"))


def test_committed_extract_is_recomputed_from_the_committed_results():
    assert COMMITTED["source"]["sha256"] == hashlib.sha256(RAW).hexdigest()
    assert render_selection.dumps(render_selection.extract(RAW)) == \
        (ROOT / render_selection.EXTRACT).read_text(encoding="utf-8")
    assert render_selection.main(["--check"]) == 0
    assert (ROOT / render_selection.FIGURE).is_file()


def test_extract_carries_the_numbers_the_readme_quotes():
    assert COMMITTED["label"].startswith("POST-HOC")
    rho = COMMITTED["spearman_momentum"]
    assert round(rho["validation_net_vs_test_net"], 2) == -0.59
    assert rho["validation_ic_vs_test_ic"] == -0.3072  # ranked before rounding: no artificial tie
    assert sorted(COMMITTED["top5_by_validation_net"]) == [14, 15, 16, 17, 19]
    row = {r["lookback"]: r for r in COMMITTED["momentum"]}
    assert all(row[k]["test_net_bps"] < 0 for k in COMMITTED["top5_by_validation_net"])
    assert (round(row[14]["validation_net_bps"], 2), round(row[14]["test_net_bps"], 2)) == (7.69, -0.15)
    assert COMMITTED["v1_grid_pick_on_validation_ic"]["id"] == "reversal-40"
    assert COMMITTED["counts_all_120"] == {"tested": 120, "positive_validation_net": 36,
                                           "positive_test_net": 39, "positive_both": 17}


def test_ranks_average_ties_and_spearman_sees_order_only():
    assert render_selection.ranks([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]
    assert render_selection.spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == 1.0
    assert render_selection.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_a_changed_result_is_caught():
    rows = json.loads(RAW)
    rows[13]["test"]["net_mean"] += 1e-4  # momentum-14's test net, one bp higher
    changed = render_selection.extract(json.dumps(rows).encode())
    assert changed["momentum"][13]["test_net_bps"] != COMMITTED["momentum"][13]["test_net_bps"]
    assert changed["source"]["sha256"] != COMMITTED["source"]["sha256"]
    assert render_selection.dumps(changed) != render_selection.dumps(COMMITTED)
