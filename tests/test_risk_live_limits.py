import json
from dataclasses import asdict

import pytest

from pmlab import risk


def _walkforward(tmp_path):
    limits = risk.RiskLimits(sigma_lo=0.1, sigma_hi=0.9)
    f = tmp_path / "walkforward.json"
    f.write_text(json.dumps({"test_days": ["d1"], "risk": {"limits_by_day": {"d1": asdict(limits)}, "micro_q": 0.95},
                             "fair_params": {"halflife": 5, "vol_mult": 1.0}}))
    return f


@pytest.mark.case
def test_live_limits_micro_fails_clearly_not_with_a_missing_module(tmp_path):
    """live_limits(micro=True) used to import pmlab.performance, which the published tree does not have, and died with
    ModuleNotFoundError after reading the walk-forward file. It must say what is missing, before reading anything."""
    wf = _walkforward(tmp_path)
    with pytest.raises(NotImplementedError, match="LIVE_VALID_FROM"):
        risk.live_limits({"halflife": 5, "vol_mult": 1.0, "sigma_floor": 0.0}, walkforward=wf, micro=True)
    with pytest.raises(NotImplementedError, match="LIVE_VALID_FROM"):         # not even a missing file comes first
        risk.live_limits({}, walkforward=tmp_path / "absent.json", micro=True)


@pytest.mark.case
def test_live_limits_without_micro_rescales_the_walk_forward_band(tmp_path):
    """The control for the test above: the same call without micro works and rescales sigma by live/walk-forward vol_mult."""
    live_fair = {"halflife": 5, "vol_mult": 2.0, "sigma_floor": 0.0}
    got = risk.live_limits(live_fair, walkforward=_walkforward(tmp_path))
    assert got.sigma_lo == pytest.approx(0.2) and got.sigma_hi == pytest.approx(1.8)
