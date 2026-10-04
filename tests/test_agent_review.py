"""The QuantOS envelope validator, copied from codex-quant-os, against this repo's loop envelopes."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from quantos_showcase import loop

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/agent-review"
spec = importlib.util.spec_from_file_location("validate_envelope", TOOL / "validate_envelope.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

LEDGER = {
    "run_id": "unit", "data_label": "SYNTHETIC",
    "campaign": {"rule": {"min_rank_ic": "1/10", "min_valid_dates": 10, "min_net_mean": "0"}},
    "provider": {"name": "replay", "provenance": "HAND-WRITTEN FIXTURE: test"},
    "hypotheses": [{"id": "H1", "card_sha256": "0" * 64}],
    "counts": dict.fromkeys(loop.COUNTS, 1),
}


def envelope():
    return loop.envelope(LEDGER, "a" * 64, "2024-02-29T16:00:00Z")


def test_schema_and_validator_require_the_same_fields():
    schema = json.loads((TOOL / "evidence-envelope.schema.json").read_text())
    assert set(schema["required"]) == set(validator.TOP_LEVEL_FIELDS) == set(schema["properties"])


def test_loop_envelope_is_valid():
    assert validator.validate_envelope(envelope()) == []


@pytest.mark.parametrize(("mutate", "error"), [
    (lambda e: e["facts"].append("Saved to /tmp/run"), "envelope:ABSOLUTE_LOCAL_PATH"),
    (lambda e: e["safety"].update(live_action_executed=True), "safety.live_action_executed:FORBIDDEN"),
    (lambda e: e.update(decision="PAPER_CANDIDATE"), "decision:EVIDENCE_CEILING"),
    (lambda e: e["facts"].append("Out-of-sample Sharpe of 2.1."), "claims:E0_EMPIRICAL"),
    (lambda e: e.update(run_state_after="HUMAN_REVIEW"), "run_state_after:STATE_CHANGE_FORBIDDEN"),
    (lambda e: e.update(promoted=True), "envelope:FIELDS"),
])
def test_validator_rejects_overclaims_and_authority(mutate, error):
    candidate = copy.deepcopy(envelope())
    mutate(candidate)
    assert error in validator.validate_envelope(candidate)


def test_command_line_exit_codes(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(envelope()))
    bad.write_text(json.dumps({**envelope(), "decision": "SHIP"}))
    run = lambda p: subprocess.run([sys.executable, str(TOOL / "validate_envelope.py"), str(p)], capture_output=True, text=True)
    assert (run(good).returncode, run(good).stdout.strip()) == (0, "VALID")
    assert run(bad).returncode == 1
