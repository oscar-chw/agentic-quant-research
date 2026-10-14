"""Invalid user specifications must not consume a proposer call or reserve a run."""

import copy
import json
from pathlib import Path

import method_contract
import pytest
from factor_research.synthetic import make_fixture
from hypothesis import given
from hypothesis import strategies as st

from quantos_showcase import loop

EXAMPLES = Path(__file__).parents[1] / "examples/loop"
BASE = json.loads((EXAMPLES / "campaign.json").read_bytes())
INVALID = [
    ("question", None), ("question", "  "), ("question", {"text": "question"}),
    ("max_lookback", 0), ("max_lookback", True), ("max_lookback", "10"), ("max_lookback", 251),
    ("signals", None), ("signals", {"momentum": True}), ("signals", [["momentum"]]),
    ("data_label", ["SYNTHETIC"]),
    ("retrieval.query", None), ("retrieval.query", " "),
    ("retrieval.n_results", True), ("retrieval.n_results", 0), ("retrieval.n_results", 129),
    ("rule.min_valid_dates", 5001),
]


def changed(field, value):
    result = copy.deepcopy(BASE)
    target = result
    parts = field.split(".")
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = value
    return result


@pytest.mark.parametrize("field,value", INVALID)
def test_invalid_campaign_names_field(field, value):
    with pytest.raises(ValueError, match=field.split(".")[-1]):
        loop.read_campaign(json.dumps(changed(field, value)).encode())


@pytest.mark.parametrize("entrypoint", ["propose", "run"])
def test_invalid_campaign_refuses_before_provider_and_all_io(tmp_path, monkeypatch, entrypoint):
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(changed("max_lookback", 0)))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid campaign reached retrieval or proposal")

    monkeypatch.setattr(loop, "select_sources", forbidden)
    provider = type("Provider", (), {"complete": forbidden})()
    with pytest.raises(ValueError, match="max_lookback"):
        if entrypoint == "propose":
            loop.propose_only(campaign, tmp_path / "absent-contract", None, provider)
        else:
            loop.run_loop(campaign, tmp_path / "absent-prices", tmp_path / "absent-contract", None,
                          tmp_path / "store", "attempt", provider)
    assert not (tmp_path / "store").exists()


def test_cli_refusal_is_actionable_before_model_call(tmp_path, monkeypatch, capsys):
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(changed("signals", None)))
    calls = []

    class Provider:
        def complete(self, *args):
            calls.append(args)
            pytest.fail("invalid campaign consumed a model call")

    monkeypatch.setattr(loop, "provider_for", lambda _: Provider())
    code = loop.main(["run", "--campaign", str(campaign), "--prices", "unused",
                      "--contract", "unused", "--store", str(tmp_path / "store"),
                      "--run-id", "attempt", "--vault", "unused", "--replay", "unused"])
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "REFUSED" and "signals" in error["error"]
    assert calls == [] and not (tmp_path / "store").exists()


@given(lookback=st.integers(1, 250), dates=st.integers(1, 5000),
       signal=st.sampled_from(["momentum", "reversal"]))
def test_admitted_numeric_boundaries_match_native_card(lookback, dates, signal):
    campaign = changed("max_lookback", lookback)
    campaign["rule"]["min_valid_dates"] = dates
    campaign["signals"] = [signal]
    admitted = loop.read_campaign(loop.canonical_json_bytes(campaign))
    proposal = dict(id="contract-witness", signal=signal, lookback=lookback,
                    hypothesis="A declared lagged signal may rank later returns.",
                    mechanism="Test only the frozen chronological validation split.",
                    assumptions=["Declared input clocks are accurate."],
                    limitations=["A valid contract is not evidence of predictive power."], citation="source")
    experiment = {"cost_bps": 0, "splits": {s: {"start": "x", "end": "x"}
                                             for s in ("train", "validation", "test")}}
    card = loop.card_for(proposal, admitted, experiment, "0" * 64, source_kind="local_derivation")
    assert method_contract.validate_card(card) == card


def test_valid_campaign_and_prompt_are_preserved():
    admitted = loop.read_campaign((EXAMPLES / "campaign.json").read_bytes())
    assert admitted == BASE
    experiment = json.loads((EXAMPLES / "synthetic-contract.json").read_bytes())["experiment"]
    assert loop.propose_prompt(admitted, experiment, []) == loop.propose_prompt(BASE, experiment, [])


def test_corrected_campaign_can_use_same_unreserved_run_id(tmp_path, capsys):
    make_fixture(tmp_path / "control", 8, 60, persistent=True)
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(changed("max_lookback", 0)))
    args = ["run", "--campaign", str(campaign), "--prices", str(tmp_path / "control/panel.csv"),
            "--contract", str(EXAMPLES / "synthetic-contract.json"), "--store", str(tmp_path / "store"),
            "--run-id", "control", "--vault", str(EXAMPLES / "vault"),
            "--replay", str(EXAMPLES / "replay.hand-written.json"), "--receipt-keys", str(tmp_path / "keys")]
    assert loop.main(args) == 2
    assert "max_lookback" in json.loads(capsys.readouterr().err)["error"]
    assert not (tmp_path / "store").exists()
    campaign.write_bytes((EXAMPLES / "campaign.json").read_bytes())
    assert loop.main(args) == 0
    capsys.readouterr()
    folder = tmp_path / "store/control"
    verified = loop.verify(folder, receipt_keys=tmp_path / "keys")
    assert verified["verified"] and verified["counts"]["promoted"] == 0
    ledger = loop.read_ledger(folder)
    assert ledger["campaign"] == BASE
    assert ledger["provider"]["provenance"].startswith("HAND-WRITTEN FIXTURE")
