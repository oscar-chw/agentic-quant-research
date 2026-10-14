"""Normal loop integration with explicitly labelled offline proposal test doubles."""

import copy
import json
from pathlib import Path

import pytest
from factor_research.synthetic import make_fixture
from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.llm import ReplayProvider
from shared_manifest import load_shared_note

from quantos_showcase import loop

EXAMPLES = Path(__file__).parents[1] / "examples/loop"
NOTE_ID = "test-method-note"
RIGHTS = {"license": "MIT", "basis": "original_summary", "redistribution": "public"}


@pytest.fixture
def shared(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    text = "# Test-only method\nUse observations available before each decision.\n"
    (root / "method.md").write_text(text)
    row = {"id": NOTE_ID, "document_type": "method_note", "title": "Test-only method",
           "path": "method.md", "sha256": sha256_bytes(text.encode()),
           "source_refs": [{"uri": "https://example.invalid/test", "kind": "public_code", "revision": "test"}],
           "rights": RIGHTS}
    manifest = {"schema": "quant-shared-notes/v1", "revision": "test", "documents": [row]}
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    make_fixture(tmp_path / "panel", 8, 60, persistent=True)
    campaign = json.loads((EXAMPLES / "campaign.json").read_bytes())
    campaign["k"] = 1
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_bytes(canonical_json_bytes(campaign))
    return {"root": root, "tmp": tmp_path, "campaign": campaign_path,
            "digest": sha256_bytes((root / "manifest.json").read_bytes())}


def note_for(shared):
    return load_shared_note(shared["root"], NOTE_ID,
                            expected_manifest_sha256=shared["digest"])["matches"][0]


def proposal_replay(shared):
    draft = {"id": "METHOD_1", "signal": "momentum", "lookback": 5,
             "hypothesis": "Test double: momentum on a synthetic trend.",
             "mechanism": "Test double: rank lagged returns with known observation times.",
             "assumptions": ["Explicitly hand-written test fixture."],
             "limitations": ["No real model or market result."], "citation": NOTE_ID}
    campaign = loop.read_campaign(shared["campaign"].read_bytes())
    experiment = json.loads((EXAMPLES / "synthetic-contract.json").read_bytes())["experiment"]
    prompt = loop.propose_prompt(campaign, experiment, [note_for(shared)])
    replay = shared["tmp"] / "proposal-only.json"
    replay.write_bytes(canonical_json_bytes({
        "schema": "qrae.llm-replay/v1", "provenance": "HAND-WRITTEN TEST DOUBLE; no critic response",
        "entries": [{"key": "propose", "prompt_sha256": sha256_bytes(prompt),
                     "response": json.dumps({"hypotheses": [draft]})}]}))
    return replay


def run(shared, provider=None, **selection):
    options = {"shared_root": shared["root"], "manifest_sha256": shared["digest"], "note_ids": [NOTE_ID]}
    options.update(selection)
    return loop.run_loop(shared["campaign"], shared["tmp"] / "panel/panel.csv",
                         EXAMPLES / "synthetic-contract.json", None, shared["tmp"] / "store", "shared",
                         provider or ReplayProvider(proposal_replay(shared)),
                         receipt_keys=shared["tmp"] / "keys", **options)


def folder_for(shared):
    return shared["tmp"] / "store/shared"


def test_normal_run_records_method_provenance_and_missing_critic_keeps_test_sealed(shared, monkeypatch):
    native_test = loop.test_card
    frozen_seen = []

    def observe(run_dir, hid, gate=False):
        folder = loop.card_dir(run_dir, hid)
        card = json.loads((folder / "card.json").read_bytes())
        registration = json.loads((folder / "registration.json").read_bytes())
        assert card["sources"][0]["kind"] == "local_derivation"
        assert registration["source_provenance"]["document_type"] == "method_note"
        frozen_seen.append(hid)
        assert not gate, "missing critic must never open the holdout"
        return native_test(run_dir, hid, gate)

    monkeypatch.setattr(loop, "test_card", observe)
    ledger = run(shared)
    assert frozen_seen == ["METHOD_1"]
    source = ledger["inputs"]["retrieved"][0]
    assert source == {key: value for key, value in loop.method_contract.shared_note_citation(note_for(shared)).items()
                      if key != "document"}
    assert "arxiv_id" not in json.dumps(ledger)
    assert source["document_id"] == NOTE_ID and source["manifest_sha256"] == shared["digest"]
    assert ledger["provider"]["provenance"] == "HAND-WRITTEN TEST DOUBLE; no critic response"
    row = ledger["hypotheses"][0]
    assert row["citation"] == NOTE_ID and row["citation_type"] == "method_note"
    assert row["status"] == "CRITIC_UNUSABLE" and row["critic"]["reason_code"] == "RUNNER_FAILURE"
    critic_result = json.loads((folder_for(shared) / "critic/outbox/shared/critic-METHOD_1.json").read_bytes())
    assert critic_result["status"] == "QUARANTINED" and critic_result["payload"] is None
    assert row["decision"]["split_status"]["test"] == "SEALED" and row["decision"]["metrics"]["test"] is None
    assert not (folder_for(shared) / "gate").exists()
    trial = folder_for(shared) / "trials/METHOD_1"
    dates = [line.split(",")[0] for line in (trial / "panel.csv").read_text().splitlines()[1:]]
    assert dates and max(dates) < "2024-02-10T16:00:00"
    assert json.loads((trial / "report.json").read_bytes())["test"]["summary"]["valid_ic_dates"] == 0
    assert loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")["verified"]
    with pytest.raises(ValueError, match="not awaiting"):
        loop.gate(folder_for(shared), promote=["METHOD_1"])


class NeverCalled:
    def complete(self, *_):
        pytest.fail("invalid shared sources must refuse before any provider call")


@pytest.mark.parametrize("mutation", ["text", "manifest", "rights", "escape", "refs", "type", "missing"])
def test_shared_source_errors_refuse_before_execution(shared, mutation):
    manifest_path = shared["root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    row = manifest["documents"][0]
    if mutation == "text":
        (shared["root"] / "method.md").write_text("tampered")
    elif mutation == "manifest":
        shared["digest"] = "0" * 64
    else:
        if mutation == "rights":
            row["rights"]["license"] = "unknown"
        elif mutation == "escape":
            row["path"] = "../outside.md"
        elif mutation == "refs":
            row["source_refs"] = []
        elif mutation == "type":
            row["document_type"] = "paper_note"
        else:
            manifest["documents"] = []
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        shared["digest"] = sha256_bytes(manifest_path.read_bytes())
    with pytest.raises(ValueError):
        run(shared, NeverCalled())
    assert not (shared["tmp"] / "store").exists()


@pytest.mark.parametrize("selection", [
    {"note_ids": []}, {"note_ids": [NOTE_ID] * 2}, {"note_ids": [f"id-{i}" for i in range(9)]},
    {"note_ids": ["n" * 65]}, {"manifest_sha256": None}, {"shared_root": None},
])
def test_selection_is_explicit_and_bounded(shared, selection):
    with pytest.raises(ValueError):
        run(shared, NeverCalled(), **selection)
    assert not (shared["tmp"] / "store").exists()


def rehash_bundle(folder):
    manifest = json.loads((folder / "manifest.json").read_bytes())
    for entry in manifest["artifacts"]:
        raw = (folder / entry["path"]).read_bytes()
        entry.update(bytes=len(raw), sha256=sha256_bytes(raw))
    (folder / "manifest.json").write_bytes(canonical_json_bytes(manifest))


@pytest.mark.parametrize("target", ["ledger_id", "ledger_type", "ledger_digest", "ledger_refs", "ledger_rights",
                                  "registration", "source", "card_kind", "prompt", "response", "selection", "downgrade"])
def test_verification_binds_shared_provenance_across_run_artifacts(shared, target):
    run(shared)
    folder = folder_for(shared)
    card_dir = loop.card_dir(folder, "METHOD_1")
    ledger = loop.read_ledger(folder)
    if target.startswith("ledger_"):
        source = ledger["inputs"]["retrieved"][0]
        field, value = {"ledger_id": ("document_id", "another-note"), "ledger_type": ("document_type", "paper_note"),
                        "ledger_digest": ("manifest_sha256", "0" * 64), "ledger_refs": ("source_refs", []),
                        "ledger_rights": ("rights", {})}[target]
        source[field] = value
    elif target == "selection":
        del ledger["inputs"]["source_selection"]
    elif target == "downgrade":
        del ledger["inputs"]["source_selection"]
        ledger["inputs"]["retrieved"] = [{"id": NOTE_ID, "sha256": note_for(shared)["source_sha256"]}]
        path = folder / "propose/prompt.json"
        prompt = json.loads(path.read_bytes())
        prompt["notes"] = [{key: value for key, value in source.items() if key in {"id", "title", "text"}}
                           for source in prompt["notes"]]
        path.write_bytes(canonical_json_bytes(prompt))
    elif target == "registration":
        path = card_dir / "registration.json"
        registration = json.loads(path.read_bytes())
        registration["source_provenance"]["manifest_sha256"] = "0" * 64
        path.write_bytes(canonical_json_bytes(registration))
        rehash_bundle(card_dir)
    elif target == "card_kind":
        path = card_dir / "card.json"
        card = json.loads(path.read_bytes())
        card["sources"][0]["kind"] = "paper_note"
        path.write_bytes(canonical_json_bytes(card))
        ledger["hypotheses"][0]["card_sha256"] = sha256_bytes(path.read_bytes())
        rehash_bundle(card_dir)
    else:
        path = {"source": card_dir / "source-0.md", "prompt": folder / "propose/prompt.json",
                "response": folder / "propose/response.txt"}[target]
        path.write_bytes(path.read_bytes() + b" ")
        if target == "source":
            rehash_bundle(card_dir)
    (folder / "ledger.json").write_bytes(canonical_json_bytes(ledger))
    (folder / "REPORT.md").write_text(loop.render(ledger))
    with pytest.raises(ValueError):
        loop.verify(folder, receipt_keys=shared["tmp"] / "keys")


def test_cli_runs_shared_proposal_only_replay(shared):
    replay = proposal_replay(shared)
    code = loop.main(["run", "--campaign", str(shared["campaign"]),
                      "--prices", str(shared["tmp"] / "panel/panel.csv"),
                      "--contract", str(EXAMPLES / "synthetic-contract.json"),
                      "--shared-root", str(shared["root"]), "--manifest-sha256", shared["digest"],
                      "--note-id", NOTE_ID, "--store", str(shared["tmp"] / "store"),
                      "--run-id", "shared", "--replay", str(replay), "--receipt-keys", str(shared["tmp"] / "keys")])
    assert code == 0
    assert loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")["verified"]


def test_shared_propose_only_and_normal_run_bind_identical_prompts(shared):
    experiment = shared["tmp"] / "experiment.json"
    experiment.write_bytes(canonical_json_bytes(
        json.loads((EXAMPLES / "synthetic-contract.json").read_bytes())["experiment"]))
    provider = ReplayProvider(proposal_replay(shared))
    early = loop.propose_only(shared["campaign"], experiment, None, provider, shared_root=shared["root"],
                              manifest_sha256=shared["digest"], note_ids=[NOTE_ID])
    later = run(shared, provider)
    assert early["prompt_sha256"] == later["provider"]["propose_prompt_sha256"]
    assert early["hypotheses"][0]["citation_type"] == "method_note"


def test_stale_shared_critic_still_aborts_and_preserves_replay(shared):
    path = proposal_replay(shared)
    replay = json.loads(path.read_bytes())
    replay["entries"].append({"key": "critic:shared:METHOD_1", "prompt_sha256": "0" * 64,
                              "response": "DELIBERATELY STALE TEST DOUBLE"})
    path.write_bytes(canonical_json_bytes(replay))
    original = path.read_bytes()
    with pytest.raises(LookupError, match="stale replay entry"):
        run(shared, ReplayProvider(path))
    assert not folder_for(shared).exists() and path.read_bytes() == original


def test_large_unicode_card_refuses_before_numerical_execution(shared, monkeypatch):
    path = proposal_replay(shared)
    replay = json.loads(path.read_bytes())
    proposal = json.loads(replay["entries"][0]["response"])
    draft = proposal["hypotheses"][0]
    draft["assumptions"] = ["雪" * 1000] * 16
    draft["limitations"] = ["雪" * 1000] * 16
    replay["entries"][0]["response"] = json.dumps(proposal)
    path.write_bytes(canonical_json_bytes(replay))
    monkeypatch.setattr(loop, "test_card", lambda *args, **kwargs: pytest.fail("oversized card must not execute"))
    with pytest.raises(ValueError, match="64 KiB"):
        run(shared, ReplayProvider(path))
    assert not folder_for(shared).exists()


@pytest.mark.parametrize("count", [2, 8])
def test_multiple_explicit_sources_retain_order_and_verify(shared, count):
    path = shared["root"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    original = manifest["documents"][0]
    ids = [NOTE_ID] + [f"test-method-{i}" for i in range(1, count)]
    manifest["documents"] = [{**original, "id": identity} for identity in ids]
    path.write_bytes(canonical_json_bytes(manifest))
    shared["digest"] = sha256_bytes(path.read_bytes())
    replay_path = proposal_replay(shared)
    replay = json.loads(replay_path.read_bytes())
    campaign = loop.read_campaign(shared["campaign"].read_bytes())
    experiment = json.loads((EXAMPLES / "synthetic-contract.json").read_bytes())["experiment"]
    notes = [load_shared_note(shared["root"], identity, expected_manifest_sha256=shared["digest"])["matches"][0]
             for identity in ids]
    replay["entries"][0]["prompt_sha256"] = sha256_bytes(loop.propose_prompt(campaign, experiment, notes))
    replay_path.write_bytes(canonical_json_bytes(replay))
    ledger = run(shared, ReplayProvider(replay_path), note_ids=ids)
    assert [row["document_id"] for row in ledger["inputs"]["retrieved"]] == ids
    assert loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")["verified"]


def test_method_holdout_injection_is_refused(shared):
    run(shared)
    folder = folder_for(shared)
    (folder / "gate/outcomes").mkdir(parents=True)
    (folder / "gate/outcomes/METHOD_1.json").write_text("{}")
    with pytest.raises(ValueError, match="reached the gate"):
        loop.verify(folder, receipt_keys=shared["tmp"] / "keys")


def test_shared_record_verification_uses_frozen_source_after_corpus_changes(shared):
    run(shared)
    (shared["root"] / "method.md").write_text("changed after the admitted snapshot")
    assert loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")["verified"]
    with pytest.raises(ValueError, match="STALE"):
        loop.select_sources({}, None, shared_root=shared["root"], manifest_sha256=shared["digest"], note_ids=[NOTE_ID])


@pytest.mark.parametrize("field,value", [("lookback", 6), ("horizon", 2)])
def test_verified_execution_parameters_must_match_proposal_and_prompt(shared, monkeypatch, field, value):
    encode = loop.canonical_json_bytes

    def wrong_frozen_contract(document):
        if isinstance(document, dict) and document.get("schema_version") == "factor-price-import/v1":
            document = copy.deepcopy(document)
            document["experiment"][field] = value
        return encode(document)

    monkeypatch.setattr(loop, "canonical_json_bytes", wrong_frozen_contract)
    run(shared)
    with pytest.raises(ValueError, match="execution parameters"):
        loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")


@pytest.mark.parametrize("target", ["prices", "contract"])
def test_verification_binds_frozen_inputs_to_admitted_inputs(shared, monkeypatch, target):
    freeze = loop.freeze

    def substitute(run_dir, draft, campaign, contract, prices, note):
        if target == "prices":
            prices = prices.rstrip(b"\r\n")
        else:
            contract = copy.deepcopy(contract)
            contract["price_semantics"]["unit"] = "substituted units"
        return freeze(run_dir, draft, campaign, contract, prices, note)

    monkeypatch.setattr(loop, "freeze", substitute)
    run(shared)
    with pytest.raises(ValueError, match="admitted input"):
        loop.verify(folder_for(shared), receipt_keys=shared["tmp"] / "keys")


def test_legacy_prompt_and_replay_keep_exact_bytes_and_paper_contract(shared):
    replay_path = EXAMPLES / "replay.hand-written.json"
    replay = json.loads(replay_path.read_bytes())
    expected = next(entry for entry in replay["entries"] if entry["key"] == "propose")
    ledger = loop.run_loop(EXAMPLES / "campaign.json", shared["tmp"] / "panel/panel.csv",
                           EXAMPLES / "synthetic-contract.json", EXAMPLES / "vault", shared["tmp"] / "legacy",
                           "control", ReplayProvider(replay_path), receipt_keys=shared["tmp"] / "keys")
    folder = shared["tmp"] / "legacy/control"
    assert sha256_bytes((folder / "propose/prompt.json").read_bytes()) == expected["prompt_sha256"]
    assert (folder / "propose/response.txt").read_text() == expected["response"]
    assert "source_selection" not in ledger["inputs"]
    assert all(set(source) == {"id", "sha256"} for source in ledger["inputs"]["retrieved"])
    card = json.loads((loop.card_dir(folder, "H1") / "card.json").read_bytes())
    assert card["sources"][0]["kind"] == "paper_note"
    assert loop.verify(folder, receipt_keys=shared["tmp"] / "keys")["verified"]
