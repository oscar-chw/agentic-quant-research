"""Prepare the real QuantOS proposer input from pinned shared method notes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from method_contract import safe_id, shared_note_citation
from qrae.artifacts import canonical_json_bytes
from shared_manifest import SharedManifestError, load_shared_note

from quantos_showcase.loop import propose_prompt, read_campaign, read_input, strict_json


def prepare(campaign_path, contract_path, root, manifest_sha256, document_ids, output):
    if not 1 <= len(document_ids) <= 8 or len(set(document_ids)) != len(document_ids):
        raise ValueError("select one to eight distinct method-note ids")
    for document_id in document_ids:
        safe_id(document_id)
    campaign_raw = read_input(Path(campaign_path), 64 * 1024)
    contract_raw = read_input(Path(contract_path), 64 * 1024)
    campaign = read_campaign(campaign_raw)
    experiment = strict_json(contract_raw)["experiment"]
    notes = []
    for document_id in document_ids:
        selected = load_shared_note(root, document_id, expected_manifest_sha256=manifest_sha256)
        if selected["status"] != "MATCHES":
            raise SharedManifestError("SHARED_NOTE_NO_MATCH", "selected method note was not found")
        notes.append(selected["matches"][0])
    prompt = propose_prompt(campaign, experiment, notes)
    request = {
        "schema": "quantos-proposal-request/v1", "state": "PROMPT_READY", "execution": "NOT_EXECUTED",
        "task": "PROPOSE_HYPOTHESES", "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "campaign_sha256": hashlib.sha256(campaign_raw).hexdigest(),
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "sources": [{key: value for key, value in shared_note_citation(note).items() if key != "document"}
                    for note in notes],
        "boundary": "Real QuantOS proposal input; provider, proposal validation and backtest have not run.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.json").write_bytes(prompt)
    (output / "request.json").write_bytes(canonical_json_bytes(request))
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--note-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = prepare(args.campaign, args.contract, args.shared_root, args.manifest_sha256,
                      args.note_id, args.output)
    print(json.dumps({"state": request["state"], "execution": request["execution"],
                      "prompt_sha256": request["prompt_sha256"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
