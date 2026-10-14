"""Deterministic hand-written replay bound to one canonical shared method note."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from method_contract import shared_note_citation
from qrae.artifacts import canonical_json_bytes
from shared_manifest import SharedManifestError, load_shared_note

NOTE_ID = "quant-pit-cutoffs-v1"
PINNED_MANIFEST = "c15abda5657cc63a953c7d8b8d34f044f0b6dcad09c253a0b9842e5beea4c70a"
PROMPT = "Explain the point-in-time cutoff and one failure it prevents; cite the bound method-note id."
REPLAY = {
    "answer": "A point-in-time cutoff limits each observation to information available at its decision time; it prevents later revisions from leaking into historical features.",
    "source_id": NOTE_ID,
}


def run(shared_root: Path, output: Path, run_id: str) -> dict:
    selected = load_shared_note(shared_root, NOTE_ID, expected_manifest_sha256=PINNED_MANIFEST)
    if selected["status"] != "MATCHES":
        raise SharedManifestError("SHARED_NOTE_NO_MATCH", "the replay source id was not found")
    note = selected["matches"][0]
    prompt_bytes = canonical_json_bytes({"prompt": PROMPT, "source_ids": [NOTE_ID]})
    record = {
        "schema": "quantos-shared-note-replay/v1",
        "run_id": run_id,
        "execution": "HAND_WRITTEN_REPLAY",
        "claim": "offline source-consumer demonstration; no live model or performance claim",
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "response": REPLAY,
        "citation": {key: value for key, value in shared_note_citation(note).items() if key != "document"},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_bytes(canonical_json_bytes(record))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if not args.run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in args.run_id):
        parser.error("run id must contain only letters, digits, dot, underscore, or hyphen")
    result = run(args.shared_root, args.output / args.run_id, args.run_id)
    print(json.dumps({"execution": result["execution"], "run": str(args.output / args.run_id / "run.json"),
                      "note_id": result["citation"]["document_id"],
                      "note_sha256": result["citation"]["source_sha256"],
                      "manifest_sha256": result["citation"]["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
