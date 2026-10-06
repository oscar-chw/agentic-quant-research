import json
import math
import os
import subprocess
import threading
from pathlib import Path

import pytest

from qrae.artifacts import ArtifactError, ArtifactRecord, ArtifactStore, verify_manifest


def test_artifact_store_is_immutable_and_commits_verified_latest(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run_001")
    result = store.write_json("result.json", {"value": 1})
    assert store.write_json("result.json", {"value": 1}) == result
    with pytest.raises(ArtifactError, match="different content"):
        store.write_json("result.json", {"value": 2})

    manifest = store.commit_manifest([result], {"evidence_tier": "E0"})
    assert not (tmp_path / "latest.json").exists()
    verify_manifest(store.run_dir)
    store.publish_latest(manifest)
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["manifest_sha256"] == manifest.sha256
    assert verify_manifest(store.run_dir)["metadata"]["evidence_tier"] == "E0"


def test_manifest_verification_detects_tampering(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run_002")
    result = store.write_text("report.md", "original")
    store.commit_manifest([result], {})
    (store.run_dir / "report.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="verification failed"):
        verify_manifest(store.run_dir)


def test_manifest_rejects_unmanifested_files_and_nonfinite_json(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run_004")
    result = store.write_json("result.json", {"value": 1})
    store.commit_manifest([result], {})
    (store.run_dir / "omitted.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="unmanifested"):
        verify_manifest(store.run_dir)
    with pytest.raises(ArtifactError, match="finite canonical JSON"):
        ArtifactStore(tmp_path, "run_005").write_json("bad.json", {"value": math.inf})


def test_manifest_commit_rejects_omitted_and_forged_records(tmp_path: Path):
    omitted_store = ArtifactStore(tmp_path, "run_omitted")
    result = omitted_store.write_json("result.json", {"value": 1})
    omitted_store.write_text("report.md", "required")
    with pytest.raises(ArtifactError, match="complete run bundle"):
        omitted_store.commit_manifest([result], {})

    forged_store = ArtifactStore(tmp_path, "run_forged")
    forged_store.write_json("result.json", {"value": 1})
    forged = ArtifactRecord("result.json", "0" * 64, 12)
    with pytest.raises(ArtifactError, match="verification failed"):
        forged_store.commit_manifest([forged], {})


def test_latest_publication_reverifies_complete_manifest(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run_unverified")
    store.write_json("result.json", {"value": 1})
    manifest = store.write_json(
        "manifest.json",
        {
            "schema_version": "1.0",
            "run_id": store.run_id,
            "metadata": {},
            "artifacts": [],
        },
    )

    with pytest.raises(ArtifactError, match="unmanifested"):
        store.publish_latest(manifest)
    assert not (tmp_path / "latest.json").exists()


def test_concurrent_immutable_writers_cannot_overwrite_each_other(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run_006")
    outcomes: list[str] = []

    def write(value: bytes) -> None:
        try:
            store.write_bytes("value.bin", value)
            outcomes.append("ok")
        except ArtifactError:
            outcomes.append("conflict")

    threads = [
        threading.Thread(target=write, args=(value,)) for value in (b"left", b"right")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["conflict", "ok"]
    assert (store.run_dir / "value.bin").read_bytes() in {b"left", b"right"}


def test_artifact_path_cannot_escape_run_directory(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run_003")
    with pytest.raises(ArtifactError, match="escapes"):
        store.write_text("../outside.txt", "no")


def test_runs_directory_link_cannot_redirect_artifacts_outside_root(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    link = root / "runs"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(external)],
            capture_output=True,
            check=False,
        )
        if created.returncode != 0:
            pytest.skip("directory junctions are unavailable on this platform")
    else:
        link.symlink_to(external, target_is_directory=True)

    with pytest.raises(ArtifactError, match="escapes"):
        ArtifactStore(root, "run_link")
    assert not (external / "run_link").exists()
