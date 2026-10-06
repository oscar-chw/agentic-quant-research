"""Run the committed synthetic fixture and demonstrate replay and tamper gates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qrae.artifacts import ArtifactError
from qrae.kernel import KernelError, run_price_baseline, verify_run


def run_demo(output: Path) -> dict:
    """Create a new demo directory; refuse to overwrite an existing path."""
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    fixture = Path(__file__).resolve().parents[1] / "examples" / "price_baseline"
    shutil.copytree(fixture, workspace / "examples" / "price_baseline")
    order = workspace / "examples" / "price_baseline" / "work_order.json"
    first = run_price_baseline(order, workspace=workspace, output_root=output / "runs")
    second = run_price_baseline(order, workspace=workspace, output_root=output / "runs")
    verified = verify_run(first["run_dir"])
    if (
        not second["idempotent_replay"]
        or first["manifest_sha256"] != second["manifest_sha256"]
    ):
        raise RuntimeError("Identical inputs did not replay the same manifest")
    if first["evidence_tier"] != "E0" or first["live_trading_authorized"]:
        raise RuntimeError(
            "Synthetic research exceeded its evidence or authority ceiling"
        )
    # Corrupt only a disposable copy; retain the original verified bundle.
    corrupted = output / "tampered-copy"
    shutil.copytree(first["run_dir"], corrupted)
    with (corrupted / "result.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    try:
        verify_run(corrupted)
    except (ArtifactError, KernelError):
        tamper_rejected = True
    else:
        raise RuntimeError("Modified result artifact was accepted")
    summary = {
        "fixture": "committed synthetic prices; not investment evidence",
        "status": verified["status"],
        "evidence_tier": first["evidence_tier"],
        "idempotent_replay": second["idempotent_replay"],
        "manifest_sha256": first["manifest_sha256"],
        "tamper_rejected": tamper_rejected,
        "live_trading_authorized": first["live_trading_authorized"],
        "run_dir": str(Path(first["run_dir"]).resolve()),
    }
    (output / "demo-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory to retain artifacts")
    args = parser.parse_args()
    if args.output:
        summary = run_demo(args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="qrae-demo-") as temp:
            summary = run_demo(Path(temp) / "demo")
            summary["run_dir"] = "temporary bundle verified and removed on exit"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
