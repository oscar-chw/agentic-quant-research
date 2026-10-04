"""Run three native workflows of this monorepo and build their shared review index.

Run from the repository root with the packages importable (see the README).
All generated fixtures/results stay in a new output directory.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

from factor_research.runs import run_trial
from imc4_analysis.cli import main as imc_main
from qrae.artifacts import canonical_json_bytes
from qrae.path_safety import has_link_or_reparse_component

from quantos_showcase.feed import run_feed
from quantos_showcase.run_index import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.repo_root.absolute(), args.output.absolute()
    if has_link_or_reparse_component(root) or has_link_or_reparse_component(output):
        raise ValueError("unlinked roots required")
    feed_example = root / "apps/quantos/examples/feed"
    imc_example = root / "packages/imc-sim/examples/community-log"
    factor_example = root / "packages/factor/examples/trend-reversal"
    required = [
        feed_example / "archive.jsonl",
        feed_example / "spec.json",
        imc_example / "sample.log",
        imc_example / "import.json",
        factor_example / "panel.csv",
        factor_example / "config.json",
    ]
    if not all(p.is_file() and not has_link_or_reparse_component(p) for p in required):
        raise ValueError(
            "run from the repository root; declared examples missing"
        )
    output.mkdir(parents=True, exist_ok=False)
    feed = run_feed(
        feed_example / "archive.jsonl", feed_example / "spec.json", output / "feed"
    )
    with contextlib.redirect_stdout(io.StringIO()) as log:
        status = imc_main(
            [
                "import-log",
                str(imc_example / "sample.log"),
                "--config",
                str(imc_example / "import.json"),
                "--out",
                str(output / "imc"),
            ]
        )
    (output / "imc-command.log").write_text(log.getvalue())
    if status != 0:
        raise ValueError("native IMC command failed; earlier outputs retained")
    factor = run_trial(
        factor_example / "panel.csv",
        factor_example / "config.json",
        output / "factor",
        "baseline",
    )
    paths = [
        dict(
            id="quote-forecast",
            kind="quantos-feed",
            path=str(output / "feed/admissions/runs" / feed["run_id"]),
        ),
        dict(id="imc4-accounting", kind="imc4-import", path=str(output / "imc")),
        dict(id="factor-trial", kind="factor-trial", path=factor["directory"]),
    ]
    for row in paths:
        row["path"] = Path(os.path.relpath(row["path"], output)).as_posix()
    selection = output / "selection.json"
    selection.write_bytes(
        canonical_json_bytes({"schema": "quantos-run-selection/v1", "entries": paths})
    )
    result = build_index(selection, output / "index", "three-home")
    (output / "demo-receipt.json").write_bytes(canonical_json_bytes(result))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "VERIFIED_INDEX" else 2


if __name__ == "__main__":
    raise SystemExit(main())
