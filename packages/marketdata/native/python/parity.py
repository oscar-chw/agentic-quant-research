"""Output parity between the C++ port and the Python reference. Exit 0 only on exact match.

1. Export the SYNTHETIC 100,000-message workload and the reference answer
   (export_workload.py also pins both to the published evidence digests).
2. Run the C++ driver on it, batch and scalar, and compare the output bytes
   with the reference's. Prices are integer ticks, so no tolerance applies.
3. Re-record the golden differential cases from the reference and compare with
   the committed file, so `make test` cannot pass against a stale oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sh(argv) -> None:
    done = subprocess.run([str(a) for a in argv], cwd=ROOT)
    if done.returncode:
        sys.exit(f"parity: command failed ({done.returncode}): {' '.join(map(str, argv))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, help="directory holding the reference replay package")
    parser.add_argument("--cli", required=True, help="built replay_cli")
    args = parser.parse_args()
    build = ROOT / "build"
    workload, expected = build / "workload.txt", build / "expected.json"

    sh([sys.executable, HERE / "export_workload.py", "--ref", args.ref, "--workload", workload, "--expected", expected])
    reference = expected.read_bytes()
    failed = False
    for mode in ("batch", "scalar"):
        out = build / f"cpp-{mode}.json"
        sh([args.cli, *(["--scalar"] if mode == "scalar" else []), workload, out])
        actual = out.read_bytes()
        same = actual == reference
        failed |= not same
        print(f"{'PASS' if same else 'FAIL'}  C++ {mode}: {len(actual)} bytes, sha256 "
              f"{hashlib.sha256(actual).hexdigest()} vs reference {hashlib.sha256(reference).hexdigest()}")

    golden = build / "golden-rerecorded.txt"
    sh([sys.executable, HERE / "make_golden.py", "--ref", args.ref, "--out", golden])
    same = golden.read_bytes() == (ROOT / "tests/golden/reference_cases.txt").read_bytes()
    failed |= not same
    print(f"{'PASS' if same else 'FAIL'}  committed golden cases match a fresh recording from the reference")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
