"""Installed application adapter for the shared quote-feature research module."""

import argparse
import json
from pathlib import Path

from qrae.quote_workflow import run_quotes, verify_quote_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--quotes", type=Path, required=True)
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--store", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--store", type=Path, required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_quotes(args.quotes, args.spec, args.store)
        else:
            result = verify_quote_run(args.store, args.run_id, args.recompute)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
