"""Command-line interface for the local QRAE QuantOS kernel."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import ArtifactError
from .codex_broker import (
    BrokerValidationError,
    OutboxConflictError,
    run_work_order,
)
from .codex_reviews import (
    ReviewBundleError,
    ingest_review_bundle,
    verify_review_bundle,
)
from .contracts import ContractError
from .data_catalog import CatalogError, DataCatalog
from .historical_data import (
    HistoricalDataError,
    catalog_historical_source,
)
from .kernel import KernelError, run_price_baseline, verify_run
from .polymarket import PolymarketAdapterError, fetch_markets
from .price_baseline import DataQualityError
from .price_import import PriceBarImportError, import_price_bars, load_adapter_contract
from .research_workflow import (
    ResearchWorkflowError,
    prepare_review_task,
    run_research_draft,
    verify_research_draft_workflow,
)
from .state import StateLogError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qrae", description="Local-first, research-only QuantOS kernel"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="execute one validated local research work order"
    )
    run.add_argument("--work-order", required=True)
    run.add_argument("--workspace", default=".")
    run.add_argument("--output-root", default=".quantos")
    verify = subparsers.add_parser("verify", help="verify an immutable run bundle")
    verify.add_argument("--run-dir", required=True)
    research_draft = subparsers.add_parser(
        "research-draft",
        help="run research and produce one authenticated local Codex review workflow",
    )
    research_draft.add_argument("--work-order", required=True)
    research_draft.add_argument("--workspace", default=".")
    research_draft.add_argument("--output-root", default=".quantos")
    research_draft.add_argument("--task-id", required=True)
    research_draft.add_argument(
        "--task-kind",
        choices=["HYPOTHESIS_REVIEW", "ADVERSARIAL_CRITIC", "REPORT_DRAFT"],
        default="REPORT_DRAFT",
    )
    research_draft.add_argument("--input", default="report.md")
    research_draft.add_argument("--enable-codex", action="store_true")
    research_draft.add_argument("--receipt-key-root")
    research_draft.add_argument("--expires-minutes", type=int, default=60)
    research_draft.add_argument("--timeout-seconds", type=int, default=120)
    research_draft.add_argument("--max-output-bytes", type=int, default=128 * 1024)
    workflow_verify = subparsers.add_parser(
        "research-draft-verify",
        help="verify a composed research/Codex workflow receipt",
    )
    workflow_verify.add_argument("--workflow", required=True)
    workflow_verify.add_argument("--receipt-key-root")
    prepare = subparsers.add_parser(
        "codex-prepare", help="prepare a bounded local Codex review task"
    )
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--task-id", required=True)
    prepare.add_argument(
        "--task-kind",
        required=True,
        choices=["HYPOTHESIS_REVIEW", "ADVERSARIAL_CRITIC", "REPORT_DRAFT"],
    )
    prepare.add_argument("--input", default="report.md")
    prepare.add_argument("--expires-minutes", type=int, default=60)
    prepare.add_argument("--timeout-seconds", type=int, default=120)
    prepare.add_argument("--max-output-bytes", type=int, default=128 * 1024)
    codex_run = subparsers.add_parser(
        "codex-run", help="run or safely defer a prepared local Codex task"
    )
    codex_run.add_argument("--work-order", required=True)
    codex_run.add_argument("--workspace", required=True)
    codex_run.add_argument("--outbox", default="codex-outbox")
    codex_run.add_argument("--enable-codex", action="store_true")
    codex_run.add_argument("--receipt-key-root")
    codex_ingest = subparsers.add_parser(
        "codex-ingest", help="authenticate and freeze one DRAFT_READY Codex review"
    )
    codex_ingest.add_argument("--run-dir", required=True)
    codex_ingest.add_argument("--task-id", required=True)
    codex_ingest.add_argument("--receipt-key-root")
    review_verify = subparsers.add_parser(
        "codex-review-verify", help="verify one immutable Codex review bundle"
    )
    review_verify.add_argument("--bundle-dir", required=True)
    review_verify.add_argument("--receipt-key-root")
    catalog_verify = subparsers.add_parser(
        "catalog-verify", help="rehash and verify a local PIT data catalog"
    )
    catalog_verify.add_argument("--catalog-root", default=".quantos/catalog")
    polymarket_fetch = subparsers.add_parser(
        "polymarket-fetch", help="fetch one bounded read-only Gamma markets page"
    )
    polymarket_fetch.add_argument("--catalog-root", default=".quantos/catalog")
    polymarket_fetch.add_argument("--request-id", required=True)
    polymarket_fetch.add_argument("--limit", type=int, default=100)
    polymarket_fetch.add_argument(
        "--active", action=argparse.BooleanOptionalAction, default=True
    )
    polymarket_fetch.add_argument(
        "--closed", action=argparse.BooleanOptionalAction, default=False
    )
    price_import = subparsers.add_parser(
        "price-bars-import",
        help="catalog one validated local observation of an HTTPS price-bar CSV",
    )
    price_import.add_argument("--catalog-root", default=".quantos/catalog")
    price_import.add_argument("--adapter-contract", required=True)
    price_import.add_argument("--request-id", required=True)
    price_import.add_argument("--source-uri", required=True)
    price_import.add_argument("--csv", required=True)
    historical_import = subparsers.add_parser(
        "historical-import",
        help="validate and freeze one hash-bound all-family historical source at E0",
    )
    historical_import.add_argument("--catalog-root", default=".quantos/catalog")
    historical_import.add_argument("--manifest", required=True)
    historical_import.add_argument("--workspace", default=".")
    historical_import.add_argument("--as-of", required=True)
    historical_import.add_argument("--request-id", required=True)
    return parser


def _utc_argument(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("as-of must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("as-of must include a timezone")
    return parsed.astimezone(timezone.utc)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: datetime | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_price_baseline(
                args.work_order,
                workspace=Path(args.workspace),
                output_root=Path(args.output_root),
                now=now,
            )
        elif args.command == "verify":
            result = verify_run(args.run_dir)
        elif args.command == "research-draft":
            result = run_research_draft(
                args.work_order,
                workspace=Path(args.workspace),
                output_root=Path(args.output_root),
                task_id=args.task_id,
                task_kind=args.task_kind,
                input_name=args.input,
                enable_codex=args.enable_codex,
                receipt_key_root=args.receipt_key_root,
                expires_minutes=args.expires_minutes,
                timeout_seconds=args.timeout_seconds,
                max_output_bytes=args.max_output_bytes,
                now=now,
            )
        elif args.command == "research-draft-verify":
            result = verify_research_draft_workflow(
                args.workflow, receipt_key_root=args.receipt_key_root
            )
        elif args.command == "codex-prepare":
            result = prepare_review_task(
                args.run_dir,
                task_id=args.task_id,
                task_kind=args.task_kind,
                input_name=args.input,
                expires_minutes=args.expires_minutes,
                timeout_seconds=args.timeout_seconds,
                max_output_bytes=args.max_output_bytes,
                now=now,
            )
        elif args.command == "codex-run":
            workspace = Path(args.workspace).resolve()
            order = json.loads(Path(args.work_order).read_text(encoding="utf-8"))
            result = run_work_order(
                order,
                workspace,
                workspace / args.outbox,
                enabled=args.enable_codex,
                receipt_key_root=args.receipt_key_root,
                now=now,
            )
        elif args.command == "codex-ingest":
            result = ingest_review_bundle(
                args.run_dir,
                task_id=args.task_id,
                receipt_key_root=args.receipt_key_root,
            )
        elif args.command == "codex-review-verify":
            result = verify_review_bundle(
                args.bundle_dir,
                receipt_key_root=args.receipt_key_root,
            )
        elif args.command == "catalog-verify":
            result = DataCatalog(args.catalog_root).verify_catalog()
        elif args.command == "polymarket-fetch":
            result = fetch_markets(
                DataCatalog(args.catalog_root),
                request_id=args.request_id,
                limit=args.limit,
                active=args.active,
                closed=args.closed,
            )
        elif args.command == "historical-import":
            result = catalog_historical_source(
                DataCatalog(args.catalog_root),
                args.manifest,
                workspace=args.workspace,
                as_of=_utc_argument(args.as_of),
                request_id=args.request_id,
                now=now,
            )
        else:
            result = import_price_bars(
                DataCatalog(args.catalog_root),
                adapter_contract=load_adapter_contract(args.adapter_contract),
                request_id=args.request_id,
                source_uri=args.source_uri,
                csv_path=args.csv,
            )
    except (
        ArtifactError,
        BrokerValidationError,
        CatalogError,
        ContractError,
        DataQualityError,
        KernelError,
        HistoricalDataError,
        OutboxConflictError,
        PolymarketAdapterError,
        PriceBarImportError,
        ResearchWorkflowError,
        ReviewBundleError,
        StateLogError,
        OSError,
        ValueError,
    ) as exc:
        code = getattr(exc, "code", exc.__class__.__name__)
        print(
            json.dumps(
                {"ok": False, "error": code, "message": str(exc)}, sort_keys=True
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
