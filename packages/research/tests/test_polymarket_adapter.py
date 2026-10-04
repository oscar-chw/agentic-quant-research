from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qrae.data_catalog import CatalogError, DataCatalog
from qrae.polymarket import (
    HttpRequest,
    HttpResponse,
    PolymarketAdapterError,
    _RejectRedirects,
    fetch_markets,
    normalize_markets,
)

UTC = timezone.utc
REQUESTED = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
RETRIEVED = REQUESTED + timedelta(seconds=2)


def market(
    market_id: str = "1001", *, updated_at: str = "2026-07-11T09:59:00Z"
) -> dict:
    return {
        "id": market_id,
        "conditionId": f"condition-{market_id}",
        "question": f"Will test market {market_id} resolve yes?",
        "slug": f"test-market-{market_id}",
        "active": True,
        "closed": False,
        "updatedAt": updated_at,
        "endDate": "2026-12-31T00:00:00Z",
        "outcomes": json.dumps(["Yes", "No"]),
        "clobTokenIds": json.dumps([f"yes-{market_id}", f"no-{market_id}"]),
        "outcomePrices": json.dumps(["0.55", "0.45"]),
        "bestBid": "0.54",
        "bestAsk": "0.56",
        "enableOrderBook": True,
        "acceptingOrders": True,
    }


def test_normalizer_creates_sorted_hashable_market_records():
    normalized, metadata = normalize_markets(
        [market("2"), market("1")], retrieved_at=RETRIEVED
    )
    lines = [json.loads(line) for line in normalized.decode("utf-8").splitlines()]

    assert [item["market_id"] for item in lines] == ["1", "2"]
    assert lines[0]["outcomes"] == [
        {"name": "Yes", "token_id": "yes-1", "indicative_price": 0.55},
        {"name": "No", "token_id": "no-1", "indicative_price": 0.45},
    ]
    assert lines[0]["best_bid"] == 0.54
    assert lines[0]["best_ask"] == 0.56
    assert len(lines[0]["source_item_sha256"]) == 64
    assert metadata == {
        "item_count": 2,
        "max_source_event_time": "2026-07-11T09:59:00Z",
        "normalized_schema": "polymarket.gamma.markets.v1",
    }


@pytest.mark.parametrize(
    "payload",
    [
        [market("1"), market("1")],
        [{**market(), "outcomes": json.dumps(["Yes"])}],
        [{**market(), "outcomePrices": json.dumps(["NaN", "0.45"])}],
        [{**market(), "bestBid": "1.2"}],
        [{**market(), "updatedAt": "2026-07-11T11:00:00Z"}],
    ],
)
def test_normalizer_rejects_duplicates_misalignment_nonfinite_bounds_and_future_data(
    payload,
):
    with pytest.raises(PolymarketAdapterError):
        normalize_markets(payload, retrieved_at=RETRIEVED)


def test_fetch_is_read_only_bounded_and_registers_e0_provenance(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    calls: list[HttpRequest] = []
    body = json.dumps([market()]).encode("utf-8")

    def transport(request: HttpRequest) -> HttpResponse:
        calls.append(request)
        return HttpResponse(
            url=request.url,
            status=200,
            headers={
                "content-type": "application/json; charset=utf-8",
                "date": "Sat, 11 Jul 2026 10:00:01 GMT",
                "etag": "fixture-etag",
                "set-cookie": "must-not-enter-provenance",
            },
            body=body,
        )

    times = iter([REQUESTED, RETRIEVED])
    snapshot = fetch_markets(
        catalog,
        request_id="pm-fetch-001",
        limit=1,
        active=True,
        closed=False,
        transport=transport,
        clock=lambda: next(times),
    )

    assert len(calls) == 1
    request = calls[0]
    assert request.url == (
        "https://gamma-api.polymarket.com/markets?limit=1&active=true&closed=false"
    )
    assert request.timeout_seconds == 20
    assert request.max_response_bytes == 8 * 1024 * 1024
    assert {key.casefold() for key in request.headers} == {"accept", "user-agent"}
    assert snapshot["evidence_ceiling"] == "E0"
    assert snapshot["availability_basis"] == "OBSERVED_AT_RETRIEVAL"
    assert snapshot["available_at"] == "2026-07-11T10:00:02Z"
    assert snapshot["cutoff"] == snapshot["available_at"]
    assert snapshot["item_count"] == 1
    assert "set-cookie" not in snapshot["response_headers"]
    assert catalog.verify_catalog()["snapshots"] == 1
    with pytest.raises(CatalogError, match="CATALOG_EVIDENCE_CEILING"):
        catalog.resolve_snapshot(
            snapshot["snapshot_id"],
            as_of=RETRIEVED,
            requested_evidence_tier="E1",
        )


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(
            url="https://evil.example/markets",
            status=200,
            headers={"content-type": "application/json"},
            body=b"[]",
        ),
        HttpResponse(
            url="https://gamma-api.polymarket.com/markets",
            status=200,
            headers={"content-type": "application/json"},
            body=b"[]",
        ),
        HttpResponse(
            url="https://gamma-api.polymarket.com/markets",
            status=500,
            headers={"content-type": "application/json"},
            body=b"[]",
        ),
        HttpResponse(
            url="https://gamma-api.polymarket.com/markets",
            status=200,
            headers={"content-type": "text/html"},
            body=b"<html></html>",
        ),
    ],
)
def test_fetch_rejects_redirect_host_status_and_content_type(
    tmp_path: Path, response: HttpResponse
):
    catalog = DataCatalog(tmp_path / "catalog")

    with pytest.raises(PolymarketAdapterError):
        fetch_markets(
            catalog,
            request_id="pm-fetch-fail",
            limit=1,
            transport=lambda request: response,
            clock=lambda: REQUESTED,
        )
    assert catalog.stats()["snapshots"] == 0


def test_fetch_rejects_invalid_limit_before_transport(tmp_path: Path):
    catalog = DataCatalog(tmp_path / "catalog")
    called = False

    def transport(request: HttpRequest) -> HttpResponse:
        nonlocal called
        called = True
        raise AssertionError("transport must not run")

    with pytest.raises(PolymarketAdapterError, match="PM_INVALID_LIMIT"):
        fetch_markets(
            catalog,
            request_id="pm-fetch-invalid",
            limit=0,
            transport=transport,
            clock=lambda: REQUESTED,
        )
    assert called is False


def test_default_redirect_handler_never_authorizes_followup_request():
    handler = _RejectRedirects()
    assert (
        handler.redirect_request(
            None, None, 302, "Found", {}, "https://127.0.0.1/private"
        )
        is None
    )
