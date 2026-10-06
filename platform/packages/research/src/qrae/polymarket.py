"""Bounded, read-only Polymarket Gamma market metadata adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import __version__
from .artifacts import canonical_json_bytes, sha256_bytes
from .data_catalog import DataCatalog


class PolymarketAdapterError(ValueError):
    """Gamma data violated the bounded read-only adapter contract."""

    def __init__(self, message: str) -> None:
        self.code = message.partition(":")[0] if ":" in message else "PM_ADAPTER_ERROR"
        super().__init__(message)


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
MARKETS_PATH = "/markets"
MAX_MARKETS = 100
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 20
NORMALIZED_SCHEMA = "polymarket.gamma.markets.v1"
ADAPTER_ID = "polymarket-gamma-markets-v1"

POLYMARKET_ADAPTER_CONTRACT: dict[str, Any] = {
    "schema_version": "1.0",
    "adapter_id": ADAPTER_ID,
    "adapter_version": "1.0.0",
    "market_family": "prediction_markets",
    "source_kind": "HTTPS_JSON",
    "base_url": GAMMA_BASE_URL,
    "allowed_hosts": ["gamma-api.polymarket.com"],
    "read_only": True,
    "auth_mode": "none",
    "data_tier": "TIER_0",
    "normalized_schema": NORMALIZED_SCHEMA,
    "license_status": "UNVERIFIED_PUBLIC_API_TERMS",
    "evidence_ceiling": "E0",
}


@dataclass(frozen=True, slots=True)
class HttpRequest:
    url: str
    headers: Mapping[str, str]
    timeout_seconds: int
    max_response_bytes: int


@dataclass(frozen=True, slots=True)
class HttpResponse:
    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[HttpRequest], HttpResponse]
Clock = Callable[[], datetime]


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PolymarketAdapterError(f"PM_INVALID_TIME: {field} must be UTC")
    if value.utcoffset() != timedelta(0):
        raise PolymarketAdapterError(f"PM_INVALID_TIME: {field} must use UTC")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise PolymarketAdapterError(
            f"PM_INVALID_TIME: {field} must be an ISO timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PolymarketAdapterError(f"PM_INVALID_TIME: {field} is invalid") from exc
    return _utc(parsed, field)


def _required_string(
    item: Mapping[str, Any], field: str, *, limit: int = 20_000
) -> str:
    value = item.get(field)
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or any(char in value for char in ("\x00", "\r"))
    ):
        raise PolymarketAdapterError(
            f"PM_INVALID_MARKET: {field} must be a bounded string"
        )
    return value


def _required_bool(item: Mapping[str, Any], field: str) -> bool:
    value = item.get(field)
    if type(value) is not bool:
        raise PolymarketAdapterError(f"PM_INVALID_MARKET: {field} must be boolean")
    return value


def _optional_bool(item: Mapping[str, Any], field: str) -> bool | None:
    if field not in item or item[field] is None:
        return None
    return _required_bool(item, field)


def _json_array(value: Any, field: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    PolymarketAdapterError(
                        f"PM_INVALID_MARKET: {field} contains {constant}"
                    )
                ),
            )
        except PolymarketAdapterError:
            raise
        except json.JSONDecodeError as exc:
            raise PolymarketAdapterError(
                f"PM_INVALID_MARKET: {field} is not a JSON array"
            ) from exc
    if not isinstance(value, list) or len(value) < 2 or len(value) > 256:
        raise PolymarketAdapterError(
            f"PM_INVALID_MARKET: {field} must be a bounded array"
        )
    return value


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise PolymarketAdapterError(f"PM_INVALID_MARKET: {field} is not numeric")
    if isinstance(value, str) and (not value.strip() or len(value) > 64):
        raise PolymarketAdapterError(f"PM_INVALID_MARKET: {field} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PolymarketAdapterError(
            f"PM_INVALID_MARKET: {field} is not numeric"
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PolymarketAdapterError(f"PM_INVALID_MARKET: {field} is outside [0, 1]")
    return number


def _optional_probability(item: Mapping[str, Any], field: str) -> float | None:
    value = item.get(field)
    if value is None:
        return None
    return _probability(value, field)


def _source_hash(item: Mapping[str, Any]) -> str:
    try:
        return sha256_bytes(canonical_json_bytes(dict(item)))
    except (TypeError, ValueError) as exc:
        raise PolymarketAdapterError(
            "PM_INVALID_MARKET: source item is not finite canonical JSON"
        ) from exc


def normalize_markets(
    payload: Sequence[Mapping[str, Any]], *, retrieved_at: datetime
) -> tuple[bytes, dict[str, Any]]:
    """Normalize a Gamma markets response into deterministic sorted JSONL."""

    retrieved = _utc(retrieved_at, "retrieved_at")
    if (
        not isinstance(payload, list)
        or len(payload) > MAX_MARKETS
        or any(not isinstance(item, Mapping) for item in payload)
    ):
        raise PolymarketAdapterError(
            "PM_INVALID_PAYLOAD: response must be a bounded object list"
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_times: list[datetime] = []
    for item in payload:
        market_id = _required_string(item, "id", limit=256)
        if market_id in seen:
            raise PolymarketAdapterError("PM_DUPLICATE_MARKET: market id is duplicated")
        seen.add(market_id)
        updated = _parse_timestamp(item.get("updatedAt"), "updatedAt")
        if updated > retrieved:
            raise PolymarketAdapterError(
                "PM_FUTURE_DATA: market update is later than retrieval"
            )
        source_times.append(updated)

        outcomes = _json_array(item.get("outcomes"), "outcomes")
        tokens = _json_array(item.get("clobTokenIds"), "clobTokenIds")
        prices = _json_array(item.get("outcomePrices"), "outcomePrices")
        if len(outcomes) != len(tokens) or len(outcomes) != len(prices):
            raise PolymarketAdapterError(
                "PM_ARRAY_ALIGNMENT: outcomes, token ids, and prices differ in length"
            )
        normalized_outcomes: list[dict[str, Any]] = []
        for index, (name, token, price) in enumerate(
            zip(outcomes, tokens, prices, strict=True)
        ):
            if not isinstance(name, str) or not name.strip() or len(name) > 1024:
                raise PolymarketAdapterError(
                    "PM_INVALID_MARKET: outcome name is invalid"
                )
            if not isinstance(token, str) or not token.strip() or len(token) > 512:
                raise PolymarketAdapterError("PM_INVALID_MARKET: token id is invalid")
            normalized_outcomes.append(
                {
                    "name": name,
                    "token_id": token,
                    "indicative_price": _probability(price, f"outcomePrices[{index}]"),
                }
            )

        best_bid = _optional_probability(item, "bestBid")
        best_ask = _optional_probability(item, "bestAsk")
        if best_bid is not None and best_ask is not None and best_bid > best_ask:
            raise PolymarketAdapterError("PM_CROSSED_QUOTES: best bid exceeds best ask")

        end_date: str | None = None
        if item.get("endDate") is not None:
            end_date = _timestamp(_parse_timestamp(item["endDate"], "endDate"))
        records.append(
            {
                "market_id": market_id,
                "condition_id": _required_string(item, "conditionId", limit=512),
                "question": _required_string(item, "question"),
                "slug": _required_string(item, "slug", limit=2048),
                "active": _required_bool(item, "active"),
                "closed": _required_bool(item, "closed"),
                "updated_at": _timestamp(updated),
                "end_date": end_date,
                "outcomes": normalized_outcomes,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "enable_order_book": _optional_bool(item, "enableOrderBook"),
                "accepting_orders": _optional_bool(item, "acceptingOrders"),
                "source_item_sha256": _source_hash(item),
            }
        )

    records.sort(key=lambda record: record["market_id"])
    normalized = b"".join(canonical_json_bytes(record) for record in records)
    return normalized, {
        "item_count": len(records),
        "max_source_event_time": _timestamp(max(source_times))
        if source_times
        else None,
        "normalized_schema": NORMALIZED_SCHEMA,
    }


def _header_map(headers: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(headers, Mapping) or len(headers) > 128:
        raise PolymarketAdapterError("PM_INVALID_HEADERS: response headers are invalid")
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise PolymarketAdapterError(
                "PM_INVALID_HEADERS: response headers are invalid"
            )
        name = key.strip().casefold()
        if name in normalized and normalized[name] != value.strip():
            raise PolymarketAdapterError(
                "PM_INVALID_HEADERS: conflicting duplicate header"
            )
        normalized[name] = value.strip()
    return normalized


def _validate_final_url(url: Any, expected_url: str) -> str:
    if not isinstance(url, str):
        raise PolymarketAdapterError("PM_REDIRECT_REJECTED: response URL is missing")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise PolymarketAdapterError(
            "PM_REDIRECT_REJECTED: response URL is malformed"
        ) from exc
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold().rstrip(".") != "gamma-api.polymarket.com"
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and port != 443)
        or parsed.path.rstrip("/") != MARKETS_PATH
        or parsed.fragment
    ):
        raise PolymarketAdapterError(
            "PM_REDIRECT_REJECTED: response left the allowed endpoint"
        )
    if url != expected_url:
        raise PolymarketAdapterError("PM_REDIRECT_REJECTED: response URL changed")
    return url


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _default_transport(request: HttpRequest) -> HttpResponse:
    outbound = Request(request.url, headers=dict(request.headers), method="GET")
    opener = build_opener(_RejectRedirects())
    try:
        with opener.open(outbound, timeout=request.timeout_seconds) as response:
            body = response.read(request.max_response_bytes + 1)
            return HttpResponse(
                url=response.geturl(),
                status=response.status,
                headers=dict(response.headers.items()),
                body=body,
            )
    except HTTPError as exc:
        body = exc.read(request.max_response_bytes + 1)
        return HttpResponse(
            url=exc.geturl(),
            status=exc.code,
            headers=dict(exc.headers.items()),
            body=body,
        )
    except PolymarketAdapterError:
        raise
    except Exception as exc:
        raise PolymarketAdapterError(
            f"PM_TRANSPORT_ERROR: {type(exc).__name__}"
        ) from exc


def _parse_http_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise PolymarketAdapterError(
            "PM_INVALID_HEADERS: HTTP Date is invalid"
        ) from exc
    return _utc(parsed, "HTTP Date")


def _decode_payload(body: bytes) -> list[Mapping[str, Any]]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PolymarketAdapterError(
                    f"PM_DUPLICATE_KEY: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PolymarketAdapterError(f"PM_NONFINITE_JSON: {constant}")
            ),
        )
    except PolymarketAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolymarketAdapterError(
            "PM_INVALID_JSON: response is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, list):
        raise PolymarketAdapterError("PM_INVALID_PAYLOAD: response root must be a list")
    return value


def fetch_markets(
    catalog: DataCatalog,
    *,
    request_id: str,
    limit: int = 100,
    active: bool = True,
    closed: bool = False,
    transport: Transport = _default_transport,
    clock: Clock = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Fetch one bounded Gamma page and persist an E0 provenance snapshot."""

    if type(limit) is not int or not 1 <= limit <= MAX_MARKETS:
        raise PolymarketAdapterError(
            f"PM_INVALID_LIMIT: limit must be 1..{MAX_MARKETS}"
        )
    if type(active) is not bool or type(closed) is not bool:
        raise PolymarketAdapterError(
            "PM_INVALID_FILTER: active and closed must be booleans"
        )
    if (
        not isinstance(catalog, DataCatalog)
        or not callable(transport)
        or not callable(clock)
    ):
        raise PolymarketAdapterError(
            "PM_INVALID_ARGUMENT: catalog, transport, or clock is invalid"
        )

    query = urlencode(
        [
            ("limit", str(limit)),
            ("active", str(active).lower()),
            ("closed", str(closed).lower()),
        ]
    )
    url = f"{GAMMA_BASE_URL}{MARKETS_PATH}?{query}"
    request = HttpRequest(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"qrae-rd/{__version__} research-only",
        },
        timeout_seconds=TIMEOUT_SECONDS,
        max_response_bytes=MAX_RESPONSE_BYTES,
    )
    requested = _utc(clock(), "requested_at")
    response = transport(request)
    retrieved = _utc(clock(), "retrieved_at")
    if not isinstance(response, HttpResponse):
        raise PolymarketAdapterError(
            "PM_INVALID_RESPONSE: transport returned the wrong type"
        )
    final_url = _validate_final_url(response.url, request.url)
    if type(response.status) is not int or response.status != 200:
        raise PolymarketAdapterError("PM_HTTP_STATUS: Gamma response was not HTTP 200")
    if not isinstance(response.body, bytes) or len(response.body) > MAX_RESPONSE_BYTES:
        raise PolymarketAdapterError(
            "PM_RESPONSE_TOO_LARGE: Gamma response exceeded the bound"
        )
    headers = _header_map(response.headers)
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if content_type != "application/json":
        raise PolymarketAdapterError(
            "PM_CONTENT_TYPE: Gamma response was not application/json"
        )
    content_length = headers.get("content-length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdigit():
            raise PolymarketAdapterError(
                "PM_INVALID_HEADERS: Content-Length is invalid"
            )
        if int(content_length) != len(response.body):
            raise PolymarketAdapterError(
                "PM_LENGTH_MISMATCH: Content-Length differs from body"
            )

    payload = _decode_payload(response.body)
    normalized, metadata = normalize_markets(payload, retrieved_at=retrieved)
    source_http_date = _parse_http_date(headers.get("date"))
    safe_headers = {
        key: value
        for key, value in headers.items()
        if key
        in {
            "cache-control",
            "content-length",
            "content-type",
            "date",
            "etag",
            "expires",
            "last-modified",
            "vary",
        }
    }
    catalog.register_adapter(POLYMARKET_ADAPTER_CONTRACT, registered_at=requested)
    max_source_event_time = (
        _parse_timestamp(metadata["max_source_event_time"], "max_source_event_time")
        if metadata["max_source_event_time"] is not None
        else None
    )
    return catalog.register_snapshot(
        request_id=request_id,
        adapter_id=ADAPTER_ID,
        source_uri=final_url,
        requested_at=requested,
        retrieved_at=retrieved,
        available_at=retrieved,
        cutoff=retrieved,
        availability_basis="OBSERVED_AT_RETRIEVAL",
        raw_bytes=response.body,
        normalized_bytes=normalized,
        normalized_schema=metadata["normalized_schema"],
        item_count=metadata["item_count"],
        max_source_event_time=max_source_event_time,
        source_http_date=source_http_date,
        response_headers=safe_headers,
    )


__all__ = [
    "ADAPTER_ID",
    "POLYMARKET_ADAPTER_CONTRACT",
    "HttpRequest",
    "HttpResponse",
    "PolymarketAdapterError",
    "fetch_markets",
    "normalize_markets",
]
