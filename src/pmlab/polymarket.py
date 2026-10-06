"""Polymarket BTC 5-minute windows: market metadata and the taker trade tape.

The Up frame: every trade is re-expressed as a trade in the Up token.
A taker BUYING Down at p holds exactly the position of a taker SELLING Up at 1 - p,
so both become sign = -1 at p_up = 1 - p. After this, one price series and one
signed flow describe the whole market.
"""
import json
import re
import time

import pandas as pd

from pmlab import WINDOW_SECONDS
from pmlab.http import get_json

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
PAGE = 1000
MAX_OFFSET = 10000   # the data-api refuses larger offsets ("max historical trades offset of 10000 exceeded")
BATCH = 100          # Gamma takes at most 100 slugs per /events request


def window_start(ts: int) -> int:
    return ts - ts % WINDOW_SECONDS


def slug(start: int) -> str:
    return f"btc-updown-5m-{start}"


def settlement_rule(m: dict) -> tuple[int | None, str | None]:
    """(seconds averaged at each end, where that was read): 60 or 30 for a Chainlink TWAP stream, 1 for point prices.

    cryptoMarketConfig.twapLookbackSeconds when present ("config"), else the resolution source and description
    ("description"): markets from 2026-08-07 to 2026-08-13 cite the btc-usd-twap-30s stream; earlier ones compare the
    price at the end with the price at the beginning. (None, None) when neither says (same reading as fair.rule_lookback).
    """
    if lookback := (m.get("cryptoMarketConfig") or {}).get("twapLookbackSeconds"):
        return int(lookback), "config"
    text = f"{m.get('resolutionSource') or ''} {m.get('description') or ''}"
    if found := re.search(r"twap-(\d+)s", text, re.IGNORECASE):
        return int(found.group(1)), "description"
    if "twap" not in text.lower() and ("price at the end of the time range" in text
                                       or re.search(r"streams/btc-usd/?(\s|$)", text)):
        return 1, "description"
    return None, None


def parse_market(event: dict) -> dict:
    """`twap_lookback` is the normalised settlement rule (60, 30 or 1 = point price; see settlement_rule)."""
    m = event["markets"][0]
    outcomes = json.loads(m["outcomes"])
    tokens = json.loads(m["clobTokenIds"])
    prices = [float(p) for p in json.loads(m["outcomePrices"])]
    start = int(event["slug"].rsplit("-", 1)[1])
    resolved = m.get("umaResolutionStatus") == "resolved" and sorted(prices) == [0.0, 1.0]
    return {
        "start": start,
        "slug": event["slug"],
        "condition_id": m["conditionId"],
        "up_token": tokens[outcomes.index("Up")],
        "down_token": tokens[outcomes.index("Down")],
        "resolved": resolved,
        "up_won": (prices[outcomes.index("Up")] == 1.0) if resolved else None,
        "twap_lookback": settlement_rule(m)[0],
        "rule_source": settlement_rule(m)[1],
        "fee_schedule": json.dumps(m.get("feeSchedule")),
        "description": m.get("description"),
        "resolutionSource": m.get("resolutionSource"),
    }


def fetch_market(start: int) -> dict | None:
    events = get_json(f"{GAMMA}/events", {"slug": slug(start)})
    return parse_market(events[0]) if events else None


def fetch_markets(starts: list[int]) -> dict[int, dict]:
    """Markets by start, BATCH slugs per request; starts with no market are absent."""
    out = {}
    for i in range(0, len(starts), BATCH):
        chunk = starts[i:i + BATCH]
        events = get_json(f"{GAMMA}/events", [("slug", slug(s)) for s in chunk] + [("limit", len(chunk))])
        for event in events:
            if event.get("markets") and (m := parse_market(event))["start"] in chunk:
                out[m["start"]] = m
    return out


def fetch_raw_trades(condition_id: str, limit: int | None = None) -> list[dict]:
    """Every taker fill of a market, newest first, up to the data-api's offset cap (MAX_OFFSET + limit rows;
    beyond it the oldest fills are missing and coverage_ok tells).

    Cloudflare (max-age 300 s) and the data-api itself cache answers by (market, takerOnly, limit,
    offset), so a repeated URL can return a page minutes old (research/live_tape.md). The page size
    cycles with the clock over 600 s, longer than either cache holds, so each call is new to both.
    A fixed `limit` is for markets closed long ago, whose cached answers are final."""
    limit = limit or PAGE - int(time.time()) % 600
    rows, offset = [], 0
    while offset <= MAX_OFFSET:
        page = get_json(f"{DATA}/trades", {"market": condition_id, "takerOnly": "true",
                                           "limit": limit, "offset": offset})
        rows.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return rows


def normalise_trades(raw: list[dict], market: dict) -> pd.DataFrame:
    """Raw taker trades (API order: newest first) -> chronological Up-frame tape."""
    cols = ["ts", "t", "seq", "outcome", "side", "price", "size",
            "p_up", "sign", "shares", "usdc", "tx"]
    if not raw:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(raw)
    df = df[df["conditionId"] == market["condition_id"]]
    df = df.drop_duplicates(["transactionHash", "asset", "side", "price", "size"])
    # Timestamps are whole seconds. Reversing the API order is the best available
    # guess at arrival order inside a second; it is a guess, and tests must not
    # depend on intra-second order.
    df = df.iloc[::-1].reset_index(drop=True)
    is_up = df["asset"] == market["up_token"]
    if not (is_up | (df["asset"] == market["down_token"])).all():
        raise ValueError("trade on a token that belongs to neither outcome")
    buy = df["side"] == "BUY"
    out = pd.DataFrame({
        "ts": df["timestamp"].astype("int64"),
        "t": df["timestamp"].astype("int64") - market["start"],
        "seq": range(len(df)),
        "outcome": is_up.map({True: "Up", False: "Down"}),
        "side": df["side"],
        "price": df["price"].astype(float),
        "size": df["size"].astype(float),
    })
    out["p_up"] = out["price"].where(is_up, 1.0 - out["price"])
    out["sign"] = (buy == is_up).map({True: 1, False: -1}).astype("int8")
    out["shares"] = out["size"]
    out["usdc"] = out["price"] * out["size"]
    out["tx"] = df["transactionHash"].values
    out = out.sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
    out["seq"] = range(len(out))
    return out[cols]


def in_window(tape: pd.DataFrame) -> pd.DataFrame:
    return tape[(tape["t"] >= 0) & (tape["t"] < WINDOW_SECONDS)].reset_index(drop=True)


def coverage_ok(tape: pd.DataFrame) -> bool:
    """True if the tape provably reaches back past the window start (no truncation)."""
    return len(tape) > 0 and tape["t"].min() < 0
