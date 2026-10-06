import json

import numpy as np
import pandas as pd
import pytest

from pmlab import binance, polymarket

UP, DOWN, CID = "111", "222", "0xabc"
MARKET = {"start": 1000, "condition_id": CID, "up_token": UP, "down_token": DOWN}


def raw(ts, asset, side, price, size, tx, cid=CID):
    return {"timestamp": ts, "asset": asset, "side": side, "price": price,
            "size": size, "transactionHash": tx, "conditionId": cid}


@pytest.mark.case
def test_window_math():
    assert polymarket.window_start(1789573965) == 1789573800
    assert polymarket.window_start(1789574100) == 1789574100
    assert polymarket.slug(1789573800) == "btc-updown-5m-1789573800"


@pytest.mark.case
def test_parse_market_maps_tokens_by_outcome_name_not_position():
    event = {"slug": "btc-updown-5m-1000", "markets": [{
        "outcomes": json.dumps(["Down", "Up"]), "clobTokenIds": json.dumps([DOWN, UP]),
        "outcomePrices": json.dumps(["0", "1"]), "umaResolutionStatus": "resolved",
        "conditionId": CID, "cryptoMarketConfig": {"twapLookbackSeconds": 60},
        "feeSchedule": {"rate": 0.07}}]}
    m = polymarket.parse_market(event)
    assert (m["start"], m["up_token"], m["down_token"]) == (1000, UP, DOWN)
    assert m["resolved"] and m["up_won"] is True and m["twap_lookback"] == 60


@pytest.mark.case
def test_unresolved_market_has_no_winner():
    event = {"slug": "btc-updown-5m-1000", "markets": [{
        "outcomes": json.dumps(["Up", "Down"]), "clobTokenIds": json.dumps([UP, DOWN]),
        "outcomePrices": json.dumps(["0.62", "0.38"]), "umaResolutionStatus": None,
        "conditionId": CID}]}
    m = polymarket.parse_market(event)
    assert not m["resolved"] and m["up_won"] is None


@pytest.mark.case
def test_normalise_puts_every_trade_in_the_up_frame():
    api_order_newest_first = [
        raw(1004, DOWN, "SELL", 0.40, 8, "d"),   # sell Down @.40  == buy  Up @.60
        raw(1003, UP, "SELL", 0.55, 5, "c"),     # sell Up   @.55
        raw(1002, DOWN, "BUY", 0.45, 20, "b"),   # buy  Down @.45  == sell Up @.55
        raw(1002, DOWN, "BUY", 0.45, 20, "b"),   # duplicate row
        raw(1001, UP, "BUY", 0.60, 10, "a"),     # buy  Up   @.60
        raw(1001, UP, "BUY", 0.99, 1, "x", cid="0xother"),
    ]
    t = polymarket.normalise_trades(api_order_newest_first, MARKET)
    assert list(t["tx"]) == ["a", "b", "c", "d"]
    assert list(t["t"]) == [1, 2, 3, 4]
    assert list(t["p_up"]) == pytest.approx([0.60, 0.55, 0.55, 0.60])
    assert list(t["sign"]) == [1, -1, -1, 1]
    assert list(t["usdc"]) == pytest.approx([6.0, 9.0, 2.75, 3.2])
    assert list(t["seq"]) == [0, 1, 2, 3]


@pytest.mark.case
def test_foreign_token_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError):
        polymarket.normalise_trades([raw(1001, "999", "BUY", 0.5, 1, "a")], MARKET)


@pytest.mark.case
def test_in_window_and_coverage():
    t = pd.DataFrame({"t": [-5, 0, 299, 300]})
    assert list(polymarket.in_window(t)["t"]) == [0, 299]
    assert polymarket.coverage_ok(t)
    assert not polymarket.coverage_ok(t[t["t"] >= 0])


@pytest.mark.case
def test_klines_gap_fill_and_price_at(monkeypatch):
    k = lambda s, c: [s * 1000, "1", "1", "1", str(c), "0.5", s * 1000 + 999, "0", 3]
    monkeypatch.setattr(binance, "get_json", lambda url, params: [k(10, 100.0), k(12, 102.0)])
    df = binance.fetch_seconds(10, 13)
    assert list(df["sec"]) == [10, 11, 12]
    assert list(df["close"]) == [100.0, 100.0, 102.0]      # second 11 had no trades
    assert list(df["n_trades"]) == [3, 0, 3]
    price_at = lambda instant: float(df.loc[df["sec"] == instant - 1, "close"].iloc[0])   # the close of [i − 1, i)
    assert price_at(12) == 100.0 and price_at(13) == 102.0


@pytest.mark.case
def test_a_kline_range_binance_returns_nothing_for_is_the_empty_gap_filled_grid(monkeypatch):
    """pricing review, binance 1: an outage or maintenance span raised IndexError instead of the promised 1 s grid."""
    monkeypatch.setattr(binance, "get_json", lambda url, params: [])
    df = binance.fetch_seconds(10, 13)
    assert list(df["sec"]) == [10, 11, 12] and df["close"].isna().all() and list(df["n_trades"]) == [0, 0, 0]
    assert df["n_trades"].dtype == "int64" and binance.parse_klines([]).empty


class _Response:
    def __init__(self, status, text="{}"):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code), response=self)


@pytest.mark.case
def test_http_retries_dropped_connections_and_server_errors_then_raises_the_last_error(monkeypatch):
    """data_eval review, http 1: a dropped connection failed a whole paginated fetch at once; retries=0 crashed."""
    import requests
    from pmlab import http
    sleeps = []
    monkeypatch.setattr(http.time, "sleep", sleeps.append)
    answers = [requests.ConnectionError("reset"), requests.Timeout("slow"), _Response(503), _Response(200, '{"ok": 1}')]

    def get(url, params=None, timeout=None):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(http._session, "get", get)
    assert http.get_json("https://example.invalid/x") == {"ok": 1} and answers == []
    assert [s for s in sleeps if s >= 1] == [1, 2, 4]                                   # backoff between attempts
    sleeps.clear()
    answers[:] = [requests.ConnectionError("reset")] * 3
    with pytest.raises(requests.ConnectionError):
        http.get_json("https://example.invalid/x", retries=3)
    assert answers == [] and [s for s in sleeps if s >= 1] == [1, 2]                   # no pause after the last attempt
    answers[:] = [_Response(503)] * 2
    with pytest.raises(requests.HTTPError, match="503"):
        http.get_json("https://example.invalid/x", retries=2)
    answers[:] = [_Response(404), _Response(200)]
    with pytest.raises(requests.HTTPError, match="404"):
        http.get_json("https://example.invalid/x")                                      # not retried
    assert len(answers) == 1
    with pytest.raises(ValueError):
        http.get_json("https://example.invalid/x", retries=0)


# A window that closed and resolved Up before these tests were written (2026-09-16).
KNOWN_START = 1789570800


@pytest.mark.integration
def test_real_window_tape_is_complete_and_sane():
    m = polymarket.fetch_market(KNOWN_START)
    assert m["resolved"] and m["up_won"] is True and m["twap_lookback"] == 60
    tape = polymarket.normalise_trades(polymarket.fetch_raw_trades(m["condition_id"]), m)
    assert polymarket.coverage_ok(tape)
    assert len(polymarket.in_window(tape)) > 500
    assert tape["p_up"].between(0, 1, inclusive="neither").all()
    assert set(tape["sign"].unique()) == {-1, 1}
    assert tape["ts"].is_monotonic_increasing
    assert tape["tx"].is_unique


@pytest.mark.integration
def test_real_underlying_is_a_full_second_grid():
    df = binance.fetch_seconds(KNOWN_START - 180, KNOWN_START + 360)
    assert len(df) == 540 and df["sec"].diff().dropna().eq(1).all()
    assert (df["close"] > 1000).all()


@pytest.mark.integration
def test_closes_extend_back_over_contiguous_cached_windows_only():
    from pmlab import dataset
    starts = dataset.starts()
    s = next(x for x in starts[10:] if all((x - 300 * k) in set(starts) for k in range(1, 6)))
    ext = dataset.closes([s], history=1200)[s]
    short = __import__("pmlab").store.cached_underlying(s).set_index("sec")["close"]
    assert ext.index.min() <= s - 1200 and ext.index.max() == short.index.max()
    assert ext.index.is_monotonic_increasing and ext.index.is_unique
    assert (ext.loc[short.index] == short).all()                       # the window's own seconds are untouched
    assert ext.index.to_series().diff().dropna().eq(1).all()            # contiguous: nothing invented



@pytest.mark.case
def test_trade_pages_bust_the_api_cache_and_recent_tapes_are_never_cached(monkeypatch, tmp_path):
    from pmlab import store
    calls = []
    def fake(url, params):
        calls.append(params)
        return [{"i": k} for k in range(params["limit"])] if params["offset"] == 0 else [{"i": -1}]
    monkeypatch.setattr(polymarket, "get_json", fake)
    monkeypatch.setattr(polymarket.time, "time", lambda: 1_800_000_123.0)
    rows = polymarket.fetch_raw_trades("c")
    assert calls[0]["limit"] == 1000 - 1_800_000_123 % 600 and calls[1]["offset"] == calls[0]["limit"]
    assert len(rows) == calls[0]["limit"] + 1
    start = 1_800_000_000
    m = {"start": start, "condition_id": "c", "up_token": "u", "down_token": "d"}
    tape = pd.DataFrame({"t": [-5, 10, 290]})
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store.polymarket, "fetch_raw_trades", lambda c: [])
    monkeypatch.setattr(store.polymarket, "normalise_trades", lambda raw, m: tape)
    monkeypatch.setattr(store, "put_tape", lambda s, df: (tmp_path / "tape").mkdir() or (tmp_path / "tape" / f"{s}.parquet").touch())
    monkeypatch.setattr(store.time, "time", lambda: start + 300 + 899)
    store.tape(m)
    assert not (tmp_path / "tape" / f"{start}.parquet").exists()
    monkeypatch.setattr(store.time, "time", lambda: start + 300 + 900)
    store.tape(m)
    assert (tmp_path / "tape" / f"{start}.parquet").exists()



@pytest.mark.case
def test_klines_are_not_cached_before_their_range_has_ended(monkeypatch, tmp_path):
    from pmlab import store
    start = 1_800_000_000
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store.binance, "fetch_seconds", lambda a, b: klines_frame(a, b))
    monkeypatch.setattr(store.time, "time", lambda: start + 300 + 60 + 4)
    store.underlying(start)
    assert not store.has("binance_1s", start)
    monkeypatch.setattr(store.time, "time", lambda: start + 300 + 60 + 5)
    store.underlying(start)
    assert store.has("binance_1s", start)



@pytest.mark.case
def test_research_windows_exclude_the_confirmation_period_unless_asked(monkeypatch, tmp_path):
    from pmlab import dataset, evaluation, store
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    before, after = evaluation.CONFIRM_START - 300, evaluation.CONFIRM_START
    for s in (before - 300, before, after):
        for d in ("tape", "binance"):
            (tmp_path / d).mkdir(exist_ok=True)
            (tmp_path / d / f"{s}.parquet").touch()
        (tmp_path / "markets").mkdir(exist_ok=True)
        pd.DataFrame([{"start": s, "resolved": True, "twap_lookback": 60}]).to_parquet(tmp_path / "markets" / f"{s}.parquet")
    assert dataset.starts() == [before - 300, before]
    assert dataset.starts(confirmation=True) == [before - 300, before, after]


# --- the day-partitioned history (pmlab.store) -----------------------------------------------

S0 = 1_786_000_200    # 2026-08-06 05:10 UTC


def klines_frame(a, b):
    """A function of the second alone, so overlapping windows agree as real klines do."""
    sec = np.arange(a, b, dtype=np.int64)
    close = 60000 + (sec * 7919 % 100003) / 100
    return pd.DataFrame({"sec": sec, "open": close - 0.5, "high": close + 1.25, "low": close - 1.5, "close": close,
                         "volume": (sec % 977) / 1e5, "n_trades": sec % 401})


def market_dict(start, lookback=60, source="config"):
    return {"start": start, "slug": polymarket.slug(start), "condition_id": f"0x{start:064x}", "up_token": f"{start}1",
            "down_token": f"{start}2", "resolved": True, "up_won": start % 600 == 0, "twap_lookback": lookback,
            "rule_source": source, "fee_schedule": json.dumps({"rate": 0.07}), "description": "d", "resolutionSource": "r"}


def tape_frame(m, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = [raw(int(m["start"] - 50 + 7 * k), [m["up_token"], m["down_token"]][k % 2], ["BUY", "SELL"][(k // 2) % 2],
               float(np.round(rng.uniform(0.01, 0.99), 7)), float(np.round(rng.uniform(1, 500), 6)), f"0x{rng.bytes(32).hex()}", cid=m["condition_id"])
           for k in range(n)][::-1]
    return polymarket.normalise_trades(rows, m)


def write_legacy(root, start, lookback=60):
    m = market_dict(start, lookback)
    legacy = {k: v for k, v in m.items() if k in ("start", "slug", "condition_id", "up_token", "down_token", "resolved",
                                                   "up_won", "twap_lookback", "fee_schedule")}
    for d, df in (("markets", pd.DataFrame([legacy])), ("tape", tape_frame(m, seed=start % 97)),
                  ("binance", klines_frame(start - 180, start + 360))):
        (root / d).mkdir(exist_ok=True)
        df.to_parquet(root / d / f"{start}.parquet", index=False)
    return m


@pytest.mark.integration
def test_history_migration_is_exact_and_legacy_files_are_read_until_verified_and_deleted(monkeypatch, tmp_path):
    from pmlab import dataset, store
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "_cache", {})
    monkeypatch.setattr(store.polymarket, "fetch_market", lambda s: pytest.fail("cached windows are never fetched"))
    starts = [S0 + 300 * k for k in range(3)] + [store.day_start("2026-08-07") - 300]   # the last one crosses midnight
    legacy = {s: write_legacy(tmp_path, s) for s in starts}
    before = {s: (store.market(s), store.tape(legacy[s]), store.underlying(s)) for s in starts}
    assert before[starts[0]][0]["rule_source"] == "config" and before[starts[0]][0]["description"] is None

    report = store.migrate(log=lambda *a: None)
    assert report == {"markets": 4, "tape": 4, "binance_1s": 4, "failed": []}
    for s in starts:
        assert store.has("tape", s) and len(store._window_rows("tape", s)) > 0
        assert store.market(s) == before[s][0]
        pd.testing.assert_frame_equal(store.tape(legacy[s]), before[s][1])
        pd.testing.assert_frame_equal(store.underlying(s), before[s][2])
    assert sorted(p.parent.name for p in (tmp_path / "history" / "tape").glob("*/data.parquet")) == \
        ["day=2026-08-06"]
    assert not list((tmp_path / "history").glob("*/*/p-*.parquet"))       # every piece consolidated
    assert store.migrate(log=lambda *a: None)["failed"] == []            # idempotent
    changed = pd.read_parquet(tmp_path / "binance" / f"{starts[1]}.parquet")
    changed.loc[5, "close"] += 0.01
    changed.to_parquet(tmp_path / "binance" / f"{starts[2]}.parquet.bak", index=False)
    (tmp_path / "binance" / f"{starts[1]}.parquet").rename(tmp_path / "binance" / "orig.bak")
    changed.to_parquet(tmp_path / "binance" / f"{starts[1]}.parquet", index=False)
    market_file = tmp_path / "markets" / f"{starts[2]}.parquet"
    original_market = market_file.read_bytes()
    pd.read_parquet(market_file).assign(up_won=lambda d: ~d["up_won"]).to_parquet(market_file, index=False)
    assert store.migrate(verify_only=True, log=lambda *a: None)["failed"] == [starts[1], starts[2]]  # a cent, an outcome
    assert store.delete_legacy(log=lambda *a: None) == 10                                   # are never deleted
    (tmp_path / "binance" / "orig.bak").rename(tmp_path / "binance" / f"{starts[1]}.parquet")
    market_file.write_bytes(original_market)

    assert store.delete_legacy(log=lambda *a: None) == 2
    assert not list(tmp_path.glob("markets/*.parquet")) and not list(tmp_path.glob("tape/*.parquet"))
    for s in starts:
        store._cache.clear()
        pd.testing.assert_frame_equal(store.tape(legacy[s]), before[s][1])
        pd.testing.assert_frame_equal(store.underlying(s), before[s][2])
    assert dataset.starts(twap_lookback=60, confirmation=True) == sorted(starts)


@pytest.mark.case
def test_stored_tapes_rebuild_every_derived_column_and_refuse_what_they_cannot_keep():
    from pmlab import store
    m = market_dict(S0)
    tape = tape_frame(m, n=64, seed=3)
    stored = store.encode("tape", tape, S0)
    assert stored.column_names == ["start", "ts", "outcome", "side", "price", "size", "tx"]
    pd.testing.assert_frame_equal(store.decode_tape(stored.to_pandas(), S0), store._tape_frame(tape))
    assert set(store.decode_tape(stored.to_pandas(), S0)["sign"]) == {-1, 1}
    bad = tape.copy()
    bad.loc[3, "tx"] = bad.loc[3, "tx"].upper().replace("0X", "0x")
    with pytest.raises(ValueError):
        store.encode("tape", bad, S0)
    k = klines_frame(S0, S0 + 100)
    pd.testing.assert_frame_equal(store.decode_klines(store.encode("binance_1s", k).to_pandas()), k)
    with pytest.raises(ValueError):
        store.encode("markets", pd.DataFrame([{**m, "slug": "btc-updown-5m-1"}]))


@pytest.mark.integration
def test_pieces_are_read_latest_first_and_consolidate_into_one_file(monkeypatch, tmp_path):
    from pmlab import store
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "_cache", {})
    m = market_dict(S0)
    first, second = tape_frame(m, n=30, seed=1), tape_frame(m, n=35, seed=2)
    store.put_tape(S0, first)
    other = market_dict(S0 + 300)
    store.put_tape(S0 + 300, tape_frame(other, n=20, seed=9))
    store.put_tape(S0, second)                                    # a later fetch of the same window wins
    pd.testing.assert_frame_equal(store.cached_tape(S0), store._tape_frame(second))
    assert store.tape_starts() == {S0, S0 + 300}
    day = store.day_name(S0)
    assert store.consolidate("tape", day) == 55
    assert [p.name for p in store._day_files("tape", day)] == ["data.parquet"]
    store._cache.clear()
    pd.testing.assert_frame_equal(store.cached_tape(S0), store._tape_frame(second))
    assert len(store.cached_tape(S0 + 300)) == 20
    # klines: seconds across midnight land in both days; a window needs every second of its range
    lo = store.day_start("2026-08-07") - 400
    store.put_klines(klines_frame(lo, lo + 900))
    assert sorted(store.days("binance_1s")) == ["2026-08-06", "2026-08-07"]
    assert store.klines(lo, lo + 900) is not None and store.klines(lo, lo + 901) is None
    assert store.underlying_starts([lo + 180, lo + 540, lo + 600]) == {lo + 180, lo + 540}
    assert store.consolidate_all(now=store.day_start("2026-08-08") + 3600) == {
        "binance_1s 2026-08-06": 400, "binance_1s 2026-08-07": 500}


@pytest.mark.case
@pytest.mark.parametrize("name,expected", [("twap60", (60, "config")), ("twap30", (30, "config")), ("point", (1, "description"))])
def test_settlement_rule_is_normalised_from_config_or_text(name, expected):
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent / "fixtures" / "rules" / f"{name}.json").read_text())
    assert polymarket.settlement_rule(fixture) == expected
    text_only = {k: v for k, v in fixture.items() if k != "cryptoMarketConfig"}
    assert polymarket.settlement_rule(text_only) == (expected[0], "description")
    assert polymarket.settlement_rule({"description": "resolves on the TWAP"}) == (None, None)


@pytest.mark.case
def test_markets_come_100_slugs_per_request_and_trades_stop_at_the_offset_cap(monkeypatch):
    calls = []

    def gamma(url, params):
        slugs = [v for k, v in params if k == "slug"]
        calls.append(len(slugs))
        return [{"slug": sl, "markets": [{"outcomes": '["Up", "Down"]', "clobTokenIds": '["1", "2"]',
                                          "outcomePrices": '["1", "0"]', "umaResolutionStatus": "resolved",
                                          "conditionId": "c", "cryptoMarketConfig": {"twapLookbackSeconds": 60}}]}
                for sl in slugs] + [{"slug": polymarket.slug(1), "markets": []}]
    monkeypatch.setattr(polymarket, "get_json", gamma)
    starts = [S0 + 300 * k for k in range(250)]
    got = polymarket.fetch_markets(starts)
    assert calls == [100, 100, 50] and sorted(got) == starts and got[S0]["twap_lookback"] == 60

    offsets = []
    monkeypatch.setattr(polymarket, "get_json", lambda url, p: offsets.append(p["offset"]) or [{}] * p["limit"])
    rows = polymarket.fetch_raw_trades("c", limit=1000)
    assert offsets == list(range(0, 10001, 1000)) and len(rows) == 11000


@pytest.mark.case
def test_all_rules_are_selectable_without_changing_the_default(monkeypatch, tmp_path):
    from pmlab import dataset, store
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "_cache", {})
    rules = {S0: 1, S0 + 300: 30, S0 + 600: 60, S0 + 900: None}
    for s, lb in rules.items():
        m = market_dict(s, lb)
        store.put_markets([m])
        store.put_tape(s, tape_frame(m))
    store.put_klines(klines_frame(S0 - 180, S0 + 900 + 360))
    assert dataset.starts() == [S0 + 600]
    assert dataset.starts(twap_lookback=30) == [S0 + 300] and dataset.starts(twap_lookback=1) == [S0]
    assert dataset.starts(twap_lookback=None) == sorted(rules)
