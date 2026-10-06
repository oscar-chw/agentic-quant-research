"""Offline checks of the Binance fetch script: URLs, checksums and parsing, no network."""
import hashlib
import importlib.util
import io
import urllib.error
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_binance_daily", ROOT / "scripts/fetch_binance_daily.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)

MS_ROW = "1704067200000,100.00,101.50,99.25,100.75,12.5,1704153599999,0,0,0,0,0\n"
US_ROW = "1735689600000000,200.00,202.00,198.00,201.00,3.0,1735775999999999,0,0,0,0,0\n"


def archive(text, name="BTCUSDT-1d-2024-01.csv"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr(name, text)
    return buffer.getvalue()


def test_monthly_archive_and_checksum_urls():
    url, checksum = fetcher.archive_url("BTCUSDT", "2024-01")
    assert url == "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    assert checksum == url + ".CHECKSUM"
    with pytest.raises(ValueError):
        fetcher.archive_url("btc/usdt", "2024-01")


def test_month_range_is_inclusive_across_a_year_end():
    assert fetcher.months("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]
    for start, end in [("2024-02", "2024-01"), ("2024-13", "2024-13"), ("2024-1", "2024-02")]:
        with pytest.raises(ValueError):
            fetcher.months(start, end)


def test_checksum_must_match_bytes_and_name():
    data = archive(MS_ROW)
    good = hashlib.sha256(data).hexdigest() + "  BTCUSDT-1d-2024-01.zip\n"
    assert fetcher.verify_archive(data, good, "BTCUSDT-1d-2024-01.zip") == hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match="mismatch"):
        fetcher.verify_archive(data + b"x", good, "BTCUSDT-1d-2024-01.zip")
    with pytest.raises(ValueError, match="malformed"):
        fetcher.verify_archive(data, good, "ETHUSDT-1d-2024-01.zip")
    with pytest.raises(ValueError, match="malformed"):
        fetcher.verify_archive(data, "not-a-digest  BTCUSDT-1d-2024-01.zip", "BTCUSDT-1d-2024-01.zip")


def test_millisecond_and_microsecond_open_times_give_the_same_calendar_day_rule():
    assert fetcher.kline_rows(archive(MS_ROW)) == [["2024-01-01", "100.00", "101.50", "99.25", "100.75", "12.5"]]
    assert fetcher.kline_rows(archive(US_ROW))[0][0] == "2025-01-01"
    with pytest.raises(ValueError, match="00:00 UTC"):
        fetcher.bar_date("1704070800000")


def test_header_row_is_skipped_and_short_rows_refused():
    header = "open_time,open,high,low,close,volume,close_time,a,b,c,d,e\n"
    assert len(fetcher.kline_rows(archive(header + MS_ROW))) == 1
    with pytest.raises(ValueError):
        fetcher.kline_rows(archive("1704067200000,1,2\n"))


def test_fetch_writes_ohlcv_files_and_refuses_a_tampered_archive(tmp_path):
    data = archive(MS_ROW)
    served = {}
    url, checksum_url = fetcher.archive_url("BTCUSDT", "2024-01")
    served[url] = data
    served[checksum_url] = (hashlib.sha256(data).hexdigest() + "  BTCUSDT-1d-2024-01.zip\n").encode()
    manifest = fetcher.fetch(["BTCUSDT"], "2024-01", "2024-01", tmp_path / "out", get=served.__getitem__)
    assert (tmp_path / "out/BTCUSDT.csv").read_text() == (
        "date,open,high,low,close,volume\n2024-01-01,100.00,101.50,99.25,100.75,12.5\n")
    assert manifest["archives"] == [{"url": url, "sha256": hashlib.sha256(data).hexdigest()}]
    served[url] = archive(MS_ROW.replace("100.75", "100.76"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetcher.fetch(["BTCUSDT"], "2024-01", "2024-01", tmp_path / "tampered", get=served.__getitem__)
    assert not (tmp_path / "tampered/BTCUSDT.csv").exists()


def test_a_404_after_listed_months_ends_the_listing_but_a_first_month_404_still_raises(tmp_path):
    served, asked = {}, []
    data = archive(MS_ROW)
    url, checksum_url = fetcher.archive_url("BTCUSDT", "2024-01")
    served[url] = data
    served[checksum_url] = (hashlib.sha256(data).hexdigest() + "  BTCUSDT-1d-2024-01.zip\n").encode()

    def get(address):
        asked.append(address)
        if address not in served:
            raise urllib.error.HTTPError(address, 404, "Not Found", {}, None)
        return served[address]

    manifest = fetcher.fetch(["BTCUSDT"], "2024-01", "2024-03", tmp_path / "out", get=get)
    assert manifest["listing_ended"] == {"BTCUSDT": {"last_month": "2024-01", "first_missing_month": "2024-02"}}
    assert manifest["files"]["BTCUSDT.csv"]["bars"] == 1
    assert not any("2024-03" in a for a in asked)  # nothing is asked for after the listing ended
    with pytest.raises(urllib.error.HTTPError):
        fetcher.fetch(["BTCUSDT"], "2023-12", "2024-01", tmp_path / "first", get=get)

    def server_error(address):
        if address == url or address == checksum_url:
            return served[address]
        raise urllib.error.HTTPError(address, 500, "Server Error", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        fetcher.fetch(["BTCUSDT"], "2024-01", "2024-02", tmp_path / "error", get=server_error)
