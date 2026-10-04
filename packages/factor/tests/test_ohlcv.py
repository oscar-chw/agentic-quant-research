from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from factor_research.contracts import decode_json
from factor_research.ohlcv import ohlcv_import, read_bars
from factor_research.prepare import convert_prices

HEADER = "date,open,high,low,close,volume\n"


def day(i):
    return (date(2024, 1, 1) + timedelta(days=i)).isoformat()


def split(a, b):
    return {"start": day(a) + "T23:59:59Z", "end": day(b) + "T23:59:59Z"}


EXPERIMENT = {"available_delay_seconds": 60, "lookback": 1, "cost_bps": 10, "candidates": ["momentum"],
              "splits": {"train": split(0, 3), "validation": split(4, 7), "test": split(8, 11)}}


class OhlcvTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def write(self, asset, first, closes):
        rows = [f"{day(first + i)},{c},{c + 1},{c - 1},{c},5\n" for i, c in enumerate(closes)]
        (self.dir / f"{asset}.csv").write_text(HEADER + "".join(rows))

    def test_imported_prices_pass_prepare_with_declared_close_and_delay(self):
        self.write("AAA", 0, [10 + i for i in range(12)])
        self.write("BBB", 0, [20 - i for i in range(12)])
        result = ohlcv_import(self.dir, EXPERIMENT)
        files = convert_prices(result["prices"], result["contract"])
        panel = files["panel.csv"].decode().splitlines()
        self.assertEqual(panel[1], "2024-01-01T23:59:59Z,AAA,2024-01-01T23:59:59Z,2024-01-02T00:00:59Z,10,")
        self.assertEqual(len(panel), 1 + 24)
        self.assertEqual(json.loads(files["config.json"])["assets"], ["AAA", "BBB"])
        manifest = decode_json(files["import-manifest.json"])
        self.assertEqual(manifest["data_source"]["availability_status"], "ASSUMED_DELAY")

    def test_asset_listed_after_training_is_excluded_not_backfilled(self):
        self.write("AAA", 0, [10 + i for i in range(12)])
        self.write("BBB", 0, [20 - i for i in range(12)])
        self.write("NEW", 5, [5 + i for i in range(7)])
        result = ohlcv_import(self.dir, EXPERIMENT)
        self.assertEqual(result["excluded_assets"], ["NEW"])
        self.assertNotIn(b"NEW", result["prices"])
        convert_prices(result["prices"], result["contract"])

    def test_listing_on_the_cutoff_day_is_excluded_when_the_delay_crosses_it(self):
        self.write("AAA", 0, [10 + i for i in range(12)])
        self.write("BBB", 0, [20 - i for i in range(12)])
        self.write("EDGE", 3, [5 + i for i in range(9)])
        result = ohlcv_import(self.dir, EXPERIMENT)
        self.assertEqual(result["excluded_assets"], ["EDGE"])
        convert_prices(result["prices"], result["contract"])

    def test_bars_before_the_calendar_do_not_establish_membership(self):
        self.write("AAA", 0, [10 + i for i in range(12)])
        self.write("BBB", 0, [20 - i for i in range(12)])
        (self.dir / "OLD.csv").write_text(HEADER + "2023-12-30,5,6,4,5,1\n" + "".join(
            f"{day(i)},{c},{c + 1},{c - 1},{c},5\n" for i, c in [(6, 7), (7, 8)]))
        result = ohlcv_import(self.dir, EXPERIMENT)
        self.assertEqual(result["excluded_assets"], ["OLD"])
        convert_prices(result["prices"], result["contract"])

    def test_rows_outside_the_declared_calendar_are_dropped(self):
        self.write("AAA", 0, [10 + i for i in range(15)])
        self.write("BBB", 0, [20 - i for i in range(15)])
        self.assertEqual(ohlcv_import(self.dir, EXPERIMENT)["rows"], 24)

    def test_invalid_bars_are_refused(self):
        cases = {
            "header": "date,close\n2024-01-01,1\n",
            "high below close": HEADER + "2024-01-01,10,9,8,10,1\n",
            "zero close": HEADER + "2024-01-01,1,1,0,0,1\n",
            "duplicate": HEADER + "2024-01-01,1,1,1,1,1\n2024-01-01,1,1,1,1,1\n",
            "date format": HEADER + "2024-1-1,1,1,1,1,1\n",
            "nan": HEADER + "2024-01-01,nan,1,1,1,1\n",
        }
        for label, text in cases.items():
            with self.subTest(label):
                path = self.dir / "X.csv"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_bars(path)

    def test_experiment_fields_are_exact(self):
        self.write("AAA", 0, [10] * 12)
        with self.assertRaises(ValueError):
            ohlcv_import(self.dir, {**EXPERIMENT, "extra": 1})


if __name__ == "__main__":
    unittest.main()
