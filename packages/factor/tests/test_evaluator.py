import copy
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import io
import json
import unittest

from factor_research.contracts import read_config, read_panel
from factor_research.evaluator import average_ranks, build_features, evaluate, rank_ic, sleeve, target_weights


def inputs():
    dates = [(datetime(2024, 1, 1, tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(18)]
    config = dict(schema_version="factor-config/v1", panel_schema="factor-panel/v1",
                  assets=["A", "B", "C"], calendar=dates, lookback=1, horizon=1,
                  cost_bps=10, candidates=["momentum", "reversal"],
                  splits={k:dict(start=dates[a], end=dates[b]) for k, a, b in
                          [("train",0,5),("validation",6,11),("test",12,17)]})
    rows = [dict(time=t, asset=a, observed_at=t, available_at=t,
                 price=str(100*(1+r)**i), group="g")
            for i,t in enumerate(dates) for a,r in zip(config["assets"],[-0.01,0,0.02])]
    return config, rows


def csv_bytes(rows):
    s=io.StringIO(newline="")
    writer=csv.DictWriter(s,fieldnames=["time","asset","observed_at","available_at","price","group"])
    writer.writeheader(); writer.writerows(rows)
    return s.getvalue().encode()


def load(data=None, rows=None):
    base, original = inputs()
    config = read_config(json.dumps(base if data is None else data).encode())
    return config, read_panel(csv_bytes(original if rows is None else rows), config)


class ArithmeticTests(unittest.TestCase):
    def test_average_ties(self):
        self.assertEqual(average_ranks([3,1,1,2]), [4,1.5,1.5,3])

    def test_independent_rank_ic(self):
        # By hand: ranks centered at 2.5; dot=15/4, both norm squared=9/2.
        self.assertAlmostEqual(rank_ic([1,1,2,3],[1,2,2,3]),float(Fraction(15,4)/Fraction(9,2)))
        self.assertAlmostEqual(rank_ic([1,2,3],[3,1,2]),-0.5)

    def test_null_constant_and_insufficient_ic(self):
        self.assertIsNone(rank_ic([1,1],[2,3]))
        self.assertIsNone(rank_ic([1,2],[3,3]))
        self.assertIsNone(rank_ic([None,2],[3,4]))
        self.assertAlmostEqual(rank_ic([None,1,2],[3,5,4]),-1)

    def test_hand_computed_positions_and_roundtrip_cost(self):
        weights=target_weights(dict(A=1,B=2,C=3),dict(A=100,B=100,C=100))
        self.assertEqual(weights,dict(A=-0.5,B=0,C=0.5))
        # qA=-.005, qC=.005; exit prices 110 and 90.
        # PnL=-.005*10 + .005*(-10)=-.1. Entry notional=1, exit=1.
        result=sleeve(weights,dict(A=.1,B=0,C=-.1),10)
        self.assertAlmostEqual(result["gross"],-.1)
        self.assertAlmostEqual(result["turnover"],2)
        self.assertAlmostEqual(result["cost"],.002)
        self.assertAlmostEqual(result["net"],-.102)

    def test_exit_cost_tracks_price_change(self):
        result=sleeve(dict(A=-.5,B=.5),dict(A=.2,B=.4),25)
        self.assertAlmostEqual(result["gross"],.1)
        self.assertAlmostEqual(result["turnover"],2.3)
        self.assertAlmostEqual(result["net"],.1-2.3*.0025)

    def test_future_missing_mark_cannot_reweight(self):
        weights=target_weights(dict(A=1,B=2,C=3),dict(A=100,B=100,C=100))
        result=sleeve(weights,dict(A=.1,B=.3,C=None),10)
        self.assertEqual(weights,dict(A=-.5,B=0,C=.5))
        self.assertEqual(result["status"],"MISSING_EXIT_MARK")
        self.assertIsNone(result["net"])

    def test_constant_signals_abstain(self):
        weights=target_weights(dict(A=1,B=1),dict(A=100,B=100))
        self.assertEqual(sleeve(weights,dict(A=None,B=None),10)["status"],"ABSTAIN")


class TimeAndContractTests(unittest.TestCase):
    def test_strict_past_feature_and_warmup(self):
        config,panel=load()
        features=build_features(panel,config,"momentum")
        self.assertTrue(all(v is None for v in features[0].values()))
        self.assertTrue(all(v is None for v in features[1].values()))
        self.assertAlmostEqual(features[2]["A"],-.01)
        self.assertAlmostEqual(features[2]["C"],.02)

    def test_late_availability_is_not_backfilled(self):
        config,rows=inputs()
        rows[3]["available_at"]=config["calendar"][3]  # A at index 1 unavailable at decision 2
        c,p=load(config,rows)
        self.assertIsNone(build_features(p,c,"momentum")[2]["A"])
        self.assertIsNotNone(build_features(p,c,"momentum")[3]["A"])

    def test_future_mutation_preserves_earlier_features_and_selection(self):
        data,rows=inputs()
        c,p=load(data,rows); before=evaluate(p,c)
        mutated=copy.deepcopy(rows)
        for row in mutated:
            if row["time"]>=data["calendar"][12]:
                row["price"]=str(10000 if row["asset"]=="A" else 1)
        c,p=load(data,mutated); after=evaluate(p,c)
        for factor in c.candidates:
            self.assertEqual(before["feature_rows"][factor][:13],after["feature_rows"][factor][:13])
        self.assertEqual(before["development"],after["development"])
        self.assertEqual(before["selection"],after["selection"])
        self.assertNotEqual(before["test"],after["test"])

    def test_labels_never_cross_splits(self):
        c,p=load(); result=evaluate(p,c)
        for factor in c.candidates:
            for name in ("train","validation"):
                part=result["development"][factor][name]
                self.assertEqual(len(part["daily"]),5)
                self.assertLessEqual(part["daily"][-1]["label_time"],c.calendar[c.splits[name][1]].isoformat().replace('+00:00','Z'))

    def test_duplicate_asset_time_rejected_even_with_offset_alias(self):
        data,rows=inputs(); duplicate=dict(rows[0]); duplicate["time"]="2024-01-01T08:00:00+08:00"
        with self.assertRaisesRegex(ValueError,"duplicate"):
            load(data,rows+[duplicate])

    def test_missing_rows_and_blank_prices_are_explicit(self):
        data,rows=inputs(); rows.pop(); rows[0]["price"]=""
        c,p=load(data,rows); report=evaluate(p,c)
        self.assertEqual(report["missing_panel_rows"],1)
        self.assertEqual(report["null_prices"],1)

    def test_missing_future_price_suppresses_aggregate(self):
        data,rows=inputs(); rows[3*14]["price"]=""
        c,p=load(data,rows); report=evaluate(p,c)
        self.assertFalse(report["test"]["summary"]["portfolio_complete"])
        self.assertIsNone(report["test"]["summary"]["net_mean"])

    def test_all_constant_validation_prevents_selection(self):
        data,rows=inputs()
        for row in rows: row["price"]="100"
        c,p=load(data,rows); report=evaluate(p,c)
        self.assertEqual(report["selection"]["status"],"NO_VALIDATION_IC")
        self.assertIsNone(report["test"])

    def test_invalid_horizons(self):
        for value in [0,-1,2,1.5,True,"1"]:
            with self.subTest(value=value):
                data,_=inputs(); data["horizon"]=value
                with self.assertRaisesRegex(ValueError,"horizon"): load(data)

    def test_invalid_split_overlap(self):
        data,_=inputs(); data["splits"]["validation"]["start"]=data["calendar"][5]
        with self.assertRaisesRegex(ValueError,"splits"): load(data)

    def test_invalid_clocks_prices_and_universe(self):
        for field,value in [("observed_at","2023-12-31T00:00:00Z"),("available_at","2024-01-01"),
                            ("price","nan"),("price","0"),("price","inf"),("asset","outside")]:
            with self.subTest(field=field,value=value):
                data,rows=inputs(); rows[0][field]=value
                with self.assertRaises(ValueError): load(data,rows)

    def test_duplicate_config_keys_and_unknown_fields(self):
        data,_=inputs(); raw=json.dumps(data)
        with self.assertRaisesRegex(ValueError,"duplicate JSON key"):
            read_config(('{"horizon":1,'+raw[1:]).encode())
        data["cost_bp"]=0
        with self.assertRaisesRegex(ValueError,"fields"): load(data)


if __name__=="__main__": unittest.main()
