import copy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factor_research.artifacts import json_bytes
from factor_research.contracts import read_config, read_panel
from factor_research.evaluator import evaluate
from factor_research.prepare import convert_prices, load_prepared, parse_clock, prepare_prices, validate_prepared
from factor_research.runs import run_trial, verify_run

EXAMPLE = Path(__file__).resolve().parents[1]/"examples/price-import"


def numerical(files):
    c=read_config(files["config.json"])
    return evaluate(read_panel(files["panel.csv"],c),c)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.raw=(EXAMPLE/"prices.csv").read_bytes()
        self.contract=json.loads((EXAMPLE/"contract.json").read_bytes())

    def convert(self, raw=None, contract=None):
        return convert_prices(self.raw if raw is None else raw,json_bytes(self.contract if contract is None else contract))

    def test_hand_written_panel_config_and_numerical_equivalence(self):
        imported=self.convert()
        self.assertEqual(imported['panel.csv'],(EXAMPLE/'manual-panel.csv').read_bytes())
        self.assertEqual(json.loads(imported['config.json']),json.loads((EXAMPLE/'manual-config.json').read_bytes()))
        manual={name:(EXAMPLE/('manual-'+name)).read_bytes() for name in ['panel.csv','config.json']}
        self.assertEqual(numerical(imported),numerical(manual))

    def test_missing_rows_blank_prices_and_assumption_visible(self):
        manifest=validate_prepared(self.convert())
        self.assertEqual(manifest['rows'],35)
        self.assertEqual(manifest['missing_rows'],1)
        self.assertEqual(manifest['null_prices'],1)
        self.assertEqual(manifest['data_source']['availability_status'],'ASSUMED_DELAY')
        self.assertFalse(manifest['data_source']['historically_point_in_time_verified'])

    def test_provided_late_clocks_retained(self):
        files=convert_prices((EXAMPLE/'prices-with-clocks.csv').read_bytes(),(EXAMPLE/'contract-with-clocks.json').read_bytes())
        manifest=validate_prepared(files)
        self.assertEqual(manifest['data_source']['availability_status'],'PROVIDED_CLOCKS_UNAUDITED')
        report=numerical(files)
        self.assertEqual(report['delayed_rows'],1)
        self.assertIsNone(report['feature_rows']['momentum'][6]['values']['A'])
        self.assertIsNotNone(numerical(self.convert())['feature_rows']['momentum'][6]['values']['A'])

    def test_input_order_invariance_retains_different_raw_hash(self):
        lines=self.raw.splitlines();shuffled=b'\n'.join([lines[0]]+list(reversed(lines[1:])))+b'\n'
        first,second=self.convert(),self.convert(shuffled)
        self.assertEqual(first['panel.csv'],second['panel.csv'])
        self.assertEqual(first['config.json'],second['config.json'])
        self.assertNotEqual(first['import-manifest.json'],second['import-manifest.json'])

    def test_future_test_prices_cannot_change_prior_features_or_selection(self):
        rows=list(csv.DictReader(io.StringIO(self.raw.decode())))
        for row in rows:
            if row['Date']>='2024-01-09':row['Close']='999' if row['Ticker']=='A' else '1'
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['Date','Ticker','Close']);writer.writeheader();writer.writerows(rows)
        before,after=numerical(self.convert()),numerical(self.convert(stream.getvalue().encode()))
        self.assertEqual(before['development'],after['development'])
        self.assertEqual(before['selection'],after['selection'])
        for f in ['momentum','reversal']:self.assertEqual(before['feature_rows'][f][:9],after['feature_rows'][f][:9])
        self.assertNotEqual(before['test'],after['test'])

    def test_test_only_asset_refused_not_added_to_universe(self):
        with self.assertRaisesRegex(ValueError,'test-only assets'):
            self.convert(self.raw+b'2024-01-12,D,100\n')

    def test_universe_cutoff_cannot_use_validation(self):
        self.contract['universe']['cutoff']='2024-01-05T16:00:00Z'
        with self.assertRaisesRegex(ValueError,'training'):self.convert()

    def test_explicit_universe_and_observed_union_are_deterministic(self):
        self.contract['universe']={'policy':'explicit','assets':['C','A','B']}
        self.contract['calendar'].pop('step_seconds');self.contract['calendar']['policy']='observed_union'
        config=json.loads(self.convert()['config.json'])
        self.assertEqual(config['assets'],['A','B','C'])
        self.assertEqual(len(config['calendar']),12)

    def test_fixed_calendar_preserves_wholly_missing_grid_date(self):
        raw=b'\n'.join(line for line in self.raw.splitlines() if not line.startswith(b'2024-01-03,'))+b'\n'
        result=self.convert(raw)
        self.assertEqual(len(json.loads(result['config.json'])['calendar']),12)
        self.assertEqual(json.loads(result['import-manifest.json'])['missing_rows'],4)

    def test_duplicate_conflict_and_unknown_columns_refused(self):
        for extra in [b'2024-01-01,A,100\n',b'2024-01-01,A,999\n']:
            with self.assertRaisesRegex(ValueError,'duplicate'):self.convert(self.raw+extra)
        with self.assertRaisesRegex(ValueError,'header'):self.convert(self.raw.replace(b'Date,Ticker,Close',b'Date,Ticker,Close,Ignored'))

    def test_no_implicit_clocks_or_delay(self):
        for availability in [{},{'mode':'infer'},{'mode':'assumed_delay'}, {'mode':'assumed_delay','observed_delay_seconds':1,'available_delay_seconds':0}]:
            with self.subTest(availability=availability):
                self.contract['availability']=availability
                with self.assertRaises(ValueError):self.convert()

    def test_absolute_split_overlap_and_missing_endpoint_refused(self):
        for start in ['2024-01-04T16:00:00Z','2024-01-05T15:00:00Z']:
            self.contract['experiment']['splits']['validation']['start']=start
            with self.assertRaises(ValueError):self.convert()

    def test_utc_offset_and_naive_time_rules(self):
        expected=datetime(2024,1,1,16,tzinfo=timezone.utc)
        self.assertEqual(parse_clock('2024-01-02T00:00:00+08:00',{'format':'iso8601','timezone':'+08:00'}),expected)
        self.assertEqual(parse_clock('2024-01-02T00:00:00',{'format':'iso8601','timezone':'+08:00'}),expected)
        with self.assertRaisesRegex(ValueError,'conflicts'):
            parse_clock('2024-01-01T16:00:00Z',{'format':'iso8601','timezone':'+08:00'})
        with self.assertRaisesRegex(ValueError,'naive'):
            parse_clock('2024-01-01T16:00:00',{'format':'iso8601','timezone':'offset_in_value'})
        with self.assertRaises(ValueError):parse_clock('2024-01-01',{'format':'iso8601','timezone':'UTC'})

    def test_wrong_timezone_hits_declared_date_grid(self):
        self.contract['timestamps']['time']['timezone']='+08:00'
        with self.assertRaisesRegex(ValueError,'calendar/grid'):self.convert()

    def test_epoch_units_explicit_and_wrong_unit_rejected(self):
        original=list(csv.DictReader(io.StringIO(self.raw.decode())))
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['Date','Ticker','Close']);writer.writeheader()
        for r in original:
            t=datetime.fromisoformat(r['Date']+'T16:00:00+00:00');r['Date']=str(int(t.timestamp()*1000));writer.writerow(r)
        raw=stream.getvalue().encode()
        self.contract['timestamps']['time']={'format':'unix','unit':'ms','timezone':'UTC'}
        self.assertEqual(self.convert(raw)['panel.csv'],(EXAMPLE/'manual-panel.csv').read_bytes())
        for unit in ['s','us','guess']:
            self.contract['timestamps']['time']['unit']=unit
            with self.assertRaises(ValueError):self.convert(raw)

    def test_mapping_conflicts_invalid_price_and_unknown_policy(self):
        broken=copy.deepcopy(self.contract);broken['columns']['price']='Date'
        with self.assertRaises(ValueError):self.convert(contract=broken)
        for price in [b'nan',b'0',b'-1']:
            with self.assertRaises(ValueError):self.convert(self.raw.replace(b'A,100',b'A,'+price,1))
        self.contract['universe']={'policy':'all_data'}
        with self.assertRaisesRegex(ValueError,'full-sample'):self.convert()


class PreparedTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.bundle=self.root/'prepared';self.store=self.root/'runs'
        prepare_prices(EXAMPLE/'prices.csv',EXAMPLE/'contract.json',self.bundle)

    def trial(self):return run_trial(None,None,self.store,'imported',prepared=self.bundle)

    def test_prepare_repeat_and_conflict_are_byte_preserving(self):
        before={p.name:p.read_bytes() for p in self.bundle.iterdir()}
        self.assertTrue(prepare_prices(EXAMPLE/'prices.csv',EXAMPLE/'contract.json',self.bundle)['reused'])
        changed=self.root/'changed.csv';changed.write_bytes((EXAMPLE/'prices.csv').read_bytes().replace(b'A,100',b'A,101',1))
        with self.assertRaisesRegex(ValueError,'conflicting'):prepare_prices(changed,EXAMPLE/'contract.json',self.bundle)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.bundle.iterdir()})

    def test_assumption_and_raw_mapping_propagate_into_v2_trial(self):
        self.assertEqual(self.trial()['status'],'SUCCESS')
        run=self.store/'imported'
        self.assertEqual((run/'raw-prices.csv').read_bytes(),(EXAMPLE/'prices.csv').read_bytes())
        self.assertEqual((run/'import-contract.json').read_bytes(),(EXAMPLE/'contract.json').read_bytes())
        self.assertTrue(verify_run(run,True)['recomputed'])
        report=json.loads((run/'report.json').read_bytes())
        self.assertEqual(report['schema_version'],'factor-report/v2')
        self.assertEqual(report['data_source']['availability_status'],'ASSUMED_DELAY')
        self.assertFalse(report['data_source']['historically_point_in_time_verified'])
        self.assertIn('Historical point-in-time verification: NO',(run/'report.md').read_text())
        self.assertTrue(self.trial()['reused'])

    def test_verified_run_uses_its_snapshots_not_mutated_prepare(self):
        self.trial()
        (self.bundle/'raw-prices.csv').write_text('damaged')
        self.assertTrue(verify_run(self.store/'imported',True)['recomputed'])
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.trial()

    def test_tampered_import_metadata_or_panel_refused(self):
        for name in ['import-manifest.json','panel.csv']:
            files=load_prepared(self.bundle);files[name]+=b' '
            with self.assertRaises(ValueError):validate_prepared(files)

    def test_raw_import_tampering_in_trial_refused(self):
        self.trial();(self.store/'imported/raw-prices.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_run(self.store/'imported')

    def test_incomplete_prepare_refuses_overwrite(self):
        other=self.root/'interrupted'
        with patch('factor_research.prepare.write_once',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):prepare_prices(EXAMPLE/'prices.csv',EXAMPLE/'contract.json',other)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            prepare_prices(EXAMPLE/'prices.csv',EXAMPLE/'contract.json',other)

    def test_run_rejects_ambiguous_input_modes(self):
        with self.assertRaisesRegex(ValueError,'prepared alone'):
            run_trial(EXAMPLE/'manual-panel.csv',EXAMPLE/'manual-config.json',self.store,'ambiguous',prepared=self.bundle)

    def test_legacy_route_cannot_accidentally_drop_prepared_assumptions(self):
        with self.assertRaisesRegex(ValueError,'retain raw source'):
            run_trial(self.bundle/'panel.csv',self.bundle/'config.json',self.store,'dropped')


if __name__=='__main__':unittest.main()
