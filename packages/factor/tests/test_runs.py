import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factor_research.cli import main
from factor_research.runs import list_trials, run_trial, verify_run
from factor_research.synthetic import make_fixture


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.fixture=self.root/'fixture'; self.store=self.root/'trials'
        make_fixture(self.fixture)

    def run_trial(self, name='one'):
        return run_trial(self.fixture/'panel.csv',self.fixture/'config.json',self.store,name)

    def bytes(self):
        return {str(p.relative_to(self.store)):p.read_bytes() for p in self.store.rglob('*') if p.is_file()}

    def test_success_recompute_and_negative_test_result(self):
        self.assertEqual(self.run_trial()['status'],'SUCCESS')
        self.assertTrue(verify_run(self.store/'one',True)['recomputed'])
        report=json.loads((self.store/'one/report.json').read_text())
        self.assertEqual(report['selection']['selected'],'momentum')
        self.assertLess(report['test']['summary']['rank_ic_mean'],0)
        self.assertLess(report['test']['summary']['net_mean'],0)
        self.assertIn('fixed unit', (self.store/'one/report.md').read_text())

    def test_same_repeat_is_byte_preserving(self):
        self.run_trial(); before=self.bytes()
        self.assertTrue(self.run_trial()['reused'])
        self.assertEqual(before,self.bytes())

    def test_conflicting_repeat_is_refused_and_preserved(self):
        self.run_trial(); before=self.bytes()
        p=self.fixture/'config.json'; data=json.loads(p.read_text()); data['cost_bps']=20
        p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'conflicting run ID'): self.run_trial()
        self.assertEqual(before,self.bytes())

    def test_failed_trial_preserved_and_listed(self):
        panel=self.fixture/'panel.csv'; lines=panel.read_text().splitlines()
        panel.write_text('\n'.join(lines+[lines[1]])+'\n')
        self.assertEqual(self.run_trial()['status'],'FAILED')
        self.assertIn('duplicate',json.loads((self.store/'one/failure.json').read_text())['message'])
        self.assertEqual(list_trials(self.store)[0]['status'],'FAILED')
        before=self.bytes(); self.assertTrue(self.run_trial()['reused']); self.assertEqual(before,self.bytes())

    def test_incomplete_trial_cannot_be_overwritten(self):
        with patch('factor_research.runs.report_for',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.run_trial()
        before=self.bytes()
        self.assertEqual(list_trials(self.store)[0]['status'],'INCOMPLETE')
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'): self.run_trial()
        self.assertEqual(before,self.bytes())

    def test_tampered_report_refused(self):
        self.run_trial(); p=self.store/'one/report.json'; p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): verify_run(self.store/'one')
        self.assertEqual(list_trials(self.store)[0]['status'],'CORRUPT')

    def test_tampered_input_refused(self):
        self.run_trial(); p=self.store/'one/panel.csv'; p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): verify_run(self.store/'one',True)

    def test_recompute_requires_exact_code(self):
        self.run_trial()
        with patch('factor_research.runs.code_identity',return_value={'code_sha256':'changed'}):
            with self.assertRaisesRegex(ValueError,'original installed code'): verify_run(self.store/'one',True)

    def test_invalid_config_records_failed_trial(self):
        (self.fixture/'config.json').write_text('{broken')
        self.assertEqual(self.run_trial()['status'],'FAILED')
        self.assertTrue((self.store/'one/completion.json').is_file())

    def test_input_fixture_refuses_overwrite(self):
        before=(self.fixture/'panel.csv').read_bytes()
        with self.assertRaises(FileExistsError): make_fixture(self.fixture)
        self.assertEqual((self.fixture/'panel.csv').read_bytes(),before)

    def test_unsafe_run_ids(self):
        for name in ['../escape','/absolute','a/b','', 'x'*65]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError,'run ID'): self.run_trial(name)

    def test_diagnostic_fixture_marks_incomplete_portfolio(self):
        folder=self.root/'diagnostic'; make_fixture(folder,diagnostics=True)
        run_trial(folder/'panel.csv',folder/'config.json',self.store,'diagnostic')
        report=json.loads((self.store/'diagnostic/report.json').read_text())
        self.assertEqual(report['delayed_rows'],1)
        self.assertEqual(report['null_prices'],1)
        self.assertFalse(report['test']['summary']['portfolio_complete'])


if __name__=='__main__': unittest.main()
