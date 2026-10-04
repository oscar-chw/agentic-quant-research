"""Native persisted-state invariants; exhaustive/fault receipts are external."""
from contextlib import ExitStack
import copy
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import selectors
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from imc4_analysis import persistence as p, persist_journal as j, quote_study as qs, cli


def raw():return files('imc4_analysis').joinpath('data/quoting_scenarios.json').read_bytes()
def saved(root):
    return {str(f.relative_to(root)):(hashlib.sha256(f.read_bytes()).hexdigest(),f.stat().st_mtime_ns) for f in root.rglob('*') if f.is_file()}


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Path(self.tmp.name).resolve()/'store'
        self.addCleanup(self.tmp.cleanup)

    def test_genesis_suffix_completion_and_readonly_verification(self):
        result=p.run(raw(),self.store,'a',stop_after_commits=0)
        self.assertEqual((result['status'],result['token']['generation']),('PAUSED_AT_BOUNDARY',0))
        root=self.store/'a';before=saved(root)
        self.assertEqual(p.checkpoint(self.store,'a')['token'],result['token'])
        self.assertEqual(p.verify(self.store,'a')['status'],'VERIFIED');self.assertEqual(saved(root),before)
        result=p.resume(self.store,'a',result['token'])
        self.assertEqual((result['status'],result['token']['generation']),('COMPLETE',126))
        before=saved(root)
        self.assertEqual(p.verify(self.store,'a',mode='recompute')['status'],'VERIFIED');self.assertEqual(saved(root),before)
        with self.assertRaises(p.ResumeError) as cm:p.resume(self.store,'a',result['token'])
        self.assertEqual(cm.exception.code,'ALREADY_COMPLETED');self.assertEqual(saved(root),before)

    def test_final_budget_and_publication_never_call_policy_or_transition(self):
        result=p.run(raw(),self.store,'a',stop_after_commits=126)
        self.assertEqual(result['status'],'COMPUTE_COMPLETE');self.assertFalse((self.store/'a/product').exists())
        with mock.patch.object(qs,'advance_step',side_effect=AssertionError('transition during publication')),mock.patch.object(qs,'quote_orders',side_effect=AssertionError('policy during publication')):
            result=p.resume(self.store,'a',result['token'])
        self.assertEqual(result['status'],'COMPLETE')
        baseline=self.store/'baseline';cli.write_study(raw(),baseline)
        expected={str(f.relative_to(baseline)):f.read_bytes() for f in baseline.rglob('*') if f.is_file()}
        output=self.store/'a/product'
        self.assertEqual({str(f.relative_to(output)):f.read_bytes() for f in output.rglob('*') if f.is_file()},expected)

    def test_stale_token_context_and_input_binding_refuse_without_advancing(self):
        result=p.run(raw(),self.store,'a',context_sha256='1'*64,stop_after_commits=4)
        current=p.resume(self.store,'a',result['token'],context_sha256='1'*64,stop_after_commits=1)
        root=self.store/'a';before=saved(root)
        with self.assertRaises(p.ResumeError) as cm:p.resume(self.store,'a',result['token'],context_sha256='1'*64)
        self.assertEqual(cm.exception.code,'STALE_CHECKPOINT');self.assertEqual(saved(root),before)
        with self.assertRaises(p.ResumeError) as cm:p.checkpoint(self.store,'a')
        self.assertEqual(cm.exception.code,'CONTEXT_MISMATCH');self.assertEqual(saved(root),before)
        self.assertEqual(current['token']['generation'],5)

    def test_rehashed_plausible_counter_cash_live_corruption_fails_effect_projection(self):
        mutations=[lambda s:s['counters'].update(area=s['counters']['area']+1),
                   lambda s:s.update(cash='1100.6'),lambda s:s['live'][0].update(remaining_units=2)]
        for i,change in enumerate(mutations):
            with self.subTest(kind=i):
                name=f'a{i}';result=p.run(raw(),self.store,name,stop_after_commits=68);root=self.store/name
                with sqlite3.connect(root/'journal.sqlite',isolation_level=None) as c:
                    g,rid,tick,prev,digest,st,tr,eh=c.execute('SELECT * FROM steps ORDER BY g DESC LIMIT 1').fetchone()
                    state=j.decode(st,8192);change(state);st=j.blob(state,8192)
                    events=b''.join(r[0] for r in c.execute('SELECT data FROM events WHERE g=? ORDER BY seq',(g,)))
                    digest=j.step_digest(g,rid,tick,prev,st,tr,events)
                    c.execute('UPDATE steps SET state=?,digest=? WHERE g=?',(st,digest,g))
                    rev,og,kind,data,opprev,_=c.execute('SELECT * FROM ops ORDER BY rev DESC LIMIT 1').fetchone()
                    d=j.decode(data,16384);d['payload']=digest;data=j.blob(d,16384)
                    od=j.sha(j.blob(dict(rev=rev,g=og,kind=kind,data_sha256=j.sha(data),prev=opprev)))
                    c.execute('UPDATE ops SET data=?,digest=? WHERE rev=?',(data,od,rev));c.execute('UPDATE head SET digest=?',(od,))
                before=saved(root)
                with self.assertRaises(p.ResumeError) as cm:p.verify(self.store,name)
                self.assertEqual(cm.exception.code,'INVALID_STATE');self.assertIn('projection',str(cm.exception));self.assertEqual(saved(root),before)

    def test_foreign_output_and_completed_output_change_refuse(self):
        result=p.run(raw(),self.store,'a');root=self.store/'a';(root/'product/unexpected.txt').write_text('user content')
        before=saved(root)
        with self.assertRaises(p.ResumeError) as cm:p.resume(self.store,'a',result['token'])
        self.assertEqual(cm.exception.code,'INCONSISTENT_COMPLETE');self.assertEqual(saved(root),before)

    def test_competing_owner_blocks_read_and_write(self):
        result=p.run(raw(),self.store,'a',stop_after_commits=0);root=self.store/'a'
        code="import fcntl,sys; f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print('ready',flush=True); sys.stdin.read(1)"
        child=subprocess.Popen([sys.executable,'-c',code,str(root/'attempt.lock')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(child.stdout,selectors.EVENT_READ);self.assertTrue(sel.select(10));self.assertEqual(child.stdout.readline().strip(),'ready')
            before=saved(root)
            for invoke in [lambda:p.checkpoint(self.store,'a'),lambda:p.resume(self.store,'a',result['token'])]:
                with self.assertRaises(p.ResumeError) as cm:invoke()
                self.assertEqual(cm.exception.code,'BUSY')
            self.assertEqual(saved(root),before)
        finally:
            child.communicate('x',timeout=10)
        self.assertEqual(child.returncode,0)

    def test_resource_refusal_precedes_next_commit(self):
        result=p.run(raw(),self.store,'a',stop_after_commits=4);root=self.store/'a';before=saved(root)
        with mock.patch.object(j,'MAX_LOGICAL',result['metrics']['logical_bytes']):
            with self.assertRaises(p.ResumeError) as cm:p.resume(self.store,'a',result['token'],stop_after_commits=1)
        self.assertEqual(cm.exception.code,'RESOURCE_LIMIT');self.assertEqual(saved(root),before)
        self.assertEqual(p.checkpoint(self.store,'a')['token'],result['token'])

    def test_actual_constraint_failure_rolls_back_transaction(self):
        p.run(raw(),self.store,'a',stop_after_commits=0);root=self.store/'a';before=(root/'journal.sqlite').read_bytes()
        with j.locked(root):
            c=j.connection(root)
            try:
                self.assertIn('TEMP_STORE=1',[r[0] for r in c.execute('PRAGMA compile_options')])
                with self.assertRaises(sqlite3.IntegrityError):
                    with j.transaction(c,root):
                        c.execute('INSERT OR ROLLBACK INTO runs VALUES(?,?,?,?)',(98,'tentative',b'{}',b'{}'))
                        c.execute('INSERT OR ROLLBACK INTO runs VALUES(?,?,?,?)',(99,'tentative',b'{}',b'{}'))
                self.assertFalse(c.in_transaction);self.assertEqual(c.execute('SELECT count(*) FROM runs').fetchone()[0],18)
            finally:c.close()
        self.assertEqual((root/'journal.sqlite').read_bytes(),before)


if __name__=='__main__':unittest.main()
