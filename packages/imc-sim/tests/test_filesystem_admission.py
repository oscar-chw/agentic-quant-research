"""Root-reproduced ownership failures and nearby filesystem admission cases."""
import hashlib
from importlib.resources import files
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from imc4_analysis import persistence as p, persist_journal as j, quote_study as qs


def snapshot(root):
    paths=[root]+list(root.rglob('*')) if root.is_dir() else [root]
    return {str(f.relative_to(root)) if f!=root else '.':
            (f.lstat().st_mtime_ns,f.lstat().st_mode,f.lstat().st_ino,f.lstat().st_nlink,
             hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None)
            for f in paths}


class FilesystemAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.store=self.base/'store'
        self.raw=files('imc4_analysis').joinpath('data/quoting_scenarios.json').read_bytes()

    def create(self,name='a',cut=67):
        result=p.run(self.raw,self.store,name,stop_after_commits=cut)
        return self.store/name,result

    def refusals_unchanged(self,name,result,extra=None):
        root=self.store/name;before=snapshot(root);outside=snapshot(extra) if extra else None
        for fn in [lambda:p.resume(self.store,name,result['token'],stop_after_commits=1),
                   lambda:p.checkpoint(self.store,name),lambda:p.verify(self.store,name)]:
            with self.assertRaises(p.ResumeError) as cm:fn()
            self.assertIn(cm.exception.code,{'OWNERSHIP_AMBIGUOUS','UNEXPECTED_OUTPUT','CORRUPT','INCONSISTENT_COMPLETE'})
            self.assertEqual(snapshot(root),before)
            if extra:self.assertEqual(snapshot(extra),outside)

    def test_root_database_alias_refuses_without_changing_either_location(self):
        root,result=self.create();outside=self.base/'outside-database'
        os.link(root/'journal.sqlite',outside)
        self.assertEqual(outside.stat().st_nlink,2)
        self.refusals_unchanged('a',result,outside)

    def test_root_unknown_anchor_refuses_without_changing_attempt(self):
        root,result=self.create(cut=0)
        (root/'anchors/user-notes.txt').write_text('Preserve this unrelated content.\n')
        self.refusals_unchanged('a',result)

    def test_other_operational_aliases_refuse_before_open_or_capture(self):
        for i,filename in enumerate(['attempt.lock','identity.json','capability.json','geometry.sqlite','journal.sqlite-journal']):
            with self.subTest(filename=filename):
                name=f'a{i}';root,result=self.create(name,0)
                if filename.endswith('-journal'):(root/filename).write_bytes(b'pending-test-journal')
                outside=self.base/f'outside-{i}';os.link(root/filename,outside)
                self.refusals_unchanged(name,result,outside)
                self.assertEqual(list((root/'forensics').iterdir()),[])

    def test_scenario_third_alias_refuses_but_valid_pair_verifies(self):
        root,result=self.create(cut=126);result=p.resume(self.store,'a',result['token'])
        scenario=root/'product/scenario.json';anchor=root/'anchors/generation-0.scenario'
        self.assertEqual((scenario.stat().st_ino,scenario.stat().st_nlink),(anchor.stat().st_ino,2))
        before=snapshot(root);self.assertEqual(p.verify(self.store,'a')['status'],'VERIFIED');self.assertEqual(snapshot(root),before)
        outside=self.base/'outside-scenario';os.link(scenario,outside)
        self.refusals_unchanged('a',result,outside)

    def test_unknown_nested_names_and_wrong_types_refuse(self):
        mutations=[lambda r:(r/'anchors/generation-0.scenario').mkdir(),
                   lambda r:(r/'anchors/generation-0.scenario').write_bytes(b'unregistered'),
                   lambda r:(r/'forensics/generation-1').write_bytes(b'wrong type'),
                   lambda r:(r/'forensics/generation-3').mkdir(),
                   lambda r:((r/'forensics/generation-1').mkdir(),(r/'forensics/generation-1/user-notes.txt').write_text('preserve')),
                   lambda r:((r/'product').mkdir(),(r/'product/unregistered-run').mkdir())]
        for i,mutation in enumerate(mutations):
            with self.subTest(case=i):
                name=f'a{i}';root,result=self.create(name,0);mutation(root);self.refusals_unchanged(name,result)

    def test_forensic_receipt_shape_hashes_and_archive_binding_are_required(self):
        def malformed(root):
            d=root/'forensics/generation-1';d.mkdir();(d/'complete.json').write_bytes(b'{}\n')
        def bad_hash(root):
            d=root/'forensics/generation-1';d.mkdir();(d/'journal.sqlite').write_bytes(b'captured')
            (d/'complete.json').write_bytes(j.blob(dict(schema='imc4-forensic/v1',kind='database-journal',files={'journal.sqlite':dict(sha256='0'*64,bytes=8)})))
        def unbound_partial(root):
            d=root/'forensics/generation-1';(d/'partial').mkdir(parents=True);(d/'partial/scenario.json').write_bytes(self.raw)
            (d/'complete.json').write_bytes(j.blob(dict(schema='imc4-forensic/v1',kind='partial-publication',files=p._file_commitments(d))))
        for i,mutation in enumerate([malformed,bad_hash,unbound_partial]):
            with self.subTest(case=i):
                name=f'a{i}';root,result=self.create(name,0);mutation(root);self.refusals_unchanged(name,result)

    def test_archived_pair_remains_bound_after_new_publication(self):
        root,result=self.create(cut=126)
        class Cut(Exception):pass
        def hook(label):
            if label=='PUB_AFTER_TRACE':raise Cut()
        with mock.patch.object(j,'_FAULT_HOOK',hook):
            with self.assertRaises(Cut):p.resume(self.store,'a',result['token'])
        token=p.checkpoint(self.store,'a')['token']
        with mock.patch.object(qs,'advance_step',side_effect=AssertionError('publication simulated')):
            complete=p.resume(self.store,'a',token)
        self.assertEqual(complete['status'],'COMPLETE')
        for anchor,scenario in [(root/'anchors/generation-0.scenario',root/'forensics/generation-1/partial/scenario.json'),
                                (root/'anchors/generation-1.scenario',root/'product/scenario.json')]:
            self.assertEqual((anchor.stat().st_ino,anchor.stat().st_nlink),(scenario.stat().st_ino,2))
        before=snapshot(root);self.assertEqual(p.verify(self.store,'a')['status'],'VERIFIED');self.assertEqual(snapshot(root),before)
        outside=self.base/'outside-archived-scenario';os.link(root/'forensics/generation-1/partial/scenario.json',outside)
        self.refusals_unchanged('a',complete,outside)


if __name__=='__main__':unittest.main()
