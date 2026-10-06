"""Finite native persisted simulation; local cooperating writers only."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import time

from . import __version__, quote_study as qs, cli
from .contracts import InputError
from .provenance import code_identity
from .report import render
from . import persist_journal as j

ResumeError = j.ResumeError


def _geometry(root):
    p=root/'geometry.sqlite';c=sqlite3.connect(p,isolation_level=None,timeout=0)
    try:
        for text in ['page_size=4096','journal_mode=DELETE','synchronous=FULL','cache_spill=OFF','temp_store=MEMORY','max_page_count=16']:
            c.execute('PRAGMA '+text).fetchall()
        c.execute('CREATE TABLE geometry(id INTEGER PRIMARY KEY, value BLOB) STRICT')
        c.execute('INSERT OR ROLLBACK INTO geometry VALUES(1,?)',(b'x'*4096,))
        c.execute('BEGIN IMMEDIATE');c.execute('UPDATE OR ROLLBACK geometry SET value=? WHERE id=1',(b'y'*4096,))
        data=Path(str(p)+'-journal').read_bytes();sector=int.from_bytes(data[20:24],'big');page=int.from_bytes(data[24:28],'big')
        if not 512<=sector<=4096 or sector&(sector-1) or page!=4096:j.fail('UNSUPPORTED','unsupported journal geometry')
        c.execute('ROLLBACK')
    finally:c.close()
    result=dict(sqlite=sqlite3.sqlite_version,sector=sector,page=page,db_journal_reserve=j.MAX_DB_JOURNAL)
    j.write_new(root/'capability.json',j.blob(result));return result


def _initialize(root, raw, plan, context):
    _geometry(root)
    records=[];parts=[]
    for rid,(s,e,p) in enumerate(plan):
        name=f"{s['id']}--{e['id']}--{p}";binding=j.blob(dict(scenario=s,execution=e,policy=p),256*1024)
        genesis=j.blob(qs.initialize_scenario(s,e,p),8192)
        records.append((rid,name,binding,genesis));parts.append(dict(rid=rid,name=name,binding_sha256=j.sha(binding),genesis_sha256=j.sha(genesis)))
    native=code_identity()['sha256']
    meta=j.blob(dict(schema='imc4-persist-meta/v1',input_sha256=j.sha(raw),native_code_sha256=native,
        runtime=j.runtime(),profile=j.PROFILE,context_sha256=context,output=str(root/'product')))
    anchor=j.sha(j.blob(dict(meta_sha256=j.sha(meta),raw_sha256=j.sha(raw),runs=parts),32768))
    ident=dict(schema='imc4-persist/v1',attempt_id=root.name,path=str(root),lock=j.identity(root/'attempt.lock'),
        input_sha256=j.sha(raw),native_code_sha256=native,version=__version__,runtime=j.runtime(),context_sha256=context,
        profile=j.PROFILE,genesis_sha256=anchor)
    if sum(len(x[2])+len(x[3]) for x in records)+len(raw)+len(meta)>j.MAX_LOGICAL:j.fail('RESOURCE_LIMIT','genesis logical bound')
    j.write_new(root/'identity.json',j.blob(ident))
    c=j.connection(root)
    try:
        with j.transaction(c,root):
            for sql in j.SQL:c.execute(sql)
            c.execute('INSERT OR ROLLBACK INTO meta VALUES(1,?,?)',(meta,raw))
            c.executemany('INSERT OR ROLLBACK INTO runs VALUES(?,?,?,?)',records)
            data=j.blob(dict(phase='RUNNING',publication=j.initial_publication(),payload=anchor),16384)
            digest=j.sha(j.blob(dict(rev=0,g=0,kind='GENESIS',data_sha256=j.sha(data),prev='')))
            c.execute('INSERT OR ROLLBACK INTO ops VALUES(0,0,?,?,?,?)',('GENESIS',data,'',digest))
            c.execute('INSERT OR ROLLBACK INTO head VALUES(1,0,0,?,?,?)',(digest,'RUNNING',j.blob(j.initial_publication(),8192)))
            j.fault('INIT_BEFORE_COMMIT')
        j.sync_directory(root)
    finally:c.close()
    return ident


def _forensic_directory(root):
    base=root/'forensics';base.mkdir(exist_ok=True)
    directories=sorted(base.iterdir())
    if len(directories)>=j.MAX_FORENSICS:j.fail('RESOURCE_LIMIT','two forensic generations already retained')
    for d in directories:
        if not d.is_dir() or not (d/'complete.json').is_file():j.fail('OWNERSHIP_AMBIGUOUS','prior forensic capture is incomplete')
    target=base/f'generation-{len(directories)+1}'
    target.mkdir(exist_ok=False);j.sync_directory(base)
    return target


def _file_commitments(root):
    return {str(p.relative_to(root)):dict(sha256=j.sha(p.read_bytes()),bytes=p.stat().st_size)
            for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'complete.json'}


def _capture(root):
    paths=[p for p in [root/'journal.sqlite',root/'journal.sqlite-journal'] if p.exists()]
    amount=sum(p.stat().st_size for p in paths)
    j.reserve(root,amount+32768);dest=_forensic_directory(root)
    for path in paths:
        before=path.stat();data=path.read_bytes();after=path.stat()
        if (before.st_ino,before.st_mtime_ns,before.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size):j.fail('BUSY','capture source changed')
        j.write_new(dest/path.name,data)
        if (dest/path.name).read_bytes()!=data:j.fail('CORRUPT','capture readback differs')
    receipt=dict(schema='imc4-forensic/v1',kind='database-journal',files=_file_commitments(dest))
    j.fault('CAPTURE_BEFORE_RECEIPT');j.write_new(dest/'complete.json',j.blob(receipt,32768))
    return str(dest.relative_to(root))


def _load(root, context, *, readonly):
    ident=j.read_identity(root,context);j.reserve(root)
    if not all((root/name).is_dir() for name in ['anchors','forensics']):j.fail('OWNERSHIP_AMBIGUOUS','missing operational directory')
    if (root/'capability.json').stat().st_size>32768:j.fail('CORRUPT','oversized capability record')
    capability=j.decode((root/'capability.json').read_bytes())
    if capability.get('sqlite')!=sqlite3.sqlite_version or capability.get('page')!=4096 or capability.get('sector') not in (512,1024,2048,4096):j.fail('UNSUPPORTED','unqualified journal geometry')
    journal=root/'journal.sqlite-journal';capture=None
    if journal.exists() and journal.stat().st_size:
        if readonly:j.fail('RECOVERY_REQUIRED','database/journal must be captured before SQLite recovery')
        # A hot journal cannot authorize an unregistered publication alias.
        # When publication objects exist, prove their binding without recovery
        # before preserving or mutating anything. Ambiguous evidence stays put.
        if (root/'product').exists() or any((root/'anchors').iterdir()) or any((root/'forensics').glob('*/partial')):
            probe=j.connection(root,readonly=True)
            try:_admit_filesystem(root,j.audit(probe,ident))
            finally:probe.close()
        capture=_capture(root)
    # Ordinary admission is entirely read-only until both journal and filesystem
    # agree. The lifetime flock prevents a cooperating writer changing either.
    c=j.connection(root,readonly=readonly or capture is None)
    try:
        if not c.execute("SELECT count(*) FROM sqlite_master WHERE name='head'").fetchone()[0]:j.fail('INITIALIZATION_INCOMPLETE','genesis not committed')
        a=j.audit(c,ident);prior_token=j.token(ident,a['head'])
        _admit_filesystem(root,a)
        if not readonly and capture is None:
            c.close();c=j.connection(root)
        if capture is not None:
            a['head']=j.operational(c,root,'RECOVERED',a['head']['phase'],a['head']['publication'],capture)
        a['recovery_prior_token']=prior_token
        return c,ident,a
    except BaseException:c.close();raise


def _budget(value):
    if value is not None and (type(value) is not int or not 0<=value<=1024):j.fail('RESOURCE_LIMIT','stop budget must be integer0..1024')


def _payloads(a):
    start=time.monotonic();runs={}
    for rid,(s,e,p) in enumerate(a['plan']):
        name=f"{s['id']}--{e['id']}--{p}"
        runs[name]=qs.finish_scenario(s,e,p,a['states'][rid],a['deltas'][rid])
    comparison,runs=qs.finish_study(a['raw'],runs)
    payloads={'scenario.json':a['raw'],'comparison.json':cli.json_bytes(comparison),'comparison.md':cli.study_markdown_bytes(comparison)}
    for row in comparison['rows']:
        name=row['run'];r=runs[name];report=r['report']
        numerical=cli.json_bytes(report);visual=render(report).encode()
        receipt=dict(schema='imc4-analysis-completion/v1',tool_version=__version__,code_identity=report['code_identity'],runtime=report['runtime'],source_sha256=report['source_sha256'],
            files={n:dict(sha256=j.sha(data),bytes=len(data)) for n,data in [('report.json',numerical),('report.html',visual)]})
        payloads.update({name+'/report.json':numerical,name+'/report.html':visual,name+'/complete.json':cli.json_bytes(receipt),name+'/normalized.jsonl':r['normalized'],
            name+'/trace.json':cli.json_bytes(dict(policy=r['policy'],policy_settings=r['policy_settings'],exogenous_sha256=row['exogenous_sha256'],normalized_sha256=row['normalized_sha256'],steps=r['trace']))})
    receipt=dict(schema='imc4-quote-study-completion/v1',tool_version=__version__,code_identity=comparison['code_identity'],runtime=comparison['runtime'],source_sha256=comparison['source_sha256'],
        files={n:dict(sha256=j.sha(data),bytes=len(data)) for n,data in sorted(payloads.items())})
    payloads['complete.json']=cli.json_bytes(receipt)
    # Conservative receipt/Markdown reserve frozen before implementation.
    bound=sum(len(v) for k,v in payloads.items() if not k.endswith('complete.json') and k!='comparison.md')+8192+2048*len(runs)+65536*(len(runs)+1)
    if bound>j.MAX_PRODUCT or sum(map(len,payloads.values()))>j.MAX_PRODUCT:j.fail('RESOURCE_LIMIT','publication pre-write bound exceeds128MiB')
    a['metrics']['derivation_seconds']=time.monotonic()-start
    return comparison,runs,payloads,bound


def _owner(root, output, generation):
    scenario=output/'scenario.json';anchor=root/'anchors'/f'generation-{generation}.scenario'
    return dict(directory=j.identity(output),scenario=j.identity(scenario),anchor=j.identity(anchor),raw_sha256=j.sha(scenario.read_bytes()))


def _owned(root, output, pub, payloads):
    owner=pub['owner']
    if owner is None or not output.is_dir():j.fail('OWNERSHIP_AMBIGUOUS','product has no committed creation binding')
    anchor=root/'anchors'/f"generation-{pub['generation']}.scenario"
    if not anchor.is_file() or not (output/'scenario.json').is_file():j.fail('OWNERSHIP_AMBIGUOUS','missing permanent scenario anchor')
    if _owner(root,output,pub['generation'])!=owner or owner['scenario']!=owner['anchor']:
        j.fail('OWNERSHIP_AMBIGUOUS','directory or scenario inode replaced')
    if anchor.stat().st_nlink!=2 or (output/'scenario.json').stat().st_nlink!=2:
        j.fail('OWNERSHIP_AMBIGUOUS','scenario must have exactly its registered anchor link')
    return _contents(output,payloads)


def _contents(output,payloads):
    expected_dirs={str(Path(k).parent) for k in payloads if '/' in k}
    total=0
    for path in output.rglob('*'):
        rel=str(path.relative_to(output));st=path.lstat()
        if path.is_symlink():j.fail('UNEXPECTED_OUTPUT','linked product member')
        if path.is_dir():
            if rel not in expected_dirs:j.fail('UNEXPECTED_OUTPUT','unexpected product directory')
            continue
        if not path.is_file() or rel not in payloads:j.fail('UNEXPECTED_OUTPUT','unexpected product member')
        if st.st_nlink>1 and rel!='scenario.json':j.fail('UNEXPECTED_OUTPUT','unexpected hard-linked payload')
        total+=st.st_size
        if total>j.MAX_PRODUCT:j.fail('RESOURCE_LIMIT','product content exceeds cap')
        data=path.read_bytes()
        if data!=payloads[rel][:len(data)]:j.fail('UNEXPECTED_OUTPUT','product content is not an expected complete or interrupted write')
    return True


def _admit_filesystem(root,a):
    """Bind the entire admitted tree to audited publication/capture history."""
    try:_filesystem_bindings(root,a)
    except ResumeError:
        if a['head']['phase']=='COMPLETE':j.fail('INCONSISTENT_COMPLETE','completed output is missing or changed')
        raise


def _filesystem_bindings(root,a):
    h=a['head'];current=h['publication'];publications=a['publications']
    expected_anchors={f'generation-{g}.scenario' for g in publications}
    actual_anchors={p.name for p in (root/'anchors').iterdir()}
    archives={p['archive']['destination']:p for p in publications.values() if p['archive'] is not None}
    active=current['archive']['destination'] if h['phase']=='ARCHIVING' else None
    containers={str(p.relative_to(root)):p for p in (root/'forensics').iterdir()}
    captures=set()
    for name,directory in containers.items():
        partial=name+'/partial';receipt=directory/'complete.json'
        if partial in archives:
            # Before rename the registered container is empty. After rename its
            # exact bound partial may exist before the forensic receipt commits.
            if partial!=active and not receipt.is_file():j.fail('OWNERSHIP_AMBIGUOUS','archived generation lacks complete receipt')
            if not (directory/'partial').exists() and partial!=active:j.fail('OWNERSHIP_AMBIGUOUS','registered archive is missing')
            if any(p.name not in {'partial','complete.json'} for p in directory.iterdir()):j.fail('UNEXPECTED_OUTPUT','archive contains unrelated capture files')
            if receipt.exists() and j.decode(receipt.read_bytes())['kind']!='partial-publication':j.fail('CORRUPT','archive receipt kind differs')
        else:
            if (directory/'partial').exists():j.fail('OWNERSHIP_AMBIGUOUS','partial generation has no committed archive binding')
            if not receipt.is_file():j.fail('OWNERSHIP_AMBIGUOUS','forensic capture is incomplete')
            if j.decode(receipt.read_bytes())['kind']!='database-journal':j.fail('CORRUPT','unbound forensic publication')
            captures.add(name)
    if not a['captures']<=captures:j.fail('CORRUPT','committed recovery capture is missing')
    if any(str(Path(n).parent) not in containers for n in archives):j.fail('OWNERSHIP_AMBIGUOUS','archive container is missing')
    try:
        if actual_anchors!=expected_anchors:j.fail('OWNERSHIP_AMBIGUOUS','anchor set differs from committed ownership history')
        output=root/'product'
        if not publications and not output.exists():return
        if output.exists() and not current['intent']:j.fail('UNEXPECTED_OUTPUT','product exists without publication intent')
        _,_,payloads,_=_payloads(a)
        for pub in publications.values():
            archive=pub['archive']
            if archive is None:target=output
            else:
                target=root/archive['destination']
                if archive['destination']==active:
                    if target.exists()==output.exists():j.fail('OWNERSHIP_AMBIGUOUS','archive source/destination identity is ambiguous')
                    if output.exists():target=output
            _owned(root,target,pub,payloads)
        if output.exists() and current['owner'] is None:
            # A known pre-owner creation window is inspectable for its token;
            # _publish still refuses any move/reuse without committed ownership.
            _contents(output,payloads)
    except ResumeError:
        if h['phase']=='COMPLETE':j.fail('INCONSISTENT_COMPLETE','completed output is missing or changed')
        raise


def _complete(output,payloads):
    actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    return actual==set(payloads) and all((output/n).read_bytes()==data for n,data in payloads.items())


def _archive(c,root,a,payloads):
    h=a['head'];pub=copy.deepcopy(h['publication']);src=root/'product'
    if h['phase']!='ARCHIVING':
        _owned(root,src,pub,payloads);j.reserve(root,32768)
        generation=_forensic_directory(root)
        pub['archive']=dict(destination=str((generation/'partial').relative_to(root)),owner=pub['owner'])
        h=j.operational(c,root,'ARCHIVE_INTENT','ARCHIVING',pub)
    dest=root/pub['archive']['destination'];container=dest.parent
    if src.exists() and not dest.exists():
        _owned(root,src,pub,payloads);j.fault('ARCHIVE_BEFORE_MOVE')
        os.rename(src,dest);j.sync_directory(root);j.sync_directory(container);j.fault('ARCHIVE_AFTER_MOVE')
    elif dest.exists() and not src.exists():_owned(root,dest,pub,payloads)
    else:j.fail('OWNERSHIP_AMBIGUOUS','archive source/destination identity is ambiguous')
    receipt=dict(schema='imc4-forensic/v1',kind='partial-publication',files=_file_commitments(container))
    target=container/'complete.json'
    if target.exists():
        if j.decode(target.read_bytes(),32768)!=receipt:j.fail('CORRUPT','archive receipt differs')
    else:j.write_new(target,j.blob(receipt,32768))
    new=j.initial_publication();new['generation']=pub['generation']+1
    a['head']=j.operational(c,root,'ARCHIVED','COMPUTE_COMPLETE',new)


def _publish(c,root,a):
    comparison,runs,payloads,bound=_payloads(a);output=root/'product'
    h=a['head'];pub=h['publication']
    if h['phase']=='ARCHIVING':_archive(c,root,a,payloads);h=a['head'];pub=h['publication']
    if output.exists():
        _owned(root,output,pub,payloads)
        if _complete(output,payloads):
            new=copy.deepcopy(pub);new['receipt_sha256']=j.sha(payloads['complete.json'])
            a['head']=j.operational(c,root,'COMPLETE','COMPLETE',new)
            return
        _archive(c,root,a,payloads);h=a['head'];pub=h['publication']
    if pub['owner'] is not None:j.fail('OWNERSHIP_AMBIGUOUS','registered product is missing')
    j.reserve(root,bound)
    if not pub['intent']:
        pub=copy.deepcopy(pub);pub['intent']=True
        a['head']=j.operational(c,root,'INTENT','PUBLISHING',pub);j.fault('PUB_AFTER_INTENT')
    anchor=root/'anchors'/f"generation-{pub['generation']}.scenario"
    if anchor.exists():j.fail('OWNERSHIP_AMBIGUOUS','unregistered generation anchor')
    written=0;start=time.monotonic()
    def created(path):j.sync_directory(root);j.fault('PUB_AFTER_MKDIR')
    def before(path,content):
        nonlocal written
        rel=str(path.relative_to(output))
        if rel not in payloads or content!=payloads[rel]:j.fail('CORRUPT','writer differs from native expected payload')
        if written+len(content)>j.MAX_PRODUCT:j.fail('RESOURCE_LIMIT','publication write cap')
        j.reserve(root,len(content));written+=len(content)
        if path==output/'complete.json':j.fault('PUB_PARTIAL_ROOT_RECEIPT')
    def after(path):
        nonlocal pub
        if path==output/'scenario.json':
            j.fault('PUB_AFTER_SCENARIO');os.link(path,anchor);j.sync_directory(anchor.parent);j.sync_directory(output)
            pub=copy.deepcopy(pub);pub['owner']=_owner(root,output,pub['generation'])
            a['head']=j.operational(c,root,'OWNER','PUBLISHING',pub);j.fault('PUB_AFTER_OWNER')
        elif path==output/'complete.json':j.fault('PUB_AFTER_ROOT_RECEIPT')
        elif path.name=='complete.json':j.fault('PUB_AFTER_NESTED_RECEIPT')
        elif path.name=='trace.json':j.fault('PUB_AFTER_TRACE')
    try:
        cli.write_study_results(a['raw'],comparison,runs,output,_on_create=created,_before_write=before,_after_write=after)
    except (OSError,ResumeError) as exc:
        a['publication_error']=str(exc);return
    _owned(root,output,pub,payloads)
    if not _complete(output,payloads):j.fail('CORRUPT','writer did not complete expected product')
    j.fault('PUB_BEFORE_COMPLETE');pub=copy.deepcopy(pub);pub['receipt_sha256']=j.sha(payloads['complete.json'])
    a['head']=j.operational(c,root,'COMPLETE','COMPLETE',pub)
    a['metrics'].update(publication_seconds=time.monotonic()-start,publication_bytes=written,publication_bound_bytes=bound)


def _advance(c,root,a,budget):
    start=time.monotonic();calls=[];db_peak=journal_peak=0
    while a['head']['g']<len(a['flat']) and (budget is None or len(calls)<budget):
        old=a['head'];g=old['g']+1;rid,tick=a['flat'][g-1];s,e,p=a['plan'][rid]
        if tick==s['end_timestamp']:j.fault('BEFORE_TERMINAL')
        state,delta=qs.advance_step(s,e,p,a['states'][rid],s['steps'][tick])
        st=j.blob(state,8192);tr=j.blob(delta['trace'],32768);events=[j.blob(v,2048) for v in delta['events']];eb=b''.join(events)
        if len(eb)>8192:j.fail('RESOURCE_LIMIT','event block cap')
        added=len(st)+len(tr)+len(eb)
        added_nodes=j.shape(state)+j.shape(delta['trace'])+sum(j.shape(v) for v in delta['events'])
        if a['metrics']['logical_bytes']+added+16384>j.MAX_LOGICAL or a['metrics']['decoded_nodes']+added_nodes+4096>j.MAX_NODES:
            j.fail('RESOURCE_LIMIT','pre-insert logical/decoded reserve')
        prev=a['step_hashes'][-1] if a['step_hashes'] else a['anchor']
        digest=j.step_digest(g,rid,tick,prev,st,tr,eb)
        with j.transaction(c,root):
            c.execute('INSERT OR ROLLBACK INTO steps VALUES(?,?,?,?,?,?,?,?)',(g,rid,tick,prev,digest,st,tr,j.sha(eb)))
            for seq,(value,data) in enumerate(zip(delta['events'],events)):
                c.execute('INSERT OR ROLLBACK INTO events VALUES(?,?,?,?,?,?)',(rid,tick,seq,value.get('fill_id'),g,data))
            j.fault('STEP_AFTER_ROWS')
            h=j.op(c,old,'STEP','COMPUTE_COMPLETE' if g==len(a['flat']) else 'RUNNING',old['publication'],digest,g)
            j.fault('STEP_AFTER_HEAD')
            _,sizes=j.tree(root);db_peak=max(db_peak,sizes.get('journal.sqlite',0));journal_peak=max(journal_peak,sizes.get('journal.sqlite-journal',0))
            j.fault('STEP_BEFORE_COMMIT')
        j.fault('STEP_AFTER_COMMIT')
        a['head']=h;a['states'][rid]=state;a['deltas'][rid].append(delta);a['step_hashes'].append(digest)
        a['metrics']['logical_bytes']+=added+c.execute('SELECT length(data) FROM ops WHERE rev=?',(h['rev'],)).fetchone()[0]
        a['metrics']['decoded_nodes']+=added_nodes+j.shape(j.decode(c.execute('SELECT data FROM ops WHERE rev=?',(h['rev'],)).fetchone()[0],16384));calls.append([state['run'],tick])
        if tick==s['end_timestamp']:j.fault('AFTER_TERMINAL')
    a['metrics'].update(suffix_calls=calls,suffix_steps=len(calls),suffix_seconds=time.monotonic()-start,
                         observed_database_peak_bytes=db_peak,observed_journal_peak_bytes=journal_peak)
    if budget is not None and len(calls)==budget:return
    if a['head']['g']==len(a['flat']):_publish(c,root,a)


def _result(root,ident,a,status=None,*,c):
    phase=a['head']['phase']
    status=status or ('PUBLICATION_INCOMPLETE' if 'publication_error' in a else
        {'RUNNING':'PAUSED_AT_BOUNDARY','COMPUTE_COMPLETE':'COMPUTE_COMPLETE','PUBLISHING':'PUBLICATION_INCOMPLETE','ARCHIVING':'PUBLICATION_INCOMPLETE','COMPLETE':'COMPLETE'}[phase])
    total,files=j.tree(root);metrics=dict(a['metrics'],owned_bytes=total,current_database_bytes=files.get('journal.sqlite',0),current_journal_bytes=files.get('journal.sqlite-journal',0),
        forensic_bytes=sum(v for k,v in files.items() if k.startswith('forensics/')),run_count=len(a['plan']),total_steps=len(a['flat']))
    metrics['logical_bytes'],metrics['stored_decoded_nodes']=j.content_sizes(c)
    result=dict(status=status,token=j.token(ident,a['head']),output=str(root/'product'),metrics=metrics)
    if 'publication_error' in a:result['error']=a['publication_error']
    return result


def _errors(call):
    try:return call()
    except ResumeError:raise
    except sqlite3.Error as exc:j.fail('RESOURCE_LIMIT' if getattr(exc,'sqlite_errorcode',None)==sqlite3.SQLITE_FULL else 'SQLITE_FAILURE',str(exc))
    except OSError as exc:j.fail('SQLITE_FAILURE',str(exc))
    except (InputError,ValueError,TypeError,KeyError,IndexError) as exc:j.fail('INVALID_STATE',str(exc))


def run(raw,store,attempt_id,*,context_sha256=None,stop_after_commits=None):
    """Exclusively initialize and advance one finite simulated attempt."""
    def work():
        _budget(stop_after_commits);plan=j.admitted(raw,initial=True)
        if context_sha256 is not None and (not isinstance(context_sha256,str) or len(context_sha256)!=64 or any(c not in '0123456789abcdef' for c in context_sha256)):j.fail('CONTEXT_MISMATCH','invalid context digest')
        root=j.attempt_path(store,attempt_id);root.parent.mkdir(parents=True,exist_ok=True)
        try:root.mkdir(exist_ok=False)
        except FileExistsError:j.fail('EXISTS','attempt path already exists')
        fd=os.open(root/'attempt.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
        with j.locked(root):
            (root/'anchors').mkdir();(root/'forensics').mkdir();_initialize(root,raw,plan,context_sha256)
            c,ident,a=_load(root,context_sha256,readonly=False)
            try:_advance(c,root,a,stop_after_commits);return _result(root,ident,a,c=c)
            finally:c.close()
    return _errors(work)


def checkpoint(store,attempt_id,*,context_sha256=None):
    def work():
        root=j.attempt_path(store,attempt_id)
        with j.locked(root,shared=True):
            c,ident,a=_load(root,context_sha256,readonly=True)
            try:return _result(root,ident,a,c=c)
            finally:c.close()
    return _errors(work)


def resume(store,attempt_id,expected_token,*,context_sha256=None,stop_after_commits=None):
    def work():
        _budget(stop_after_commits);root=j.attempt_path(store,attempt_id)
        with j.locked(root):
            c,ident,a=_load(root,context_sha256,readonly=False)
            try:
                if expected_token!=a['recovery_prior_token']:j.fail('STALE_CHECKPOINT','token differs from committed head')
                if a['head']['phase']=='COMPLETE':
                    _,_,payloads,_=_payloads(a)
                    try:_owned(root,root/'product',a['head']['publication'],payloads);complete=_complete(root/'product',payloads)
                    except ResumeError:j.fail('INCONSISTENT_COMPLETE','completed output is missing or changed')
                    if not complete:j.fail('INCONSISTENT_COMPLETE','completed output is missing or changed')
                    j.fail('ALREADY_COMPLETED','completed attempt is unchanged')
                _advance(c,root,a,stop_after_commits);return _result(root,ident,a,c=c)
            finally:c.close()
    return _errors(work)


def verify(store,attempt_id,*,context_sha256=None,mode='state'):
    def work():
        if mode not in ('state','recompute'):j.fail('UNSUPPORTED','unknown verification mode')
        root=j.attempt_path(store,attempt_id)
        with j.locked(root,shared=True):
            c,ident,a=_load(root,context_sha256,readonly=True)
            try:
                if a['head']['phase']=='COMPLETE':
                    comparison,runs,payloads,_=_payloads(a)
                    try:_owned(root,root/'product',a['head']['publication'],payloads);complete=_complete(root/'product',payloads)
                    except ResumeError:j.fail('INCONSISTENT_COMPLETE','completed output is missing or changed')
                    if not complete:j.fail('INCONSISTENT_COMPLETE','completed output is missing or changed')
                if mode=='recompute':
                    recomparison,reruns=qs.run_study(a['raw'])
                    if a['head']['g']==len(a['flat']):
                        comparison,runs,_,_=_payloads(a)
                        if recomparison!=comparison or reruns!=runs:j.fail('CORRUPT','independent native computation differs')
                    else:
                        for rid,(s,e,p) in enumerate(a['plan']):
                            name=f"{s['id']}--{e['id']}--{p}";prefix=[d['trace'] for d in a['deltas'][rid]]
                            if reruns[name]['trace'][:len(prefix)]!=prefix:j.fail('CORRUPT','independent prefix computation differs')
                    a['metrics']['recompute_scope']='Independent compute parity; not suffix-only recovery evidence'
                return _result(root,ident,a,'VERIFIED',c=c)
            finally:c.close()
    return _errors(work)
