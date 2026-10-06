"""Bounded local SQLite storage and read-only committed-effect audit."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import stat
import time

from . import __version__, quote_study as qs
from .contracts import InputError, no_duplicate_keys
from .provenance import code_identity
from .persist_audit import project

MiB = 1024 * 1024
MAX_DB_JOURNAL = 64 * MiB
MAX_PRODUCT = 128 * MiB
MAX_ATTEMPT = 512 * MiB
MAX_PAGES = 3840
MAX_LOGICAL = 8 * MiB
MAX_NODES = 500000
MAX_FORENSICS = 2
PROFILE = dict(page_size=4096, max_pages=3840, db_journal=64*MiB, product=128*MiB,
               attempt=512*MiB, logical=8*MiB, forensic_generations=2, runs=18, steps=1024,
               state_bytes=8192, trace_bytes=32768, event_block_bytes=8192, event_bytes=2048,
               depth=32, record_nodes=4096, total_nodes=500000)
SQL = [
 'CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK(id=1), body BLOB NOT NULL, raw BLOB NOT NULL) STRICT',
 'CREATE TABLE runs (rid INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, binding BLOB NOT NULL, genesis BLOB NOT NULL) STRICT',
 'CREATE TABLE steps (g INTEGER PRIMARY KEY, rid INTEGER NOT NULL REFERENCES runs(rid), tick INTEGER NOT NULL, prev TEXT NOT NULL, digest TEXT NOT NULL, state BLOB NOT NULL, trace BLOB NOT NULL, events_sha TEXT NOT NULL, UNIQUE(rid,tick)) STRICT',
 'CREATE TABLE events (rid INTEGER NOT NULL, tick INTEGER NOT NULL, seq INTEGER NOT NULL, fill_id TEXT, g INTEGER NOT NULL REFERENCES steps(g), data BLOB NOT NULL, PRIMARY KEY(rid,tick,seq), UNIQUE(rid,fill_id), FOREIGN KEY(rid) REFERENCES runs(rid)) STRICT',
 'CREATE TABLE ops (rev INTEGER PRIMARY KEY, g INTEGER NOT NULL, kind TEXT NOT NULL, data BLOB NOT NULL, prev TEXT NOT NULL, digest TEXT NOT NULL) STRICT',
 'CREATE TABLE head (id INTEGER PRIMARY KEY CHECK(id=1), rev INTEGER NOT NULL, g INTEGER NOT NULL, digest TEXT NOT NULL, phase TEXT NOT NULL, publication BLOB NOT NULL) STRICT',
]
PHASES = ('RUNNING','COMPUTE_COMPLETE','PUBLISHING','ARCHIVING','COMPLETE')
_FAULT_HOOK = None  # Child-local tests only; no CLI/environment dispatch.


class ResumeError(InputError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def fail(code, message): raise ResumeError(code, message)
def fault(label):
    if _FAULT_HOOK is not None: _FAULT_HOOK(label)
def sha(value): return hashlib.sha256(value).hexdigest()


def shape(value):
    stack=[(value,0)];nodes=0
    while stack:
        v,d=stack.pop();nodes+=1
        if d>32 or nodes>4096:fail('RESOURCE_LIMIT','JSON structure exceeds record bound')
        if isinstance(v,dict):stack.extend((x,d+1) for x in v.values())
        elif isinstance(v,list):stack.extend((x,d+1) for x in v)
        elif v is not None and type(v) not in (str,int,bool):fail('CORRUPT','unsupported JSON scalar')
    return nodes


def blob(value, limit=32768):
    shape(value);b=qs.encoded(value)
    if len(b)>limit:fail('RESOURCE_LIMIT','serialized record exceeds bound')
    return b


def decode(value, limit=32768):
    if type(value) is not bytes or len(value)>limit:fail('CORRUPT','invalid bounded SQL blob')
    try:
        v=json.loads(value,object_pairs_hook=no_duplicate_keys)
        if blob(v,limit)!=value:fail('CORRUPT','noncanonical JSON record')
        return v
    except (ValueError,TypeError,RecursionError) as exc:
        if isinstance(exc,ResumeError):raise
        fail('CORRUPT',str(exc))


def identity(path):
    s=path.lstat()
    return dict(dev=s.st_dev,ino=s.st_ino,birthtime=str(getattr(s,'st_birthtime',None)))


def safe_path(value):
    path=Path(value).absolute()
    for p in reversed([path]+list(path.parents)):
        if p.is_symlink():fail('OWNERSHIP_AMBIGUOUS','symlink path component')
    return path


def attempt_path(store, attempt_id):
    if not isinstance(attempt_id,str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}',attempt_id):
        fail('IDENTITY_MISMATCH','invalid attempt identifier')
    return safe_path(store)/attempt_id


def _product_member(parts, directory):
    """Coarse writer grammar; exact run names/content require the audited plan."""
    roots={'scenario.json','comparison.json','comparison.md','complete.json'}
    if directory:return len(parts)==0 or (len(parts)==1 and parts[0] not in roots)
    # Defer exact payload names to state-bound admission so COMPLETE corruption
    # retains its public INCONSISTENT_COMPLETE outcome. Links/types are not deferred.
    return len(parts)==1 or (len(parts)==2 and parts[0] not in roots)


def _forensic_receipts(root, files, generations):
    # Missing receipts are transient during creation. Public admission must bind
    # those windows to the journal; a present receipt must always be exact.
    for name in generations:
        prefix='forensics/'+name+'/'
        receipt=prefix+'complete.json'
        if receipt not in files:continue
        if files[receipt]>32768:fail('RESOURCE_LIMIT','forensic receipt exceeds cap')
        value=decode((root/receipt).read_bytes(),32768)
        if not isinstance(value,dict) or set(value)!={'schema','kind','files'} or value['schema']!='imc4-forensic/v1':
            fail('CORRUPT','forensic receipt shape')
        if not isinstance(value['files'],dict) or any(not isinstance(v,dict) or set(v)!={'sha256','bytes'} or
                type(v['bytes']) is not int or v['bytes']<0 or not isinstance(v['sha256'],str) or
                not re.fullmatch('[0-9a-f]{64}',v['sha256']) for v in value['files'].values()):
            fail('CORRUPT','forensic commitment types')
        actual={n[len(prefix):]:dict(sha256=sha((root/n).read_bytes()),bytes=size)
                for n,size in files.items() if n.startswith(prefix) and n!=receipt}
        names=set(actual)
        if value['kind']=='database-journal':
            valid='journal.sqlite' in names and names<={'journal.sqlite','journal.sqlite-journal'} and not (root/prefix/'partial').exists()
        elif value['kind']=='partial-publication':
            valid='partial/scenario.json' in names and all(n.startswith('partial/') for n in names)
        else:valid=False
        if not valid or value['files']!=actual:fail('CORRUPT','forensic receipt membership or commitment differs')


def tree(root):
    """Bound the complete layout and potential link pairs before file access."""
    total=0;nodes=0;files={};groups={};generations=set();products={}
    operational={'identity.json','attempt.lock','journal.sqlite','journal.sqlite-journal','capability.json','geometry.sqlite','geometry.sqlite-journal'}
    if not root.exists():return total,files
    for p in root.rglob('*'):
        nodes+=1
        if nodes>1024:fail('RESOURCE_LIMIT','attempt tree exceeds node cap')
        s=p.lstat()
        if stat.S_ISLNK(s.st_mode) or not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):
            fail('OWNERSHIP_AMBIGUOUS','unexpected filesystem object')
        rel=str(p.relative_to(root));parts=p.relative_to(root).parts;directory=stat.S_ISDIR(s.st_mode)
        if parts[0] in operational:
            allowed=len(parts)==1 and not directory
        elif parts[0]=='anchors':
            allowed=(len(parts)==1 and directory) or (len(parts)==2 and not directory and re.fullmatch(r'generation-[0-2]\.scenario',parts[1]))
        elif parts[0]=='product':
            allowed=_product_member(parts[1:],directory)
            if not directory:products['product']=products.get('product',0)+s.st_size
        elif parts[0]=='forensics':
            allowed=len(parts)==1 and directory
            if len(parts)>=2 and re.fullmatch(r'generation-[12]',parts[1]):
                generations.add(parts[1]);tail=parts[2:]
                allowed=(not tail and directory) or (len(tail)==1 and tail[0] in {'complete.json','journal.sqlite','journal.sqlite-journal'} and not directory)
                if tail and tail[0]=='partial':
                    allowed=_product_member(tail[1:],directory)
                    key='/'.join(parts[:3])
                    if not directory:products[key]=products.get(key,0)+s.st_size
        else:allowed=False
        if not allowed:fail('UNEXPECTED_OUTPUT','unexpected attempt member or object type: '+rel)
        if stat.S_ISREG(s.st_mode):
            total+=s.st_size;files[rel]=s.st_size
            if total>MAX_ATTEMPT:fail('RESOURCE_LIMIT','owned attempt content exceeds cap')
            if s.st_nlink!=1:groups.setdefault((s.st_dev,s.st_ino),[]).append((parts,s.st_nlink))
    for members in groups.values():
        anchors=[p for p,n in members if len(p)==2 and p[0]=='anchors']
        scenarios=[p for p,n in members if p==('product','scenario.json') or
                   (len(p)==4 and p[0]=='forensics' and p[2:]==('partial','scenario.json'))]
        if len(members)!=2 or any(n!=2 for p,n in members) or len(anchors)!=1 or len(scenarios)!=1:
            fail('OWNERSHIP_AMBIGUOUS','unregistered hard-link alias or incomplete scenario pair')
    if files.get('journal.sqlite',0)>MAX_PAGES*4096 or files.get('journal.sqlite',0)+files.get('journal.sqlite-journal',0)>MAX_DB_JOURNAL:
        fail('RESOURCE_LIMIT','database/journal exceeds cap')
    if any(size>MAX_PRODUCT for size in products.values()):fail('RESOURCE_LIMIT','product content exceeds cap')
    if 'generation-2' in generations and 'generation-1' not in generations:fail('OWNERSHIP_AMBIGUOUS','noncontiguous forensic generations')
    for name in generations:
        prefix='forensics/'+name+'/'
        captured=files.get(prefix+'journal.sqlite',0)+files.get(prefix+'journal.sqlite-journal',0)
        if captured>MAX_DB_JOURNAL:fail('RESOURCE_LIMIT','forensic database/journal exceeds cap')
        if any(n.startswith(prefix+'partial/') for n in files) and any(prefix+n in files for n in ['journal.sqlite','journal.sqlite-journal']):
            fail('UNEXPECTED_OUTPUT','mixed forensic generation kinds')
    _forensic_receipts(root,files,generations)
    return total,files


def reserve(root, extra=0):
    total,files=tree(root)
    active=files.get('journal.sqlite',0)+files.get('journal.sqlite-journal',0)
    if total-active+MAX_DB_JOURNAL+extra>MAX_ATTEMPT:fail('RESOURCE_LIMIT','insufficient bounded attempt reserve')
    return total,files


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_new(path, data):
    reserve(path if path.is_dir() else _root_of(path),len(data))
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    sync_directory(path.parent)


def _root_of(path):
    # All internal writes descend from an existing stable attempt lock.
    for parent in path.parents:
        if (parent/'attempt.lock').exists():return parent
    fail('INITIALIZATION_INCOMPLETE','no attempt owner')


@contextmanager
def locked(root, *, shared=False):
    safe_path(root);tree(root)
    path=root/'attempt.lock'
    if not path.is_file():fail('INITIALIZATION_INCOMPLETE','missing stable lock')
    fd=os.open(path,(os.O_RDONLY if shared else os.O_RDWR) | os.O_NOFOLLOW)
    try:
        try:fcntl.flock(fd,(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        except BlockingIOError:fail('BUSY','attempt is owned by another invocation')
        s=os.fstat(fd);p=path.lstat()
        if (s.st_dev,s.st_ino)!=(p.st_dev,p.st_ino) or s.st_nlink!=1:fail('OWNERSHIP_AMBIGUOUS','lock file replaced or aliased')
        tree(root)
        yield
    finally:os.close(fd)


def connection(root, *, readonly=False):
    db=root/'journal.sqlite'
    if readonly:
        c=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True,isolation_level=None,timeout=0)
        c.execute('PRAGMA query_only=ON')
    else:
        c=sqlite3.connect(db,isolation_level=None,timeout=0)
        for pragma in ['page_size=4096','journal_mode=DELETE','synchronous=FULL','foreign_keys=ON','cache_spill=OFF','temp_store=MEMORY','busy_timeout=0','mmap_size=0',f'max_page_count={MAX_PAGES}']:
            c.execute('PRAGMA '+pragma).fetchall()
        required=dict(page_size=4096,journal_mode='delete',synchronous=2,foreign_keys=1,cache_spill=0,temp_store=2,busy_timeout=0,max_page_count=MAX_PAGES)
        if any(c.execute('PRAGMA '+k).fetchone()[0]!=v for k,v in required.items()):
            c.close();fail('UNSUPPORTED','SQLite effective settings differ')
    compile_options={row[0] for row in c.execute('PRAGMA compile_options')}
    if not compile_options.intersection({'TEMP_STORE=1','TEMP_STORE=2','TEMP_STORE=3'}):
        c.close();fail('UNSUPPORTED','SQLite TEMP_STORE compile option is unsupported')
    c.execute('PRAGMA temp_store=MEMORY')
    if c.execute('PRAGMA temp_store').fetchone()[0]!=2:
        c.close();fail('UNSUPPORTED','SQLite temp_store is not MEMORY')
    c.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,512*1024)
    c.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH,16384)
    c.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH,50)
    c.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED,0)
    return c


def runtime():
    return dict(implementation=platform.python_implementation(),python_version=platform.python_version(),
                sqlite=sqlite3.sqlite_version,serializer='sorted-compact-json-newline/v1',accounting_precision=50,matching_precision=80)


def admitted(raw, *, initial=False):
    try:study=qs.load_study(raw,_check_policy=initial)
    except InputError as exc:fail('INVALID_STATE',str(exc))
    plan=[(s,e,p) for s in study['scenarios'] for e in study['executions'] for p in ('symmetric','inventory')]
    if len(plan)>18 or sum(len(s['steps']) for s,e,p in plan)>1024:fail('RESOURCE_LIMIT','resumable run/step admission exceeded')
    return plan


def initial_publication():return dict(generation=0,intent=False,owner=None,archive=None,receipt_sha256=None)


def validate_publication(pub):
    if not isinstance(pub,dict) or set(pub)!={'generation','intent','owner','archive','receipt_sha256'}:
        fail('CORRUPT','publication shape')
    if type(pub['generation']) is not int or not 0<=pub['generation']<=2 or type(pub['intent']) is not bool:
        fail('CORRUPT','publication generation/type')
    owner=pub['owner']
    if owner is not None:
        if not isinstance(owner,dict) or set(owner)!={'directory','scenario','anchor','raw_sha256'}:
            fail('CORRUPT','publication owner shape')
        for name in ['directory','scenario','anchor']:
            value=owner[name]
            if not isinstance(value,dict) or set(value)!={'dev','ino','birthtime'} or type(value['dev']) is not int or type(value['ino']) is not int or not isinstance(value['birthtime'],str):
                fail('CORRUPT','publication inode binding')
        if owner['anchor']!=owner['scenario']:fail('CORRUPT','anchor is not the scenario inode')
    archive=pub['archive']
    if archive is not None:
        if not isinstance(archive,dict) or set(archive)!={'destination','owner'} or archive['owner']!=owner or not isinstance(archive['destination'],str) or not re.fullmatch(r'forensics/generation-[12]/partial',archive['destination']):
            fail('CORRUPT','archive intent shape/path')
    receipt=pub['receipt_sha256']
    if receipt is not None and (not isinstance(receipt,str) or not re.fullmatch('[0-9a-f]{64}',receipt)):
        fail('CORRUPT','publication receipt identity')


def read_identity(root, context):
    if context is not None and (not isinstance(context,str) or not re.fullmatch('[0-9a-f]{64}',context)):
        fail('CONTEXT_MISMATCH','context must be SHA256 or null')
    path=root/'identity.json'
    if not path.is_file():fail('INITIALIZATION_INCOMPLETE','missing identity commitment')
    if path.stat().st_size>32768:fail('CORRUPT','oversized identity record')
    x=decode(path.read_bytes(),32768)
    expected={'schema','attempt_id','path','lock','input_sha256','native_code_sha256','version','runtime','context_sha256','profile','genesis_sha256'}
    if set(x)!=expected or x['schema']!='imc4-persist/v1':fail('IDENTITY_MISMATCH','unknown identity schema')
    if x['context_sha256']!=context:fail('CONTEXT_MISMATCH','external context differs')
    if x['path']!=str(root) or x['attempt_id']!=root.name or x['lock']!=identity(root/'attempt.lock'):
        fail('IDENTITY_MISMATCH','attempt/lock path binding differs')
    if x['native_code_sha256']!=code_identity()['sha256'] or x['version']!=__version__ or x['runtime']!=runtime() or x['profile']!=PROFILE:
        fail('IDENTITY_MISMATCH','native code/runtime/resource profile differs')
    return x


def head(c):
    row=c.execute('SELECT rev,g,digest,phase,publication FROM head WHERE id=1').fetchone()
    if row is None:fail('INITIALIZATION_INCOMPLETE','no committed head')
    rev,g,digest,phase,pub=row
    if type(rev) is not int or type(g) is not int or phase not in PHASES:fail('CORRUPT','head type/phase')
    return dict(rev=rev,g=g,digest=digest,phase=phase,publication=decode(pub,8192))


def op(c, old, kind, phase, publication, payload, g=None):
    g=old['g'] if g is None else g;rev=old['rev']+1
    if rev>1151:fail('RESOURCE_LIMIT','operational revision cap')
    data=blob(dict(phase=phase,publication=publication,payload=payload),16384)
    digest=sha(blob(dict(rev=rev,g=g,kind=kind,data_sha256=sha(data),prev=old['digest'])))
    c.execute('INSERT OR ROLLBACK INTO ops VALUES(?,?,?,?,?,?)',(rev,g,kind,data,old['digest'],digest))
    cursor=c.execute('UPDATE OR ROLLBACK head SET rev=?,g=?,digest=?,phase=?,publication=? WHERE id=1 AND rev=? AND digest=?',
        (rev,g,digest,phase,blob(publication,8192),old['rev'],old['digest']))
    if cursor.rowcount!=1:fail('STALE_CHECKPOINT','expected SQL head changed')
    return dict(rev=rev,g=g,digest=digest,phase=phase,publication=publication)


@contextmanager
def transaction(c, root):
    reserve(root)
    try:
        c.execute('BEGIN IMMEDIATE');yield
        c.execute('COMMIT')
    except BaseException:
        if c.in_transaction:c.execute('ROLLBACK')
        raise


def content_sizes(c):
    logical=0;nodes=0
    for table,columns in [('meta',['body','raw']),('runs',['binding','genesis']),('steps',['state','trace']),('events',['data']),('ops',['data']),('head',['publication'])]:
        for column in columns:
            for (value,) in c.execute('SELECT '+column+' FROM '+table):
                logical+=len(value)
                if not (table=='meta' and column=='raw'):nodes+=shape(decode(value,256*1024))
    return logical,nodes


def operational(c,root,kind,phase,pub,payload=None):
    old=head(c)
    logical,nodes=content_sizes(c)
    new_data=blob(dict(phase=phase,publication=pub,payload=payload),16384)
    if logical+len(new_data)+len(blob(pub,8192))>MAX_LOGICAL or nodes+shape(decode(new_data,16384))+shape(pub)>MAX_NODES:fail('RESOURCE_LIMIT','operational record reserve')
    with transaction(c,root):new=op(c,old,kind,phase,pub,payload)
    return new


def token(ident,h):
    return dict(schema='imc4-persist-token/v1',attempt_id=ident['attempt_id'],generation=h['g'],revision=h['rev'],
                head_sha256=h['digest'],input_sha256=ident['input_sha256'],native_code_sha256=ident['native_code_sha256'],context_sha256=ident['context_sha256'])


def step_digest(g,rid,tick,prev,state,trace,event_bytes):
    return sha(blob(dict(g=g,rid=rid,tick=tick,prev=prev,state_sha256=sha(state),trace_sha256=sha(trace),events_sha256=sha(event_bytes))))


def audit(c, ident):
    began=time.monotonic()
    try:
        schemas=[r[0] for r in c.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name")]
        if sorted(schemas)!=sorted(SQL):fail('CORRUPT','unexpected SQL schema')
        if c.execute("SELECT count(*) FROM sqlite_master WHERE type IN ('trigger','view') OR (type='index' AND sql IS NOT NULL)").fetchone()[0]:fail('CORRUPT','unexpected active schema')
        if c.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:fail('CORRUPT','SQLite quick_check failed')
        h=head(c)
        for table,limit in [('meta',1),('runs',18),('steps',1024),('events',7168),('ops',1152),('head',1)]:
            if c.execute('SELECT count(*) FROM '+table).fetchone()[0]>limit:fail('RESOURCE_LIMIT','table count cap')
        for table,columns in [('meta',{'body':32768,'raw':256*1024}),('runs',{'binding':256*1024,'genesis':8192}),('steps',{'state':8192,'trace':32768}),('events',{'data':2048}),('ops',{'data':16384}),('head',{'publication':8192})]:
            for column,limit in columns.items():
                maximum=c.execute('SELECT max(length('+column+')) FROM '+table).fetchone()[0]
                if maximum is not None and maximum>limit:fail('RESOURCE_LIMIT','SQL blob exceeds pre-decode bound')
        row=c.execute('SELECT body,raw FROM meta WHERE id=1').fetchone()
        if row is None:fail('INITIALIZATION_INCOMPLETE','missing meta')
        body,raw=row;meta=decode(body,32768)
        if type(raw) is not bytes or len(raw)>256*1024 or sha(raw)!=ident['input_sha256']:fail('IDENTITY_MISMATCH','raw input differs')
        plan=admitted(raw);bindings=[];states=[];deltas=[[] for _ in plan]
        rows=c.execute('SELECT rid,name,binding,genesis FROM runs ORDER BY rid').fetchall()
        if len(rows)!=len(plan):fail('CORRUPT','run count differs')
        logical=len(body)+len(raw)+len(blob(h['publication'],8192));nodes=shape(meta)+shape(h['publication']);genesis_parts=[]
        for rid,((s,e,p),row) in enumerate(zip(plan,rows)):
            i,name,binding,genesis=row;bound=decode(binding,256*1024);st=decode(genesis,8192)
            expected=dict(scenario=s,execution=e,policy=p)
            if type(i) is not int or i!=rid or name!=f"{s['id']}--{e['id']}--{p}" or bound!=expected or st!=qs.initialize_scenario(s,e,p):fail('CORRUPT','run/genesis binding differs')
            bindings.append(expected);states.append(st);genesis_parts.append(dict(rid=i,name=name,binding_sha256=sha(binding),genesis_sha256=sha(genesis)))
            logical+=len(binding)+len(genesis);nodes+=shape(bound)+shape(st)
        anchor=sha(blob(dict(meta_sha256=sha(body),raw_sha256=sha(raw),runs=genesis_parts),32768))
        if anchor!=ident['genesis_sha256'] or meta!=dict(schema='imc4-persist-meta/v1',input_sha256=ident['input_sha256'],native_code_sha256=ident['native_code_sha256'],runtime=ident['runtime'],profile=ident['profile'],context_sha256=ident['context_sha256'],output=str(Path(ident['path'])/'product')):
            fail('IDENTITY_MISMATCH','genesis/meta commitment mismatch')
        flat=[(rid,t) for rid,(s,e,p) in enumerate(plan) for t in range(len(s['steps']))]
        steps=c.execute('SELECT g,rid,tick,prev,digest,state,trace,events_sha FROM steps ORDER BY g').fetchall()
        if len(steps)!=h['g'] or not 0<=h['g']<=len(flat):fail('CORRUPT','head/step count mismatch')
        step_hashes=[];prev=anchor;event_count=0
        for index,row in enumerate(steps,1):
            g,rid,tick,prev_hash,digest,state_raw,trace_raw,events_sha=row
            if any(type(x) is not int for x in [g,rid,tick]) or g!=index or (rid,tick)!=flat[index-1] or prev_hash!=prev:fail('CORRUPT','step order/lineage')
            event_rows=c.execute('SELECT rid,tick,seq,fill_id,g,data FROM events WHERE g=? ORDER BY seq',(g,)).fetchall()
            if not 1<=len(event_rows)<=7:fail('CORRUPT','event block count')
            events=[];eb=b''
            for sequence,(ri,ti,se,fill_id,eg,data) in enumerate(event_rows):
                event=decode(data,2048)
                if any(type(x) is not int for x in [ri,ti,se,eg]) or (ri,ti,se,eg)!=(rid,tick,sequence,g) or fill_id!=event.get('fill_id') or (event.get('timestamp'),event.get('sequence'))!=(tick,sequence):fail('CORRUPT','event identity/order')
                events.append(event);eb+=data;nodes+=shape(event)
            if len(eb)>8192 or sha(eb)!=events_sha:fail('CORRUPT','event block commitment')
            st=decode(state_raw,8192);tr=decode(trace_raw,32768)
            if step_digest(g,rid,tick,prev,state_raw,trace_raw,eb)!=digest:fail('CORRUPT','step digest mismatch')
            s,e,p=plan[rid]
            if project(s,e,p,states[rid],tr,events)!=st:fail('INVALID_STATE','committed effect projection mismatch')
            states[rid]=st;deltas[rid].append(dict(trace=tr,events=events));prev=digest;step_hashes.append(digest)
            logical+=len(state_raw)+len(trace_raw)+len(eb);nodes+=shape(st)+shape(tr);event_count+=len(events)
            if logical>MAX_LOGICAL or nodes>MAX_NODES:fail('RESOURCE_LIMIT','journal logical/decoded cap')
        if event_count!=c.execute('SELECT count(*) FROM events').fetchone()[0]:fail('CORRUPT','orphan event rows')
        operations=c.execute('SELECT rev,g,kind,data,prev,digest FROM ops ORDER BY rev').fetchall()
        if len(operations)!=h['rev']+1:fail('CORRUPT','operation count mismatch')
        last='';generation=0;last_data=None;publications={};captures=set()
        for index,(rev,g,kind,data,prior,digest) in enumerate(operations):
            if type(rev) is not int or type(g) is not int or rev!=index or prior!=last:fail('CORRUPT','operation order/lineage')
            d=decode(data,16384)
            if set(d)!={'phase','publication','payload'} or d['phase'] not in PHASES:fail('CORRUPT','operation shape')
            validate_publication(d['publication'])
            if index==0:
                if kind!='GENESIS' or g!=0 or d!=dict(phase='RUNNING',publication=initial_publication(),payload=anchor):fail('CORRUPT','genesis op mismatch')
            elif kind=='STEP':
                if last_data['phase']!='RUNNING' or d['publication']!=last_data['publication']:fail('CORRUPT','step lifecycle mismatch')
                generation+=1
                if g!=generation or d['payload']!=step_hashes[g-1] or d['phase']!=('COMPUTE_COMPLETE' if g==len(flat) else 'RUNNING'):fail('CORRUPT','STEP operation mismatch')
            elif kind not in ('INTENT','OWNER','ARCHIVE_INTENT','ARCHIVED','COMPLETE','RECOVERED') or g!=generation:fail('CORRUPT','operational generation/kind mismatch')
            if index and kind!='STEP':
                previous_pub=last_data['publication'];current_pub=d['publication'];previous_phase=last_data['phase']
                if kind=='RECOVERED':
                    if d['phase']!=previous_phase or current_pub!=previous_pub:fail('CORRUPT','recovery lifecycle mismatch')
                    if not isinstance(d['payload'],str) or not re.fullmatch(r'forensics/generation-[12]',d['payload']):fail('CORRUPT','recovery capture path')
                    captures.add(d['payload'])
                elif kind=='INTENT':
                    expected_pub=dict(previous_pub,intent=True)
                    if previous_phase!='COMPUTE_COMPLETE' or d['phase']!='PUBLISHING' or current_pub!=expected_pub:fail('CORRUPT','intent lifecycle mismatch')
                elif kind=='OWNER':
                    expected_pub=dict(previous_pub,owner=current_pub['owner'])
                    if previous_phase!='PUBLISHING' or d['phase']!='PUBLISHING' or previous_pub['owner'] is not None or current_pub['owner'] is None or current_pub!=expected_pub:fail('CORRUPT','owner lifecycle mismatch')
                elif kind=='ARCHIVE_INTENT':
                    if previous_phase!='PUBLISHING' or d['phase']!='ARCHIVING' or current_pub!=dict(previous_pub,archive=current_pub['archive']) or current_pub['archive'] is None:fail('CORRUPT','archive lifecycle mismatch')
                elif kind=='ARCHIVED':
                    expected_pub=initial_publication();expected_pub['generation']=previous_pub['generation']+1
                    if previous_phase!='ARCHIVING' or d['phase']!='COMPUTE_COMPLETE' or current_pub!=expected_pub:fail('CORRUPT','archived lifecycle mismatch')
                elif kind=='COMPLETE':
                    if previous_phase!='PUBLISHING' or d['phase']!='COMPLETE' or current_pub!=dict(previous_pub,receipt_sha256=current_pub['receipt_sha256']) or current_pub['receipt_sha256'] is None or current_pub['owner'] is None:fail('CORRUPT','completion lifecycle mismatch')
            expected=sha(blob(dict(rev=rev,g=g,kind=kind,data_sha256=sha(data),prev=prior)))
            if digest!=expected:fail('CORRUPT','operation digest mismatch')
            if d['publication']['owner'] is not None:publications[d['publication']['generation']]=d['publication']
            last=digest;last_data=d;logical+=len(data);nodes+=shape(d)
        if generation!=h['g'] or last!=h['digest'] or last_data['phase']!=h['phase'] or last_data['publication']!=h['publication']:fail('CORRUPT','head does not match operation chain')
        if logical>MAX_LOGICAL or nodes>MAX_NODES:fail('RESOURCE_LIMIT','journal logical/decoded cap')
        return dict(raw=raw,plan=plan,states=states,deltas=deltas,flat=flat,head=h,step_hashes=step_hashes,anchor=anchor,publications=publications,captures=captures,
                    metrics=dict(prefix_steps=len(steps),prefix_events=event_count,logical_bytes=logical,decoded_nodes=nodes,audit_seconds=time.monotonic()-began))
    except (InputError,KeyError,IndexError,TypeError,ValueError) as exc:
        if isinstance(exc,ResumeError):raise
        fail('INVALID_STATE',str(exc))
