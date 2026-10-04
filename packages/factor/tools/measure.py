"""Bounded installed-runtime baseline; each child process has a 60s timeout."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def child(args):
    from factor_research.runs import run_trial
    start=time.perf_counter()
    result=run_trial(args.panel,args.config,args.store,args.run_id)
    elapsed=time.perf_counter()-start
    if result['status']!='SUCCESS':raise RuntimeError('benchmark trial did not succeed')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report_path=Path(result['directory'])/'report.json'
    data=json.loads(report_path.read_text())
    finite=sum(v is not None for rows in data['feature_rows'].values() for row in rows for v in row['values'].values())
    print(json.dumps(dict(elapsed_seconds=elapsed,peak_rss_bytes=peak if sys.platform=='darwin' else peak*1024,
                          feature_cells=sum(len(row['values']) for rows in data['feature_rows'].values() for row in rows),
                          nonnull_feature_values=finite,report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                          report_bytes=report_path.stat().st_size,trial_bytes=sum(p.stat().st_size for p in report_path.parent.iterdir()),
                          input_rows=data['observation_rows'],null_prices=data['null_prices'],delayed_rows=data['delayed_rows'],
                          identity=result['identity'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--child',action='store_true')
    for name in ['panel','config','store','run-id']:p.add_argument('--'+name)
    args=p.parse_args()
    if args.child:return child(args)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    data=dict(platform=platform.platform(),python=sys.version,executable=sys.executable,
              clock='time.perf_counter wall time around installed run_trial including read/hash/parse/evaluate/report/write/read-back',
              rss_method='fresh child process resource.getrusage(RUSAGE_SELF).ru_maxrss; process peak, not function allocations',
              cache_state='fresh Python process each repeat, OS file cache uncontrolled; first read versus later likely-warm repeats',
              synthetic=True,generator='deterministic formula; no random sampling/seed',optimization=False,workloads=[])
    for dates in [100,1000]:
        fixture=root/f'panel-{dates}';store=root/f'trials-{dates}'
        subprocess.run([sys.executable,'-m','factor_research','fixture',str(fixture),'--assets','100','--dates',str(dates),'--diagnostics'],check=True,capture_output=True,timeout=60)
        records=[]
        for repeat in range(3):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--panel',str(fixture/'panel.csv'),'--config',str(fixture/'config.json'),
                 '--store',str(store),'--run-id',f'repeat-{repeat}']
            result=subprocess.run(cmd,text=True,capture_output=True,timeout=60,check=True)
            records.append(dict(repeat=repeat,**json.loads(result.stdout)))
        assert len({r['report_sha256'] for r in records})==1,'identical input report drift'
        reference=subprocess.run([sys.executable,str(Path(__file__).with_name('reference_check.py')),str(store/'repeat-0')],text=True,capture_output=True,check=True,timeout=60)
        ref=json.loads(reference.stdout)
        workload=dict(assets=100,dates=dates,input_bytes=(fixture/'panel.csv').stat().st_size,
                      config_bytes=(fixture/'config.json').stat().st_size,repeats=records,reference=ref)
        data['workloads'].append(workload)
        (root/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(dates=dates,rows=100*dates,elapsed=[r['elapsed_seconds'] for r in records],reference=ref['status'])),flush=True)
    physical=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert physical<1024**3,'workload storage cap exceeded'
    data['physical_bytes_before_receipt_update']=physical
    (root/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
