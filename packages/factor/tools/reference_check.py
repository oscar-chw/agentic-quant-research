"""Independent exhaustive oracle over a saved run; imports no factor-lab code.

Ranks count pairwise less/equal values (O(N^2)); portfolio uses explicit shares
and entry/exit dollar cash flows rather than the evaluator's return formula.
"""
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys


def check(folder):
    folder=Path(folder)
    cfg=json.loads((folder/'config.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    dt=lambda s:datetime.fromisoformat(s.replace('Z','+00:00')).astimezone(timezone.utc)
    times=[dt(t) for t in cfg['calendar']]; assets=cfg['assets']; n=len(assets)
    with (folder/'panel.csv').open(newline='') as f:
        panel={(dt(r['time']),r['asset']):r for r in csv.DictReader(f)}
    checks=0; max_delta=0.0
    def equal(actual,expected):
        nonlocal checks,max_delta
        checks+=1
        if expected is None:
            assert actual is None,(actual,expected)
        elif isinstance(expected,(float,int)):
            assert actual is not None
            delta=abs(actual-expected); max_delta=max(delta,max_delta)
            assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-11),(actual,expected)
        else: assert actual==expected,(actual,expected)
    def price(i,a,decision):
        r=panel.get((times[i],a))
        return float(r['price']) if r and r['price'] and dt(r['available_at'])<=decision else None
    def ranks(xs):
        return [1+sum(y<x for y in xs)+(sum(y==x for y in xs)-1)/2 for x in xs]
    def corr(xs,ys):
        pairs=[(x,y) for x,y in zip(xs,ys) if x is not None and y is not None]
        if len(pairs)<2:return None
        rx=ranks([p[0] for p in pairs]); ry=ranks([p[1] for p in pairs])
        mx=sum(rx)/len(rx); my=sum(ry)/len(ry)
        vx=sum((x-mx)**2 for x in rx); vy=sum((y-my)**2 for y in ry)
        return sum((x-mx)*(y-my) for x,y in zip(rx,ry))/math.sqrt(vx*vy) if vx and vy else None
    features={}
    for factor in cfg['candidates']:
        features[factor]=[]
        for i,t in enumerate(times):
            row={}
            for a in assets:
                value=None
                if i>=cfg['lookback']+1:
                    p1=price(i-1,a,t); p0=price(i-1-cfg['lookback'],a,t)
                    if p1 is not None and p0 is not None:
                        value=(p1-p0)/p0 * (1 if factor=='momentum' else -1)
                row[a]=value
                equal(report['feature_rows'][factor][i]['values'][a],value)
            features[factor].append(row)
    summary_refs={}
    daily_count=0
    parts=[(f,s,p) for f,sets in report['development'].items() for s,p in sets.items()]
    if report['test'] is not None:parts.append((report['selection']['selected'],'test',report['test']))
    for f,s,part in parts:
        lo=times.index(dt(cfg['splits'][s]['start'])); hi=times.index(dt(cfg['splits'][s]['end']))
        equal(len(part['daily']),hi-lo)
        ics=[]; nets=[]; grosses=[]; turnovers=[]; costs=[]; cover=[]; fcov=[]; active=missing=0
        for i,d in zip(range(lo,hi),part['daily']):
            daily_count+=1; signal=features[f][i]
            equal(dt(d['time']),times[i]); equal(dt(d['label_time']),times[i+1])
            entry={a:price(i,a,times[i]) for a in assets}
            exit={a:price(i+1,a,times[i+1]) for a in assets}
            returns={a:(exit[a]-entry[a])/entry[a] if entry[a] is not None and exit[a] is not None else None for a in assets}
            ic=corr(list(signal.values()),list(returns.values())); equal(d['rank_ic'],ic)
            if ic is not None:ics.append(ic)
            fcount=sum(v is not None for v in signal.values())
            pairs=sum(signal[a] is not None and returns[a] is not None for a in assets)
            equal(d['feature_count'],fcount); equal(d['ic_pair_count'],pairs)
            equal(d['feature_coverage'],fcount/n); equal(d['ic_coverage'],pairs/n)
            fcov.append(fcount/n);cover.append(pairs/n)
            eligible=[a for a in assets if signal[a] is not None and entry[a] is not None]
            equal(d['entry_eligible_count'],len(eligible))
            rk=ranks([signal[a] for a in eligible]); centered=[r-(len(rk)+1)/2 for r in rk]
            total=sum(abs(r) for r in centered)
            weights={a:0.0 for a in assets}
            if total: weights.update({a:r/total for a,r in zip(eligible,centered)})
            for a in assets:equal(d['weights'][a],weights[a])
            held=[a for a in assets if weights[a]]
            if any(exit[a] is None for a in held):
                equal(d['portfolio']['status'],'MISSING_EXIT_MARK');missing+=1
                for name in ['gross','net','turnover','cost']:equal(d['portfolio'][name],None)
            else:
                shares={a:weights[a]/entry[a] for a in held}
                gross=sum(shares[a]*(exit[a]-entry[a]) for a in held)
                turnover=sum(abs(shares[a])*entry[a]+abs(shares[a])*exit[a] for a in held)
                cost=turnover*cfg['cost_bps']/10000; net=gross-cost
                for name,value in [('gross',gross),('net',net),('turnover',turnover),('cost',cost)]:equal(d['portfolio'][name],value)
                equal(d['portfolio']['status'],'ACTIVE' if held else 'ABSTAIN')
                active+=bool(held); grosses.append(gross);nets.append(net);turnovers.append(turnover);costs.append(cost)
        mean=lambda xs:sum(xs)/len(xs) if xs else None
        refs=dict(intervals=hi-lo,valid_ic_dates=len(ics),rank_ic_mean=mean(ics),feature_coverage_mean=mean(fcov),
                  ic_coverage_mean=mean(cover),active_intervals=active,missing_exit_intervals=missing,
                  portfolio_complete=missing==0,gross_mean=mean(grosses) if not missing else None,
                  net_mean=mean(nets) if not missing else None,turnover_mean=mean(turnovers) if not missing else None,
                  cost_mean=mean(costs) if not missing else None)
        for k,v in refs.items():equal(part['summary'][k],v)
        summary_refs[(f,s)]=refs
    scored=[f for f in cfg['candidates'] if summary_refs[(f,'validation')]['rank_ic_mean'] is not None]
    winner=max(scored,key=lambda f:summary_refs[(f,'validation')]['rank_ic_mean']) if scored else None
    equal(report['selection']['selected'],winner)
    return dict(status='PASS_INDEPENDENT_NUMERICAL_REFERENCE',checks=checks,checked_feature_cells=len(times)*n*len(cfg['candidates']),
                checked_daily_evaluations=daily_count,max_absolute_delta=max_delta,
                method='pairwise rank counts; explicit share/cash-flow portfolio; no factor_research imports')


if __name__=='__main__':print(json.dumps(check(sys.argv[1]),indent=2))
