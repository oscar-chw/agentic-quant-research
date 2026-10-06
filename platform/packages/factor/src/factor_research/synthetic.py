"""Deterministic synthetic trend regime, by default switching to alternation; no market dataset."""
import csv
from datetime import datetime,timedelta,timezone
import io
import math
from pathlib import Path
from .runs import json_bytes, write_once


def make_fixture(folder, assets=4, dates=24, diagnostics=False, persistent=False):
    if not 3<=assets<=200 or not 12<=dates<=5000 or assets*dates>100000:
        raise ValueError("synthetic fixture requires 3..200 assets, 12..5000 dates and <=100000 rows")
    names=[f"asset_{i:03d}" for i in range(assets)]
    times=[datetime(2024,1,1,16,tzinfo=timezone.utc)+timedelta(days=i) for i in range(dates)]
    calendar=[t.isoformat().replace('+00:00','Z') for t in times]
    cut=dates//3
    config=dict(schema_version="factor-config/v1",panel_schema="factor-panel/v1",assets=names,
                calendar=calendar,lookback=1,horizon=1,cost_bps=10,candidates=["momentum","reversal"],
                splits={name:dict(start=calendar[a],end=calendar[b]) for name,a,b in
                        [("train",0,cut-1),("validation",cut,2*cut-1),("test",2*cut,dates-1)]})
    text=io.StringIO(newline="")
    writer=csv.writer(text); writer.writerow(["time","asset","observed_at","available_at","price","group"])
    prices=[100.0]*assets
    for day,now in enumerate(times):
        for i,name in enumerate(names):
            scale=(i-(assets-1)/2)/assets
            trend=.012*scale
            # Feature return ends at t-1; label return ends at t+1. A two-up,
            # two-down regime reverses across that two-grid-step separation.
            # persistent=True keeps the trend in the test split: a positive control
            # in which a lagged momentum signal should survive out of sample.
            rate=trend+.0001*math.sin(day+i) if day<2*cut or persistent else trend*(1 if day%4<2 else -1)
            if day: prices[i]*=1+rate
            available=now+timedelta(days=2) if diagnostics and i==0 and day==cut+1 else now
            price="" if diagnostics and i==assets-1 and day==2*cut+2 else format(prices[i],'.12g')
            writer.writerow([calendar[day],name,calendar[day],available.isoformat().replace('+00:00','Z'),price,f"group_{i%2}"])
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=False)
    write_once(folder/"panel.csv",text.getvalue().encode())
    write_once(folder/"config.json",json_bytes(config))
    return dict(directory=str(folder),rows=assets*dates,assets=assets,dates=dates,diagnostics=diagnostics,persistent=persistent,
                generator="deterministic formula, no random sampling",seed=None)
