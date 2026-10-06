import argparse
import json
import sys
from .runs import list_trials, run_trial, verify_run
from .synthetic import make_fixture
from .prepare import prepare_prices
from .ohlcv import write_ohlcv_import
from .allocation_workflow import run_allocation, verify_allocation


def main(argv=None):
    parser=argparse.ArgumentParser(description="Offline factor research; no platform or live execution")
    commands=parser.add_subparsers(dest="command",required=True)
    allocate=commands.add_parser("allocate",help="compare constrained score allocation with rank weighting on a verified research run")
    allocate.add_argument("--research-run",required=True); allocate.add_argument("--scenario",required=True); allocate.add_argument("--out",required=True)
    allocation_verify=commands.add_parser("verify-allocation",help="verify and recompute a saved allocation comparison")
    allocation_verify.add_argument("directory")
    prepare=commands.add_parser("prepare",help="import ordinary price CSV under an explicit clock/universe/split contract")
    prepare.add_argument("--prices",required=True); prepare.add_argument("--contract",required=True)
    prepare.add_argument("--out",required=True)
    ohlcv=commands.add_parser("from-ohlcv",help="write a price CSV and import contract from a directory of daily OHLCV CSVs")
    ohlcv.add_argument("--dir",required=True); ohlcv.add_argument("--experiment",required=True); ohlcv.add_argument("--out",required=True)
    fixture=commands.add_parser("fixture",help="create a new deterministic synthetic fixture directory")
    fixture.add_argument("directory"); fixture.add_argument("--assets",type=int,default=4)
    fixture.add_argument("--dates",type=int,default=24); fixture.add_argument("--diagnostics",action="store_true")
    fixture.add_argument("--persistent",action="store_true",help="keep the trend regime through the test split (positive control)")
    run=commands.add_parser("run",help="register and evaluate a trial; preserve failed/incomplete IDs")
    run.add_argument("--panel"); run.add_argument("--config"); run.add_argument("--prepared")
    run.add_argument("--store",required=True); run.add_argument("--run-id",required=True)
    verify=commands.add_parser("verify",help="verify immutable artifacts; optionally recompute with original code")
    verify.add_argument("directory"); verify.add_argument("--recompute",action="store_true")
    listing=commands.add_parser("trials"); listing.add_argument("store")
    args=parser.parse_args(argv)
    try:
        if args.command=="allocate": result=run_allocation(args.research_run,args.scenario,args.out)
        elif args.command=="verify-allocation": result=verify_allocation(args.directory)
        elif args.command=="from-ohlcv": result=write_ohlcv_import(args.dir,args.experiment,args.out)
        elif args.command=="prepare": result=prepare_prices(args.prices,args.contract,args.out)
        elif args.command=="fixture": result=make_fixture(args.directory,args.assets,args.dates,args.diagnostics,args.persistent)
        elif args.command=="run": result=run_trial(args.panel,args.config,args.store,args.run_id,prepared=args.prepared)
        elif args.command=="verify": result=verify_run(args.directory,args.recompute)
        else: result=list_trials(args.store)
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
        return 2 if isinstance(result,dict) and result.get("status")=="FAILED" else 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status="REFUSED",error=str(exc))),file=sys.stderr)
        return 2


if __name__=="__main__": sys.exit(main())
