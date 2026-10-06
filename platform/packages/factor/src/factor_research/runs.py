"""Append-only-by-application trial directories with a final completion manifest."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from . import __version__
from .artifacts import json_bytes, read_bounded, sha, sync_directory, write_once
from .contracts import read_config, read_panel
from .evaluator import evaluate
from .prepare import load_prepared, validate_prepared, SOURCE_MEMBERS
from .reports import human_report


def code_identity():
    package = Path(__file__).parent
    files = {p.name:sha(p.read_bytes()) for p in sorted(package.glob("*.py"))}
    return dict(code_sha256=sha(json_bytes(files)), code_files=files, version=__version__)


def report_for(panel_raw, config_raw, identity, source_files=None):
    config=read_config(config_raw)
    report=evaluate(read_panel(panel_raw,config),config)
    report["provenance"]=identity
    if source_files is not None:
        report["schema_version"]="factor-report/v2"
        report["data_source"]=json.loads(source_files["import-manifest.json"])["data_source"]
    return report


def verify_run(folder, recompute=False):
    folder=Path(folder)
    final=folder/"completion.json"
    if not final.is_file():
        raise ValueError("INCOMPLETE trial: completion manifest absent")
    if folder.is_symlink() or any(p.is_symlink() for p in folder.iterdir()):
        raise ValueError("symlink in trial directory")
    completion=json.loads(final.read_bytes())
    schema=completion.get("schema_version")
    imported=schema=="factor-completion/v2"
    if schema not in ("factor-completion/v1","factor-completion/v2") or completion.get("status") not in ("SUCCESS","FAILED"):
        raise ValueError("invalid completion manifest")
    files=completion["files"]
    expected={"registration.json","panel.csv","config.json"}
    if imported:
        expected |= SOURCE_MEMBERS
    expected |= {"report.json","report.md"} if completion["status"]=="SUCCESS" else {"failure.json"}
    if not expected<=set(files) or set(files)-expected-{"report.json","report.md"}:
        raise ValueError("unexpected manifest members")
    if {p.name for p in folder.iterdir()} != set(files)|{"completion.json"}:
        raise ValueError("unexpected trial directory contents")
    for name, digest in files.items():
        if Path(name).name != name or sha((folder/name).read_bytes()) != digest:
            raise ValueError("artifact hash mismatch: "+name)
    registration=json.loads((folder/"registration.json").read_bytes())
    if registration.get("schema_version") != ("factor-trial/v2" if imported else "factor-trial/v1"):
        raise ValueError("trial/completion schema mismatch")
    identity=registration["identity"]
    if (files["panel.csv"]!=identity["input_sha256"] or files["config.json"]!=identity["config_sha256"]
            or sha(json_bytes(registration["code_files"]))!=identity["code_sha256"]):
        raise ValueError("input/config/code identity mismatch")
    source_files=None
    if imported:
        source_files={name:(folder/name).read_bytes() for name in SOURCE_MEMBERS|{"panel.csv","config.json"}}
        manifest=validate_prepared(source_files)
        if identity.get("import_sha256") != sha(source_files["import-manifest.json"]):
            raise ValueError("import provenance identity mismatch")
    elif "import_sha256" in identity:
        raise ValueError("import identity requires v2 source snapshots")
    if completion["status"]=="SUCCESS":
        report=json.loads((folder/"report.json").read_bytes())
        if report["provenance"]!=identity or report["schema_version"]!=("factor-report/v2" if imported else "factor-report/v1"):
            raise ValueError("report provenance mismatch")
        if imported and report.get("data_source") != manifest["data_source"]:
            raise ValueError("report availability assumption mismatch")
        if recompute:
            if code_identity()["code_sha256"]!=identity["code_sha256"]:
                raise ValueError("recompute requires the original installed code hash")
            actual=report_for((folder/"panel.csv").read_bytes(),(folder/"config.json").read_bytes(),identity,source_files)
            if json_bytes(actual)!=(folder/"report.json").read_bytes():
                raise ValueError("report recomputation differs")
    return dict(run_id=registration["run_id"],status=completion["status"],
                directory=str(folder),identity=identity,recomputed=recompute and completion["status"]=="SUCCESS")


def run_trial(panel_path, config_path, store, run_id, prepared=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",run_id):
        raise ValueError("run ID must be 1..64 safe filename characters")
    source_files=None
    if prepared is not None:
        if panel_path is not None or config_path is not None:
            raise ValueError("use --prepared alone or --panel with --config")
        source_files=load_prepared(prepared)
        panel_raw,config_raw=source_files["panel.csv"],source_files["config.json"]
    else:
        if panel_path is None or config_path is None:
            raise ValueError("use --prepared or provide both --panel and --config")
        if any((Path(p).resolve().parent/"import-manifest.json").exists() for p in (panel_path,config_path)):
            raise ValueError("prepared files require --prepared to retain raw source and availability assumptions")
        panel_raw,config_raw=(read_bounded(Path(p)) for p in (panel_path,config_path))
    code=code_identity()
    identity=dict(input_sha256=sha(panel_raw),config_sha256=sha(config_raw),
                  code_sha256=code["code_sha256"],version=code["version"])
    if source_files is not None:
        identity["import_sha256"]=sha(source_files["import-manifest.json"])
    store=Path(store); store.mkdir(parents=True,exist_ok=True)
    folder=store/run_id
    try: folder.mkdir()
    except FileExistsError:
        result=verify_run(folder)
        if result["identity"]!=identity:
            raise ValueError("conflicting run ID; existing trial preserved")
        return dict(result,reused=True)
    # Reserve identity before parsing. Interruption leaves a visible incomplete trial.
    registration=dict(schema_version="factor-trial/v2" if source_files is not None else "factor-trial/v1",run_id=run_id,
                      created_at=datetime.now(timezone.utc).isoformat(),identity=identity,code_files=code["code_files"])
    write_once(folder/"registration.json",json_bytes(registration))
    write_once(folder/"panel.csv",panel_raw)
    write_once(folder/"config.json",config_raw)
    if source_files is not None:
        for name in sorted(SOURCE_MEMBERS):
            write_once(folder/name,source_files[name])
    started=time.perf_counter()
    status="SUCCESS"
    try:
        report=report_for(panel_raw,config_raw,identity,source_files)
        write_once(folder/"report.json",json_bytes(report))
        write_once(folder/"report.md",human_report(report).encode())
    except Exception as exc:
        status="FAILED"
        write_once(folder/"failure.json",json_bytes(dict(error_type=type(exc).__name__,message=str(exc))))
    files={p.name:sha(p.read_bytes()) for p in sorted(folder.iterdir())}
    write_once(folder/"completion.json",json_bytes(dict(schema_version="factor-completion/v2" if source_files is not None else "factor-completion/v1",status=status,
                                                        elapsed_seconds=time.perf_counter()-started,files=files)))
    sync_directory(folder); sync_directory(store)
    return dict(verify_run(folder),reused=False)


def list_trials(store):
    rows=[]
    for folder in sorted(Path(store).iterdir()):
        if folder.is_dir():
            try: rows.append(verify_run(folder))
            except (ValueError,KeyError,OSError,TypeError) as exc:
                rows.append(dict(run_id=folder.name,status="INCOMPLETE" if not (folder/"completion.json").exists() else "CORRUPT",reason=str(exc)))
    return rows
