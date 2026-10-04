"""Source-bound consumption of native IMC persistence; no second simulation ledger."""

from __future__ import annotations

import fcntl
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path

from qrae.artifacts import canonical_json_bytes, sha256_bytes
from qrae.path_safety import has_link_or_reparse_component
from qrae.quote_features import strict_json
from qrae.quote_workflow import read_input

from quantos_showcase import execution

CAPS = {
    "wrapper.lock": 0,
    "request.json": 65536,
    "scenario.json": 262144,
    "source.md": 262144,
    "registration.json": 65536,
    "REPORT.md": 1048576,
    "complete.json": 65536,
}
MAX_METADATA = 4 * 1024 * 1024
SNAPSHOTS = {"request.json", "scenario.json", "source.md", "registration.json"}
TOKEN_FIELDS = {
    "schema",
    "attempt_id",
    "generation",
    "revision",
    "head_sha256",
    "input_sha256",
    "native_code_sha256",
    "context_sha256",
}


class PersistError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


def fail(code, message):
    raise PersistError(code, message)


def native():
    execution.native()
    from imc4_analysis import persistence

    return persistence


def code_identity():
    return sha256_bytes(
        canonical_json_bytes(
            {
                "persisted": sha256_bytes(Path(__file__).read_bytes()),
                "shared_execution": execution.implementation_identity(),
            }
        )
    )


def folder(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9_-]{0,63}", run_id
    ):
        fail(
            "IDENTITY_MISMATCH",
            "persisted ID must be lowercase, starting with a letter",
        )
    path = Path(root).absolute() / "persisted/runs" / run_id
    if has_link_or_reparse_component(path):
        fail("OWNERSHIP_AMBIGUOUS", "unlinked wrapper path required")
    return path


def identity(path):
    value = path.lstat()
    return {
        "dev": value.st_dev,
        "ino": value.st_ino,
        "birthtime": str(getattr(value, "st_birthtime", None)),
    }


def layout(path):
    """Inspect only the fixed wrapper grammar, never recurse through native state."""
    if has_link_or_reparse_component(path) or not path.is_dir():
        fail("OWNERSHIP_AMBIGUOUS", "regular unlinked wrapper directory required")
    names, total = set(), 0
    with os.scandir(path) as entries:
        for entry in entries:
            name = entry.name
            if name not in CAPS and name != "native":
                fail("OWNERSHIP_AMBIGUOUS", "unexpected wrapper member: " + name)
            names.add(name)
            value = entry.stat(follow_symlinks=False)
            if name == "native":
                if not stat.S_ISDIR(value.st_mode):
                    fail(
                        "OWNERSHIP_AMBIGUOUS",
                        "native store must be an unlinked directory",
                    )
                with os.scandir(path / name) as attempts:
                    for attempt in attempts:
                        if attempt.name != path.name or not attempt.is_dir(
                            follow_symlinks=False
                        ):
                            fail(
                                "OWNERSHIP_AMBIGUOUS", "unexpected native-store member"
                            )
            else:
                if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                    fail(
                        "OWNERSHIP_AMBIGUOUS",
                        "wrapper metadata must be regular and single-link",
                    )
                if value.st_size > CAPS[name]:
                    fail("RESOURCE_LIMIT", "wrapper file cap: " + name)
                total += value.st_size
    if total > MAX_METADATA:
        fail("RESOURCE_LIMIT", "wrapper metadata exceeds 4 MiB")
    return names, total


@contextmanager
def locked(path, *, shared=False):
    names, _ = layout(path)
    if "wrapper.lock" not in names:
        fail("INITIALIZATION_INCOMPLETE", "missing wrapper lifetime lock")
    lock = path / "wrapper.lock"
    fd = os.open(lock, (os.O_RDONLY if shared else os.O_RDWR) | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(
                fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
            )
        except BlockingIOError:
            fail("BUSY", "another wrapper invocation owns this attempt")
        opened, current = os.fstat(fd), lock.lstat()
        if (opened.st_dev, opened.st_ino) != (
            current.st_dev,
            current.st_ino,
        ) or opened.st_nlink != 1:
            fail("OWNERSHIP_AMBIGUOUS", "wrapper lock replaced or aliased")
        layout(path)
        yield
    finally:
        os.close(fd)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, name, raw):
    _, total = layout(path)
    if name not in CAPS or len(raw) > CAPS[name] or total + len(raw) > MAX_METADATA:
        fail("RESOURCE_LIMIT", "wrapper publication reserve exceeded")
    fd = os.open(
        path / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path)


def registration(path, request, scenario, note):
    return {
        "schema": "quantos-persist-registration/v1",
        "method": execution.METHOD,
        "run_id": path.name,
        "path": str(path),
        "directory": identity(path),
        "lock": identity(path / "wrapper.lock"),
        "native_store": identity(path / "native"),
        "product": f"native/{path.name}/product",
        "wrapper_sha256": code_identity(),
        "native_version": execution.IMC_VERSION,
        "native_code_sha256": execution.IMC_CODE,
        "request_sha256": sha256_bytes(request),
        "scenario_sha256": sha256_bytes(scenario),
        "source_sha256": sha256_bytes(note),
        "scope": "Local source context before native execution; no authenticity or external preregistration proof",
    }


def load_context(path):
    native()
    names, _ = layout(path)
    if not SNAPSHOTS | {"native", "wrapper.lock"} <= names:
        fail(
            "INITIALIZATION_INCOMPLETE",
            "registration snapshots are incomplete; preserve attempt",
        )
    data = {name: read_input(path / name, CAPS[name]) for name in SNAPSHOTS}
    request = execution.validate_request(strict_json(data["request.json"]))
    note = execution.read_source(request, path, frozen=True)
    expected = registration(path, data["request.json"], data["scenario.json"], note)
    if canonical_json_bytes(expected) != data["registration.json"]:
        fail(
            "CONTEXT_MISMATCH", "frozen input, source, code, path or ownership differs"
        )
    return {
        "record": expected,
        "digest": sha256_bytes(data["registration.json"]),
        "request": request,
        "data": data,
    }


def token_envelope(path, context, token):
    return {
        "schema": "quantos-persist-checkpoint/v1",
        "run_id": path.name,
        "context_sha256": context["digest"],
        "token": token,
    }


def validate_token(value, path, context):
    if isinstance(value, dict) and value.get("schema") == "quantos-persist-result/v1":
        value = value.get("checkpoint")
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "run_id", "context_sha256", "token"}
        or value["schema"] != "quantos-persist-checkpoint/v1"
        or value["run_id"] != path.name
        or value["context_sha256"] != context["digest"]
    ):
        fail("CONTEXT_MISMATCH", "typed wrapper checkpoint required")
    token = value["token"]
    if (
        not isinstance(token, dict)
        or set(token) != TOKEN_FIELDS
        or token["schema"] != "imc4-persist-token/v1"
        or token["attempt_id"] != path.name
    ):
        fail("INVALID_TOKEN", "exact native token schema required")
    for name in ("generation", "revision"):
        if type(token[name]) is not int or not 0 <= token[name] < 2**63:
            fail("INVALID_TOKEN", "bounded native token counter required")
    for name in ("head_sha256", "input_sha256", "native_code_sha256", "context_sha256"):
        if not isinstance(token[name], str) or not re.fullmatch(
            r"[0-9a-f]{64}", token[name]
        ):
            fail("INVALID_TOKEN", "native token digest required")
    if (
        token["input_sha256"] != context["record"]["scenario_sha256"]
        or token["native_code_sha256"] != execution.IMC_CODE
        or token["context_sha256"] != context["digest"]
    ):
        fail("CONTEXT_MISMATCH", "native token input/code/context differs")
    return token


def product_summary(path, context):
    product = path / context["record"]["product"]
    completion = read_input(product / "complete.json", 65536)
    comparison_raw = read_input(product / "comparison.json", 262144)
    comparison = strict_json(comparison_raw)
    receipt = strict_json(completion)
    if (
        comparison["source_sha256"] != context["record"]["scenario_sha256"]
        or comparison["code_identity"]["sha256"] != execution.IMC_CODE
        or receipt["files"]["comparison.json"]
        != {"sha256": sha256_bytes(comparison_raw), "bytes": len(comparison_raw)}
    ):
        fail("CONTEXT_MISMATCH", "native product differs from registered scenario/code")
    return comparison, completion


def expected_publication(path, context, result):
    comparison, native_complete = product_summary(path, context)
    report = execution.render(
        context["request"],
        comparison,
        product=context["record"]["product"],
        persisted=True,
    ).encode()
    complete = canonical_json_bytes(
        {
            "schema": "quantos-persist-complete/v1",
            "registration_sha256": context["digest"],
            "native_token": result["token"],
            "native_complete_sha256": sha256_bytes(native_complete),
            "product": context["record"]["product"],
            "report_sha256": sha256_bytes(report),
            "snapshots": {
                name: sha256_bytes(raw) for name, raw in sorted(context["data"].items())
            },
        }
    )
    if len(report) > CAPS["REPORT.md"] or len(complete) > CAPS["complete.json"]:
        fail("RESOURCE_LIMIT", "wrapper finalization size exceeds cap")
    return {"REPORT.md": report, "complete.json": complete}, comparison


def publication_state(path, expected):
    names, _ = layout(path)
    if "complete.json" in names and "REPORT.md" not in names:
        fail(
            "WRAPPER_PUBLICATION_AMBIGUOUS", "receipt without report is not completion"
        )
    for name, raw in expected.items():
        if name in names and read_input(path / name, CAPS[name]) != raw:
            fail(
                "WRAPPER_PUBLICATION_AMBIGUOUS",
                "partial or different " + name + " is preserved",
            )
    return "complete.json" in names


def finalize(path, context, result):
    """Reconcile only missing files or byte-exact complete files, never partial bytes."""
    expected, _ = expected_publication(path, context, result)
    if publication_state(path, expected):
        fail("ALREADY_COMPLETED", "wrapper completion already exists; verify only")
    for name, raw in expected.items():
        if not (path / name).exists():
            write_new(path, name, raw)


def view(path, context, result, *, verification="NATIVE_STATE"):
    state, rows = result["status"], None
    if state == "COMPLETE":
        expected, comparison = expected_publication(path, context, result)
        state = (
            "COMPLETED"
            if publication_state(path, expected)
            else "NATIVE_COMPLETE_WRAPPER_INCOMPLETE"
        )
        rows = comparison["rows"]
    elif any((path / name).exists() for name in ("REPORT.md", "complete.json")):
        fail(
            "WRAPPER_PUBLICATION_AMBIGUOUS",
            "wrapper publication precedes native completion",
        )
    answer = {
        "schema": "quantos-persist-result/v1",
        "execution": state,
        "run_id": path.name,
        "directory": str(path),
        "product": str(path / context["record"]["product"]),
        "checkpoint": token_envelope(path, context, result["token"]),
        "native": result,
        "verification": verification,
        "rows": rows,
        "lifecycle": "native-committed-suffix/v1",
    }
    if state == "COMPLETED":
        answer["report"] = str(path / "REPORT.md")
    return answer


def observe(path, context, *, verify_mode=None):
    api = native()
    result = api.checkpoint(
        path / "native", path.name, context_sha256=context["digest"]
    )
    if result["status"] == "COMPLETE" or verify_mode is not None:
        verified = api.verify(
            path / "native",
            path.name,
            context_sha256=context["digest"],
            mode=verify_mode or "state",
        )
        if result["token"] != verified["token"]:
            fail("BUSY", "native head changed during verification")
    return result


def run(request_path, scenario_path, root, run_id, *, stop_after_commits=None):
    api = native()
    path = folder(root, run_id)
    if path.exists():
        fail("EXISTS", "wrapper path exists; never reset or migrate it")
    request_raw = read_input(request_path, 65536)
    request = execution.validate_request(strict_json(request_raw))
    note = execution.read_source(request, request_path.absolute().parent)
    raw = read_input(scenario_path, execution.MAX_INPUT)
    execution.admit(raw)  # Initial feasibility only; never called by continuation.
    budget(stop_after_commits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    fd = os.open(
        path / "wrapper.lock",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    os.close(fd)
    sync_directory(path)
    with locked(path):
        (path / "native").mkdir()
        for name, data in (
            ("request.json", request_raw),
            ("scenario.json", raw),
            ("source.md", note),
        ):
            write_new(path, name, data)
        write_new(
            path,
            "registration.json",
            canonical_json_bytes(registration(path, request_raw, raw, note)),
        )
        context = load_context(path)
        result = api.run(
            raw,
            path / "native",
            run_id,
            context_sha256=context["digest"],
            stop_after_commits=stop_after_commits,
        )
        if result["status"] == "COMPLETE":
            verified = api.verify(
                path / "native", run_id, context_sha256=context["digest"], mode="state"
            )
            if verified["token"] != result["token"]:
                fail("BUSY", "native head changed before finalization")
            finalize(path, context, result)
        return view(path, context, result)


def checkpoint(root, run_id):
    path = folder(root, run_id)
    with locked(path, shared=True):
        context = load_context(path)
        return view(path, context, observe(path, context))


def resume(root, run_id, expected_token, *, stop_after_commits=None):
    budget(stop_after_commits)
    path = folder(root, run_id)
    with locked(path):
        context = load_context(path)
        token = validate_token(expected_token, path, context)
        api = native()
        try:
            current = observe(path, context)
        except api.ResumeError as exc:
            if exc.code != "RECOVERY_REQUIRED" or any(
                (path / n).exists() for n in ("REPORT.md", "complete.json")
            ):
                raise
            current = None  # Native owns capture/recovery and stale-token validation.
        if current is not None:
            if token != current["token"]:
                fail("STALE_CHECKPOINT", "token differs from committed native head")
            current_view = view(path, context, current)
            if current_view["execution"] == "COMPLETED":
                fail("ALREADY_COMPLETED", "completed wrapper is unchanged; verify only")
            if current["status"] == "COMPLETE":
                finalize(path, context, current)
                return view(path, context, current)
        result = api.resume(
            path / "native",
            run_id,
            token,
            context_sha256=context["digest"],
            stop_after_commits=stop_after_commits,
        )
        if result["status"] == "COMPLETE":
            verified = api.verify(
                path / "native", run_id, context_sha256=context["digest"], mode="state"
            )
            if verified["token"] != result["token"]:
                fail("BUSY", "native head changed before finalization")
            finalize(path, context, result)
        return view(path, context, result)


def budget(value):
    if value is not None and (type(value) is not int or not 0 <= value <= 1024):
        fail("RESOURCE_LIMIT", "stop budget must be integer 0..1024")


def verify(root, run_id, *, mode="state", evidence=None):
    if mode not in ("state", "recompute"):
        fail("UNSUPPORTED", "verification mode must be state or recompute")
    path = folder(root, run_id)
    with locked(path, shared=True):
        context = load_context(path)
        result = view(
            path,
            context,
            observe(path, context, verify_mode=mode),
            verification="NATIVE_STATE" if mode == "state" else "NATIVE_RECOMPUTED",
        )
        if evidence is not None:
            for name in sorted(
                SNAPSHOTS
                | (
                    {"REPORT.md", "complete.json"}
                    if result["execution"] == "COMPLETED"
                    else set()
                )
            ):
                evidence.bind(path / name, "wrapper registration/completion")
            if result["execution"] == "COMPLETED":
                for name in ("complete.json", "comparison.json", "comparison.md"):
                    evidence.bind(
                        path / context["record"]["product"] / name,
                        "native product commitment",
                    )
        return result
