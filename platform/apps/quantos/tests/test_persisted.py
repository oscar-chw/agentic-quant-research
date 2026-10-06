"""Installed native consumer contracts, with one bounded real child process."""

import io
import json
import os
import re
import selectors
import sqlite3
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
from qrae.artifacts import canonical_json_bytes, sha256_bytes

from quantos_showcase import execution
from quantos_showcase import persisted as p
from quantos_showcase.run_adapters import Evidence, inspect_run
from quantos_showcase.run_index import build_index, verify_index

pytest.importorskip("imc4_analysis")
from imc4_analysis import cli  # noqa: E402
from imc4_analysis import persist_journal as journal
from imc4_analysis import persistence as native
from imc4_analysis import quote_study as qs

EXAMPLES = Path(__file__).parents[1] / "examples/execution"


def saved(root, *, times=True):
    return {
        str(f.relative_to(root)): (sha256_bytes(f.read_bytes()), f.stat().st_mtime_ns)
        if times
        else sha256_bytes(f.read_bytes())
        for f in root.rglob("*")
        if f.is_file() and not f.is_symlink()
    }


def record(name, value):
    if os.environ.get("QUANTOS_TEST_EVIDENCE"):
        dest = Path(os.environ["QUANTOS_TEST_EVIDENCE"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / (name + ".json")).write_bytes(canonical_json_bytes(value))


@contextmanager
def no_simulation(*, no_native_mutation=False):
    with ExitStack() as stack:
        names = [
            (qs, "advance_step"),
            (qs, "quote_orders"),
            (qs, "run_study"),
            (execution, "admit"),
        ]
        if no_native_mutation:
            names += [(native, "run"), (native, "resume")]
        for module, name in names:
            stack.enter_context(
                patch.object(
                    module, name, side_effect=AssertionError("forbidden call: " + name)
                )
            )
        yield


@pytest.fixture
def inputs(tmp_path):
    base = tmp_path / "input"
    base.mkdir()
    for name in ("request.json", "source.md"):
        (base / name).write_bytes((EXAMPLES / name).read_bytes())
    data = json.loads((EXAMPLES / "scenario.json").read_bytes())
    data["scenarios"] = data["scenarios"][:1]
    data["executions"] = data["executions"][:1]
    (base / "scenario.json").write_bytes(canonical_json_bytes(data))
    return base / "request.json", base / "scenario.json", tmp_path / "store"


def start(inputs, name="a", budget=0):
    return p.run(*inputs, name, stop_after_commits=budget)


def row(result, base):
    return inspect_run(
        {"id": "selected", "kind": "quantos-persisted", "path": result["directory"]},
        base,
    )


def assert_refusal_unchanged(inputs, result, code=None):
    root = Path(result["directory"])
    before = saved(root)
    with no_simulation(), pytest.raises(ValueError) as error:
        p.resume(inputs[2], root.name, result["checkpoint"])
    if code:
        assert getattr(error.value, "code", None) == code
    assert saved(root) == before
    return str(error.value)


def committed(result):
    db = Path(result["directory"]) / "native" / result["run_id"] / "journal.sqlite"
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
        g, st, trace = conn.execute(
            "SELECT g,state,trace FROM steps ORDER BY g DESC LIMIT 1"
        ).fetchone()
        return {"generation": g, "state": json.loads(st), "trace": json.loads(trace)}


def invoke(args):
    output = io.StringIO()
    with redirect_stdout(output):
        status = execution.main(list(map(str, args)))
    value = json.loads(output.getvalue())
    assert status == 0, value
    return value


def test_installed_cli_g67_g68_suffix_context_and_94_file_parity(tmp_path):
    started = time.monotonic()
    store = tmp_path / "walkthrough"
    request, scenario = EXAMPLES / "request.json", EXAMPLES / "scenario.json"
    observed = []

    def bound_call(name, original):
        def call(*args, **kwargs):
            # Every public native call sees already-frozen wrapper bytes.
            native_store = Path(args[1] if name == "run" else args[0])
            wrapper = native_store.parent
            reg = (wrapper / "registration.json").read_bytes()
            assert kwargs["context_sha256"] == sha256_bytes(reg)
            assert (wrapper / "request.json").read_bytes() == request.read_bytes()
            assert (wrapper / "source.md").read_bytes() == request.with_name(
                "source.md"
            ).read_bytes()
            if name == "run":
                assert not (native_store / "demo").exists()
            observed.append(
                {
                    "method": name,
                    "context": kwargs["context_sha256"],
                    "mode": kwargs.get("mode"),
                }
            )
            return original(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for name in ("run", "checkpoint", "resume", "verify"):
            stack.enter_context(
                patch.object(
                    native, name, side_effect=bound_call(name, getattr(native, name))
                )
            )
        initial = invoke(
            [
                "persist-run",
                "--request",
                request,
                "--scenario",
                scenario,
                "--store",
                store,
                "--run-id",
                "demo",
                "--stop-after",
                67,
            ]
        )
        assert initial["execution"] == "PAUSED_AT_BOUNDARY"
        before = saved(store)
        with no_simulation(no_native_mutation=True):
            checked = invoke(["checkpoint", "--store", store, "--run-id", "demo"])
        assert saved(store) == before
        assert checked["checkpoint"] == initial["checkpoint"]
        g67 = committed(checked)
        assert (
            g67["generation"],
            g67["state"]["cash"],
            g67["state"]["inventory"],
            g67["state"]["total_fees"],
        ) == (67, "801.8", 2, "0.2")
        live = {o["order_id"]: o for o in g67["state"]["live"]}
        assert live["inventory-2-sell"]["cancel_due"] == 4
        token_path = tmp_path / "token.json"
        token_path.write_bytes(canonical_json_bytes(checked["checkpoint"]))
        with (
            patch.object(
                execution, "admit", side_effect=AssertionError("initial admission")
            ),
            patch.object(qs, "advance_step", wraps=qs.advance_step) as advance,
            patch.object(qs, "quote_orders", wraps=qs.quote_orders) as policy,
        ):
            next_result = invoke(
                [
                    "resume",
                    "--store",
                    store,
                    "--run-id",
                    "demo",
                    "--token",
                    token_path,
                    "--stop-after",
                    1,
                ]
            )
            assert advance.call_count == policy.call_count == 1
            assert next_result["native"]["metrics"]["suffix_calls"] == [
                [g67["state"]["run"], 4]
            ]
        g68 = committed(next_result)
        assert (
            g68["generation"],
            g68["state"]["cash"],
            g68["state"]["inventory"],
            g68["state"]["total_fees"],
        ) == (68, "1100.5", -1, "0.5")
        assert [
            (o["order_id"], o["remaining_units"]) for o in g68["state"]["live"]
        ] == [("inventory-4-sell", 1)]
        token_path.write_bytes(canonical_json_bytes(next_result))
        with (
            patch.object(
                execution, "admit", side_effect=AssertionError("initial admission")
            ),
            patch.object(qs, "advance_step", wraps=qs.advance_step) as advance,
            patch.object(qs, "quote_orders", wraps=qs.quote_orders) as policy,
        ):
            complete = invoke(
                ["resume", "--store", store, "--run-id", "demo", "--token", token_path]
            )
            assert advance.call_count == policy.call_count == 58
            assert complete["native"]["metrics"]["suffix_steps"] == 58
        assert complete["execution"] == "COMPLETED"
        assert any(r["summary"]["pnl"] == "-14.6" for r in complete["rows"])
        with no_simulation(no_native_mutation=True):
            verified = invoke(
                ["verify-persisted", "--store", store, "--run-id", "demo"]
            )
        assert verified["verification"] == "NATIVE_STATE"
    direct = tmp_path / "direct"
    cli.write_study(scenario.read_bytes(), direct)
    product = Path(complete["product"])
    assert saved(product, times=False) == saved(direct, times=False)
    assert len(saved(product)) == 94
    wrapper = Path(complete["directory"])
    assert all(
        (wrapper / href).is_file()
        for href in re.findall(r"\]\(([^)]+)\)", (wrapper / "REPORT.md").read_text())
    )
    selection = tmp_path / "selection.json"
    selection.write_bytes(
        canonical_json_bytes(
            {
                "schema": "quantos-run-selection/v1",
                "entries": [
                    {"id": "quotes", "kind": "quantos-persisted", "path": str(wrapper)}
                ],
            }
        )
    )
    with no_simulation(no_native_mutation=True):
        index = build_index(selection, tmp_path / "index", "review")
        assert verify_index(tmp_path / "index", "review")["status"] == "VERIFIED_INDEX"
    index_json = json.loads((Path(index["directory"]) / "index.json").read_bytes())
    assert index_json["combined_score"] is None
    entry = index_json["entries"][0]
    assert entry["result_state"] == "COMPLETED"
    assert not any(
        "journal.sqlite" in name or "/anchors/" in name or "/forensics/" in name
        for name in entry["evidence"]
    )
    record(
        "walkthrough",
        {
            "g67": g67,
            "g68": g68,
            "initial": initial,
            "one_step": next_result,
            "complete": complete,
            "index": index,
            "native_public_calls": observed,
            "parity_files": 94,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def test_frozen_original_note_and_explicit_recompute(inputs):
    result = start(inputs)
    inputs[0].with_name("source.md").write_text(
        "The original has changed; snapshots remain authoritative."
    )
    with patch.object(
        execution, "admit", side_effect=AssertionError("initial admission")
    ):
        result = p.resume(inputs[2], "a", result["checkpoint"])
    root = Path(result["directory"])
    before = saved(root)
    with no_simulation(no_native_mutation=True):
        assert p.verify(inputs[2], "a")["execution"] == "COMPLETED"
    with patch.object(qs, "advance_step", wraps=qs.advance_step) as advance:
        assert (
            p.verify(inputs[2], "a", mode="recompute")["verification"]
            == "NATIVE_RECOMPUTED"
        )
        assert advance.call_count == 6
    assert saved(root) == before
    assert_refusal_unchanged(inputs, result, "ALREADY_COMPLETED")


@pytest.mark.parametrize(
    "name", ["source.md", "request.json", "scenario.json", "registration.json"]
)
def test_changed_registered_bytes_refuse(inputs, name):
    result = start(inputs)
    target = Path(result["directory"]) / name
    target.write_bytes(target.read_bytes() + b" ")
    assert_refusal_unchanged(inputs, result)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrapper_code",
        "native_code",
        "native_context",
        "token_context",
        "token_code",
        "token_bool",
        "token_extra",
        "token_schema",
        "stale",
    ],
)
def test_identity_and_typed_stale_tokens_preserve(inputs, mutation):
    result = start(inputs)
    with ExitStack() as stack:
        if mutation == "wrapper_code":
            stack.enter_context(patch.object(p, "code_identity", return_value="0" * 64))
        elif mutation == "native_code":
            stack.enter_context(patch.object(execution, "IMC_CODE", "0" * 64))
        elif mutation == "native_context":
            path = Path(result["directory"]) / "native/a/identity.json"
            value = json.loads(path.read_bytes())
            value["context_sha256"] = "0" * 64
            path.write_bytes(canonical_json_bytes(value))
        elif mutation == "token_context":
            result["checkpoint"]["context_sha256"] = "0" * 64
        elif mutation == "token_code":
            result["checkpoint"]["token"]["native_code_sha256"] = "0" * 64
        elif mutation == "token_bool":
            result["checkpoint"]["token"]["generation"] = False
        elif mutation == "token_extra":
            result["checkpoint"]["token"]["extra"] = 0
        elif mutation == "token_schema":
            result["checkpoint"]["schema"] = "native-token"
        else:
            p.resume(inputs[2], "a", result["checkpoint"], stop_after_commits=1)
        assert_refusal_unchanged(inputs, result)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "native_sibling",
        "symlink",
        "hardlink",
        "lock_alias",
        "lock_replace",
        "directory_replace",
        "db_alias",
        "native_unknown",
        "capture_incomplete",
        "oversize",
    ],
)
def test_ambiguous_ownership_preserves_bytes_and_mtimes(inputs, tmp_path, mutation):
    result = start(inputs)
    root = Path(result["directory"])
    if mutation == "unknown":
        (root / "user.txt").write_text("unrelated")
    elif mutation == "native_sibling":
        (root / "native/other").mkdir()
    elif mutation == "symlink":
        target = root / "source.md"
        original = tmp_path / "original.md"
        target.rename(original)
        target.symlink_to(original)
    elif mutation in {"hardlink", "lock_alias", "db_alias"}:
        name = {
            "hardlink": "source.md",
            "lock_alias": "wrapper.lock",
            "db_alias": "native/a/journal.sqlite",
        }[mutation]
        os.link(root / name, tmp_path / "alias")
    elif mutation == "lock_replace":
        (root / "wrapper.lock").rename(tmp_path / "old.lock")
        (root / "wrapper.lock").write_bytes(b"")
    elif mutation == "directory_replace":
        original = root.with_name("held")
        root.rename(original)
        root.mkdir()
        for child in original.iterdir():
            child.rename(root / child.name)
    elif mutation == "native_unknown":
        (root / "native/a/user.txt").write_text("unrelated")
    elif mutation == "capture_incomplete":
        (root / "native/a/forensics/generation-1").mkdir()
    else:
        (root / "request.json").write_bytes(b"x" * (65536 + 1))
    assert_refusal_unchanged(inputs, result)


@pytest.mark.parametrize(
    "label,generation", [("STEP_BEFORE_COMMIT", 1), ("STEP_AFTER_COMMIT", 2)]
)
def test_native_commit_acknowledgement_semantics(inputs, label, generation):
    result = start(inputs, budget=1)
    db = Path(result["directory"]) / "native/a/journal.sqlite"
    old_bytes = db.read_bytes()

    def fail(point):
        if point == label:
            raise SystemExit(71)

    with patch.object(journal, "_FAULT_HOOK", fail), pytest.raises(SystemExit):
        p.resume(inputs[2], "a", result["checkpoint"], stop_after_commits=1)
    with no_simulation(no_native_mutation=True):
        current = p.checkpoint(inputs[2], "a")
    assert current["checkpoint"]["token"]["generation"] == generation
    if generation == 1:
        assert db.read_bytes() == old_bytes
        assert current["checkpoint"] == result["checkpoint"]
    else:
        assert db.read_bytes() != old_bytes
        assert_refusal_unchanged(inputs, result, "STALE_CHECKPOINT")
    with patch.object(qs, "advance_step", wraps=qs.advance_step) as advance:
        final = p.resume(inputs[2], "a", current["checkpoint"])
        assert advance.call_count == 6 - generation
    record(
        "commit-" + label.lower(), {"committed_generation": generation, "result": final}
    )


@pytest.mark.parametrize(
    "state", ["PAUSED_AT_BOUNDARY", "COMPUTE_COMPLETE", "PUBLICATION_INCOMPLETE"]
)
def test_lifecycle_index_and_zero_step_publication(inputs, tmp_path, state):
    result = start(inputs, budget=0 if state == "PAUSED_AT_BOUNDARY" else 6)
    if state == "PUBLICATION_INCOMPLETE":

        def stop(point):
            if point == "PUB_AFTER_TRACE":
                raise OSError("owned partial publication")

        with patch.object(journal, "_FAULT_HOOK", stop):
            result = p.resume(inputs[2], "a", result["checkpoint"])
    assert result["execution"] == state
    with no_simulation(no_native_mutation=True):
        assert row(result, tmp_path)["result_state"] == state
        selection = tmp_path / "selection.json"
        selection.write_bytes(
            canonical_json_bytes(
                {
                    "schema": "quantos-run-selection/v1",
                    "entries": [
                        {
                            "id": "selected",
                            "kind": "quantos-persisted",
                            "path": result["directory"],
                        }
                    ],
                }
            )
        )
        index = build_index(selection, tmp_path / "index", "incomplete")
    assert index["status"] == "PARTIAL"
    if state != "PAUSED_AT_BOUNDARY":
        with no_simulation():
            final = p.resume(inputs[2], "a", result["checkpoint"])
        assert final["execution"] == "COMPLETED"
    record("index-" + state.lower(), {"index": index, "state": state})


def test_native_preownership_refusal_propagates(inputs):
    result = start(inputs, budget=6)

    def stop(point):
        if point == "PUB_AFTER_MKDIR":
            raise SystemExit(72)

    with patch.object(journal, "_FAULT_HOOK", stop), pytest.raises(SystemExit):
        p.resume(inputs[2], "a", result["checkpoint"])
    current = p.checkpoint(inputs[2], "a")
    assert current["execution"] == "PUBLICATION_INCOMPLETE"
    assert_refusal_unchanged(inputs, current, "OWNERSHIP_AMBIGUOUS")


def test_real_child_lock_and_exit_before_wrapper_finalization(inputs, tmp_path):
    result = start(inputs, budget=0)
    token = tmp_path / "child-token.json"
    token.write_bytes(canonical_json_bytes(result["checkpoint"]))
    code = """import fcntl,json,os,sys
from pathlib import Path
from quantos_showcase import persisted as p
store=Path(sys.argv[1]); lock=open(p.folder(store,'a')/'wrapper.lock','rb')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
print('locked',flush=True); sys.stdin.read(1); lock.close()
def exit_at_finalization(*args): os._exit(73)
p.finalize=exit_at_finalization
p.resume(store,'a',json.loads(Path(sys.argv[2]).read_bytes()))
"""
    started = time.monotonic()
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(inputs[2]), str(token)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            assert selector.select(10), "child did not establish ownership"
            assert child.stdout.readline().strip() == "locked"
        before = saved(Path(result["directory"]))
        for call in (
            lambda: p.checkpoint(inputs[2], "a"),
            lambda: p.resume(inputs[2], "a", result["checkpoint"]),
        ):
            with pytest.raises(p.PersistError) as error:
                call()
            assert error.value.code == "BUSY"
        assert saved(Path(result["directory"])) == before
        stdout, stderr = child.communicate("x", timeout=40)
        assert child.returncode == 73, (stdout, stderr)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
    root = Path(result["directory"])
    native_before = saved(root / "native")
    with no_simulation(no_native_mutation=True):
        current = p.checkpoint(inputs[2], "a")
        assert current["execution"] == "NATIVE_COMPLETE_WRAPPER_INCOMPLETE"
        assert (
            row(current, tmp_path)["result_state"]
            == "NATIVE_COMPLETE_WRAPPER_INCOMPLETE"
        )
        final = p.resume(inputs[2], "a", current["checkpoint"])
        assert final["execution"] == "COMPLETED"
    assert saved(root / "native") == native_before
    assert time.monotonic() - started < 60
    record(
        "child-exit",
        {
            "returncode": child.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": current,
            "recovered": final,
            "native_files_unchanged": True,
            "child_processes": 1,
            "policy_run_resume_calls_in_recovery": 0,
        },
    )


@pytest.mark.parametrize(
    "window",
    ["full_report", "partial_report", "partial_receipt", "receipt_without_report"],
)
def test_wrapper_finalization_windows_preserved(inputs, tmp_path, window):
    with (
        patch.object(p, "finalize", side_effect=SystemExit(74)),
        pytest.raises(SystemExit),
    ):
        start(inputs, budget=None)
    current = p.checkpoint(inputs[2], "a")
    root = Path(current["directory"])
    context = p.load_context(root)
    expected, _ = p.expected_publication(root, context, current["native"])
    if window != "receipt_without_report":
        (root / "REPORT.md").write_bytes(
            expected["REPORT.md"][:30]
            if window == "partial_report"
            else expected["REPORT.md"]
        )
    if window in {"partial_receipt", "receipt_without_report"}:
        (root / "complete.json").write_bytes(
            expected["complete.json"][:20]
            if window == "partial_receipt"
            else expected["complete.json"]
        )
    if window == "full_report":
        native_before = saved(root / "native")
        with no_simulation(no_native_mutation=True):
            assert (
                p.resume(inputs[2], "a", current["checkpoint"])["execution"]
                == "COMPLETED"
            )
        assert saved(root / "native") == native_before
    else:
        assert_refusal_unchanged(inputs, current, "WRAPPER_PUBLICATION_AMBIGUOUS")
        assert row(current, tmp_path)["result_state"] == "REFUSED"


def test_evidence_never_traverses_operational_state_and_corrupt_complete_refuses(
    inputs, tmp_path
):
    result = start(inputs, budget=None)
    seen = []
    original = Evidence.bind

    def bind(self, path, role, **kwargs):
        assert not any(
            part in str(path)
            for part in (
                "journal.sqlite",
                "/anchors/",
                "/forensics/",
                "/identity.json",
                "/attempt.lock",
            )
        )
        seen.append(str(path))
        return original(self, path, role, **kwargs)

    with patch.object(Evidence, "bind", bind), no_simulation(no_native_mutation=True):
        assert row(result, tmp_path)["result_state"] == "COMPLETED"
    assert any(name.endswith("/registration.json") for name in seen)
    target = Path(result["product"]) / "comparison.json"
    target.write_bytes(target.read_bytes() + b" ")
    assert_refusal_unchanged(inputs, result)
    assert row(result, tmp_path)["result_state"] == "REFUSED"


@pytest.mark.parametrize("budget", [-1, 1025, True])
def test_invalid_budget_before_mutation(inputs, budget):
    with pytest.raises(p.PersistError) as error:
        start(inputs, budget=budget)
    assert error.value.code == "RESOURCE_LIMIT" and not inputs[2].exists()
    result = start(inputs)
    before = saved(inputs[2])
    with pytest.raises(p.PersistError) as error:
        p.resume(inputs[2], "a", result["checkpoint"], stop_after_commits=budget)
    assert error.value.code == "RESOURCE_LIMIT" and saved(inputs[2]) == before


def test_strict_ids_native_limits_and_existing_attempt(inputs):
    for name in ("../bad", "Upper", "1a", "", "a" * 65):
        with pytest.raises(ValueError):
            start(inputs, name=name)
    assert not inputs[2].exists()
    result = start(inputs)
    before = saved(inputs[2])
    with pytest.raises(p.PersistError) as error:
        start(inputs)
    assert error.value.code == "EXISTS" and saved(inputs[2]) == before
    with (
        patch.object(
            journal, "MAX_LOGICAL", result["native"]["metrics"]["logical_bytes"]
        ),
        patch.object(qs, "advance_step", wraps=qs.advance_step) as advance,
    ):
        with pytest.raises(native.ResumeError) as error:
            p.resume(inputs[2], "a", result["checkpoint"], stop_after_commits=1)
        assert error.value.code == "RESOURCE_LIMIT"
        assert (
            advance.call_count <= 1
        )  # Admission or the native pre-insert reserve may refuse.
    assert saved(inputs[2]) == before
    assert journal.MAX_DB_JOURNAL == 64 * 1024**2
    assert journal.MAX_PRODUCT == 128 * 1024**2
    assert journal.MAX_ATTEMPT == 512 * 1024**2
    assert p.MAX_METADATA == 4 * 1024**2


@pytest.mark.parametrize("kind", ["runs", "steps", "input_bytes", "source_digest"])
def test_initial_admission_refuses_before_wrapper_creation(inputs, kind):
    data = json.loads(inputs[1].read_bytes())
    if kind == "runs":
        data["scenarios"] = [
            dict(data["scenarios"][0], id="s" + str(i)) for i in range(10)
        ]
        inputs[1].write_bytes(canonical_json_bytes(data))
    elif kind == "steps":
        s = data["scenarios"][0]
        steps = [dict(s["steps"][0], timestamp=i) for i in range(514)]
        steps[-1].update(buyer_units=0, seller_units=0)
        s.update(steps=steps, end_timestamp=513)
        inputs[1].write_bytes(canonical_json_bytes(data))
    elif kind == "input_bytes":
        inputs[1].write_bytes(b" " * (262144 + 1))
    else:
        data = json.loads(inputs[0].read_bytes())
        data["source"]["sha256"] = "0" * 64
        inputs[0].write_bytes(canonical_json_bytes(data))
    with (
        patch.object(
            native, "run", side_effect=AssertionError("native must not start")
        ),
        pytest.raises(ValueError),
    ):
        start(inputs)
    assert not inputs[2].exists()


def test_root_frozen_stress_and_larger_inputs(tmp_path):
    root = os.environ.get("QUANTOS_FROZEN_NATIVE_INPUTS")
    if not root:
        pytest.skip("root-owned stress inputs are an external installed gate")
    outcomes = []
    for name in ("root-stress", "root-larger"):
        scenario = Path(root) / name / "scenario.json"
        before = scenario.read_bytes()
        started = time.monotonic()
        result = p.run(
            EXAMPLES / "request.json",
            scenario,
            tmp_path / name,
            "a",
            stop_after_commits=1,
        )
        with (
            patch.object(qs, "advance_step", wraps=qs.advance_step) as advance,
            patch.object(
                execution, "admit", side_effect=AssertionError("initial admission")
            ),
        ):
            final = p.resume(tmp_path / name, "a", result["checkpoint"])
            assert advance.call_count == final["native"]["metrics"]["total_steps"] - 1
        direct = tmp_path / (name + "-direct")
        cli.write_study(before, direct)
        assert saved(Path(final["product"]), times=False) == saved(direct, times=False)
        assert scenario.read_bytes() == before
        outcomes.append(
            {
                "name": name,
                "input_sha256": sha256_bytes(before),
                "result": final,
                "elapsed_seconds": time.monotonic() - started,
                "parity_files": len(saved(direct)),
            }
        )
    record("root-fixed-inputs", outcomes)
