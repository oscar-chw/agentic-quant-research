"""Interrupt cleanup uses a real local child; no Codex/model process is started."""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from qrae import codex_broker as broker


def test_interruption_reaps_the_child_before_joining_its_output_reader(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    children = []
    watchdog_used = threading.Event()
    timers = []

    def start(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        real_wait = child.wait
        first = True

        def interrupted_wait(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt
            return real_wait(*args, **kwargs)

        def watchdog():
            if child.poll() is None:
                watchdog_used.set()
                child.kill()

        timer = threading.Timer(2, watchdog)
        timer.start()
        timers.append(timer)
        monkeypatch.setattr(child, "wait", interrupted_wait)
        return child

    monkeypatch.setattr(broker.subprocess, "Popen", start)
    try:
        with pytest.raises(KeyboardInterrupt):
            broker._bounded_subprocess_run(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                input_bytes=b"", cwd=str(tmp_path), timeout=10,
                env=dict(os.environ), max_output_bytes=1024)
        assert len(children) == 1 and children[0].poll() is not None
        assert not watchdog_used.is_set(), "cleanup waited for an external watchdog to kill the child"
    finally:
        for timer in timers:
            timer.cancel()
            timer.join()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)


def test_deadline_includes_stdout_held_by_a_descendant(tmp_path):
    started = time.monotonic()
    program = ('import subprocess,sys; '
               'subprocess.Popen([sys.executable,"-c","import time; time.sleep(3)"]); '
               'print("parent exited",flush=True)')
    with pytest.raises(subprocess.TimeoutExpired):
        broker._bounded_subprocess_run(
            [sys.executable, "-c", program], input_bytes=b"", cwd=str(tmp_path),
            timeout=1, env=dict(os.environ), max_output_bytes=4096)
    # The child's three-second sleep is the negative control; a one-second run
    # deadline must fire first, allowing ordinary process-reaping overhead.
    assert time.monotonic() - started < 2


def test_interruption_during_pipe_join_cleans_the_local_group(tmp_path, monkeypatch):
    real_popen, real_join = subprocess.Popen, threading.Thread.join
    children = []
    interrupted = False

    def start(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def join(thread, *args, **kwargs):
        nonlocal interrupted
        if thread.name == "qrae-codex-stdout" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return real_join(thread, *args, **kwargs)

    monkeypatch.setattr(broker.subprocess, "Popen", start)
    monkeypatch.setattr(threading.Thread, "join", join)
    program = ('import subprocess,sys; '
               'subprocess.Popen([sys.executable,"-c","import time; time.sleep(3)"])')
    try:
        with pytest.raises(KeyboardInterrupt):
            broker._bounded_subprocess_run(
                [sys.executable, "-c", program], input_bytes=b"", cwd=str(tmp_path),
                timeout=1, env=dict(os.environ), max_output_bytes=4096)
        assert interrupted and children[0].poll() is not None
        assert not any(t.is_alive() and t.name.startswith("qrae-codex-")
                       for t in threading.enumerate())
        if os.name != "nt":
            with pytest.raises(ProcessLookupError):
                os.killpg(children[0].pid, 0)
    finally:
        for child in children:
            if os.name != "nt":
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                child.kill()
            child.wait(timeout=3)
