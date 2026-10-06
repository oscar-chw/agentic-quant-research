import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# A child pytest in which the shared HTTP session cannot connect and retries do not sleep.
CHILD = """
import sys, time
import requests
time.sleep = lambda s: None
from pmlab import http

def down(*a, **k):
    raise requests.ConnectionError("network is down")

http._session.get = down
import pytest
sys.exit(pytest.main(["-q", "-rs", "-p", "no:cacheprovider", "tests/test_data.py",
                      "-k", "real_window_tape or real_underlying"]))
"""


@pytest.mark.integration
def test_the_two_live_api_tests_skip_with_a_reason_when_the_network_is_down():
    """The Polymarket and Binance tests call live APIs. With no network (or an HTTP error from the host, as GitHub's
    runners got from Binance) each must skip and say why; they used to go red for a cause outside this code."""
    run = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "SKIPPED [2]" in out and "network is down" in out, out      # both skipped, with the reason
