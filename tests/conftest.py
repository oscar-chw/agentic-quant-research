"""Deselect, by name, the tests whose inputs are not published here (study scripts, study outputs, the recorded
market-data cache, the private heading index). The list is tests/unpublished.txt; each line says what is missing.

A stale line fails the run: a renamed test must not silently drop out of the list and leave a claim looking tested."""
from pathlib import Path

import pytest

LIST = Path(__file__).with_name("unpublished.txt")


def _unpublished() -> dict[str, str]:
    out = {}
    for line in LIST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            node, _, why = line.partition("#")
            out[node.strip()] = why.strip()
    return out


@pytest.hookimpl(tryfirst=True)  # before -k / -m filtering, so the stale check sees every collected test of a file
def pytest_collection_modifyitems(config, items):
    wanted = _unpublished()
    collected = {it.nodeid.split("[")[0] for it in items}   # a parametrised test is listed once, by function
    files = {n.split("::")[0] for n in collected}
    stale = sorted(n for n in wanted if n.split("::")[0] in files and n not in collected)
    if stale:
        raise pytest.UsageError("tests/unpublished.txt names tests that do not exist: " + ", ".join(stale))
    drop = [it for it in items if it.nodeid.split("[")[0] in wanted]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = [it for it in items if it.nodeid.split("[")[0] not in wanted]
