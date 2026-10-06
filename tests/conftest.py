"""Deselect, by name, the tests whose inputs are not published here (study scripts, study outputs, the recorded
market-data cache, the private heading index). The list is tests/unpublished.txt; each line says what is missing.

A stale line fails the run: a renamed test must not silently drop out of the list and leave a claim looking tested."""
import ast
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


def _defined(file: str) -> set[str]:
    """The test functions a file defines, read from its source, so that staleness does not depend on what this run selected."""
    path = LIST.parent.parent / file
    return {n.name for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.hookimpl(tryfirst=True)  # before -k / -m filtering
def pytest_collection_modifyitems(config, items):
    wanted = _unpublished()
    collected = {it.nodeid.split("[")[0] for it in items}   # a parametrised test is listed once, by function
    files = {n.split("::")[0] for n in collected}
    # Stale means "the file has no such test function", not "this run did not collect it": running one node id
    # collects one test, and every other listed test of that file would otherwise read as stale and abort the session.
    defined = {f: _defined(f) for f in files}
    stale = sorted(n for n in wanted if n.split("::")[0] in files and n.split("::")[1] not in defined[n.split("::")[0]])
    if stale:
        raise pytest.UsageError("tests/unpublished.txt names tests that do not exist: " + ", ".join(stale))
    drop = [it for it in items if it.nodeid.split("[")[0] in wanted]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = [it for it in items if it.nodeid.split("[")[0] not in wanted]
