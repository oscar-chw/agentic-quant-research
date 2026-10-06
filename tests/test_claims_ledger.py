import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "claims.py"
spec = importlib.util.spec_from_file_location("claims_script", SCRIPT)
claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claims)

DEFINED = {"test_a": {"test_one", "test_two"}}
LISTED = {("test_a", "test_two")}


def claim(*tests, status="done"):
    return {"id": "I1.1", "chapter": 1, "section": "1.1", "kind": "x", "status": status, "tests": sorted(tests)}


def outcome(c, results):
    return claims.outcome(c, results, DEFINED, LISTED)


@pytest.mark.case
def test_a_cited_test_in_a_file_that_is_not_here_is_not_in_repo_not_reference_only():
    """It used to land in 'reference only', the bucket for a deliberately unpublished test, so a typo and a missing file
    looked like a known gap. Only tests/unpublished.txt (or a skip) makes a reference; a missing file is its own count."""
    assert outcome(claim(("test_gone", "test_x")), {}) == "not in repo"
    assert outcome(claim(("test_a", "test_two")), {}) == "reference only"           # listed in unpublished.txt
    assert outcome(claim(("test_a", "test_one")), {("test_a", "test_one"): "skipped"}) == "reference only"
    assert outcome(claim(("test_a", "test_one"), ("test_gone", "test_x")), {("test_a", "test_one"): "passed"}) == "partly run"
    assert outcome(claim(("test_a", "test_one")), {("test_a", "test_one"): "passed"}) == "passing"
    assert outcome(claim(("test_a", "test_one"), ("test_gone", "test_x")), {("test_a", "test_one"): "failed"}) == "failing"
    assert outcome(claim(("test_gone", "test_x"), status="deferred"), {}) == "no test"


@pytest.mark.case
def test_a_cited_test_a_published_file_does_not_define_is_unknown():
    """A typo or a rename: the file is here, the function is not. Neither a result nor an unpublished.txt line explains it."""
    assert claims.classify(("test_a", "test_one_typo"), {}, DEFINED, LISTED) == "unknown"
    assert claims.classify(("test_a", "test_one"), {}, DEFINED, LISTED) == "unrun"
    assert claims.classify(("test_gone", "test_x"), {}, DEFINED, LISTED) == "absent"


@pytest.mark.case
def test_unknown_tests_make_the_run_exit_1(monkeypatch, tmp_path):
    rows = [claim(("test_a", "test_one_typo"))]
    monkeypatch.setattr(claims, "load", lambda: rows)
    monkeypatch.setattr(claims, "published", lambda: DEFINED)
    monkeypatch.setattr(claims, "unpublished", lambda: LISTED)
    junit = tmp_path / "j.xml"
    junit.write_text('<testsuites><testsuite><testcase classname="tests.test_a" name="test_one"/></testsuite></testsuites>')
    monkeypatch.setattr(claims.sys, "argv", ["claims.py", "--junit", str(junit)])
    assert claims.main() == 1
    rows[:] = [claim(("test_a", "test_one"))]
    assert claims.main() == 0                                                       # control: the same run, a real citation


@pytest.mark.case
def test_every_citation_into_a_published_test_file_names_a_test_that_file_defines():
    """The ledger as committed: no cited id points into a file that is here but lacks the function."""
    assert claims.unknown_tests(claims.load(), {}) == []
