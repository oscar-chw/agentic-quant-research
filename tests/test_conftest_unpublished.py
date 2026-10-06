import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _pytest(cwd, *args):
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.integration
def test_one_node_id_runs_even_though_the_file_has_other_unpublished_tests():
    """Selecting one test by node id (what -rfE prints after a CI failure) collected one test, so every other listed
    test of that file read as stale and the session aborted with 'names tests that do not exist'."""
    listed = (REPO / "tests" / "unpublished.txt").read_text()
    other = "tests/test_meta_label.py::test_the_filter_check_takes_only_signals_whose_probability_clears_their_cost"
    assert other in listed                                                  # the premise: a sibling is listed too
    run = _pytest(REPO, "--collect-only", "-q", "tests/test_meta_label.py::test_bagged_trees_can_be_bagged_one_tree_forests_as_in_the_book")
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "tests/test_meta_label.py: 1"               # exactly the one selected test, collected


@pytest.mark.integration
def test_a_listed_test_that_was_renamed_still_fails_the_run_whatever_is_selected(tmp_path):
    """The stale check must keep its teeth: a line naming a function the file no longer defines aborts the session,
    for a whole-file run and for a single node id alike."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text((REPO / "tests" / "conftest.py").read_text())
    (tmp_path / "tests" / "test_x.py").write_text("def test_new_name():\n    pass\n\ndef test_other():\n    pass\n")
    (tmp_path / "tests" / "unpublished.txt").write_text("tests/test_x.py::test_old_name  # renamed away\n")
    for target in ("tests/test_x.py", "tests/test_x.py::test_other"):
        run = _pytest(tmp_path, target)
        assert run.returncode != 0 and "tests/test_x.py::test_old_name" in run.stdout + run.stderr, target
    (tmp_path / "tests" / "unpublished.txt").write_text("tests/test_x.py::test_new_name  # a real one\n")
    ok = _pytest(tmp_path, "tests/test_x.py::test_other")                  # control: the same file with a true line passes
    assert ok.returncode == 0, ok.stdout + ok.stderr
