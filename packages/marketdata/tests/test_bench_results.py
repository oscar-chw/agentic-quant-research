"""`make native-bench` must never overwrite a committed results file."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native/bench"))
from run_bench import write_new  # noqa: E402


def test_a_second_run_on_the_same_date_gets_a_new_file(tmp_path):
    committed = tmp_path / "results-2026-10-04.json"
    committed.write_text("committed\n")
    first = write_new(tmp_path, "2026-10-04", "run 2\n")
    second = write_new(tmp_path, "2026-10-04", "run 3\n")
    assert (first.name, second.name) == ("results-2026-10-04-2.json", "results-2026-10-04-3.json")
    assert committed.read_text() == "committed\n"
    assert (first.read_text(), second.read_text()) == ("run 2\n", "run 3\n")
    assert write_new(tmp_path, "2026-10-05", "x\n").name == "results-2026-10-05.json"
