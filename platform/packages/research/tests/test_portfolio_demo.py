from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_demo_replay_tamper_detection_and_existing_output(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "portfolio_demo.py"
    spec = importlib.util.spec_from_file_location("portfolio_demo", source)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    output = tmp_path / "demo"
    result = demo.run_demo(output)
    assert result["idempotent_replay"] is True
    assert result["tamper_rejected"] is True
    before = (output / "demo-summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        demo.run_demo(output)
    assert (output / "demo-summary.json").read_bytes() == before
