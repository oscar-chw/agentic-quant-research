import re
import tomllib
from pathlib import Path

import pytest


@pytest.mark.case
def test_the_build_pin_allows_only_setuptools_that_read_an_spdx_license_string():
    """`license = "MIT"` (PEP 639) is rejected by setuptools before 77, so a build allowing 68 cannot build the package.
    CI runs from PYTHONPATH=src and never builds, so nothing else would notice."""
    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert isinstance(cfg["project"]["license"], str)                        # the SPDX form that needs the new setuptools
    pins = [r for r in cfg["build-system"]["requires"] if r.startswith("setuptools")]
    assert len(pins) == 1
    assert int(re.fullmatch(r"setuptools>=(\d+)(\.\d+)*", pins[0]).group(1)) >= 77
