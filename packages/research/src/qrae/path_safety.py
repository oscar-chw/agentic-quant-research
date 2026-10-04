"""Small cross-platform helpers for rejecting links and Windows reparse points."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if stat.S_ISLNK(metadata.st_mode):
        return True
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(flag and attributes & flag)


def has_link_or_reparse_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if is_link_or_reparse(current):
            return True
    return False


__all__ = ["has_link_or_reparse_component", "is_link_or_reparse"]
