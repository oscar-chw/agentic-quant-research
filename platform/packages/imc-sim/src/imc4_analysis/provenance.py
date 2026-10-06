"""Identify the actual installed Python sources, independently of version labels."""
import hashlib
import json
from pathlib import Path


def code_identity(package_directory=None):
    root = Path(package_directory) if package_directory is not None else Path(__file__).resolve().parent
    files = [{"path": file.relative_to(root).as_posix(), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
             for file in sorted(root.rglob("*.py")) if "__pycache__" not in file.parts]
    if not files:
        raise OSError("installed Python source fingerprint is unavailable")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files,
            "algorithm": "SHA-256 of canonical sorted relative-path and per-file SHA-256 JSON manifest",
            "scope": "Installed imc4_analysis Python source files only; excludes fixture, metadata, interpreter and stdlib. Normal unpacked wheel/source installation required."}
