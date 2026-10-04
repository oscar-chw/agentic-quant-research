"""Shared local artifact primitives; no external services or mutable registry."""
import hashlib
import json
import os


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_once(path, raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_bounded(path):
    with path.open("rb") as f:
        raw = f.read(16*1024*1024+1)
    if len(raw) > 16*1024*1024:
        raise ValueError("input file exceeds 16 MiB")
    return raw
