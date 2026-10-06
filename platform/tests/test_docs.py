"""Docs that point into code must keep pointing at the right line."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOOP = "apps/quantos/src/quantos_showcase/loop.py"
# [`token`, loop.py:N](path#LN): the token must sit on line N, and the anchor must say N too.
REF = re.compile(r"\[`([^`]+)`, loop\.py:(\d+)\]\(([^)#]+)#L(\d+)\)")


def loop_refs(text):
    return REF.findall(text)


def stale(doc):
    lines = (ROOT / LOOP).read_text(encoding="utf-8").splitlines()
    bad = []
    for token, line, target, anchor in loop_refs(doc.read_text(encoding="utf-8")):
        n = int(line)
        if anchor != line or (doc.parent / target).resolve() != (ROOT / LOOP).resolve() \
                or not 1 <= n <= len(lines) or token not in lines[n - 1]:
            bad.append(f"{token} @ {line}")
    return bad


def test_walkthrough_line_references_resolve():
    doc = ROOT / "docs/how-a-hypothesis-dies.md"
    assert len(loop_refs(doc.read_text(encoding="utf-8"))) >= 7
    assert stale(doc) == []


def test_a_moved_line_is_caught(tmp_path):
    doc = tmp_path / "docs" / "page.md"
    doc.parent.mkdir()
    link = str(ROOT / LOOP)
    doc.write_text(f"[`def passes_validation`, loop.py:213]({link}#L213) "
                   f"[`def gate`, loop.py:478]({link}#L477)", encoding="utf-8")
    assert stale(doc) == ["def passes_validation @ 213", "def gate @ 478"]


def git(*args):
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def linked(doc):
    return {(doc.parent / t.split("#")[0]).resolve() for t in re.findall(r"\]\(([^)\s]+)\)", doc.read_text(encoding="utf-8"))
            if not re.match(r"[a-z]+:|#", t)}


def unindexed(index, tracked):
    return sorted(p for p in tracked if p != "docs/README.md" and not re.match(r"packages/[^/]+/docs/", p)
                  and (ROOT / p).resolve() not in linked(index))


def test_docs_index_lists_every_markdown_file():
    listing = git("ls-files", "*.md")
    if listing.returncode:
        pytest.skip("not a git checkout")
    index = ROOT / "docs/README.md"
    assert unindexed(index, listing.stdout.split()) == []
    assert unindexed(index, ["docs/README.md", "docs/forgotten.md"]) == ["docs/forgotten.md"]


# The repository was published with fresh history, so git order cannot evidence the
# pre-registration; the SHA-256 table in the design history does. A registered file that
# drifts from its recorded digest, or a protocol file left out of the table, must fail.
HISTORY = ROOT / "docs/design-history.md"
DIGEST_ROW = re.compile(r"^\| \[([^\]]+)\]\([^)]+\) \| [^|]+ \| `([0-9a-f]{64})` \|$", re.M)


def drifted(rows):
    import hashlib
    return [path for path, digest in rows
            if not (ROOT / path).is_file() or hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest]


def test_registered_files_match_their_recorded_sha256():
    text = HISTORY.read_text(encoding="utf-8")
    rows = DIGEST_ROW.findall(text)
    listed = {path for path, _ in rows}
    assert len(rows) == len(listed) == 13
    for study in sorted((ROOT / "results").iterdir()):
        for name in ("protocol.json", "campaign.json", "experiment.json"):
            assert f"results/{study.name}/{name}" in listed
    assert drifted(rows) == []
    path, digest = rows[0]
    flipped = digest[:-1] + ("0" if digest[-1] != "0" else "1")
    assert drifted([(path, flipped)]) == [path]
    assert drifted([("results/missing.json", digest)]) == ["results/missing.json"]


# A commit hash in a doc or a source file points into history this repository does
# not publish. The only ones allowed are those a registered protocol.json itself cites,
# which cannot be edited and which the design history explains, plus the pinned revisions
# of public upstream repositories (imc-sim reads its log format at one; protocol v2's
# amendment 5 pins the LLM arm's open weights at the other).
COMMIT = re.compile(r"(?<![\w.\-…])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}(?![\w.\-…])")
HEX_ALPHABET = "0123456789abcdef"  # a hex-digit check in code, not a hash
PUBLIC_UPSTREAM = {"0094c681f8cd019889761e6431a1a47ea151aaa8",  # nabayansaha/imc-prosperity-4-backtester
                   "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",  # huggingface.co/Qwen/Qwen3.8-27B
                   "8306c42"}  # oscar-chw/alpha-gp-lab: 2eeb94a after the 2026-10-04/06 history rewrites (same tree and date)
# Names of artefacts that were never published: a reader cannot follow them.
UNPUBLISHED = re.compile(r"quantos release|probe receipt|batch-0\d", re.I)
SKIP_DIRS = {"build", ".hypothesis", "__pycache__", ".pytest_cache"}


def commit_refs(paths):
    return {(str(p.relative_to(ROOT)), h) for p in paths for h in COMMIT.findall(p.read_text(encoding="utf-8"))
            if h != HEX_ALPHABET}


def source_files(*dirs):
    """Every tracked file under dirs; outside a git checkout, every file not in a build or cache dir."""
    listing = git("ls-files", "-z", *dirs)
    if listing.returncode == 0:
        return sorted(ROOT / p for p in listing.stdout.split("\0") if p)
    return sorted(p for d in dirs for p in (ROOT / d).rglob("*")
                  if p.is_file() and not SKIP_DIRS & set(p.relative_to(ROOT).parts))


def scanned():
    return [ROOT / "README.md", *sorted(ROOT.glob("docs/*.md")), *sorted(ROOT.glob("docs/evidence/*.json")),
            *sorted(ROOT.glob("results/*/*.md")), *sorted(ROOT.glob("tools/*.py")),
            *source_files("packages", "apps")]


def unpublished(paths, allowed):
    return sorted(ref for ref in commit_refs(paths) if ref[1] not in allowed)


def test_docs_and_sources_cite_no_unpublished_commit():
    protocols = sorted(ROOT.glob("results/*/protocol.json"))
    allowed = {h for _, h in commit_refs(protocols)} - PUBLIC_UPSTREAM
    assert allowed == {"2ec30c8", "2eeb94a"}
    paths = scanned()
    assert len(paths) >= 230 and any("packages/" in str(p) for p in paths) and any("apps/" in str(p) for p in paths)
    assert unpublished(paths, allowed | PUBLIC_UPSTREAM) == []
    assert COMMIT.findall("pre-registered in commit 6a69c1b, before") == ["6a69c1b"]
    assert COMMIT.findall("sha256 279cc869…3849, 8.9e-16, `" + "ab" * 32 + "`") == []


def test_a_planted_commit_in_package_source_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    planted = tmp_path / "packages/x/src/mod.py"
    planted.parent.mkdir(parents=True)
    planted.write_text('# ported from commit 6a69c1b\nOK = "' + HEX_ALPHABET + '"\n', encoding="utf-8")
    assert unpublished(source_files("packages"), set()) == [("packages/x/src/mod.py", "6a69c1b")]


def test_sources_name_no_unpublished_artefact():
    hits = [(str(p.relative_to(ROOT)), m) for p in scanned() for m in UNPUBLISHED.findall(p.read_text(encoding="utf-8"))]
    assert hits == []
    assert UNPUBLISHED.findall("the batch-03 probe receipt, in the quantos release") == [
        "batch-03", "probe receipt", "quantos release"]
