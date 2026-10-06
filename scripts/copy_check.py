#!/usr/bin/env python3
"""Fail if any tracked text file repeats 16 or more consecutive words of the book.

    python scripts/copy_check.py --book <your-copy-of-the-book>.txt [--run 16] [--report 10]

The book is copyrighted and is not in this repository: pass the path of a plain-text copy you own (for example
`pdftotext` of the PDF). Words are runs of letters, digits and apostrophes, lower-cased; a line-break hyphen is
joined first, so "informa-\\ntion" is one word. Every git-tracked (or untracked, not ignored) .md, .py, .txt, .toml, .ini, .yml and .sh file is
compared. Exit 1 on any run of --run words; --report N also lists every file whose longest shared run is N words or
more, so near misses are visible. Prints file names, line numbers and run lengths only, never the book's words.
Exit 2 when the book path is missing: an absent source must not read as a pass.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

WORD = re.compile(r"[a-z0-9']+")
EXTS = (".md", ".py", ".txt", ".toml", ".ini", ".yml", ".sh")
K = 8  # index granularity: shared runs shorter than this are ordinary English and are not tracked


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"-\s*\n\s*", "", text).lower()


def words_with_lines(text: str) -> tuple[list[str], list[int]]:
    words, lines = [], []
    for n, line in enumerate(normalise(text).split("\n"), 1):
        for w in WORD.findall(line):
            words.append(w)
            lines.append(n)
    return words, lines


def longest_runs(doc: list[str], book: list[str], index: dict) -> list[tuple[int, int]]:
    """(start, length) of every maximal shared run of at least K words, by extending K-word seeds."""
    out, i = [], 0
    while i <= len(doc) - K:
        best = 0
        for j in index.get(tuple(doc[i:i + K]), ()):
            n = K
            while i + n < len(doc) and j + n < len(book) and doc[i + n] == book[j + n]:
                n += 1
            best = max(best, n)
        if best:
            out.append((i, best))
            i += best - K + 1
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    ap.add_argument("--run", type=int, default=16)
    ap.add_argument("--report", type=int, default=0)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    book_path = Path(args.book)
    if not book_path.is_file():
        print(f"book text not found: {book_path}")
        return 2
    book = WORD.findall(normalise(book_path.read_text(errors="ignore")))
    index: dict[tuple, list[int]] = {}
    for j in range(len(book) - K + 1):
        index.setdefault(tuple(book[j:j + K]), []).append(j)
    files = subprocess.run(["git", "-C", args.root, "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True, check=True).stdout.split()
    bad, scanned, overall = 0, 0, 0
    for rel in files:
        if not rel.endswith(EXTS):
            continue
        scanned += 1
        doc, lines = words_with_lines((Path(args.root) / rel).read_text(errors="ignore"))
        runs = longest_runs(doc, book, index)
        top = max((n for _, n in runs), default=0)
        overall = max(overall, top)
        hits = [(lines[i], n) for i, n in runs if n >= args.run]
        bad += bool(hits)
        if hits:
            print(f"FAIL {rel}: " + ", ".join(f"line {ln} ({n} words)" for ln, n in hits))
        elif args.report and top >= args.report:
            print(f"     {rel}: longest shared run {top} words")
    print(f"copy check: {scanned} files, {len(book)} book words, longest shared run {overall} words, "
          f"limit {args.run - 1}: {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
