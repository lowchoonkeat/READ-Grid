"""Regenerate paper_package/MANIFEST.txt from the files on disk.

The manifest is the only artefact that lets someone confirm a copied package is
the package that produced the numbers, so it has to be generated, not edited.
Regenerating it by hand is how a stale hash survives into a submission.

Keeps the header block verbatim from the existing file and rewrites the table, so
the only diff is the rows whose bytes or hashes actually changed.

Usage
-----
    python analysis/regen_manifest.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "paper_package"
MAN = PKG / "MANIFEST.txt"
SKIP = {"MANIFEST.txt"}
# The path column must be at least as wide as the longest name, or a long filename runs
# into the byte count and the row stops being parseable. The Gemini report is 70 chars.
MIN_PATH_W, BYTE_W = 51, 6


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    old = [s.rstrip("\r") for s in MAN.read_text(encoding="utf-8").split("\n")]
    # preamble = the three descriptive lines above the column header; keep them verbatim
    preamble = old[: next(i for i, s in enumerate(old) if s.startswith("path"))]

    files = sorted(
        (p for p in PKG.rglob("*") if p.is_file() and p.name not in SKIP
         and p.suffix not in {".zip"}),
        key=lambda p: p.relative_to(PKG).as_posix(),
    )
    rows, total = [], 0
    width = max(MIN_PATH_W, max(len(p.relative_to(PKG).as_posix()) for p in files) + 2)
    for p in files:
        rel = p.relative_to(PKG).as_posix()
        n = p.stat().st_size
        total += n
        rows.append(f"{rel:<{width}}{n:>{BYTE_W}}  {sha256(p)}")

    footer = [
        "-" * (width + BYTE_W + 2 + 64),
        f"{len(files)} files, {total} bytes total ({total / 1024:.1f} KiB)",
        "",
        "note: MANIFEST.txt does not list itself. paper_package.zip is the archive of this",
        "folder; its own sha256 is written to paper_package.zip.sha256 next to the zip.",
    ]
    header = preamble + [f"{'path':<{width}}{'bytes':>{BYTE_W}}  sha256"]
    # newline="" keeps LF on Windows; the default would silently write CRLF and make the
    # manifest the only file in the package with different line endings.
    with MAN.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header + rows + footer) + "\n")
    print(f"{len(files)} files, {total} bytes -> {MAN}")


if __name__ == "__main__":
    main()
