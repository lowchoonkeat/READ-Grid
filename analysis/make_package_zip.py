"""Zip paper_package/ and write its sha256 next to the archive.

Two properties matter for a package that travels to another machine:
  1. the archive must contain exactly the files MANIFEST.txt lists, so the manifest
     can be used to verify the extraction;
  2. the archive must be reproducible, so the same folder always yields the same
     bytes. Fixed timestamps and sorted entry order give that.

Refuses to run if MANIFEST.txt is older than any file it lists, because that means
someone edited the package after regenerating the manifest and the hashes no
longer describe what is inside.

Usage
-----
    python analysis/make_package_zip.py
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "paper_package"
ZIP = ROOT / "paper_package.zip"
SUMF = ROOT / "paper_package.zip.sha256"
STAMP = (2026, 10, 18, 0, 0, 0)


def main() -> None:
    man = PKG / "MANIFEST.txt"
    files = sorted(p for p in PKG.rglob("*") if p.is_file())
    stale = [p for p in files if p.name != "MANIFEST.txt" and p.stat().st_mtime > man.stat().st_mtime]
    if stale:
        raise SystemExit(
            "MANIFEST.txt is older than " + ", ".join(p.name for p in stale[:5]) +
            "\nrun analysis/regen_manifest.py first"
        )

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo("paper_package/" + p.relative_to(PKG).as_posix(), STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())

    h = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    SUMF.write_text(f"{h}  {ZIP.name}\n", encoding="utf-8")
    print(f"{len(files)} files -> {ZIP} ({ZIP.stat().st_size} bytes)")
    print(f"sha256 {h}")


if __name__ == "__main__":
    main()
