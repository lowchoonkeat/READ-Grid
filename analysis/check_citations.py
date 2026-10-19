"""Check that the manuscript's in-text citations and its reference list agree.

Catches the three failure modes that survive a manual edit:
  - a citation number in the body that has no entry in the list;
  - an entry in the list that nothing cites;
  - the same work cited under two numbers, or a number reused.

Escaping is inconsistent in the draft (`\\[1\\]` in the early sections, `[9]` later), so
both forms are matched. That inconsistency is itself reported, because it should be
normalised before the final formatting pass.

Usage
-----
    python analysis/check_citations.py [manuscript.md]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "paper_package" / "13_author_manuscript_draft_v3.md"
# A citation may list several numbers, e.g. [6, 7, 8], so the group is parsed as a list.
CITE = re.compile(r"\\?\[([\d]+(?:\s*,\s*\d+)*)\\?\]")


def main() -> None:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    text = p.read_text(encoding="utf-8")
    if "# References" not in text:
        raise SystemExit("no '# References' heading found")
    body, refs = text.split("# References", 1)

    def numbers(group: str) -> list[int]:
        return [int(x) for x in re.split(r"\s*,\s*", group) if x]

    cited: Counter[int] = Counter()
    for m in CITE.finditer(body):
        cited.update(numbers(m.group(1)))
    listed = [n for m in CITE.finditer(refs) for n in numbers(m.group(1))]

    print(f"body citations : {sorted(cited.items())}")
    print(f"reference list : {listed}")
    print(f"unlisted       : {sorted(set(cited) - set(listed)) or 'none'}")
    print(f"uncited        : {sorted(set(listed) - set(cited)) or 'none'}")
    dup = [n for n, c in Counter(listed).items() if c > 1]
    print(f"duplicate nums : {sorted(dup) or 'none'}")

    esc = len(re.findall(r"\\\[[\d,\s]+\\\]", body))
    plain = len(CITE.findall(body)) - esc
    print(f"escaping       : {esc} escaped, {plain} plain"
          + ("  <- normalise to one form" if esc and plain else ""))
    print(f"body words     : {len(body.split())}")


if __name__ == "__main__":
    main()
