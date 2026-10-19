"""Second-pass keyword harvest for the gaps whose first query returned noise.

G1, G3, G5, G6 screened cleanly on the first pass. G2 did not: the first query
returned droop control, consensus dispatch and protection relaying -- the
literature of *microgrid control*, not of *what a controller does when the
command path fails*. That is a query-vocabulary failure, not a library gap, so
it gets a second pass before the gap is declared partly closed.

Each case below is a different guess at how the literature names the thing:

  G2-a  the *response* to failure   (autonomous / degraded / fallback operation)
  G2-b  the *property* being sought (resilience of a cyber-physical microgrid)
  G2-c  the mature parent field     (networked control under packet loss)
  G2-d  the grid-specific mechanism (seamless transfer between modes)
  G3-b  the authority vocabulary    (leases, fencing, mutual exclusion)

Rows are appended to refs/refs_candidates.csv through the same dedup() as the
first pass, so `via` reads `keyword2` and the gap tag merges rather than
duplicating.

Usage
-----
    python analysis/harvest_pass2.py
"""

from __future__ import annotations

import csv
import sys
import time

from lit_review_harvest import OUT, SELECT, _get, _row, dedup

CASES = [
    {
        "gap": "G2", "via": "keyword2", "n": 10,
        "why": "G2 second pass: names the response to a failed command path.",
        "q": ('("communication failure" OR "communication loss" OR "loss of communication") '
              'AND ("autonomous operation" OR "degraded operation" OR fallback OR '
              '"fail-operational")'),
        "extra": "publication_year:2012-2026,cited_by_count:>15,has_abstract:true",
    },
    {
        "gap": "G2", "via": "keyword2", "n": 15,
        "why": "G2 second pass: cyber-physical resilience of a microgrid.",
        "q": ('("communication failure" OR "cyber-physical") AND (resilience OR '
              '"resilient control") AND (microgrid OR "distribution system")'),
        "extra": "publication_year:2015-2026,cited_by_count:>30,has_abstract:true",
    },
    {
        "gap": "G2", "via": "keyword2", "n": 12,
        "why": "G2 second pass: networked control under packet loss, the parent field.",
        "q": ('("networked control" OR "networked control system") AND '
              '("packet loss" OR "packet dropout" OR "communication failure") AND '
              '(compensation OR "fault-tolerant" OR stability)'),
        "extra": "publication_year:2005-2026,cited_by_count:>60,has_abstract:true",
    },
    {
        "gap": "G2", "via": "keyword2", "n": 10,
        "why": "G2 second pass: seamless transfer between operating modes.",
        "q": ('("seamless transfer" OR "seamless switching") AND '
              '(microgrid OR "distribution network" OR inverter)'),
        "extra": "publication_year:2010-2026,cited_by_count:>30,has_abstract:true",
    },
    {
        "gap": "G3", "via": "keyword2", "n": 10,
        "why": "G3 second pass: lease / fencing / mutual exclusion vocabulary.",
        "q": ('("distributed lock" OR leases OR "fencing token") AND '
              '("mutual exclusion" OR "fault-tolerant" OR "cache consistency")'),
        "extra": "publication_year:1985-2026,cited_by_count:>40,has_abstract:true",
    },
]


def main() -> None:
    rows: list[dict] = []
    prev = OUT / "refs_candidates.csv"
    if prev.exists():
        with prev.open(encoding="utf-8") as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
        print(f"merging into {len(rows)} existing candidates")

    added = 0
    for c in CASES:
        flt = f'title_and_abstract.search:{c["q"]},{c["extra"]}'
        try:
            r = _get({"filter": flt, "per-page": c["n"],
                      "sort": "cited_by_count:desc", "select": SELECT})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{c['gap']}] ERROR {exc}")
            time.sleep(1.0)
            continue
        got = r.get("results", [])
        print(f"  [{c['gap']}] hits={r['meta']['count']:6d}  kept={len(got):2d}  "
              f"({c['why'][:52]})")
        for w in got:
            rows.append(_row(c["gap"], c["via"], w, c["why"]))
            added += 1
        time.sleep(1.0)

    rows = dedup(rows)
    with prev.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n+{added} raw rows -> {len(rows)} unique candidates in {prev}")


if __name__ == "__main__":
    main()
