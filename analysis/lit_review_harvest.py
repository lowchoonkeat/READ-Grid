"""Literature-gap harvest for P359 / READ-Grid v2.

Two mechanisms, deliberately separated because they fail in different places:

  1. KEYWORD   -- gap queries for domain literature (G1, G2, G3, G5, G6, G9).
                  Cheap, broad, works. Does NOT work for methodology gaps.
  2. CITATION  -- backward/forward citation chase from anchor papers (G4).
                  This is the only mechanism that reaches the evaluation-methodology
                  canon, because the vocabulary there is not searchable ("tuning
                  budget" returns 36 hits; "margin" returns 560k).

Neither mechanism covers standards (ANSI C84.1) or software docs (OpenDSS,
dss-python). Those are fetched by hand; see 17_literature_gap_plan.md.

Usage
-----
    python analysis/lit_review_harvest.py            # both passes, writes refs/
    python analysis/lit_review_harvest.py keyword
    python analysis/lit_review_harvest.py citation

Outputs
-------
    refs/refs_candidates.csv    one row per candidate, gap-tagged, verdict blank
    refs/gap_coverage.md        per-gap counts + what is still missing
    refs/harvest_log.json       query strings and hit counts, for the audit trail

Polite pool: set OPENALEX_MAILTO, otherwise a placeholder is sent. Without an API
key the request shares a small daily budget with every other user on the IP, so
the script sleeps 1 s between calls and never paginates past `max_pages`.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "research@example.org")
SLEEP = 1.0
SELECT = "id,doi,title,publication_year,cited_by_count,primary_location,type,open_access,authorships"
# referenced_works is a long list; only request it for the anchor lookup, otherwise every
# 50-row batch carries it for nothing.
ANCHOR_SELECT = SELECT + ",referenced_works"

# Citers of an evaluation-methodology paper are mostly papers that *use* it, not papers
# that *argue about* evaluation. This filter is the cheap co-citation proxy.
FWD_FILTER = (r"(evaluat|compar|benchmark|statistic|significan|confidence|variance|seed|"
              r"reproduc|report|tuning|hyperparameter|power|equivalen|non-?inferior|margin|"
              r"holdout|reuse|selective|bias|robust)")
OUT = Path(__file__).resolve().parent.parent / "refs"


# --------------------------------------------------------------------------- #
# gap definitions
# --------------------------------------------------------------------------- #

KEYWORD_GAPS = [
    {
        "id": "G1",
        "label": "Edge autonomy for feeder-level demand response",
        "why": "Supports the §2.1 negation: offline edge learning is not itself novel.",
        "q": '("edge intelligence" OR "edge computing" OR "edge device") AND ("demand response" OR "load control")',
        "extra": "publication_year:2015-2026,cited_by_count:>20,has_abstract:true",
        "n": 20,
    },
    {
        "id": "G2",
        "label": "Local fallback under communication failure",
        "why": "Supports the §2.2 negation: switching to local on comms loss is not itself novel.",
        "q": '("communication failure" OR "communication outage" OR "loss of communication") AND (microgrid OR "distribution network" OR "local control")',
        "extra": "publication_year:2014-2026,cited_by_count:>15",
        "n": 20,
    },
    {
        "id": "G3",
        "label": "Authority semantics: lease, epoch, fencing",
        "why": "Vocabulary for §3.1; must NOT be cited as proof the arbiter is correct.",
        "q": '("leader election" OR "fencing token" OR "distributed lock") AND ("control plane" OR "distributed system")',
        "extra": "publication_year:2000-2026,cited_by_count:>30",
        "n": 15,
    },
    {
        "id": "G5",
        "label": "Action masking / constrained action sets",
        "why": "Cites §3.4 admissible-action masking; currently uncited.",
        "q": '("action masking" OR "invalid action") AND (reinforcement learning OR "policy gradient")',
        "extra": "publication_year:2017-2026,cited_by_count:>10",
        "n": 12,
    },
    {
        "id": "G6",
        "label": "Batch fitted-Q and linear value approximation",
        "why": "Cites the algorithm in §3.4; currently uncited.",
        "q": '("fitted Q" OR "batch reinforcement learning") AND ("value function" OR "least squares")',
        "extra": "publication_year:2000-2026,cited_by_count:>30",
        "n": 12,
    },
    {
        "id": "G9",
        "label": "Rule-based versus learned control in demand response",
        "why": "The closest prior work to this paper; §2 currently has none of it.",
        "q": '("rule-based" OR heuristic) AND ("reinforcement learning") AND ("demand response" OR "building control" OR "voltage control")',
        "extra": "publication_year:2015-2026,cited_by_count:>25",
        "n": 20,
    },
]

# Anchors chosen because each is the load-bearing citation for one sub-gap of G4.
# Chasing their references + citers reconstructs the canon without keyword search.
CITATION_ANCHORS = [
    # G4a -- matched tuning budget / fair empirical comparison.
    # Backward chase from this anchor is contaminated: its references include BERT and
    # other systems it *evaluated*, not the methodology it *argues for*. ref_filter keeps
    # only references whose title is about evaluation practice.
    ("G4a", "10.18653/v1/d19-1224",
     r"(evaluat|compar|benchmark|tuning|hyperparameter|statistic|report|reproduc|experiment|significan|variance|robust|protocol)"),
    # G4b -- test-set reuse, winner's curse, selective inference
    ("G4b", "10.1126/science.aaa9375", None),
    # G4c -- seed count, CIs, clustered bootstrap
    ("G4c", "10.1609/aaai.v32i1.11694", None),
    ("G4c", "10.48550/arxiv.2108.13264", None),
    # G4d -- non-inferiority / equivalence margin
    ("G4d", "10.1007/bf01068419", None),
]


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

def _get(params: dict) -> dict:
    params = dict(params)
    params["mailto"] = MAILTO
    url = BASE + "?" + urllib.parse.urlencode(params, safe=":,()<>\"'|")
    req = urllib.request.Request(url, headers={"User-Agent": f"p359-harvest (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as fh:
        return json.load(fh)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _row(gap: str, via: str, w: dict, note: str = "") -> dict:
    auth = w.get("authorships") or []
    first = auth[0]["author"]["display_name"] if auth else ""
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}).get("display_name") or ""
    return {
        "gap": gap,
        "via": via,
        "openalex_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "year": w.get("publication_year") or "",
        "first_author": first,
        "title": w.get("title") or "",
        "venue": src,
        "type": w.get("type") or "",
        "cited_by": w.get("cited_by_count") or 0,
        "is_oa": bool((w.get("open_access") or {}).get("is_oa")),
        "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
        "verdict": "",
        "note": note,
    }


# --------------------------------------------------------------------------- #
# pass 1: keyword
# --------------------------------------------------------------------------- #

def run_keyword(log: dict) -> list[dict]:
    rows: list[dict] = []
    for gap in KEYWORD_GAPS:
        flt = f'title_and_abstract.search:{gap["q"]},{gap["extra"]}'
        try:
            r = _get({"filter": flt, "per-page": gap["n"], "sort": "cited_by_count:desc",
                      "select": SELECT})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{gap['id']}] ERROR {exc}")
            log["keyword"].append({"gap": gap["id"], "error": str(exc), "query": gap["q"]})
            time.sleep(SLEEP)
            continue
        n = r["meta"]["count"]
        got = len(r.get("results", []))
        print(f"  [{gap['id']}] hits={n:6d}  kept={got}")
        log["keyword"].append({"gap": gap["id"], "query": gap["q"], "filter": gap["extra"],
                               "hits": n, "kept": got, "label": gap["label"]})
        for w in r.get("results", []):
            rows.append(_row(gap["id"], "keyword", w, gap["why"]))
        time.sleep(SLEEP)
    return rows


# --------------------------------------------------------------------------- #
# pass 2: citation chase
# --------------------------------------------------------------------------- #

def _fetch_by_ids(ids: list[str]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        flt = "openalex_id:" + "|".join(chunk)
        try:
            r = _get({"filter": flt, "per-page": 50, "select": SELECT})
        except Exception as exc:  # noqa: BLE001
            print(f"    batch ERROR {exc}")
            continue
        out.extend(r.get("results", []))
        time.sleep(SLEEP)
    return out


def run_citation(log: dict, keep: int = 12) -> list[dict]:
    rows: list[dict] = []
    for gap, doi, ref_filter in CITATION_ANCHORS:
        try:
            r = _get({"filter": f"doi:{doi}", "per-page": 1, "select": ANCHOR_SELECT})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{gap}] anchor {doi} ERROR {exc}")
            log["citation"].append({"gap": gap, "anchor_doi": doi, "error": str(exc)})
            time.sleep(SLEEP)
            continue
        res = r.get("results", [])
        if not res:
            print(f"  [{gap}] anchor {doi} NOT FOUND")
            log["citation"].append({"gap": gap, "anchor_doi": doi, "error": "not found"})
            time.sleep(SLEEP)
            continue

        anchor = res[0]
        wid = anchor["id"].rsplit("/", 1)[-1]
        rows.append(_row(gap, "anchor", anchor, "anchor paper"))

        # backward
        refs = anchor.get("referenced_works") or []
        back = _fetch_by_ids(refs[:80]) if refs else []
        if ref_filter:
            keep_re = re.compile(ref_filter, re.I)
            back = [w for w in back if keep_re.search(w.get("title") or "")]
        back.sort(key=lambda w: w.get("cited_by_count") or 0, reverse=True)
        for w in back[:keep]:
            rows.append(_row(gap, "backward", w, f"cites {anchor.get('title','')[:40]}"))

        # forward
        try:
            fwd = _get({"filter": f"cites:{wid},publication_year:2015-2026",
                        "per-page": 50, "sort": "cited_by_count:desc", "select": SELECT})
            frows_all = fwd.get("results", [])
            fn = fwd["meta"]["count"]
        except Exception as exc:  # noqa: BLE001
            frows_all, fn = [], 0
            print(f"    forward ERROR {exc}")
        # Citers of a methodology anchor include every paper that cites it in passing
        # (e.g. DQN papers citing Henderson for the environment). Keep only citers whose
        # title is itself about evaluation practice; this is the co-citation proxy.
        keep_fwd = re.compile(FWD_FILTER, re.I)
        frows = [w for w in frows_all if keep_fwd.search(w.get("title") or "")][:keep]
        for w in frows:
            rows.append(_row(gap, "forward", w, f"cites {anchor.get('title','')[:40]}"))

        print(f"  [{gap}] {anchor.get('title','')[:52]:54s} refs={len(refs):3d} "
              f"back={len(back[:keep]):2d} citers={fn:4d} fwd={len(frows):2d}")
        log["citation"].append({"gap": gap, "anchor_doi": doi, "anchor_id": wid,
                                "anchor_title": anchor.get("title"),
                                "n_referenced": len(refs), "n_citers": fn})
        time.sleep(SLEEP)
    return rows


# --------------------------------------------------------------------------- #
# dedup + write
# --------------------------------------------------------------------------- #

def dedup(rows: list[dict]) -> list[dict]:
    seen_doi: dict[str, dict] = {}
    seen_title: dict[str, dict] = {}
    out: list[dict] = []
    for r in rows:
        key_t = _norm_title(r["title"])
        hit = None
        if r["doi"] and r["doi"].lower() in seen_doi:
            hit = seen_doi[r["doi"].lower()]
        elif key_t and key_t in seen_title:
            hit = seen_title[key_t]
        if hit is not None:
            if r["gap"] not in hit["gap"]:
                hit["gap"] = hit["gap"] + "|" + r["gap"]
            if r["via"] not in hit["via"]:
                hit["via"] = hit["via"] + "|" + r["via"]
            # keep the richer metadata of the two
            for k in ("doi", "venue", "year", "cited_by"):
                if not hit.get(k) and r.get(k):
                    hit[k] = r[k]
            continue
        out.append(r)
        if r["doi"]:
            seen_doi[r["doi"].lower()] = r
        if key_t:
            seen_title[key_t] = r
    out.sort(key=lambda r: (r["gap"], -int(r["cited_by"] or 0)))
    return out


def write_csv(rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "refs_candidates.csv"
    cols = list(rows[0].keys()) if rows else []
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def write_coverage(rows: list[dict], log: dict) -> Path:
    p = OUT / "gap_coverage.md"
    per: dict[str, list[dict]] = {}
    for r in rows:
        for g in r["gap"].split("|"):
            per.setdefault(g, []).append(r)
    lines = [
        "# Gap coverage after harvest",
        "",
        "Generated by `analysis/lit_review_harvest.py`. `verdict` in the CSV is blank on purpose:",
        "OpenAlex cannot decide relevance, only surface candidates. Fill it by reading.",
        "",
        "| gap | candidates | with DOI | OA full text | top hit (by citations) |",
        "|---|---:|---:|---:|---|",
    ]
    for g in sorted(per):
        rs = per[g]
        top = rs[0]["title"][:58] if rs else "-"
        lines.append(f"| {g} | {len(rs)} | {sum(1 for r in rs if r['doi'])} | "
                     f"{sum(1 for r in rs if r['is_oa'])} | {top} |")
    lines += ["", "## Query log", "", "| gap | mechanism | hits | kept |", "|---|---|---:|---:|"]
    for e in log.get("keyword", []):
        lines.append(f"| {e['gap']} | keyword | {e.get('hits','ERR')} | {e.get('kept','-')} |")
    for e in log.get("citation", []):
        if "error" in e:
            lines.append(f"| {e['gap']} | citation | ERR ({e['error']}) | - |")
        else:
            lines.append(f"| {e['gap']} | citation | refs={e['n_referenced']} citers={e['n_citers']} | - |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    log: dict = {"keyword": [], "citation": [], "mailto": MAILTO}
    rows: list[dict] = []

    # A single-pass run must not wipe the other pass's rows: reload what is on disk and
    # let dedup() merge. Only `all` rebuilds from scratch.
    prev = OUT / "refs_candidates.csv"
    if mode != "all" and prev.exists():
        with prev.open(encoding="utf-8") as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
        print(f"merging into {len(rows)} existing candidates")

    if mode in ("all", "keyword"):
        print("pass 1: keyword")
        rows += run_keyword(log)
    if mode in ("all", "citation"):
        print("pass 2: citation chase")
        rows += run_citation(log)

    rows = dedup(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "harvest_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    csv_p = write_csv(rows)
    cov_p = write_coverage(rows, log)
    print(f"\n{len(rows)} unique candidates -> {csv_p}")
    print(f"coverage table -> {cov_p}")


if __name__ == "__main__":
    main()
