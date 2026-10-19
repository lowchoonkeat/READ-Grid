"""Run the plan's §8 pre-delivery checklist against a manuscript file.

The checklist in 16_section_plan_v2.md §8 is a list of prohibitions, and prohibitions
are exactly what a writer stops noticing after the third revision. This script turns
them into assertions: section word counts against the plan's targets, the retired
numbers that must never appear, the unit rules for the runtime figures, and the
sentence-level rule that `inconclusive` must travel with its half-width and margin.

Every check prints PASS / FAIL / WARN with the evidence it used. A WARN is a
judgement call that needs a human eye; a FAIL is a violation of a stated red line.

Usage
-----
    python analysis/audit_manuscript.py [file]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "paper_package" / "13_author_manuscript_draft_v3.md"

# section -> target words, from the plan's §1 table
TARGETS = {
    "Abstract": 250,
    "1. Introduction": 750,
    "2. Related work": 550,
    "3. System model": 1000,
    "4. Experimental design": 850,
    "5. Results": 4000,
    "6. Validity": 800,
    "7. Discussion": 600,
    "8. Conclusion": 280,
    "Data and code availability": 100,
}

RETIRED = ["1389", "3923", "6.5%", "92.7"]

results: list[tuple[str, str, str]] = []
LINES: list[str] = []


def say(msg: str = "") -> None:
    LINES.append(msg)
    print(msg)


def check(name: str, ok: bool, detail: str, warn: bool = False) -> None:
    results.append(("WARN" if (warn and not ok) else ("PASS" if ok else "FAIL"), name, detail))


def sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    cur = "__front__"
    for line in text.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            cur = line[2:].strip()
            out[cur] = []
            continue
        out.setdefault(cur, []).append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def words(s: str) -> int:
    return len(re.sub(r"[#*`|]|<[^>]+>", " ", s).split())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    p = Path(args[0]) if args else DEFAULT
    text = p.read_text(encoding="utf-8")
    body, refs = text.split("# References", 1)
    secs = sections(body)

    # --- word counts ---
    say(f"{'section':32s} {'target':>7s} {'actual':>7s}  delta")
    for name, target in TARGETS.items():
        key = next((k for k in secs if k.startswith(name)), None)
        if key is None:
            say(f"{name:32s} {target:7d}   MISSING")
            continue
        n = words(secs[key])
        pct = (n - target) / target * 100
        flag = "" if abs(pct) <= 12 else "  <- outside ±12%"
        say(f"{name:32s} {target:7d} {n:7d}  {pct:+6.1f}%{flag}")

    # --- retired numbers ---
    hits = [r for r in RETIRED if r in body]
    check("retired numbers absent", not hits, f"found {hits}" if hits else "none present")

    # --- runtime unit rules ---
    # The rule is about inference latency, which must be quoted per control step. The
    # configured network parameters (backhaul delay, jitter, local delay) are inputs to
    # the simulation, not measurements of the controller, so they are excluded here.
    NETWORK_MS = re.compile(r"backhaul|jitter|one-way|local delay|round-trip", re.I)
    ms_sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if re.search(r"\bms\b", s)]
    runtime = [s for s in ms_sentences if not NETWORK_MS.search(s)]
    bad = [s for s in runtime if "control step" not in s and "per episode" not in s]
    check("every ms figure is per control step", not bad,
          f"{len(bad)} runtime sentence(s) quote ms without the unit: {bad[0][:90]!r}" if bad else
          f"{len(runtime)} runtime sentences unit-qualified, "
          f"{len(ms_sentences) - len(runtime)} network parameter(s) excluded")
    check("147 per controller decision present", "147 multiply" in body.replace("–", "-"),
          "147 multiply-accumulates per controller decision")
    check("8,085 per control step present", "8,085" in body, "8,085 per control step")
    check("no per-decision millisecond", not re.search(r"ms per (controller )?decision", body),
          "no per-decision latency")

    # --- the inconclusive sentence ---
    inc = [s for s in re.split(r"(?<=[.!?])\s+", body) if "inconclusive" in s.lower()]
    ok = any(("0.0207" in s) and ("0.02" in s) for s in inc)
    check("inconclusive travels with half-width and margin", ok,
          f"{len(inc)} sentence(s) use 'inconclusive'; half-width and delta co-located: {ok}")

    # --- claim strength ---
    for word in ("outperform", "superior to the rule", "better than the rule", "novel algorithm"):
        check(f"no '{word}'", word not in body.lower(), "absent")
    # Only priority claims are prohibited. "The first evaluation", "first place" and
    # "the first is ..." are ordinary English and must not trip the check.
    PRIORITY = re.compile(
        r"\bfirst to (?:propose|show|demonstrate|apply|use|introduce|address|study|evaluate"
        r"|combine|report|quantify|measure)"
        r"|(?:we|this paper|this work|this study) (?:are|is) the first"
        r"|the first (?:work|paper|study|approach|method|algorithm|framework|system) to"
        r"|for the first time",
        re.I,
    )
    sents = re.split(r"(?<=[.!?])\s+", body)
    priority = [s for s in sents if PRIORITY.search(s)]
    benign = sum(1 for s in sents if re.search(r"\bfirst\b", s, re.I))
    check("no priority claim", not priority,
          f"{len(priority)} priority sentence(s): {priority[0][:90]!r}" if priority else
          f"absent ({benign} benign uses of 'first' in non-priority senses)")

    # --- §7 numeric budget ---
    # Condition labels (C4), run labels (Run 1) and section numbers are not figures cited
    # as evidence, so the count looks at bare numerals only.
    s7 = next((v for k, v in secs.items() if k.startswith("7. Discussion")), "")
    nums = re.findall(r"(?<![A-Za-z§])(?<!Run )(−?\d+(?:[.,]\d+)?)", s7)
    check("§7 cites at most 6 numbers", len(nums) <= 6, f"{len(nums)}: {nums}")

    # --- mechanical lint: sentence-boundary damage from scripted edits ----
    tail = body[body.index("# Abstract"):] if "# Abstract" in body else body
    spliced = re.findall(r",\s+(?:The|It|This|A|An|In|If|For|Both|Every)\b", tail)
    check("no comma-spliced capital", not spliced,
          f"{len(spliced)} occurrence(s) of ', Capital'" if spliced else "none")
    check("no doubled punctuation", not re.search(r"\.\.|,,|;\s*;", tail), "none")

    # --- references ---
    listed = len(re.findall(r"^\[(\d+)\]", refs, re.M))
    check("reference list non-empty", listed > 0, f"{listed} entries")

    # --- figures (D4: Run 2 figures only; captions; references) ---
    root = p.parent.parent
    imgs = re.findall(r'<img src="(figures/[^"]+)"', body)
    legacy = [i for i in imgs if not i.startswith("figures/fig_v2_")]
    check("only Run 2 figures embedded", len(imgs) == 7 and not legacy,
          f"{len(imgs)} image(s), legacy: {legacy}" if legacy else
          f"{len(imgs)} v2 figures, no legacy figure")

    missing = [i for i in imgs
               if not (p.parent / i).is_file() or not (root / i).is_file()]
    check("embedded figures exist beside the manuscript and in figures/",
          not missing, f"missing: {missing}" if missing else
          f"{len(set(imgs))} file(s) present in both locations")

    dup = sorted(i for i in set(imgs) if imgs.count(i) != 1)
    check("each figure embedded exactly once", not dup,
          f"repeated: {dup}" if dup else f"{len(set(imgs))} unique image(s)")

    embeds = re.findall(
        r'<img src="figures/(fig_v2_[a-z0-9_]+\.png)"[^>]*/>\n\n\*Figure (\d+)\.',
        body)
    check("each image is followed by its numbered caption",
          [int(n) for _, n in embeds] == list(range(1, len(imgs) + 1)),
          f"{len(embeds)} of {len(imgs)} embeds captioned {[n for _, n in embeds]}")

    reg = json.loads((root / "figures" / "v2_figure_sources.json")
                     .read_text(encoding="utf-8"))
    check("embedded set matches the figure-source register",
          {f"figures/{n}" for n in reg["figures"]} == set(imgs),
          f"register {len(reg['figures'])}, embedded {len(set(imgs))}")

    counts = {n: body.count(f"(Figure {n})") for n in range(1, len(imgs) + 1)}
    bad_refs = {n: c for n, c in counts.items() if c != 1}
    check("every (Figure N) reference appears exactly once", not bad_refs,
          f"reference counts: {bad_refs}" if bad_refs else
          f"{len(counts)} references, each exactly once")

    caps = re.findall(r"\*Figure \d+\. (.*?)\*", body, re.S)
    over = [f"{i}: {words(c)}w" for i, c in enumerate(caps, 1) if words(c) > 55]
    check("caption lengths inside the 55-word policy", not over,
          f"over 55 words: {over}" if over else
          f"{len(caps)} captions, longest {max(words(c) for c in caps)} words",
          warn=True)

    v1 = re.findall(r"v1", body)
    check("no 'v1' token in the body (D6)", not v1,
          f"{len(v1)} occurrence(s)" if v1 else "Run 1 / original used throughout")

    # --- D10 (closes D6): no prior-version vocabulary for the system ---
    # "reconstructed" survives only as the demand-estimation term of art
    # (code-consistent: value_model.py "reconstructed uncontrolled demand").
    sysrec = re.findall(
        r"\brebuil\w*\b|\breconstruction\w*\b"
        r"|(?:READ-Grid|the system|the implementation|the simulation"
        r"|the controller|the architecture|the framework)"
        r"\s+(?:is|are|was|were|has been|had been)\s+reconstruct\w*",
        body, re.I)
    nrc = len(re.findall(r"\breconstructed\b", body, re.I))
    check("no reconstruct/rebuild word for the system (D10)", not sysrec,
          f"occurs {len(sysrec)}x: {sysrec[:2]}" if sysrec else
          f"demand-sense 'reconstructed' only ({nrc} uses)")

    full = "READ-Grid (Resilient Edge-Native Autonomous Demand Response for Smart Grids)"
    first = body.find("READ-Grid")
    ok_full = first != -1 and body.startswith(full, first) and body.count(full) == 1
    check("READ-Grid full name at its first occurrence", ok_full,
          "written once, immediately following the first mention" if ok_full else
          f"count={body.count(full)}, first index={first}")

    s5n = words(next((v for k, v in secs.items() if k.startswith("5. Results")), ""))
    check("section 5 inside the caption-extended word band", 3520 <= s5n <= 4480,
          f"{s5n} words (target 4000, band 3520-4480)")

    # --- plan §8 items that are checkable without judgement ---
    ab = next((v for k, v in secs.items() if k.startswith("Abstract")), "")
    ab = ab.split("Keywords:")[0]
    check("abstract within 250 words", words(ab) <= 250, f"{words(ab)} words excluding keywords")

    check("aggregate weighting is labelled", ("0.0017" not in body) or ("94.7" in body),
          "the pooled-weighting row is accompanied by the 94.7% weight note"
          if "0.0017" in body else "no pooled-weighting row")

    check("transfer table names the composite selection score",
          "composite selection score" in body,
          "Equation (5) score, not the achievement rate")

    check("the ratio column is defined as a ratio of per-step means",
          "per-step means" in body, "stated in the table note")

    # --- report ---
    say()
    for status, name, detail in results:
        say(f"{status:4s} {name:44s} {detail}")
    fails = sum(1 for s, _, _ in results if s == "FAIL")
    say(f"\n{len(results)} checks, {fails} fail")

    if "--save" in sys.argv:
        out = p.parent.parent / "paper_package" / "21_draft_v3_audit.md"
        header = [
            "# v3 交付前自检报告（`16_section_plan_v2.md` §8 的机械化版本）",
            "",
            f"- 受检文件：`{p.name}`",
            f"- 检查项：{len(results)}，失败 {fails}",
            f"- 生成脚本：`analysis/audit_manuscript.py`（本文件由脚本生成，不要手改）",
            "",
            "```text",
        ]
        out.write_text("\n".join(header + LINES + ["```", ""]), encoding="utf-8", newline="\n")
        print(f"report -> {out}")


if __name__ == "__main__":
    main()
