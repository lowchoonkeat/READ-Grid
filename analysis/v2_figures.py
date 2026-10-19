"""v2 figures.

Every figure is regenerated from the v2 artefacts on disk. Nothing is entered by
hand: every value is read or recomputed here, and every quantitative claim a
figure makes is guarded by a fail-loud check recorded in
``figures/v2_figure_sources.json``.

The Run 1 figures (``figures/fig_*.png``) are retained untouched and are NOT
regenerated, because their producing source is not recoverable
(``docs/v1_source_provenance.md``).

Figures:
  fig_v2_training_ladder.png   the pre-registered training ladder (12 rows x 5 seeds)
  fig_v2_val_arms.png          validation-split arms, 5-seed means, both criteria
  fig_v2_gap_closure.png       archived Run 1 shortfall vs the Run 2 validation
  fig_v2_baselines.png         the strengthened baseline set vs the learner
  fig_v2_primary_forest.png    held-out paired contrasts (equal-weight and pooled)
  fig_v2_outage_coverage.png   DR-step density vs the configured outage windows
  fig_v2_transfer.png          transfer: learner minus per-condition re-tuned rule

Usage:
    python analysis/v2_figures.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

RUNS = ROOT / "runs_v2"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# a restrained, print-friendly palette; no colour encodes a finding by itself
C_MAIN = "#2c5f8d"
C_ACCENT = "#b5651d"
C_REF = "#6b7280"
C_POS = "#2f7d4f"
C_NEG = "#a4303f"
GRID = dict(axis="y", alpha=0.3, linewidth=0.6)

SOURCES: dict[str, str] = {}
CHECKS: dict[str, object] = {}


def _check(name: str, ok: bool, detail: object = "") -> None:
    """Fail loud: a figure must not be written if any check fails."""
    if not ok:
        raise SystemExit(f"CHECK FAILED: {name}  [{detail}]")
    CHECKS[name] = detail if detail != "" else True
    print(f"  ok  {name}" + (f"  [{detail}]" if detail != "" else ""))


def _save(fig, name: str, source: str) -> None:
    fig.savefig(FIGS / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    SOURCES[name] = source
    print(f"  wrote figures/{name}")


def _minus(s: str) -> str:
    """Typographic minus for display text (the console stays ASCII)."""
    return s.replace("-", "\u2212")


# ------------------------------------------------ 1. validation arms (Table 2)
def fig_val_arms() -> None:
    d = json.loads((RUNS / "validation_scores_v2.json").read_text(encoding="utf-8"))
    by = {r["policy_id"]: r for r in d}

    def agg(prefix):
        rows = [by[f"{prefix}_s{k}"] for k in range(5)]
        return (float(np.mean([r["score"] for r in rows])),
                float(np.mean([r["achievement_rate"] for r in rows])))

    # the values published in Table 2 of the manuscript; the figure must
    # reproduce them from the per-seed records, not restate them
    TABLE2 = {
        "C0": (0.1105, 0.1105, 1), "C1": (0.5371, 0.5689, 1),
        "C2": (0.4349, 0.4673, 5), "C3": (0.5342, 0.5659, 1),
        "C3b": (0.5423, 0.5747, 1), "C4": (0.4402, 0.4731, 5),
        "C5": (0.1230, 0.1293, 5), "C7": (0.5413, 0.5734, 1),
        "C2v2": (0.5365, 0.5681, 5), "C4v2": (0.5341, 0.5661, 5),
    }
    rows, mism = [], []
    for arm, (s_lit, a_lit, n) in TABLE2.items():
        if n > 1:
            s, a = agg(arm)
        else:
            s, a = float(by[arm]["score"]), float(by[arm]["achievement_rate"])
        if f"{s:.4f}" != f"{s_lit:.4f}" or f"{a:.4f}" != f"{a_lit:.4f}":
            mism.append(f"{arm}: {s:.4f}/{a:.4f} vs {s_lit:.4f}/{a_lit:.4f}")
        rows.append((arm, s, a, n))
    _check("val_arms: Table 2 score/achievement values reproduced",
           not mism, f"{len(TABLE2) * 2 - len(mism)}/{len(TABLE2) * 2}")
    rows.sort(key=lambda r: -r[1])

    labels = [r[0] for r in rows]
    score = [r[1] for r in rows]
    ach = [r[2] for r in rows]
    ns = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, score, width=0.4, color=C_MAIN, label="selection score (Eq. 5)")
    ax.bar(x + 0.2, ach, width=0.4, color=C_ACCENT, label="raw achievement rate")
    for xi, s, a, n in zip(x, score, ach, ns):
        ax.text(xi - 0.2, s + 0.006, f"{s:.4f}", ha="center", fontsize=7.5)
        # C0 has identical values and C5 differs by 0.0063, so their paired
        # labels sit almost at the same height; stagger those two pairs
        dy = 0.024 if abs(s - a) < 0.008 else 0.006
        ax.text(xi + 0.2, a + dy, f"{a:.4f}", ha="center", fontsize=7.5)
        if n > 1:
            ax.text(xi, -0.055, f"{n} seeds", ha="center", fontsize=7,
                    color=C_REF)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.10, 0.74)
    ax.set_ylabel("value")
    ax.set_title("Corrected policy set on the validation split (21 episodes): "
                 "both selection criteria shown", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(**GRID)
    ax.axhline(0, color="black", linewidth=0.6)
    fig.text(0.01, -0.02,
             "source: runs_v2/validation_scores_v2.json (learned arms are "
             "five-seed means)", fontsize=7, color=C_REF)
    _save(fig, "fig_v2_val_arms.png", "runs_v2/validation_scores_v2.json")


# ------------------------------------- 2. gap closure (Run 1 archive vs Run 2)
def fig_gap_closure() -> None:
    # the archived Run 1 paired difference, read live from the analysis table
    csv_path = ROOT / "analysis" / "tables" / "main_seedcollapsed_paired_comparisons.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    ix = {k: i for i, k in enumerate(lines[0].split(","))}
    row = None
    for ln in lines[1:]:
        f = ln.split(",")
        if (f[ix["comparison"]] == "READ-Grid vs guarded rule fallback"
                and f[ix["metric"]] == "achievement_rate"):
            row = f
            break
    if row is None:
        raise SystemExit("CHECK FAILED: gap_closure: archived CSV row not found")
    v1_m = float(row[ix["mean_diff"]])
    v1_lo = float(row[ix["ci_low"]])
    v1_hi = float(row[ix["ci_high"]])
    _check("gap_closure: archived C4-C3 rounds to -0.1332 [-0.2038, -0.0698]",
           f"{v1_m:+.4f}" == "-0.1332" and f"{v1_lo:+.4f}" == "-0.2038"
           and f"{v1_hi:+.4f}" == "-0.0698",
           f"mean={v1_m:+.4f} ci=[{v1_lo:+.4f}, {v1_hi:+.4f}]")

    d = json.loads((RUNS / "validation_scores_v2.json").read_text(encoding="utf-8"))
    by = {r["policy_id"]: r for r in d}

    def mean(prefix, key):
        return float(np.mean([by[f"{prefix}_s{k}"][key] for k in range(5)]))

    v2_c3 = mean("C4v2", "achievement_rate") - float(by["C3"]["achievement_rate"])
    v2_c3b = mean("C4v2", "achievement_rate") - float(by["C3b"]["achievement_rate"])
    _check("gap_closure: Run 2 validation differences +0.0002 / -0.0086",
           f"{v2_c3:+.4f}" == "+0.0002" and f"{v2_c3b:+.4f}" == "-0.0086",
           f"C4v2-C3={v2_c3:+.4f} C4v2-C3b={v2_c3b:+.4f}")

    pairs = [
        ("C4 \u2212 C3\n(Run 1)", v1_m),
        ("C4v2 \u2212 C3\n(Run 2 validation)", v2_c3),
        ("C4v2 \u2212 C3b\n(Run 2 validation)", v2_c3b),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(pairs))
    vals = [p[1] for p in pairs]
    cols = [C_NEG if v < 0 else C_POS for v in vals]
    ax.bar(x, vals, width=0.5, color=cols)
    ax.errorbar([0], [v1_m], yerr=[[v1_m - v1_lo], [v1_hi - v1_m]],
                fmt="none", ecolor="black", elinewidth=1.1, capsize=4)
    ax.text(0.30, v1_m, _minus(f"{v1_m:+.4f}"), fontsize=9, fontweight="bold",
            va="center")
    ax.text(0.30, v1_m - 0.014, _minus(f"[{v1_lo:+.4f}, {v1_hi:+.4f}]"),
            fontsize=7.5, color=C_REF, va="center")
    ax.text(1, v2_c3 + 0.006, _minus(f"{v2_c3:+.4f}"), ha="center", fontsize=9,
            fontweight="bold")
    ax.text(2, v2_c3b - 0.016, _minus(f"{v2_c3b:+.4f}"), ha="center", fontsize=9,
            fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=9)
    ax.set_ylabel("achievement-rate difference (A \u2212 B)")
    ax.set_ylim(-0.225, 0.035)
    ax.set_title("Learner-minus-rule achievement-rate difference:\nthe archived "
                 "Run 1 shortfall and the Run 2 validation split", fontsize=10)
    ax.grid(**GRID)
    fig.text(0.01, -0.03,
             "sources: analysis/tables/main_seedcollapsed_paired_comparisons.csv "
             "(Run 1, episode-clustered 95% interval) + "
             "runs_v2/validation_scores_v2.json (Run 2, point estimates)",
             fontsize=7, color=C_REF)
    _save(fig, "fig_v2_gap_closure.png",
          "analysis/tables/main_seedcollapsed_paired_comparisons.csv + "
          "runs_v2/validation_scores_v2.json")


# ------------------------------------- 3. the pre-registered training ladder
LADDER_PAIRS = [("L0", "V0"), ("L1a", "V1"), ("L1b", "V2"), ("L1c", "V3"),
                ("L2", "V4"), ("L3a", "M1"), ("L3b", "M2"), ("L4a", "M4"),
                ("L4b", "M5"), ("L4c", "M6")]

RX_LEARNER = re.compile(
    r"^\s+(L\w+)\s+\S+\s+seed\s+(\d+)\s+ach\s+([\d.]+)\s+score\s+([\d.]+)\s*$")
RX_VM = re.compile(r"^([VM]\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")


def _read_learner_log(path: Path) -> dict:
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RX_LEARNER.match(ln)
        if m:
            out[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


def _read_vm_logs(paths) -> dict:
    out = {}
    for path in paths:
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = RX_VM.match(ln)
            if m:
                out[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return out


def fig_ladder() -> None:
    lad = json.loads((RUNS / "s3_ladder.json").read_text(encoding="utf-8"))
    rows = lad["rows"] if isinstance(lad, dict) and "rows" in lad else lad
    _check("ladder: 12 rows, five seeds each",
           len(rows) == 12 and all(len(r["seeds"]) == 5 for r in rows),
           f"{len(rows)} rows")

    # the sandbox text logs are the producing record; the figure may not be
    # drawn unless the JSON still agrees with them (the reproduction claim
    # itself stays in Section 5.1, not in the figure)
    lg = _read_learner_log(RUNS / "sandbox_logs" / "s3_learner.txt")
    vm = _read_vm_logs([RUNS / "sandbox_logs" / "ablate.txt",
                        RUNS / "sandbox_logs" / "mask_rerun.txt"])
    codes = [r["row"].split()[0] for r in rows]
    missing = [f"{c}/{s}" for c in codes for s in range(5) if (c, s) not in lg]
    missing += [f"{n}s{s}" for n, v in LADDER_PAIRS for s in range(3)
                if (v, s) not in vm]
    _check("ladder: both logs carry every needed row", not missing,
           f"{len(missing)} missing")

    cross = [f"{n}/{v} seed {s}" for n, v in LADDER_PAIRS for s in range(3)
             if f"{lg[(n, s)][0]:.4f}" != f"{vm[(v, s)]:.4f}"]
    _check("ladder: sandbox logs agree across programs (10 pairs x 3 seeds)",
           len(cross) == 0, f"{30 - len(cross)}/30")
    mult = []
    for r in rows:
        c = r["row"].split()[0]
        if sorted(f"{p['achievement_rate']:.4f}" for p in r["per_seed"]) != \
           sorted(f"{lg[(c, s)][0]:.4f}" for s in range(5)):
            mult.append(f"{c} ach")
        if sorted(f"{p['score']:.4f}" for p in r["per_seed"]) != \
           sorted(f"{lg[(c, s)][1]:.4f}" for s in range(5)):
            mult.append(f"{c} score")
    _check("ladder: JSON per-seed values match the logs (12 rows x 5, as multisets)",
           not mult, f"{24 - len(mult)}/24 row-metrics")

    l0 = float(next(r for r in rows if r["row"].startswith("L0 "))["score_mean"])
    _check("ladder: L0 reference score is 0.4346", f"{l0:.4f}" == "0.4346",
           f"L0={l0:.4f}")

    order = sorted(rows, key=lambda r: float(r["score_mean"]))
    names, descs, vals = [], [], []
    for r in order:
        code, desc = r["row"].split(" ", 1)
        desc = desc.strip().replace("v1-like", "original-style")
        names.append(code)
        descs.append(desc)
        vals.append(float(r["score_mean"]))
    _check("ladder: displayed labels carry no v1 token",
           not any("v1" in d for d in descs),
           f"{sum('v1' in d for d in descs)}/12 labels")
    _check("ladder: every row carries a configuration code",
           len(set(names)) == 12 and all(n.startswith("L") for n in names),
           f"{len(set(names))} codes")

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    y = np.arange(len(names))
    ax.barh(y, vals, color=C_MAIN, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi, f"{v:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.axvline(l0, color=C_REF, linestyle="--", linewidth=1.0)
    ax.text(l0 + 0.007, 1.5, "L0 original-style reference",
            fontsize=7.5, color=C_REF, va="center", ha="left")
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("selection score, five-seed mean")
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    fig.text(0.01, -0.02,
             "source: runs_v2/s3_ladder.json, cross-checked live against the "
             "sandbox logs.\nSandbox scores from the normalised single-group "
             "replica; not comparable to Tables 3 and 4.",
             fontsize=7, color=C_REF)
    _save(fig, "fig_v2_training_ladder.png", "runs_v2/s3_ladder.json")


# ------------------------- 4. outage coverage (why the extension fault exists)
def fig_outage_coverage() -> None:
    import yaml
    from readgrid.workload import load_episodes

    ec = json.loads((RUNS / "s7_ec.json").read_text(encoding="utf-8"))
    ov = ec["dr_overlap"]
    lo_s, hi_s = (float(v) for v in ov["f3_window_s"])
    _check("outage: frozen F3 window is 18:30-20:30",
           (lo_s, hi_s) == (66600.0, 73800.0),
           f"window_s=[{lo_s:.0f}, {hi_s:.0f}]")
    thr = float(ov["dr_trigger_fraction"])
    _check("outage: DR-step trigger fraction is 0.8", thr == 0.8,
           f"fraction={thr}")
    n_full = int(ov["n_fully_covered"])
    n_full_ep = sum(1 for e in ov["per_episode"] if e["dr_fully_covered"])
    _check("outage: no episode's DR window is fully covered",
           n_full == 0 and n_full_ep == 0,
           f"top-level={n_full}, per-episode={n_full_ep}/{len(ov['per_episode'])}")

    # the windows must agree with the protocol files that defined them
    proto = yaml.safe_load((ROOT / "configs" / "protocol_v2.yaml")
                           .read_text(encoding="utf-8"))
    f3 = proto["faults"]["F3"]
    _check("outage: s7_ec window agrees with protocol_v2.yaml F3",
           float(f3["start_s"]) == lo_s
           and float(f3["start_s"] + f3["duration_s"]) == hi_s,
           f"F3 start={f3['start_s']} dur={f3['duration_s']}")
    ext = yaml.safe_load((ROOT / "configs" / "protocol_v2_ec.yaml")
                         .read_text(encoding="utf-8"))
    f3e = ext["faults"]["F3E"]
    _check("outage: F3E extension window is 17:00-24:00",
           f3e["kind"] == "outage" and float(f3e["start_s"]) == 61200.0
           and float(f3e["duration_s"]) == 25200.0,
           f"F3E start={f3e['start_s']} dur={f3e['duration_s']}")

    # overlap recomputed live from the same shapes, so the drawn density and
    # the recorded numbers can never disagree
    split = json.loads((ROOT / "runs" / "workload_split.json")
                       .read_text(encoding="utf-8"))
    test_ids = set(split["test"])
    eps_data = load_episodes()
    hist = np.zeros(1440)
    inside = tot = 0
    lo_i, hi_i = int(lo_s) // 60, int(hi_s) // 60
    for eid in test_ids:
        sh = np.asarray(eps_data[eid].shape, dtype=float)
        dr = sh >= thr
        hist += dr.astype(float)
        tot += int(dr.sum())
        inside += int(dr[lo_i:hi_i].sum())
    frac = inside / tot if tot else float("nan")
    byt = ov["by_split"]["test"]
    _check("outage: test-split overlap reproduced live",
           (inside, tot) == (int(byt["n_dr_steps_inside_F3"]),
                             int(byt["n_dr_steps"]))
           and abs(frac - float(byt["dr_step_overlap_fraction"])) < 1e-12,
           f"{inside}/{tot} = {frac:.4f}")

    hours = np.arange(1440) / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    ax = axes[0]
    ax.fill_between(hours, hist, color=C_MAIN, alpha=0.85)
    ax.set_ylabel("DR steps\n(21 held-out\nepisodes)")
    ax.set_title("Demand-response steps by hour of day (21 held-out episodes)\n"
                 "and the configured outage windows used in the study",
                 fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.6)

    ax2 = axes[1]
    ax2.barh([0.55], [(hi_s - lo_s) / 3600.0], left=[lo_s / 3600.0], height=0.22,
             color=C_NEG, label="F3 (frozen) 18:30\u201320:30")
    ax2.barh([0.15], [float(f3e["duration_s"]) / 3600.0],
             left=[float(f3e["start_s"]) / 3600.0], height=0.22, color=C_ACCENT,
             label="F3E (extension) 17:00\u201324:00")
    ax2.set_yticks([])
    ax2.set_xlim(0, 24)
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.set_xlabel("hour of simulated day")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.grid(axis="x", alpha=0.3, linewidth=0.6)

    fig.text(0.01, -0.02,
             f"source: workload shapes + runs/workload_split.json + "
             f"runs_v2/s7_ec.json; F3 covers {inside}/{tot} = {frac:.4f} of "
             f"test-split DR steps ({frac * 100:.1f} %); "
             f"{n_full_ep}/{len(ov['per_episode'])} episodes fully covered",
             fontsize=7, color=C_REF)
    _save(fig, "fig_v2_outage_coverage.png",
          "runs_v2/s7_ec.json + runs/workload_split.json + "
          "configs/protocol_v2.yaml (F3) + configs/protocol_v2_ec.yaml (F3E) + "
          "workload shapes")


# ---------------------------------- 5. E-A transfer: learner vs re-tuned rule
# condition -> (mean, ci_low, ci_high, resolved), the published rounded values
TRANSFER_EXPECT = [
    ("nominal",     "-0.0082", "-0.0283", "+0.0126", False),
    ("delta_0.08",  "+0.0212", "-0.0104", "+0.0482", False),
    ("delta_0.16",  "+0.0400", "+0.0266", "+0.0548", True),
    ("budget_0.02", "-0.0116", "-0.0225", "-0.0006", True),
    ("budget_0.04", "+0.0139", "-0.0050", "+0.0325", False),
    ("budget_0.06", "+0.0328", "+0.0194", "+0.0451", True),
]
TRANSFER_LABELS = {
    "nominal": "nominal",
    "delta_0.08": "target relaxed, \u03b4_DR = 8%",
    "delta_0.16": "target tightened, \u03b4_DR = 16%",
    "budget_0.02": "budget 2%",
    "budget_0.04": "budget 4%",
    "budget_0.06": "budget 6%",
}


def fig_transfer() -> None:
    iv = json.loads((RUNS / "s7_ea_intervals.json").read_text(encoding="utf-8"))
    res6 = json.loads((RUNS / "s6_results.json").read_text(encoding="utf-8"))
    delta = float(res6["delta_frozen"])
    conds = iv["conditions"]
    _check("transfer: six conditions, one learner seed, frozen bootstrap",
           len(conds) == 6 and iv["learner_seeds"] == ["C4v2_s0"]
           and int(iv["bootstrap"]["n_resamples"]) == 5000
           and float(iv["bootstrap"]["ci_level"]) == 0.95 and delta == 0.02,
           f"conditions={len(conds)} seeds={iv['learner_seeds']} delta={delta}")
    mism = []
    for c in conds:
        exp = next((e for e in TRANSFER_EXPECT if e[0] == c["condition"]), None)
        if exp is None:
            mism.append(f"{c['condition']}: unexpected condition")
            continue
        got = (f"{float(c['mean_diff']):+.4f}", f"{float(c['ci_low']):+.4f}",
               f"{float(c['ci_high']):+.4f}", bool(c["resolved"]))
        if got != exp[1:]:
            mism.append(f"{c['condition']}")
        if int(c["n_units"]) != 21:
            mism.append(f"{c['condition']}:n")
    _check("transfer: 6 conditions x (3 values + resolved + n_units)",
           not mism, f"{30 - len(mism)}/30")

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.axvspan(-delta, delta, color=C_REF, alpha=0.16, zorder=0)
    ypos = list(range(len(conds)))[::-1]
    for yi, c in zip(ypos, conds):
        m = float(c["mean_diff"])
        lo = float(c["ci_low"])
        hi = float(c["ci_high"])
        ax.errorbar([m], [yi], xerr=[[m - lo], [hi - m]], fmt="o", ms=7.5,
                    mfc=(C_MAIN if c["resolved"] else "white"), mec=C_MAIN,
                    ecolor=C_MAIN, elinewidth=1.2, capsize=3, zorder=3)
        ax.text(hi + 0.0035, yi, _minus(f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"),
                va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([TRANSFER_LABELS[c["condition"]] for c in conds],
                       fontsize=9)
    ax.set_ylim(-0.6, len(conds) - 0.4)
    ax.set_xlim(-0.045, 0.096)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("selection-score difference (learner \u2212 re-tuned rule)")
    ax.set_title("Transfer across operating points: learner, never retrained, "
                 "minus the per-condition re-tuned rule\nselection score, "
                 "episode-clustered 95% intervals (post-hoc)", fontsize=10)
    handles = [
        Line2D([], [], marker="o", linestyle="none", mfc=C_MAIN, mec=C_MAIN,
               ms=7, label="interval excludes zero"),
        Line2D([], [], marker="o", linestyle="none", mfc="white", mec=C_MAIN,
               ms=7, label="interval crosses zero"),
        Patch(facecolor=C_REF, alpha=0.16,
              label=f"equivalence zone \u00b1\u03b4 = {delta:.2f}"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    fig.text(0.01, -0.03,
             "source: runs_v2/s7_ea_intervals.json (post-hoc, one learner seed, "
             "five deterministic faults, 21 episodes; no retraining)",
             fontsize=7, color=C_REF)
    _save(fig, "fig_v2_transfer.png",
          "runs_v2/s7_ea_intervals.json (post-hoc intervals)")


# --------------------------------------- 6. the strengthened baseline set
def fig_baselines() -> None:
    d = json.loads((RUNS / "validation_scores_v2.json").read_text(encoding="utf-8"))
    by = {r["policy_id"]: r for r in d}

    def m(prefix, key):
        return float(np.mean([by[f"{prefix}_s{k}"][key] for k in range(5)]))

    vals = {"C7": float(by["C7"]["achievement_rate"]),
            "C3b": float(by["C3b"]["achievement_rate"]),
            "C3": float(by["C3"]["achievement_rate"]),
            "C4v2": m("C4v2", "achievement_rate"),
            "C4": m("C4", "achievement_rate")}
    expect = {"C7": "0.5734", "C3b": "0.5747", "C3": "0.5659",
              "C4v2": "0.5661", "C4": "0.4731"}
    bad = [k for k in expect if f"{vals[k]:.4f}" != expect[k]]
    _check("baselines: validation achievement values reproduce", not bad,
           f"{len(expect) - len(bad)}/{len(expect)} values")

    t_rule = json.loads((RUNS / "s4_rule_tuning_table.json")
                        .read_text(encoding="utf-8"))
    t_learn = json.loads((RUNS / "s3_learner_tuning_table.json")
                         .read_text(encoding="utf-8"))
    _check("baselines: both tuning tables carry 144 candidates",
           len(t_rule) == 144 and len(t_learn) == 144,
           f"rule={len(t_rule)} learner={len(t_learn)}")

    rows = [
        ("C7 parameter-free\n(no tuning budget)", vals["C7"]),
        ("C3b rule++\n(144 candidates)", vals["C3b"]),
        ("C3 original rule\n(hand-set)", vals["C3"]),
        ("C4v2 learner\n(144 candidates, 5-seed mean)", vals["C4v2"]),
        ("C4 original learner\n(5-seed mean)", vals["C4"]),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(len(rows))
    cols = [C_MAIN if "learner" not in lab else C_ACCENT for lab, _ in rows]
    ax.bar(x, [v for _, v in rows], width=0.55, color=cols)
    for xi, (lab, v) in zip(x, rows):
        ax.text(xi, v + 0.006, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in rows], fontsize=8)
    ax.set_ylim(0, 0.64)
    ax.set_ylabel("raw achievement rate (validation split)")
    ax.set_title("Baseline set on the validation split\n"
                 "the tuned rule and the learner evaluated the same 144 "
                 "candidates", fontsize=10)
    ax.grid(**GRID)
    fig.text(0.01, -0.03,
             "source: runs_v2/validation_scores_v2.json; candidate budgets from "
             "runs_v2/s4_rule_tuning_table.json + "
             "runs_v2/s3_learner_tuning_table.json", fontsize=7, color=C_REF)
    _save(fig, "fig_v2_baselines.png",
          "runs_v2/validation_scores_v2.json + runs_v2/s4_rule_tuning_table.json "
          "+ runs_v2/s3_learner_tuning_table.json")


# ----------------------------- 7. held-out paired contrasts (forest plot)
def _equal_fault_arm(df: pd.DataFrame, arm: str, seeds: tuple):
    """Per-episode frame with equal weight per fault id; seed rows averaged.

    This is the pre-registered weighting (each fault id contributes equally
    within an episode), recomputed here so the figure's primary row is a live
    reproduction, not a table lookup.
    """
    def per_episode_equal_fault(pid: str):
        sub = df[df["policy_id"] == pid]
        vals = {}
        for eid, g in sub.groupby("episode_id"):
            rates = []
            for _, h in g.groupby("fault_id"):
                el = h["eligible_steps"].sum()
                if el:
                    rates.append(h["achieved_steps"].sum() / el)
            vals[eid] = float(np.mean(rates)) if rates else float("nan")
        return pd.Series(vals, dtype=float)

    if not seeds:
        return per_episode_equal_fault(arm)
    frames = [per_episode_equal_fault(f"{arm}_s{s}") for s in seeds]
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    stacked = np.stack([f.loc[common].to_numpy(dtype=float) for f in frames])
    return pd.Series(stacked.mean(axis=0), index=common, dtype=float)


def _classify(low: float, high: float, delta: float) -> str:
    """Six-condition classification with the D-rule (copy of s6_analysis_v2)."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return "inconclusive"
    half = (high - low) / 2.0
    power_ok = half <= delta
    if low > 0.0:
        label = "superior"
    elif (low > -delta) and (high < delta):
        label = "equivalent"
    elif low > -delta:
        label = "non-inferior"
    elif high < -delta:
        label = "inferior beyond margin"
    elif high < 0.0:
        label = "inferior"
    else:
        label = "inconclusive"
    if (not power_ok) and label in ("non-inferior", "equivalent", "inconclusive"):
        label = "inconclusive (insufficient power for delta)"
    return label


# classification label -> the tag shown on the figure
TAG = {"superior": "superior", "equivalent": "equivalent",
       "inferior beyond margin": "inferior\u2020",
       "inconclusive (insufficient power for delta)": "inconclusive*"}


def fig_primary_forest() -> None:
    import analyze as az
    from readgrid.protocol import load_protocol

    res = json.loads((RUNS / "s6_results.json").read_text(encoding="utf-8"))
    delta = float(res["delta_frozen"])
    _check("forest: delta frozen at 0.02", delta == 0.02, f"delta={delta}")

    # --- the equal-weight-per-fault recount of the primary contrast, live
    prot = load_protocol(ROOT / "configs" / "protocol_v2.yaml")
    n_res = int(prot.raw["analysis"]["bootstrap_resamples"])
    ci = float(prot.raw["analysis"]["ci_level"])
    df = az.load_rows(RUNS / "stage_main_v2.jsonl")
    fa = _equal_fault_arm(df, "C4v2", tuple(range(5)))
    fb = _equal_fault_arm(df, "C3b", ())
    r = az.cluster_bootstrap_diff(
        pd.DataFrame({"achievement_rate": fa}),
        pd.DataFrame({"achievement_rate": fb}),
        "achievement_rate", n_resamples=n_res, ci_level=ci,
        seed_key=("primary", "C4v2", "C3b"))
    rm = float(r["mean_diff"])
    rl = float(r["ci_low"])
    rh = float(r["ci_high"])
    recount_half = (rh - rl) / 2.0
    _check("forest: equal-weight recount reproduces -0.0040 [-0.0239, +0.0174]",
           int(r["n_units"]) == 21 and f"{rm:+.4f}" == "-0.0040"
           and f"{rl:+.4f}" == "-0.0239" and f"{rh:+.4f}" == "+0.0174"
           and f"{recount_half:.4f}" == "0.0207",
           f"n={r['n_units']} mean={rm:+.4f} [{rl:+.4f}, {rh:+.4f}] "
           f"half={recount_half:.4f}")

    # --- the stored rows, and the classification cross-check
    def find(pool, a, b):
        for rr in pool:
            if rr.get("a") == a and rr.get("b") == b:
                return rr
        raise SystemExit(f"CHECK FAILED: forest: row {a} vs {b} absent")

    sec = res["secondary_contrasts"]
    used = [(a, b, find(sec, a, b)) for a, b in
            [("C4v2", "C2v2"), ("C3b", "C3"), ("C7", "C3b"), ("C4v2", "C4"),
             ("C4v2", "C5"), ("C4v2", "C0"), ("C4", "C3")]]
    prim = res["primary_contrast"][0]
    used = [("C4v2", "C3b", prim)] + used
    bad = [f"{a}-{b}" for a, b, rr in used
           if _classify(float(rr["ci_low"]), float(rr["ci_high"]), delta)
           != rr["classification"]["label"]]
    _check("forest: local classification matches stored labels "
           "(1 primary + 7 secondary)", not bad, f"{len(used) - len(bad)}/8")
    _check("forest: recount classification is the D-rule downgrade",
           _classify(rl, rh, delta) == "inconclusive (insufficient power for delta)",
           _classify(rl, rh, delta))

    EXPECT = {
        ("C4v2", "C3b"): ("-0.0017", "-0.0218", "+0.0200"),
        ("C4v2", "C2v2"): ("+0.0057", "+0.0009", "+0.0121"),
        ("C3b", "C3"): ("+0.0058", "-0.0070", "+0.0174"),
        ("C7", "C3b"): ("+0.0049", "-0.0023", "+0.0162"),
        ("C4v2", "C4"): ("+0.1152", "+0.0726", "+0.1619"),
        ("C4v2", "C5"): ("+0.4483", "+0.3664", "+0.5254"),
        ("C4v2", "C0"): ("+0.4597", "+0.3771", "+0.5405"),
        ("C4", "C3"): ("-0.1110", "-0.1644", "-0.0620"),
    }
    bad = [f"{a}-{b}" for a, b, rr in used
           if (f"{float(rr['mean_diff']):+.4f}", f"{float(rr['ci_low']):+.4f}",
               f"{float(rr['ci_high']):+.4f}") != EXPECT[(a, b)]]
    _check("forest: 8 stored rows reproduce the published run values",
           not bad, f"{24 - 3 * len(bad)}/24 rounded values")

    # --- assemble the two panels
    def disp(label, rr, filled=True):
        lab = TAG.get(rr["classification"]["label"])
        if lab is None:
            raise SystemExit("CHECK FAILED: forest: unmapped classification "
                             f"'{rr['classification']['label']}'")
        return {"label": label, "m": float(rr["mean_diff"]),
                "lo": float(rr["ci_low"]), "hi": float(rr["ci_high"]),
                "tag": lab, "filled": filled}

    bypair = {(a, b): rr for a, b, rr in used}
    panel_a = [
        disp("C3b \u2212 C3\nrule++ vs original rule", bypair[("C3b", "C3")]),
        disp("C4v2 \u2212 C2v2\nvalue of the local fallback",
             bypair[("C4v2", "C2v2")]),
        disp("C7 \u2212 C3b\nparameter-free vs rule++", bypair[("C7", "C3b")]),
        disp("C4v2 \u2212 C3b\npooled eligible-step weight",
             bypair[("C4v2", "C3b")], filled=False),
        {"label": "C4v2 \u2212 C3b\nequal weight per fault id", "m": rm,
         "lo": rl, "hi": rh, "tag": TAG[_classify(rl, rh, delta)],
         "filled": True},
    ]
    panel_b = [
        disp("C4v2 \u2212 C0\nvs uncontrolled", bypair[("C4v2", "C0")]),
        disp("C4v2 \u2212 C5\nbinned vs concatenated", bypair[("C4v2", "C5")]),
        disp("C4v2 \u2212 C4\ntraining-pipeline correction",
             bypair[("C4v2", "C4")]),
        disp("C4 \u2212 C3\noriginal learner vs original rule",
             bypair[("C4", "C3")]),
    ]

    fig = plt.figure(figsize=(8.6, 7.4))
    # explicit rects keep the two right-hand text columns inside the canvas, so
    # the artwork is not widened by its own annotations
    axA = fig.add_axes([0.105, 0.560, 0.500, 0.300])
    axB = fig.add_axes([0.105, 0.140, 0.500, 0.240])

    def draw(ax, rws, xlim, outside=False):
        """Draw one panel; return the stat/tag texts so the two right-hand
        columns can be placed from measured widths (no hand-tuned offsets).

        ``outside=True`` puts both text columns beyond the right spine, which
        lets the axis itself be scaled to the data instead of being padded to
        leave room for the numbers.
        """
        n = len(rws)
        ys = np.arange(n)[::-1].astype(float)
        stx, tgx = [], []
        for yi, rw in zip(ys, rws):
            m, lo, hi = rw["m"], rw["lo"], rw["hi"]
            ax.errorbar([m], [yi], xerr=[[m - lo], [hi - m]], fmt="o", ms=7.5,
                        mfc=(C_MAIN if rw["filled"] else "white"), mec=C_MAIN,
                        ecolor=C_MAIN, elinewidth=1.2, capsize=3, zorder=3)
            if outside:
                stx.append((yi, hi, ax.text(
                    1.02, yi, _minus(f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"),
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7.5)))
            else:
                stx.append((yi, hi, ax.text(
                    0.60, yi, _minus(f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"),
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7.5)))
            tgx.append(ax.text(1.02 if outside else 0.995, yi, rw["tag"],
                               transform=ax.get_yaxis_transform(),
                               ha="left" if outside else "right",
                               va="center", fontsize=7.5, color=C_REF))
        ax.set_yticks(ys)
        ax.set_yticklabels([rw["label"] for rw in rws], fontsize=8)
        ax.set_ylim(-0.65, n - 0.35)
        ax.set_xlim(*xlim)
        ax.axvline(0, color="black", linewidth=0.8, zorder=1)
        ax.grid(axis="x", alpha=0.3, linewidth=0.6)
        ax.set_xlabel("achievement-rate difference (A \u2212 B)")
        return stx, tgx

    axA.axvspan(-delta, delta, color=C_REF, alpha=0.15, zorder=0)
    sa = draw(axA, panel_a, (-0.034, 0.030), outside=True)
    sb = draw(axB, panel_b, (-0.20, 0.60), outside=True)

    # the number column ends where the widest tag column begins, both measured
    # from the rendered figure, so no text pair can collide at any layout
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    for ax, (stx, tgx), outside in ((axA, sa, True), (axB, sb, True)):
        if outside:
            numcol = (max(t.get_window_extent(renderer=ren).width for _, _, t in stx)
                      / ax.bbox.width + 0.070)
            for (yi, _, _), t in zip(stx, tgx):
                t.set_position((1.02 + numcol, yi))
            continue
        tagcol = (max(t.get_window_extent(renderer=ren).width for t in tgx)
                  / ax.bbox.width + 0.022)
        for yi, _, t in stx:
            t.set_position((0.995 - tagcol, yi))
    fig.canvas.draw()
    clears = []
    for ax, (stx, _) in ((axA, sa), (axB, sb)):
        for yi, hi, t in stx:
            bb = t.get_window_extent(renderer=ren)
            hi_px = ax.transData.transform((hi, yi))[0]
            clears.append(bb.x0 - hi_px)
    _check("forest: number columns clear every interval and tag column "
           f"({len(panel_a) + len(panel_b)} rows)",
           min(clears) >= 8.0, f"min clearance {min(clears):.0f}px")
    axA.set_title("(a) comparisons at the margin (\u00b1\u03b4 shown)",
                  fontsize=9.5)
    axB.set_title("(b) comparisons far beyond \u03b4", fontsize=9.5)
    fig.suptitle("Held-out achievement-rate differences: 21 episodes, "
                 "episode-clustered 95% intervals", fontsize=11, y=0.985)
    handles = [
        Line2D([], [], marker="o", linestyle="none", mfc=C_MAIN, mec=C_MAIN,
               ms=7, label="equal weight per fault id"),
        Line2D([], [], marker="o", linestyle="none", mfc="white", mec=C_MAIN,
               ms=7, label="pooled eligible-step"),
        Patch(facecolor=C_REF, alpha=0.15,
              label=f"\u00b1\u03b4 = {delta:.2f} band (panel a)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.045))
    fig.text(0.02, -0.085,
             "* CI half-width exceeds \u03b4 = 0.02, so delta-defined conclusions "
             "are suppressed (D-rule).  \u2020 CI entirely below \u2212\u03b4.\n"
             "values: runs_v2/s6_results.json; the equal-weight recount is computed "
             "here from runs_v2/stage_main_v2.jsonl\n(episode-clustered bootstrap, "
             "seed_key ('primary','C4v2','C3b')).",
             ha="left", fontsize=7, color=C_REF)
    _save(fig, "fig_v2_primary_forest.png",
          "runs_v2/s6_results.json + runs_v2/stage_main_v2.jsonl "
          "(equal-weight recount)")


def main() -> int:
    print("v2 figures")
    fig_val_arms()
    fig_gap_closure()
    fig_ladder()
    fig_outage_coverage()
    fig_transfer()
    fig_baselines()
    fig_primary_forest()
    out = {
        "figures": sorted(SOURCES),
        "sources": SOURCES,
        "checks": CHECKS,
        "script": "analysis/v2_figures.py",
        "note": "regenerated from v2 artefacts; the Run 1 figures are NOT "
                "regenerated because their producing source is not recoverable "
                "(docs/v1_source_provenance.md)",
    }
    (FIGS / "v2_figure_sources.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote figures/v2_figure_sources.json  "
          f"({len(SOURCES)} figures, {len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
