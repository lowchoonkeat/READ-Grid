"""Phase 12: regenerate figures from the raw stage logs.

Every figure records the source run file it was built from in its caption, so no
value in the manuscript is hand-entered.

Run:  python analysis/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "analysis"))
FIG = ROOT / "figures"
RUNS = ROOT / "runs"

from analyze import per_episode, load_rows  # noqa: E402

POLICY_ORDER = ["C0", "C1", "C2_s0", "C3", "C4_s0", "C4_s1", "C4_s2", "C5_s0"]
PRETTY = {
    "C0": "C0 uncontrolled",
    "C1": "C1 cloud rule only",
    "C2_s0": "C2 cloud learned only",
    "C3": "C3 rule + rule fallback",
    "C4_s0": "C4 READ-Grid (s0)",
    "C4_s1": "C4 READ-Grid (s1)",
    "C4_s2": "C4 READ-Grid (s2)",
    "C5_s0": "C5 concatenated ablation",
}
C_RG = "#c0392b"      # READ-Grid emphasis (Chinese convention: red = the "good" one here is a label, not a price)
C_OTHER = "#4a6fa5"


def _read(path: Path) -> pd.DataFrame:
    return load_rows(path)


def _rel(p: Path) -> str:
    """Repo-relative path for a caption, robust to relative inputs."""
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def fig_achievement(df: pd.DataFrame, out: Path) -> None:
    pols = [p for p in POLICY_ORDER if p in set(df["policy_id"])]
    pe = {p: per_episode(df, p) for p in pols}
    rng = np.random.default_rng(7)
    means, los, his = [], [], []
    for p in pols:
        v = pe[p]["achievement_rate"].dropna().to_numpy()
        means.append(v.mean())
        b = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(axis=1)
        los.append(np.quantile(b, 0.025))
        his.append(np.quantile(b, 0.975))
    x = np.arange(len(pols))
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    colors = [C_RG if p.startswith("C4") else C_OTHER for p in pols]
    ax.bar(x, means, color=colors, width=0.62)
    ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                fmt="none", ecolor="#222", capsize=4, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(p, p) for p in pols], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("DR achievement rate\n(fraction of eligible steps at or below the cap)")
    n_ep = df["episode_id"].nunique()
    n_fc = df.groupby(["episode_id", "fault_id", "comm_seed"]).ngroups / max(n_ep, 1)
    ax.set_title("Held-out target achievement by policy\n"
                 f"source: {_rel(out)} | "
                 f"{n_ep} episodes x {n_fc:.0f} fault conditions each",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_achievement_by_policy.png")
    plt.close(fig)


def fig_by_fault(df: pd.DataFrame, out: Path) -> None:
    order = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
    fam = {
        "F0": "connected", "F1": "outage 5 min", "F2": "outage 30 min",
        "F3": "outage 120 min", "F4": "10% loss", "F5": "burst loss",
        "F6": "delay+jitter", "F7": "asymmetric", "F8": "flapping 5/5",
        "F9": "delayed dups", "F10": "stale cache",
    }
    focus = [p for p in ["C1", "C3", "C4_s0", "C5_s0"] if p in set(df["policy_id"])]
    pe = {p: per_episode(df, p) for p in focus}
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=150)
    w = 0.2
    for i, p in enumerate(focus):
        vals = []
        for f in order:
            sub = pe[p]
            idx = [e for e in sub.index]
            # per-episode achievement restricted to this fault: recompute from rows
            g = df[(df["policy_id"] == p) & (df["fault_id"] == f)]
            el = g["eligible_steps"].sum()
            vals.append(g["achieved_steps"].sum() / el if el else np.nan)
        ax.bar(np.arange(len(order)) + (i - 1.5) * w, vals, width=w,
               label=PRETTY.get(p, p), color=C_RG if p.startswith("C4") else None)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{f}\n{fam[f]}" for f in order], fontsize=8)
    ax.set_ylabel("DR achievement rate")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Achievement by fault condition\nsource: {_rel(out)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_achievement_by_fault.png")
    plt.close(fig)


def fig_authority(df: pd.DataFrame, out: Path) -> None:
    focus = [p for p in ["C1", "C3", "C4_s0", "C5_s0"] if p in set(df["policy_id"])]
    order = ["F0", "F1", "F2", "F3", "F8", "F9", "F10"]
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    w = 0.2
    for i, p in enumerate(focus):
        vals = [df[(df["policy_id"] == p) & (df["fault_id"] == f)][
            "frac_time_local_authority"].mean() for f in order]
        ax.bar(np.arange(len(order)) + (i - 1.5) * w, vals, width=w,
               label=PRETTY.get(p, p), color=C_RG if p.startswith("C4") else None)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("fraction of control steps with\nedge authority")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Edge authority share by fault\nsource: {_rel(out)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_authority_share.png")
    plt.close(fig)


def fig_tradeoff(df: pd.DataFrame, out: Path) -> None:
    pols = [p for p in POLICY_ORDER if p in set(df["policy_id"])]
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)
    for p in pols:
        g = df[df["policy_id"] == p]
        ach = g["achieved_steps"].sum() / max(g["eligible_steps"].sum(), 1)
        curt = g["curtail_energy_kwh"].sum() / max(g["baseline_energy_kwh"].sum(), 1) * 100
        ax.scatter(curt, ach, s=70, color=C_RG if p.startswith("C4") else C_OTHER,
                   zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(p, (curt, ach), textcoords="offset points", xytext=(7, 4), fontsize=8.5)
    ax.set_xlabel("curtailment energy as % of baseline energy")
    ax.set_ylabel("DR achievement rate")
    ax.grid(alpha=0.3)
    ax.set_title(f"Achievement vs curtailment burden\nsource: {_rel(out)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tradeoff.png")
    plt.close(fig)


def fig_sensitivity(df: pd.DataFrame, out: Path) -> None:
    if "factor" not in df or df["factor"].dropna().empty:
        return
    d = df[df["factor"].notna() & (df["factor"] != "")]
    factors = [f for f in d["factor"].unique() if not str(f).startswith("drop_")]
    if not factors:
        return
    n = len(factors)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), dpi=150,
                             squeeze=False)
    for k, f in enumerate(factors):
        ax = axes[k // cols][k % cols]
        sub = d[d["factor"] == f]
        levels = list(dict.fromkeys(sub["factor_level"]))
        ys, xs = [], []
        for lv in levels:
            g = sub[sub["factor_level"] == lv]
            el = g["eligible_steps"].sum()
            ys.append(g["achieved_steps"].sum() / el if el else np.nan)
            xs.append(lv)
        ax.plot(range(len(xs)), ys, "o-", color=C_RG)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=8.5)
        ax.set_title(str(f), fontsize=9.5)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        if k % cols == 0:
            ax.set_ylabel("achievement rate")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"One-factor-at-a-time sensitivity (probe = C4_s0)\n"
                 f"source: {_rel(out)}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig_sensitivity.png")
    plt.close(fig)


def fig_ablation(abl: pd.DataFrame, main_df: pd.DataFrame, out: Path) -> None:
    """Guarded READ-Grid vs its unguarded variants, paired on identical cells.

    The ablation cells are a strict subset of the main-grid cells, so the guarded
    reference (C4_s0) is read from ``stage_main.jsonl`` and restricted to the
    (episode, fault, comm_seed) cells the ablation actually ran. Both source
    files are named in the caption.
    """
    keys = ["episode_id", "fault_id", "comm_seed"]
    cells = abl[keys].drop_duplicates()
    ref = main_df.merge(cells, on=keys, how="inner")
    ref = ref[ref["policy_id"] == "C4_s0"]
    pool = pd.concat([abl, ref], ignore_index=True)

    arms = [("C4_s0", "C4 guarded\n(reference)"),
            ("A1_C4_unguarded", "A1 no authority\nfence"),
            ("A2_C4_cache_unchecked", "A2 cache age\nunchecked"),
            ("A3_C4_unguarded_cache_unchecked", "A3 both guards\nremoved")]
    arms = [(p, lab) for p, lab in arms if p in set(pool["policy_id"])]
    if len(arms) < 2:
        return
    rng = np.random.default_rng(11)
    means, los, his, labs = [], [], [], []
    for p, lab in arms:
        pe = per_episode(pool, p, 1)
        v = pe["achievement_rate"].dropna().to_numpy()
        if not len(v):
            continue
        means.append(float(v.mean()))
        b = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(axis=1)
        los.append(float(np.quantile(b, 0.025)))
        his.append(float(np.quantile(b, 0.975)))
        labs.append(lab)
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    colors = [C_RG] + [C_OTHER] * (len(labs) - 1)
    ax.bar(x, means, color=colors, width=0.6)
    ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                fmt="none", ecolor="#222", capsize=4, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("DR achievement rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Ablation of the authority and cache guards\n"
                 f"source: {_rel(out)} + runs/stage_main.jsonl (C4_s0)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)


def fig_budget_matched(c4t: pd.DataFrame, main_df: pd.DataFrame, out: Path) -> None:
    """Frozen C4, the budget-matched C4T, and the rule it is compared against.

    C4T lives in its own stage file, so the frozen arms are pulled from
    ``stage_main.jsonl`` on the identical ``(episode, fault, comm_seed)`` cells.
    The caption names both source files.
    """
    keys = ["episode_id", "fault_id", "comm_seed"]
    cells = c4t[keys].drop_duplicates()
    ref = main_df.merge(cells, on=keys, how="inner")
    pool = pd.concat([c4t, ref], ignore_index=True)

    arms = [("C4T", "C4T budget-matched\n(18 configs searched)"),
            ("C4_s0", "C4 frozen\n(s0)"),
            ("C4_s1", "C4 frozen\n(s1)"),
            ("C3", "C3 tuned rule"),
            ("C2_s0", "C2 cloud learned only")]
    arms = [(p, lab) for p, lab in arms if p in set(pool["policy_id"])]
    if len(arms) < 2:
        return
    rng = np.random.default_rng(23)
    means, los, his, labs, cols = [], [], [], [], []
    for p, lab in arms:
        pe = per_episode(pool, p, 1)
        v = pe["achievement_rate"].dropna().to_numpy()
        if not len(v):
            continue
        means.append(float(v.mean()))
        b = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(axis=1)
        los.append(float(np.quantile(b, 0.025)))
        his.append(float(np.quantile(b, 0.975)))
        labs.append(lab)
        cols.append(C_RG if p in ("C4T", "C4_s0", "C4_s1") else C_OTHER)
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.bar(x, means, color=cols, width=0.6)
    ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                fmt="none", ecolor="#222", capsize=4, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel("DR achievement rate")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Does equalising the tuning budget change the outcome?\n"
                 f"source: {_rel(out)} + runs/stage_main.jsonl",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_budget_matched.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    src: dict[str, str] = {}
    main_log = RUNS / "stage_main.jsonl"
    if main_log.exists():
        df = _read(main_log)
        for name, fn in [
            ("fig_achievement_by_policy.png", fig_achievement),
            ("fig_achievement_by_fault.png", fig_by_fault),
            ("fig_authority_share.png", fig_authority),
            ("fig_tradeoff.png", fig_tradeoff),
        ]:
            fn(df, main_log)
            made.append(name)
            src[name] = str(main_log.relative_to(ROOT))
    sens = RUNS / "stage_sensitivity.jsonl"
    if sens.exists():
        fig_sensitivity(_read(sens), sens)
        made.append("fig_sensitivity.png")
        src["fig_sensitivity.png"] = str(sens.relative_to(ROOT))
    abl = RUNS / "stage_ablation.jsonl"
    if abl.exists() and main_log.exists():
        fig_ablation(_read(abl), _read(main_log), abl)
        made.append("fig_ablation.png")
        src["fig_ablation.png"] = (f"{abl.relative_to(ROOT)} + "
                                   f"{main_log.relative_to(ROOT)} (C4_s0)")
    c4t = RUNS / "stage_c4t.jsonl"
    if c4t.exists() and main_log.exists():
        fig_budget_matched(_read(c4t), _read(main_log), c4t)
        made.append("fig_budget_matched.png")
        src["fig_budget_matched.png"] = (f"{c4t.relative_to(ROOT)} + "
                                         f"{main_log.relative_to(ROOT)}")
    (FIG / "figure_sources.json").write_text(json.dumps(
        {"figures": made,
         "sources": src,
         "script": "analysis/figures.py"}, indent=2), encoding="utf-8")
    print("[figures] wrote:", ", ".join(made) if made else "(none)")


if __name__ == "__main__":
    main()
