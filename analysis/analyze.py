"""Phase 10: metrics aggregation and paired statistical analysis.

Input : a JSONL stage file produced by ``src/run_experiments.py``
        (one row per (policy, episode, fault, comm_seed) simulation).
Output: ``analysis/tables/*.csv`` and ``analysis/summary.md``.

Design decisions
----------------
* The **independent unit is the workload episode**, not the time step and not
  the individual simulation. Every stochastic fault is replayed with the same
  held-out communication seeds across policies, and every policy sees the same
  episode set, so a comparison is *paired within (episode, fault, comm_seed)*.
* Confidence intervals come from an **episode-cluster bootstrap**: episodes are
  resampled with replacement and every simulation belonging to a resampled
  episode travels with it. This is the resampling scheme the protocol freezes
  (``analysis.bootstrap_unit = episode_cluster``).
* Rate metrics are aggregated as **ratios of sums within an episode**
  (``sum(achieved_steps) / sum(eligible_steps)``), never as a mean of per-run
  rates, so a run with few eligible steps cannot dominate.
* The **primary** metric is ``achievement_rate``. The secondary family is
  corrected with **Holm-Bonferroni** at the frozen ``ci_level``.
* Undefined quantities are reported as ``nan``, never as an invented ratio
  (e.g. a policy with zero eligible steps has no achievement rate).

Run:  python analysis/analyze.py runs/stage_main.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))

TABLES = ROOT / "analysis" / "tables"
OUT = ROOT / "analysis"

# ---------------------------------------------------------------- metric spec
# name -> (aggregation within episode, source column(s))
RATIO_METRICS = {
    "achievement_rate": ("achieved_steps", "eligible_steps"),
    "mean_exceedance": ("sum_exceedance", "eligible_steps"),
    # v2 fix F-2: renamed from "retention". This is actual/baseline scheduled
    # energy, NOT outage-service retention.
    "energy_ratio": ("actual_energy_kwh", "baseline_energy_kwh"),
    # Outage-focus window (outage + the 30 min that follow it).
    "outage_achievement_rate": ("outage_achieved_steps", "outage_eligible_steps"),
    "outage_energy_ratio": ("outage_actual_energy_kwh", "outage_baseline_energy_kwh"),
}
SUM_METRICS = [
    "curtail_energy_kwh", "baseline_energy_kwh", "actual_energy_kwh",
    "n_switches", "n_authority_switches", "eligible_steps", "achieved_steps",
    "sum_exceedance", "n_commands_accepted", "n_commands_rejected",
    "n_commands_offered", "n_convergence_failures", "n_steps_solved",
    "voltage_violation_s", "line_violation_s",
    "outage_eligible_steps", "outage_achieved_steps", "outage_switches",
    "outage_baseline_energy_kwh", "outage_actual_energy_kwh",
]
MIN_METRICS = ["vmin_pu"]
MAX_METRICS = ["vmax_pu", "max_line_loading_pu"]
MEAN_METRICS = [
    "heartbeat_detect_delay_s", "authority_transfer_delay_s", "recovery_delay_s",
    "mean_command_latency_s", "burden_used_fraction", "outage_window_s",
]
# fraction-of-time metrics: episode mean of the per-run value
FRAC_METRICS = ["frac_time_local_authority", "frac_time_cloud_authority"]

PRIMARY = "achievement_rate"
SECONDARY = [
    "mean_exceedance", "curtail_energy_kwh", "n_switches", "energy_ratio",
    "vmin_pu", "max_line_loading_pu", "frac_time_local_authority",
    "outage_achievement_rate", "outage_energy_ratio", "outage_switches",
]

# comparisons: (label, policy_a, policy_b, direction_hint)
# direction_hint is documentation only; the sign is always A - B.
COMPARISONS = [
    ("READ-Grid vs cloud-only rule", "C4", "C1"),
    ("READ-Grid vs guarded rule fallback", "C4", "C3"),
    ("READ-Grid vs cloud-only learned", "C4", "C2"),
    ("READ-Grid vs concatenated ablation", "C4", "C5"),
    ("READ-Grid vs uncontrolled", "C4", "C0"),
    ("cloud-only rule vs uncontrolled", "C1", "C0"),
    # Phase 11 ablations. ``A1``/``A2``/``A3`` are the unguarded variants of the
    # seed-matched READ-Grid instance; their reference arm is C4_s0, read from
    # stage_main.jsonl on exactly the same (episode, fault, comm_seed) cells.
    ("ablation: guarded vs heartbeat-only", "A1", "C4"),
    ("ablation: cache-age enforced vs unchecked", "A2", "C4"),
    ("ablation: both guards removed", "A3", "C4"),
]


# ------------------------------------------------------------------- loading
def load_rows(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"{path} contains no rows")
    df = df[df["ok"].astype(bool)].copy()
    df["policy_id"] = df["policy_id"].astype(str)
    return df


def per_episode(df: pd.DataFrame, policy: str, n_reps: int = 1) -> pd.DataFrame:
    """Collapse one policy's simulations to one row per episode.

    ``n_reps`` is the number of replications that were collapsed into this
    policy label (used by ``--group-by-seed``, where C4_s0..s2 are folded into
    C4). Rate metrics stay ratios of sums, but *sum* metrics are divided by
    ``n_reps`` so a seed-averaged energy is not triple-counted.
    """
    sub = df[df["policy_id"] == policy]
    if sub.empty:
        return pd.DataFrame()
    out = {"episode_id": []}
    for col in RATIO_METRICS:
        out[col] = []
    for col in SUM_METRICS + MIN_METRICS + MAX_METRICS + MEAN_METRICS + FRAC_METRICS:
        out[col] = []
    for eid, g in sub.groupby("episode_id"):
        out["episode_id"].append(eid)
        for name, (num, den) in RATIO_METRICS.items():
            d = g[den].sum()
            out[name].append(float(g[num].sum() / d) if d else float("nan"))
        for col in SUM_METRICS:
            out[col].append(float(g[col].sum() / n_reps))
        for col in MIN_METRICS:
            out[col].append(float(g[col].min()))
        for col in MAX_METRICS:
            out[col].append(float(g[col].max()))
        for col in MEAN_METRICS + FRAC_METRICS:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            out[col].append(float(vals.mean()) if len(vals) else float("nan"))
    return pd.DataFrame(out).set_index("episode_id")


def n_replications(df: pd.DataFrame, policy: str) -> int:
    """How many rows share the same (episode, fault, comm_seed) for a policy."""
    sub = df[df["policy_id"] == policy]
    if sub.empty:
        return 1
    keys = ["episode_id", "fault_id", "comm_seed"]
    groups = sub.groupby(keys).size()
    if groups.empty:
        return 1
    return max(1, int(round(float(groups.mean()))))


# ---------------------------------------------------------------- aggregation
def describe(pe: pd.DataFrame) -> dict:
    """Summarise a per-episode frame for one policy."""
    row = {"n_episodes": int(len(pe))}
    for name in RATIO_METRICS:
        v = pe[name].dropna()
        row[name] = float(v.mean()) if len(v) else float("nan")
    for col in SUM_METRICS:
        row[col] = float(pe[col].mean())
    for col in MIN_METRICS:
        row[col] = float(pe[col].min())
    for col in MAX_METRICS:
        row[col] = float(pe[col].max())
    for col in MEAN_METRICS + FRAC_METRICS:
        v = pe[col].dropna()
        row[col] = float(v.mean()) if len(v) else float("nan")
    # ratio of totals across all episodes (a second, denominator-weighted view)
    tot_el = pe["eligible_steps"].sum()
    row["achievement_rate_pooled"] = (float(pe["achieved_steps"].sum() / tot_el)
                                      if tot_el else float("nan"))
    tot_base = pe["baseline_energy_kwh"].sum()
    row["curtail_fraction_pooled"] = (float(pe["curtail_energy_kwh"].sum() / tot_base)
                                      if tot_base else float("nan"))
    return row


def per_fault_table(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive breakdown per ``(policy_id, fault_id)``.

    Rate metrics are ratios of sums over the episodes and communication seeds
    in the cell (not a mean of per-run rates), matching the per-episode
    aggregation; ``n_reps`` records how many simulation rows the cell contains.
    """
    rows = []
    for (pid, fid), g in df.groupby(["policy_id", "fault_id"]):
        row = {"policy_id": pid, "fault_id": fid, "n_runs": int(len(g)),
               "n_episodes": int(g["episode_id"].nunique())}
        for name, (num, den) in RATIO_METRICS.items():
            d = g[den].sum()
            row[name] = float(g[num].sum() / d) if d else float("nan")
        for col in SUM_METRICS:
            row[col] = float(g[col].sum())
        for col in MIN_METRICS:
            row[col] = float(g[col].min())
        for col in MAX_METRICS:
            row[col] = float(g[col].max())
        for col in MEAN_METRICS + FRAC_METRICS:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[col] = float(vals.mean()) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["policy_id", "fault_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ bootstrap
def _seed_for(*parts: object) -> int:
    """Deterministic per-comparison RNG seed.

    A single shared RNG consumed in loop order makes every confidence interval
    depend on how many comparisons ran before it, so reordering or subsetting the
    comparison list silently changes the intervals. Deriving the seed from the
    comparison identity instead makes each interval reproducible on its own.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def cluster_bootstrap_diff(pe_a: pd.DataFrame, pe_b: pd.DataFrame, metric: str,
                           *, n_resamples: int, ci_level: float,
                           rng: np.random.Generator | None = None,
                           seed_key: object | None = None) -> dict:
    """Paired difference A - B, CI from resampling episodes with replacement.

    ``seed_key`` (any hashable identifying the comparison) takes precedence over
    ``rng``: when given, the resampling stream is derived from it so the result
    does not depend on call order.
    """
    if seed_key is not None:
        rng = np.random.default_rng(_seed_for(seed_key))
    elif rng is None:
        rng = np.random.default_rng(0)
    common = pe_a.index.intersection(pe_b.index)
    if len(common) == 0:
        return {"n_units": 0, "mean_diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_defined": 0}
    a = pe_a.loc[common, metric].to_numpy(dtype=float)
    b = pe_b.loc[common, metric].to_numpy(dtype=float)
    d = a - b
    ok = ~np.isnan(d)
    n_def = int(ok.sum())
    if n_def == 0:
        return {"n_units": int(len(common)), "mean_diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_defined": 0}
    d = d[ok]
    mean_diff = float(d.mean())
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    boots = d[idx].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    # two-sided bootstrap p-value for H0: mean difference == 0
    # (proportion of resampled means on the opposite side of zero, doubled)
    frac_le = float((boots <= 0).mean())
    frac_ge = float((boots >= 0).mean())
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    if p == 0.0:
        p = float(1.0 / n_resamples)  # resolution floor, reported as such
    return {"n_units": int(len(common)), "n_defined": n_def,
            "mean_diff": mean_diff, "ci_low": lo, "ci_high": hi,
            "boot_p": p, "sig_at_ci": bool(lo > 0.0 or hi < 0.0)}


def holm_bonferroni(pvals: list[float], alpha: float) -> list[float]:
    """Adjusted p-values under Holm-Bonferroni (step-down, monotone)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def md_table(df: pd.DataFrame) -> str:
    """Minimal Markdown table writer (avoids a ``tabulate`` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------- sensitivity
def sensitivity_table(df: pd.DataFrame, *, n_resamples: int, ci_level: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Per-(factor, level) summary for the one-factor-at-a-time stage.

    The sensitivity stage re-runs the *probe* policy (C4_s0) at perturbed
    protocol values, so the policy-id comparison table is not meaningful here.
    What matters is how the probe's held-out behaviour moves as one frozen value
    is perturbed. Every level is evaluated on the same episodes and faults, so
    the levels are paired within a factor and only the protocol value differs.
    """
    if "factor" not in df.columns:
        return pd.DataFrame()
    d = df[df["factor"].notna() & (df["factor"].astype(str) != "")]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for factor in list(dict.fromkeys(d["factor"])):
        sub_f = d[d["factor"] == factor]
        for level in list(dict.fromkeys(sub_f["factor_level"])):
            g = sub_f[sub_f["factor_level"] == level]
            # collapse to one row per episode (one policy per level, so n_reps=1)
            pol = g["policy_id"].iloc[0]
            pe = per_episode(g, pol, 1)
            if pe.empty:
                continue
            ach = pe["achievement_rate"].dropna().to_numpy(dtype=float)
            if len(ach):
                lrng = np.random.default_rng(_seed_for("sens", factor, level))
                b = ach[lrng.integers(0, len(ach), size=(n_resamples, len(ach)))]
                lo = float(np.quantile(b.mean(axis=1), (1 - ci_level) / 2))
                hi = float(np.quantile(b.mean(axis=1), 1 - (1 - ci_level) / 2))
            else:
                lo = hi = float("nan")
            rows.append({
                "factor": factor,
                "level": level,
                "policy_id": pol,
                "n_rows": int(len(g)),
                "n_episodes": int(pe.shape[0]),
                "achievement_rate": float(np.mean(ach)) if len(ach) else float("nan"),
                "ach_ci_low": lo,
                "ach_ci_high": hi,
                "mean_exceedance": float(pe["mean_exceedance"].mean()),
                "curtail_energy_kwh": float(pe["curtail_energy_kwh"].mean()),
                "n_switches": float(pe["n_switches"].mean()),
                "vmin_pu": float(pe["vmin_pu"].mean()),
                "frac_time_local_authority":
                    float(pe["frac_time_local_authority"].mean()),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main flow
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_jsonl", type=Path)
    ap.add_argument("--protocol", type=Path,
                    default=ROOT / "configs" / "protocol.yaml")
    ap.add_argument("--label", default=None)
    ap.add_argument("--reference", type=Path, action="append", default=None,
                    help="extra JSONL stage file(s) whose rows are pooled with "
                         "the stage file. Used by the ablation stage, whose "
                         "reference arms live in stage_main.jsonl: the ablation "
                         "cells are a subset of the main-grid cells, so the "
                         "contrast stays paired on identical cells.")
    ap.add_argument("--policies", nargs="*", default=None,
                    help="restrict the descriptive table to these policy ids")
    ap.add_argument("--group-by-seed", action="store_true",
                    help="collapse C4_s* / C5_s* / C2_s* to C4 / C5 / C2 by "
                         "averaging seeds before comparison")
    ap.add_argument("--sensitivity", action="store_true",
                    help="force the per-(factor, level) sensitivity table even "
                         "if the stage file has no `factor` column")
    ap.add_argument("--baseline", default=None,
                    help="also compare every policy in the stage file against "
                         "this policy id (paired). Use for stages whose arms are "
                         "dynamically named (e.g. SR_* in the retrain stage); "
                         "combine with --reference to pull the baseline rows in "
                         "from another stage file.")
    ap.add_argument("--per-fault", action="store_true",
                    help="also write a per-(policy, fault) descriptive table. "
                         "Rate metrics are ratios of sums, consistent with the "
                         "per-episode aggregation, so a fault with few eligible "
                         "steps cannot dominate.")
    args = ap.parse_args(argv)

    import yaml
    proto = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    an = proto["analysis"]
    n_res = int(an["bootstrap_resamples"])
    ci = float(an["ci_level"])
    rng = np.random.default_rng(20261014)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.stage_jsonl)
    if args.reference:
        refs = [load_rows(p) for p in args.reference]
        ref = pd.concat(refs, ignore_index=True)
        # keep only the reference cells that the stage file actually used, so the
        # pooled frame is exactly paired (same episodes, faults and comm seeds)
        keys = ["episode_id", "fault_id", "comm_seed"]
        cells = df[keys].drop_duplicates()
        ref = ref.merge(cells, on=keys, how="inner")
        df = pd.concat([df, ref], ignore_index=True)
        print(f"[analyze] pooled {len(ref)} reference rows from "
              f"{', '.join(p.name for p in args.reference)}")
    label = args.label or args.stage_jsonl.stem
    print(f"[analyze] {args.stage_jsonl.name}: {len(df)} rows, "
          f"{df['policy_id'].nunique()} policies, "
          f"{df['episode_id'].nunique()} episodes, "
          f"{df['fault_id'].nunique()} faults")

    if args.group_by_seed:
        df = df.copy()
        df["policy_id"] = df["policy_id"].str.replace(r"_s\d+$", "", regex=True)

    policies = args.policies or sorted(df["policy_id"].unique())
    nrep = {p: n_replications(df, p) for p in policies}
    pe = {p: per_episode(df, p, nrep[p]) for p in policies}
    pe = {p: v for p, v in pe.items() if not v.empty}

    # ---- descriptive table --------------------------------------------------
    desc = pd.DataFrame({p: describe(v) for p, v in pe.items()}).T
    desc.insert(0, "n_replications", [nrep[p] for p in desc.index])
    desc.index.name = "policy_id"
    desc.to_csv(TABLES / f"{label}_policy_summary.csv")
    print(f"[analyze] wrote tables/{label}_policy_summary.csv")

    # ---- sensitivity table (when the stage file is a sensitivity grid) ------
    sens_df = pd.DataFrame()
    if args.sensitivity or "factor" in df.columns:
        sens_df = sensitivity_table(df, n_resamples=n_res, ci_level=ci, rng=rng)
        if not sens_df.empty:
            sens_df.to_csv(TABLES / f"{label}_sensitivity.csv", index=False)
            print(f"[analyze] wrote tables/{label}_sensitivity.csv "
                  f"({len(sens_df)} factor/level rows)")

    # ---- per-(policy, fault) breakdown --------------------------------------
    if args.per_fault:
        pf = per_fault_table(df)
        pf.to_csv(TABLES / f"{label}_per_fault.csv", index=False)
        print(f"[analyze] wrote tables/{label}_per_fault.csv ({len(pf)} rows)")

    # ---- pairwise comparisons ----------------------------------------------
    # resolve comparison endpoints to concrete policy ids present in the data
    def resolve(tag: str) -> list[str]:
        return [p for p in policies if p == tag or p.startswith(tag)]

    rows = []
    comparisons = list(COMPARISONS)
    base_arms: list[str] = []
    if args.baseline:
        def _is_baseline(p: str) -> bool:
            return p == args.baseline or p.startswith(args.baseline)
        base_arms = [p for p in policies if _is_baseline(p)]
        for p in policies:
            if not _is_baseline(p):
                comparisons.append((f"{p} vs {args.baseline}", p, args.baseline))
    for cname, a_tag, b_tag in comparisons:
        a_ids, b_ids = resolve(a_tag), resolve(b_tag)
        if not a_ids or not b_ids:
            continue
        for a in a_ids:
            for b in b_ids:
                for metric in [PRIMARY] + SECONDARY:
                    if metric not in pe[a].columns or metric not in pe[b].columns:
                        continue
                    r = cluster_bootstrap_diff(
                        pe[a], pe[b], metric, n_resamples=n_res, ci_level=ci,
                        seed_key=(a, b, metric))
                    rows.append({
                        "comparison": cname, "policy_a": a, "policy_b": b,
                        "metric": metric, **r,
                        "mean_a": float(np.nanmean(pe[a][metric].to_numpy(dtype=float)))
                        if metric in pe[a] else float("nan"),
                        "mean_b": float(np.nanmean(pe[b][metric].to_numpy(dtype=float)))
                        if metric in pe[b] else float("nan"),
                    })
    cmp_df = pd.DataFrame(rows)
    if cmp_df.empty:
        cmp_df = pd.DataFrame(columns=[
            "comparison", "policy_a", "policy_b", "metric", "n_units",
            "n_defined", "mean_diff", "ci_low", "ci_high", "boot_p",
            "sig_at_ci", "mean_a", "mean_b"])
        print("[analyze] WARNING: no comparison resolved in this stage file")
    cmp_df.to_csv(TABLES / f"{label}_paired_comparisons.csv", index=False)
    print(f"[analyze] wrote tables/{label}_paired_comparisons.csv")

    # ---- multiplicity: Holm-Bonferroni on the secondary family --------------
    # Family = the secondary metrics for the *seed-collapsed* headline pair
    # (READ-Grid vs cloud-only rule) plus the representation ablation.
    fam_names = [
        "READ-Grid vs cloud-only rule",
        "READ-Grid vs guarded rule fallback",
        "READ-Grid vs concatenated ablation",
    ]
    if args.baseline:
        # each dynamically named arm is its own perturbation hypothesis, so the
        # family is the set of arm-vs-baseline contrasts
        fam_names += [f"{p} vs {args.baseline}" for p in policies
                      if p not in base_arms]
    fam_mask = (cmp_df["comparison"].isin(fam_names)) & \
        (cmp_df["metric"].isin(SECONDARY))
    fam = cmp_df[fam_mask].copy()
    fam_note = None
    if not fam.empty:
        pv = fam["boot_p"].fillna(1.0).tolist()
        fam["p_holm"] = holm_bonferroni(pv, 1.0 - ci)
        fam["sig_holm"] = fam["p_holm"] < (1.0 - ci)
        fam.to_csv(TABLES / f"{label}_holm.csv", index=False)
        print(f"[analyze] wrote tables/{label}_holm.csv "
              f"({len(fam)} secondary tests corrected)")
        # The family is inflated when seed-specific instances are compared
        # pairwise (C4_s0 x C5_s0, C4_s0 x C5_s1, ...). Those are replications of
        # one contrast, not independent hypotheses, so the multiplicity-corrected
        # headline family is the one produced with --group-by-seed. Say so rather
        # than let a reader read the inflated count as the study's test family.
        seed_a = fam["policy_a"].str.contains(r"_s\d+$", regex=True)
        seed_b = fam["policy_b"].str.contains(r"_s\d+$", regex=True)
        n_seed_pairs = int((seed_a | seed_b).sum())
        if n_seed_pairs:
            fam_note = (f"{n_seed_pairs} of the {len(fam)} corrected rows compare "
                        f"seed-specific instances (C4_s0 vs C5_s1, ...). Those are "
                        f"replications of one contrast, not independent hypotheses, "
                        f"so the multiplicity-corrected **headline** family is the "
                        f"one produced with `--group-by-seed`.")
            print(f"[analyze] NOTE: {n_seed_pairs} of {len(fam)} family rows involve "
                  f"seed-specific instances; run with --group-by-seed for the "
                  f"multiplicity-corrected headline family")

    # ---- markdown summary ---------------------------------------------------
    lines = [f"# Analysis — `{args.stage_jsonl.name}`", ""]
    n_ok = int(df["ok"].sum()) if "ok" in df else len(df)
    lines.append(f"* rows kept: {len(df)} (ok={n_ok})")
    lines.append(f"* policies: {df['policy_id'].nunique()}; "
                 f"independent episodes: {df['episode_id'].nunique()}; "
                 f"fault conditions per policy/episode: "
                 f"{len(df) / max(df['policy_id'].nunique() * df['episode_id'].nunique(), 1):.1f}")
    lines.append(f"* bootstrap: {n_res} resamples, episode-cluster, CI {ci:.0%}")
    if fam_note:
        lines.append(f"* multiplicity note: {fam_note}")
    lines += ["", "## Per-policy summary", "",
              md_table(desc.reset_index())]
    if not sens_df.empty:
        lines += ["", "## Sensitivity (one factor at a time)", "",
                  "Each level is evaluated on the same episodes and faults as the "
                  "frozen value, so levels are paired within a factor; only the "
                  "protocol value differs. CI is an episode-cluster bootstrap on "
                  "the achievement rate.", "",
                  md_table(sens_df)]
    lines += ["", "## Paired differences (A − B)", ""]
    show = cmp_df[cmp_df["metric"].isin([PRIMARY] + SECONDARY)].copy()
    show = show[["comparison", "policy_a", "policy_b", "metric", "n_units",
                 "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
                 "sig_at_ci"]]
    lines.append(md_table(show))
    (OUT / f"summary_{label}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[analyze] wrote summary_{label}.md")


if __name__ == "__main__":
    main()
