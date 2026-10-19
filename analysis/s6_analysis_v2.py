"""S6: the single held-out evaluation analysis for the v2 policy set.

Input : ``runs_v2/stage_main_v2.jsonl`` (one row per policy x test episode x
        fault condition).
Output: ``runs_v2/s6_results.json``, ``runs_v2/s6_table.md`` and a printed
        summary.

Protocol compliance (all values are read from the frozen protocol, none are
chosen here):

* independent unit  = workload episode, resampled with replacement
                      (``analysis.bootstrap_unit = episode_cluster``);
* resamples         = ``analysis.bootstrap_resamples`` (5000);
* level             = ``analysis.ci_level`` (0.95);
* multiplicity      = ``analysis.multiplicity`` (Holm-Bonferroni);
* primary contrast  = the frozen learned arm vs rule++ (C3b), metric
                      ``achievement_rate``;
* non-inferiority margin delta = 0.02, FROZEN. It is never raised. If the
  measured half-width exceeds delta the conclusion is reported as
  "inconclusive (insufficient power)", per plan S2.4.

Six-condition classification (plan S5.2, corrected order)
--------------------------------------------------------
The five conditions are evaluated INDEPENDENTLY and then reported by priority
(superior > equivalent > non-inferior > inferior beyond margin > inferior >
inconclusive). "Equivalent" is strictly stronger than "non-inferior" and is
tested separately -- it is never inferred from non-inferiority.

Documented reading (D-rule, stated in the stage report): conditions that are
*defined by* delta (non-inferior, equivalent) are only reportable when the CI
half-width is at most delta. Conditions that do not depend on delta (superior,
inferior) and the strongly-negative "inferior beyond margin" are reported as
measured, with the power flag attached.

Usage:
    python analysis/s6_analysis_v2.py [stage_jsonl]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import analyze as az                                              # noqa: E402
from readgrid.protocol import load_protocol                        # noqa: E402

PROTO = ROOT / "configs" / "protocol_v2.yaml"
RUNS = ROOT / "runs_v2"
STAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS / "stage_main_v2.jsonl"

DELTA = 0.02                      # plan S2.4, frozen
PRIMARY = "achievement_rate"

# arm -> seeds.  Empty tuple means a single, seed-free instance.
ARMS = {
    "C0": (),
    "C1": (),
    "C3": (),
    "C3b": (),
    "C7": (),
    "C2": tuple(range(5)),
    "C4": tuple(range(5)),
    "C5": tuple(range(5)),
    "C2v2": tuple(range(5)),
    "C4v2": tuple(range(5)),
}

# (label, a, b) -- sign is always a - b
PRIMARY_CONTRAST = ("C4v2 vs C3b (primary)", "C4v2", "C3b")
SECONDARY_CONTRASTS = [
    ("C4v2 vs C4 (v2 learner vs v1 learner)", "C4v2", "C4"),
    ("C4v2 vs C1 (learner vs cloud-only rule)", "C4v2", "C1"),
    ("C4v2 vs C2v2 (value of the local fallback)", "C4v2", "C2v2"),
    ("C4v2 vs C5 (binned vs concatenated)", "C4v2", "C5"),
    ("C4v2 vs C0 (vs uncontrolled)", "C4v2", "C0"),
    ("C3b vs C3 (rule++ vs v1 rule)", "C3b", "C3"),
    ("C3b vs C1 (rule++ vs cloud-only rule)", "C3b", "C1"),
    ("C7 vs C3b (parameter-free vs tuned rule)", "C7", "C3b"),
    ("C4 vs C3 (v1 learner vs v1 rule)", "C4", "C3"),
    ("C2v2 vs C4v2 (cloud-only v2)", "C2v2", "C4v2"),
]
SECONDARY_METRICS = ["mean_exceedance", "curtail_energy_kwh", "n_switches",
                     "energy_ratio", "vmin_pu", "max_line_loading_pu",
                     "frac_time_local_authority", "outage_achievement_rate",
                     "outage_energy_ratio", "outage_switches"]


# --------------------------------------------------------------------- helpers
def classify(low: float, high: float, delta: float) -> dict:
    """Six-condition classification, evaluated independently then by priority."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return {"label": "inconclusive", "reason": "CI undefined",
                "conditions": {}, "power_ok": False}
    half = (high - low) / 2.0
    power_ok = half <= delta
    conds = {
        "1_superior (L > 0)": low > 0.0,
        "2_non_inferior (L > -delta)": low > -delta,
        "3_equivalent (L > -delta and U < +delta)": (low > -delta) and (high < delta),
        "4_inferior (U < 0)": high < 0.0,
        "5_inferior_beyond_margin (U < -delta)": high < -delta,
    }
    if conds["1_superior (L > 0)"]:
        label = "superior"
    elif conds["3_equivalent (L > -delta and U < +delta)"]:
        label = "equivalent"
    elif conds["2_non_inferior (L > -delta)"]:
        label = "non-inferior"
    elif conds["5_inferior_beyond_margin (U < -delta)"]:
        label = "inferior beyond margin"
    elif conds["4_inferior (U < 0)"]:
        label = "inferior"
    else:
        label = "inconclusive"
    # D-rule: delta-defined conclusions need the power to resolve delta.
    if (not power_ok) and label in ("non-inferior", "equivalent", "inconclusive"):
        label = "inconclusive (insufficient power for delta)"
    return {"label": label, "ci_half_width": half, "delta": delta,
            "power_ok": bool(power_ok), "conditions": conds,
            "ci_low": low, "ci_high": high}


def arm_frame(df: pd.DataFrame, arm: str, seeds: tuple) -> pd.DataFrame:
    """Per-episode frame for an arm, seed-averaged when the arm has seeds.

    For a seeded arm each seed is collapsed to one row per episode first (ratios
    of sums within the episode, ``analyze.per_episode`` convention), then the
    seed rows are averaged. The result is one value per episode, so the
    episode-cluster bootstrap still resamples whole episodes.
    """
    if not seeds:
        return az.per_episode(df, arm)
    frames = []
    for s in seeds:
        pid = f"{arm}_s{s}"
        if (df["policy_id"] == pid).any():
            frames.append(az.per_episode(df, pid))
    if not frames:
        return pd.DataFrame()
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    stacked = np.stack([f.loc[common].to_numpy(dtype=float) for f in frames])
    return pd.DataFrame(stacked.mean(axis=0), index=common,
                        columns=frames[0].columns)


def compare(pe_a, pe_b, metric, protocol, key) -> dict:
    r = az.cluster_bootstrap_diff(
        pe_a, pe_b, metric,
        n_resamples=int(protocol.raw["analysis"]["bootstrap_resamples"]),
        ci_level=float(protocol.raw["analysis"]["ci_level"]),
        seed_key=key)
    r["metric"] = metric
    return r


def pooled(df: pd.DataFrame, arm: str, seeds: tuple) -> dict:
    """Frozen selection objective + headline rates for one arm, pooled."""
    if not seeds:
        sub = df[df["policy_id"] == arm]
    else:
        sub = df[df["policy_id"].isin([f"{arm}_s{s}" for s in seeds])]
    if sub.empty:
        return {}
    cfg = protocol.raw["analysis"]["selection_score"]
    n = len(sub)
    elig = sub["eligible_steps"].sum()
    ach = sub["achieved_steps"].sum()
    base = sub["baseline_energy_kwh"].sum()
    curt = sub["curtail_energy_kwh"].sum()
    sw = sub["n_switches"].sum()
    ar = float(ach / elig) if elig else float("nan")
    cf = float(curt / base) if base else float("nan")
    sp = float(sw / protocol.n_groups / max(n, 1))
    oel = sub["outage_eligible_steps"].sum()
    return {
        "n_runs": int(n),
        "achievement_rate": ar,
        "curtail_fraction": cf,
        "switches_per_group_run": sp,
        "score": float(cfg["w_achievement"] * ar
                       - cfg["w_curtail_fraction"] * cf
                       - cfg["w_switch_per_group"] * sp),
        "outage_achievement_rate": (float(sub["outage_achieved_steps"].sum() / oel)
                                    if oel else float("nan")),
        "outage_eligible_steps": int(oel),
        "outage_window_s_mean": float(pd.to_numeric(
            sub["outage_window_s"], errors="coerce").dropna().mean())
        if "outage_window_s" in sub.columns else float("nan"),
        "energy_ratio": float(sub["actual_energy_kwh"].sum() / base) if base else float("nan"),
        "curtail_energy_kwh": float(curt),
        "n_switches": float(sw),
        "n_authority_switches": float(sub["n_authority_switches"].sum()),
        "vmin_pu": float(sub["vmin_pu"].min()),
        "max_line_loading_pu": float(sub["max_line_loading_pu"].max()),
        "voltage_violation_s": float(sub["voltage_violation_s"].sum()),
        "line_violation_s": float(sub["line_violation_s"].sum()),
        "mean_exceedance": float(sub["sum_exceedance"].sum() / elig) if elig else float("nan"),
        "burden_used_fraction": float(pd.to_numeric(
            sub["burden_used_fraction"], errors="coerce").dropna().mean()),
        "frac_time_local_authority": float(pd.to_numeric(
            sub["frac_time_local_authority"], errors="coerce").dropna().mean()),
        "policy_inference_ms_mean": float(pd.to_numeric(
            sub["policy_inference_s"], errors="coerce").dropna().mean()) * 1e3,
        "policy_inference_ms_p95": float(np.nanpercentile(
            pd.to_numeric(sub["policy_inference_s"], errors="coerce").dropna(), 95)) * 1e3,
        "mean_command_latency_ms": float(pd.to_numeric(
            sub["mean_command_latency_s"], errors="coerce").dropna().mean()) * 1e3,
        "heartbeat_detect_delay_s": float(pd.to_numeric(
            sub["heartbeat_detect_delay_s"], errors="coerce").dropna().mean()),
        "authority_transfer_delay_s": float(pd.to_numeric(
            sub["authority_transfer_delay_s"], errors="coerce").dropna().mean()),
        "recovery_delay_s": float(pd.to_numeric(
            sub["recovery_delay_s"], errors="coerce").dropna().mean()),
    }


# ------------------------------------------------------- edge inference budget
def edge_cost() -> dict:
    """Multiply-accumulate count per decision, by arm, from the frozen sizes."""
    d_v2, d_v1, na = 49, 8, 3
    return {
        "note": "MACs per single decision; the cloud controller and the local "
                "fallback each run once per decision interval (60 s).",
        "C4v2": {"cloud_macs": d_v2 * na, "local_macs": d_v2 * na,
                 "total_macs": 2 * d_v2 * na,
                 "params": 2 * d_v2 * na, "repr": f"{d_v2}-dim binned"},
        "C2v2": {"cloud_macs": d_v2 * na, "local_macs": 0,
                 "total_macs": d_v2 * na, "params": d_v2 * na,
                 "repr": f"{d_v2}-dim binned"},
        "C4": {"cloud_macs": d_v1 * na, "local_macs": d_v1 * na,
               "total_macs": 2 * d_v1 * na, "params": 2 * d_v1 * na,
               "repr": f"{d_v1}-dim raw"},
        "C2": {"cloud_macs": d_v1 * na, "local_macs": 0,
               "total_macs": d_v1 * na, "params": d_v1 * na,
               "repr": f"{d_v1}-dim raw"},
        "C5": {"cloud_macs": d_v1 * na * na, "local_macs": d_v1 * na * na,
               "total_macs": 2 * d_v1 * na * na, "params": 2 * d_v1 * na * na,
               "repr": f"{d_v1}x{na} concatenated"},
        "C3/C3b": {"cloud_macs": 0, "local_macs": 0, "total_macs": 0,
                   "params": 3, "repr": "3 thresholds + hysteresis"},
        "C7": {"cloud_macs": na, "local_macs": na, "total_macs": 2 * na,
               "params": 0, "repr": "no parameters"},
        "C0/C1": {"cloud_macs": 0, "local_macs": 0, "total_macs": 0,
                  "params": 0, "repr": "no controller"},
    }


# ------------------------------------------------------------------------ main
protocol = load_protocol(PROTO)
df = az.load_rows(STAGE)
print(f"stage rows: {len(df)} ok")
print(f"policies: {sorted(df['policy_id'].unique().tolist())}")
print(f"episodes: {df['episode_id'].nunique()}   faults: {df['fault_id'].nunique()}")

present = {a: s for a, s in ARMS.items()
           if (not s and (df["policy_id"] == a).any())
           or (s and all((df["policy_id"] == f"{a}_s{x}").any() for x in s))}
missing = sorted(set(ARMS) - set(present))
if missing:
    print(f"!! arms absent from the stage file (excluded): {missing}")

frames = {a: arm_frame(df, a, s) for a, s in present.items()}
frames = {a: f for a, f in frames.items() if len(f)}
for a, f in frames.items():
    print(f"  {a:<6} episodes {len(f)}  seeds {len(present[a]) or 1}")

head = {a: pooled(df, a, present[a]) for a in frames}
print(f"\n{'arm':<6}{'score':>9}{'ach':>9}{'curt':>9}{'sw/g/r':>9}"
      f"{'outAch':>9}{'energy':>9}{'infer ms':>10}")
for a in sorted(head, key=lambda x: -(head[x]["score"] or -9)):
    h = head[a]
    print(f"{a:<6}{h['score']:>9.4f}{h['achievement_rate']:>9.4f}"
          f"{h['curtail_fraction']:>9.4f}{h['switches_per_group_run']:>9.4f}"
          f"{h['outage_achievement_rate']:>9.4f}{h['energy_ratio']:>9.4f}"
          f"{h['policy_inference_ms_mean']:>10.3f}")

# ------------------------------------------------------------------ contrasts
def run_contrasts(pairs):
    out = []
    for label, a, b in pairs:
        if a not in frames or b not in frames:
            continue
        r = compare(frames[a], frames[b], PRIMARY, protocol, ("primary", a, b))
        r["label"] = label
        r["a"], r["b"] = a, b
        r["classification"] = classify(r["ci_low"], r["ci_high"], DELTA)
        out.append(r)
    return out


primary = run_contrasts([PRIMARY_CONTRAST])
secondary = run_contrasts(SECONDARY_CONTRASTS)

pvals = [r["boot_p"] for r in secondary if math.isfinite(r.get("boot_p", float("nan")))]
adj = az.holm_bonferroni(pvals, float(protocol.raw["analysis"]["ci_level"]))
it = iter(adj)
for r in secondary:
    r["holm_adjusted_p"] = next(it, float("nan")) if math.isfinite(
        r.get("boot_p", float("nan"))) else float("nan")

print("\nPRIMARY CONTRAST (achievement_rate, episode-clustered bootstrap, "
      f"delta = {DELTA})")
for r in primary:
    c = r["classification"]
    print(f"  {r['label']}")
    print(f"    mean diff {r['mean_diff']:+.4f}  95% CI [{r['ci_low']:+.4f}, "
          f"{r['ci_high']:+.4f}]  half-width {c['ci_half_width']:.4f}  "
          f"power_ok {c['power_ok']}")
    print(f"    CONCLUSION: {c['label'].upper()}")
    for k, v in c["conditions"].items():
        print(f"      {k}: {v}")

print("\nSECONDARY CONTRASTS (achievement_rate), Holm-Bonferroni corrected")
print(f"  {'label':<46}{'diff':>9}{'CI low':>9}{'CI high':>9}{'p':>8}{'holm p':>9}  conclusion")
for r in secondary:
    c = r["classification"]
    print(f"  {r['label']:<46}{r['mean_diff']:>+9.4f}{r['ci_low']:>9.4f}"
          f"{r['ci_high']:>9.4f}{r['boot_p']:>8.4f}{r['holm_adjusted_p']:>9.4f}"
          f"  {c['label']}")

# ---------------------------------------------------- H3 outage-window (E-C)
print("\nH3  ARCHITECTURE VALUE ON THE OUTAGE WINDOW (outage_achievement_rate)")
h3 = []
for label, a, b in [("C3 - C1 (rule: local fallback value)", "C3", "C1"),
                    ("C4v2 - C2v2 (learner: local fallback value)", "C4v2", "C2v2"),
                    ("C3b - C1 (rule++: local fallback value)", "C3b", "C1"),
                    ("C4 - C2 (v1 learner: local fallback value)", "C4", "C2")]:
    if a not in frames or b not in frames:
        continue
    r = compare(frames[a], frames[b], "outage_achievement_rate", protocol,
                ("h3", a, b))
    r["label"] = label
    r["classification"] = classify(r["ci_low"], r["ci_high"], DELTA)
    h3.append(r)
    print(f"  {label:<46}{r['mean_diff']:>+9.4f} [{r['ci_low']:+.4f}, "
          f"{r['ci_high']:+.4f}]  -> {r['classification']['label']}")

# ------------------------------------------------------------- H4 DP bracket
# C8 is a REFERENCE, not a simulated arm: the plan says to restate it, not to
# recompute it. The values below are the ones recorded in the S2 report /
# runs_v2/sandbox_logs/dp_reference.txt and dp_hires.txt (sandbox, 21 val
# episodes). They are reported next to the test-split scores purely so a reader
# can see the remaining head-room; NO test is performed on this gap.
DP_BRACKET_VAL = {
    "source": "runs_v2/sandbox_logs/dp_reference.txt + dp_hires.txt (VAL, 21 episodes)",
    "units": "pooled achievement rate = sum_i achieved_i / sum_i eligible_i",
    "bracket": [
        {"nb": 150, "ceil_feasible": 0.5375, "floor_optimistic": 0.8583},
        {"nb": 300, "ceil_feasible": 0.6419, "floor_optimistic": 0.8389},
        {"nb": 600, "ceil_feasible": 0.6524, "floor_optimistic": 0.7560},
        {"nb": 1200, "ceil_feasible": 0.6603, "floor_optimistic": 0.7103},
        {"nb": 2400, "ceil_feasible": 0.6658, "floor_optimistic": 0.6896},
        {"nb": 4800, "ceil_feasible": 0.6672, "floor_optimistic": 0.6789},
    ],
    "note": ("ceil is the value of a policy that is FEASIBLE in the real budget "
             "(0/21 replay mismatches); floor is the optimistic discretisation. "
             "Neither is a proven upper bound, and neither is a policy that could "
             "be deployed: both see the whole day in advance."),
}
dp = None
for cand in (RUNS / "s7_ea_sandbox.json", RUNS / "dp_hires.json"):
    if cand.exists():
        dp = json.loads(cand.read_text(encoding="utf-8"))
        break
print("\nH4  GAP TO THE DP BRACKET (reported, NOT tested)")
print(f"  {DP_BRACKET_VAL['source']}")
for row in DP_BRACKET_VAL["bracket"]:
    print(f"    nb={row['nb']:>5}  ceil {row['ceil_feasible']:.4f}  "
          f"floor {row['floor_optimistic']:.4f}")
print(f"  {DP_BRACKET_VAL['note']}")
if dp is not None and "nominal" in dp:
    n = dp["nominal"]
    print(f"  cross-check from {cand.name}: nominal-condition nb=600 "
          f"ceil {n['dp']['ceil']:.4f} / floor {n['dp']['floor']:.4f} "
          f"(must equal the nb=600 row above)")

# --------------------------------------------------- secondary metric family
print("\nSECONDARY METRIC FAMILY (C4v2 - C3b), descriptive with Holm-corrected p")
sec = []
if "C4v2" in frames and "C3b" in frames:
    for m in SECONDARY_METRICS:
        if m not in frames["C4v2"].columns or m not in frames["C3b"].columns:
            continue
        r = compare(frames["C4v2"], frames["C3b"], m, protocol, ("sec", m))
        r["metric"] = m
        sec.append(r)
    mp = [r["boot_p"] for r in sec]
    ma = az.holm_bonferroni(mp, float(protocol.raw["analysis"]["ci_level"]))
    for r, a_ in zip(sec, ma):
        r["holm_adjusted_p"] = a_
        print(f"  {r['metric']:<28}{r['mean_diff']:>+12.5f} "
              f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]  p={r['boot_p']:.4f} "
              f"holm={a_:.4f}")
else:
    print("  skipped: the C4v2 and C3b arms are not both present in the stage file")

out = {
    "stage_file": str(STAGE),
    "protocol_sha256": protocol.sha256,
    "n_rows": int(len(df)),
    "n_episodes": int(df["episode_id"].nunique()),
    "n_faults": int(df["fault_id"].nunique()),
    "arms": {a: {"seeds": list(present[a]), "n_episodes": int(len(frames[a]))}
             for a in frames},
    "headline": head,
    "primary_contrast": primary,
    "secondary_contrasts": secondary,
    "h3_outage_window": h3,
    "dp_bracket": DP_BRACKET_VAL,
    "dp_bracket_crosscheck": (dp.get("nominal", {}).get("dp") if dp else None),
    "secondary_metrics_c4v2_vs_c3b": sec,
    "edge_inference_cost": edge_cost(),
    "delta_frozen": DELTA,
    "evaluation_nature": "post_hoc_exploratory (see docs/test_exposure_record.md)",
}
(RUNS / "s6_results.json").write_text(json.dumps(out, indent=2, default=float),
                                     encoding="utf-8")
print("\nwrote runs_v2/s6_results.json")
print("STAGE S6 COMPLETE")
