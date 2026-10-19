"""S7 / E-C stress: analysis of the "outage swallows the whole evening DR window"
scenario (fault F3E, 17:00-24:00), run in the corrected full simulator.

This is the analysis half of ``s7_ec_stress_v2.py`` (which only simulates).
It answers the plan's question directly: when the outage covers the entire
evening DR block, do the arm rankings and the outage-window metrics move?

Input : ``runs_v2/stage_ec_stress.jsonl``  (8 arms x 21 test episodes x F3E)
Output: ``runs_v2/s7_ec_stress_results.json`` + a printed table.

Everything is computed on the SAME 21 test episodes as S6, so the contrast
C3b (frozen rule++) vs the others is episode-paired. Because the stress run uses
a single deterministic fault (F3E, comm seed 0) and one seed per learned arm,
the CIs here are NARROWER than S6's and are NOT comparable to them; they are
reported as a within-scenario ranking, not as a new primary result.

Usage:
    python analysis/s7_ec_stress_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import analyze as az                                              # noqa: E402
from readgrid.protocol import load_protocol                        # noqa: E402

RUNS = ROOT / "runs_v2"
STAGE = RUNS / "stage_ec_stress.jsonl"
PROTO = load_protocol(ROOT / "configs" / "protocol_v2.yaml")

# The learned arms are present as single seeds (s0) in this scenario.
ARMS = ["C0", "C1", "C3", "C3b", "C7", "C4v2_s0", "C2v2_s0", "C4_s0"]
LABEL = {"C4v2_s0": "C4v2", "C2v2_s0": "C2v2", "C4_s0": "C4"}
# The stress run reuses the S6 "main" stage semantics, so the frozen objective
# applies unchanged. Rule arms carry no seed.
CONTRASTS = [("C4v2_s0 vs C3b", "C4v2_s0", "C3b"),
             ("C4_s0 vs C3b", "C4_s0", "C3b"),
             ("C2v2_s0 vs C3b", "C2v2_s0", "C3b"),
             ("C7 vs C3b", "C7", "C3b"),
             ("C3 vs C3b", "C3", "C3b"),
             ("C1 vs C3b", "C1", "C3b"),
             ("C0 vs C3b", "C0", "C3b")]


def headline(df: pd.DataFrame, pid: str) -> dict:
    sub = df[df["policy_id"] == pid]
    if sub.empty:
        return {}
    cfg = PROTO.raw["analysis"]["selection_score"]
    n = len(sub)
    elig = float(sub["eligible_steps"].sum())
    ach = float(sub["achieved_steps"].sum())
    base = float(sub["baseline_energy_kwh"].sum())
    curt = float(sub["curtail_energy_kwh"].sum())
    sw = float(sub["n_switches"].sum())
    ar = ach / elig if elig else float("nan")
    cf = curt / base if base else float("nan")
    sp = sw / PROTO.n_groups / max(n, 1)
    oel = float(sub["outage_eligible_steps"].sum())
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
        "outage_energy_ratio": float(sub["outage_actual_energy_kwh"].sum()
                                     / sub["outage_baseline_energy_kwh"].sum()),
        "outage_switches": float(sub["outage_switches"].sum()),
        "outage_window_s_mean": float(pd.to_numeric(
            sub["outage_window_s"], errors="coerce").dropna().mean()),
        "energy_ratio": float(sub["actual_energy_kwh"].sum() / base) if base else float("nan"),
        "vmin_pu": float(sub["vmin_pu"].min()),
        "max_line_loading_pu": float(sub["max_line_loading_pu"].max()),
        "voltage_violation_s": float(sub["voltage_violation_s"].sum()),
        "frac_time_local_authority": float(pd.to_numeric(
            sub["frac_time_local_authority"], errors="coerce").dropna().mean()),
        "n_authority_switches": float(sub["n_authority_switches"].sum()),
    }


def main() -> int:
    df = az.load_rows(STAGE)
    print(f"stress rows: {len(df)} ok   episodes {df['episode_id'].nunique()}   "
          f"faults {sorted(df['fault_id'].unique())}")
    present = [a for a in ARMS if (df["policy_id"] == a).any()]
    missing = sorted(set(ARMS) - set(present))
    if missing:
        print(f"!! arms absent: {missing}")

    head = {a: headline(df, a) for a in present}
    print(f"\n{'arm':<10}{'score':>9}{'ach':>9}{'curt':>9}{'sw/g/r':>9}"
          f"{'outAch':>9}{'outEner':>9}{'outSw':>8}{'vmin':>8}")
    for a in sorted(head, key=lambda x: -(head[x]["score"] or -9)):
        h = head[a]
        print(f"{LABEL.get(a, a):<10}{h['score']:>9.4f}{h['achievement_rate']:>9.4f}"
              f"{h['curtail_fraction']:>9.4f}{h['switches_per_group_run']:>9.4f}"
              f"{h['outage_achievement_rate']:>9.4f}{h['outage_energy_ratio']:>9.4f}"
              f"{h['outage_switches']:>8.0f}{h['vmin_pu']:>8.4f}")

    frames = {a: az.per_episode(df, a) for a in present}
    contrasts = []
    print(f"\nCONTRASTS vs C3b (episode-paired bootstrap, achievement_rate)")
    print(f"  {'contrast':<20}{'diff':>9}{'CI low':>9}{'CI high':>9}{'p':>8}")
    for label, a, b in CONTRASTS:
        if a not in frames or b not in frames:
            continue
        r = az.cluster_bootstrap_diff(
            frames[a], frames[b], "achievement_rate",
            n_resamples=int(PROTO.raw["analysis"]["bootstrap_resamples"]),
            ci_level=float(PROTO.raw["analysis"]["ci_level"]),
            seed_key=("ec_stress", a, b))
        r["label"] = label
        contrasts.append(r)
        print(f"  {label:<20}{r['mean_diff']:>+9.4f}{r['ci_low']:>9.4f}"
              f"{r['ci_high']:>9.4f}{r['boot_p']:>8.4f}")

    # Does the outage-window picture differ from the whole-episode picture?
    print("\nRANK SHIFT under the stress scenario (frozen objective, whole episode)")
    by_score = sorted(present, key=lambda a: -head[a]["score"])
    by_out = sorted(present, key=lambda a: -head[a]["outage_achievement_rate"])
    print("  by score        :", " > ".join(LABEL.get(a, a) for a in by_score))
    print("  by outage ach   :", " > ".join(LABEL.get(a, a) for a in by_out))

    man = json.loads((RUNS / "ec_stress_manifest.json").read_text(encoding="utf-8"))
    out = {
        "stage_file": str(STAGE),
        "n_rows": int(len(df)),
        "n_episodes": int(df["episode_id"].nunique()),
        "fault": man.get("fault"), "fault_spec": man.get("fault_spec"),
        "protocol_ec_sha256": man.get("protocol_ec_sha256"),
        "protocol_ec_base_sha256": man.get("protocol_ec_base_sha256"),
        "headline": head,
        "contrasts_vs_C3b": contrasts,
        "rank_by_score": [LABEL.get(a, a) for a in by_score],
        "rank_by_outage_achievement": [LABEL.get(a, a) for a in by_out],
        "caveats": [
            "single deterministic fault F3E (comm seed 0) and one seed per learned arm, "
            "so the CIs are narrower than S6's and are not comparable to them",
            "F3E is an EXTENSION fault not present in the frozen protocol_v2.yaml; the "
            "frozen protocol and its S0 hash are unmodified",
            "reported as a within-scenario ranking, not as a new primary result",
        ],
    }
    (RUNS / "s7_ec_stress_results.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote runs_v2/s7_ec_stress_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
