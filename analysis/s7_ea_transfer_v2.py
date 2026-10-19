"""S7 / E-A: transfer robustness under operating-condition shift.

The nominal operating point is ``target.delta_dr = 0.12`` (the external cap is
``(1 - delta_dr) * nom``) and ``constraints.burden_cap_fraction = 0.03`` (the
daily curtailment budget). E-A deploys policies that were frozen at the nominal
point into six shifted conditions and asks whether the ranking survives:

    delta_dr  in {0.08, 0.16}          (cap factor 0.92 / 0.84)
    budget    in {0.02, 0.04, 0.06}    (2 % / 4 % / 6 %)

Comparators (the plan's list):
    * rule NOT re-tuned      -- the frozen C3b parameters (0.00, 0.30, 0.00)
    * rule re-tuned per condition -- the 144-point grid re-run inside each
      condition, selected on VAL with the frozen selection objective
    * learner NOT re-trained -- the frozen C4v2 / C4 checkpoints
    * C7 (parameter-free) and C0 (uncontrolled)
    * C8, the DP bracket, restated per condition (ceil = feasible, floor =
      optimistic; never called an upper bound)

Documented scope decisions (stated in the S7 report, not hidden):

1. **The full-simulator deployment grid uses the five DETERMINISTIC fault
   conditions only** (F0, F1, F2, F3, F8). E-A's question is whether the ranking
   survives a change in the operating point, which the deterministic faults
   answer directly; the stochastic-fault replication is already carried at the
   nominal point in S6. This keeps E-A to ~4.4k simulations instead of ~60k.
2. **The per-condition rule re-tuning selects on VAL.** VAL was already consumed
   by S3/S4 (see docs/test_exposure_record.md), so this is post-hoc and is
   labelled as such wherever it is reported. The frozen arm is never re-selected.
3. **The learner is never retrained or re-tuned for any condition.** That is the
   point of the experiment.

Commands:
    python analysis/s7_ea_transfer_v2.py sandbox      # rule re-tune + DP per condition
    python analysis/s7_ea_transfer_v2.py fullsim      # deployment runs (OpenDSS)
    python analysis/s7_ea_transfer_v2.py report       # comparison table
    python analysis/s7_ea_transfer_v2.py intervals    # paired CIs (post-hoc)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_v2"
SANDBOX = ROOT / "sandbox" / "p359_sandbox"

NOMINAL = {"delta_dr": 0.12, "budget": 0.03}
CONDITIONS = [
    ("nominal", 0.12, 0.03),
    ("delta_0.08", 0.08, 0.03),
    ("delta_0.16", 0.16, 0.03),
    ("budget_0.02", 0.12, 0.02),
    ("budget_0.04", 0.12, 0.04),
    ("budget_0.06", 0.12, 0.06),
]
DET_FAULTS = [("F0", 0), ("F1", 0), ("F2", 0), ("F3", 0), ("F8", 0)]
EA_ARMS = ["C0", "C3b", "C3b_retuned", "C7", "C4v2_s0", "C4_s0"]

# --------------------------------------------------------------- full simulator
def run_fullsim(workers: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("P359_PROTOCOL", str(ROOT / "configs" / "protocol_v2.yaml"))
    import run_v2 as v2
    import run_experiments as rx

    sand = json.loads((RUNS / "s7_ea_sandbox.json").read_text(encoding="utf-8"))
    split = rx.load_split()
    base_pols = v2.v2_policy_set(v2.load_learner_cfg(), v2.rule_params_from_winner(), 5)
    by_id = {p["policy_id"]: p for p in base_pols}

    jobs = []
    for name, delta, budget in CONDITIONS:
        cond = {"target.delta_dr": float(delta),
                "constraints.burden_cap_fraction": float(budget)}
        arms = {
            "C0": by_id["C0"],
            "C3b": by_id["C3b"],
            "C7": by_id["C7"],
            "C4v2_s0": by_id["C4v2_s0"],
            "C4_s0": by_id["C4_s0"],
            "C3b_retuned": dict(by_id["C3b"],
                                policy_id="C3b_retuned",
                                policy_label=f"C3b rule re-tuned inside {name}"),
        }
        r = sand[name]["retuned_rule"]
        rp = [r["th_low"], r["th_high"], r["hyst"], 1.20, 0.0]
        arms["C3b_retuned"]["cloud"] = {"type": "rule", "params": rp}
        arms["C3b_retuned"]["local"] = {"type": "rule", "params": rp}
        for pid, p in arms.items():
            q = dict(p)
            q["policy_id"] = f"{pid}@{name}"
            ov = dict(p.get("protocol_override") or {})
            ov.update(cond)
            q["protocol_override"] = ov
            jobs += rx.expand([q], split["test"], DET_FAULTS, scale=1.0,
                              stage=f"ea_{name}")
    print(f"[E-A] {len(jobs)} jobs = {len(CONDITIONS)} conditions x "
          f"{len(EA_ARMS)} arms x {len(split['test'])} test episodes x "
          f"{len(DET_FAULTS)} deterministic faults", flush=True)
    st = rx.dispatch(jobs, RUNS / "stage_ea_transfer.jsonl", workers=workers, chunk=2)
    man = {"stage": "ea_transfer", "n_jobs": len(jobs), "n_errors": st["n_err"],
           "wall_s": st["wall_s"], "conditions": CONDITIONS,
           "deterministic_faults": DET_FAULTS, "arms": EA_ARMS,
           "src_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src").rglob("*.py"))},
           "split_sha256": hashlib.sha256(
               (ROOT / "runs" / "workload_split.json").read_bytes()).hexdigest(),
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (RUNS / "ea_transfer_manifest.json").write_text(json.dumps(man, indent=2),
                                                    encoding="utf-8")
    print("[E-A] wrote runs_v2/ea_transfer_manifest.json", flush=True)


def run_report() -> None:
    import numpy as np
    import pandas as pd
    sand = json.loads((RUNS / "s7_ea_sandbox.json").read_text(encoding="utf-8"))
    df = pd.read_json(RUNS / "stage_ea_transfer.jsonl", lines=True)
    df = df[df["ok"].astype(bool)].copy()
    df["cond"] = df["policy_id"].str.split("@").str[1]
    df["arm"] = df["policy_id"].str.split("@").str[0]

    def score(g):
        el = g["eligible_steps"].sum()
        base = g["baseline_energy_kwh"].sum()
        ar = g["achieved_steps"].sum() / el if el else float("nan")
        cf = g["curtail_energy_kwh"].sum() / base if base else float("nan")
        sp = g["n_switches"].sum() / 50 / max(len(g), 1)
        return ar - cf - 1e-4 * sp

    rows = []
    for (cond, arm), g in df.groupby(["cond", "arm"]):
        rows.append({"condition": cond, "arm": arm, "n_runs": len(g),
                     "fullsim_score": score(g)})
    tab = pd.DataFrame(rows)
    piv = tab.pivot(index="condition", columns="arm", values="fullsim_score")
    order = [c[0] for c in CONDITIONS]
    piv = piv.reindex(order)
    print("E-A  FULL-SIMULATOR SCORE (test split, deterministic faults only)")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nE-A  RANK SHIFT vs the nominal condition (arm ordering)")
    for cond in order:
        if cond not in piv.index:
            continue
        s = piv.loc[cond].dropna().sort_values(ascending=False)
        print(f"  {cond:<14} " + "  ".join(f"{a}={v:.4f}" for a, v in s.items()))

    print("\nE-A  FROZEN vs RE-TUNED RULE, and the DP bracket")
    print(f"  {'condition':<14}{'frozen rule':>12}{'re-tuned':>10}"
          f"{'retuned params':>22}{'DP ceil':>9}{'DP floor':>10}")
    out = []
    for name, delta, budget in CONDITIONS:
        s = sand[name]
        sub = piv.loc[name] if name in piv.index else pd.Series(dtype=float)
        print(f"  {name:<14}{sub.get('C3b', float('nan')):>12.4f}"
              f"{sub.get('C3b_retuned', float('nan')):>10.4f}"
              f"{str((s['retuned_rule']['th_low'], s['retuned_rule']['th_high'], s['retuned_rule']['hyst'])):>22}"
              f"{s['dp']['ceil']:>9.4f}{s['dp']['floor']:>10.4f}")
        out.append({"condition": name, "delta_dr": delta, "budget_fraction": budget,
                    "fullsim": {a: (float(sub.get(a)) if a in sub else None)
                                for a in EA_ARMS},
                    "sandbox_retuned_rule": s["retuned_rule"],
                    "sandbox_frozen_rule": s["frozen_rule"],
                    "sandbox_c7": s["c7"], "dp": s["dp"]})
    (RUNS / "s7_ea_results.json").write_text(json.dumps(out, indent=2, default=float),
                                             encoding="utf-8")
    tab.to_csv(RUNS / "s7_ea_table.csv", index=False)
    print("\nwrote runs_v2/s7_ea_results.json and runs_v2/s7_ea_table.csv")
    print("STAGE S7-EA COMPLETE")


def run_intervals() -> None:
    """Paired, episode-clustered intervals for the transfer contrasts (post-hoc).

    The transfer table is a point-estimate ordering. This stage supplies the
    interval for every shifted condition under the *same frozen bootstrap* the
    held-out analysis uses (5,000 resamples, episode clusters, deterministic
    seed key), so every interval quoted in the manuscript can be regenerated
    from the archived records without re-running a single simulation.

    Metric: the frozen selection score, computed **per episode** as
    ``achievement - curtailment fraction - 1e-4 * switches/group/run``, because
    the transfer table is a score table, not an achievement table. The learner
    arm is the single frozen seed ``C4v2_s0``; this is post-hoc and the interval
    describes that seed, not the training distribution.
    """
    import pandas as pd

    sys.path.insert(0, str(ROOT / "analysis"))
    import analyze as az

    df = pd.read_json(RUNS / "stage_ea_transfer.jsonl", lines=True)
    df = df[df["ok"].astype(bool)].copy()
    df["cond"] = df["policy_id"].str.split("@").str[1]
    df["arm"] = df["policy_id"].str.split("@").str[0]

    def ep_score(g) -> float:
        el = g["eligible_steps"].sum()
        base = g["baseline_energy_kwh"].sum()
        ar = g["achieved_steps"].sum() / el if el else float("nan")
        cf = g["curtail_energy_kwh"].sum() / base if base else float("nan")
        sp = g["n_switches"].sum() / 50.0 / max(len(g), 1)
        return ar - cf - 1e-4 * sp

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for (cond, arm), g in df.groupby(["cond", "arm"]):
        frames[(cond, arm)] = g.groupby("episode_id").apply(ep_score).rename("score").to_frame()

    rows = []
    print("E-A  PAIRED INTERVALS, C4v2_s0 - C3b_retuned (frozen selection score)")
    print(f"  {'condition':<14}{'diff':>9}{'ci_low':>9}{'ci_high':>9}{'p':>8}{'n':>4}"
          f"{'resolved':>10}")
    for name, _delta, _budget in CONDITIONS:
        a = frames[(name, "C4v2_s0")]
        b = frames[(name, "C3b_retuned")]
        r = az.cluster_bootstrap_diff(a, b, "score", n_resamples=5000,
                                      ci_level=0.95, seed_key=("ea_score", name))
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
        resolved = bool(lo > 0.0 or hi < 0.0)
        rows.append({
            "condition": name,
            "contrast": "C4v2_s0 - C3b_retuned",
            "metric": "frozen selection score (per episode)",
            "mean_diff": float(r["mean_diff"]),
            "ci_low": lo, "ci_high": hi,
            "ci_half_width": (hi - lo) / 2.0,
            "boot_p": float(r["boot_p"]),
            "n_units": int(r["n_units"]),
            "c4v2_mean": float(a["score"].mean()),
            "c3b_retuned_mean": float(b["score"].mean()),
            "resolved": resolved,
        })
        print(f"  {name:<14}{r['mean_diff']:>9.4f}{lo:>9.4f}{hi:>9.4f}"
              f"{r['boot_p']:>8.4f}{int(r['n_units']):>4}{str(resolved):>10}")

    doc = {
        "stage": "ea_intervals",
        "source": "runs_v2/stage_ea_transfer.jsonl",
        "nature": "post-hoc; not part of the pre-registered analysis",
        "learner_seeds": ["C4v2_s0"],
        "bootstrap": {"n_resamples": 5000, "ci_level": 0.95,
                      "unit": "episode cluster", "seed_key": "('ea_score', condition)"},
        "conditions": rows,
    }
    (RUNS / "s7_ea_intervals.json").write_text(
        json.dumps(doc, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RUNS / "s7_ea_intervals.csv", index=False)
    print("\nwrote runs_v2/s7_ea_intervals.json and runs_v2/s7_ea_intervals.csv")
    print("STAGE S7-EA INTERVALS COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sandbox", "fullsim", "report", "intervals"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"sandbox": lambda w: run_sandbox(),
     "fullsim": run_fullsim,
     "report": lambda w: run_report(),
     "intervals": lambda w: run_intervals()}[a.stage](a.workers)
