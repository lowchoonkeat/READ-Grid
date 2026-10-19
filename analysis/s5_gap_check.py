"""S5 Gate G2: full-simulator VAL results vs the sandbox prediction.

The plan (S5) says: put the frozen candidates into the corrected OpenDSS
simulator, run F0 / F2 / F3 on the VAL split, and compare against the sandbox
prediction -- a difference greater than 0.01 must be investigated.

What is actually comparable, stated up front
--------------------------------------------
* The sandbox replica models a **single normalised group in connected mode**.
  It has no faults, no outages and no flapping. Therefore the comparison is made
  on the **F0 (no-fault) rows only**; the F2/F3 rows are reported separately and
  are NOT part of the gate arithmetic.
* The sandbox rule/learned replicas also omit the rule's **voltage override**
  (`v_thresh = 1.20`, i.e. a 0.06 pu deviation). If the full simulator ever
  drives a group's voltage deviation past that threshold, the two sides are
  solving different problems on those steps. The script therefore reports the
  worst per-run voltage deviation observed on the F0 rows so the reader can see
  whether the override could have fired at all.
* The frozen learner was retrained **5 times** (seeds 0..4). The sandbox
  prediction for seed k is `runs_v2/s3_final_sandbox.json` seeds[k]; the full
  simulator prediction for seed k is the `C4v2_s{k}` rows.

Usage:
    python analysis/s5_gap_check.py [tol]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from readgrid.protocol import load_protocol                       # noqa: E402

PROTO = ROOT / "configs" / "protocol_v2.yaml"
RUNS = ROOT / "runs_v2"
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01


def score(df, protocol) -> dict:
    cfg = protocol.raw["analysis"]["selection_score"]
    n = len(df)
    elig = df["eligible_steps"].sum()
    ach = df["achieved_steps"].sum()
    base = df["baseline_energy_kwh"].sum()
    curt = df["curtail_energy_kwh"].sum()
    sw = df["n_switches"].sum()
    ar = float(ach / elig) if elig else float("nan")
    cf = float(curt / base) if base else float("nan")
    sp = float(sw / protocol.n_groups / max(n, 1))
    return {
        "n_runs": int(n),
        "achievement_rate": ar,
        "curtail_fraction": cf,
        "switches_per_group_run": sp,
        "score": float(cfg["w_achievement"] * ar
                       - cfg["w_curtail_fraction"] * cf
                       - cfg["w_switch_per_group"] * sp),
    }


def main() -> int:
    proto = load_protocol(PROTO)
    df = pd.read_json(RUNS / "stage_validate_v2.jsonl", lines=True)
    ok = df[df["ok"].astype(bool)].copy()
    print(f"rows: {len(df)} total, {len(ok)} ok, {len(df) - len(ok)} failed")
    print("fault conditions present:",
          sorted(ok["fault_id"].astype(str).unique().tolist()))

    f0 = ok[ok["fault_id"].astype(str) == "F0"]
    if not len(f0):
        print("!! no F0 rows: the gate cannot be evaluated against the sandbox")
        return 2

    # voltage-deviation visibility (does the replica's missing override matter?)
    if "vmin_pu" in f0.columns:
        vdev = (1.0 - f0["vmin_pu"].astype(float))
        print(f"\nF0 voltage deviation (1 - vmin_pu): max {vdev.max():.4f} pu, "
              f"p95 {np.nanpercentile(vdev, 95):.4f} pu; "
              f"runs past v_thresh=0.06 pu: {int((vdev > 0.06).sum())}/{len(f0)}")

    full = {pid: score(g, proto) for pid, g in f0.groupby("policy_id")}

    sand = json.loads((RUNS / "s3_final_sandbox.json").read_text(encoding="utf-8"))
    refs = sand["references"]

    # D22: the sandbox divided a SINGLE group's switch count by n_groups, so its
    # switch term is 50x too small (the full simulator divides a 50-group total
    # by n_groups). The correction removes 1e-4 * 49 * sw from the sandbox score.
    # Both versions are shown; the gate is evaluated on the CORRECTED one.
    def _corr(d):
        return d["score"] - 1e-4 * 49 * d["switches_per_group_run"]

    pred = {}
    for s in sand["seeds"]:
        pred[f"C4v2_s{s['seed']}"] = {
            "score": s["val"]["score"], "score_corrected": _corr(s["val"]),
            "achievement_rate": s["val"]["achievement_rate"],
            "source": f"sandbox frozen 5-seed retrain, seed {s['seed']}"}
    for key, label in (("C3b", "sandbox rule (0.00,0.30,0.00)"),
                       ("C3", "sandbox rule (0.20,0.60,0.20)"),
                       ("C7", "sandbox C7 (no parameters)")):
        pred[key] = {"score": refs[key]["val"]["score"],
                     "score_corrected": _corr(refs[key]["val"]),
                     "achievement_rate": refs[key]["val"]["achievement_rate"],
                     "source": label}

    print(f"\n{'policy':<12}{'full':>9}{'sand':>9}{'d raw':>9}"
          f"{'sand corr':>11}{'d corr':>9}{'full ach':>10}{'sand ach':>10}{'d ach':>9}  verdict")
    rows, worst, worst_c = [], 0.0, 0.0
    for pid in sorted(pred, key=lambda p: (p.split('_')[0], p)):
        if pid not in full:
            print(f"  {pid:<10}  (missing in full-sim results)")
            continue
        f, p = full[pid], pred[pid]
        ds = f["score"] - p["score"]
        dc = f["score"] - p["score_corrected"]
        da = f["achievement_rate"] - p["achievement_rate"]
        bad = abs(dc) > TOL
        worst = max(worst, abs(ds))
        worst_c = max(worst_c, abs(dc))
        print(f"  {pid:<10}{f['score']:>9.4f}{p['score']:>9.4f}{ds:>+9.4f}"
              f"{p['score_corrected']:>11.4f}{dc:>+9.4f}"
              f"{f['achievement_rate']:>10.4f}{p['achievement_rate']:>10.4f}{da:>+9.4f}"
              f"  {'GAP' if bad else 'ok'}")
        rows.append({"policy_id": pid, "full": f, "sandbox": p,
                     "d_score_raw": ds, "d_score_corrected": dc,
                     "d_achievement": da, "over_tol": bool(bad),
                     "sandbox_source": p["source"]})

    arm = {}
    for pid, f in full.items():
        base = pid.split("_")[0]
        arm.setdefault(base, []).append(f["score"])
    print("\narm means over training seeds (full simulator, F0 only)")
    for base in sorted(arm):
        a = np.array(arm[base])
        print(f"  {base:<8} n={len(a)}  score {a.mean():.4f}  [{a.min():.4f}-{a.max():.4f}]")

    non_f0 = {pid: score(g, proto) for pid, g in
              ok[ok["fault_id"].astype(str) != "F0"].groupby("policy_id")}
    print("\nNOT part of the gate -- F2/F3 rows (faulted, sandbox cannot model):")
    print(f"  {'policy':<12}{'score':>9}{'ach':>9}{'outage ach':>12}")
    for pid in sorted(non_f0):
        g = ok[(ok["policy_id"] == pid) & (ok["fault_id"].astype(str) != "F0")]
        oa = (g["outage_achieved_steps"].sum() / g["outage_eligible_steps"].sum()
              if g["outage_eligible_steps"].sum() else float("nan"))
        print(f"  {pid:<12}{non_f0[pid]['score']:>9.4f}"
              f"{non_f0[pid]['achievement_rate']:>9.4f}{oa:>12.4f}")

    out = {"tol": TOL, "n_f0_runs": int(len(f0)), "rows": rows,
           "worst_abs_d_score_raw": float(worst),
           "worst_abs_d_score_corrected": float(worst_c),
           "gate_pass": bool(worst_c <= TOL),
           "gate_pass_raw": bool(worst <= TOL),
           "n_over_tol": int(sum(r["over_tol"] for r in rows)),
           "d22_note": ("sandbox scores are 1e-4*49*sw too high because the sandbox divided "
                        "a single group's switch count by n_groups; the gate uses the "
                        "corrected values, and both are reported")}
    (RUNS / "s5_gap_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nGATE G2 (D22-corrected): worst |d score| = {worst_c:.4f} vs tol {TOL} -> "
          f"{'PASS' if out['gate_pass'] else 'FAIL'} "
          f"({out['n_over_tol']} policy/seeds over tol)")
    print(f"          (uncorrected worst |d score| = {worst:.4f} -> "
          f"{'PASS' if out['gate_pass_raw'] else 'FAIL'})")
    print("wrote runs_v2/s5_gap_check.json")
    return 0 if out["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
