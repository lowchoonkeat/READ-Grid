"""Phase 9: training, rule tuning and validation.

Stages
------
1. Build the frozen episode-level train/validation/test split (stratified by
   load regime). The split is written to ``runs/workload_split.json`` and never
   changed afterwards.
2. Tune the transparent rule on the **validation** split only (grid search).
3. Train the four learned value-model families with batch fitted-Q, on the
   **training** split only, using connected-operation traces:
   ``cloud_linear``, ``cloud_concatenated``, ``local_linear``,
   ``local_concatenated`` x 5 independent training seeds.
4. Score every frozen policy instance on the **validation** split.

The held-out test split is never touched by this script.

Run:  python src/train_policies.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
from readgrid.controllers import default_rule_grid  # noqa: E402
from readgrid.protocol import load_protocol  # noqa: E402
from readgrid.workload import load_episodes, stratified_split  # noqa: E402

RUNS = ROOT / "runs"
MODELS = ROOT / "models"


# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def selection_score(df: pd.DataFrame, protocol) -> dict:
    """Aggregate a result frame and apply the frozen selection objective."""
    cfg = protocol.raw["analysis"]["selection_score"]
    n = len(df)
    elig = df["eligible_steps"].sum()
    ach = df["achieved_steps"].sum()
    base = df["baseline_energy_kwh"].sum()
    curt = df["curtail_energy_kwh"].sum()
    sw = df["n_switches"].sum()
    achievement_rate = float(ach / elig) if elig else float("nan")
    curt_frac = float(curt / base) if base else float("nan")
    sw_per_group_run = float(sw / protocol.n_groups / max(n, 1))
    score = (cfg["w_achievement"] * achievement_rate
             - cfg["w_curtail_fraction"] * curt_frac
             - cfg["w_switch_per_group"] * sw_per_group_run)
    return {
        "n_runs": int(n),
        "achievement_rate": achievement_rate,
        "mean_exceedance": float(df["sum_exceedance"].sum() / elig) if elig else float("nan"),
        "curtail_energy_kwh": float(curt),
        "curtail_fraction": curt_frac,
        "baseline_energy_kwh": float(base),
        "energy_ratio": float(df["actual_energy_kwh"].sum() / base) if base else float("nan"),
        "n_switches": float(sw),
        "switches_per_group_run": sw_per_group_run,
        "vmin_pu": float(df["vmin_pu"].min()),
        "max_line_loading_pu": float(df["max_line_loading_pu"].max()),
        "n_convergence_failures": int(df["n_convergence_failures"].sum()),
        "score": float(score),
    }


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reuse-rule-tuning", action="store_true",
                    help="reuse runs/rule_tuning_table.json instead of re-running "
                         "the validation grid search (the rule is not a learned "
                         "policy, so its tuning is unaffected by a change to the "
                         "transition collector)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    RUNS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol()
    print(f"protocol sha256 = {protocol.sha256}", flush=True)
    n_seeds = int(protocol.raw["value_model"]["n_train_seeds"])

    # ---- 1. split ----------------------------------------------------------
    eps = load_episodes()
    split = stratified_split(
        eps,
        n_train=int(protocol.raw["split"]["n_train_episodes"]),
        n_val=int(protocol.raw["split"]["n_val_episodes"]),
        n_test=int(protocol.raw["split"]["n_test_episodes"]))
    (RUNS / "workload_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"split: train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}", flush=True)

    # ---- 2. rule tuning (validation only) ----------------------------------
    tbl = RUNS / "rule_tuning_table.json"
    if args.reuse_rule_tuning and tbl.exists():
        print("\n== stage 1/4: rule tuning (reused) ==", flush=True)
        agg = json.loads(tbl.read_text(encoding="utf-8"))
        agg.sort(key=lambda r: -r["score"])
        best = agg[0]
        rt = None
        print(f"  best rule {best['params']} score={best['score']:.4f} "
              f"ach={best['achievement_rate']:.3f}", flush=True)
    else:
        print("\n== stage 1/4: rule tuning ==", flush=True)
        rx.dispatch([j for j in _rule_jobs(protocol, split["val"])],
                    RUNS / "stage_rule_tune.jsonl", workers=args.workers)
        rt = read_jsonl(RUNS / "stage_rule_tune.jsonl")
        ok = rt[rt["ok"].astype(bool)]
        agg = []
        for variant, g in ok.groupby("variant"):
            row = selection_score(g, protocol)
            row["params"] = json.loads(variant)
            agg.append(row)
        agg.sort(key=lambda r: -r["score"])
        best = agg[0]
        print(f"  best rule {best['params']} score={best['score']:.4f} "
              f"ach={best['achievement_rate']:.3f}", flush=True)
        tbl.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # ---- 3. learned models --------------------------------------------------
    print("\n== stage 2/4: fitted-Q training ==", flush=True)
    specs = []
    for scope in ("cloud", "local"):
        for kind in ("linear", "concatenated"):
            for seed in range(n_seeds):
                specs.append({"scope": scope, "kind": kind, "seed": seed,
                              "train_episodes": split["train"]})
    trained = rx.train_stage(specs, MODELS, workers=args.workers)
    train_log = [entry for r in sorted(trained, key=lambda r: (r["scope"], r["kind"], r["seed"]))
                 for entry in r["log"]]
    (MODELS / "training_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")

    # ---- 4. validation of the frozen policy instances -----------------------
    print("\n== stage 3/4: validation of the frozen policy set ==", flush=True)
    policies = rx.policy_instances(protocol, rule_params=best["params"], n_train_seeds=n_seeds)
    val_faults = [("F0", 0), ("F2", 0), ("F2", 1), ("F3", 1)]
    jobs = rx.expand(policies, split["val"], val_faults, stage="validate")
    rx.dispatch(jobs, RUNS / "stage_validate.jsonl", workers=args.workers)

    vd = read_jsonl(RUNS / "stage_validate.jsonl")
    vok = vd[vd["ok"].astype(bool)]
    val_scores = {}
    for pid, g in vok.groupby("policy_id"):
        val_scores[pid] = selection_score(g, protocol)
    for pid in sorted(val_scores):
        s = val_scores[pid]
        print(f"  val {pid:<8} score={s['score']:.4f} ach={s['achievement_rate']:.3f} "
              f"curt_frac={s['curtail_fraction']:.4f} sw/g/r={s['switches_per_group_run']:.3f}",
              flush=True)

    # ---- 5. manifest --------------------------------------------------------
    print("\n== stage 4/4: manifest ==", flush=True)
    manifest = {
        "protocol_sha256": protocol.sha256,
        "protocol_version": protocol.raw["meta"]["protocol_version"],
        "best_rule_params": best["params"],
        "best_rule_score": best["score"],
        "n_train_seeds": n_seeds,
        "n_val_episodes": len(split["val"]),
        "n_train_episodes": len(split["train"]),
        "n_test_episodes": len(split["test"]),
        "val_faults": [list(f) for f in val_faults],
        "models": sorted(p.name for p in MODELS.glob("*.npy")),
        "validation_scores": val_scores,
        "rule_grid_size": len(default_rule_grid()),
        "errors": {
            "rule_tune": (int((~rt["ok"].astype(bool)).sum()) if rt is not None else 0),
            "validate": int((~vd["ok"].astype(bool)).sum()),
        },
        "rule_tuning_reused": bool(args.reuse_rule_tuning and tbl.exists()),
    }
    (MODELS / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\ntraining pipeline complete in {time.perf_counter() - t0:.0f}s", flush=True)


def _rule_jobs(protocol, val_ids: list[str]) -> list[dict]:
    faults = [("F0", 0), ("F2", 0), ("F2", 1)]
    jobs = []
    for i, params in enumerate(default_rule_grid()):
        p = list(params.as_tuple())
        ctl = {"type": "rule", "params": p}
        for eid in val_ids:
            for fid, seed in faults:
                jobs.append({
                    "stage": "rule_tune", "policy_id": f"rule{i:02d}",
                    "policy_label": f"rule{i:02d}", "variant": json.dumps(p),
                    "cloud": ctl, "local": ctl,
                    "cloud_kind": "rule", "local_kind": "rule",
                    "episode_id": eid, "fault_id": fid, "comm_seed": seed,
                    "scale": 1.0, "rng_seed": seed,
                })
    return jobs


if __name__ == "__main__":
    main()
