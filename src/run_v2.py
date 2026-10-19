"""P359 / READ-Grid v2 runner: frozen-learner training, validation (S5) and the
single held-out evaluation (S6).

Everything this script writes goes to ``runs_v2/`` and ``models_v2/``. The v1
artefacts in ``runs/`` and ``models/`` are read-only inputs (the frozen split and
the frozen v1 checkpoints used as the historical C4 arm).

The learned arm's hyper-parameters are NOT hard-coded here: they are read from
``configs/learner_v2_frozen.yaml``, the artefact produced by the pre-registered
S3 sweep, and applied as dotted-path protocol overrides. That keeps
``configs/protocol_v2.yaml`` byte-identical to the hash recorded at S0.

Stages
------
  train     train the frozen learner (C4v2 / C2v2) on the frozen TRAIN split
  validate  S5 gate: frozen policies on the VAL split under F0 / F2 / F3
  main      S6: the single held-out evaluation on the TEST split (all faults)

Run:
    python src/run_v2.py train    --workers 14
    python src/run_v2.py validate --workers 14
    python src/run_v2.py main     --workers 14
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx                                    # noqa: E402
from readgrid.protocol import load_protocol                      # noqa: E402
from readgrid.value_model import STATE_DIM, feature_dim          # noqa: E402

RUNS_V2 = ROOT / "runs_v2"
MODELS_V2 = ROOT / "models_v2"
MODELS_V1 = ROOT / "models"
LEARNER_CFG = ROOT / "configs" / "learner_v2_frozen.yaml"
PROTOCOL_V2 = ROOT / "configs" / "protocol_v2.yaml"
RULE_WINNER = RUNS_V2 / "s4_rule_winner.json"

# Make every worker process (and this one) base itself on the v2 protocol. The
# v1 default is untouched, so v1 runs keep their recorded hash.
os.environ.setdefault("P359_PROTOCOL", str(PROTOCOL_V2))

# The validation conditions used for BOTH families' candidate scoring. Identical
# to the v1 rule-tuning conditions so the "same frozen objective on the same
# validation split" claim is literally true.
VAL_FAULTS = [("F0", 0), ("F2", 0), ("F2", 1), ("F3", 1)]

# The v1 arms must run under the v1 representation, not the v2 default. Written
# as an explicit per-policy override so it is recorded in every result row.
V1_OVERRIDE = {"value_model.feature_variant": "v1"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_learner_cfg() -> dict:
    return yaml.safe_load(LEARNER_CFG.read_text(encoding="utf-8"))


def rule_params_from_winner() -> list[float]:
    """The frozen C3b rule++ parameters, chosen by the pre-registered S4 rule."""
    w = json.loads(RULE_WINNER.read_text(encoding="utf-8"))["winner"]
    return [w["th_low"], w["th_high"], w["hyst"], 1.20, 0.0]


def selection_score(df, protocol) -> dict:
    """The frozen selection objective, unchanged from src/train_policies.py."""
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
    return {"n_runs": int(n), "achievement_rate": achievement_rate,
            "curtail_fraction": curt_frac, "n_switches": float(sw),
            "switches_per_group_run": sw_per_group_run, "score": float(score),
            "energy_ratio": float(df["actual_energy_kwh"].sum() / base) if base else float("nan"),
            "outage_achievement_rate": float(
                df["outage_achieved_steps"].sum() / df["outage_eligible_steps"].sum())
            if df["outage_eligible_steps"].sum() else float("nan")}


# --------------------------------------------------------------------------- #
# policy set
# --------------------------------------------------------------------------- #
def v2_policy_set(cfg: dict, rule_params: list[float], n_seeds: int,
                  *, include_learned: bool = True) -> list[dict]:
    """The frozen v2 policy set, with every arm carrying its own provenance."""
    rp = list(rule_params)
    rule_ctl = {"type": "rule", "params": rp}
    c7_ctl = {"type": "minimal_sufficient"}
    dep = cfg["deployment"]

    out = [
        {"policy_id": "C0", "policy_label": "C0 uncontrolled",
         "cloud": None, "local": None, "cloud_kind": None, "local_kind": None},
        {"policy_id": "C1", "policy_label": "C1 cloud rule (no fallback)",
         "cloud": rule_ctl, "local": None, "cloud_kind": "rule", "local_kind": None},
        {"policy_id": "C3", "policy_label": "C3 v1 rule + rule fallback (guarded)",
         "cloud": {"type": "rule", "params": [0.20, 0.60, 0.20, 1.20, 0.0]},
         "local": {"type": "rule", "params": [0.20, 0.60, 0.20, 1.20, 0.0]},
         "cloud_kind": "rule", "local_kind": "rule"},
        {"policy_id": "C3b", "policy_label": "C3b rule++ (144-candidate winner)",
         "cloud": rule_ctl, "local": rule_ctl,
         "cloud_kind": "rule", "local_kind": "rule"},
        {"policy_id": "C7", "policy_label": "C7 minimal sufficient level (no params)",
         "cloud": c7_ctl, "local": c7_ctl,
         "cloud_kind": "minimal_sufficient", "local_kind": "minimal_sufficient"},
    ]
    if not include_learned:
        return out

    def _v2_model(name: str) -> dict:
        return {"type": "model", "path": str(MODELS_V2 / name), "vm": dep["vm"],
                "name": name, "state_dim": int(dep["state_dim"]),
                "mask_admissible": bool(dep["mask_admissible"])}

    def _v1_model(name: str, vm: str) -> dict:
        # the frozen v1 checkpoints: 8-dim raw features, unmasked greedy.
        # The override is written out explicitly rather than relying on the v2
        # protocol default, so every result row carries its own provenance.
        return {"type": "model", "path": str(MODELS_V1 / name), "vm": vm,
                "name": name, "state_dim": STATE_DIM, "mask_admissible": False,
                "protocol_override": V1_OVERRIDE}

    for seed in range(n_seeds):
        out.append({
            "policy_id": f"C4v2_s{seed}",
            "policy_label": f"C4v2 frozen learner seed {seed}",
            "cloud": _v2_model(f"v2_cloud_linear_seed{seed}.npy"),
            "local": _v2_model(f"v2_local_linear_seed{seed}.npy"),
            "cloud_kind": "linear", "local_kind": "linear", "train_seed": seed,
            "protocol_override": cfg["protocol_override"],
        })
        out.append({
            "policy_id": f"C2v2_s{seed}",
            "policy_label": f"C2v2 frozen cloud learner seed {seed}",
            "cloud": _v2_model(f"v2_cloud_linear_seed{seed}.npy"),
            "local": None, "cloud_kind": "linear", "local_kind": None,
            "train_seed": seed,
            "protocol_override": cfg["protocol_override"],
        })
        out.append({
            "policy_id": f"C4_s{seed}",
            "policy_label": f"C4 v1 frozen checkpoint seed {seed}",
            "cloud": _v1_model(f"cloud_linear_seed{seed}.npy", "action_dependent"),
            "local": _v1_model(f"local_linear_seed{seed}.npy", "action_dependent"),
            "cloud_kind": "linear", "local_kind": "linear", "train_seed": seed,
        })
        out.append({
            "policy_id": f"C2_s{seed}",
            "policy_label": f"C2 v1 cloud-only checkpoint seed {seed}",
            "cloud": _v1_model(f"cloud_linear_seed{seed}.npy", "action_dependent"),
            "local": None, "cloud_kind": "linear", "local_kind": None,
            "train_seed": seed,
        })
        out.append({
            "policy_id": f"C5_s{seed}",
            "policy_label": f"C5 v1 concatenated ablation seed {seed}",
            "cloud": _v1_model(f"cloud_concatenated_seed{seed}.npy", "concatenated"),
            "local": _v1_model(f"local_concatenated_seed{seed}.npy", "concatenated"),
            "cloud_kind": "concatenated", "local_kind": "concatenated",
            "train_seed": seed,
        })
    for p in out:
        p.setdefault("guard", "guarded")
        p.setdefault("cache_enforced", True)
    return out


def write_manifest(name: str, stage: str, stats: dict, n_jobs: int,
                   policy_ids: list[str], n_faults: int, n_eps: int,
                   *, extra: dict | None = None) -> Path:
    man = {
        "name": name, "stage": stage,
        "n_jobs": int(n_jobs), "n_policies": len(policy_ids),
        "n_fault_conditions": int(n_faults), "n_episodes": int(n_eps),
        "policies": sorted(policy_ids),
        "protocol_sha256": load_protocol(PROTOCOL_V2).sha256,
        "protocol_v2_sha256": sha256_file(ROOT / "configs" / "protocol_v2.yaml"),
        "learner_config_sha256": sha256_file(LEARNER_CFG),
        "split_sha256": sha256_file(ROOT / "runs" / "workload_split.json"),
        "data_index_sha256": sha256_file(ROOT / "data" / "processed" / "workload_episodes.csv")
        if (ROOT / "data" / "processed" / "workload_episodes.csv").exists() else None,
        "src_sha256": {str(p.relative_to(ROOT)): sha256_file(p)
                       for p in sorted((ROOT / "src").rglob("*.py"))},
        "checkpoint_sha256": {p.name: sha256_file(p)
                              for p in sorted(MODELS_V2.glob("*.npy"))},
        "wall_s": stats.get("wall_s"), "n_errors": stats.get("n_err"),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        man.update(extra)
    path = RUNS_V2 / f"{name}_manifest.json"
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_train(workers: int) -> None:
    cfg = load_learner_cfg()
    split = rx.load_split()
    MODELS_V2.mkdir(parents=True, exist_ok=True)
    n_seeds = int(cfg["n_final_train_seeds"])
    ov = cfg["protocol_override"]
    specs = []
    for seed in range(n_seeds):
        for scope in ("cloud", "local"):
            specs.append({"scope": scope, "kind": "linear", "seed": seed,
                          "train_episodes": split["train"],
                          "protocol_override": ov, "tag": "v2"})
    print(f"[v2] TRAIN: {len(specs)} models "
          f"({n_seeds} seeds x 2 scopes), frozen learner config "
          f"{sha256_file(LEARNER_CFG)[:16]}", flush=True)
    t0 = time.perf_counter()
    trained = rx.train_stage(specs, MODELS_V2, workers=workers)
    log = [e for r in sorted(trained, key=lambda r: (r["scope"], r["seed"]))
           for e in r["log"]]
    (MODELS_V2 / "training_log_v2.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8")
    write_manifest("train_v2", "train", {"wall_s": time.perf_counter() - t0,
                                         "n_err": 0},
                   len(specs), [f"v2_{s}_linear_seed{k}" for k in range(n_seeds)
                                for s in ("cloud", "local")],
                   0, len(split["train"]), extra={"n_train_seeds": n_seeds})
    print(f"[v2] TRAIN done in {time.perf_counter() - t0:.0f}s", flush=True)


def _run_stage(stage: str, episode_ids: list[str], faults, workers: int,
               out_name: str) -> tuple[dict, list[str]]:
    cfg = load_learner_cfg()
    proto = load_protocol(PROTOCOL_V2)
    pols = v2_policy_set(cfg, rule_params_from_winner(),
                         int(cfg["n_final_train_seeds"]))
    jobs = rx.expand(pols, episode_ids, faults, scale=1.0, stage=stage)
    print(f"[v2] {stage.upper()}: {len(jobs)} jobs "
          f"({len(pols)} policies x {len(episode_ids)} episodes x "
          f"{len(faults)} fault conditions)", flush=True)
    st = rx.dispatch(jobs, RUNS_V2 / f"stage_{out_name}.jsonl",
                     workers=workers, chunk=2)
    write_manifest(out_name, stage, st, len(jobs),
                   [p["policy_id"] for p in pols], len(faults),
                   len(episode_ids))
    return st, [p["policy_id"] for p in pols]


def stage_validate(workers: int) -> None:
    split = rx.load_split()
    _run_stage("validate_v2", split["val"], VAL_FAULTS, workers, "validate_v2")
    report_val(load_protocol(PROTOCOL_V2))


def stage_main(workers: int) -> None:
    split = rx.load_split()
    proto = load_protocol(PROTOCOL_V2)
    faults = rx.fault_grid(proto)
    _run_stage("main_v2", split["test"], faults, workers, "main_v2")


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def report_val(protocol) -> None:
    import pandas as pd
    df = pd.read_json(RUNS_V2 / "stage_validate_v2.jsonl", lines=True)
    ok = df[df["ok"].astype(bool)]
    rows = []
    for pid, g in ok.groupby("policy_id"):
        s = selection_score(g, protocol)
        s["policy_id"] = pid
        rows.append(s)
    tbl = sorted(rows, key=lambda r: -r["score"])
    print(f"\n[v2] VALIDATION (VAL split, {len(VAL_FAULTS)} fault conditions, "
          f"frozen selection_score)")
    print(f"  {'policy':<12}{'score':>9}{'ach':>9}{'curt':>9}{'sw/g/r':>10}")
    for r in tbl:
        print(f"  {r['policy_id']:<12}{r['score']:>9.4f}{r['achievement_rate']:>9.4f}"
              f"{r['curtail_fraction']:>9.4f}{r['switches_per_group_run']:>10.4f}")
    by = {r["policy_id"]: r for r in tbl}
    groups = {}
    for pid, r in by.items():
        base = pid.split("_")[0]
        groups.setdefault(base, []).append(r)
    print("\n[v2] arm means over training seeds")
    for base in sorted(groups):
        g = groups[base]
        a = np.array([r["achievement_rate"] for r in g])
        s = np.array([r["score"] for r in g])
        print(f"  {base:<8} n={len(g)}  score {s.mean():.4f}  ach {a.mean():.4f}"
              f"  [{a.min():.4f}-{a.max():.4f}]")
    (RUNS_V2 / "validation_scores_v2.json").write_text(
        json.dumps(tbl, indent=2), encoding="utf-8")
    print(f"\nwrote runs_v2/validation_scores_v2.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "validate", "main", "report"])
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    {"train": stage_train,
     "validate": stage_validate,
     "main": stage_main,
     "report": lambda w: report_val(load_protocol(PROTOCOL_V2))}[args.stage](args.workers)
