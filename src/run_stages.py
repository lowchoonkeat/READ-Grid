"""Held-out evaluation stages: main grid, Phase 11 ablations, Phase 9 sensitivity.

Usage
-----
    python src/run_stages.py main        --workers 14
    python src/run_stages.py ablation    --workers 14
    python src/run_stages.py sensitivity --workers 14
    python src/run_stages.py sensitivity-retrain --workers 14

``main`` is the only stage that touches the held-out test split. It is launched
once, after the protocol is frozen, and never re-run with different settings.

Policy set for the main grid
----------------------------
Seven frozen instances, all evaluated over the *same* 185 (fault, comm-seed)
conditions on the *same* 21 held-out episodes so every comparison is paired:

    C0        uncontrolled reference
    C1        cloud-only rule                       (no edge fallback)
    C2_s0     cloud-only corrected-linear           (no edge fallback)
    C3        cloud rule + local rule, guarded      (non-strawman comparator)
    C4_s0     READ-Grid  (cloud linear + local linear, guarded)
    C4_s1     READ-Grid, second training seed
    C5_s0     cloud/local concatenated ablation

The remaining training seeds (C2_s1..s4, C4_s2..s4, C5_s1..s4) are scored on
the validation split and are covered in the supplementary seed-robustness stage
(``python src/run_stages.py seedrobust``). The reduction from 18 to 7 main-grid
instances is a compute-budget decision recorded in
``docs/experiment_protocol.md`` (change log P16); it does not affect the
pre-registered primary contrast (C4 vs C1, C4 vs C3), which keeps all 30
held-out communication seeds. Instances are dispatched in an order that
completes the primary contrast first, so a truncated run is still analysable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
from readgrid.protocol import load_protocol  # noqa: E402

RUNS = ROOT / "runs"
MODELS = ROOT / "models"

MAIN_POLICIES = ["C0", "C1", "C3", "C4_s0", "C4_s1", "C2_s0", "C5_s0"]

# The ablation variants are evaluated on a reduced but still paired fault set:
# the five deterministic faults plus the six stochastic faults at 10 held-out
# communication seeds each (instead of 30). The reference rows for the same
# (episode, fault, seed) cells already exist in stage_main.jsonl, so the
# ablation contrast is paired on exactly those cells.
ABLATION_STOCHASTIC_SEEDS = 10

# One-factor-at-a-time sensitivity grid. ``None`` level == the frozen value.
# (factor, dotted protocol key, [(level_label, value), ...], note)
SENSITIVITY = [
    ("heartbeat_timeout", "authority.takeover_after_s",
     [("2s", 2.0), ("3s", 3.0), ("5s", 5.0), ("8s", 8.0)],
     "suspect_after_s is moved with it to keep the 2 s detection margin"),
    ("recovery_guard", "authority.recovery_stable_s",
     [("5s", 5.0), ("10s", 10.0), ("20s", 20.0)], ""),
    ("cache_age", "target.cache_max_age_s",
     [("60s", 60.0), ("300s", 300.0), ("900s", 900.0)], ""),
    ("flexible_fraction", "flexibility.flexible_fraction",
     [("0.10", 0.10), ("0.20", 0.20), ("0.30", 0.30)], ""),
    ("min_dwell", "authority.min_dwell_s",
     [("0s", 0.0), ("10s", 10.0), ("30s", 30.0)], ""),
]

SENS_FAULTS = ["F0", "F1", "F2", "F3", "F8"]   # deterministic faults only


# --------------------------------------------------------------------- helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths) if p.is_file()}


def source_hashes() -> dict[str, str]:
    files = list((ROOT / "src").rglob("*.py"))
    files += list((ROOT / "configs").glob("*.yaml"))
    files += list((ROOT / "configs").glob("*.dss"))
    return hash_tree(files)


def model_hashes() -> dict[str, str]:
    return hash_tree(list(MODELS.glob("*.npy")))


def load_best_rule() -> list[float]:
    man = json.loads((MODELS / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    return list(man["best_rule_params"])


def select(policies: list[dict], ids: list[str]) -> list[dict]:
    want = set(ids)
    out = [p for p in policies if p["policy_id"] in want]
    missing = want - {p["policy_id"] for p in out}
    if missing:
        raise SystemExit(f"missing policy instances: {sorted(missing)}")
    return out


def write_manifest(name: str, stage: str, stats: dict, n_jobs: int,
                   policies: list[str], faults: int, episodes: int,
                   extra: dict | None = None) -> None:
    man = {
        "stage": stage,
        "protocol_sha256": load_protocol().sha256,
        "protocol_version": load_protocol().raw["meta"]["protocol_version"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_jobs": n_jobs,
        "n_errors": stats.get("n_err", 0),
        "wall_s": round(stats.get("wall_s", 0.0), 1),
        "policies": policies,
        "n_fault_conditions": faults,
        "n_episodes": episodes,
        "source_hashes": source_hashes(),
        "model_hashes": model_hashes(),
    }
    if extra:
        man.update(extra)
    (RUNS / f"{name}_manifest.json").write_text(json.dumps(man, indent=2),
                                                encoding="utf-8")
    print(f"[stages] wrote {name}_manifest.json")


# ------------------------------------------------------------------------ main
def stage_main(workers: int) -> None:
    proto = load_protocol()
    split = rx.load_split()
    rp = load_best_rule()
    pols = rx.policy_instances(proto, rule_params=rp, n_train_seeds=5)
    pols = select(pols, MAIN_POLICIES)
    faults = rx.fault_grid(proto)
    eps = split["test"]
    jobs = rx.expand(pols, eps, faults, scale=1.0, stage="main")
    print(f"[stages] MAIN: {len(jobs)} jobs = {len(pols)} policies x {len(eps)} "
          f"test episodes x {len(faults)} fault conditions")
    st = rx.dispatch(jobs, RUNS / "stage_main.jsonl", workers=workers, chunk=2)
    write_manifest("main", "main", st, len(jobs), MAIN_POLICIES,
                   len(faults), len(eps),
                   extra={"best_rule_params": rp, "scale": 1.0})


def stage_seedrobust(workers: int) -> None:
    """Supplementary: the remaining training seeds, deterministic faults only."""
    proto = load_protocol()
    split = rx.load_split()
    rp = load_best_rule()
    pols = rx.policy_instances(proto, rule_params=rp, n_train_seeds=5)
    keep = [f"C2_s{s}" for s in (1, 2, 3, 4)] + \
           [f"C4_s{s}" for s in (3, 4)] + \
           [f"C5_s{s}" for s in (1, 2, 3, 4)]
    pols = select(pols, keep)
    faults = [(f, 0) for f in SENS_FAULTS]
    jobs = rx.expand(pols, split["test"], faults, scale=1.0, stage="seedrobust")
    print(f"[stages] SEEDROBUST: {len(jobs)} jobs")
    st = rx.dispatch(jobs, RUNS / "stage_seedrobust.jsonl", workers=workers, chunk=2)
    write_manifest("seedrobust", "seedrobust", st, len(jobs), keep,
                   len(faults), len(split["test"]))


# ------------------------------------------------------------------- ablation
def _ablation_faults(proto) -> list[tuple[str, int]]:
    out = [(f, 0) for f in proto.raw["deterministic_faults"]]
    for f in proto.raw["stochastic_faults"]:
        out += [(f, s) for s in range(ABLATION_STOCHASTIC_SEEDS)]
    return out


def stage_ablation(workers: int) -> None:
    proto = load_protocol()
    split = rx.load_split()
    rp = load_best_rule()
    base = rx.policy_instances(proto, rule_params=rp, n_train_seeds=5)
    c4 = select(base, ["C4_s0"])[0]
    faults = _ablation_faults(proto)
    eps = split["test"]

    variants = []
    v = dict(c4)
    v.update({"policy_id": "A1_C4_unguarded", "policy_label": "A1 C4 heartbeat-only",
              "guard": "heartbeat_only", "variant": "guarded_vs_heartbeat_only"})
    variants.append(v)
    v = dict(c4)
    v.update({"policy_id": "A2_C4_cache_unchecked",
              "policy_label": "A2 C4 cache-age unchecked",
              "cache_enforced": False, "variant": "cache_enforced_vs_unchecked"})
    variants.append(v)
    v = dict(c4)
    v.update({"policy_id": "A3_C4_unguarded_cache_unchecked",
              "policy_label": "A3 C4 both guards removed",
              "guard": "heartbeat_only", "cache_enforced": False,
              "variant": "both_guards_removed"})
    variants.append(v)

    jobs = rx.expand(variants, eps, faults, scale=1.0, stage="ablation")
    print(f"[stages] ABLATION: {len(jobs)} jobs = {len(variants)} variants x "
          f"{len(eps)} episodes x {len(faults)} fault conditions")
    st = rx.dispatch(jobs, RUNS / "stage_ablation.jsonl", workers=workers, chunk=2)
    write_manifest("ablation", "ablation", st, len(jobs),
                   [v["policy_id"] for v in variants], len(faults), len(eps),
                   extra={"stochastic_seeds": ABLATION_STOCHASTIC_SEEDS,
                          "note": "reference rows are C4_s0/C5_s0/C3/C1 in "
                                  "stage_main.jsonl; the ablation cells are a "
                                  "subset of the main-grid cells, so every "
                                  "contrast is paired on identical "
                                  "(episode, fault, comm_seed) cells"})


# ---------------------------------------------------------------- sensitivity
def _sens_jobs(proto, split, rp, factor, key, levels, note):
    base = rx.policy_instances(proto, rule_params=rp, n_train_seeds=5)
    c4 = select(base, ["C4_s0"])[0]
    faults = [(f, 0) for f in SENS_FAULTS]
    jobs = []
    for lab, val in levels:
        ov = {key: val}
        if factor == "heartbeat_timeout":
            # keep the documented 2 s detection margin between suspect and takeover
            ov["authority.suspect_after_s"] = max(0.0, float(val) - 2.0)
        for eid in split["test"]:
            for fid, seed in faults:
                j = dict(c4)
                j.update({"policy_id": f"S_{factor}_{lab}",
                          "policy_label": f"sensitivity {factor}={lab}",
                          "episode_id": eid, "fault_id": fid, "comm_seed": seed,
                          "scale": 1.0, "stage": "sensitivity", "rng_seed": seed,
                          "factor": factor, "factor_level": lab,
                          "protocol_override": ov})
                jobs.append(j)
    return jobs


def stage_sensitivity(workers: int) -> None:
    proto = load_protocol()
    split = rx.load_split()
    rp = load_best_rule()
    all_jobs = []
    for factor, key, levels, note in SENSITIVITY:
        all_jobs += _sens_jobs(proto, split, rp, factor, key, levels, note)
    print(f"[stages] SENSITIVITY: {len(all_jobs)} jobs over "
          f"{len(SENSITIVITY)} factors, probe policy C4_s0, "
          f"faults {SENS_FAULTS}")
    st = rx.dispatch(all_jobs, RUNS / "stage_sensitivity.jsonl",
                     workers=workers, chunk=2)
    write_manifest("sensitivity", "sensitivity", st, len(all_jobs),
                   ["C4_s0"], len(SENS_FAULTS), len(split["test"]),
                   extra={"factors": [{"factor": f, "key": k,
                                       "levels": [l for l, _ in lv]}
                                      for f, k, lv, _ in SENSITIVITY]})


def stage_workload_scale(workers: int) -> None:
    """Workload-scale sensitivity (0.8 / 1.0 / 1.2) for the main policy set."""
    proto = load_protocol()
    split = rx.load_split()
    rp = load_best_rule()
    pols = select(rx.policy_instances(proto, rule_params=rp, n_train_seeds=5),
                  ["C0", "C1", "C3", "C4_s0", "C5_s0"])
    faults = [(f, 0) for f in SENS_FAULTS]
    jobs = []
    for scale in (0.8, 1.2):
        js = rx.expand(pols, split["test"], faults, scale=scale,
                       stage="sensitivity_scale")
        for j in js:
            j["factor"] = "workload_scale"
            j["factor_level"] = f"{scale:.1f}"
        jobs += js
    print(f"[stages] SCALE SENSITIVITY: {len(jobs)} jobs")
    st = rx.dispatch(jobs, RUNS / "stage_scale.jsonl", workers=workers, chunk=2)
    write_manifest("scale", "sensitivity_scale", st, len(jobs),
                   [p["policy_id"] for p in pols], len(faults), len(split["test"]),
                   extra={"scales": [0.8, 1.2],
                          "note": "scale 1.0 is stage_main.jsonl"})


# ------------------------------------------------- sensitivity requiring retrain
def stage_sensitivity_retrain(workers: int) -> None:
    """Reward-weight and feature-removal sensitivity.

    Both factors change the *training* objective or state definition, so a
    frozen checkpoint cannot answer them: the models must be retrained. The
    retrain uses the frozen training split, the frozen fitted-Q budget and the
    same 5 seeds; only the perturbed factor changes.
    """
    proto = load_protocol()
    split = rx.load_split()
    train_ids = split["train"]
    rp = load_best_rule()
    out = MODELS / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    specs = []
    # (a) reward weight on curtailment burden: 0.10 / 0.20(frozen) / 0.40
    for lab, wb in (("0.10", 0.10), ("0.40", 0.40)):
        for scope in ("cloud", "local"):
            specs.append({"scope": scope, "kind": "linear", "seed": 0,
                          "train_episodes": train_ids,
                          "protocol_override": {"reward.w_burden": wb},
                          "tag": f"reward_wburden_{lab}"})
    # (b) feature removal: drop each non-bias state feature in turn
    from readgrid.value_model import FEATURE_NAMES
    for idx in range(1, len(FEATURE_NAMES)):
        mask = [1.0] * len(FEATURE_NAMES)
        mask[idx] = 0.0
        for scope in ("cloud", "local"):
            specs.append({"scope": scope, "kind": "linear", "seed": 0,
                          "train_episodes": train_ids,
                          "feature_mask": mask,
                          "tag": f"drop_{FEATURE_NAMES[idx]}"})
    print(f"[stages] RETRAIN SENSITIVITY: {len(specs)} models")
    trained = rx.train_stage(specs, out, workers=workers)
    (out / "retrain_log.json").write_text(
        json.dumps([{"tag": r["tag"], "scope": r["scope"], "kind": r["kind"],
                     "final_rmse": r["log"][-1]["bellman_rmse"]}
                    for r in trained], indent=2), encoding="utf-8")

    # evaluate each retrained pair on the deterministic faults
    faults = [(f, 0) for f in SENS_FAULTS]
    jobs = []
    for r in trained:
        tag = r["tag"]
        # train_stage() saves as f"{tag}_{scope}_{kind}_seed{seed}.npy"
        path = out / f"{tag}_{r['scope']}_{r['kind']}_seed{r['seed']}.npy"
        if not path.exists():
            print(f"  WARNING: missing checkpoint {path.name}; arm skipped")
            continue
        jobs.append((tag, r["scope"], path))
    # group by tag into cloud/local pairs
    by_tag: dict[str, dict] = {}
    for tag, scope, path in jobs:
        by_tag.setdefault(tag, {})[scope] = path

    ejobs = []
    for tag, pair in sorted(by_tag.items()):
        cloud = {"type": "model", "path": str(pair["cloud"]), "vm": "action_dependent",
                 "name": f"{tag}_cloud"}
        local = {"type": "model", "path": str(pair["local"]), "vm": "action_dependent",
                 "name": f"{tag}_local"}
        pol = {"policy_id": f"SR_{tag}", "policy_label": f"retrain {tag}",
               "cloud": cloud, "local": local, "cloud_kind": "linear",
               "local_kind": "linear", "guard": "guarded", "cache_enforced": True,
               "factor": tag, "factor_level": "perturbed"}
        ejobs += rx.expand([pol], split["test"], faults, scale=1.0,
                           stage="sensitivity_retrain")
    print(f"[stages] evaluating {len(ejobs)} retrained-policy jobs")
    st = rx.dispatch(ejobs, RUNS / "stage_sensitivity_retrain.jsonl",
                     workers=workers, chunk=2)
    write_manifest("sensitivity_retrain", "sensitivity_retrain", st, len(ejobs),
                   sorted(by_tag), len(faults), len(split["test"]),
                   extra={"note": "frozen reference is C4_s0 in stage_main.jsonl"})


# ------------------------------------------------------------------- cli ----
# Equal-budget learned tuning (reviewer R-W2.3). The rule receives an 18-candidate
# threshold search on the validation split; before this stage the learned policy
# received no structural hyper-parameter search at all (only training seeds
# varied), so the C4-vs-C3 comparison handicapped the learned arm. These 18
# configurations give the learned policy the same budget *in kind and in size*,
# on the same validation split, under the same frozen objective.
LEARNED_TUNING_GRID = [
    {"reward.w_burden": wb, "reward.w_switch": ws, "value_model.gamma": g}
    for wb in (0.10, 0.20, 0.40)      # burden weight
    for ws in (0.00, 0.05, 0.15)      # switching weight
    for g in (0.90, 0.97)             # discount
]
# identical to the rule grid's validation conditions (train_policies.py)
TUNE_VAL_FAULTS = [("F0", 0), ("F2", 0), ("F2", 1), ("F3", 1)]


def stage_tune_learned(workers: int, *,
                       retrain: bool = False) -> None:
    """Search 18 learned hyper-parameter configurations on the validation split.

    The winning configuration becomes the budget-matched learned arm ``C4T``.
    Every configuration is scored with the frozen ``analysis.selection_score`` on
    exactly the four (fault, seed) conditions the rule grid used, so the two
    search budgets are comparable. ``--retrain`` additionally evaluates the winner
    on the held-out grid.
    """
    proto = load_protocol()
    split = rx.load_split()
    train_ids = split["train"]
    out = MODELS / "tuning"
    out.mkdir(parents=True, exist_ok=True)

    specs = []
    for i, ov in enumerate(LEARNED_TUNING_GRID):
        for scope in ("cloud", "local"):
            specs.append({"scope": scope, "kind": "linear", "seed": 0,
                          "train_episodes": train_ids,
                          "protocol_override": ov, "tag": f"cfg{i:02d}"})
    print(f"[stages] TUNE-LEARNED: {len(specs)} models "
          f"({len(LEARNED_TUNING_GRID)} configs x 2 scopes), the same 18-candidate "
          f"budget size the rule grid used (docs/experiment_protocol.md 9.6)")
    trained = rx.train_stage(specs, out, workers=workers)

    arms = []
    for i, ov in enumerate(LEARNED_TUNING_GRID):
        tag = f"cfg{i:02d}"
        c_p = out / f"{tag}_cloud_linear_seed0.npy"
        l_p = out / f"{tag}_local_linear_seed0.npy"
        if not (c_p.exists() and l_p.exists()):
            continue
        arms.append({
            "policy_id": f"LT_{tag}", "policy_label": f"tuned-learned {tag}",
            "cloud": {"type": "model", "path": str(c_p),
                      "vm": "action_dependent", "name": f"{tag}_cloud"},
            "local": {"type": "model", "path": str(l_p),
                      "vm": "action_dependent", "name": f"{tag}_local"},
            "cloud_kind": "linear", "local_kind": "linear",
            "guard": "guarded", "cache_enforced": True,
            "protocol_override": ov,
        })
    if not arms:
        raise SystemExit("no arm could be assembled; training produced no files")

    jobs = rx.expand(arms, split["val"], TUNE_VAL_FAULTS, scale=1.0,
                     stage="tune_learned")
    print(f"[stages] TUNE-LEARNED: {len(jobs)} validation jobs "
          f"({len(arms)} configs x {len(split['val'])} episodes x "
          f"{len(TUNE_VAL_FAULTS)} fault conditions)")
    st = rx.dispatch(jobs, RUNS / "stage_tune_learned.jsonl", workers=workers,
                     chunk=2)
    write_manifest("tune_learned", "tune_learned", st, len(jobs),
                   [a["policy_id"] for a in arms], len(TUNE_VAL_FAULTS),
                   len(split["val"]),
                   extra={"grid": LEARNED_TUNING_GRID, "retrain": bool(retrain),
                          "note": "validation split only; frozen objective is "
                                  "analysis.selection_score"})

    # ---- score the grid with the frozen selection objective -----------------
    import pandas as pd
    df = pd.read_json(RUNS / "stage_tune_learned.jsonl", lines=True)
    rows = []
    for i, ov in enumerate(LEARNED_TUNING_GRID):
        pid = f"LT_cfg{i:02d}"
        g = df[df["policy_id"] == pid]
        if g.empty:
            continue
        ach = float(g["achieved_steps"].sum() / max(g["eligible_steps"].sum(), 1))
        curt = float(g["curtail_energy_kwh"].sum()
                     / max(g["baseline_energy_kwh"].sum(), 1e-9))
        sw = float(g["n_switches"].sum()) / 50.0 / max(len(g), 1)
        w = proto.raw["analysis"]["selection_score"]
        score = (w["w_achievement"] * ach - w["w_curtail_fraction"] * curt
                 - w["w_switch_per_group"] * sw)
        rows.append({"config": f"cfg{i:02d}",
                     "reward.w_burden": ov["reward.w_burden"],
                     "reward.w_switch": ov["reward.w_switch"],
                     "value_model.gamma": ov["value_model.gamma"],
                     "achievement_rate": ach, "curtail_fraction": curt,
                     "switches_per_group_run": sw, "score": score,
                     "policy_id": pid})
    table = (pd.DataFrame(rows).sort_values("score", ascending=False)
             .reset_index(drop=True))
    table["rank"] = table.index + 1
    (RUNS / "learned_tuning_table.json").write_text(
        json.dumps({"grid_size": len(LEARNED_TUNING_GRID),
                    "val_faults": [list(f) for f in TUNE_VAL_FAULTS],
                    "ranking": table.to_dict("records"),
                    "best_config": table.iloc[0]["config"]}, indent=2),
        encoding="utf-8")
    print(f"[stages] TUNE-LEARNED: best = {table.iloc[0]['config']} "
          f"(score {table.iloc[0]['score']:.4f}) -> runs/learned_tuning_table.json")

    # ---- optionally evaluate the winner on the held-out grid ----------------
    if not retrain:
        return
    best = table.iloc[0]["config"]
    arm = [a for a in arms if a["policy_id"] == f"LT_{best}"][0]
    c4t = dict(arm)
    c4t.update({"policy_id": "C4T", "policy_label": "C4T learned, budget-matched"})
    jobs = rx.expand([c4t], split["test"], rx.fault_grid(proto), scale=1.0,
                     stage="main")
    print(f"[stages] C4T HELD-OUT: {len(jobs)} jobs "
          f"(1 policy x {len(split['test'])} episodes x "
          f"{len(rx.fault_grid(proto))} fault conditions)")
    st = rx.dispatch(jobs, RUNS / "stage_c4t.jsonl", workers=workers, chunk=2)
    write_manifest("c4t_heldout", "main", st, len(jobs), ["C4T"],
                   len(rx.fault_grid(proto)), len(split["test"]),
                   extra={"best_config": best,
                          "note": "budget-matched learned arm, held-out"})


# ------------------------------------------------------------------------ cli
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["main", "ablation", "sensitivity",
                                      "sensitivity-retrain", "seedrobust",
                                      "scale", "tune-learned"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--retrain", action="store_true",
                    help="with stage=tune-learned, also run the winning "
                         "configuration on the held-out grid as C4T")
    args = ap.parse_args(argv)
    if args.stage == "tune-learned":
        stage_tune_learned(args.workers, retrain=args.retrain)
        return
    fn = {
        "main": stage_main,
        "ablation": stage_ablation,
        "sensitivity": stage_sensitivity,
        "sensitivity-retrain": stage_sensitivity_retrain,
        "seedrobust": stage_seedrobust,
        "scale": stage_workload_scale,
    }[args.stage]
    fn(args.workers)


if __name__ == "__main__":
    main()
