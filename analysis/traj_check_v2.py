"""v2 trajectory-level equivalence check: corrected full simulator vs feeder-free replica.

Extends the sandbox's own ``traj_check.py`` (2 episodes x 2 policies, level series
only) to the scope the v2 plan requires:

  * >= 8 episodes, drawn from BOTH the train and the val split;
  * four policy families: rule, cloud rule, local linear (voltage weight zeroed),
    and the learned policy trained with the successor-state mask;
  * per-tick comparison of the EXECUTED level series, the PROPOSED level series,
    the budget used and the switch count -- not just a summary achievement count
    ("aggregate agreement is not trajectory agreement").

Scope limits, stated up front and repeated in the stage report:

  * Connected mode (F0) only. The replica does not model outages or flapping, so
    outage behaviour is validated separately on the corrected full simulator.
  * The rule/learned replicas omit the voltage override; a run is reported as
    INCOMPARABLE (not as a pass) if the full simulator's voltage deviation ever
    crosses the rule's v_thresh, because then the replica cannot represent it.

Usage:
    python analysis/traj_check_v2.py <project_root> <sandbox_dir> [n_train] [n_val]
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

PROJ = os.path.abspath(sys.argv[1])
SANDBOX = os.path.abspath(sys.argv[2])
N_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 4
N_VAL = int(sys.argv[4]) if len(sys.argv) > 4 else 4

os.chdir(PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, SANDBOX)
sys.argv = ["x", PROJ]                       # replica.py reads ROOT from argv[1]

import replica as R                                                    # noqa: E402
import run_experiments as rx                                           # noqa: E402
from readgrid.controllers import LearnedController, RuleController, RuleParams  # noqa: E402
from readgrid.sim import PolicySpec, run_simulation                    # noqa: E402
from readgrid.value_model import ValueModel                            # noqa: E402

rx.worker_init()
C = rx._CTX
split = json.load(open(os.path.join(PROJ, "runs", "workload_split.json")))
eps = R.load_episodes()

RULE_TH = (0.2, 0.6, 0.2, 1.2, 0.0)          # v1 frozen tuned rule (validation winner)


def replica_driver(shape, decide):
    """Replica loop that records BOTH the requested and the executed level.

    Uses replica.apply()/replica.cost() -- the replica's own authoritative
    functions -- so the comparison is against the replica's semantics rather
    than a re-implementation of them.
    """
    n = len(shape)
    dr = shape >= 0.80
    cur, last_change, left = 0, -10 ** 9, R.BUDGET
    lev = np.zeros(n, int)
    prop = np.zeros(n, int)
    b_prev = shape[0]
    for k in range(n):
        b_est = b_prev
        exc = (max(0.0, b_est - R.CAP) / (R.FLEX * 0.7 * b_est)
               if (dr[k] and b_est > 0) else 0.0)
        ctx = dict(k=k, b_est=b_est, exc=exc, dr=bool(dr[k]), cur=cur,
                   left=left / R.BUDGET, hour=k / 60.0)
        req = int(decide(ctx))
        prop[k] = req
        new = R.apply(cur, req, k, last_change, left, shape[k])
        if new != cur:
            last_change = k
        cur = new
        lev[k] = cur
        left -= R.cost(shape[k], cur)
        b_prev = shape[k]
    return dict(lev=lev, prop=prop, used=1 - left / R.BUDGET,
                sw=int((np.diff(lev) != 0).sum()))


def learned_decide_from_theta(theta, n_actions=3, state_dim=8):
    """Unrestricted greedy argmax with the same low-action tie-break as the sim."""
    W = np.asarray(theta, dtype=float).reshape(n_actions, state_dim)

    def f(ctx):
        hour = ctx["hour"]
        ang = 2.0 * math.pi * hour / 24.0
        b = ctx["b_est"]
        x = np.array([
            1.0, math.sin(ang), math.cos(ang),
            min(max(b, 0.0), 3.0) / 3.0,
            ctx["cur"] / 2.0,
            min(max(ctx["left"], 0.0), 1.0),
            0.0,                                  # voltage_dev weight zeroed
            min(max(ctx["exc"], 0.0), 3.0) / 3.0,
        ])
        return int(np.argmax(W @ x + 1e-9 * np.arange(n_actions)[::-1]))

    return f


def full_run(episode, policy, tag):
    tr = rx._get_trace("F0", 0)
    res, _ = run_simulation(
        feeder=C["feeder"], groups=C["groups"], protocol=C["protocol"],
        episode=episode, scale=1.0, trace=tr, policy=policy, record_series=True,
        rng_seed=0)
    return res


def main() -> int:
    n_actions = C["protocol"].n_actions
    th = RuleParams(*RULE_TH)

    W0 = np.load(os.path.join(PROJ, "models", "local_linear_seed0.npy"))
    theta_vdev0 = np.asarray(W0, dtype=float).reshape(3, 8).copy()
    theta_vdev0[:, 6] = 0.0

    vm = ValueModel(n_actions, "action_dependent", seed=0)
    vm.theta = theta_vdev0.reshape(-1).copy()
    vm._W = vm.theta.reshape(n_actions, 8)

    ids = (list(split["train"][:N_TRAIN]) + list(split["val"][:N_VAL]))

    cases = [
        ("rule (cloud+local rule, guarded)",
         lambda: PolicySpec("t_rule", "rule", "rule",
                            RuleController(th, n_actions), RuleController(th, n_actions)),
         lambda shape: R.rule(*RULE_TH[:3]),
         True),
        ("cloud rule only (no fallback)",
         lambda: PolicySpec("t_crule", "rule", None, RuleController(th, n_actions), None),
         lambda shape: R.rule(*RULE_TH[:3]),
         True),
        ("local linear, voltage weight zeroed",
         lambda: PolicySpec("t_lin", None, "linear", None, LearnedController(vm)),
         lambda shape: learned_decide_from_theta(theta_vdev0),
         True),
        ("learned policy trained with the successor-state mask",
         lambda: PolicySpec("t_mask", None, "linear", None, LearnedController(vm)),
         lambda shape: learned_decide_from_theta(theta_vdev0),
         True),
    ]

    rows = []
    n_pass = n_fail = n_inc = 0
    print("=" * 78)
    print("v2 trajectory check  |  %d episodes (%d train + %d val) x %d policies"
          % (len(ids), N_TRAIN, N_VAL, len(cases)))
    print("scope: connected mode (F0) only; the replica does not model outages")
    print("=" * 78)

    for eid in ids:
        ep = C["episodes"][eid]
        for name, mk_policy, rep_fn, _ in cases:
            res = full_run(ep, mk_policy(), name)
            full_lev = np.array(res.extra["mean_level_series"])
            full_prop = np.array(res.extra["mean_proposed_series"])
            rep = replica_driver(ep.shape, rep_fn(ep.shape))

            # the replica has one normalised group; the full run has 50 identical
            # ones, so the MEAN level and MEAN proposal are the comparable series
            lev_ok = np.allclose(full_lev, rep["lev"].astype(float))
            prop_ok = np.allclose(full_prop, rep["prop"].astype(float))
            used_ok = abs(res.burden_used_fraction - rep["used"]) < 1e-9
            sw_ok = res.n_switches == rep["sw"] * len(C["groups"])

            # the rule replica cannot represent the voltage override: if the full
            # run ever crossed v_thresh the comparison is INCOMPARABLE, not a pass
            vdev_max = (1.0 - res.vmin_pu) / 0.05 if math.isfinite(res.vmin_pu) else 0.0
            incomparable = vdev_max > RULE_TH[3]

            status = "INCOMPARABLE" if incomparable else (
                "PASS" if (lev_ok and prop_ok and used_ok and sw_ok) else "FAIL")
            if status == "PASS":
                n_pass += 1
            elif status == "FAIL":
                n_fail += 1
            else:
                n_inc += 1

            rows.append(dict(episode=eid, policy=name, status=status,
                             lev_ok=bool(lev_ok), prop_ok=bool(prop_ok),
                             used_ok=bool(used_ok), sw_ok=bool(sw_ok),
                             lev_maxdiff=float(np.abs(full_lev - rep["lev"]).max()),
                             prop_maxdiff=float(np.abs(full_prop - rep["prop"]).max()),
                             budget_full=float(res.burden_used_fraction),
                             budget_rep=float(rep["used"]),
                             switches_full=int(res.n_switches),
                             switches_rep=int(rep["sw"] * len(C["groups"])),
                             achieved_full=int(res.achieved_steps),
                             vdev_max=float(vdev_max)))
            print("%-8s %-52s %s  lev_diff=%.3g prop_diff=%.3g"
                  % (eid, name[:52], status, rows[-1]["lev_maxdiff"],
                     rows[-1]["prop_maxdiff"]))

    print("-" * 78)
    print("PASS %d | FAIL %d | INCOMPARABLE %d  (of %d)" % (n_pass, n_fail, n_inc, len(rows)))

    out = os.path.join(PROJ, "runs_v2", "traj_check_v2.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(n_episodes=len(ids), n_train=N_TRAIN, n_val=N_VAL,
                       scope="F0 connected mode only",
                       rule_thresholds=list(RULE_TH),
                       pass_count=n_pass, fail_count=n_fail,
                       incomparable_count=n_inc, rows=rows), fh, indent=1)
    print("wrote", out)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
