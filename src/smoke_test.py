"""Quick end-to-end smoke test (short horizon) for the READ-Grid rebuild.

Run:  python src/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))

from readgrid.arbiter import Command  # noqa: E402
from readgrid.comm import generate_fault_trace  # noqa: E402
from readgrid.controllers import LearnedController, NullController, RuleController, RuleParams  # noqa: E402
from readgrid.feeder import FeederModel  # noqa: E402
from readgrid.mapping import build_mapping  # noqa: E402
from readgrid.protocol import Protocol, load_protocol  # noqa: E402
from readgrid.sim import PolicySpec, run_simulation  # noqa: E402
from readgrid.value_model import ValueModel  # noqa: E402
from readgrid.workload import load_episodes  # noqa: E402


def short_protocol(hours: float = 24.0) -> Protocol:
    p = load_protocol()
    raw = deepcopy(p.raw)
    raw["workload"]["episode_length_s"] = int(hours * 3600)
    return Protocol(raw=raw, path=p.path, sha256=p.sha256)


def main() -> None:
    t0 = time.perf_counter()
    print("== mapping ==")
    groups = build_mapping()
    print(f"  groups={len(groups)} loads={sum(len(g.groups if False else g.load_names) for g in groups)}")
    print(f"  total kW={sum(g.base_kw for g in groups):.2f} kvar={sum(g.base_kvar for g in groups):.2f}")
    print(f"  domains={sorted({g.domain_id for g in groups})}")

    proto = short_protocol(24.0)
    print(f"== feeder (episode {proto.episode_length_s}s, {proto.steps_per_episode} steps) ==")
    feeder = FeederModel(groups, proto)
    prov = feeder.provenance()
    print(f"  loads={prov['n_loads']} lines={prov['n_lines']} unrated={prov['n_unrated_lines']}")
    print(f"  total base kW={prov['total_base_kw']:.2f}")
    print(f"  rated line NormAmps set: {sorted(set(feeder.rated_normamps))}")
    print(f"  unrated lines: {sorted(feeder.unrated_lines)}")

    print("== episodes ==")
    eps = load_episodes()
    eid = sorted(eps)[0]
    ep = eps[eid]
    dr = ep.shape >= proto.dr_trigger_fraction
    print(f"  {eid} regime={ep.regime} clip={ep.clip_fraction:.4f} shape mean={ep.shape.mean():.3f} "
          f"DR fraction={dr.mean():.3f} s_ref(median DR)={np.median(ep.shape[dr]):.3f}")

    print("== fault traces ==")
    for fid in ("F0", "F1", "F4", "F8"):
        cfg = proto.raw["faults"][fid]
        tr = generate_fault_trace(fid, cfg, proto, seed=1)
        delivered = int(np.sum(~np.isnan(tr.hb_arrival)))
        print(f"  {fid}: hb delivered {delivered}/{tr.n_seconds}  "
              f"c2e delivered {int(np.sum(~np.isnan(tr.c2e_arrival)))}/{tr.n_ticks}")

    print("== policies ==")
    c0 = PolicySpec("C0", None, None, None, None)
    rule = RuleController(RuleParams())
    c1 = PolicySpec("C1", "rule", None, rule, None)
    c3 = PolicySpec("C3", "rule", "rule", rule, rule)
    lin = LearnedController(ValueModel(3, "action_dependent", seed=0), "lin")
    c4 = PolicySpec("C4", "linear", "linear", lin, lin)

    tr0 = generate_fault_trace("F0", proto.raw["faults"]["F0"], proto, seed=1)
    tr1 = generate_fault_trace("F1", proto.raw["faults"]["F1"], proto, seed=1)

    for pol, tr in [(c0, tr0), (c1, tr0), (c3, tr1), (c4, tr1)]:
        t1 = time.perf_counter()
        res, _ = run_simulation(feeder=feeder, groups=groups, protocol=proto,
                                episode=ep, scale=1.0, trace=tr, policy=pol)
        dt = time.perf_counter() - t1
        print(f"  {pol.policy_id:>3} {tr.fault_id}: elig={res.eligible_steps} "
              f"ach={res.achieved_steps} exc={res.sum_exceedance:.3f} "
              f"E={res.actual_energy_kwh:.1f}/{res.baseline_energy_kwh:.1f} kWh "
              f"curt={res.curtail_energy_kwh:.2f} sw={res.n_switches} "
              f"vmin={res.vmin_pu:.4f} lmax={res.max_line_loading_pu:.3f} "
              f"conv_fail={res.n_convergence_failures} "
              f"local_frac={res.frac_time_local_authority:.2f} "
              f"acc/rej={res.n_commands_accepted}/{res.n_commands_rejected} "
              f"[{dt:.2f}s wall, {res.wall_s:.2f}s sim]")
        if res.reject_reasons:
            print(f"        rejects: {res.reject_reasons}")

    print(f"\ntotal wall {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
