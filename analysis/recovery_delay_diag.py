"""Diagnostic for the recovery-delay metric (change-log P21).

Runs one guarded policy on one outage fault and prints the arbiter's
``link_up_since`` evolution and the transfer times, so the value of
``recovery_delay_s`` can be read off the event stream rather than inferred.

Run:  python analysis/recovery_delay_diag.py
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import make_trace                                  # noqa: E402
from readgrid.arbiter import DomainArbiter                        # noqa: E402
from readgrid.controllers import RuleController, RuleParams       # noqa: E402
from readgrid.feeder import FeederModel                           # noqa: E402
from readgrid.mapping import build_mapping                        # noqa: E402
from readgrid.protocol import Protocol, load_protocol             # noqa: E402
from readgrid.sim import PolicySpec, run_simulation               # noqa: E402
from readgrid.workload import load_episodes                       # noqa: E402


def main() -> None:
    proto = load_protocol()
    raw = copy.deepcopy(proto.raw)
    raw["faults"]["F2"]["start_s"] = 1800.0          # move the outage inside the day
    p = Protocol(raw=raw, path=proto.path, sha256=proto.sha256)
    tr = make_trace("F2", p, seed=0)

    arr = np.asarray(tr.hb_arrival, dtype=float)
    mask = ~np.isnan(arr)
    print("first/last heartbeat arrival overall: %.3f .. %.3f"
          % (arr[mask][0], arr[mask][-1]))
    seg = arr[1790:2000]
    m = ~np.isnan(seg)
    print("arrivals in 1790-2000 s: %d (first %.3f, last %.3f)"
          % (int(m.sum()), seg[m][0], seg[m][-1]))
    gap = np.diff(arr[mask])
    print("heartbeat gap max %.3f s at t=%.0f s"
          % (float(gap.max()), float(arr[mask][1:][int(gap.argmax())])))

    log = []
    orig_hb = DomainArbiter.on_heartbeat
    orig_loss = DomainArbiter.on_heartbeat_loss
    orig_tick = DomainArbiter.on_tick

    def on_hb(self, t, source="cloud"):
        before = (self.link_up_since, tuple(self.transfer_times[-1:]))
        ev = orig_hb(self, t, source)
        log.append(("hb  ", round(t, 3), before, self.link_up_since,
                    tuple(self.transfer_times[-1:])))
        return ev

    def on_loss(self, t):
        orig_loss(self, t)
        log.append(("loss", round(t, 3), None, self.link_up_since,
                    None))

    def on_tick(self, t):
        ev = orig_tick(self, t)
        if self.transfer_times:
            log.append(("tick", round(t, 3), None, None,
                        tuple(self.transfer_times[-1:])))
        return ev

    DomainArbiter.on_heartbeat = on_hb
    DomainArbiter.on_heartbeat_loss = on_loss
    DomainArbiter.on_tick = on_tick
    try:
        ctl = RuleController(RuleParams(*[0.2, 0.6, 0.2, 1.2, 0.0]), n_actions=3)
        pol = PolicySpec("guarded", "rule", "rule", ctl, ctl)
        groups = build_mapping()
        eps = load_episodes()
        res, _ = run_simulation(feeder=FeederModel(groups, p), groups=groups,
                                protocol=p, episode=eps[sorted(eps)[0]],
                                scale=1.0, trace=tr, policy=pol)
    finally:
        DomainArbiter.on_heartbeat = orig_hb
        DomainArbiter.on_heartbeat_loss = orig_loss
        DomainArbiter.on_tick = orig_tick

    print("\nrecovery_delay_s = %r" % res.recovery_delay_s)
    print("authority_transfer_delay_s = %r" % res.authority_transfer_delay_s)
    print("frac_time_local_authority = %.5f" % res.frac_time_local_authority)
    print("\narbiter event stream (t, kind, link_up_since before/after, transfers):")
    for e in log:
        if e[0] == "tick" or (e[3] is not None and e[2] is None) or e[2] is None:
            continue
    # print only the informative slice: the last 40 lines
    for e in log[-40:]:
        print("   %s t=%-9s link_before=%-8s link_after=%-8s transfers=%s"
              % (e[0], e[1], e[2], e[3], e[4]))


if __name__ == "__main__":
    main()
