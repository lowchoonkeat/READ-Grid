"""Export raw authority/command event traces for the connected / outage / flapping
/ recovery cases (v2 plan S1.3).

The v2 plan requires the raw event timeline, not just the aggregated metrics:
the claim under test is that authority moves atomically, that stale or duplicated
old commands are fenced by epoch and sequence, and that recovery requires a
stable-link interval rather than a single packet. None of that is visible in a
summary row.

Scope: corrected full simulator only. The feeder-free replica does not model
outages or flapping at all, so it cannot be used for these cases.

Output: runs_v2/event_traces/<fault>__<episode>.jsonl plus an index JSON.

Usage:
    python analysis/export_event_traces.py [episode_id ...]
"""
from __future__ import annotations

import json
import os
import sys

PROJ = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "src"))

import run_experiments as rx                                            # noqa: E402
from readgrid.controllers import RuleController, RuleParams             # noqa: E402
from readgrid.sim import PolicySpec, run_simulation                     # noqa: E402

OUT_DIR = os.path.join(PROJ, "runs_v2", "event_traces")

# F0 connected | F1/F2/F3 outage | F8 flapping | F9 delayed duplicates | F10 stale cache
FAULTS = ["F0", "F1", "F2", "F3", "F8", "F9", "F10"]
DEFAULT_EPISODES = ["LVP002", "LVP005"]


def main() -> int:
    rx.worker_init()
    C = rx._CTX
    os.makedirs(OUT_DIR, exist_ok=True)

    eps = sys.argv[1:] or DEFAULT_EPISODES
    th = RuleParams()
    index = []

    for eid in eps:
        episode = C["episodes"][eid]
        for fid in FAULTS:
            trace = rx._get_trace(fid, 0)
            pol = PolicySpec("trace", "rule", "rule",
                             RuleController(th, C["protocol"].n_actions),
                             RuleController(th, C["protocol"].n_actions))
            res, _ = run_simulation(
                feeder=C["feeder"], groups=C["groups"], protocol=C["protocol"],
                episode=episode, scale=1.0, trace=trace, policy=pol,
                record_events=True, rng_seed=0)

            events = res.extra.get("events", [])
            path = os.path.join(OUT_DIR, f"{fid}__{eid}.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for e in events:
                    fh.write(json.dumps(e, sort_keys=True) + "\n")

            index.append(dict(
                fault=fid, episode=eid, path=os.path.relpath(path, PROJ),
                n_events=len(events),
                n_authority_switches=int(res.n_authority_switches),
                n_commands_accepted=int(res.n_commands_accepted),
                n_commands_rejected=int(res.n_commands_rejected),
                reject_reasons=res.reject_reasons,
                outage_window_s=float(res.outage_window_s),
                heartbeat_detect_delay_s=(None if res.heartbeat_detect_delay_s != res.heartbeat_detect_delay_s
                                          else float(res.heartbeat_detect_delay_s)),
                authority_transfer_delay_s=(None if res.authority_transfer_delay_s != res.authority_transfer_delay_s
                                            else float(res.authority_transfer_delay_s)),
                recovery_delay_s=(None if res.recovery_delay_s != res.recovery_delay_s
                                  else float(res.recovery_delay_s)),
            ))
            print("%-5s %-8s events=%-5d transfers=%d accepted=%d rejected=%d"
                  % (fid, eid, len(events), res.n_authority_switches,
                     res.n_commands_accepted, res.n_commands_rejected))

    with open(os.path.join(OUT_DIR, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    print("wrote", os.path.join(OUT_DIR, "index.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
