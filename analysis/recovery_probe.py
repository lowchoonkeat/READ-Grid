"""Targeted probe for the corrected ``recovery_delay_s`` metric (change-log P21).

The held-out artefacts were produced before the P19/P21 corrections, so their
``recovery_delay_s`` column is NaN (pre-P19 sampling) and their
``authority_transfer_delay_s`` for C0 is a meaningless mean (pre-P19 prime).
This probe re-runs a *small, identical* slice of the held-out grid — two
policies on the three outage faults over all 21 held-out episodes — so the
corrected metric has a real, current number attached to it. It is deliberately
small (126 jobs) and is reported as a targeted measurement, not as a result of
the main grid.

Run:  python analysis/recovery_probe.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx                                    # noqa: E402
from run_stages import (load_protocol, load_best_rule,          # noqa: E402
                        write_manifest, MODELS)

RUNS = ROOT / "runs"


def main() -> None:
    proto = load_protocol()
    split = rx.load_split()
    rp = load_best_rule()
    base = rx.policy_instances(proto, rule_params=rp, n_train_seeds=5)
    arms = [p for p in base if p["policy_id"] in ("C4_s0", "C3")]
    faults = [("F1", 0), ("F2", 0), ("F3", 0)]
    jobs = rx.expand(arms, split["test"], faults, scale=1.0,
                     stage="recovery_probe")
    print(f"[probe] {len(jobs)} jobs = {len(arms)} policies x "
          f"{len(split['test'])} episodes x {len(faults)} fault conditions")
    st = rx.dispatch(jobs, RUNS / "stage_recovery_probe.jsonl", workers=14,
                     chunk=2)
    write_manifest("recovery_probe", "recovery_probe", st, len(jobs),
                  [a["policy_id"] for a in arms], len(faults),
                  len(split["test"]),
                  extra={"note": "targeted measurement of the P21-corrected "
                                 "recovery_delay_s metric; small slice of the "
                                 "held-out grid, not a new result"})


if __name__ == "__main__":
    main()
