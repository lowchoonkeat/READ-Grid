"""Diagnostic: why do the authority-timing sensitivity axes come out flat?

Runs C4_s0 on the 720 s outage (F3) under three takeover thresholds and prints
the observed local-authority share, then repeats the same thresholds at 1 s
resolution against the arbiter alone. Writes no artefact; it exists so the
sensitivity result in docs/experiment_report.md §10 is explained rather than
guessed at.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))

from readgrid.arbiter import DomainArbiter                      # noqa: E402
from readgrid.protocol import load_protocol                     # noqa: E402
import run_experiments as rx                                     # noqa: E402

if __name__ == "__main__":
    rx.worker_init()
    proto = load_protocol()
    split = __import__("json").loads(
        (ROOT / "runs" / "workload_split.json").read_text(encoding="utf-8"))
    c4 = [p for p in rx.policy_instances(proto, n_train_seeds=5)
          if p["policy_id"] == "C4_s0"][0]
    for val in (2.0, 5.0, 8.0):
        ov = {"authority.takeover_after_s": val,
              "authority.suspect_after_s": max(0.0, val - 2.0)}
        job = dict(c4)
        job.update({"episode_id": split["test"][0], "fault_id": "F3",
                    "comm_seed": 0, "scale": 1.0, "protocol_override": ov})
        row = rx.run_job(job)
        print("end-to-end takeover=%.0fs  frac_local=%.5f  ach=%.4f  "
              "transfer_delay=%.1fs"
              % (val, row["frac_time_local_authority"], row["achievement_rate"],
                 row["authority_transfer_delay_s"]))

    print()
    for val in (2.0, 5.0, 8.0):
        p = rx.protocol_with_override(proto, {
            "authority.takeover_after_s": val,
            "authority.suspect_after_s": max(0.0, val - 2.0)})
        arb = DomainArbiter("d0", p)
        for t in range(1, 31):
            arb.on_tick(float(t))
        print("arbiter-only takeover=%.0fs -> transfers %s"
              % (val, [(round(t, 1), a) for t, a in arb.transfer_times]))
