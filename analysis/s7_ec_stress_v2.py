"""S7 / E-C stress scenario: the outage swallows the whole evening DR window.

Why a separate protocol file: the frozen fault set has no fault that covers the
evening DR block. F3 (18:30-20:30) covers only 21.7 % of the test split's DR
steps, so the scenario the plan asks for ("the outage covers the whole evening
DR window") does not exist in the frozen design.

``configs/protocol_v2_ec.yaml`` is the frozen ``protocol_v2.yaml`` plus exactly
one new fault:

    F3E = outage, start_s 61200 (17:00), duration_s 25200 (7 h) -> 17:00-24:00

That window covers the entire evening block by construction, and the evening
block holds 56.4 % of the test split's DR steps (measured, see the S7 report).
The frozen protocol is NOT modified, so its S0 hash stays valid; every other key
of the extension equals the frozen value (asserted at build time).

The sandbox cannot model outages at all, so this scenario runs in the corrected
full simulator only.

Run:
    python analysis/s7_ec_stress_v2.py --workers 12
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO_EC = ROOT / "configs" / "protocol_v2_ec.yaml"
RUNS = ROOT / "runs_v2"
ARMS = ["C0", "C1", "C3", "C3b", "C7", "C4v2_s0", "C2v2_s0", "C4_s0"]


def main(workers: int) -> int:
    # Must be set BEFORE importing run_experiments so the workers build the F3E
    # trace from the extension protocol.
    os.environ["P359_PROTOCOL"] = str(PROTO_EC)
    sys.path.insert(0, str(ROOT / "src"))
    import run_v2 as v2
    import run_experiments as rx

    proto = v2.load_protocol(PROTO_EC)
    assert "F3E" in proto.raw["faults"], "extension protocol missing F3E"
    split = rx.load_split()
    all_pols = v2.v2_policy_set(v2.load_learner_cfg(),
                                v2.rule_params_from_winner(), 5)
    by_id = {p["policy_id"]: p for p in all_pols}
    pols = [by_id[a] for a in ARMS]
    faults = [("F3E", 0)]
    jobs = rx.expand(pols, split["test"], faults, scale=1.0, stage="ec_stress")
    print(f"[E-C stress] {len(jobs)} jobs = {len(pols)} arms x "
          f"{len(split['test'])} test episodes x 1 fault (F3E 17:00-24:00)",
          flush=True)
    st = rx.dispatch(jobs, RUNS / "stage_ec_stress.jsonl", workers=workers, chunk=2)
    man = {"stage": "ec_stress", "n_jobs": len(jobs), "n_errors": st["n_err"],
           "wall_s": st["wall_s"], "arms": ARMS, "fault": "F3E",
           "fault_spec": proto.raw["faults"]["F3E"],
           "protocol_ec_sha256": hashlib.sha256(PROTO_EC.read_bytes()).hexdigest(),
           "protocol_ec_base_sha256": proto.raw["meta"]["ec_extension_base_sha256"],
           "split_sha256": hashlib.sha256(
               (ROOT / "runs" / "workload_split.json").read_bytes()).hexdigest(),
           "src_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src").rglob("*.py"))},
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (RUNS / "ec_stress_manifest.json").write_text(json.dumps(man, indent=2),
                                                 encoding="utf-8")
    print("[E-C stress] wrote runs_v2/ec_stress_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    raise SystemExit(main(ap.parse_args().workers))
