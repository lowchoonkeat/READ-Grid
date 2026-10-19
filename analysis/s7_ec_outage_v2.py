"""S7 / E-C: outage-focus metrics and the raw flapping event trace.

Three deliverables, all from data that already exists (no new simulation):

1. Outage-window metrics per arm -- achievement, switch count and energy ratio
   restricted to "outage + the 30 minutes that follow recovery", which the
   simulator already accumulates in the ``outage_*`` columns. Broken down by
   outage duration class (F1 5 min, F2 30 min, F3 2 h).
2. The "the outage covers the whole evening DR window" stress scenario. F3 is an
   outage from 18:30 to 20:30, so whether it really covers the DR window is a
   question about the workload, not about the policy: this script measures the
   overlap from the episode shapes themselves.
3. The raw F8 flapping timeline: every authority transition with its epoch and
   the takeover / recovery delays, read from ``runs_v2/event_traces/F8__*.jsonl``.
   Aggregates cannot show that authority moves atomically, so the raw trace is
   the evidence.

Usage:
    python analysis/s7_ec_outage_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from readgrid.protocol import load_protocol                       # noqa: E402
from readgrid.workload import load_episodes                        # noqa: E402

RUNS = ROOT / "runs_v2"
TRACES = RUNS / "event_traces"
PROTO = load_protocol(ROOT / "configs" / "protocol_v2.yaml")
STAGE = RUNS / "stage_main_v2.jsonl"

OUTAGE_CLASS = {"F1": "5 min", "F2": "30 min", "F3": "2 h"}
OUTAGE_START_S = {"F1": 66600.0, "F2": 66600.0, "F3": 66600.0}
OUTAGE_DUR_S = {"F1": 300.0, "F2": 1800.0, "F3": 7200.0}


def arm_of(pid: str) -> str:
    return pid.split("_")[0]


def outage_overlap_with_dr() -> dict:
    """How much of each episode's DR window is covered by the F3 outage."""
    eps = load_episodes()
    dr_frac = float(PROTO.raw["target"]["dr_trigger_fraction"])
    rows = []
    for eid, ep in eps.items():
        shape = np.asarray(ep.shape, dtype=float)
        dr = shape >= dr_frac
        n_dr = int(dr.sum())
        lo = int(OUTAGE_START_S["F3"] // 60)
        hi = int((OUTAGE_START_S["F3"] + OUTAGE_DUR_S["F3"]) // 60)
        covered = int(dr[lo:hi].sum())
        rows.append({"episode_id": eid, "n_dr_steps": n_dr,
                     "n_dr_steps_inside_F3": covered,
                     "dr_fully_covered": bool(n_dr > 0 and covered == n_dr),
                     "dr_overlap_fraction": (covered / n_dr) if n_dr else float("nan")})
    # The S6 held-out evaluation is on the TEST split, so report the split-wise
    # DR-step overlap as well as the whole-workload mean: the pooled number over
    # 100 episodes is not the number that the S6 claim rests on.
    split_path = ROOT / "runs" / "workload_split.json"
    by_split: dict = {}
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        for name, ids in split.items():
            if not isinstance(ids, list):
                continue
            tot = sum(r["n_dr_steps"] for r in rows if r["episode_id"] in set(ids))
            ins = sum(r["n_dr_steps_inside_F3"] for r in rows if r["episode_id"] in set(ids))
            by_split[name] = {"n_episodes": len(ids), "n_dr_steps": int(tot),
                              "n_dr_steps_inside_F3": int(ins),
                              "dr_step_overlap_fraction": (ins / tot) if tot else float("nan")}
    return {"dr_trigger_fraction": dr_frac,
            "f3_window_s": [OUTAGE_START_S["F3"],
                            OUTAGE_START_S["F3"] + OUTAGE_DUR_S["F3"]],
            "n_episodes": len(rows),
            "n_fully_covered": int(sum(r["dr_fully_covered"] for r in rows)),
            "mean_overlap_fraction": float(np.nanmean(
                [r["dr_overlap_fraction"] for r in rows])),
            "by_split": by_split,
            "per_episode": rows}


def outage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("policy_id"):
        oel = g["outage_eligible_steps"].sum()
        row = {
            "policy_id": pid, "arm": arm_of(pid), "n_runs": int(len(g)),
            "outage_eligible_steps": int(oel),
            "outage_achievement_rate": (float(g["outage_achieved_steps"].sum() / oel)
                                        if oel else float("nan")),
            "outage_energy_ratio": (float(g["outage_actual_energy_kwh"].sum()
                                          / g["outage_baseline_energy_kwh"].sum())
                                    if g["outage_baseline_energy_kwh"].sum() else float("nan")),
            "outage_switches": float(g["outage_switches"].sum()) if "outage_switches" in g else float("nan"),
            "outage_window_s_mean": float(pd.to_numeric(
                g["outage_window_s"], errors="coerce").dropna().mean())
            if "outage_window_s" in g else float("nan"),
            "heartbeat_detect_delay_s": float(pd.to_numeric(
                g["heartbeat_detect_delay_s"], errors="coerce").dropna().mean())
            if "heartbeat_detect_delay_s" in g else float("nan"),
            "authority_transfer_delay_s": float(pd.to_numeric(
                g["authority_transfer_delay_s"], errors="coerce").dropna().mean())
            if "authority_transfer_delay_s" in g else float("nan"),
            "recovery_delay_s": float(pd.to_numeric(
                g["recovery_delay_s"], errors="coerce").dropna().mean())
            if "recovery_delay_s" in g else float("nan"),
        }
        for fid, cls in OUTAGE_CLASS.items():
            sub = g[g["fault_id"].astype(str) == fid]
            o = sub["outage_eligible_steps"].sum()
            row[f"outage_ach_{fid}({cls})"] = (float(sub["outage_achieved_steps"].sum() / o)
                                               if o else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("outage_achievement_rate", ascending=False)


def flapping_trace() -> dict:
    """Raw authority timeline for F8, straight from the exported event stream."""
    out = {}
    if not TRACES.exists():
        return {"note": f"{TRACES} does not exist"}
    for path in sorted(TRACES.glob("F8__*.jsonl")):
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # authority transitions: a domain's authority changes from the previous value
        last: dict = {}
        transitions = []
        rejections: dict = {}
        for e in events:
            d = e.get("domain_id")
            if d is None:
                continue
            a = e.get("authority")
            if d not in last:
                last[d] = a
                continue
            if a != last[d]:
                transitions.append({"t_s": float(e["t"]), "domain_id": d,
                                    "from": last[d], "to": a,
                                    "epoch": int(e.get("epoch", -1)),
                                    "sequence": int(e.get("sequence", -1))})
                last[d] = a
            if not e.get("accepted", True):
                key = str(e.get("detail", "unknown"))
                rejections[key] = rejections.get(key, 0) + 1
        # Atomicity: at any instant, each DOMAIN must have exactly ONE authority.
        # The key is therefore (t, domain), not t alone -- different domains are
        # allowed (and expected) to hold different authorities during a takeover,
        # so keying on t alone flags every multi-domain timestamp as a violation.
        # That was the bug in the first version of this counter: it reported
        # 1435/1435 timestamps "carrying two authorities" purely because five
        # domains were each listed once under the same instant.
        per_td: dict = {}
        for e in events:
            d = e.get("domain_id")
            if d is None:
                continue
            per_td.setdefault((float(e["t"]), d), set()).add(e.get("authority"))
        dup = sum(1 for v in per_td.values() if len(v) > 1)
        takeovers = [x for x in transitions if x["to"] == "local"]
        recoveries = [x for x in transitions if x["to"] == "cloud"]
        out[path.name] = {
            "n_events": len(events),
            "n_authority_transitions": len(transitions),
            "n_takeovers_cloud_to_local": len(takeovers),
            "n_recoveries_local_to_cloud": len(recoveries),
            "epochs_seen": sorted({int(e.get("epoch", -1)) for e in events}),
            "rejections_by_reason": rejections,
            "timestamps_with_two_authorities": int(dup),
            "first_20_transitions": transitions[:20],
            "first_takeover_s": takeovers[0]["t_s"] if takeovers else None,
            "first_recovery_s": recoveries[0]["t_s"] if recoveries else None,
        }
    return out


def main() -> int:
    print("=" * 78)
    print("E-C.1  THE F3 OUTAGE vs THE EVENING DR WINDOW (measured on the workload)")
    print("=" * 78)
    ov = outage_overlap_with_dr()
    print(f"  DR trigger: shape >= {ov['dr_trigger_fraction']}")
    print(f"  F3 window:  {ov['f3_window_s'][0]:.0f}-{ov['f3_window_s'][1]:.0f} s "
          f"({ov['f3_window_s'][0] / 3600:.2f}h-{ov['f3_window_s'][1] / 3600:.2f}h)")
    print(f"  episodes:   {ov['n_episodes']}")
    print(f"  DR window FULLY inside the F3 outage: {ov['n_fully_covered']}/{ov['n_episodes']}")
    print(f"  mean fraction of the DR window covered: {ov['mean_overlap_fraction']:.4f}")
    if ov.get("by_split"):
        print("  by split (DR steps inside F3 / total DR steps):")
        for name, s in ov["by_split"].items():
            print(f"    {name:<6} n={s['n_episodes']:>3}  "
                  f"{s['n_dr_steps_inside_F3']:>5}/{s['n_dr_steps']:>5} "
                  f"= {s['dr_step_overlap_fraction']:.4f}")
    worst = sorted(ov["per_episode"], key=lambda r: r["dr_overlap_fraction"])[:5]
    print("  least covered episodes:")
    for r in worst:
        print(f"    {r['episode_id']}  DR steps {r['n_dr_steps']:>4}  inside F3 "
              f"{r['n_dr_steps_inside_F3']:>4}  frac {r['dr_overlap_fraction']:.3f}")

    print()
    print("=" * 78)
    print("E-C.2  OUTAGE-WINDOW METRICS PER ARM (outage + the following 30 min)")
    print("=" * 78)
    df = pd.read_json(STAGE, lines=True)
    df = df[df["ok"].astype(bool)].copy()
    tab = outage_table(df)
    tab.to_csv(RUNS / "s7_ec_outage_table.csv", index=False)
    cols = ["policy_id", "outage_achievement_rate", "outage_energy_ratio",
            "outage_switches", "outage_window_s_mean", "heartbeat_detect_delay_s",
            "authority_transfer_delay_s", "recovery_delay_s"]
    print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("  by outage duration class (outage_achievement_rate)")
    cls_cols = ["policy_id"] + [c for c in tab.columns if c.startswith("outage_ach_")]
    print(tab[cls_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("=" * 78)
    print("E-C.3  RAW F8 FLAPPING TIMELINE (authority is atomic; epochs fence)")
    print("=" * 78)
    ft = flapping_trace()
    for name, info in ft.items():
        if "note" in info:
            print(f"  {info['note']}")
            continue
        print(f"  {name}")
        print(f"    events {info['n_events']}  transitions {info['n_authority_transitions']} "
              f"({info['n_takeovers_cloud_to_local']} takeovers, "
              f"{info['n_recoveries_local_to_cloud']} recoveries)")
        print(f"    epochs seen {info['epochs_seen']}")
        print(f"    (t, domain) instants with more than one authority "
              f"[atomicity violations]: {info['timestamps_with_two_authorities']}")
        print(f"    rejections {info['rejections_by_reason']}")
        print(f"    first takeover {info['first_takeover_s']} s, "
              f"first recovery {info['first_recovery_s']} s")

    payload = {"dr_overlap": ov, "outage_table": tab.to_dict(orient="records"),
               "flapping_trace": ft}
    (RUNS / "s7_ec.json").write_text(json.dumps(payload, indent=2, default=float),
                                     encoding="utf-8")
    print("\nwrote runs_v2/s7_ec.json and runs_v2/s7_ec_outage_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
