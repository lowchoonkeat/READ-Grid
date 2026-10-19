"""Pilot test A: feasi  bility of demand-response visibility on IEEE 123.

Purpose
-------
The Gemini audit report claims that because actions curtail only 40 %/70 % of a
20 % flexible slice (7.4 %-14 % of group demand), a feeder-wide curtailment can
never exceed ~8.5 % of total load when only 50 of 91 load elements are mapped,
and that this keeps bus voltages inside limits so that *no* DR policy can
demonstrate physical effect.

This script tests that claim with the actual solver rather than accepting it.

Method
------
1. Compile the verified IEEE 123 feeder.
2. Apply a uniform curtailment fraction to *all* 91 loads (constant power factor)
   at several levels: 0, 2, 5, 10, 14, 20, 30, 50, 75, 100 %.
3. Re-solve and record min/max per-unit bus voltage, line loading (max I/NormAmps)
   and total losses.
4. Repeat under Fixed-Tap (ControlMode=STATIC, taps frozen at the solved state)
   and Active-Regulator (ControlMode=EVENT) conditions.
5. Repeat at workload scales 0.8 / 1.0 / 1.2.

Output: data/processed/pilot_curtailment_sensitivity.csv plus a console summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dss_engine import dss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
WORK = ROOT / "data" / "processed" / "ieee123_work"
OUT = ROOT / "data" / "processed"


def iter_coll(coll):
    return iter(coll)


def snapshot_loads(c):
    return [(ld.Name, float(ld.kW), float(ld.kvar)) for ld in iter_coll(c.Loads)]


def restore_loads(c, base):
    for name, kw, kvar in base:
        dss.Text.Command = f"Load.{name}.kW={kw:.9g}"
        dss.Text.Command = f"Load.{name}.kvar={kvar:.9g}"


def curtail_all(c, base, fraction):
    """Curtail `fraction` of every load at constant power factor."""
    for name, kw, kvar in base:
        f = 1.0 - fraction
        dss.Text.Command = f"Load.{name}.kW={kw * f:.9g}"
        dss.Text.Command = f"Load.{name}.kvar={kvar * f:.9g}"


def solve_and_measure(c, control_mode):
    dss.Text.Command = "Set mode=snapshot"
    dss.Text.Command = "Set MaxControlIter=300"
    dss.Text.Command = f"Set ControlMode={control_mode}"
    c.Solution.Solve()
    conv = bool(c.Solution.Converged)
    if not conv:
        return {"converged": False}

    vpu = np.asarray(c.AllBusVmagPu, dtype=float)
    nodes = list(c.AllNodeNames)
    pairs = [(n, float(v)) for n, v in zip(nodes, vpu) if 0.05 < v < 2.0]
    vmin = min(pairs, key=lambda t: t[1]) if pairs else (None, np.nan)
    vmax = max(pairs, key=lambda t: t[1]) if pairs else (None, np.nan)

    # line loading: max |I| / NormAmps over all lines and phases
    worst = 0.0
    worst_line = None
    n_over_norm = 0
    n_over_emerg = 0
    for ln in iter_coll(c.Lines):
        norm = float(ln.NormAmps)
        emerg = float(ln.EmergAmps)
        if norm <= 0:
            continue
        cur = np.asarray(c.ActiveCktElement.CurrentsMagAng, dtype=float)
        imax = float(np.max(cur)) if cur.size else 0.0
        ratio = imax / norm
        if ratio > worst:
            worst, worst_line = ratio, ln.Name
        if ratio > 1.0:
            n_over_norm += 1
        if emerg > 0 and imax > emerg:
            n_over_emerg += 1

    tp = [float(v) for v in c.TotalPower]
    ls = [float(v) for v in c.Losses]
    return {
        "converged": True,
        "vmin_pu": vmin[1], "vmin_node": vmin[0],
        "vmax_pu": vmax[1], "vmax_node": vmax[0],
        "p_loss_kW": ls[0] / 1000.0,
        "q_loss_kvar": ls[1] / 1000.0,
        "substation_kW": tp[0], "substation_kvar": tp[1],
        "max_line_loading_pu": worst, "worst_line": worst_line,
        "n_lines_over_normamps": n_over_norm,
        "n_lines_over_emergamps": n_over_emerg,
    }


def main():
    dss.Text.Command = "Clear"
    dss.Text.Command = "Set DefaultBaseFreq=60"
    dss.Text.Command = f'Compile [{WORK / "IEEE123Master.dss"}]'
    dss.Text.Command = "CalcVoltageBases"
    c = dss.ActiveCircuit
    base = snapshot_loads(c)
    print(f"compiled; {len(base)} loads, base total {sum(k for _, k, _ in base):.1f} kW")

    fractions = [0.0, 0.02, 0.05, 0.10, 0.14, 0.20, 0.30, 0.50, 0.75, 1.00]
    rows = []

    # fix the tap state once, then measure with regulators frozen
    dss.Text.Command = "Set mode=snapshot"
    dss.Text.Command = "Set MaxControlIter=300"
    dss.Text.Command = "Set ControlMode=STATIC"
    c.Solution.Solve()
    # Taps live on the regulator transformer; record both the tap position and
    # the per-unit tap ratio so the fixed-tap regime can be restored exactly.
    frozen_taps = {}
    for t in iter_coll(c.Transformers):
        frozen_taps[t.Name] = float(t.Tap)
    print("frozen transformer tap ratios:", frozen_taps)

    for scale in [0.8, 1.0, 1.2]:
        # apply workload scale to the base case
        for name, kw, kvar in base:
            dss.Text.Command = f"Load.{name}.kW={kw * scale:.9g}"
            dss.Text.Command = f"Load.{name}.kvar={kvar * scale:.9g}"
        scaled = snapshot_loads(c)

        for frac in fractions:
            for mode, label in [("STATIC", "fixed_tap"), ("EVENT", "active_reg")]:
                if mode == "STATIC":
                    # pin transformer taps back to the frozen solution
                    # RegControl.TapNumber is not exposed by this build; freezing
                    # the transformer tap plus keeping ControlMode=STATIC is what
                    # holds the regulator position fixed.
                    for tname, tap in frozen_taps.items():
                        dss.Text.Command = f"Transformer.{tname}.Tap={tap:.9g}"
                curtail_all(c, scaled, frac)
                m = solve_and_measure(c, mode)
                rows.append({"workload_scale": scale, "curtail_fraction": frac,
                             "regulator_regime": label, **m})
                if mode == "STATIC":
                    print(f"  scale={scale} curtail={frac:>5.0%} {label:>10}: "
                          f"vmin={m['vmin_pu']:.5f} vmax={m['vmax_pu']:.5f} "
                          f"line={m['max_line_loading_pu']:.3f} "
                          f"P_loss={m['p_loss_kW']:.1f} kW")
            restore_loads(c, scaled)
        restore_loads(c, base)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pilot_curtailment_sensitivity.csv", index=False)
    print("\nwrote", OUT / "pilot_curtailment_sensitivity.csv")

    print("\n=== Q1: does a 14% uniform curtailment move voltage measurably? ===")
    sub = df[(df.regulator_regime == "fixed_tap")]
    for scale in [0.8, 1.0, 1.2]:
        s0 = sub[(sub.workload_scale == scale) & (sub.curtail_fraction == 0.0)]
        s14 = sub[(sub.workload_scale == scale) & (sub.curtail_fraction == 0.14)]
        if len(s0) and len(s14):
            dv = float(s14.vmin_pu.iloc[0]) - float(s0.vmin_pu.iloc[0])
            print(f"  scale={scale}: vmin {float(s0.vmin_pu.iloc[0]):.5f} -> "
                  f"{float(s14.vmin_pu.iloc[0]):.5f}  (delta = {dv:+.5f} p.u.)")

    print("\n=== Q2: at which curtailment does vmin cross 0.95 p.u.? ===")
    for scale in [0.8, 1.0, 1.2]:
        s = sub[sub.workload_scale == scale].sort_values("curtail_fraction")
        below = s[s.vmin_pu < 0.95]
        print(f"  scale={scale}: min vmin over all curtail levels = "
              f"{s.vmin_pu.min():.5f}; levels below 0.95 = {len(below)}")

    print("\n=== Q3: worst voltage at the LOWEST curtailment (i.e. baseline) ===")
    print(sub[sub.curtail_fraction == 0.0][["workload_scale", "vmin_pu", "vmin_node",
                                            "vmax_pu", "max_line_loading_pu"]]
          .to_string(index=False))

    print("\n=== Q4: does curtailment ever RAISE vmin above the upper limit? ===")
    print(f"  global vmax over all runs = {df.vmax_pu.max():.5f} p.u.")


if __name__ == "__main__":
    main()
