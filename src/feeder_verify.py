"""Phase 2 feeder verification for the IEEE 123-bus case.

Verifies: compilation, convergence, phase connections, voltage bases, load
totals, regulator/capacitor settings, and line/transformer ratings.

The raw feeder files are never modified. A normalised working copy is written
to ``data/processed/ieee123_work/`` and only that copy is patched.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dss_engine import dss, ENGINE_PACKAGE, engine_version  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
RAW = ROOT / "data" / "raw" / "ieee123"
LINE_CODES = ROOT / "data" / "raw" / "IEEELineCodes.DSS"
OUT = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_workdir() -> Path:
    """Copy the feeder and normalise the LineCodes indirection.

    The shipped ``IEEELineCodes.DSS`` inside the 123Bus folder is a one-line
    stub (``redirect ../IEEELineCodes.DSS``). We resolve it locally so no path
    escapes the project folder and behaviour is platform-independent.
    """
    work = OUT / "ieee123_work"
    work.mkdir(parents=True, exist_ok=True)
    for p in RAW.iterdir():
        shutil.copy2(p, work / p.name)
    shutil.copy2(LINE_CODES, work / "IEEELineCodes_shared.DSS")
    (work / "IEEELineCodes.DSS").write_text(
        "! normalised by src/feeder_verify.py\nredirect IEEELineCodes_shared.DSS\n",
        encoding="utf-8",
    )
    return work


def iter_coll(coll):
    """Iterate a DSS collection.

    ``dss-python`` collections support the iterator protocol directly. Using
    ``coll.First`` / ``coll.Next`` as statements does not advance the cursor in
    this binding, so the iterator protocol is used instead.
    """
    return iter(coll)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    rep["engine"] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "engine_package": ENGINE_PACKAGE,
        "engine_version": engine_version(),
    }
    print("engine:", ENGINE_PACKAGE)
    print(engine_version()[:150])

    rep["files"] = {}
    for p in sorted(list(RAW.iterdir()) + [LINE_CODES]):
        rep["files"][str(p.relative_to(ROOT)).replace("\\", "/")] = sha256(p)

    work = prepare_workdir()
    rep["workdir"] = str(work).replace("\\", "/")

    c = dss.ActiveCircuit

    # --- compile ---
    dss.Text.Command = "Clear"
    dss.Text.Command = "Set DefaultBaseFreq=60"
    dss.Text.Command = f'Compile [{work / "IEEE123Master.dss"}]'
    rep["compile"] = {
        "error": dss.Text.Result,
        "num_ckt_elements": int(c.NumCktElements),
        "num_buses": int(c.NumBuses),
        "num_nodes": int(c.NumNodes),
    }
    print("compile error:", repr(dss.Text.Result))
    print("elements:", c.NumCktElements, "buses:", c.NumBuses, "nodes:", c.NumNodes)

    # --- voltage bases ---
    dss.Text.Command = "CalcVoltageBases"
    vb = {}
    for b in c.AllBusNames:
        c.SetActiveBus(b)
        vb[b] = round(float(c.Buses.kVBase or 0.0), 4)
    rep["voltage_bases"] = {"distinct": sorted(set(vb.values()), reverse=True),
                            "by_bus": vb}
    print("voltage bases (distinct):", sorted(set(vb.values()), reverse=True))

    # --- solve ---
    dss.Text.Command = "Set mode=snapshot"
    dss.Text.Command = "Set MaxControlIter=100"
    rep["solution"] = {}

    dss.Text.Command = "Set ControlMode=STATIC"
    c.Solution.Solve()
    rep["solution"]["static_control_converged"] = bool(c.Solution.Converged)
    print("STATIC ctrl converged:", c.Solution.Converged)

    dss.Text.Command = "Set ControlMode=Event"
    c.Solution.Solve()
    rep["solution"].update({
        "event_control_converged": bool(c.Solution.Converged),
        "event_control_iterations": int(c.Solution.ControlIterations),
        "solver_iterations": int(c.Solution.Iterations),
        "total_power": [float(v) for v in c.TotalPower],
        "losses": [float(v) for v in c.Losses],
    })
    print("EVENT ctrl converged:", c.Solution.Converged,
          "ctrl_iters:", c.Solution.ControlIterations,
          "solver_iters:", c.Solution.Iterations)
    print("TotalPower kW/kvar:", [round(float(v), 3) for v in c.TotalPower])
    print("Losses W/var:", [round(float(v), 1) for v in c.Losses])

    # --- line codes: what ratings are in effect? ---
    lc_rows = [{"name": lc.Name, "nphases": int(lc.Phases),
                "normamps": float(lc.NormAmps), "emergamps": float(lc.EmergAmps)}
               for lc in iter_coll(c.LineCodes)]
    rep["linecodes"] = lc_rows
    print("linecodes:", [(r["name"], r["normamps"], r["emergamps"]) for r in lc_rows])

    # --- loads ---
    load_rows = []
    for ld in iter_coll(c.Loads):
        load_rows.append({
            "name": ld.Name,
            "bus": c.ActiveCktElement.BusNames[0],
            "phases": int(c.ActiveCktElement.NumPhases),
            "kw": float(ld.kW), "kvar": float(ld.kvar), "pf": float(ld.PF),
            "model": int(ld.Model), "kv": float(ld.kV), "class": int(ld.Class),
            "isdelta": bool(ld.IsDelta),
            "kva": float(ld.kva),
            "conn": "delta" if bool(ld.IsDelta) else "wye",
        })
    tot_kw = sum(r["kw"] for r in load_rows)
    tot_kvar = sum(r["kvar"] for r in load_rows)
    rep["loads"] = {
        "count": len(load_rows), "total_kW": tot_kw, "total_kvar": tot_kvar,
        "by_phases": {p: sum(1 for r in load_rows if r["phases"] == p)
                      for p in sorted({r["phases"] for r in load_rows})},
        "by_model": {m: sum(1 for r in load_rows if r["model"] == m)
                     for m in sorted({r["model"] for r in load_rows})},
        "by_conn": {k: sum(1 for r in load_rows if r["conn"] == k)
                    for k in sorted({r["conn"] for r in load_rows})},
        "distinct_buses": len({r["bus"] for r in load_rows}),
        "rows": load_rows,
    }
    print(f"loads={len(load_rows)} total_kW={tot_kw:.2f} total_kvar={tot_kvar:.2f}")
    print("  by phases:", rep["loads"]["by_phases"])
    print("  by model :", rep["loads"]["by_model"])
    print("  by conn  :", rep["loads"]["by_conn"])
    print("  distinct loaded buses:", rep["loads"]["distinct_buses"])

    # --- lines ---
    line_rows = [{"name": ln.Name, "linecode": ln.LineCode,
                  "length": float(ln.Length), "units": str(ln.Units),
                  "phases": int(ln.Phases), "normamps": float(ln.NormAmps),
                  "emergamps": float(ln.EmergAmps)}
                 for ln in iter_coll(c.Lines)]
    used: dict[str, int] = {}
    for r in line_rows:
        used[r["linecode"]] = used.get(r["linecode"], 0) + 1
    rep["lines"] = {
        "count": len(line_rows),
        "distinct_normamps": sorted({r["normamps"] for r in line_rows}),
        "distinct_emergamps": sorted({r["emergamps"] for r in line_rows}),
        "used_linecodes": dict(sorted(used.items())),
        "rows": line_rows,
    }
    print("lines:", len(line_rows))
    print("  distinct NormAmps :", rep["lines"]["distinct_normamps"])
    print("  distinct EmergAmps:", rep["lines"]["distinct_emergamps"])
    print("  used linecodes    :", rep["lines"]["used_linecodes"])

    # --- transformers ---
    xf = []
    for t in iter_coll(c.Transformers):
        xf.append({
            "name": t.Name,
            "kva": float(t.kva),
            "numwindings": int(t.NumWindings),
            "buses": list(c.ActiveCktElement.BusNames),
            "xhl": float(t.Xhl),
            "xht": float(t.Xht),
            "xlt": float(t.Xlt),
            "kv_high": float(t.kV),
            "kva_rated": float(t.kva),
            "max_tap": float(t.MaxTap),
            "min_tap": float(t.MinTap),
            "num_taps": int(t.NumTaps),
            "tap": float(t.Tap),
            "hs_rating_A": float(c.ActiveCktElement.NormalAmps),
            "hs_emerg_A": float(c.ActiveCktElement.EmergAmps),
        })
    rep["transformers"] = xf
    print("transformers:", [(t["name"], t["kva"]) for t in xf])

    # --- regulators ---
    rc = [{"name": r.Name, "transformer": r.Transformer,
           "winding": int(r.Winding),
           "forward_vreg": float(r.ForwardVreg),
           "forward_band": float(r.ForwardBand),
           "reverse_vreg": float(r.ReverseVreg),
           "reverse_band": float(r.ReverseBand),
           "delay": float(r.Delay), "tap_delay": float(r.TapDelay),
           "maxtapchange": int(r.MaxTapChange),
           "tapnum": int(r.TapNumber),
           "voltage_limit": float(r.VoltageLimit),
           "monitored_bus": r.MonitoredBus,
           "ptr": float(r.PTratio), "ctp": float(r.CTPrimary)}
          for r in iter_coll(c.RegControls)]
    rep["regulators"] = rc
    print("regulators:", [(r["name"], r["forward_vreg"], r["delay"],
                           r["maxtapchange"]) for r in rc])

    # --- capacitors ---
    caps = [{"name": x.Name, "kvar": float(x.kvar), "numsteps": int(x.NumSteps),
             "states": [int(s) for s in x.States],
             "buses": list(c.ActiveCktElement.BusNames)}
            for x in iter_coll(c.Capacitors)]
    rep["capacitors"] = caps
    print("capacitors:", [(x["name"], x["kvar"], x["states"]) for x in caps])

    # --- voltage extrema at nominal load ---
    vpu = np.asarray(c.AllBusVmagPu, dtype=float)
    names = list(c.AllNodeNames)
    pairs = [(n, float(v)) for n, v in zip(names, vpu) if 0.05 < v < 2.0]
    if pairs:
        vmin = min(pairs, key=lambda t: t[1])
        vmax = max(pairs, key=lambda t: t[1])
        rep["solution"].update({
            "vmin_pu": vmin[1], "vmin_node": vmin[0],
            "vmax_pu": vmax[1], "vmax_node": vmax[0],
            "vmin_by_node": dict(pairs),
        })
        print(f"voltage {vmin[1]:.5f} p.u. @ {vmin[0]} .. {vmax[1]:.5f} p.u. @ {vmax[0]}")

    with open(OUT / "feeder_verification.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("wrote", OUT / "feeder_verification.json")


if __name__ == "__main__":
    main()
