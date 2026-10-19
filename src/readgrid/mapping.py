"""Logical metering/control group mapping.

Maps **all 91** IEEE 123-bus spot loads onto **50 logical groups** and those
groups onto **5 edge domains**, preserving each load's original phase connection
and aggregating the baseline P and Q exactly.

The mapping is *virtual*: the 50 groups are logical control partitions, not 50
real residential households. The task prompt forbids describing them as real
households and this module does not.

Mapping rule (deterministic, documented): loads are sorted by
(numeric bus number, phase) and cut into contiguous blocks so that groups stay
spatially local along the bus numbering. Because 91 = 50 + 41, the first 41
groups receive two loads and the remaining nine receive one.

The conservation check (sum of group P == sum of load P, likewise Q) is asserted
at build time and recorded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # project root, derived from this file (portable)
VERIFY_JSON = ROOT / "data" / "processed" / "feeder_verification.json"
OUT_DIR = ROOT / "data" / "processed"


@dataclass(frozen=True)
class Group:
    group_id: str
    domain_id: str
    load_names: tuple[str, ...]
    buses: tuple[str, ...]
    phases: tuple[int, ...]
    base_kw: float
    base_kvar: float

    @property
    def power_factor(self) -> float:
        s = (self.base_kw ** 2 + self.base_kvar ** 2) ** 0.5
        return self.base_kw / s if s > 0 else 1.0


def _bus_sort_key(bus: str) -> tuple[int, int]:
    """Sort a bus label like ``65.1`` by (bus number, phase)."""
    m = re.match(r"^(\d+)(?:r)?(?:\.(\d+))?", str(bus))
    if not m:
        return (10 ** 6, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


def build_mapping(verify_json: Path | str = VERIFY_JSON,
                  n_groups: int = 50,
                  n_domains: int = 5) -> list[Group]:
    data = json.loads(Path(verify_json).read_text(encoding="utf-8"))
    rows = data["loads"]["rows"]

    ordered = sorted(rows, key=lambda r: (_bus_sort_key(r["bus"]), r["name"]))

    # contiguous cut: 91 = 50 + 41 -> first 41 groups get 2 loads
    n_loads = len(ordered)
    base = n_loads // n_groups          # 1
    extra = n_loads % n_groups          # 41
    sizes = [base + 1] * extra + [base] * (n_groups - extra)

    groups: list[Group] = []
    cursor = 0
    for gi, size in enumerate(sizes):
        chunk = ordered[cursor:cursor + size]
        cursor += size
        gid = f"G{gi + 1:02d}"
        did = f"E{gi // (n_groups // n_domains) + 1}"
        groups.append(Group(
            group_id=gid,
            domain_id=did,
            load_names=tuple(r["name"] for r in chunk),
            buses=tuple(r["bus"] for r in chunk),
            phases=tuple(int(r["phases"]) for r in chunk),
            base_kw=round(sum(float(r["kw"]) for r in chunk), 6),
            base_kvar=round(sum(float(r["kvar"]) for r in chunk), 6),
        ))

    assert cursor == n_loads, "mapping did not consume every load"

    # ---- conservation check -------------------------------------------------
    sum_group_kw = sum(g.base_kw for g in groups)
    sum_group_kvar = sum(g.base_kvar for g in groups)
    sum_load_kw = sum(float(r["kw"]) for r in rows)
    sum_load_kvar = sum(float(r["kvar"]) for r in rows)
    assert abs(sum_group_kw - sum_load_kw) < 1e-6, (sum_group_kw, sum_load_kw)
    assert abs(sum_group_kvar - sum_load_kvar) < 1e-6, (sum_group_kvar, sum_load_kvar)
    assert len({g.group_id for g in groups}) == n_groups
    assert len({g.domain_id for g in groups}) == n_domains
    # every load appears exactly once
    flat = [n for g in groups for n in g.load_names]
    assert len(flat) == len(set(flat)) == n_loads
    return groups


def mapping_frames(groups: list[Group]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (load-level mapping, group-level summary)."""
    load_rows = []
    for g in groups:
        for name, bus, ph in zip(g.load_names, g.buses, g.phases):
            load_rows.append({
                "group_id": g.group_id,
                "domain_id": g.domain_id,
                "load_name": name,
                "bus": bus,
                "phase_node": bus.split(".")[-1] if "." in bus else "1",
                "phases": ph,
            })
    load_df = pd.DataFrame(load_rows)

    grp_rows = []
    for g in groups:
        grp_rows.append({
            "group_id": g.group_id,
            "domain_id": g.domain_id,
            "n_loads": len(g.load_names),
            "load_names": "|".join(g.load_names),
            "buses": "|".join(g.buses),
            "base_kw": g.base_kw,
            "base_kvar": g.base_kvar,
            "power_factor": round(g.power_factor, 6),
            "flexible_kw": round(g.base_kw * 0.20, 6),
        })
    grp_df = pd.DataFrame(grp_rows)
    return load_df, grp_df


def write_mapping(out_dir: Path | str = OUT_DIR, **kw) -> list[Group]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = build_mapping(**kw)
    load_df, grp_df = mapping_frames(groups)
    load_df.to_csv(out / "meter_to_load_mapping.csv", index=False)
    grp_df.to_csv(out / "group_summary.csv", index=False)

    summary = {
        "n_groups": len(groups),
        "n_domains": len({g.domain_id for g in groups}),
        "n_loads_mapped": int(len(load_df)),
        "total_kw": round(sum(g.base_kw for g in groups), 6),
        "total_kvar": round(sum(g.base_kvar for g in groups), 6),
        "mapping_rule": "bus_sorted_contiguous",
        "note": ("Logical control partitions, not real residential households. "
                 "Original phase connections preserved; group sums equal the "
                 "original load totals exactly."),
        "groups_per_domain": {
            d: sum(1 for g in groups if g.domain_id == d)
            for d in sorted({g.domain_id for g in groups})
        },
    }
    (out / "mapping_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return groups


if __name__ == "__main__":
    gs = write_mapping()
    print(f"groups={len(gs)} loads={sum(len(g.load_names) for g in gs)} "
          f"kW={sum(g.base_kw for g in gs):.2f} kvar={sum(g.base_kvar for g in gs):.2f}")
