"""OpenDSS feeder wrapper for the IEEE 123-bus case.

Responsibilities
----------------
* compile the *unmodified* feeder from a project-local working copy;
* apply the version-controlled rating patch (see ``configs/ieee123_ratings_patch.dss``)
  and record which lines remain UNRATED;
* expose a single ``step`` operation: set every group's P and Q, solve, measure.

The wrapper never modifies the raw input files. It never treats a software
default rating as a verified equipment rating: after the patch is applied the
per-line rating provenance is recorded explicitly.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dss_engine import dss, ENGINE_PACKAGE, engine_version  # noqa: E402

from .mapping import Group  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # project root, derived from this file (portable)

# Kersting 50 degC ampacities used by the rating patch (provenance: referenced).
LINE_CODE_RATING_A = {
    "1": 530.0, "2": 530.0, "3": 530.0, "4": 530.0,
    "5": 530.0, "6": 530.0, "7": 530.0, "8": 530.0,
    "9": 230.0, "10": 230.0, "11": 230.0,
    "12": 175.0,
}
RATING_PROVENANCE = {
    "1": ("530", "source_defined"), "2": ("530", "source_defined"),
    "3": ("530", "source_defined"), "4": ("530", "source_defined"),
    "5": ("530", "source_defined"), "6": ("530", "source_defined"),
    "7": ("530", "source_defined"), "8": ("530", "source_defined"),
    "9": ("230", "source_defined"), "10": ("230", "source_defined"),
    "11": ("230", "source_defined"), "12": ("175", "source_defined"),
}


@dataclass
class SolveResult:
    converged: bool
    solver_iterations: int
    control_iterations: int
    vmin_pu: float
    vmin_node: str
    vmax_pu: float
    vmax_node: str
    max_line_loading_pu: float
    worst_line: str
    n_lines_over_norm: int
    n_lines_over_emerg: int
    losses_kw: float
    losses_kvar: float
    substation_kw: float
    substation_kvar: float
    domain_vmin_pu: dict[str, float] = field(default_factory=dict)


class FeederModel:
    """Thin, fast wrapper around one compiled IEEE 123-bus circuit."""

    def __init__(self, groups: list[Group], protocol, *, verbose: bool = False):
        self.protocol = protocol
        self.groups = groups
        self.verbose = verbose

        work = ROOT / protocol.raw["feeder"]["workdir"]
        master = work / protocol.raw["feeder"]["master"]
        patch = ROOT / protocol.raw["feeder"]["rating_patch"]

        self.master_path = master
        self.patch_path = patch
        self.master_sha256 = _sha256(master)
        self.patch_sha256 = _sha256(patch)

        self.c = dss.ActiveCircuit

        dss.Text.Command = "Clear"
        dss.Text.Command = "Set DefaultBaseFreq=60"
        dss.Text.Command = f"Compile [{master}]"
        if dss.Text.Result.strip():
            raise RuntimeError(f"feeder compile error: {dss.Text.Result!r}")

        # ---- rating patch -----------------------------------------------------
        # 1) patch the line-code definitions
        for line in patch.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            dss.Text.Command = s
        # 2) OpenDSS does NOT propagate a LineCode rating onto Line objects that
        #    were defined without their own NormAmps: every Line keeps the 400 A
        #    software default. Each conductor-bearing Line is therefore rated
        #    explicitly from its line code. Lines with no line code (the eight
        #    zero-length switch segments) stay UNRATED and are excluded from the
        #    loading metrics.
        for name in list(self.c.Lines.AllNames):
            self.c.Lines.Name = name
            code = str(self.c.Lines.LineCode)
            if code in LINE_CODE_RATING_A:
                r = LINE_CODE_RATING_A[code]
                dss.Text.Command = f"Line.{name}.NormAmps={r:g}"
                dss.Text.Command = f"Line.{name}.EmergAmps={r:g}"
        dss.Text.Command = "CalcVoltageBases"
        dss.Text.Command = "Set mode=snapshot"
        dss.Text.Command = f"Set MaxControlIter={int(protocol.raw['feeder']['max_control_iter'])}"
        dss.Text.Command = f"Set ControlMode={protocol.raw['feeder']['control_mode']}"

        # ---- static structure -------------------------------------------------
        self.load_names: list[str] = list(self.c.Loads.AllNames)
        self.line_names: list[str] = list(self.c.Lines.AllNames)
        self.node_names: list[str] = list(self.c.AllNodeNames)

        # base (unmodified) per-load P/Q
        self.base_kw: dict[str, float] = {}
        self.base_kvar: dict[str, float] = {}
        for name in self.load_names:
            self.c.Loads.Name = name
            self.base_kw[name] = float(self.c.Loads.kW)
            self.base_kvar[name] = float(self.c.Loads.kvar)

        # group -> load names (validated against the mapping)
        self.group_loads = {g.group_id: list(g.load_names) for g in groups}

        # ---- flat load actuation plan ----------------------------------------
        # Precomputed once so ``set_loads`` does no dict/string work per step.
        # Each entry is (load_name, group_id, share_of_group_baseline_kw).
        self._load_plan: list[tuple[str, str, float]] = []
        for g in groups:
            tot = sum(self.base_kw[n] for n in g.load_names)
            if tot <= 0:
                continue
            for n in g.load_names:
                self._load_plan.append((n, g.group_id, self.base_kw[n] / tot))

        # line rating provenance + unrated lines
        self.line_normamps: dict[str, float] = {}
        self.line_linecode: dict[str, str] = {}
        self.unrated_lines: set[str] = set()
        for name in self.line_names:
            self.c.Lines.Name = name
            self.line_linecode[name] = str(self.c.Lines.LineCode)
            na = float(self.c.Lines.NormAmps)
            self.line_normamps[name] = na
            if str(self.c.Lines.LineCode) == "" or self.c.Lines.IsSwitch:
                self.unrated_lines.add(name)

        self.rated_lines = [n for n in self.line_names if n not in self.unrated_lines]
        self.rated_normamps = [self.line_normamps[n] for n in self.rated_lines]

        # ---- flat line loading plan ------------------------------------------
        # Only rated lines are swept. Each entry is
        # (name, normamps, emerg_over_norm, (magnitude index tuple, ...)).
        self._line_plan: list[tuple[str, float, float, tuple[int, ...]]] = []
        for name in self.rated_lines:
            self.c.Lines.Name = name
            na = self.line_normamps[name]
            if na <= 0:
                continue
            ce = self.c.ActiveCktElement
            nc = int(ce.NumConductors)
            if nc <= 0:
                continue
            emerg = float(ce.EmergAmps)
            self._line_plan.append(
                (name, na, (emerg / na if emerg > 0 else 1.0),
                 tuple(range(0, 2 * nc, 2))))

        # node -> domain (for local voltage measurements)
        self.node_domain: dict[str, str] = {}
        for g in groups:
            for bus in g.buses:
                node = bus if bus in set(self.node_names) else None
                if node is None:
                    cand = [n for n in self.node_names
                            if n == bus or n.startswith(bus + ".")]
                    node = cand[0] if cand else None
                if node is not None:
                    self.node_domain[node] = g.domain_id
        self._node_index = {n: i for i, n in enumerate(self.node_names)}
        self._domain_nodes: dict[str, list[int]] = {}
        for node, dom in self.node_domain.items():
            self._domain_nodes.setdefault(dom, []).append(self._node_index[node])
        self.domain_ids = sorted({g.domain_id for g in groups})

        # nominal totals at the documented baseline
        self.total_base_kw = sum(self.base_kw.values())
        self.total_base_kvar = sum(self.base_kvar.values())

        self._last_mults: dict[str, float] = {g.group_id: 1.0 for g in groups}

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Restore every load to its documented baseline and re-solve."""
        loads = self.c.Loads
        base_kw = self.base_kw
        base_kvar = self.base_kvar
        for name in self.load_names:
            loads.Name = name
            loads.kW = base_kw[name]
            loads.kvar = base_kvar[name]
        self._last_mults = {g.group_id: 1.0 for g in self.groups}
        self.c.Solution.Solve()

    # ------------------------------------------------------------ load update
    def set_loads(self, demand_kw: dict[str, float], demand_kvar: dict[str, float]) -> None:
        """Set every group's aggregate P/Q (kW, kvar) directly.

        ``demand_kw[g]`` is the *total* active demand of group ``g`` across all
        of its loads. The per-load split preserves each load's share of the
        group baseline, so the group sum is exact and phase connections are
        untouched.

        Implementation note: this uses the OpenDSS *property* API rather than
        ``Text.Command`` strings. Both produce identical load values (verified in
        ``tests/test_feeder_reset_and_convergence.py``) but the property API is
        about four times faster, which is what makes the full 11-fault held-out
        grid affordable.
        """
        loads = self.c.Loads
        q_ratio = {gid: (demand_kvar[gid] / kw if kw > 1e-12 else 0.0)
                   for gid, kw in demand_kw.items()}
        for name, gid, share in self._load_plan:
            kw = demand_kw[gid] * share
            loads.Name = name
            loads.kW = kw
            loads.kvar = kw * q_ratio[gid]

    # ------------------------------------------------------------------ solve
    def solve(self) -> SolveResult:
        c = self.c
        c.Solution.Solve()
        conv = bool(c.Solution.Converged)
        res = SolveResult(
            converged=conv,
            solver_iterations=int(c.Solution.Iterations),
            control_iterations=int(c.Solution.ControlIterations),
            vmin_pu=float("nan"), vmin_node="", vmax_pu=float("nan"), vmax_node="",
            max_line_loading_pu=float("nan"), worst_line="",
            n_lines_over_norm=0, n_lines_over_emerg=0,
            losses_kw=float("nan"), losses_kvar=float("nan"),
            substation_kw=float("nan"), substation_kvar=float("nan"),
        )
        if not conv:
            return res

        # ---- voltages ---------------------------------------------------------
        vpu = list(c.AllBusVmagPu)
        names = self.node_names
        best_min, best_max = float("inf"), -float("inf")
        arg_min, arg_max = "", ""
        for n, v in zip(names, vpu):
            v = float(v)
            if not (0.05 < v < 2.0):
                continue
            if v < best_min:
                best_min, arg_min = v, n
            if v > best_max:
                best_max, arg_max = v, n
        res.vmin_pu, res.vmin_node = best_min, arg_min
        res.vmax_pu, res.vmax_node = best_max, arg_max

        for dom, idxs in self._domain_nodes.items():
            dv = min((float(vpu[i]) for i in idxs if 0.05 < float(vpu[i]) < 2.0),
                     default=float("nan"))
            res.domain_vmin_pu[dom] = dv

        # ---- line loading -----------------------------------------------------
        # Only lines with a documented rating are swept; the eight code-less
        # switch segments are excluded (they carry no rating and no conductor).
        ce = c.ActiveCktElement
        lines = c.Lines
        mx, worst, over_n, over_e = 0.0, "", 0, 0
        for name, na, emerg_ratio, idx in self._line_plan:
            lines.Name = name
            cur = ce.CurrentsMagAng
            nidx = len(idx)
            if nidx == 1:
                peak = cur[0]
            elif nidx == 2:
                a, b = cur[0], cur[2]
                peak = a if a > b else b
            elif nidx == 3:
                a, b, d = cur[0], cur[2], cur[4]
                peak = a if a > b else b
                if d > peak:
                    peak = d
            else:
                peak = 0.0
                for i in idx:
                    v = cur[i]
                    if v > peak:
                        peak = v
            pu = peak / na
            if pu > mx:
                mx, worst = pu, name
            if pu > 1.0:
                over_n += 1
            if pu > emerg_ratio:
                over_e += 1
        res.max_line_loading_pu, res.worst_line = mx, worst
        res.n_lines_over_norm, res.n_lines_over_emerg = over_n, over_e

        # ---- power ------------------------------------------------------------
        tp = list(c.TotalPower)
        lo = list(c.Losses)
        res.substation_kw, res.substation_kvar = float(tp[0]), float(tp[1])
        res.losses_kw, res.losses_kvar = float(lo[0]) / 1000.0, float(lo[1]) / 1000.0
        return res

    # ---------------------------------------------------------------- helpers
    def provenance(self) -> dict:
        return {
            "engine_package": ENGINE_PACKAGE,
            "engine_version": engine_version(),
            "master": str(self.master_path).replace("\\", "/"),
            "master_sha256": self.master_sha256,
            "rating_patch": str(self.patch_path).replace("\\", "/"),
            "rating_patch_sha256": self.patch_sha256,
            "n_loads": len(self.load_names),
            "n_lines": len(self.line_names),
            "n_unrated_lines": len(self.unrated_lines),
            "unrated_lines": sorted(self.unrated_lines),
            "total_base_kw": self.total_base_kw,
            "total_base_kvar": self.total_base_kvar,
            "linecode_rating_provenance": RATING_PROVENANCE,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
