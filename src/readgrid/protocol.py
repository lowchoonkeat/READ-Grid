"""Frozen protocol loading and derived constants.

All numeric assumptions live in ``configs/protocol.yaml``. This module is the
single place that reads them, so every experiment run can record the exact
protocol hash it used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]   # project root, derived from this file (portable)
DEFAULT_PROTOCOL = ROOT / "configs" / "protocol.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Protocol:
    """Immutable view over the frozen protocol YAML."""

    raw: dict[str, Any]
    path: Path
    sha256: str

    # ---- convenience accessors -------------------------------------------------
    @property
    def n_groups(self) -> int:
        return int(self.raw["topology"]["n_groups"])

    @property
    def n_domains(self) -> int:
        return int(self.raw["topology"]["n_domains"])

    @property
    def groups_per_domain(self) -> int:
        return int(self.raw["topology"]["groups_per_domain"])

    @property
    def flexible_fraction(self) -> float:
        return float(self.raw["flexibility"]["flexible_fraction"])

    @property
    def action_levels(self) -> list[float]:
        return [float(x) for x in self.raw["flexibility"]["action_levels"]]

    @property
    def n_actions(self) -> int:
        return len(self.action_levels)

    def group_multiplier(self, level: int) -> float:
        """Group demand multiplier for an action level index."""
        return 1.0 - self.flexible_fraction * self.action_levels[level]

    @property
    def multipliers(self) -> list[float]:
        return [self.group_multiplier(i) for i in range(self.n_actions)]

    @property
    def control_interval_s(self) -> int:
        return int(self.raw["workload"]["control_interval_s"])

    @property
    def episode_length_s(self) -> int:
        return int(self.raw["workload"]["episode_length_s"])

    @property
    def heartbeat_interval_s(self) -> int:
        return int(self.raw["workload"]["heartbeat_interval_s"])

    @property
    def steps_per_episode(self) -> int:
        return self.episode_length_s // self.control_interval_s

    @property
    def delta_dr(self) -> float:
        return float(self.raw["target"]["delta_dr"])

    @property
    def dr_trigger_fraction(self) -> float:
        return float(self.raw["target"]["dr_trigger_fraction"])

    @property
    def cache_max_age_s(self) -> float:
        return float(self.raw["target"]["cache_max_age_s"])

    @property
    def ramp_max_level_step(self) -> int:
        return int(self.raw["constraints"]["ramp_max_level_step"])

    @property
    def dwell_time_s(self) -> float:
        return float(self.raw["constraints"]["dwell_time_s"])

    @property
    def burden_cap_fraction(self) -> float:
        return float(self.raw["constraints"]["burden_cap_fraction"])

    @property
    def max_flexible_fraction(self) -> float:
        """Largest curtailment a single group can apply, as a fraction of demand."""
        return self.flexible_fraction * max(self.action_levels)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.raw)


def load_protocol(path: Path | str | None = None) -> Protocol:
    p = Path(path) if path is not None else DEFAULT_PROTOCOL
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Protocol(raw=raw, path=p, sha256=_sha256(p))
