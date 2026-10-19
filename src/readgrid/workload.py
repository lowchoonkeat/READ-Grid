"""24-hour workload episodes from public IEEE European LV test-feeder profiles.

Provenance
----------
``data/raw/lv_profiles/Daily_1min_100profiles`` in the public
``dss-extensions/electricdss-tst`` repository. These are **public benchmark
profiles** for the IEEE European LV test feeder. They are NOT measured household
smart-meter traces and are never described as such here.

Shape normalisation (new study assumption)
------------------------------------------
Each published profile is a normalised per-unit active-power *shape*. It is

1. divided by its own mean,
2. clipped to a documented band ``[shape_clip_low, shape_clip_high]`` so the
   extreme short spikes (raw peak-to-mean up to 54) are not carried into the
   study, and
3. divided by its own 95th percentile, so that

       "workload scale 1.0" means the 95th-percentile daily demand equals the
       documented IEEE 123-bus baseline load (3490 kW / 1920 kVAr).

Step 3 is what keeps the feeder inside a physically sensible operating band: the
IEEE 123-bus main line already runs at ~1.19 pu of its referenced 50 degC
conductor rating at the documented nominal load, so a daily cycle that peaked far
above nominal would be meaningless. The convention is recorded in
``docs/experiment_protocol.md``. The fraction of samples clipped is reported.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # project root, derived from this file (portable)
PROC = ROOT / "data" / "processed"
EPISODES_CSV = PROC / "workload_episodes.csv"
INDEX_CSV = PROC / "workload_episodes_index.csv"
META_JSON = PROC / "workload_episodes_meta.json"


@dataclass
class Episode:
    episode_id: str
    regime: str
    shape: np.ndarray          # (1440,) daily multiplier, p95 = 1.0
    clip_fraction: float

    @property
    def p95_multiplier(self) -> float:
        return float(np.percentile(self.shape, 95))

    @property
    def peak_multiplier(self) -> float:
        return float(self.shape.max())


def load_episodes(clip_low: float = 0.20, clip_high: float = 2.50) -> dict[str, Episode]:
    idx = pd.read_csv(INDEX_CSV)
    long = pd.read_csv(EPISODES_CSV)
    shapes: dict[str, np.ndarray] = {}
    for eid, grp in long.groupby("episode_id", sort=True):
        g = grp.sort_values("minute")
        shapes[eid] = g["p_pu"].to_numpy(dtype=float)

    out: dict[str, Episode] = {}
    for _, row in idx.iterrows():
        eid = row["episode_id"]
        raw = shapes[eid]
        mean = raw.mean()
        unit = raw / mean if mean > 0 else raw
        n_clip = int(np.sum((unit < clip_low) | (unit > clip_high)))
        unit = np.clip(unit, clip_low, clip_high)
        p95 = float(np.percentile(unit, 95))
        norm = unit / p95 if p95 > 0 else unit
        out[eid] = Episode(
            episode_id=eid,
            regime=str(row["regime"]),
            shape=norm,
            clip_fraction=n_clip / unit.size,
        )
    return out


def stratified_split(episodes: dict[str, Episode], *,
                     n_train: int, n_val: int, n_test: int,
                     seed: int = 20261014) -> dict[str, list[str]]:
    """Disjoint train/val/test split stratified by load regime.

    Splitting happens at the *episode* level, never by randomly splitting
    adjacent time steps of the same trace (Phase 9 requirement).

    Allocation uses largest-remainder rounding so the per-regime counts sum
    exactly to the requested totals, and each regime never contributes more
    episodes than it has. Any episodes left over are held back and are not used
    by the study.
    """
    total_requested = n_train + n_val + n_test
    assert total_requested <= len(episodes), (total_requested, len(episodes))

    by_regime: dict[str, list[str]] = {}
    for eid, ep in episodes.items():
        by_regime.setdefault(ep.regime, []).append(eid)
    regimes = sorted(by_regime)
    n_total = len(episodes)

    rng = np.random.default_rng(seed)
    for r in regimes:
        by_regime[r] = sorted(by_regime[r])
        rng.shuffle(by_regime[r])

    def allocate(n_target: int) -> list[int]:
        exact = [n_target * len(by_regime[r]) / n_total for r in regimes]
        base = [int(np.floor(x)) for x in exact]
        remainder = n_target - sum(base)
        order = sorted(range(len(regimes)), key=lambda i: (-(exact[i] - base[i]), i))
        for i in order[:remainder]:
            base[i] += 1
        assert sum(base) == n_target
        return base

    k_tr = allocate(n_train)
    k_va = allocate(n_val)
    k_te = allocate(n_test)

    # never take more than a regime has
    buckets = [k_tr, k_va, k_te]
    for i, r in enumerate(regimes):
        avail = len(by_regime[r])
        while sum(b[i] for b in buckets) > avail:
            j = max(range(3), key=lambda b: buckets[b][i])
            assert buckets[j][i] > 0, f"cannot satisfy the split for regime {r!r}"
            buckets[j][i] -= 1
    assert sum(k_tr) == n_train, (sum(k_tr), n_train)
    assert sum(k_va) == n_val, (sum(k_va), n_val)
    assert sum(k_te) == n_test, (sum(k_te), n_test)

    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for i, r in enumerate(regimes):
        ids = by_regime[r]
        a, b = k_tr[i], k_tr[i] + k_va[i]
        c = b + k_te[i]
        split["train"] += ids[:a]
        split["val"] += ids[a:b]
        split["test"] += ids[b:c]

    for k in split:
        split[k] = sorted(split[k])
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))
    assert sum(len(v) for v in split.values()) == total_requested
    return split


def dr_windows(shape: np.ndarray, dr_trigger_fraction: float) -> np.ndarray:
    """Boolean array: is minute t a declared DR opportunity?

    Derived from the *uncontrolled baseline* shape and therefore identical for
    every policy evaluated on the same (episode, scale).
    """
    return shape >= dr_trigger_fraction


def episode_meta() -> dict:
    return json.loads(META_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    eps = load_episodes()
    print("episodes:", len(eps))
    sp = stratified_split(eps, n_train=40, n_val=18, n_test=18)
    for k, v in sp.items():
        print(f"  {k}: {len(v)}")
    clip = np.mean([e.clip_fraction for e in eps.values()])
    print(f"mean clipped fraction: {clip:.4f}")
