"""Build the workload episode library from the IEEE European LV test feeder profiles.

Provenance
----------
Source: IEEE PES Distribution Test Feeders — IEEE European Low Voltage Test Feeder,
as distributed in the dss-extensions/electricdss-tst repository under
``Version8/Distrib/IEEETestCases/LVTestCase/Daily_1min_100profiles``.

These are **public benchmark profiles** for the IEEE European LV test feeder.
They are NOT measured household smart-meter traces, and they must never be
described as such. Each file is a normalised daily active-power shape at a
1-minute resolution (1440 samples), published as part of the IEEE benchmark.

Output
------
``data/processed/workload_episodes.parquet`` (or .csv fallback) holding one row
per (episode_id, minute) with the normalised per-unit active-power shape, plus a
``workload_episodes_index.csv`` giving the per-episode summary statistics used
for stratified splitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
SRC = ROOT / "data" / "raw" / "lv_profiles" / "Daily_1min_100profiles"
OUT = ROOT / "data" / "processed"
MINUTES = 1440
N_PROFILES = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_profile(path: Path) -> np.ndarray:
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vals.append(float(line))
    arr = np.asarray(vals, dtype=float)
    if arr.size != MINUTES:
        raise ValueError(f"{path.name}: expected {MINUTES} samples, got {arr.size}")
    return arr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("load_profile_*.txt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if len(files) != N_PROFILES:
        raise RuntimeError(f"expected {N_PROFILES} profiles, found {len(files)}")

    shapes = {}
    hashes = {}
    for p in files:
        pid = int(p.stem.split("_")[-1])
        shapes[pid] = load_profile(p)
        hashes[p.name] = sha256(p)

    mat = np.vstack([shapes[i] for i in range(1, N_PROFILES + 1)])  # (100, 1440)

    # --- how independent are the 100 profiles? ---
    # Pairwise correlation gives an honest answer about effective sample size.
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(mat)
    off = corr[np.triu_indices_from(corr, k=1)]
    # count exact duplicates
    seen = {}
    dup_groups = {}
    for i in range(mat.shape[0]):
        key = hashlib.sha256(mat[i].tobytes()).hexdigest()
        seen.setdefault(key, []).append(i + 1)
    for k, v in seen.items():
        if len(v) > 1:
            dup_groups[str(v)] = len(v)

    rows = []
    for pid in range(1, N_PROFILES + 1):
        s = shapes[pid]
        rows.append({
            "episode_id": f"LVP{pid:03d}",
            "profile_index": pid,
            "mean_pu": float(s.mean()),
            "std_pu": float(s.std()),
            "min_pu": float(s.min()),
            "max_pu": float(s.max()),
            "peak_to_mean": float(s.max() / s.mean()) if s.mean() > 0 else np.nan,
            "afternoon_mean_12_18": float(s[720:1080].mean()),
            "evening_mean_18_22": float(s[1080:1320].mean()),
            "night_mean_0_6": float(s[0:360].mean()),
            "day_energy_pu_min": float(s.sum()),
            "argmax_minute": int(np.argmax(s)),
        })
    idx = pd.DataFrame(rows)

    # stratify into low / mid / high load regimes by day energy terciles
    idx["regime"] = pd.qcut(idx["day_energy_pu_min"], q=3,
                            labels=["low", "mid", "high"]).astype(str)

    # --- long form ---
    long_rows = []
    for pid in range(1, N_PROFILES + 1):
        s = shapes[pid]
        long_rows.append(pd.DataFrame({
            "episode_id": f"LVP{pid:03d}",
            "minute": np.arange(MINUTES),
            "p_pu": s,
        }))
    long = pd.concat(long_rows, ignore_index=True)

    idx.to_csv(OUT / "workload_episodes_index.csv", index=False)
    try:
        long.to_parquet(OUT / "workload_episodes.parquet", index=False)
        long_file = "workload_episodes.parquet"
    except Exception:
        long.to_csv(OUT / "workload_episodes.csv", index=False)
        long_file = "workload_episodes.csv"

    meta = {
        "source": "IEEE European LV Test Feeder, Daily_1min_100profiles",
        "repo": "https://github.com/dss-extensions/electricdss-tst",
        "path": "Version8/Distrib/IEEETestCases/LVTestCase/Daily_1min_100profiles",
        "retrieved": "2026-10-14",
        "profile_count": N_PROFILES,
        "samples_per_profile": MINUTES,
        "resolution": "1 minute",
        "quantity": "normalised active-power shape (per-unit shape, useactual=no)",
        "licence": "IEEE PES test feeder benchmark, distributed in the public "
                   "dss-extensions/electricdss-tst repository",
        "is_measured_household_data": False,
        "provenance_note": (
            "Public IEEE benchmark profiles. These are NOT measured household "
            "smart-meter traces and must not be described as such."
        ),
        "file_hashes": hashes,
        "independence": {
            "pairwise_correlation_mean": float(np.mean(off)),
            "pairwise_correlation_min": float(np.min(off)),
            "pairwise_correlation_max": float(np.max(off)),
            "pairwise_correlation_p90": float(np.percentile(off, 90)),
            "exact_duplicate_groups": dup_groups,
            "n_exact_duplicate_groups": len(dup_groups),
        },
        "long_file": long_file,
    }
    (OUT / "workload_episodes_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"profiles: {N_PROFILES} x {MINUTES}")
    print("day-energy tercile regime counts:")
    print(idx["regime"].value_counts().to_string())
    print("\npairwise correlation across the 100 shapes:")
    print(f"  mean {np.mean(off):+.4f}  min {np.min(off):+.4f}  "
          f"max {np.max(off):+.4f}  p90 {np.percentile(off, 90):+.4f}")
    print(f"  exact duplicate groups: {len(dup_groups)}  {dup_groups}")
    print("\npeak-to-mean ratio: "
          f"min {idx.peak_to_mean.min():.2f} median {idx.peak_to_mean.median():.2f} "
          f"max {idx.peak_to_mean.max():.2f}")
    print(f"\nwrote {OUT / 'workload_episodes_index.csv'}")
    print(f"wrote {OUT / long_file}")
    print(f"wrote {OUT / 'workload_episodes_meta.json'}")


if __name__ == "__main__":
    main()
