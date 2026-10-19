"""Communication model and fault traces (Phases 4 and 8).

The communication model is deliberately **simplified**: message sizes, bandwidth
and queueing are not modelled. What *is* modelled is what the study needs:

* a 1-second heartbeat stream cloud -> edge;
* per-control-tick state reports edge -> cloud;
* per-control-tick actuation commands cloud -> edge;
* one-way backhaul delay 50 ms + jitter, local-link delay 5 ms;
* the F0..F10 fault matrix.

Fault traces are **generated once per (fault, seed) and replayed identically
across every paired policy**, which is what the task prompt requires for
stochastic scenarios. The generated trace is a concrete artefact (saved under
``data/processed/fault_traces/``), not a re-draw inside each policy.

Sub-second events are *not* quantised to the 60 s power-flow interval: heartbeat
arrival times carry their full millisecond delay and the watchdog thresholds are
evaluated against those exact times.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # project root, derived from this file (portable)
TRACE_DIR = ROOT / "data" / "processed" / "fault_traces"


@dataclass
class FaultTrace:
    fault_id: str
    seed: int
    n_seconds: int
    n_ticks: int
    tick_s: int
    hb_sent: np.ndarray        # (n_seconds,) bool  - heartbeat sent at second s
    hb_arrival: np.ndarray     # (n_seconds,) float - arrival time or NaN
    e2c_arrival: np.ndarray    # (n_ticks,) float   - state-report arrival or NaN
    c2e_arrival: np.ndarray    # (n_ticks,) float   - command arrival or NaN
    c2e_dup_arrival: np.ndarray  # (n_ticks,) float - duplicate arrival or NaN
    c2e_dup: np.ndarray        # (n_ticks,) bool
    link_up: np.ndarray        # (n_seconds,) bool - is the link usable at second s

    def outage_mask(self, n_seconds: int) -> np.ndarray:
        return ~self.link_up[:n_seconds]


def _delay(rng: np.random.Generator, base_ms: float, jitter_ms: float) -> float:
    if jitter_ms <= 0:
        return base_ms / 1000.0
    return (base_ms + rng.uniform(-jitter_ms, jitter_ms)) / 1000.0


def generate_fault_trace(fault_id: str, cfg: dict, protocol, seed: int) -> FaultTrace:
    tick_s = int(protocol.control_interval_s)
    n_seconds = int(protocol.episode_length_s)
    n_ticks = n_seconds // tick_s
    rng = np.random.default_rng(seed)

    base_c2e = float(protocol.raw["comms"]["backhaul_oneway_delay_ms"])
    jit_c2e = float(protocol.raw["comms"]["backhaul_jitter_ms"])
    base_e2c = base_c2e
    jit_e2c = jit_c2e
    local_ms = float(protocol.raw["comms"]["local_oneway_delay_ms"])

    kind = cfg["kind"]

    # ---- per-second link usability + per-direction loss ----------------------
    hb_loss = np.zeros(n_seconds, dtype=bool)       # True == dropped
    c2e_loss = np.zeros(n_ticks, dtype=bool)
    e2c_loss = np.zeros(n_ticks, dtype=bool)
    extra_delay_ms = 0.0
    extra_jitter_ms = 0.0
    dup_p = 0.0
    replay_delay_s = 0.0

    tick_seconds = np.arange(n_ticks) * tick_s

    if kind == "connected":
        pass

    elif kind == "outage":
        s0 = int(cfg["start_s"])
        s1 = min(n_seconds, s0 + int(cfg["duration_s"]))
        hb_loss[s0:s1] = True
        c2e_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = True
        e2c_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = True

    elif kind == "packet_loss":
        p = float(cfg["p"])
        hb_loss = rng.random(n_seconds) < p
        c2e_loss = rng.random(n_ticks) < p
        e2c_loss = rng.random(n_ticks) < p

    elif kind == "burst_loss":
        p_gb = float(cfg["p_good_to_bad"])
        p_bg = float(cfg["p_bad_to_good"])
        p_lb = float(cfg["p_loss_bad"])
        bad = False
        hb_loss = np.zeros(n_seconds, dtype=bool)
        for s in range(n_seconds):
            bad = (rng.random() < p_bg) if bad else (rng.random() < p_gb)
            if bad:
                hb_loss[s] = rng.random() < p_lb
        for k, s in enumerate(tick_seconds):
            c2e_loss[k] = hb_loss[min(n_seconds - 1, int(s))]
            e2c_loss[k] = hb_loss[min(n_seconds - 1, int(s))]

    elif kind == "delay_jitter":
        extra_delay_ms = float(cfg["extra_delay_ms"])
        extra_jitter_ms = float(cfg["extra_jitter_ms"])

    elif kind == "asymmetric":
        p_c2e = float(cfg["cloud_to_edge_loss"])
        p_e2c = float(cfg["edge_to_cloud_loss"])
        hb_loss = rng.random(n_seconds) < p_c2e
        c2e_loss = rng.random(n_ticks) < p_c2e
        e2c_loss = rng.random(n_ticks) < p_e2c

    elif kind == "flapping":
        down = int(cfg["down_s"])
        up = int(cfg["up_s"])
        s0 = int(cfg["start_s"])
        s1 = min(n_seconds, s0 + int(cfg["total_s"]))
        period = down + up
        for s in range(s0, s1):
            if (s - s0) % period < down:
                hb_loss[s] = True
        c2e_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = True  # most ticks land in a down phase
        # refine: a tick is lost iff its second falls in a down phase
        for k, s in enumerate(tick_seconds):
            if s0 <= s < s1 and (s - s0) % period < down:
                c2e_loss[k] = True
            elif s0 <= s < s1:
                c2e_loss[k] = False
        e2c_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = False

    elif kind == "delayed_duplicates":
        replay_delay_s = float(cfg["delay_s"])
        dup_p = float(cfg["duplicate_p"])
        if "outage_start_s" in cfg:
            s0 = int(cfg["outage_start_s"])
            s1 = min(n_seconds, s0 + int(cfg["outage_s"]))
            hb_loss[s0:s1] = True
            c2e_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = True
            e2c_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = True

    elif kind == "stale_cache":
        s0 = int(cfg["outage_start_s"])
        s1 = min(n_seconds, s0 + int(cfg["outage_s"]))
        hb_loss[s0:s1] = True
        c2e_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = True
        e2c_loss[(tick_seconds >= s0) & (tick_seconds < s1)] = True

    else:
        raise ValueError(f"unknown fault kind {kind!r}")

    # ---- arrival times -------------------------------------------------------
    hb_arrival = np.full(n_seconds, np.nan)
    for s in range(n_seconds):
        if hb_loss[s]:
            continue
        d = _delay(rng, base_c2e + extra_delay_ms, jit_c2e + extra_jitter_ms)
        hb_arrival[s] = s + d

    e2c_arrival = np.full(n_ticks, np.nan)
    c2e_arrival = np.full(n_ticks, np.nan)
    c2e_dup_arrival = np.full(n_ticks, np.nan)
    c2e_dup = np.zeros(n_ticks, dtype=bool)
    for k, s in enumerate(tick_seconds):
        if not e2c_loss[k]:
            e2c_arrival[k] = s + local_ms / 1000.0 + _delay(
                rng, base_e2c + extra_delay_ms, jit_e2c + extra_jitter_ms)
        if not c2e_loss[k]:
            d = _delay(rng, base_c2e + extra_delay_ms, jit_c2e + extra_jitter_ms)
            if replay_delay_s > 0:
                d += replay_delay_s
            c2e_arrival[k] = s + local_ms / 1000.0 + d
            if replay_delay_s > 0 and rng.random() < dup_p:
                c2e_dup[k] = True
                c2e_dup_arrival[k] = c2e_arrival[k] + 2.0 + rng.random()

    # link usable at second s  ==  some heartbeat arrives within the last second
    link_up = np.zeros(n_seconds, dtype=bool)
    arr = hb_arrival[~np.isnan(hb_arrival)]
    if arr.size:
        idx = np.clip(np.searchsorted(arr, np.arange(n_seconds, dtype=float)) - 1, 0, arr.size - 1)
        recent = (np.arange(n_seconds) - arr[idx]) <= 1.0
        link_up = recent & (arr[idx] <= np.arange(n_seconds))
    else:
        link_up[:] = False

    return FaultTrace(
        fault_id=fault_id, seed=int(seed), n_seconds=n_seconds, n_ticks=n_ticks,
        tick_s=tick_s, hb_sent=~hb_loss, hb_arrival=hb_arrival,
        e2c_arrival=e2c_arrival, c2e_arrival=c2e_arrival,
        c2e_dup_arrival=c2e_dup_arrival, c2e_dup=c2e_dup,
        link_up=link_up,
    )


def heartbeat_state_timeline(trace: FaultTrace, protocol) -> dict:
    """Exact authority-transition timeline from the pre-generated heartbeat trace.

    Returns arrays over whole seconds:
      ``last_hb_age``  age (s) of the last *valid* heartbeat arrival at second s
      ``suspect``      boolean, age >= suspect_after_s
      ``takeover``     boolean, age >= takeover_after_s
    and the list of transition times.

    Sub-second delay is preserved because arrivals are compared against their
    real timestamps; the watchdog is therefore evaluated on the event clock, not
    on the 60 s power-flow grid.
    """
    n = trace.n_seconds
    arr = trace.hb_arrival
    valid = ~np.isnan(arr)
    idx = np.where(valid, np.arange(n), -1)
    last_idx = np.maximum.accumulate(idx)

    age = np.full(n, np.inf)
    ok = last_idx >= 0
    age[ok] = np.arange(n)[ok] - arr[last_idx[ok]]

    suspect_after = float(protocol.raw["authority"]["suspect_after_s"])
    takeover_after = float(protocol.raw["authority"]["takeover_after_s"])
    suspect = age >= suspect_after
    takeover = age >= takeover_after

    def _edges(mask: np.ndarray) -> list[int]:
        if mask.size == 0:
            return []
        d = np.diff(mask.astype(np.int8))
        return list(np.where(d == 1)[0] + 1)

    return {
        "age": age,
        "suspect": suspect,
        "takeover": takeover,
        "suspect_edges": _edges(suspect),
        "takeover_edges": _edges(takeover),
        "recover_edges": _edges(~suspect & (np.arange(n) > 0)),
    }


def save_trace(trace: FaultTrace) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    p = TRACE_DIR / f"{trace.fault_id}_seed{trace.seed}.npz"
    np.savez_compressed(
        p,
        fault_id=trace.fault_id, seed=trace.seed, n_seconds=trace.n_seconds,
        n_ticks=trace.n_ticks, tick_s=trace.tick_s,
        hb_sent=trace.hb_sent, hb_arrival=trace.hb_arrival,
        e2c_arrival=trace.e2c_arrival, c2e_arrival=trace.c2e_arrival,
        c2e_dup_arrival=trace.c2e_dup_arrival, c2e_dup=trace.c2e_dup,
        link_up=trace.link_up,
    )
    return p
