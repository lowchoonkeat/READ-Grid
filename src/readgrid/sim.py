"""Multirate closed-loop simulator (Phases 4, 5 and 6).

Timeline of one 24-hour episode
-------------------------------
* physical workload sampled every ``control_interval_s`` = 60 s (1440 steps);
* heartbeat stream every 1 s on the event clock;
* authority state changes resolved on the event clock at millisecond precision
  from the pre-generated, replayed fault trace;
* one power flow per step, solved *after* the accepted actuation is applied.

Causality
---------
No controller ever sees future workload or future fault information. The cloud
only sees state reports that actually arrived over the simulated link; the edge
only sees its own measurements plus the DR schedule it has received. Cached
values carry an explicit age and the age is logged.

Command path
------------
Every actuation command (cloud or local) is submitted to the domain arbiter and
either accepted or rejected. There is no cloud-to-actuator bypass. Proposed and
executed actions are recorded separately.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .arbiter import CONNECTED, LOCAL_FALLBACK, Command, DomainArbiter
from .comm import FaultTrace, heartbeat_state_timeline
from .reward import step_reward, step_reward_achievement
from .value_model import (BINNED_DIM, state_features, state_features_binned,
                          state_features_v2)


def _clip_int(v: int, lo: int, hi: int) -> int:
    """Integer clamp without NumPy scalar overhead (hot path)."""
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class PolicySpec:
    policy_id: str
    cloud_kind: str | None = None        # 'rule' | 'linear' | 'concatenated' | None
    local_kind: str | None = None
    cloud_ctl: object | None = None
    local_ctl: object | None = None
    guard: str = "guarded"               # or 'heartbeat_only'
    cache_enforced: bool = True
    explore_eps: float = 0.0             # epsilon-greedy exploration during training
    # v2: draw exploration actions from the ADMISSIBLE set rather than uniformly
    # over all levels. Matches the sandbox behaviour policy; without it the
    # learner spends its exploration budget on requests the actuator clamps away.
    explore_admissible: bool = False
    label: str = ""
    # Phase 9 feature-removal sensitivity. A multiplicative mask applied to the
    # state vector after it is built. ``None`` means "no feature removed".
    # Zeroing a feature makes its coefficient unidentifiable for a linear model
    # (ridge drives it to zero), which is exactly "the feature is absent".
    feature_mask: tuple[float, ...] | None = None

    def describe(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "cloud_kind": self.cloud_kind,
            "local_kind": self.local_kind,
            "guard": self.guard,
            "cache_enforced": self.cache_enforced,
            "explore_eps": self.explore_eps,
            "label": self.label or self.policy_id,
            "feature_mask": self.feature_mask,
        }


@dataclass
class RunResult:
    policy_id: str
    episode_id: str
    scale: float
    fault_id: str
    comm_seed: int
    episode_regime: str = ""
    # target / achievement
    eligible_steps: int = 0
    achieved_steps: int = 0
    sum_exceedance: float = 0.0
    n_steps: int = 0
    # energy
    baseline_energy_kwh: float = 0.0
    actual_energy_kwh: float = 0.0
    curtail_energy_kwh: float = 0.0
    burden_cap_kwh: float = 0.0
    # outage-focus window (v2 fix F-2): every outage interval plus the 30
    # minutes that follow it. This is the window in which the guarded edge
    # fallback is supposed to earn its keep, so it is reported separately from
    # the all-day averages (which dilute the architectural effect to ~0.5% of
    # the day). Fields are 0 / NaN when the run has no outage at all (F0).
    outage_window_s: float = 0.0
    outage_eligible_steps: int = 0
    outage_achieved_steps: int = 0
    outage_switches: int = 0
    outage_baseline_energy_kwh: float = 0.0
    outage_actual_energy_kwh: float = 0.0
    # switching / burden
    n_switches: int = 0
    burden_used_fraction: float = 0.0
    # physics
    vmin_pu: float = float("nan")
    vmax_pu: float = float("nan")
    vmin_node: str = ""
    voltage_violation_s: float = 0.0
    max_line_loading_pu: float = float("nan")
    worst_line: str = ""
    line_violation_s: float = 0.0
    n_convergence_failures: int = 0
    n_steps_solved: int = 0
    # authority
    n_authority_switches: int = 0
    heartbeat_detect_delay_s: float = float("nan")
    authority_transfer_delay_s: float = float("nan")
    recovery_delay_s: float = float("nan")
    frac_time_local_authority: float = 0.0
    frac_time_cloud_authority: float = 0.0
    # commands
    n_commands_offered: int = 0
    n_commands_accepted: int = 0
    n_commands_rejected: int = 0
    reject_reasons: dict = field(default_factory=dict)
    mean_command_latency_s: float = float("nan")
    # runtime
    policy_inference_s: float = 0.0
    wall_s: float = 0.0
    max_physical_sampling_s: float = 60.0
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("reject_reasons", "extra")}
        d["reject_reasons"] = ";".join(f"{k}={v}" for k, v in sorted(self.reject_reasons.items()))
        d["achievement_rate"] = (self.achieved_steps / self.eligible_steps
                                 if self.eligible_steps else float("nan"))
        d["mean_exceedance"] = (self.sum_exceedance / self.eligible_steps
                                if self.eligible_steps else float("nan"))
        # v2 fix F-2: this ratio is actual/baseline scheduled energy. It is NOT
        # outage-service retention, so it is named for what it measures.
        d["energy_ratio"] = (self.actual_energy_kwh / self.baseline_energy_kwh
                             if self.baseline_energy_kwh else float("nan"))
        # Outage-focus metrics (outage window + the 30 min that follow it).
        d["outage_achievement_rate"] = (
            self.outage_achieved_steps / self.outage_eligible_steps
            if self.outage_eligible_steps else float("nan"))
        d["outage_energy_ratio"] = (
            self.outage_actual_energy_kwh / self.outage_baseline_energy_kwh
            if self.outage_baseline_energy_kwh else float("nan"))
        return d


def _max_level_within_budget(level: int, budget_left_kwh: float, base_kw: float,
                             flex: float, levels: list[float], tick_s: int) -> int:
    for cand in range(level, -1, -1):
        inc = base_kw * flex * levels[cand] * tick_s / 3600.0
        if inc <= budget_left_kwh + 1e-12:
            return cand
    return 0


def _admissible_levels(cur: int, since_change_s: float, budget_left_kwh: float,
                       base_kw: float, flex: float, levels: list[float],
                       tick_s: int, ramp_max_step: int, dwell_s: float,
                       n_actions: int) -> tuple:
    """Actions the actuator constraints would accept at a given state.

    Mirrors the enforcement order used at actuation (ramp -> dwell -> budget) so
    the set can be used to mask a Bellman target over the SUCCESSOR state
    (v2 fix F-3). Without the mask the fitted-Q target maximises over actions
    that the environment would never execute, which biases the value estimate.

    Returns a non-empty sorted tuple: when the budget is exhausted even at the
    current level, the only executable action is whatever the budget clamp
    returns, so that single action is reported instead of an empty set.
    """
    out = []
    for a in range(n_actions):
        if abs(a - cur) > ramp_max_step:
            continue
        if a != cur and since_change_s < dwell_s:
            continue
        if _max_level_within_budget(a, budget_left_kwh, base_kw, flex, levels,
                                    tick_s) != a:
            continue
        out.append(a)
    if not out:
        out = [_max_level_within_budget(cur, budget_left_kwh, base_kw, flex,
                                        levels, tick_s)]
    return tuple(sorted(set(out)))


def _domain_admissible(members: list, level: dict, last_change_t: dict,
                       burden_cap: dict, burden_used: dict, nom_kw: dict,
                       base_kw_now: float, flex: float, levels: list[float],
                       tick_s: int, protocol, n_actions: int,
                       t: float) -> set:
    """Admissible DOMAIN actions (v2 fix F-3, domain scope).

    A cloud command carries ONE level for every member group, and each member
    then clamps that request to its own ramp/dwell/budget limits. The level the
    domain actually realises is therefore ``round(mean(clamped per-member
    levels))``, and a domain action is admissible exactly when it is a FIXED
    POINT of that map -- the same "the environment would execute this action as
    itself" condition :func:`_admissible_levels` applies per group. When the map
    has no fixed point the realised level of the neutral request is used, so the
    returned set is never empty.
    """
    def _realise(a_req: int) -> int:
        ex = []
        for g in members:
            req = a_req
            cur = level[g]
            if req - cur > protocol.ramp_max_level_step:
                req = cur + protocol.ramp_max_level_step
            elif cur - req > protocol.ramp_max_level_step:
                req = cur - protocol.ramp_max_level_step
            if t - last_change_t[g] < protocol.dwell_time_s and req != cur:
                req = cur
            req = _max_level_within_budget(req, burden_cap[g] - burden_used[g],
                                           nom_kw[g] * base_kw_now, flex, levels,
                                           tick_s)
            ex.append(_clip_int(req, 0, n_actions - 1))
        return int(round(sum(ex) / len(ex)))

    adm = {a for a in range(n_actions) if _realise(a) == a}
    return adm or {_realise(0)}


def _held_or_default(arb, t: float, members: list[str]) -> dict:
    """Hold the last valid set-point until its documented expiry, then default."""
    lc = arb.last_accepted_cmd
    if lc is not None and not lc.is_expired(t) and isinstance(lc.requested_action, dict):
        return {g: int(lc.requested_action.get(g, 0)) for g in members}
    return {g: 0 for g in members}


def run_simulation(*, feeder, groups, protocol, episode, scale: float,
                   trace: FaultTrace, policy: PolicySpec,
                   collect_transitions: bool = False,
                   collect_for: str | None = None,
                   record_series: bool = False,
                   record_events: bool = False,
                   rng_seed: int = 0) -> tuple[RunResult, list | None]:
    t_wall0 = time.perf_counter()
    rng = np.random.default_rng(rng_seed)
    n_ticks = protocol.steps_per_episode
    tick_s = protocol.control_interval_s
    shape = np.asarray(episode.shape[:n_ticks], dtype=float)
    n_actions = protocol.n_actions
    flex = protocol.flexible_fraction
    levels = protocol.action_levels
    max_level = max(levels)
    mults = protocol.multipliers
    inv_mults = [1.0 / m for m in mults]
    w = protocol.raw["reward"]
    w_track = float(w["w_track"])
    w_burden = float(w["w_burden"])
    w_switch = float(w["w_switch"])
    # ---- v2 learning knobs (default to the v1 behaviour) --------------------
    vm_cfg = protocol.raw.get("value_model", {})
    feature_variant = str(vm_cfg.get("feature_variant", "v1"))
    use_v2_feat = feature_variant.startswith("v2")
    use_binned = feature_variant == "v2_binned"
    dwell_feature = bool(vm_cfg.get("dwell_feature", False))
    reward_mode = str(w.get("mode", "hinge"))
    w_achievement = float(w.get("w_achievement", 1.0))
    w_curtail = float(w.get("w_curtail", w_burden))

    def _learn_reward(*, net_kw, baseline_kw, target_kw, curtailed_kw,
                      level_changed, in_dr):
        """Learning reward; mode selected by ``reward.mode`` in the protocol."""
        if reward_mode == "achievement":
            return step_reward_achievement(
                net_kw=net_kw, baseline_kw=baseline_kw, target_kw=target_kw,
                curtailed_kw=curtailed_kw, level_changed=level_changed,
                flexible_fraction=flex, max_level=max_level, in_dr=in_dr,
                w_achievement=w_achievement, w_curtail=w_curtail,
                w_switch=w_switch).total
        return step_reward(
            net_kw=net_kw, baseline_kw=baseline_kw, target_kw=target_kw,
            curtailed_kw=curtailed_kw, level_changed=level_changed,
            flexible_fraction=flex, max_level=max_level,
            w_track=w_track, w_burden=w_burden, w_switch=w_switch).total

    # v2: collect transitions from ONE representative group / domain. Every group
    # has cap_kw = cap_frac * nom_kw, burden_cap = frac * nom_kw * hours * lf and
    # a budget increment proportional to nom_kw, and every group in an episode
    # shares one normalised shape, so all 50 groups and all 5 domains follow the
    # SAME normalised trajectory -- the other 49 copies carry no extra state
    # information. This is also what the sandbox models (one normalised group),
    # so it is required for the G1 equivalence claim to transfer to training.
    collect_repr = bool(vm_cfg.get("collect_representative_group", False))
    lease_s = float(protocol.raw["comms"]["cloud_lease_s"])
    ansi_lo = float(protocol.raw["metrics"]["ansi_lower_pu"])
    ansi_hi = float(protocol.raw["metrics"]["ansi_upper_pu"])
    loading_lim = float(protocol.raw["metrics"]["loading_limit_pu"])
    kwh_per_tick = tick_s / 3600.0
    # Phase 9 feature-removal sensitivity: ``None`` in every main-grid run.
    fmask = (np.asarray(policy.feature_mask, dtype=float)
             if policy.feature_mask is not None else None)

    gids = [g.group_id for g in groups]
    dom_of = {g.group_id: g.domain_id for g in groups}
    doms = sorted({g.domain_id for g in groups})
    dom_groups = {d: [g for g in gids if dom_of[g] == d] for d in doms}
    base_kvar = {g.group_id: g.base_kvar for g in groups}
    nom_kw = {g.group_id: g.base_kw * scale for g in groups}
    dom_nom = {d: sum(nom_kw[g] for g in dom_groups[d]) for d in doms}

    # ---- DR schedule (day-ahead, identical for all policies) ----------------
    # All three declared quantities (opportunity windows, reduction target and
    # the daily energy budget) come from published dataset constants and the
    # documented nominal baseline. None of them is read from the realised
    # episode, so no controller receives future workload information.
    dr = shape >= protocol.dr_trigger_fraction
    s_ref = float(protocol.raw["target"]["dr_s_ref"])
    cap_frac = 1.0 - protocol.delta_dr
    cap_kw = {g.group_id: cap_frac * nom_kw[g.group_id] * s_ref for g in groups}
    delta_eff = np.where(dr, protocol.delta_dr, 0.0)

    fb_h0, fb_h1 = protocol.raw["target"]["static_fallback_window"]
    static_window = np.zeros(n_ticks, dtype=bool)
    static_window[int(fb_h0) * 60:min(int(fb_h1) * 60, n_ticks)] = True
    fb_delta = float(protocol.raw["target"]["static_fallback_delta"])

    tl = heartbeat_state_timeline(trace, protocol)

    # ---- precomputed per-step delivery flags --------------------------------
    # Heartbeat arrivals inside control step k come only from sends inside the
    # same second-range (the one-way delay is far below 1 s), so the per-step
    # "did a heartbeat arrive and when" question is answered once, up front,
    # instead of by slicing a NumPy array on every one of the 1440 steps.
    hb_grid = trace.hb_arrival[:n_ticks * tick_s].reshape(n_ticks, tick_s)
    hb_mask = ~np.isnan(hb_grid)
    hb_has = hb_mask.any(axis=1)
    hb_last = np.where(hb_mask, hb_grid, -np.inf).max(axis=1)

    # ---- causal visibility of heartbeat arrivals (v2 fix F-1) ---------------
    # A decision taken at t = k * tick_s may only use heartbeats that have
    # ALREADY ARRIVED at or before t. The previous implementation took the LAST
    # arrival inside window k -- which can be up to tick_s - 1 seconds in the
    # FUTURE relative to the window start -- and applied it at the window start,
    # so a later arrival could influence an earlier decision.
    #
    # Arrivals are now shifted by exactly one control step: everything that
    # arrived during window k-1 becomes visible at the tick-k decision. This is
    # strictly causal and conservative (an arrival is never acted on before it
    # happens); the resulting authority delays stay quantised to the control
    # grid, which is the documented limitation (see P359_SUBMISSION_BLOCKERS.md
    # item 3: no event-time or millisecond takeover claim is made).
    hb_visible_has = np.zeros(n_ticks, dtype=bool)
    hb_visible_t = np.full(n_ticks, -np.inf, dtype=float)
    if n_ticks > 1:
        hb_visible_has[1:] = hb_has[:-1]
        hb_visible_t[1:] = hb_last[:-1]
    c2e_ok = ~np.isnan(trace.c2e_arrival)
    e2c_ok = ~np.isnan(trace.e2c_arrival)
    c2e_dup = trace.c2e_dup

    # The arbiter starts in the authority of whichever controller the policy
    # actually has. A policy with no cloud controller (local-only diagnostic
    # configurations) therefore starts with the edge holding actuation authority.
    init_auth = "cloud" if policy.cloud_kind is not None else "local"
    arbiters = {d: DomainArbiter(d, protocol, initial_authority=init_auth) for d in doms}
    strict = (policy.guard == "guarded")

    level = {g: 0 for g in gids}
    last_change_t = {g: -1e9 for g in gids}
    burden_used = {g: 0.0 for g in gids}
    # Declared daily energy budget: 3% of the group's *declared* nominal daily
    # energy (nominal_kw x episode hours x documented load factor). Independent
    # of the realised episode.
    declared_hours = protocol.episode_length_s / 3600.0
    declared_lf = float(protocol.raw["constraints"]["declared_load_factor"])
    burden_cap = {g: protocol.burden_cap_fraction * nom_kw[g] * declared_hours * declared_lf
                  for g in gids}

    res = RunResult(policy_id=policy.policy_id, episode_id=episode.episode_id,
                    scale=scale, fault_id=trace.fault_id, comm_seed=trace.seed,
                    n_steps=n_ticks, episode_regime=episode.regime,
                    burden_cap_kwh=float(sum(burden_cap.values())))

    # ---- outage-focus window (v2 fix F-2) -----------------------------------
    # Union of every contiguous outage interval plus the 30 minutes that follow
    # it, expressed on the control grid. This is the window in which the guarded
    # edge fallback is supposed to earn its keep, so it is reported separately
    # from the all-day average (which dilutes the architectural effect to about
    # 0.5% of the day). A run with no outage at all (F0, or a stochastic fault
    # whose link never drops) yields an all-False mask; the outage metrics then
    # stay at 0 / NaN and are reported as undefined rather than invented.
    # A down-run counts as an outage only if it lasts at least one control
    # interval. Every trace starts with link_up[0] == False while the link is
    # being established, so without this minimum every run -- including the
    # fully connected F0 -- would report a spurious outage window (and its
    # 30-minute tail). A blip shorter than the control interval cannot create a
    # fallback episode in the first place, because the arbiter needs
    # suspect_after_s + takeover_after_s seconds before authority moves.
    OUTAGE_TAIL_S = 1800
    OUTAGE_MIN_S = int(tick_s)
    n_sec = n_ticks * tick_s
    down_sec = ~np.asarray(trace.link_up[:n_sec], dtype=bool)
    focus_sec = np.zeros(n_sec, dtype=bool)
    if down_sec.any():
        idx = np.flatnonzero(down_sec)
        starts = idx[np.r_[True, np.diff(idx) > 1]]
        ends = idx[np.r_[np.diff(idx) > 1, True]]
        for s, e in zip(starts, ends):
            if int(e) - int(s) + 1 < OUTAGE_MIN_S:
                continue
            focus_sec[s:min(n_sec, int(e) + 1 + OUTAGE_TAIL_S)] = True
    outage_focus = focus_sec.reshape(n_ticks, tick_s).any(axis=1)
    res.outage_window_s = float(int(outage_focus.sum()) * tick_s)

    feeder.reset()
    sol = feeder.solve()
    res.n_steps_solved += 1
    if not sol.converged:
        res.n_convergence_failures += 1
    p_obs = {g: nom_kw[g] * shape[0] for g in gids}
    domain_vmin = {d: sol.domain_vmin_pu.get(d, float("nan")) for d in doms}

    cloud_report: dict[str, dict] = {}
    cloud_epoch_known = {d: 0 for d in doms}
    seq = {"cloud": 0, "local": 0}
    last_cmd: dict[str, Command] = {}
    latency_samples: list[float] = []
    # The DR schedule is a day-ahead DECLARED signal, available from t = 0
    # (configs/protocol.yaml, "target"). Its cache is therefore considered
    # received at t = 0 rather than never; without this the causal heartbeat
    # shift (F-1) would leave the cache unbound on the first tick, when no
    # arrival has yet become visible.
    last_schedule_time = 0.0
    last_hb_valid_t = 0.0
    detect_delays: list[float] = []
    transfer_delays: list[float] = []
    recovery_delays: list[float] = []
    time_local = 0
    time_cloud = 0
    transitions: list = []
    prev_x: dict = {}
    prev_a: dict = {}
    prev_r: dict = {}
    prev_level = {g: 0 for g in gids}
    prev_hour = {g: 0.0 for g in gids}
    prev_cloud_x: dict = {}
    prev_cloud_a: dict = {}
    prev_cloud_r: dict = {}
    prev_cloud_level = {g: 0 for g in gids}
    prev_hour_d: dict = {}
    prev_state = {d: CONNECTED for d in doms}
    # Authority tracking for the transfer/recovery delay metrics. This must be
    # the *previous authority*, not the previous state: a policy with no cloud
    # controller starts (and stays) in local authority while its state remains
    # CONNECTED, so a state-based test fires on every tick and records a mean of
    # "time since the last heartbeat" as if it were a transfer delay. Only a real
    # authority change is a transfer.
    prev_auth = {d: arbiters[d].authority for d in doms}
    series_mean_level: list[float] = [] if record_series else []
    series_curt_kw: list[float] = [] if record_series else []
    series_proposed: list[float] = [] if record_series else []
    event_log: list[dict] = [] if record_events else []

    for k in range(n_ticks):
        t = k * tick_s
        hour = (t / 3600.0) % 24.0
        hb_arrived = bool(hb_visible_has[k])
        if hb_arrived:
            hb_t = float(hb_visible_t[k])
            last_schedule_time = hb_t
            last_hb_valid_t = hb_t

        # ---- event-driven authority -----------------------------------------
        for d in doms:
            arb = arbiters[d]
            if hb_arrived:
                evs = arb.on_heartbeat(hb_t)
            else:
                arb.on_heartbeat_loss(t)
                evs = []
            before = arb.state
            evs = list(evs) + list(arb.on_tick(t))
            if record_events:
                for e in evs:
                    event_log.append({
                        "t": float(e.t), "domain_id": e.domain_id,
                        "kind": e.kind, "detail": e.detail,
                        "authority": arb.authority, "epoch": int(arb.epoch),
                        "state": arb.state,
                    })
            # A heartbeat arrival inside the current 60 s tick window can carry a
            # timestamp later than the tick start, so these differences are
            # clamped at zero. The delays are therefore quantised to the control
            # grid (an upper bound of the event-time delay); the arbiter's own
            # threshold semantics are tested at 1 s resolution in test_arbiter.py
            # and test_comm.py.
            if arb.state == LOCAL_FALLBACK and before != LOCAL_FALLBACK:
                detect_delays.append(max(0.0, t - last_hb_valid_t))
            if arb.authority == "local" and prev_auth[d] != "local":
                transfer_delays.append(max(0.0, t - last_hb_valid_t))
            if arb.authority == "cloud" and prev_auth[d] == "local":
                # Authority is restored by a *heartbeat* event, not by the tick:
                # on_heartbeat() performs the transfer at the arrival timestamp
                # once the link has been valid for recovery_stable_s. The
                # measurable latency is therefore the event time of that
                # transfer minus the first valid heartbeat after the loss.
                # Measuring from the tick start instead gives a value that can be
                # *smaller* than the guard (down to ~0 s), because the arrival
                # that triggers the transfer can sit up to one control interval
                # after the tick in which the transfer is observed.
                recovery_delays.append(max(0.0, arb.last_transfer_time
                                           - (arb.link_up_since
                                              if arb.link_up_since is not None
                                              else arb.last_transfer_time)))
            prev_state[d] = arb.state
            prev_auth[d] = arb.authority

        authority = {d: arbiters[d].authority for d in doms}
        for d in doms:
            if authority[d] == "local":
                time_local += 1
            else:
                time_cloud += 1

        # ---- schedule available to the edge ---------------------------------
        cache_age = t - last_schedule_time
        cache_ok = (not policy.cache_enforced) or (cache_age <= protocol.cache_max_age_s)
        in_dr_now = bool(dr[k]) if cache_ok else bool(static_window[k])
        d_eff_now = float(delta_eff[k]) if cache_ok else (fb_delta if static_window[k] else 0.0)

        # ---- decisions -------------------------------------------------------
        t_inf0 = time.perf_counter()
        proposed = {g: 0 for g in gids}
        feat_now: dict = {}
        feat_cloud_now: dict = {}

        cloud_actions: dict[str, int] = {}
        adm_dom_now: dict = {}
        if policy.cloud_kind is not None:
            for d in doms:
                rep = cloud_report.get(d)
                if rep is None:
                    cloud_actions[d] = 0
                    continue
                b_dom = sum(rep["base_est"].values())
                exc = 0.0
                if in_dr_now and b_dom > 0:
                    c_dom = sum(cap_kw[g] for g in dom_groups[d])
                    exc = max(0.0, b_dom - c_dom) / (flex * max_level * b_dom)
                vdev = float(rep.get("voltage_dev", 0.0))
                budget = float(rep.get("remaining_budget", 1.0))
                cur = int(rep.get("current_level", 0))
                ctx = {"exceedance": exc, "voltage_dev": vdev,
                       "remaining_budget": budget, "current_level": cur, "hour": rep.get("hour", hour)}
                # Domain-level admissible set (v2 fix F-3). A cloud command
                # applies ONE level to every member group, and each member then
                # clamps that request to its own ramp/dwell/budget limits, so the
                # domain action the environment actually realises is
                # round(mean(clamped per-member levels)). A domain action is
                # admissible exactly when it is a FIXED POINT of that map -- the
                # same "the environment would execute this action as itself"
                # condition the per-group helper uses. When the map has no fixed
                # point the realised level of the neutral request is used, so the
                # set is never empty. Computed BEFORE decide() so a masked cloud
                # controller and admissible exploration can use it.
                if policy.explore_admissible or (collect_transitions
                                                 and collect_for == "cloud"):
                    adm_dom_now[d] = tuple(sorted(_domain_admissible(
                        dom_groups[d], level, last_change_t, burden_cap,
                        burden_used, nom_kw, shape[k], flex, levels, tick_s,
                        protocol, n_actions, t)))
                if policy.cloud_kind in ("linear", "concatenated"):
                    if use_binned:
                        # Domain-level dwell: a cloud command applies ONE level to
                        # every member, so the domain can only change level once
                        # the most constrained member may change. The minimum of
                        # the members' elapsed-change times is that quantity.
                        dom_dwell = min(int((t - last_change_t[g]) // tick_s)
                                        for g in dom_groups[d])
                        ctx["features"] = state_features_binned(
                            exceedance=exc, level=cur,
                            since_change_ticks=dom_dwell,
                            remaining_budget=budget, in_dr=bool(in_dr_now))
                    elif use_v2_feat:
                        # The cloud has no per-group dwell timer of its own; the
                        # dwell slot is the documented constant when the v2 layout
                        # is used at the cloud scope.
                        ctx["features"] = state_features_v2(
                            hour=rep.get("hour", hour),
                            demand_ratio=b_dom / max(1e-9, dom_nom[d]),
                            level=cur, n_actions=n_actions, remaining_budget=budget,
                            exceedance=exc, dwell_frac=0.5)
                    else:
                        ctx["features"] = state_features(
                            hour=rep.get("hour", hour), demand_ratio=b_dom / max(1e-9, dom_nom[d]),
                            level=cur, n_actions=n_actions, remaining_budget=budget,
                            voltage_dev=vdev, exceedance=exc)
                    if fmask is not None:
                        ctx["features"] = ctx["features"] * fmask
                ctx["admissible"] = adm_dom_now.get(d, tuple(range(n_actions)))
                a = int(policy.cloud_ctl.decide(ctx))
                if policy.explore_eps > 0 and rng.random() < policy.explore_eps:
                    a = (int(rng.choice(ctx["admissible"]))
                         if policy.explore_admissible
                         else int(rng.integers(0, n_actions)))
                cloud_actions[d] = a
                if (collect_transitions and collect_for == "cloud"
                        and policy.cloud_kind in ("linear", "concatenated")):
                    # Only the learned kinds carry a feature vector; a rule or
                    # minimal-sufficient controller has none, so asking for one
                    # here would raise KeyError instead of recording nothing.
                    feat_cloud_now[d] = ctx["features"]

        local_actions: dict[str, int] = {}
        adm_now: dict = {}
        if policy.local_kind is not None:
            for g in gids:
                d = dom_of[g]
                b_est = p_obs[g] * inv_mults[level[g]]
                exc = 0.0
                if in_dr_now and b_est > 0:
                    exc = max(0.0, b_est - cap_kw[g]) / (flex * max_level * b_est)
                dvm = domain_vmin[d]
                vdev = ((1.0 - dvm) / 0.05) if math.isfinite(dvm) else 0.0
                rem = max(0.0, 1.0 - burden_used[g] / max(1e-9, burden_cap[g]))
                # Successor-state admissible set (v2 fix F-3): evaluated at the
                # DECISION state (pre-actuation level, dwell timer and budget),
                # which is exactly the state the emitted transition records as X2
                # one tick later. Computed BEFORE decide() so a masked learned
                # controller can restrict its greedy choice to it.
                adm_g = _admissible_levels(
                    cur=level[g], since_change_s=t - last_change_t[g],
                    budget_left_kwh=burden_cap[g] - burden_used[g],
                    base_kw=nom_kw[g] * shape[k], flex=flex, levels=levels,
                    tick_s=tick_s,
                    ramp_max_step=protocol.ramp_max_level_step,
                    dwell_s=protocol.dwell_time_s, n_actions=n_actions)
                ctx = {"exceedance": exc, "voltage_dev": vdev,
                       "remaining_budget": rem, "current_level": level[g],
                       "hour": hour, "admissible": tuple(adm_g)}
                if policy.local_kind in ("linear", "concatenated"):
                    since_ticks = int((t - last_change_t[g]) // tick_s)
                    if use_binned:
                        ctx["features"] = state_features_binned(
                            exceedance=exc, level=level[g],
                            since_change_ticks=since_ticks,
                            remaining_budget=rem, in_dr=bool(in_dr_now))
                    elif use_v2_feat:
                        dwell_frac = (min(3, since_ticks) / 3.0
                                      if dwell_feature else 0.5)
                        ctx["features"] = state_features_v2(
                            hour=hour, demand_ratio=b_est / max(1e-9, nom_kw[g]),
                            level=level[g], n_actions=n_actions,
                            remaining_budget=rem, exceedance=exc,
                            dwell_frac=dwell_frac)
                    else:
                        ctx["features"] = state_features(
                            hour=hour, demand_ratio=b_est / max(1e-9, nom_kw[g]),
                            level=level[g], n_actions=n_actions, remaining_budget=rem,
                            voltage_dev=vdev, exceedance=exc)
                    if fmask is not None:
                        ctx["features"] = ctx["features"] * fmask
                a = int(policy.local_ctl.decide(ctx))
                if policy.explore_eps > 0 and rng.random() < policy.explore_eps:
                    a = (int(rng.choice(adm_g)) if policy.explore_admissible
                         else int(rng.integers(0, n_actions)))
                local_actions[g] = a
                if collect_transitions and policy.local_kind in ("linear", "concatenated"):
                    # Same guard as the cloud site: non-learned kinds have no
                    # feature vector, so recording one would raise KeyError.
                    feat_now[g] = ctx["features"]
                    adm_now[g] = tuple(adm_g)

        t_inf1 = time.perf_counter()
        res.policy_inference_s += (t_inf1 - t_inf0)

        # ---- route through the arbiter ---------------------------------------
        for d in doms:
            arb = arbiters[d]
            members = dom_groups[d]
            delivered = bool(c2e_ok[k])
            vec = None
            if arb.authority == "cloud" and policy.cloud_kind is not None and delivered:
                seq["cloud"] += 1
                cmd = Command(source="cloud",
                              requested_action={g: cloud_actions.get(d, 0) for g in members},
                              authority_epoch=cloud_epoch_known[d],   # cloud uses the epoch it knows
                              sequence_number=seq["cloud"],
                              creation_time=t, expiry_time=t + lease_s, domain_id=d)
                ok, _why = arb.accept(cmd, t, strict=strict)
                res.n_commands_offered += 1
                latency_samples.append(float(trace.c2e_arrival[k] - t))
                if record_events:
                    event_log.append({
                        "t": float(t), "domain_id": d,
                        "kind": "cloud_command", "detail": _why,
                        "accepted": bool(ok), "epoch": int(cmd.authority_epoch),
                        "authority": arb.authority, "sequence": int(cmd.sequence_number),
                    })
                if ok:
                    res.n_commands_accepted += 1
                    vec = dict(cmd.requested_action)
                    last_cmd[d] = cmd
                else:
                    res.n_commands_rejected += 1
                # a duplicated old copy of the previous command arrives too (F9)
                if c2e_dup[k] and d in last_cmd:
                    ok2, _why2 = arb.accept(last_cmd[d], t, strict=strict)
                    res.n_commands_offered += 1
                    if record_events:
                        event_log.append({
                            "t": float(t), "domain_id": d,
                            "kind": "cloud_command_duplicate", "detail": _why2,
                            "accepted": bool(ok2), "epoch": int(last_cmd[d].authority_epoch),
                            "authority": arb.authority,
                            "sequence": int(last_cmd[d].sequence_number),
                        })
                    if ok2:
                        res.n_commands_accepted += 1
                    else:
                        res.n_commands_rejected += 1
            elif arb.authority == "local" and policy.local_kind is not None:
                seq["local"] += 1
                cmd = Command(source="local",
                              requested_action={g: local_actions.get(g, 0) for g in members},
                              authority_epoch=arb.epoch, sequence_number=seq["local"],
                              creation_time=t, expiry_time=t + lease_s, domain_id=d)
                ok, _why = arb.accept(cmd, t, strict=strict)
                res.n_commands_offered += 1
                if record_events:
                    event_log.append({
                        "t": float(t), "domain_id": d,
                        "kind": "local_command", "detail": _why,
                        "accepted": bool(ok), "epoch": int(arb.epoch),
                        "authority": arb.authority, "sequence": int(cmd.sequence_number),
                    })
                if ok:
                    res.n_commands_accepted += 1
                    vec = dict(cmd.requested_action)
                else:
                    res.n_commands_rejected += 1
            if vec is None:
                # no delivered command and no local policy for this domain:
                # hold the last valid set-point until its documented expiry,
                # then use the common predeclared default (action 0).
                vec = _held_or_default(arb, t, members)
            for g in members:
                proposed[g] = int(vec.get(g, 0))

        # ---- constraints + actuation -----------------------------------------
        # Proposed actions (what the authority asked for) and executed actions
        # (what the actuator constraints allowed) are recorded separately.
        executed = {}
        for g in gids:
            cur = level[g]
            req = _clip_int(int(proposed[g]), 0, n_actions - 1)
            if req - cur > protocol.ramp_max_level_step:
                req = cur + protocol.ramp_max_level_step
            elif cur - req > protocol.ramp_max_level_step:
                req = cur - protocol.ramp_max_level_step
            if t - last_change_t[g] < protocol.dwell_time_s and req != cur:
                req = cur
            b = nom_kw[g] * shape[k]
            req = _max_level_within_budget(req, burden_cap[g] - burden_used[g],
                                           b, flex, levels, tick_s)
            executed[g] = _clip_int(req, 0, n_actions - 1)
            if executed[g] != cur:
                last_change_t[g] = t
                res.n_switches += 1
                if outage_focus[k]:
                    res.outage_switches += 1
            level[g] = executed[g]

        # The *executed* domain action: the level the domain actually applied
        # after the actuator constraints. The cloud learner must be given this,
        # not the raw commanded level, otherwise its (s, a, r) triple mixes a
        # command that was never applied with the reward of the action that was.
        # (The local learner already uses the executed per-group level.)
        cloud_exec = {d: int(round(sum(executed[g] for g in dom_groups[d])
                                   / len(dom_groups[d]))) for d in doms}

        # ---- physics ---------------------------------------------------------
        demand, demand_q = {}, {}
        curt_step_kw = 0.0
        for g in gids:
            b = nom_kw[g] * shape[k]
            m = mults[level[g]]
            demand[g] = b * m
            demand_q[g] = base_kvar[g] * scale * shape[k] * m
            curt_g = b * flex * levels[level[g]]
            curt_step_kw += curt_g
            inc_kwh = curt_g * kwh_per_tick
            burden_used[g] += inc_kwh
            res.curtail_energy_kwh += inc_kwh
        if record_series:
            series_mean_level.append(sum(level.values()) / len(gids))
            series_curt_kw.append(curt_step_kw)
            series_proposed.append(sum(proposed.values()) / len(gids))
        feeder.set_loads(demand, demand_q)
        sol = feeder.solve()
        res.n_steps_solved += 1
        if not sol.converged:
            res.n_convergence_failures += 1

        p_obs = dict(demand)
        domain_vmin = {d: sol.domain_vmin_pu.get(d, float("nan")) for d in doms}

        # ---- metrics ---------------------------------------------------------
        base_total = sum(nom_kw[g] * shape[k] for g in gids)
        act_total = sum(demand[g] for g in gids)
        cap_total = sum(cap_kw.values())
        res.baseline_energy_kwh += base_total * kwh_per_tick
        res.actual_energy_kwh += act_total * kwh_per_tick
        if outage_focus[k]:
            res.outage_baseline_energy_kwh += base_total * kwh_per_tick
            res.outage_actual_energy_kwh += act_total * kwh_per_tick
        if dr[k]:
            res.eligible_steps += 1
            res.sum_exceedance += max(0.0, act_total - cap_total) / max(1e-9, flex * max_level * base_total)
            if act_total <= cap_total + 1e-9:
                res.achieved_steps += 1
                if outage_focus[k]:
                    res.outage_achieved_steps += 1
            if outage_focus[k]:
                res.outage_eligible_steps += 1
        if sol.converged:
            vmin, vmax = sol.vmin_pu, sol.vmax_pu
            if math.isfinite(vmin) and (not math.isfinite(res.vmin_pu) or vmin < res.vmin_pu):
                res.vmin_pu, res.vmin_node = vmin, sol.vmin_node
            if math.isfinite(vmax) and (not math.isfinite(res.vmax_pu) or vmax > res.vmax_pu):
                res.vmax_pu = vmax
            if vmin < ansi_lo or vmax > ansi_hi:
                res.voltage_violation_s += tick_s
            ml = sol.max_line_loading_pu
            if math.isfinite(ml) and (not math.isfinite(res.max_line_loading_pu)
                                      or ml > res.max_line_loading_pu):
                res.max_line_loading_pu, res.worst_line = ml, sol.worst_line
            if ml > loading_lim:
                res.line_violation_s += tick_s

        # ---- cloud state report (arrives over the simulated link) ------------
        if e2c_ok[k]:
            for d in doms:
                # the cloud learns the arbiter's current authority epoch from the
                # report it actually received, so its knowledge lags the transfer
                cloud_epoch_known[d] = arbiters[d].epoch
                mem = dom_groups[d]
                nmem = len(mem)
                bsum = 0.0
                rsum = 0.0
                lsum = 0.0
                for g in mem:
                    bsum += p_obs[g] * inv_mults[level[g]]
                    rsum += max(0.0, 1.0 - burden_used[g] / max(1e-9, burden_cap[g]))
                    lsum += level[g]
                dvm = domain_vmin[d]
                cloud_report[d] = {
                    "base_est": {g: p_obs[g] * inv_mults[level[g]] for g in mem},
                    "voltage_dev": ((1.0 - dvm) / 0.05) if math.isfinite(dvm) else 0.0,
                    "remaining_budget": rsum / nmem,
                    "current_level": int(round(lsum / nmem)),
                    "hour": hour,
                }

        if collect_transitions and policy.local_kind in ("linear", "concatenated"):
            # emit the previous step's transition now that its successor state is known
            r_now: dict = {}
            for g in gids:
                b = nom_kw[g] * shape[k]
                r_now[g] = _learn_reward(
                    net_kw=demand[g], baseline_kw=b,
                    target_kw=(cap_kw[g] if in_dr_now else b),
                    curtailed_kw=b * flex * levels[level[g]],
                    level_changed=(level[g] != prev_level[g]),
                    in_dr=bool(in_dr_now))
            for g in ([gids[0]] if collect_repr else gids):
                if g in prev_x:
                    transitions.append((prev_x[g], prev_a[g], prev_r[g],
                                        feat_now.get(g, prev_x[g]), False, prev_hour[g],
                                        adm_now.get(g, tuple(range(n_actions)))))
            prev_x = {g: feat_now[g] for g in gids if g in feat_now}
            prev_a = {g: int(level[g]) for g in gids}
            prev_r = dict(r_now)
            prev_level = {g: int(level[g]) for g in gids}
            prev_hour = {g: hour for g in gids}

        if collect_transitions and collect_for == "cloud":
            r_dom: dict = {}
            for d in doms:
                members = dom_groups[d]
                r_dom[d] = float(np.mean([
                    _learn_reward(net_kw=demand[g], baseline_kw=nom_kw[g] * shape[k],
                                  target_kw=(cap_kw[g] if in_dr_now else nom_kw[g] * shape[k]),
                                  curtailed_kw=nom_kw[g] * shape[k] * flex * levels[level[g]],
                                  level_changed=(level[g] != prev_cloud_level.get(g, 0)),
                                  in_dr=bool(in_dr_now))
                    for g in members]))
            for d in ([doms[0]] if collect_repr else doms):
                if d in prev_cloud_x:
                    transitions.append((prev_cloud_x[d], prev_cloud_a[d], prev_cloud_r[d],
                                        feat_cloud_now.get(d, prev_cloud_x[d]), False,
                                        prev_hour_d.get(d, hour),
                                        adm_dom_now.get(d, tuple(range(n_actions)))))
            prev_cloud_x = dict(feat_cloud_now)
            prev_cloud_a = dict(cloud_exec)
            prev_cloud_r = dict(r_dom)
            prev_cloud_level = {g: int(level[g]) for g in gids}
            prev_hour_d = {d: hour for d in doms}

    # ---- finalise -----------------------------------------------------------
    if collect_transitions and policy.local_kind in ("linear", "concatenated"):
        for g in ([gids[0]] if collect_repr else gids):
            if g in prev_x:
                transitions.append((prev_x[g], prev_a[g], prev_r[g], prev_x[g], True,
                                    prev_hour[g], tuple(range(n_actions))))
    if collect_transitions and collect_for == "cloud":
        for d in ([doms[0]] if collect_repr else doms):
            if d in prev_cloud_x:
                transitions.append((prev_cloud_x[d], prev_cloud_a[d], prev_cloud_r[d],
                                    prev_cloud_x[d], True, prev_hour_d.get(d, 0.0),
                                    tuple(range(n_actions))))
    res.burden_used_fraction = sum(burden_used.values()) / max(1e-9, sum(burden_cap.values()))
    tot = time_local + time_cloud
    res.frac_time_local_authority = time_local / max(1, tot)
    res.frac_time_cloud_authority = time_cloud / max(1, tot)
    res.n_authority_switches = sum(a.n_transfers for a in arbiters.values())
    for a in arbiters.values():
        for key, v in a.reject_reasons.items():
            res.reject_reasons[key] = res.reject_reasons.get(key, 0) + v
    if latency_samples:
        res.mean_command_latency_s = float(np.mean(latency_samples))
    if detect_delays:
        res.heartbeat_detect_delay_s = float(np.mean(detect_delays))
    if transfer_delays:
        res.authority_transfer_delay_s = float(np.mean(transfer_delays))
    if recovery_delays:
        res.recovery_delay_s = float(np.mean(recovery_delays))
    res.wall_s = time.perf_counter() - t_wall0
    res.extra = {"n_domains": len(doms), "s_ref": s_ref, "dr_fraction": float(dr.mean()),
                 "cache_max_age_s": protocol.cache_max_age_s}
    if record_series:
        res.extra["mean_level_series"] = series_mean_level
        res.extra["curt_kw_series"] = series_curt_kw
        res.extra["mean_proposed_series"] = series_proposed
    if record_events:
        res.extra["events"] = event_log
    return res, (transitions if collect_transitions else None)
