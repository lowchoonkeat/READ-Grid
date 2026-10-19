"""Locally-observable learning reward.

Design (Phase 5 of the task prompt)
-----------------------------------
* The reward contains **only** terms the edge controller can compute from its own
  measurements: one-sided target-tracking exceedance, incremental curtailment
  burden and switching.
* Physical / device constraints (line thermal loading, voltage limits) are **not**
  part of the reward. They are enforced outside the learner (ramp, dwell and the
  cumulative daily burden cap are enforced by the arbiter) and reported as
  metrics. This is the "keep physical/device constraints separate from the
  reward" requirement.
* Every term is normalised by the group's *current* maximum flexible curtailment
  ``DeltaP_flex = B_g(t) * flexible_fraction * max_level`` so the terms are
  comparable and dimensionless.

Tracking term
-------------
One-sided hinge on the *reconstructed uncontrolled demand* ``B_g(t)`` (see
``docs/audit_resolution.md`` A4/A6), not on the post-actuation net demand::

    track = max(0, P_net - T_g) / DeltaP_flex
    T_g   = the declared DR demand cap inside a DR window, else B_g(t)

Because the action set can only curtail, a symmetric penalty would charge the
controller for an unachievable positive deviation. Outside a DR window the cap
is set to the uncontrolled demand itself, so the tracking term is identically
zero and no curtailment is ever rewarded off-window.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardTerms:
    track: float
    burden: float
    switch: float
    total: float


def step_reward(*, net_kw: float, baseline_kw: float, target_kw: float,
                curtailed_kw: float, level_changed: bool,
                flexible_fraction: float, max_level: float,
                w_track: float, w_burden: float, w_switch: float) -> RewardTerms:
    flex_cap = baseline_kw * flexible_fraction * max_level
    if flex_cap <= 1e-9:
        return RewardTerms(0.0, 0.0, 0.0, 0.0)

    track = max(0.0, net_kw - target_kw) / flex_cap
    burden = max(0.0, curtailed_kw) / flex_cap
    switch = 1.0 if level_changed else 0.0

    total = -(w_track * track + w_burden * burden + w_switch * switch)
    return RewardTerms(float(track), float(burden), float(switch), float(total))


def step_reward_achievement(*, net_kw: float, baseline_kw: float, target_kw: float,
                            curtailed_kw: float, level_changed: bool,
                            flexible_fraction: float, max_level: float,
                            in_dr: bool, w_achievement: float, w_curtail: float,
                            w_switch: float) -> RewardTerms:
    """Achievement-indicator reward (v2 option, ``reward.mode: achievement``).

    The tracking hinge is replaced by a *binary* indicator that pays only when
    the post-actuation net demand is at or below the declared cap inside a DR
    window, penalised by the same normalised curtailment burden and switching
    count the hinge reward uses::

        r = w_achievement * 1[in_dr and P_net <= T] - w_curtail * burden - w_switch * switch

    ``burden`` is normalised exactly as in :func:`step_reward`, so the curtailment
    price ``w_curtail`` is on the same scale in both modes. This is the reward
    family the sandbox S3 sweep labels ``ach c.<price>``.
    """
    flex_cap = baseline_kw * flexible_fraction * max_level
    if flex_cap <= 1e-9:
        return RewardTerms(0.0, 0.0, 0.0, 0.0)

    achieved = 1.0 if (in_dr and net_kw <= target_kw + 1e-9) else 0.0
    burden = max(0.0, curtailed_kw) / flex_cap
    switch = 1.0 if level_changed else 0.0

    total = w_achievement * achieved - w_curtail * burden - w_switch * switch
    return RewardTerms(float(achieved), float(burden), float(switch), float(total))
