"""Controllers C0..C5 (Phase 5).

Every controller implements the same interface::

    decide(ctx) -> action level in {0, 1, ..., n_actions-1}

where ``ctx`` is a plain dict of *locally available* quantities for a local
controller and *cloud-available* quantities for a cloud controller. No controller
ever receives future workload or future fault information.

* ``C0`` uncontrolled reference: always level 0.
* ``C1`` cloud rule policy.
* ``C2`` cloud action-dependent linear policy.
* ``C3`` local rule policy (guarded authority).
* ``C4`` local action-dependent linear policy (guarded authority).
* ``C5`` local concatenated-feature policy (diagnostic ablation, **not** a DQN).

The rule is a transparent dual-input hysteretic threshold rule on the
reconstructed uncontrolled demand and the local per-unit voltage (see
``docs/audit_resolution.md`` A4). Deadbands are wider than the largest actuation
step so the actuation itself cannot cross a release threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .value_model import ValueModel


# --------------------------------------------------------------------- rule ---
@dataclass
class RuleParams:
    th_low: float = 0.30      # normalised exceedance for level 1
    th_high: float = 0.70     # normalised exceedance for level 2
    hyst: float = 0.25        # release margin (must exceed the largest actuation step)
    v_thresh: float = 1.20    # voltage-deviation trigger (units of 0.05 pu)
    budget_guard: float = 0.0  # if remaining budget fraction <= this, force level 0

    def as_tuple(self) -> tuple:
        return (self.th_low, self.th_high, self.hyst, self.v_thresh, self.budget_guard)


class RuleController:
    """Dual-input hysteretic threshold rule."""

    name = "rule"

    def __init__(self, params: RuleParams, n_actions: int = 3):
        self.p = params
        self.n_actions = n_actions

    def decide(self, ctx: dict) -> int:
        exc = float(ctx["exceedance"])
        vdev = float(ctx.get("voltage_dev", 0.0))
        budget = float(ctx.get("remaining_budget", 1.0))
        cur = int(ctx.get("current_level", 0))

        if budget <= self.p.budget_guard:
            target = 0
        elif exc > self.p.th_high:
            target = 2
        elif exc > self.p.th_low:
            target = 1
        else:
            target = 0

        # hysteresis: do not step down unless the signal falls well below the
        # threshold that caused the step up
        if cur > target:
            margin = self.p.hyst
            stay = (exc > (self.p.th_high - margin) if cur == 2
                    else exc > (self.p.th_low - margin))
            if stay:
                target = cur

        # voltage override: a low local voltage is a stress signal -> curtail more
        if vdev > self.p.v_thresh:
            target = min(self.n_actions - 1, max(target, cur + (1 if cur < self.n_actions - 1 else 0)))

        if target < 0:
            return 0
        if target > self.n_actions - 1:
            return self.n_actions - 1
        return int(target)


# ----------------------------------------------------------------- learned ---
class LearnedController:
    """Wraps a fitted :class:`ValueModel`; greedy action with low-level tie break.

    ``mask_admissible`` (v2) restricts the greedy choice to the actions the
    actuator constraints (ramp, dwell, budget) would actually execute. It is the
    decision-time companion of the masked Bellman target: the target never
    teaches Q for actions the environment cannot take, so letting the policy pick
    one of those actions would read an unlearned Q value. The mask is a
    deterministic function of the constraint parameters already in the protocol,
    not a tunable, and it is applied identically to every learned arm.
    """

    def __init__(self, model: ValueModel, name: str = "learned",
                 mask_admissible: bool = False):
        self.model = model
        self.name = name
        self.mask_admissible = bool(mask_admissible)

    def decide(self, ctx: dict) -> int:
        x = ctx["features"]
        if self.mask_admissible:
            adm = ctx.get("admissible")
            if adm:
                q = self.model.q_values(x)
                return int(max(adm, key=lambda z: q[z] - 1e-9 * z))
        return self.model.greedy_tiebreak(x)


class NullController:
    name = "null"

    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions

    def decide(self, ctx: dict) -> int:
        return 0


class MinimalSufficientController:
    """C7: the lowest level that is sufficient, with no parameters.

    It curtails exactly as much as the *declared* external target requires and
    never more: it returns the smallest level ``L`` whose post-actuation demand
    meets the cap, with no look-ahead, no learning and no hysteresis. It is the
    "no-parameters" reference the v2 plan asks for, and it is deliberately not
    tuned on anything.

    Derivation. With ``exc = max(0, b - cap) / (flex * max_level * b)`` the
    condition ``b * (1 - flex * level_L) <= cap`` rearranges to
    ``level_L >= exc * max_level``. So the rule needs only the exceedance
    already in the context, and it never depends on a reconstructed kW figure.
    """

    name = "minimal_sufficient"

    def __init__(self, levels: list[float], flex: float, n_actions: int = 3):
        self.levels = list(levels)
        self.max_level = max(self.levels)
        self.flex = float(flex)
        self.n_actions = n_actions

    def decide(self, ctx: dict) -> int:
        exc = float(ctx.get("exceedance", 0.0))
        if exc <= 0.0:
            return 0
        need = exc * self.max_level
        for L in range(self.n_actions):
            if self.levels[L] + 1e-12 >= need:
                return int(L)
        return int(self.n_actions - 1)


# ------------------------------------------------------------ tuning grid ---
def default_rule_grid() -> list[RuleParams]:
    """Candidate rule configurations for validation tuning.

    The grid is deliberately small and *documented*: every candidate is scored on
    the validation split and the winner is frozen. The same tuning budget (number
    of candidate evaluations) is granted to the learned policy.
    """
    out = []
    for th_low in (0.20, 0.30, 0.40):
        for th_high in (0.60, 0.75, 0.90):
            for hyst in (0.20, 0.30):
                out.append(RuleParams(th_low=th_low, th_high=th_high,
                                      hyst=hyst, v_thresh=1.20, budget_guard=0.0))
    return out


def extended_rule_grid() -> list[RuleParams]:
    """v2 rule++ candidate set: the full 6 x 6 x 4 = 144 point enumeration.

    The v1 grid (``default_rule_grid``) held 18 points and its winner sat on the
    lower boundary, so the rule arm's tuning budget was never actually exhausted.
    v2 grants the rule family the same 144 evaluations the learned family gets
    (docs/v2_preregistration.md, "tuning budget"). All 144 points are returned --
    including the 12 with ``th_high <= th_low``, which are semantically degenerate
    (level 1 can never be reached, so the rule collapses to a single threshold) --
    so that "144 candidates" is literally true for both families and no candidate
    is silently dropped before scoring.
    """
    out = []
    for th_low in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        for th_high in (0.2, 0.3, 0.4, 0.5, 0.6, 0.75):
            for hyst in (0.0, 0.1, 0.2, 0.3):
                out.append(RuleParams(th_low=th_low, th_high=th_high,
                                      hyst=hyst, v_thresh=1.20, budget_guard=0.0))
    return out
