"""Value-function representations and batch fitted-Q fitting.

Two representations are implemented, exactly as the task prompt specifies:

1. **Corrected action-dependent linear model** (the proposed model)::

       x(s) = [1, s^T]^T
       Q(s, a) = theta_a^T x(s)

   Coefficients are *action specific*, so ``argmax_a Q(s, a)`` depends on ``s``.

2. **Concatenated-feature ablation** (diagnostic only)::

       Q(s, a) = u^T s + v_a

   Because ``Q(s,a) - Q(s,b) = v_a - v_b``, the greedy ranking is independent of
   the state at fixed weights. This is a diagnostic ablation, **not** a DQN, and
   is never labelled as one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- state feature layout ----------------------------------------------------
# All eight features are locally observable at the edge controller. There is no
# fabricated temperature feature (the task prompt forbids it and no matched
# temperature series exists) and no invented frequency signal: the stress proxy
# is the observed per-unit voltage deviation.
FEATURE_NAMES = [
    "bias",
    "sin_tod",
    "cos_tod",
    "demand_ratio",       # reconstructed uncontrolled demand / nominal
    "current_level",      # currently applied action level
    "remaining_budget",   # fraction of the daily curtailment budget still unused
    "voltage_dev",        # (1 - vmin_local) / 0.05, an observable stress proxy
    "exceedance",         # normalised shortfall of reconstructed demand vs the DR cap
]
STATE_DIM = len(FEATURE_NAMES)

# ---- v2 state feature layout -------------------------------------------------
# Same dimensionality, so a v2 model is a drop-in replacement for a v1 model.
# The only structural change is slot 6: the voltage-deviation proxy is DROPPED
# (it is a feeder-derived signal the edge does not own, and v2 fixes the learner
# to locally observable quantities only) and replaced either by a constant or by
# the dwell feature -- the minutes since the last level change, capped at 3.
FEATURE_NAMES_V2 = [
    "bias",
    "sin_tod",
    "cos_tod",
    "demand_ratio",
    "current_level",
    "remaining_budget",
    "dwell_frac",         # min(3, minutes since last level change) / 3, or 0.5
    "exceedance",
]


def state_features_v2(*, hour: float, demand_ratio: float, level: int,
                      n_actions: int, remaining_budget: float,
                      exceedance: float, dwell_frac: float = 0.5) -> np.ndarray:
    """v2 state vector: identical layout to :func:`state_features` except slot 6.

    ``dwell_frac`` must be ``min(3, minutes_since_last_level_change) / 3`` when
    the dwell feature is enabled, and the constant ``0.5`` when it is not (the
    sandbox's ``raw`` layout). All inputs are locally observable at the edge.
    """
    x = np.empty(STATE_DIM, dtype=float)
    x[0] = 1.0
    ang = 2.0 * math.pi * hour / 24.0
    x[1] = math.sin(ang)
    x[2] = math.cos(ang)
    x[3] = _clamp(demand_ratio, 0.0, 3.0) / 3.0
    x[4] = level / max(1, n_actions - 1)
    x[5] = _clamp(remaining_budget, 0.0, 1.0)
    x[6] = _clamp(dwell_frac, 0.0, 1.0)
    x[7] = _clamp(exceedance, 0.0, 3.0) / 3.0
    return x


# ---- binned ("one-hot") layout ----------------------------------------------
# The representation selected by the pre-registered S3 sweep. It is a faithful
# port of the sandbox `Feat('binned')` builder with `use_pace=False`: the
# look-ahead "pace" block is a future-information leak and is never built.
#
#   slot 0                bias
#   1  .. 31              one-hot over 31 exceedance bins (edges below)
#   32 .. 34              one-hot over the current level (3 actions)
#   35 .. 36              one-hot over (minutes since last change >= 3)
#   37 .. 46              one-hot over deciles of the remaining budget fraction
#   47                    the DR-window flag
#   48                    exceedance gated by "level == 2"
#
# There is deliberately no explicit time-of-day term: in the sandbox layout the
# clock enters only through the DR flag. This is recorded as an observation in
# docs/v2_stage_S3_report.md rather than "fixed", because the representation was
# chosen by the frozen selection rule before its properties were inspected.
EXC_EDGES = np.r_[np.arange(0.0, 1.5, 0.05), 1.5, 9.0]
BINNED_DIM = 1 + (len(EXC_EDGES) - 1) + 3 + 2 + 10 + 1 + 1


def state_features_binned(*, exceedance: float, level: int, since_change_ticks: int,
                          remaining_budget: float, in_dr: bool) -> np.ndarray:
    """The selected binned state vector (``BINNED_DIM`` = 49)."""
    x = np.zeros(BINNED_DIM, dtype=float)
    x[0] = 1.0
    i = 1
    nb = len(EXC_EDGES) - 1
    exc = _clamp(exceedance, 0.0, 9.0)
    x[i + min(nb - 1, max(0, int(np.searchsorted(EXC_EDGES, exc, 'right')) - 1))] = 1.0
    i += nb
    x[i + int(level)] = 1.0
    i += 3
    x[i + int(since_change_ticks >= 3)] = 1.0
    i += 2
    lf = _clamp(remaining_budget, 0.0, 1.0)
    x[i + min(9, int(lf * 10))] = 1.0
    i += 10
    x[i] = 1.0 if in_dr else 0.0
    i += 1
    x[i] = exc if level == 2 else 0.0
    return x


def feature_dim(variant: str) -> int:
    """State dimension of a feature variant, so models are built at the right size."""
    return BINNED_DIM if variant == "v2_binned" else STATE_DIM


def _clamp(v: float, lo: float, hi: float) -> float:
    """Scalar clamp without NumPy scalar overhead (hot path)."""
    return lo if v < lo else (hi if v > hi else v)


def state_features(*, hour: float, demand_ratio: float, level: int, n_actions: int,
                   remaining_budget: float, voltage_dev: float,
                   exceedance: float) -> np.ndarray:
    """Build the documented state vector.

    All inputs are locally observable at the edge controller. No future workload
    or future fault information is used.

    Implemented with ``math`` and built-in comparisons rather than NumPy scalar
    operations: this function is called once per group per control step
    (50 x 1440 = 72,000 calls per episode), where NumPy scalar overhead
    dominated the profile.
    """
    x = np.empty(STATE_DIM, dtype=float)
    x[0] = 1.0
    ang = 2.0 * math.pi * hour / 24.0
    x[1] = math.sin(ang)
    x[2] = math.cos(ang)
    x[3] = _clamp(demand_ratio, 0.0, 3.0) / 3.0
    x[4] = level / max(1, n_actions - 1)
    x[5] = _clamp(remaining_budget, 0.0, 1.0)
    x[6] = _clamp(voltage_dev, 0.0, 2.0)
    x[7] = _clamp(exceedance, 0.0, 3.0) / 3.0
    return x


class ValueModel:
    """Least-squares linear value model with two representation kinds."""

    def __init__(self, n_actions: int, kind: str = "action_dependent",
                 state_dim: int = STATE_DIM, seed: int = 0):
        if kind not in ("action_dependent", "concatenated"):
            raise ValueError(kind)
        self.n_actions = int(n_actions)
        self.kind = kind
        self.state_dim = int(state_dim)
        self.rng = np.random.default_rng(seed)
        if kind == "action_dependent":
            self.n_params = self.n_actions * self.state_dim
            self.theta = np.zeros(self.n_params, dtype=float)
            self._W = self.theta.reshape(self.n_actions, self.state_dim)
        else:
            self.n_params = self.state_dim + self.n_actions
            self.theta = np.zeros(self.n_params, dtype=float)
        # deterministic low-action tie break, precomputed once
        self._tiebreak = 1e-9 * np.arange(self.n_actions, dtype=float)[::-1]

    # ------------------------------------------------------------- design
    def design_row(self, x: np.ndarray, a: int) -> np.ndarray:
        if self.kind == "action_dependent":
            row = np.zeros(self.n_params, dtype=float)
            row[a * self.state_dim:(a + 1) * self.state_dim] = x
            return row
        row = np.zeros(self.n_params, dtype=float)
        row[:self.state_dim] = x
        row[self.state_dim + a] = 1.0
        return row

    def design(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], self.n_params), dtype=float)
        for i in range(X.shape[0]):
            out[i] = self.design_row(X[i], int(A[i]))
        return out

    # ---------------------------------------------------------- prediction
    def q_values(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "action_dependent":
            return self._W @ x
        base = float(self.theta[:self.state_dim] @ x)
        return base + self.theta[self.state_dim:]

    def q_batch(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self.q_values(X[i]) for i in range(X.shape[0])])

    def greedy(self, x: np.ndarray) -> int:
        return int(np.argmax(self.q_values(x)))

    def greedy_tiebreak(self, x: np.ndarray) -> int:
        """Greedy with a deterministic low-action tie break."""
        q = self.q_values(x)
        return int(np.argmax(q + self._tiebreak))

    # -------------------------------------------------------------- fitting
    def fit(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray,
            ridge: float = 1e-6) -> "ValueModel":
        D = self.design(X, A)
        DtD = D.T @ D + ridge * np.eye(self.n_params)
        DtY = D.T @ Y
        self.theta = np.linalg.solve(DtD, DtY)
        if self.kind == "action_dependent":
            self._W = self.theta.reshape(self.n_actions, self.state_dim)
        return self

    def copy(self) -> "ValueModel":
        m = ValueModel(self.n_actions, self.kind, self.state_dim)
        m.theta = self.theta.copy()
        if m.kind == "action_dependent":
            m._W = m.theta.reshape(m.n_actions, m.state_dim)
        return m


def fitted_q(model: ValueModel, *, sample_fn, n_iterations: int, gamma: float,
             n_samples: int, rng: np.random.Generator,
             epsilon_start: float = 0.30, epsilon_end: float = 0.05,
             terminal_no_bootstrap: bool = True) -> list[float]:
    """Batch fitted-Q iteration.

    ``sample_fn(model, epsilon, rng, n_samples)`` must return a tuple
    ``(X, A, R, X2, DONE)`` of transitions collected from the environment using
    an epsilon-greedy behaviour policy derived from ``model``. The trainer never
    sees the environment internals, which keeps the learning loop honest: it can
    only use the transitions it is given.
    """
    history: list[float] = []
    for it in range(n_iterations):
        frac = it / max(1, n_iterations - 1)
        eps = epsilon_start + (epsilon_end - epsilon_start) * frac
        X, A, R, X2, DONE = sample_fn(model, eps, rng, n_samples)
        # Bellman target
        Q2 = model.q_batch(X2)
        target = R + gamma * np.max(Q2, axis=1) * (1.0 - DONE if terminal_no_bootstrap
                                                   else 1.0)
        model.fit(X, A, target)
        pred = np.array([model.q_values(X[i])[int(A[i])] for i in range(len(A))])
        history.append(float(np.sqrt(np.mean((pred - target) ** 2))))
    return history
