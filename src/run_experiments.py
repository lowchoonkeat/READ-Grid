"""Parallel experiment engine for the READ-Grid rebuild.

One worker process owns one OpenDSS circuit and one copy of the workload
library; jobs are described by plain dictionaries so they can be dispatched with
``multiprocessing.Pool``. Results are appended to a JSONL file as they complete,
so a long run can be inspected (and resumed) while it is still going.

Stages
------
``rule_tune``   validation-only grid search over the transparent rule
``train``       fitted-Q training of the learned value models (train split only)
``validate``    validation-only scoring of the trained policies
``main``        the frozen held-out evaluation (test split, F0-F10)
``ablation``    Phase 11 ablation variants
``sensitivity`` Phase 9 one-factor-at-a-time sensitivity

Nothing in this module touches the test split except the ``main`` stage, and
that stage is only launched after the protocol has been frozen.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root, derived from this file (portable)
sys.path.insert(0, str(ROOT / "src"))

from readgrid.comm import generate_fault_trace  # noqa: E402
from readgrid.controllers import (LearnedController, MinimalSufficientController,
                                  RuleController, RuleParams)  # noqa: E402
from readgrid.feeder import FeederModel  # noqa: E402
from readgrid.mapping import build_mapping  # noqa: E402
from readgrid.protocol import Protocol, load_protocol  # noqa: E402
from readgrid.sim import PolicySpec, run_simulation  # noqa: E402
from readgrid.value_model import STATE_DIM, ValueModel, feature_dim  # noqa: E402
from readgrid.workload import load_episodes  # noqa: E402

RUNS = ROOT / "runs"
MODELS = ROOT / "models"
SPLIT_JSON = RUNS / "workload_split.json"

_CTX: dict = {}


# --------------------------------------------------------------------------- #
# worker side
# --------------------------------------------------------------------------- #
def worker_init() -> None:
    # ``P359_PROTOCOL`` lets a v2 run use configs/protocol_v2.yaml as its base
    # while leaving the v1 default untouched. The override mechanism then layers
    # the frozen learner configuration on top, and every result row records both
    # the base hash and the override.
    proto = load_protocol(os.environ.get("P359_PROTOCOL") or None)
    groups = build_mapping()
    _CTX.clear()
    _CTX.update({
        "protocol": proto,
        "groups": groups,
        "feeder": FeederModel(groups, proto),
        "episodes": load_episodes(),
        "traces": {},
        "models": {},
        "proto_overrides": {},
    })


class Always:
    def __init__(self, level: int) -> None:
        self.level = level
        self.name = f"always{level}"

    def decide(self, ctx):
        return self.level


class Probe:
    """Returns a fixed level inside DR windows and 0 otherwise.

    Used by sensitivity runs that need a *controlled* behaviour rather than a
    tuned policy.
    """

    def __init__(self, level: int, on_exceedance: float = 0.0) -> None:
        self.level = level
        self.th = on_exceedance
        self.name = "probe"

    def decide(self, ctx):
        return self.level if float(ctx.get("exceedance", 0.0)) > self.th else 0


def _get_trace(fault_id: str, seed: int):
    key = (fault_id, seed)
    tr = _CTX["traces"].get(key)
    if tr is None:
        proto = _CTX["protocol"]
        tr = generate_fault_trace(fault_id, proto.raw["faults"][fault_id], proto, seed)
        _CTX["traces"][key] = tr
    return tr


def protocol_with_override(proto: Protocol, override: dict | None) -> Protocol:
    """Return a copy of ``proto`` with dotted-path keys replaced.

    Used only by the Phase 9 one-factor-at-a-time sensitivity stage. The
    override is recorded on every result row so a reader can see exactly which
    frozen value was perturbed. The feeder does not depend on any of the keys
    that are perturbed (it reads only the ``feeder`` section), so no circuit
    rebuild is required.
    """
    if not override:
        return proto
    key = json.dumps(override, sort_keys=True)
    cache = _CTX.setdefault("proto_overrides", {})
    cached = cache.get(key)
    if cached is not None:
        return cached
    import copy
    import dataclasses
    raw = copy.deepcopy(proto.raw)
    for dotted, value in override.items():
        node = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    new = dataclasses.replace(proto, raw=raw)
    cache[key] = new
    return new


def _make_ctl(spec):
    if spec is None:
        return None
    kind = spec["type"]
    n_actions = _CTX["protocol"].n_actions
    if kind == "rule":
        return RuleController(RuleParams(*spec["params"]), n_actions)
    if kind == "minimal_sufficient":
        return MinimalSufficientController(_CTX["protocol"].action_levels,
                                           _CTX["protocol"].flexible_fraction,
                                           n_actions)
    if kind == "always":
        return Always(int(spec["level"]))
    if kind == "probe":
        return Probe(int(spec["level"]), float(spec.get("on_exceedance", 0.0)))
    if kind == "model":
        path = str(spec["path"])
        mask_adm = bool(spec.get("mask_admissible", False))
        cache_key = f"{path}|mask={mask_adm}"
        cached = _CTX["models"].get(cache_key)
        if cached is None:
            vm = ValueModel(n_actions, spec["vm"],
                            state_dim=int(spec.get("state_dim", STATE_DIM)), seed=0)
            vm.theta = np.load(path)
            if vm.kind == "action_dependent":
                vm._W = vm.theta.reshape(vm.n_actions, vm.state_dim)
            cached = LearnedController(vm, spec.get("name", "learned"),
                                       mask_admissible=mask_adm)
            _CTX["models"][cache_key] = cached
        return cached
    raise ValueError(f"unknown controller type {kind!r}")


def run_job(job: dict) -> dict:
    """Execute one simulation job and return a flat result row."""
    t0 = time.perf_counter()
    try:
        proto = protocol_with_override(_CTX["protocol"], job.get("protocol_override"))
        groups = _CTX["groups"]
        ep = _CTX["episodes"][job["episode_id"]]
        tr = _get_trace(job["fault_id"], int(job.get("comm_seed", 0)))
        cloud = _make_ctl(job.get("cloud"))
        local = _make_ctl(job.get("local"))
        pol = PolicySpec(
            policy_id=job["policy_id"],
            cloud_kind=job.get("cloud_kind"),
            local_kind=job.get("local_kind"),
            cloud_ctl=cloud,
            local_ctl=local,
            guard=job.get("guard", "guarded"),
            cache_enforced=bool(job.get("cache_enforced", True)),
            explore_eps=float(job.get("explore_eps", 0.0)),
            label=job.get("policy_label", job["policy_id"]),
            feature_mask=(tuple(job["feature_mask"]) if job.get("feature_mask")
                          else None),
        )
        res, _ = run_simulation(
            feeder=_CTX["feeder"], groups=groups, protocol=proto, episode=ep,
            scale=float(job.get("scale", 1.0)), trace=tr, policy=pol,
            rng_seed=int(job.get("rng_seed", 0)),
        )
        row = res.to_row()
        row["ok"] = True
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001
        row = {"policy_id": job.get("policy_id", "?"),
               "episode_id": job.get("episode_id", "?"),
               "fault_id": job.get("fault_id", "?"),
               "comm_seed": job.get("comm_seed", -1),
               "scale": job.get("scale", float("nan")),
               "ok": False,
               "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()}
    for k in ("stage", "policy_id", "policy_label", "episode_id", "scale", "fault_id",
              "comm_seed", "rng_seed", "train_seed", "guard", "cache_enforced",
              "variant", "factor", "factor_level", "seed_role"):
        if k in job:
            row[k] = job[k]
    if job.get("protocol_override"):
        row["protocol_override"] = json.dumps(job["protocol_override"], sort_keys=True)
    if job.get("feature_mask"):
        row["feature_mask"] = json.dumps(job["feature_mask"])
    row["job_wall_s"] = time.perf_counter() - t0
    return row


# --------------------------------------------------------------------------- #
# driver side
# --------------------------------------------------------------------------- #
class ResultWriter:
    """Append-only JSONL writer with a live progress counter."""

    def __init__(self, path: Path, total: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fh = open(path, "w", encoding="utf-8")
        self.total = total
        self.n = 0
        self.n_err = 0
        self.t0 = time.perf_counter()

    def write(self, row: dict) -> None:
        self.fh.write(json.dumps(row, default=_json_default) + "\n")
        self.n += 1
        if not row.get("ok", True):
            self.n_err += 1
        if self.n % 25 == 0 or self.n == self.total:
            self.fh.flush()
            el = time.perf_counter() - self.t0
            rate = self.n / el if el > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else float("inf")
            print(f"    {self.n}/{self.total}  {rate:.1f} jobs/s  "
                  f"elapsed {el:.0f}s  eta {eta:.0f}s  errors {self.n_err}",
                  flush=True)

    def close(self) -> None:
        self.fh.flush()
        self.fh.close()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return str(o)


def dispatch(jobs: list[dict], out_path: Path, *, workers: int, chunk: int = 4) -> dict:
    total = len(jobs)
    writer = ResultWriter(out_path, total)
    print(f"  dispatching {total} jobs to {workers} workers -> {out_path.name}", flush=True)
    t0 = time.perf_counter()
    with Pool(processes=workers, initializer=worker_init) as pool:
        for row in pool.imap_unordered(run_job, jobs, chunksize=chunk):
            writer.write(row)
    writer.close()
    el = time.perf_counter() - t0
    print(f"  done: {writer.n} rows, {writer.n_err} errors, {el:.0f}s "
          f"({writer.n / max(el, 1e-9):.1f} jobs/s)", flush=True)
    return {"n": writer.n, "n_err": writer.n_err, "wall_s": el}


# --------------------------------------------------------------------------- #
# training (worker side)
# --------------------------------------------------------------------------- #
def training_episode_order(train_ids: list, seed: int, scope: str) -> list:
    """Seed-dependent training-episode order (v2 fix F-4).

    In v1 the training loop cycled ``train_ids[(it * 3 + j) % len(train_ids)]``,
    which is the SAME sequence for every seed. The five "training seeds" then
    differed only in exploration noise and in the rollout rng, so they were not
    independent replications of the data order. This helper permutes the episode
    order per (seed, scope) so a seed really re-draws the data order too.

    Kept as a module-level function so the property is directly testable.
    """
    rng = np.random.default_rng(7000 + int(seed) * 131 + (0 if scope == "local" else 1))
    order = list(train_ids)
    rng.shuffle(order)
    return order


def masked_bellman_target(R: np.ndarray, Q2: np.ndarray, D: np.ndarray,
                          gamma: float, masks: list) -> np.ndarray:
    """Bellman target masked by the SUCCESSOR state's admissible action set.

    v2 fix F-3. ``masks[i]`` is the tuple of actions the actuator constraints
    would accept at the successor state of transition ``i``. Maximising over all
    actions instead (the v1 behaviour) takes the max over levels the environment
    would never execute, which inflates the target and biases the fit.

    Kept as a module-level function so the masking is directly testable.
    """
    if len(masks) != Q2.shape[0]:
        raise ValueError("one admissible set is required per transition")
    mask = np.full(Q2.shape, -1e9, dtype=float)
    for i, adm in enumerate(masks):
        mask[i, list(adm)] = 0.0
    return R + gamma * np.max(Q2 + mask, axis=1) * (1.0 - D)


def mask_bits_matrix(bits: np.ndarray, n_actions: int = 3) -> np.ndarray:
    """Boolean admissible-action matrix from packed bitmasks (bit ``a`` = action ``a``).

    Coverage + replay training accumulates well over a million transitions. A
    Python list of per-transition admissible sets costs ~80 bytes each; a packed
    ``int8`` bitmask costs one byte, and the mask matrix can then be built
    vectorised instead of in a Python loop.
    """
    out = np.zeros((bits.shape[0], n_actions), dtype=bool)
    for a in range(n_actions):
        out[:, a] = (bits & (1 << a)) != 0
    return out


def masked_bellman_target_matrix(R: np.ndarray, Q2: np.ndarray, D: np.ndarray,
                                 gamma: float, mask_bool: np.ndarray) -> np.ndarray:
    """Vectorised :func:`masked_bellman_target` for a precomputed mask matrix."""
    if mask_bool.shape != Q2.shape:
        raise ValueError("mask matrix must have the same shape as Q2")
    return R + gamma * np.max(Q2 + np.where(mask_bool, 0.0, -1e9), axis=1) * (1.0 - D)


def episode_arrays(T: list) -> dict:
    """Compact per-episode transition block (``X2`` is derived by shifting ``X``).

    Transition ``i`` is ``(x_i, a_i, r_i, x_{i+1}, done_i, hour, adm_{i+1})``, so
    storing ``X`` alone is enough: ``X2[:-1] = X[1:]`` and the terminal row is
    its own successor. That halves the largest buffer in the trainer.

    ``X`` is stored as ``float32`` to match the sandbox's transition store
    exactly; the normal equations are still accumulated in ``float64``. Matching
    the dtype keeps the S5 gate (sandbox prediction vs full simulator) a test of
    the implementation rather than of floating-point rounding.
    """
    return {
        "X": np.asarray([t[0] for t in T], dtype=np.float32),
        "A": np.asarray([t[1] for t in T], dtype=int),
        "R": np.asarray([t[2] for t in T], dtype=float),
        "D": np.asarray([t[4] for t in T], dtype=float),
        "M": np.asarray([sum(1 << int(a) for a in t[6]) for t in T], dtype=np.int8),
    }


def _successor(X: np.ndarray) -> np.ndarray:
    X2 = np.empty_like(X)
    X2[:-1] = X[1:]
    X2[-1] = X[-1]
    return X2


def _fit_streaming(model: "ValueModel", buf: list, gamma: float, ridge: float,
                   return_rmse: bool = True) -> float:
    """Ridge least squares by streaming normal-equation accumulation.

    For an action-dependent linear model the full design matrix is block
    diagonal across actions, so ``D^T D`` and ``D^T y`` can be accumulated one
    action at a time. This is algebraically identical to ``ValueModel.fit`` on
    the concatenated design, but never materialises the ``(n, n_actions *
    state_dim)`` design matrix -- which for coverage + replay training would be
    well over a gigabyte.

    Returns the pre-update batch TD RMSE (a convergence diagnostic).
    """
    d = model.state_dim
    na = model.n_actions
    G = [np.zeros((d, d)) for _ in range(na)]
    bvec = [np.zeros(d) for _ in range(na)]
    cnt = [0] * na
    sq = 0.0
    n_tot = 0
    W = model._W                       # (n_actions, state_dim)
    for ep in buf:
        X = ep["X"]
        X2 = _successor(X)
        Q2 = X2 @ W.T
        Q2 = Q2 + np.where(mask_bits_matrix(ep["M"], na), 0.0, -1e9)
        y = ep["R"] + gamma * np.max(Q2, axis=1) * (1.0 - ep["D"])
        if return_rmse:
            q_all = X @ W.T
            q_taken = q_all[np.arange(X.shape[0]), ep["A"]]
            sq += float(np.sum((q_taken - y) ** 2))
            n_tot += X.shape[0]
        for a in range(na):
            idx = np.where(ep["A"] == a)[0]
            cnt[a] += idx.size
            if idx.size < 5:
                continue
            Xa = X[idx]
            G[a] += Xa.T @ Xa
            bvec[a] += Xa.T @ y[idx]
    Wn = W.copy()
    for a in range(na):
        if cnt[a] < 5:
            continue
        Wn[a, :] = np.linalg.solve(G[a] + ridge * np.eye(d), bvec[a])
    model._W = Wn
    if model.kind == "action_dependent":
        model.theta = Wn.reshape(-1)
    return float(np.sqrt(sq / n_tot)) if n_tot else float("nan")


def train_model(spec: dict) -> dict:
    """Train one learned value model with batch fitted-Q inside a worker.

    ``scope='cloud'`` trains in a cloud-only configuration (the cloud always
    holds authority); ``scope='local'`` trains in a local-only configuration
    (the edge always holds authority). Both are trained on connected-operation
    traces, so neither learner ever sees the fault traces used for testing.

    Only the transitions the environment hands back are used; the trainer never
    inspects environment internals.
    """
    proto = protocol_with_override(_CTX["protocol"], spec.get("protocol_override"))
    groups = _CTX["groups"]
    feeder = _CTX["feeder"]
    episodes = _CTX["episodes"]

    kind = spec["kind"]              # 'linear' | 'concatenated'
    scope = spec["scope"]            # 'cloud' | 'local'
    seed = int(spec["seed"])
    vm_kind = "action_dependent" if kind == "linear" else "concatenated"
    vm_cfg = proto.raw["value_model"]

    model = ValueModel(proto.n_actions, vm_kind,
                       state_dim=feature_dim(str(vm_cfg.get("feature_variant", "v1"))),
                       seed=seed)
    rng = np.random.default_rng(9000 + seed * 17 + (0 if scope == "local" else 1))
    n_iter = int(vm_cfg["fitted_q_iterations"])
    eps0 = float(vm_cfg["epsilon_start"])
    eps1 = float(vm_cfg["epsilon_end"])
    gamma = float(vm_cfg["gamma"])
    # ---- v2 training-loop switches (all default to the v1 behaviour) --------
    # coverage: every iteration sweeps ALL training episodes instead of the v1
    #   3-episode window, so the final model is not fitted on the last window only.
    # replay:   the transition buffer accumulates across iterations, so an
    #   iteration refits on everything seen so far rather than only on the
    #   freshly collected batch.
    coverage = bool(vm_cfg.get("coverage", False))
    replay = bool(vm_cfg.get("replay", False))
    mask_admissible = bool(vm_cfg.get("mask_admissible", False))
    explore_admissible = bool(vm_cfg.get("explore_admissible", False))
    train_ids = spec["train_episodes"]
    fmask = tuple(spec["feature_mask"]) if spec.get("feature_mask") else None

    log: list[dict] = []
    # v2 fix F-4: the training-episode order must vary with the seed. In v1 the
    # order was `train_ids[(it * 3 + j) % len(train_ids)]` for EVERY seed, so the
    # five "training seeds" differed only in exploration noise and in the rollout
    # rng -- they were not independent replications of the data order.
    train_order = training_episode_order(train_ids, seed, scope)
    ridge = float(vm_cfg.get("ridge", 1e-6))
    buf: list = []
    for it in range(n_iter):
        eps = eps0 + (eps1 - eps0) * it / max(1, n_iter - 1)
        pick = ([episodes[e] for e in train_order] if coverage
                else [episodes[train_order[(it * 3 + j) % len(train_order)]]
                      for j in range(3)])
        new: list = []
        for ep in pick:
            tr = _get_trace("F0", seed)
            ctl = LearnedController(model, f"{scope}_{kind}",
                                    mask_admissible=mask_admissible)
            if scope == "local":
                pol = PolicySpec(f"train_{kind}_local", None, kind, None, ctl,
                                 explore_eps=eps, feature_mask=fmask,
                                 explore_admissible=explore_admissible)
            else:
                pol = PolicySpec(f"train_{kind}_cloud", kind, None, ctl, None,
                                 explore_eps=eps, feature_mask=fmask,
                                 explore_admissible=explore_admissible)
            _, T = run_simulation(
                feeder=feeder, groups=groups, protocol=proto, episode=ep, scale=1.0,
                trace=tr, policy=pol, collect_transitions=True, collect_for=scope,
                rng_seed=int(rng.integers(1 << 30)))
            if T:
                new.append(episode_arrays(T))
        buf = (buf + new) if replay else new
        n_tr = int(sum(e["X"].shape[0] for e in buf))
        if vm_kind == "action_dependent":
            # v2: stream the normal equations instead of building the full design
            # matrix. Algebraically identical for an action-dependent model, but
            # it bounds memory when coverage + replay accumulate >10^6 rows.
            rmse = _fit_streaming(model, buf, gamma, ridge)
        else:
            X = np.vstack([e["X"] for e in buf])
            A = np.concatenate([e["A"] for e in buf])
            R = np.concatenate([e["R"] for e in buf])
            D = np.concatenate([e["D"] for e in buf])
            X2 = np.vstack([_successor(e["X"]) for e in buf])
            Mb = np.vstack([mask_bits_matrix(e["M"], model.n_actions) for e in buf])
            Q2 = model.q_batch(X2)
            # v2 fix F-3: mask the Bellman target with the successor state's
            # admissible action set. v1 maximised over ALL actions, including
            # ones the actuator constraints would never execute.
            target = masked_bellman_target_matrix(R, Q2, D, gamma, Mb)
            model.fit(X, A, target, ridge=ridge)
            pred = np.array([model.q_values(X[i])[int(A[i])] for i in range(len(A))])
            rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        log.append({"scope": scope, "kind": kind, "seed": seed, "iteration": it,
                    "epsilon": eps, "n_episodes": int(len(pick)),
                    "n_transitions": n_tr, "bellman_rmse": rmse})
    return {"scope": scope, "kind": kind, "seed": seed,
            "theta": model.theta.tolist(), "n_params": int(model.n_params),
            "tag": spec.get("tag", ""), "log": log}


def train_stage(specs: list[dict], out_dir: Path, *, workers: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  training {len(specs)} models on {workers} workers", flush=True)
    t0 = time.perf_counter()
    results = []
    with Pool(processes=workers, initializer=worker_init) as pool:
        for r in pool.imap_unordered(train_model, specs, chunksize=1):
            theta = np.asarray(r["theta"], dtype=float)
            tag = r.get("tag") or ""
            prefix = f"{tag}_" if tag else ""
            np.save(out_dir / f"{prefix}{r['scope']}_{r['kind']}_seed{r['seed']}.npy", theta)
            results.append(r)
            print(f"    trained {tag or '-'} {r['scope']}/{r['kind']} seed={r['seed']} "
                  f"final_rmse={r['log'][-1]['bellman_rmse']:.4f}", flush=True)
    print(f"  training done in {time.perf_counter() - t0:.0f}s", flush=True)
    return results


# --------------------------------------------------------------------------- #
# job construction helpers
# --------------------------------------------------------------------------- #
def load_split() -> dict:
    return json.loads(SPLIT_JSON.read_text(encoding="utf-8"))


def policy_instances(protocol: Protocol, *, with_learned: bool = True,
                     rule_params: list[float] | None = None,
                     n_train_seeds: int | None = None,
                     guard: str = "guarded",
                     cache_enforced: bool = True,
                     family: str | None = None) -> list[dict]:
    """The frozen policy set.

    Architecture (see ``docs/experiment_protocol.md`` section "Policy set"):

    ===== ================================================================
    C0    uncontrolled reference
    C1    cloud rule only, **no** edge fallback  (dependency reference)
    C2    cloud corrected-linear only, no edge fallback
    C3    cloud rule **+** local rule fallback, guarded  (non-strawman
          local comparator required by reviewer W2)
    C4    cloud corrected-linear **+** local corrected-linear fallback,
          guarded  ->  **READ-Grid, the proposed deployment candidate**
    C5    cloud concatenated + local concatenated fallback, guarded
          (diagnostic ablation of the original defective representation)
    ===== ================================================================

    ``family`` restricts the returned set, e.g. ``family="C4"``.
    """
    n_seeds = n_train_seeds if n_train_seeds is not None else int(
        protocol.raw["value_model"]["n_train_seeds"])
    rp = rule_params
    rule_ctl = {"type": "rule", "params": rp}

    out = [
        {"policy_id": "C0", "policy_label": "C0 uncontrolled",
         "cloud": None, "local": None, "cloud_kind": None, "local_kind": None},
        {"policy_id": "C1", "policy_label": "C1 cloud rule (no fallback)",
         "cloud": rule_ctl, "local": None,
         "cloud_kind": "rule", "local_kind": None},
        {"policy_id": "C3", "policy_label": "C3 rule + rule fallback (guarded)",
         "cloud": rule_ctl, "local": rule_ctl,
         "cloud_kind": "rule", "local_kind": "rule"},
    ]
    if not with_learned:
        return [p for p in out if family is None or p["policy_id"] == family]

    def _model(name: str, vm: str) -> dict:
        return {"type": "model", "path": str(MODELS / name), "vm": vm, "name": name}

    for seed in range(n_seeds):
        out.append({
            "policy_id": f"C2_s{seed}", "policy_label": f"C2 cloud linear seed {seed}",
            "cloud": _model(f"cloud_linear_seed{seed}.npy", "action_dependent"),
            "local": None, "cloud_kind": "linear", "local_kind": None,
            "train_seed": seed,
        })
        out.append({
            "policy_id": f"C4_s{seed}", "policy_label": f"READ-Grid seed {seed} (C4)",
            "cloud": _model(f"cloud_linear_seed{seed}.npy", "action_dependent"),
            "local": _model(f"local_linear_seed{seed}.npy", "action_dependent"),
            "cloud_kind": "linear", "local_kind": "linear",
            "train_seed": seed,
        })
        out.append({
            "policy_id": f"C5_s{seed}", "policy_label": f"C5 concatenated ablation seed {seed}",
            "cloud": _model(f"cloud_concatenated_seed{seed}.npy", "concatenated"),
            "local": _model(f"local_concatenated_seed{seed}.npy", "concatenated"),
            "cloud_kind": "concatenated", "local_kind": "concatenated",
            "train_seed": seed,
        })
    for p in out:
        p["guard"] = guard
        p["cache_enforced"] = cache_enforced
    if family is not None:
        out = [p for p in out if p["policy_id"] == family or p["policy_id"].startswith(family + "_")]
    return out


def fault_grid(protocol: Protocol, *, comm_seeds: int | None = None) -> list[tuple[str, int]]:
    """(fault_id, comm_seed) pairs. Deterministic faults use one seed."""
    seeds = comm_seeds if comm_seeds is not None else int(
        protocol.raw["split"]["test_comm_seeds"])
    out = []
    for fid in protocol.raw["deterministic_faults"]:
        out.append((fid, 0))
    for fid in protocol.raw["stochastic_faults"]:
        out.extend((fid, s) for s in range(seeds))
    return out


def expand(policies: list[dict], episode_ids: list[str], faults: list[tuple[str, int]],
           *, scale: float = 1.0, stage: str = "main",
           protocol: Protocol | None = None) -> list[dict]:
    jobs = []
    for pol in policies:
        for eid in episode_ids:
            for fid, seed in faults:
                job = dict(pol)
                job.update({"episode_id": eid, "fault_id": fid, "comm_seed": seed,
                            "scale": scale, "stage": stage, "rng_seed": seed})
                jobs.append(job)
    return jobs
