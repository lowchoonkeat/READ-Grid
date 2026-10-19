"""[v2 / S3] Learner-v2 design and training, in the sandbox. TRAIN only for fitting,
VAL only for selection. The old TEST split is never read here.

Two exercises, both pre-registered in docs/v2_preregistration.md:

  A) ladder L0..L6   -- descriptive, 5 seeds per row, reproduces and extends the
                        plan's reference table (ablate V-rows + mask_rerun M-rows).
  B) learner family  -- 144 candidates = 4 prices x 4 variants x 3 iterations x 3
                        seeds, every one scored with the FROZEN
                        analysis.selection_score on VAL, all scores archived.

Design decisions (documented in docs/v2_stage_S3_report.md):
  * D12  Every candidate is scored with the frozen selection objective
         (achievement - curtail_fraction - 1e-4 * switches/group/run), NOT with
         raw achievement. Same objective for the rule family in S4.
  * D13  The v2 deployment semantics restrict the greedy choice to the admissible
         set (the companion of the masked Bellman target). The unmasked greedy
         score is also recorded per candidate as a diagnostic row.
  * D14  L6 (direct policy search) searches on a TRAIN subsample and is reported
         on full TRAIN and VAL, so VAL stays honest.

Run:
    P359_ROOT=<project root> python -I s3_learner_v2.py selfcheck
    P359_ROOT=<project root> python -I s3_learner_v2.py ladder
    P359_ROOT=<project root> python -I s3_learner_v2.py family
    P359_ROOT=<project root> python -I s3_learner_v2.py l6
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

_CMD = sys.argv[1] if len(sys.argv) > 1 else 'selfcheck'   # capture BEFORE rewriting argv
sys.argv = ['x', os.environ.get('P359_ROOT', '.')]
sys.path.insert(0, '.')
import numpy as np                      # noqa: E402
import replica as R                     # noqa: E402

ROOT = sys.argv[1]
SPLIT = json.load(open(ROOT + '/runs/workload_split.json'))
EPS = R.load_episodes()
TR, VA = SPLIT['train'], SPLIT['val']
N_GROUPS = 50                            # configs/protocol_v2.yaml: system.n_groups
OUT = ROOT + '/runs_v2'
KWH = R.TICK / 3600.0

# frozen selection objective, verbatim from configs/protocol_v2.yaml
W_ACH, W_CURT, W_SW = 1.0, 1.0, 0.0001


def step_cost_frac(L):
    return R.LEVELS[L] / 0.7


# ------------------------------------------------------------------ features ---
class FeatV2:
    """Feature builders. 'raw' keeps the sandbox raw8 layout with the voltage
    feature replaced by a constant (voltage_dev is DROPPED in v2); 'raw_dwell'
    puts the dwell feature (minutes since the last level change, capped at 3) in
    that slot; 'binned' is the sandbox one-hot set with use_pace=False."""

    def __init__(self, kind):
        self.kind = kind
        if kind in ('raw', 'raw_dwell'):
            self.dim = 8
            return
        self.exc_edges = np.r_[np.arange(0, 1.5, 0.05), 1.5, 9]
        nb = len(self.exc_edges) - 1
        self.dim = 1 + nb + 3 + 2 + 10 + 2      # use_pace=False

    def __call__(self, exc, cur, c_since, leftfrac, dr, b_est, hour):
        if self.kind in ('raw', 'raw_dwell'):
            a = 2 * math.pi * hour / 24
            dwell = min(3, c_since) / 3 if self.kind == 'raw_dwell' else 0.5
            return np.array([1, math.sin(a), math.cos(a),
                             min(max(b_est, 0), 3) / 3, cur / 2,
                             min(max(leftfrac, 0), 1), dwell,
                             min(max(exc, 0), 3) / 3])
        x = np.zeros(self.dim)
        x[0] = 1
        i = 1
        nb = len(self.exc_edges) - 1
        x[i + min(nb - 1, max(0, np.searchsorted(self.exc_edges, exc, 'right') - 1))] = 1
        i += nb
        x[i + cur] = 1
        i += 3
        x[i + int(c_since >= 3)] = 1
        i += 2
        x[i + min(9, int(leftfrac * 10))] = 1
        i += 10
        x[i] = float(dr)
        i += 1
        x[i] = exc * (cur == 2)
        return x


# ------------------------------------------------------------------- dynamics ---
def admissible(cur, c_since, left, b_est):
    ok = []
    for a in range(3):
        if abs(a - cur) > 1:
            continue
        if a != cur and c_since < 3:
            continue
        if R.cost(b_est, a) > left + 1e-12:
            continue
        ok.append(a)
    return ok or [cur if R.cost(b_est, cur) <= left else 0]


def trajectory(shape, F, W, eps_, rng, reward_cfg, **kw):
    """Rollout plus the accounting the frozen selection objective needs.

    Semantics identical to ``fqi_lib.rollout``: transition i carries the
    admissible set of its SUCCESSOR state, the terminal transition is flagged
    done, and ``store_exec`` selects executed vs proposed action labels."""
    n = len(shape)
    dr = shape >= 0.80
    cur, last_change, left = 0, -10 ** 9, R.BUDGET
    b_prev = shape[0]
    lev = np.zeros(n, dtype=np.int8)
    X = np.zeros((n, F.dim), dtype=np.float32)
    A = np.zeros(n, dtype=np.int8)
    Rw = np.zeros(n, dtype=np.float32)
    D = np.zeros(n, dtype=bool)
    M = np.zeros((n, 3), dtype=bool)
    store_exec = kw.get('store_exec', True)
    explore_adm = kw.get('explore_adm', True)
    unrestricted = kw.get('unrestricted', False)
    collect = kw.get('collect', True)
    prev_x = prev_r = prev_exec = None
    for k in range(n):
        b_est = b_prev
        exc = max(0.0, b_est - R.CAP) / (R.FLEX * 0.7 * b_est) if (dr[k] and b_est > 0) else 0.0
        c_since = min(3, k - last_change) if last_change > -10 ** 8 else 3
        x = F(exc, cur, c_since, left / R.BUDGET, dr[k], b_est, k / 60.0)
        adm = admissible(cur, c_since, left, b_est)
        if W is None:
            a = int(rng.choice(adm)) if explore_adm else int(rng.integers(0, 3))
        else:
            q = x @ W
            if rng.random() < eps_:
                a = int(rng.choice(adm)) if explore_adm else int(rng.integers(0, 3))
            elif unrestricted:
                a = int(np.argmax(q + 1e-9 * np.arange(3)[::-1]))
            else:
                a = int(max(adm, key=lambda z: q[z] - 1e-9 * z))
        new = R.apply(cur, a, k, last_change, left, shape[k])
        sw = int(new != cur)
        if new != cur:
            last_change = k
        cur = new
        lev[k] = cur
        left -= R.cost(shape[k], cur)
        ach = float(dr[k] and shape[k] * R.MULTS[cur] <= R.CAP + 1e-9)
        if reward_cfg.get('hinge'):
            r = (-reward_cfg['w_track'] * max(0.0, shape[k] * R.MULTS[cur] - R.CAP)
                 / (R.FLEX * 0.7 * shape[k]) * float(dr[k])
                 - reward_cfg['w_cost'] * step_cost_frac(cur) - reward_cfg['w_sw'] * sw)
        else:
            r = (reward_cfg['w_ach'] * ach
                 - reward_cfg['w_cost'] * step_cost_frac(cur) - reward_cfg['w_sw'] * sw)
        if collect and prev_x is not None:
            X[k - 1] = prev_x
            A[k - 1] = prev_exec if store_exec else prev_a
            Rw[k - 1] = prev_r
            for z in adm:
                M[k - 1, z] = True
        prev_x, prev_r, prev_a, prev_exec = x, r, a, cur
        b_prev = shape[k]
    if collect:
        X[n - 1] = prev_x
        A[n - 1] = prev_exec if store_exec else prev_a
        Rw[n - 1] = prev_r
        D[n - 1] = True
        M[n - 1, :] = True
    return dict(X=X, A=A, Rw=Rw, D=D, M=M, lev=lev, shape=shape,
                left=left, cur=cur)


# -------------------------------------------------------------------- fitting ---
def _accumulate(buf, W, gamma, ridge, dim, masked=True):
    G = [np.zeros((dim, dim)) for _ in range(3)]
    b = [np.zeros(dim) for _ in range(3)]
    cnt = [0, 0, 0]
    for ep in buf:
        X, A, Rw, D, M = ep['X'], ep['A'], ep['Rw'], ep['D'], ep['M']
        X2 = np.empty_like(X)
        X2[:-1] = X[1:]
        X2[-1] = X[-1]
        Q2 = (X2.astype(np.float64) @ W)
        if masked:
            Q2 = Q2 + np.where(M, 0.0, -1e9)
        y = Rw + gamma * Q2.max(1) * (1.0 - D)
        for a in range(3):
            idx = np.where(A == a)[0]
            cnt[a] += idx.size
            if idx.size < 5:
                continue
            Xa = X[idx].astype(np.float64)
            G[a] += Xa.T @ Xa
            b[a] += Xa.T @ y[idx]
    return G, b, cnt


def fit(F, reward_cfg, gamma, iters, seed, ridge, all_episodes=True, replay=True,
        n_ep=3, e0=0.4, e1=0.05, store_exec=True, explore_adm=True,
        unrestricted=False, masked_target=True):
    rng = np.random.default_rng(seed)
    W = np.zeros((F.dim, 3))
    buf = []
    for it in range(iters):
        e = e0 + (e1 - e0) * it / max(1, iters - 1)
        pick = TR if all_episodes else [TR[(it * 3 + j) % len(TR)] for j in range(n_ep)]
        new = []
        for eid in pick:
            new.append(trajectory(EPS[eid].shape, F, W if it > 0 else None, e, rng,
                                  reward_cfg, store_exec=store_exec,
                                  explore_adm=explore_adm,
                                  unrestricted=unrestricted, collect=True))
        buf = (buf + new) if replay else new
        G, b, cnt = _accumulate(buf, W, gamma, ridge, F.dim, masked=masked_target)
        Wn = W.copy()
        for a in range(3):
            if cnt[a] < 5:
                continue
            Wn[:, a] = np.linalg.solve(G[a] + ridge * np.eye(F.dim), b[a])
        W = Wn
    return W


# ----------------------------------------------------------------- evaluation ---
def score_on(F, W, ids, unrestricted):
    """Frozen selection objective aggregated over `ids`."""
    rng = np.random.default_rng(1)
    ach = el = 0
    curt = base = 0.0
    sw = 0
    for eid in ids:
        sh = EPS[eid].shape
        tr = trajectory(sh, F, W, 0.0, rng, dict(w_ach=1, w_cost=0, w_sw=0),
                        store_exec=True, explore_adm=True,
                        unrestricted=unrestricted, collect=False)
        lev = tr['lev']
        ach += int(((sh * np.array([R.MULTS[l] for l in lev]) <= R.CAP + 1e-9)
                    & (sh >= 0.80)).sum())
        el += int((sh >= 0.80).sum())
        curt += float(np.sum(sh * R.FLEX * np.array([R.LEVELS[l] for l in lev]) * KWH))
        base += float(np.sum(sh) * KWH)
        sw += int((np.diff(lev) != 0).sum())
    ach_rate = ach / el if el else float('nan')
    curt_frac = curt / base if base else float('nan')
    sw_pgr = sw / N_GROUPS / max(len(ids), 1)
    return dict(achievement_rate=ach_rate, curtail_fraction=curt_frac,
                switches_per_group_run=sw_pgr,
                score=W_ACH * ach_rate - W_CURT * curt_frac - W_SW * sw_pgr,
                n_episodes=len(ids))


def score_policy_frozen(ids, decide):
    """Frozen selection objective for a parameter-free / rule policy (same
    accounting as ``score_on``, but driven by ``R.run`` so the replica's own
    rule and C7 helpers can be scored identically to S4)."""
    ach = el = 0
    curt = base = 0.0
    sw = 0
    for eid in ids:
        sh = EPS[eid].shape
        r = R.run(sh, decide)
        lev = r['lev']
        ach += r['ach']
        el += r['elig']
        curt += float(np.sum(sh * R.FLEX * np.array([R.LEVELS[l] for l in lev]) * KWH))
        base += float(np.sum(sh) * KWH)
        sw += int((np.diff(lev) != 0).sum())
    ar = ach / el if el else float('nan')
    cf = curt / base if base else float('nan')
    sp = sw / N_GROUPS / max(len(ids), 1)
    return dict(achievement_rate=ar, curtail_fraction=cf,
                switches_per_group_run=sp,
                score=W_ACH * ar - W_CURT * cf - W_SW * sp, n_episodes=len(ids))


def _c7_decide(c):
    """C7 minimal sufficient level, identical to S4's definition."""
    b = c['b_est']
    for L in range(3):
        if b * R.MULTS[L] <= R.CAP + 1e-9:
            return L
    return 2


# --------------------------------------------------------------------- configs ---
HINGE = lambda p: dict(hinge=True, w_track=1.0, w_cost=p, w_sw=0.05, w_ach=0.0)
ACH = lambda p: dict(w_ach=1.0, w_cost=p, w_sw=0.02)

# 4 variants x 4 prices = 16, x 3 iterations x 3 seeds = 144 candidates
VARIANTS = {
    'raw_hinge':     dict(feat='raw',        reward=HINGE, gamma=0.97, ridge=1e-6),
    'raw_ach':       dict(feat='raw',        reward=ACH,   gamma=0.99, ridge=1e-6),
    'binned_ach':    dict(feat='binned',     reward=ACH,   gamma=0.99, ridge=1.0),
    'raw_dwell_ach': dict(feat='raw_dwell',  reward=ACH,   gamma=0.99, ridge=1e-6),
}
PRICES = (0.30, 0.45, 0.60, 0.90)
ITERS = (12, 16, 20)
SEEDS = (0, 1, 2)


def _family_job(job):
    name, price, iters, seed = job
    spec = VARIANTS[name]
    F = FeatV2(spec['feat'])
    rc = spec['reward'](price)
    t0 = time.time()
    W = fit(F, rc, spec['gamma'], iters, seed, spec['ridge'],
            all_episodes=True, replay=True, store_exec=True, explore_adm=True,
            unrestricted=False, masked_target=True, e0=0.5, e1=0.05)
    v = score_on(F, W, VA, unrestricted=False)
    vd = score_on(F, W, VA, unrestricted=True)
    tr = score_on(F, W, TR, unrestricted=False)
    return dict(variant=name, price=price, iters=iters, seed=seed,
                val=v, val_unmasked_greedy=vd, train=tr,
                wall_s=round(time.time() - t0, 1))


# ------------------------------------------------------------------ L0..L5 rows ---
LADDER = {
    'L0  v1-like (3 eps/iter, no replay, exec action, random explore, unmasked greedy)':
        dict(feat='raw', reward=HINGE(0.20), gamma=0.97, iters=8, ridge=1e-6,
             all_episodes=False, replay=False, n_ep=3, store_exec=True,
             explore_adm=False, unrestricted=True, masked_target=False),
    'L1a +replay only':
        dict(feat='raw', reward=HINGE(0.20), gamma=0.97, iters=8, ridge=1e-6,
             all_episodes=False, replay=True, n_ep=3, store_exec=True,
             explore_adm=False, unrestricted=True, masked_target=False),
    'L1b +coverage only (all 58 eps/iter)':
        dict(feat='raw', reward=HINGE(0.20), gamma=0.97, iters=8, ridge=1e-6,
             all_episodes=True, replay=False, n_ep=3, store_exec=True,
             explore_adm=False, unrestricted=True, masked_target=False),
    'L1c coverage+replay':
        dict(feat='raw', reward=HINGE(0.20), gamma=0.97, iters=8, ridge=1e-6,
             all_episodes=True, replay=True, n_ep=3, store_exec=True,
             explore_adm=False, unrestricted=True, masked_target=False),
    'L2  store PROPOSED action (independent ablation, not used by default)':
        dict(feat='raw', reward=HINGE(0.20), gamma=0.97, iters=8, ridge=1e-6,
             all_episodes=True, replay=True, n_ep=3, store_exec=False,
             explore_adm=False, unrestricted=True, masked_target=False),
    'L3a raw8 + hinge + successor mask':
        dict(feat='raw', reward=HINGE(0.20), gamma=0.97, iters=12, ridge=1e-6,
             all_episodes=True, replay=True, store_exec=False,
             explore_adm=True, unrestricted=False, masked_target=True),
    'L3b raw8 + achievement indicator + successor mask':
        dict(feat='raw', reward=ACH(0.30), gamma=0.99, iters=12, ridge=1e-6,
             all_episodes=True, replay=True, store_exec=False,
             explore_adm=True, unrestricted=False, masked_target=True),
    'L4a binned + achievement c.30 + mask':
        dict(feat='binned', reward=ACH(0.30), gamma=0.99, iters=12, ridge=1.0,
             all_episodes=True, replay=True, store_exec=False,
             explore_adm=True, unrestricted=False, masked_target=True),
    'L4b binned + achievement c.45 + mask':
        dict(feat='binned', reward=ACH(0.45), gamma=0.99, iters=12, ridge=1.0,
             all_episodes=True, replay=True, store_exec=False,
             explore_adm=True, unrestricted=False, masked_target=True),
    'L4c binned + achievement c.60 + mask':
        dict(feat='binned', reward=ACH(0.60), gamma=0.99, iters=12, ridge=1.0,
             all_episodes=True, replay=True, store_exec=False,
             explore_adm=True, unrestricted=False, masked_target=True),
    'L4d binned + achievement c.90 + mask':
        dict(feat='binned', reward=ACH(0.90), gamma=0.99, iters=12, ridge=1.0,
             all_episodes=True, replay=True, store_exec=False,
             explore_adm=True, unrestricted=False, masked_target=True),
    'L5  raw8 + dwell feature + achievement c.45 + mask':
        dict(feat='raw_dwell', reward=ACH(0.45), gamma=0.99, iters=12,
             ridge=1e-6, all_episodes=True, replay=True, store_exec=False,
             explore_adm=True, unrestricted=False, masked_target=True),
}
LADDER_SEEDS = (0, 1, 2, 3, 4)     # 5 seeds per row (plan S3)


def _ladder_job(job):
    name, seed = job
    cfg = LADDER[name]
    F = FeatV2(cfg['feat'])
    kw = {k: v for k, v in cfg.items()
          if k in ('all_episodes', 'replay', 'n_ep', 'store_exec',
                   'explore_adm', 'unrestricted', 'masked_target')}
    # epsilon schedule: the sandbox's own fit_fqi uses 0.5->0.05, ablate.py uses
    # 0.4->0.05. mask_targets=True identifies the fqi_lib rows, False the ablate rows.
    kw['e0'] = cfg.get('e0', 0.5 if cfg['masked_target'] else 0.4)
    kw['e1'] = cfg.get('e1', 0.05)
    W = fit(F, cfg['reward'], cfg['gamma'], cfg['iters'], seed, cfg['ridge'], **kw)
    return name, seed, score_on(F, W, VA, unrestricted=cfg['unrestricted'])


# ------------------------------------------------------------------ L6 direct search ---
def _l6_policy(theta):
    th_low, th_high, hyst, bud_lo, bud_hi = theta

    def f(c):
        e, cur, bf = c['exc'], c['cur'], c['left']
        if bf <= bud_lo:
            return 0
        t = 2 if e > th_high else (1 if e > th_low else 0)
        if bf <= bud_hi:
            t = min(t, 1)
        if cur > t:
            stay = (e > th_high - hyst) if cur == 2 else (e > th_low - hyst)
            if stay:
                t = cur
        return max(0, min(2, t))
    return f


def _rule_score(ids, theta):
    ach = el = 0
    curt = base = 0.0
    sw = 0
    dec = _l6_policy(theta)
    for eid in ids:
        sh = EPS[eid].shape
        r = R.run(sh, dec)
        lev = r['lev']
        ach += r['ach']
        el += r['elig']
        curt += float(np.sum(sh * R.FLEX * np.array([R.LEVELS[l] for l in lev]) * KWH))
        base += float(np.sum(sh) * KWH)
        sw += int((np.diff(lev) != 0).sum())
    ar = ach / el if el else 0.0
    cf = curt / base if base else 0.0
    sp = sw / N_GROUPS / max(len(ids), 1)
    return W_ACH * ar - W_CURT * cf - W_SW * sp


def _l6_job(seed):
    """(1+lambda) evolution strategy on a TRAIN subsample (D14)."""
    rng = np.random.default_rng(seed)
    search_ids = TR[:8]                      # bounded search set, documented
    lam = 8
    theta = np.array([0.05, 0.30, 0.05, 0.00, 0.05])
    lo = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.6, 1.2, 0.6, 0.30, 1.00])
    sigma = np.array([0.08, 0.12, 0.06, 0.03, 0.10])
    best = theta.copy()
    best_s = _rule_score(search_ids, best)
    n_eval = 1200
    for it in range(n_eval // lam):
        cand = np.clip(theta + rng.normal(0, 1, (lam, 5)) * sigma, lo, hi)
        scores = np.array([_rule_score(search_ids, c) for c in cand])
        j = int(np.argmax(scores))
        if scores[j] > best_s:
            best_s, best = scores[j], cand[j].copy()
            theta = cand[j].copy()
        else:
            sigma *= 0.99
    return dict(seed=seed, theta=list(best),
                train=_rule_score(TR, best), val=_rule_score(VA, best))


# ------------------------------------------------------------------------ selfcheck ---
def selfcheck():
    """Reimplemented rollout/fit must reproduce the sandbox's own numbers."""
    import fqi_lib as L
    ok = True
    # V3 coverage+replay, ablate semantics, seed 0 -> 0.5701 (ablate.txt)
    cfg = LADDER['L1c coverage+replay']
    F = FeatV2('raw')
    W = fit(F, cfg['reward'], 0.97, 8, 0, 1e-6, all_episodes=True, replay=True,
            n_ep=3, store_exec=True, explore_adm=False, unrestricted=True,
            masked_target=False)
    got = score_on(F, W, VA, unrestricted=True)['achievement_rate']
    print('selfcheck V3 (expect 0.5701): %.4f  %s' % (got, 'OK' if abs(got - 0.5701) < 5e-4 else 'MISMATCH'))
    ok &= abs(got - 0.5701) < 5e-4
    # M5 binned + ach c.45 + mask, seed 0 -> 0.5765 (mask_rerun.txt)
    F = FeatV2('binned')
    W = fit(F, ACH(0.45), 0.99, 12, 0, 1.0, all_episodes=True, replay=True,
            store_exec=False, explore_adm=True, unrestricted=False,
            masked_target=True, e0=0.5, e1=0.05)
    got = score_on(F, W, VA, unrestricted=False)['achievement_rate']
    print('selfcheck M5 (expect 0.5765): %.4f  %s' % (got, 'OK' if abs(got - 0.5765) < 5e-4 else 'MISMATCH'))
    ok &= abs(got - 0.5765) < 5e-4
    print('SELFCHECK', 'PASS' if ok else 'FAIL')
    return ok


# ----------------------------------------------------------------------------- main ---
def run_ladder():
    jobs = [(n, s) for n in LADDER for s in LADDER_SEEDS]
    out = {}
    with Pool(3) as p:
        for name, seed, v in p.imap_unordered(_ladder_job, jobs):
            out.setdefault(name, []).append(v)
            print('  %-6s seed %d  ach %.4f  score %.4f' % (name[:6], seed,
                  v['achievement_rate'], v['score']), flush=True)
    rows = []
    print('\nLADDER (VAL, 5 seeds) -- frozen selection_score')
    for n in LADDER:
        vs = out[n]
        a = np.array([x['achievement_rate'] for x in vs])
        s = np.array([x['score'] for x in vs])
        rows.append(dict(row=n, seeds=LADDER_SEEDS,
                         ach_mean=float(a.mean()), ach_min=float(a.min()),
                         ach_max=float(a.max()), score_mean=float(s.mean()),
                         score_min=float(s.min()), score_max=float(s.max()),
                         per_seed=vs))
        print('%s\n   ach %.4f [%.4f-%.4f]  score %.4f [%.4f-%.4f]' %
              (n, a.mean(), a.min(), a.max(), s.mean(), s.min(), s.max()))
    json.dump(rows, open(OUT + '/s3_ladder.json', 'w'), indent=2)
    print('\nwrote runs_v2/s3_ladder.json')


def run_family():
    jobs = [(n, p, i, s) for n in VARIANTS for p in PRICES
            for i in ITERS for s in SEEDS]
    print('learner family: %d candidates' % len(jobs), flush=True)
    rows = []
    t0 = time.time()
    with Pool(3) as p:
        for k, r in enumerate(p.imap_unordered(_family_job, jobs), 1):
            rows.append(r)
            if k % 12 == 0 or k == len(jobs):
                print('  %d/%d  elapsed %.0fs' % (k, len(jobs), time.time() - t0),
                      flush=True)
    rows.sort(key=lambda r: -r['val']['score'])
    for i, r in enumerate(rows):
        r['rank_val'] = i + 1
    tr_sorted = sorted(rows, key=lambda r: -r['train']['score'])
    for i, r in enumerate(tr_sorted):
        r['rank_train'] = i + 1
    json.dump(rows, open(OUT + '/s3_learner_tuning_table.json', 'w'), indent=2)
    print('\nTOP 10 by VAL selection_score (winner\'s-curse contaminated)')
    for r in rows[:10]:
        print('  %-14s p=%.2f it=%2d seed %d | val score %.4f ach %.4f curt %.4f'
              % (r['variant'], r['price'], r['iters'], r['seed'],
                 r['val']['score'], r['val']['achievement_rate'],
                 r['val']['curtail_fraction']))
    print('\nTOP 5 by TRAIN selection_score (leak-free selection)')
    for r in tr_sorted[:5]:
        print('  %-14s p=%.2f it=%2d seed %d | train %.4f -> val %.4f'
              % (r['variant'], r['price'], r['iters'], r['seed'],
                 r['train']['score'], r['val']['score']))
    print('\nwrote runs_v2/s3_learner_tuning_table.json (%d rows)' % len(rows))


def run_l6():
    with Pool(3) as p:
        res = list(p.imap_unordered(_l6_job, LADDER_SEEDS))
    res.sort(key=lambda r: r['seed'])
    for r in res:
        print('  L6 seed %d  theta %s  train %.4f  val %.4f'
              % (r['seed'], ['%.3f' % t for t in r['theta']], r['train'], r['val']))
    tv = np.array([r['val'] for r in res])
    print('L6 direct policy search: val score %.4f [%.4f-%.4f] (5 seeds)'
          % (tv.mean(), tv.min(), tv.max()))
    json.dump(res, open(OUT + '/s3_l6.json', 'w'), indent=2)
    print('wrote runs_v2/s3_l6.json')


def run_final():
    """5-seed retrain of the FROZEN winner (plan S3: 'freeze the single selected
    candidate, retrain with 5 seeds'). This is the sandbox prediction the S5 gate
    (Gate G2) is compared against, so it must use exactly the frozen config.

    Also re-scores the frozen C3b rule and C7 in the sandbox, so the gap check has
    all three reference points on the same 21 val episodes.
    """
    frozen = json.load(open(OUT + '/s3_learner_winner.json'))
    w = frozen['winner']
    spec = VARIANTS[w['variant']]
    F = FeatV2(spec['feat'])
    rc = spec['reward'](w['price'])
    print('frozen winner: %s price=%.2f iters=%d (from %s)'
          % (w['variant'], w['price'], w['iters'], frozen['selection_rule']),
          flush=True)
    rows = []
    t0 = time.time()
    for seed in LADDER_SEEDS:
        W = fit(F, rc, spec['gamma'], w['iters'], seed, spec['ridge'],
                all_episodes=True, replay=True, store_exec=True, explore_adm=True,
                unrestricted=False, masked_target=True, e0=0.5, e1=0.05)
        va = score_on(F, W, VA, unrestricted=False)
        tr = score_on(F, W, TR, unrestricted=False)
        rows.append(dict(seed=seed, val=va, train=tr))
        print('  seed %d | val score %.4f ach %.4f curt %.4f sw/g/r %.4f | train %.4f'
              % (seed, va['score'], va['achievement_rate'], va['curtail_fraction'],
                 va['switches_per_group_run'], tr['score']), flush=True)
    sv = np.array([r['val']['score'] for r in rows])
    sa = np.array([r['val']['achievement_rate'] for r in rows])
    print('FROZEN 5-seed retrain (sandbox): val score %.4f [%.4f-%.4f]  '
          'ach %.4f [%.4f-%.4f]  (%.0fs)'
          % (sv.mean(), sv.min(), sv.max(), sa.mean(), sa.min(), sa.max(),
             time.time() - t0), flush=True)

    ref = {}
    for name, th in (('C3b', (0.00, 0.30, 0.00)), ('C3', (0.20, 0.60, 0.20))):
        ref[name] = dict(params=list(th),
                         val=score_policy_frozen(VA, R.rule(*th)),
                         train=score_policy_frozen(TR, R.rule(*th)))
        print('%s rule %s: val score %.4f ach %.4f'
              % (name, th, ref[name]['val']['score'],
                 ref[name]['val']['achievement_rate']), flush=True)
    ref['C7'] = dict(val=score_policy_frozen(VA, _c7_decide),
                     train=score_policy_frozen(TR, _c7_decide))
    print('C7 minimal sufficient level: val score %.4f ach %.4f'
          % (ref['C7']['val']['score'], ref['C7']['val']['achievement_rate']),
          flush=True)

    json.dump(dict(frozen_winner=w, seeds=rows,
                   val_score_mean=float(sv.mean()),
                   val_achievement_mean=float(sa.mean()),
                   references=ref),
              open(OUT + '/s3_final_sandbox.json', 'w'), indent=2)
    print('wrote runs_v2/s3_final_sandbox.json')


if __name__ == '__main__':
    {'selfcheck': selfcheck, 'ladder': run_ladder,
     'family': run_family, 'l6': run_l6, 'final': run_final}[_CMD]()
