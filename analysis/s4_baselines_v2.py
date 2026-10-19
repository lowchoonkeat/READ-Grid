"""[v2 / S4] Stronger baselines, sandbox, TRAIN/VAL only.

  C3b  rule++      -- full enumeration of the extended grid
                      th_low  in {0, .05, .10, .15, .20, .30}
                      th_high in {.2, .3, .4, .5, .6, .75}
                      hyst    in {0, .1, .2, .3}          -> 6*6*4 = 144 points
                      All 144 are scored (D10: no pre-filtering), on TRAIN and on
                      VAL, with the FROZEN analysis.selection_score.
  C7   minimal sufficient level heuristic -- no parameters, no lookahead, no
                      learning: the lowest level L with demand*MULTS[L] <= CAP.
  C8   DP reference -- already produced in S1/S2 (ceil = feasible, floor =
                      optimistic); restated here for the record, not recomputed.

Run:
    P359_ROOT=<project root> python -I s4_baselines_v2.py
"""
import itertools
import json
import os
import sys

sys.argv = ['x', os.environ.get('P359_ROOT', '.')]
sys.path.insert(0, '.')
import numpy as np                      # noqa: E402
import replica as R                     # noqa: E402

ROOT = sys.argv[1]
SPLIT = json.load(open(ROOT + '/runs/workload_split.json'))
EPS = R.load_episodes()
TR, VA = SPLIT['train'], SPLIT['val']
N_GROUPS = 50
KWH = R.TICK / 3600.0
W_ACH, W_CURT, W_SW = 1.0, 1.0, 0.0001     # frozen, configs/protocol_v2.yaml
OUT = ROOT + '/runs_v2'


def score_policy(ids, decide):
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
                score=W_ACH * ar - W_CURT * cf - W_SW * sp)


def c7_decide(c):
    """Minimal sufficient level: no parameters, no lookahead, no learning."""
    b = c['b_est']
    for L in range(3):
        if b * R.MULTS[L] <= R.CAP + 1e-9:
            return L
    return 2


def main():
    grid = list(itertools.product([0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                                  [0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
                                  [0.0, 0.1, 0.2, 0.3]))
    assert len(grid) == 144, len(grid)
    degenerate = [g for g in grid if g[1] <= g[0]]
    rows = []
    for g in grid:
        dec = R.rule(*g)
        tr = score_policy(TR, dec)
        va = score_policy(VA, dec)
        rows.append(dict(th_low=g[0], th_high=g[1], hyst=g[2],
                         degenerate=bool(g[1] <= g[0]),
                         train=tr, val=va))
    json.dump(rows, open(OUT + '/s4_rule_tuning_table.json', 'w'), indent=2)

    by_train = sorted(rows, key=lambda r: -r['train']['score'])
    by_val = sorted(rows, key=lambda r: -r['val']['score'])
    for i, r in enumerate(by_train):
        r['rank_train'] = i + 1
    for i, r in enumerate(by_val):
        r['rank_val'] = i + 1
    json.dump(rows, open(OUT + '/s4_rule_tuning_table.json', 'w'), indent=2)

    print('C3b rule++ : %d grid points scored (%d degenerate th_high<=th_low, D10)'
          % (len(rows), len(degenerate)))
    print('\nTOP 8 by TRAIN score (leak-free selection)')
    for r in by_train[:8]:
        print('  (%.2f,%.2f,%.2f) deg=%-5s train %.4f  val %.4f  ach %.4f curt %.4f'
              % (r['th_low'], r['th_high'], r['hyst'], r['degenerate'],
                 r['train']['score'], r['val']['score'],
                 r['val']['achievement_rate'], r['val']['curtail_fraction']))
    print('\nTOP 8 by VAL score (plan S2.5 literal; winner\'s-curse contaminated)')
    for r in by_val[:8]:
        print('  (%.2f,%.2f,%.2f) deg=%-5s val %.4f  train %.4f  ach %.4f curt %.4f'
              % (r['th_low'], r['th_high'], r['hyst'], r['degenerate'],
                 r['val']['score'], r['train']['score'],
                 r['val']['achievement_rate'], r['val']['curtail_fraction']))

    v1 = dict(th_low=0.20, th_high=0.60, hyst=0.20)
    v1r = [r for r in rows if r['th_low'] == v1['th_low']
           and r['th_high'] == v1['th_high'] and r['hyst'] == v1['hyst']][0]
    print('\nC3 (v1 grid winner 0.20/0.60/0.20): train %.4f  val %.4f  ach %.4f curt %.4f'
          % (v1r['train']['score'], v1r['val']['score'],
             v1r['val']['achievement_rate'], v1r['val']['curtail_fraction']))

    c7 = dict(train=score_policy(TR, c7_decide), val=score_policy(VA, c7_decide))
    print('\nC7 minimal sufficient level (no parameters): train %.4f  val %.4f  '
          'ach %.4f curt %.4f sw/g/r %.4f'
          % (c7['train']['score'], c7['val']['score'], c7['val']['achievement_rate'],
             c7['val']['curtail_fraction'], c7['val']['switches_per_group_run']))
    json.dump(c7, open(OUT + '/s4_c7.json', 'w'), indent=2)

    print('\nC8 DP reference (from S1/S2, not recomputed):')
    print('   nb=600 ceil 0.6524 (feasible)  floor 0.7560 (optimistic)')
    print('   nb=4800 ceil 0.6672  floor 0.6789')
    print('\nwrote runs_v2/s4_rule_tuning_table.json, runs_v2/s4_c7.json')


if __name__ == '__main__':
    main()
