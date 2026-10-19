"""[v2 / S3-S4] Deterministic re-ranking and freezing of the two selected candidates.

The sweeps print their tables in `imap_unordered` completion order, so candidates
with EQUAL scores come out in a non-reproducible order. Selection must not depend
on that, so both families are re-ranked here under one explicit, documented rule:

    score descending, then iterations ascending (parsimony), then price
    ascending, then seed ascending, then variant name ascending.

Writes runs_v2/s3_learner_winner.json and runs_v2/s4_rule_winner.json, and prints
the head-to-head comparison that S5 and S6 are built on.
"""
import json
import os
import sys

ROOT = os.environ.get('P359_ROOT', '.')
OUT = ROOT + '/runs_v2'

VARIANTS = ['binned_ach', 'raw_ach', 'raw_hinge', 'raw_dwell_ach']
PRICES = [0.30, 0.45, 0.60, 0.90]
ITERS = [12, 16, 20]
SEEDS = [0, 1, 2]


def learner_key(r):
    return (-r['val']['score'], r['iters'], r['price'], r['seed'],
            VARIANTS.index(r['variant']))


def rule_key(r):
    return (-r['val']['score'], r['th_low'], r['th_high'], r['hyst'])


def main():
    fam = json.load(open(OUT + '/s3_learner_tuning_table.json'))
    assert len(fam) == 144, len(fam)
    fam_ranked = sorted(fam, key=learner_key)
    win = fam_ranked[0]
    win_train = sorted(fam, key=lambda r: (-r['train']['score'], r['iters'],
                                           r['price'], r['seed'],
                                           VARIANTS.index(r['variant'])))[0]

    print('=== LEARNER FAMILY (144 candidates, frozen selection_score on VAL) ===')
    print('val-selected winner (pre-registered rule):')
    print('  variant=%s price=%.2f iters=%d seed=%d' %
          (win['variant'], win['price'], win['iters'], win['seed']))
    print('  val  score %.4f  ach %.4f  curt %.4f  sw/g/r %.5f' %
          (win['val']['score'], win['val']['achievement_rate'],
           win['val']['curtail_fraction'], win['val']['switches_per_group_run']))
    print('  train score %.4f  ach %.4f' %
          (win['train']['score'], win['train']['achievement_rate']))
    print('  unmasked-greedy diagnostic: val score %.4f ach %.4f' %
          (win['val_unmasked_greedy']['score'],
           win['val_unmasked_greedy']['achievement_rate']))
    print('runner-up (loses the documented tie-break):')
    r2 = fam_ranked[1]
    print('  variant=%s price=%.2f iters=%d seed=%d  val %.4f' %
          (r2['variant'], r2['price'], r2['iters'], r2['seed'], r2['val']['score']))
    print('train-selected winner (leak-free robustness row):')
    print('  variant=%s price=%.2f iters=%d seed=%d  train %.4f -> val %.4f' %
          (win_train['variant'], win_train['price'], win_train['iters'],
           win_train['seed'], win_train['train']['score'],
           win_train['val']['score']))

    json.dump(dict(selection_rule='val score desc, iters asc, price asc, seed asc, '
                                  'variant name asc',
                   n_candidates=len(fam), winner=win, runner_up=r2,
                   train_selected_winner=win_train),
              open(OUT + '/s3_learner_winner.json', 'w'), indent=2)

    rules = json.load(open(OUT + '/s4_rule_tuning_table.json'))
    assert len(rules) == 144, len(rules)
    rules_ranked = sorted(rules, key=rule_key)
    rwin = rules_ranked[0]
    rwin_train = sorted(rules, key=lambda r: (-r['train']['score'], r['th_low'],
                                              r['th_high'], r['hyst']))[0]
    print('\n=== RULE FAMILY (144 grid points, frozen selection_score on VAL) ===')
    print('val-selected winner: (th_low=%.2f, th_high=%.2f, hyst=%.2f)' %
          (rwin['th_low'], rwin['th_high'], rwin['hyst']))
    print('  val  score %.4f  ach %.4f  curt %.4f' %
          (rwin['val']['score'], rwin['val']['achievement_rate'],
           rwin['val']['curtail_fraction']))
    print('  train score %.4f' % rwin['train']['score'])
    print('train-selected winner: (%.2f, %.2f, %.2f) train %.4f -> val %.4f' %
          (rwin_train['th_low'], rwin_train['th_high'], rwin_train['hyst'],
           rwin_train['train']['score'], rwin_train['val']['score']))
    json.dump(dict(selection_rule='val score desc, th_low asc, th_high asc, hyst asc',
                   n_candidates=len(rules), winner=rwin,
                   train_selected_winner=rwin_train),
              open(OUT + '/s4_rule_winner.json', 'w'), indent=2)

    print('\n=== HEAD TO HEAD on VAL (frozen selection_score) ===')
    d_score = win['val']['score'] - rwin['val']['score']
    d_ach = win['val']['achievement_rate'] - rwin['val']['achievement_rate']
    print('  learned  (C4v2)  score %.4f  ach %.4f' %
          (win['val']['score'], win['val']['achievement_rate']))
    print('  rule++   (C3b)   score %.4f  ach %.4f' %
          (rwin['val']['score'], rwin['val']['achievement_rate']))
    print('  delta            score %+.4f  ach %+.4f' % (d_score, d_ach))
    print('  NOTE: the two criteria disagree in sign, and the VAL winner is '
          'contaminated by 144-way selection.')
    print('\nwrote runs_v2/s3_learner_winner.json, runs_v2/s4_rule_winner.json')


if __name__ == '__main__':
    main()
