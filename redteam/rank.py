"""Apply the audit prompt's rank formula; this is not gym evaluator code.

Cohort-dependent numbers are illustrative, not an assertion about unseen future
attempt variances. Positive/negative finite Pareto dominance is cohort-invariant.
Missing and nonfinite values are skipped; nonfinite skips are explicitly marked
as an audit assumption because only missing-value handling was specified.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = 'tless_coop_ablation'

def load(p):
    return json.loads(p.read_text())

def metrics(path):
    return load(path).get('experiments', {}).get(EXP, {}).get('results', {}).get('alignpose', {})

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value

def rank(row, ref, defs, std, weight):
    value = 0.
    skipped = []
    for metric, info in defs.items():
        if not info['coefficient']:
            continue
        v = row.get(metric)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            skipped.append(metric)
            continue
        if std[metric] <= 1e-15:
            continue
        direction = 1. if info['higher_is_better'] else -1.
        value += weight*abs(info['coefficient'])*direction*(v-ref[metric])/std[metric]
    return value, skipped

def main():
    experiment = load(ROOT/'scoring/reference.json')['experiments'][EXP]
    defs, weight = experiment['metrics'], experiment['weight']
    ref = metrics(ROOT/'redteam/reference_seed0/scores_test.json')
    rows, statuses = {}, {}
    for f in sorted((ROOT/'redteam').glob('*/status.json')):
        name = f.parent.name
        statuses[name] = load(f)
        p = f.parent/'scores_test.json'
        rows[name] = metrics(p) if p.exists() else {}
    p = ROOT/'redteam/reference_seed0/scores_test.json'
    rows['supplied_baseline'] = load(p)['experiments'][EXP]['results']['single_view_candidate']
    cohorts = {
        'all_audit_attempts': list(rows),
        'scientific_controls': [k for k in rows if not k.startswith('shortcut_')],
    }
    summary = {'reference': ref, 'formula': 'sum(weight * abs(coefficient) * direction * (value-reference)/population_std)',
               'cohort_note': 'Only measured tless experiment contributes; other experiments are absent. Unseen future cohort variances are unknown.',
               'cohorts': {}, 'attempts': {}, 'statuses': statuses}
    for label, names in cohorts.items():
        std = {}
        for key in defs:
            values = [rows[n].get(key) for n in names]
            finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            std[key] = float(np.std(finite)) if finite else 0.
        summary['cohorts'][label] = {'members': names, 'population_std': std}
        for name in names:
            delta, skipped = rank(rows[name], ref, defs, std, weight)
            row = summary['attempts'].setdefault(name, {'metrics': rows[name]})
            row[label+'_rank_delta'] = delta
            row['skipped_metrics'] = skipped
    for name, row in summary['attempts'].items():
        pair_std = {k: abs(row['metrics'].get(k, ref[k])-ref[k])/2
                    if isinstance(row['metrics'].get(k, ref[k]), (int, float)) else 0.
                    for k in defs}
        pair_delta, _ = rank(row['metrics'], ref, defs, pair_std, weight)
        row['pairwise_rank_delta'] = pair_delta
    seeds = [rows[f'reference_seed{i}'] for i in range(3)]
    summary['seed_metric_spread'] = {k: max(s[k] for s in seeds)-min(s[k] for s in seeds) for k in ref}
    for cohort in cohorts:
        values = [summary['attempts'][f'reference_seed{i}'][cohort+'_rank_delta'] for i in range(3)]
        summary.setdefault('seed_rank_spread', {})[cohort] = max(values)-min(values)
    (ROOT/'redteam/summary.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False))
    for name, row in summary['attempts'].items():
        print(name, 'AR=', row['metrics'].get('AR'), 'AP=', row['metrics'].get('AP'),
              'ADD=', row['metrics'].get('mean_add_error'),
              'rank=', round(row['all_audit_attempts_rank_delta'], 6),
              'exit=', statuses.get(name, {}).get('test_exit_code'))

if __name__ == '__main__':
    main()
