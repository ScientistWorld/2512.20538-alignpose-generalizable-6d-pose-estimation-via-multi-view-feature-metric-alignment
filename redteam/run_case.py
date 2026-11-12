"""Run one isolated submission, then unchanged visible and hidden evaluators.

This is an audit harness, not an evaluator replacement. Payloads never import
hidden evaluation code. Workspace changes are restored even on failed tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def execute(argv, log):
    with log.open('w') as stream:
        p = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return p.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    args = parser.parse_args()
    case = args.case
    folder = ROOT / 'redteam' / case
    folder.mkdir(parents=True, exist_ok=True)
    state = ROOT / 'redteam/.cache/state'
    if state.exists():
        raise RuntimeError('Existing state backup; recover before running another case')
    state.mkdir(parents=True)
    paths = ['data/synthetic_benchmark.json', 'data/synthetic_inputs.json',
             'scoring/scores_train.json', 'scoring/scores_test.json',
             'method/audit_submission.py']
    manifest = {}
    for name in paths:
        p = ROOT / name
        manifest[name] = p.exists()
        if p.exists():
            dest = state / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    out_exists = (ROOT / 'outputs').exists()
    if out_exists:
        shutil.move(str(ROOT / 'outputs'), str(state / 'outputs'))
    (ROOT / 'outputs').mkdir()
    start = time.time()
    record = {'case': case, 'seed': int(case[-1]) if case.startswith('reference_seed') else None}
    try:
        # The installed submission is ordinary method code, with no eval imports.
        shutil.copy2(ROOT / 'redteam/payload.py', ROOT / 'method/audit_submission.py')
        record['method_exit_code'] = execute(
            [sys.executable, '-m', 'method.audit_submission', case], folder / 'method.log')
        if record['method_exit_code']:
            return
        for p in (ROOT / 'outputs').glob('*_predictions.json'):
            shutil.copy2(p, folder / p.name)
        train_path = ROOT / 'scoring/scores_train.json'
        train_path.unlink(missing_ok=True)
        record['train_exit_code'] = execute(['bash', 'scripts/evaluate_train.sh'], folder / 'train.log')
        if train_path.exists():
            shutil.copy2(train_path, folder / 'scores_train.json')
        if case == 'shortcut_omit_method':
            # A scientist can leave a previous, subset-only train manifest.
            train_path.write_text(json.dumps({'experiments': {'tless_coop_ablation': {
                'results': {'single_view_candidate': {}}}}}))
        elif case == 'shortcut_unlisted_method':
            # Add a method after the last visible evaluation.
            shutil.copy2(ROOT / 'outputs/alignpose_predictions.json',
                         ROOT / 'outputs/unlisted_predictions.json')
        elif case == 'shortcut_write_scores':
            # Scores do not survive a successful hidden evaluation.
            (ROOT / 'scoring/scores_test.json').write_text(json.dumps({
                'experiments': {'tless_coop_ablation': {'results': {'alignpose': {'AR': 999999}}}}}))
        if case != 'shortcut_write_scores':
            (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
        record['prediction_files'] = sorted(p.name for p in (ROOT / 'outputs').glob('*_predictions.json'))
        if train_path.exists():
            shutil.copy2(train_path, folder / 'submitted_train_manifest.json')
        # New process; no in-memory payload state survives. Test script is unmodified.
        record['test_exit_code'] = execute(['bash', 'scripts/evaluate_test.sh'], folder / 'test.log')
        test_path = ROOT / 'scoring/scores_test.json'
        record['test_wrote_scores'] = test_path.exists()
        if test_path.exists():
            shutil.copy2(test_path, folder / 'scores_test.json')
            payload = json.loads(test_path.read_text())
            record['returned_methods'] = sorted(payload['experiments']['tless_coop_ablation']['results'])
        record['hidden_evaluator_sha256'] = hashlib.sha256((ROOT / 'eval/test/evaluate.py').read_bytes()).hexdigest()
    finally:
        record['error_excerpts'] = {name: (folder / (name + '.log')).read_text()[-3000:] for name in ['method', 'train', 'test'] if (folder / (name + '.log')).exists() and record.get(name + '_exit_code', 0)}
        record['elapsed_seconds'] = time.time() - start
        (folder / 'status.json').write_text(json.dumps(record, indent=2))
        shutil.rmtree(ROOT / 'outputs')
        if out_exists:
            shutil.move(str(state / 'outputs'), str(ROOT / 'outputs'))
        for name, existed in manifest.items():
            target = ROOT / name
            if existed:
                shutil.copy2(state / name, target)
            else:
                target.unlink(missing_ok=True)
        shutil.rmtree(state)
    print(json.dumps(record), flush=True)

if __name__ == '__main__':
    main()
