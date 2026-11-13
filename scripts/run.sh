#!/bin/bash
# Scheduler job for the hardening iteration (final run). Resumable: the node may
# be preempted, so every stage records completion and the evaluators/method
# cache finished work under tmp/job_state_<code hash> (results are identical to a clean run).
#   1. regenerate the visible train split with the current generator
#   2. scripts/evaluate_train.sh code path  -> scoring/scores_train.json
#   3. scripts/evaluate_test.sh code path   -> scoring/scores_test.json (+ scores.json)
#   4. hidden-test audit of better/weaker variants and shortcuts (hardening/submissions)
#   5. paired bootstrap + rank deltas       -> hardening/results/summary.{json,md}
set -euo pipefail
cd /home/user
# the test evaluator moves eval/test/ out of the workspace while submissions run;
# keep a copy so a preemption at that moment cannot lose it
mkdir -p tmp
if [ -f eval/test/evaluate.py ]; then
  rm -rf tmp/eval_test_backup; cp -r eval/test tmp/eval_test_backup
elif [ -f tmp/eval_test_backup/evaluate.py ]; then
  rm -rf eval/test; cp -r tmp/eval_test_backup eval/test
fi
# resume state is keyed on the code that produces the results, so a stale cache is never reused
key=$(cat eval/test/*.py eval/train/*.py method/*.py baseline/*.py submissions/*.sh hardening/submissions/*.sh \
      hardening/shortcuts.py | sha1sum | cut -c1-12)
state=tmp/job_state_$key
mkdir -p "$state" hardening/results
export ALIGNPOSE_CACHE_DIR=/home/user/$state/alignpose_cache
echo "cpus $(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
./scripts/download.sh --offline
if [ ! -f "$state/train_generated" ]; then
  python -m eval.test.scenegen --make-train 2>&1 | tail -2
  touch "$state/train_generated"
fi
if [ ! -f "$state/train_done" ]; then
  python -m eval.train.evaluate --resume-dir "$state/train" 2>&1 | grep --line-buffered -v "Loading weights" \
    | tee hardening/results/evaluate_train.log
  touch "$state/train_done"
fi
if [ ! -f "$state/test_done" ]; then
  python -m eval.test.evaluate --resume-dir "$state/test" 2>&1 | grep --line-buffered -v "Loading weights" \
    | tee hardening/results/evaluate_test.log
  cp scoring/scores_test.json scoring/scores.json
  touch "$state/test_done"
fi
if [ ! -f "$state/audit_done" ]; then
  python -m eval.test.evaluate --submissions-dir hardening/submissions --resume-dir "$state/test" \
    --output hardening/results/scores_test_audit.json --details hardening/results/details_test_audit.json \
    --save-predictions hardening/results/test_predictions 2>&1 | grep --line-buffered -v "Loading weights" \
    | tee hardening/results/audit_test.log
  touch "$state/audit_done"
fi
python hardening/analyze.py --scores hardening/results/scores_test_audit.json \
  --details hardening/results/details_test_audit.json --out hardening/results/summary.json --boot 300 \
  2>&1 | tee hardening/results/summary.md
