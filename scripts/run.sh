#!/bin/bash
# Scheduler job for the hardening iteration. Resumable: the node may be
# preempted, so every stage records completion and the evaluators/method cache
# finished work under tmp/job_state (results are identical to a clean run).
set -euo pipefail
cd /home/user
state=tmp/job_state
mkdir -p "$state" hardening/results
export ALIGNPOSE_CACHE_DIR=/home/user/$state/alignpose_cache
echo "cpus $(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
if [ ! -f "$state/train_generated" ]; then
  python -m eval.test.scenegen --make-train 2>&1 | tail -2
  touch "$state/train_generated"
fi
./scripts/download.sh --offline
if [ ! -f "$state/train_done" ]; then
  # same code path as scripts/evaluate_train.sh, plus resume cache
  python -m eval.train.evaluate --resume-dir "$state/train" 2>&1 | grep -v "Loading weights" | tee hardening/results/evaluate_train.log
  touch "$state/train_done"
fi
if [ ! -f "$state/test_done" ]; then
  # same code path as scripts/evaluate_test.sh, plus resume cache
  python -m eval.test.evaluate --resume-dir "$state/test" 2>&1 | grep -v "Loading weights" | tee hardening/results/evaluate_test.log
  cp scoring/scores_test.json scoring/scores.json
  touch "$state/test_done"
fi
if [ ! -f "$state/audit_done" ]; then
  python -m eval.test.evaluate --submissions-dir hardening/submissions --resume-dir "$state/test" \
    --output hardening/results/scores_test_audit.json --details hardening/results/details_test_audit.json \
    --save-predictions hardening/results/test_predictions 2>&1 | grep -v "Loading weights" | tee hardening/results/audit_test.log
  touch "$state/audit_done"
fi
python hardening/analyze.py --scores hardening/results/scores_test_audit.json \
  --details hardening/results/details_test_audit.json --out hardening/results/summary.json \
  2>&1 | tee hardening/results/summary.md
