#!/bin/bash
# Scheduler job for the hardening iteration: regenerate the visible train split,
# score the reference + baselines on train and on the hidden test, then audit
# variants and shortcuts on the hidden test.
set -euo pipefail
cd /home/user
mkdir -p hardening/results
python - <<'PY'
import os
print("cpus", len(os.sched_getaffinity(0)))
PY
start=$(date +%s)
python -m eval.test.scenegen --make-train 2>&1 | tail -2
echo "train generation took $(( $(date +%s) - start ))s"
./scripts/download.sh --offline
start=$(date +%s)
./scripts/evaluate_train.sh 2>&1 | tee hardening/results/evaluate_train.log
echo "evaluate_train took $(( $(date +%s) - start ))s"
start=$(date +%s)
./scripts/evaluate_test.sh 2>&1 | tee hardening/results/evaluate_test.log
echo "evaluate_test took $(( $(date +%s) - start ))s"
cp scoring/scores_test.json scoring/scores.json
start=$(date +%s)
python -m eval.test.evaluate --submissions-dir hardening/submissions \
  --output hardening/results/scores_test_audit.json --details hardening/results/details_test_audit.json \
  --save-predictions hardening/results/test_predictions 2>&1 | tee hardening/results/audit_test.log
echo "audit took $(( $(date +%s) - start ))s"
python hardening/analyze.py --scores hardening/results/scores_test_audit.json \
  --details hardening/results/details_test_audit.json --out hardening/results/summary.json \
  2>&1 | tee hardening/results/summary.md
