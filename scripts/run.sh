#!/bin/bash
# Scheduler job (smoke stage of the hardening iteration).
set -euo pipefail
cd /home/user
mkdir -p tmp/smoke
python - <<'PY'
import os, numpy, torch, transformers, PIL
print("numpy", numpy.__version__, "torch", torch.__version__, "transformers", transformers.__version__, "PIL", PIL.__version__)
print("cpus", len(os.sched_getaffinity(0)), "os.cpu_count", os.cpu_count())
PY
start=$(date +%s)
python -m eval.test.scenegen --make-train 2>&1 | tail -5
echo "train generation took $(( $(date +%s) - start ))s"
./scripts/download.sh --offline
start=$(date +%s)
python -m eval.test.evaluate --smoke --submissions-dir hardening/submissions \
  --output tmp/smoke/scores.json --details tmp/smoke/details.json --save-predictions tmp/smoke/preds 2>&1 | tee tmp/smoke/smoke.log
echo "smoke took $(( $(date +%s) - start ))s"
start=$(date +%s)
python -m eval.train.evaluate --details tmp/smoke/train_details.json --save-predictions tmp/smoke/train_preds 2>&1 | tee tmp/smoke/train.log
echo "train evaluation took $(( $(date +%s) - start ))s"
