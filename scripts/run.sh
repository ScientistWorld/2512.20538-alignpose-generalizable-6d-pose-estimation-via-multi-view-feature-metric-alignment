#!/bin/bash
# Scheduler job (hardening, calibration step): dev-split calibration study.
# Resumable: finished generations and submission runs are cached under tmp/diag.
set -euo pipefail
cd /home/user
echo "cpus $(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
./scripts/download.sh --offline
python hardening/diag_calibration.py --out hardening/results/diag_calibration.json 2>&1 | grep -v "Loading weights"
