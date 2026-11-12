#!/bin/bash
# Job script executed by the scheduler.

set -euo pipefail

cd /home/user
./scripts/download.sh --offline
./scripts/baseline.sh
./scripts/method.sh
./scripts/evaluate.sh
./scripts/evaluate_train.sh
./scripts/evaluate_test.sh
