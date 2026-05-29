#!/bin/bash
# Job script executed by the scheduler.

set -euo pipefail

cd /home/user
./scripts/evaluate_train.sh
./scripts/evaluate_test.sh
