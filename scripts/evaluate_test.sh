#!/bin/bash
# Evaluate held-out test-slice prediction artifacts and write scores_test.json.
# This script imports only eval/ code and reads outputs produced by methods.

set -euo pipefail

cd "$(dirname "$0")/.."

python -m eval.test.evaluate \
  --dataset data/synthetic_benchmark.json \
  --outputs-dir outputs \
  --train-scores scoring/scores_train.json \
  --output scoring/scores_test.json
