#!/bin/bash
# Evaluate visible train-slice prediction artifacts and write scores_train.json.
# This script imports only eval/ code and reads outputs produced by methods.

set -euo pipefail

cd "$(dirname "$0")/.."

python -m eval.train.evaluate \
  --dataset data/synthetic_benchmark.json \
  --outputs-dir outputs \
  --output scoring/scores_train.json
