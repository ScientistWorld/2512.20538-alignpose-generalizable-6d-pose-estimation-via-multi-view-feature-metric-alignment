#!/bin/bash
# Evaluate prediction artifacts and write scoring/scores.json.
# This script imports only eval/ code and reads outputs produced by methods.

set -euo pipefail

cd "$(dirname "$0")/.."

python -m eval.pose_metrics \
  --dataset data/synthetic_benchmark.json \
  --outputs-dir outputs \
  --output scoring/scores.json
