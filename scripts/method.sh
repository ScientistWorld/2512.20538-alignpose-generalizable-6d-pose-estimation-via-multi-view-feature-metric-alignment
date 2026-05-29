#!/bin/bash
# Run the AlignPose-style multi-view aggregation and feature-metric refinement.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs

python -m method.alignpose \
  --dataset data/synthetic_benchmark.json \
  --output outputs/alignpose_predictions.json
