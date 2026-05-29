#!/bin/bash
# Run the unrefined single-view candidate baseline.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs

python -m baseline.single_view \
  --dataset data/synthetic_inputs.json \
  --output outputs/baseline_predictions.json
