#!/bin/bash
# Run the image-free baselines on the visible train inputs.
# Usage: scripts/baseline.sh [INPUT_DIR] [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
input_dir="${1:-data/train}"
out_dir="${2:-outputs}"
mkdir -p "$out_dir"
for name in single_view_candidate multiview_aggregation multiview_aggregation_nms multiview_consensus; do
  bash "submissions/${name}.sh" "$input_dir" "$out_dir/${name}_train_predictions.json"
done
