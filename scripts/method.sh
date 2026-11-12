#!/bin/bash
# Run the reference AlignPose reproduction on the visible train inputs.
# Usage: scripts/method.sh [INPUT_DIR] [OUTPUT_JSON]
set -euo pipefail
cd "$(dirname "$0")/.."
input_dir="${1:-data/train}"
output="${2:-outputs/alignpose_train_predictions.json}"
mkdir -p "$(dirname "$output")"
bash submissions/alignpose.sh "$input_dir" "$output"
