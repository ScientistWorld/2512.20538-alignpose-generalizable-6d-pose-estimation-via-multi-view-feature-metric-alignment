#!/bin/bash
# Estimated download size: ~85 MB. Disk usage: ~100 MB including generated data.
# Downloads a local DINOv2-S feature extractor and generates the deterministic
# synthetic multi-view benchmark used by the reduced reproduction.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/downloads

model_dir="data/downloads/dinov2-small"
base_url="https://huggingface.co/facebook/dinov2-small/resolve/main"
mkdir -p "$model_dir"
[ -f "$model_dir/config.json" ] || curl -L --fail "$base_url/config.json" -o "$model_dir/config.json"
[ -f "$model_dir/preprocessor_config.json" ] || curl -L --fail "$base_url/preprocessor_config.json" -o "$model_dir/preprocessor_config.json"
[ -f "$model_dir/model.safetensors" ] || curl -L --fail "$base_url/model.safetensors" -o "$model_dir/model.safetensors"

if [ ! -f data/synthetic_benchmark.json ]; then
  python data/synthetic_alignpose.py --output data/synthetic_benchmark.json --scenes 12 --views 4
fi
