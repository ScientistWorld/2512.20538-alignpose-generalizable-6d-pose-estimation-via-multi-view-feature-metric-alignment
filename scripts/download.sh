#!/bin/bash
# Estimated download size: ~90 MB (DINOv2-S weights). Disk usage: ~100 MB.
# Fetches the frozen DINOv2-S feature extractor used by the reference method
# into data/downloads/. Idempotent: existing files are kept.
# The visible train split (data/train/) and its labels (eval/train/) are small
# and tracked in git; the hidden test split is generated at test time by
# eval/test/, so no dataset download is needed.
#
# Usage: scripts/download.sh            # fetch missing files
#        scripts/download.sh --offline  # only verify (compute nodes have no internet)

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/downloads

offline=false
if [ "${1:-}" = "--offline" ]; then
  offline=true
fi

model_dir="data/downloads/dinov2-small"
base_url="https://huggingface.co/facebook/dinov2-small/resolve/main"
mkdir -p "$model_dir"
for file in config.json preprocessor_config.json model.safetensors; do
  if [ ! -s "$model_dir/$file" ]; then
    if [ "$offline" = true ]; then
      echo "Missing $model_dir/$file. Run scripts/download.sh on an internet-enabled node first." >&2
      exit 1
    fi
    curl -L --fail --retry 3 "$base_url/$file" -o "$model_dir/$file"
  fi
done

for f in data/train/index.json eval/train/train_gt.json; do
  if [ ! -s "$f" ]; then
    echo "Missing $f (tracked in git). Restore it with: git checkout -- $f" >&2
    exit 1
  fi
done
echo "download.sh: all assets present"
