#!/bin/bash
# Estimated download size: ~85 MB. Disk usage: ~100 MB including generated data.
# Downloads a local DINOv2-S feature extractor and generates the deterministic
# synthetic multi-view benchmark used by the reduced reproduction.

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
  if [ ! -f "$model_dir/$file" ]; then
    if [ "$offline" = true ]; then
      echo "Missing $model_dir/$file. Run scripts/download.sh on an internet-enabled node before submitting." >&2
      exit 1
    fi
    curl -L --fail "$base_url/$file" -o "$model_dir/$file"
  fi
done

if [ ! -f data/synthetic_benchmark.json ] || ! python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("data/synthetic_benchmark.json").read_text())
raise SystemExit(0 if data["scenes"][0]["views"][0]["candidates"][0].get("T_CO") else 1)
PY
then
  python data/synthetic_alignpose.py --output data/synthetic_benchmark.json --scenes 12 --views 4
fi

if [ ! -f data/synthetic_inputs.json ] || python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("data/synthetic_inputs.json").read_text())
candidate = data["scenes"][0]["views"][0]["candidates"][0]
raise SystemExit(0 if candidate.get("T_WO") else 1)
PY
then
  python data/make_method_inputs.py \
    --source data/synthetic_benchmark.json \
    --output data/synthetic_inputs.json
fi
