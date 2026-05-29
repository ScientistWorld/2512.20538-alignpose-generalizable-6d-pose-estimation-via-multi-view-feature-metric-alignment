#!/bin/bash
# Estimated download size: 0 bytes. Disk usage: ~15 MB generated locally.
# The benchmark uses a deterministic synthetic multi-view pose dataset because
# the full BOP candidate files and heavyweight single-view estimators used by
# the paper are not bundled with the paper. No network fetch is required.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/downloads

if [ ! -f data/synthetic_benchmark.json ]; then
  python data/synthetic_alignpose.py --output data/synthetic_benchmark.json --scenes 12 --views 4
fi
