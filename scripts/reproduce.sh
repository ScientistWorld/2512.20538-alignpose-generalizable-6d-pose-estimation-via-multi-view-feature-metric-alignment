#!/bin/bash
# End-to-end reproduction: fetch weights, then score the reference and the
# baselines on the visible train split and on the hidden test split.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/download.sh
./scripts/evaluate.sh
