#!/bin/bash
# End-to-end reproduction script for the reduced AlignPose benchmark.

set -euo pipefail

cd "$(dirname "$0")/.."

./scripts/download.sh
./scripts/baseline.sh
./scripts/method.sh
./scripts/evaluate.sh
