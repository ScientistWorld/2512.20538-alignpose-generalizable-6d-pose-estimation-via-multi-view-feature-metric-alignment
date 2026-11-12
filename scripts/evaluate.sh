#!/bin/bash
# Convenience wrapper: train evaluation, hidden-test evaluation, and a copy of
# the test scores as the canonical scoring/scores.json.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/evaluate_train.sh
./scripts/evaluate_test.sh
cp scoring/scores_test.json scoring/scores.json
