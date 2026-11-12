#!/bin/bash
# Held-out test evaluation. Takes no arguments.
# Generates the hidden test scenes at test time (eval/test/), runs every
# submission on disk (submissions/*.sh) on its own copy of those inputs,
# validates the raw detections, recomputes AR/AP and writes
# scoring/scores_test.json. Failed or invalid runs get AR = AP = 0.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m eval.test.evaluate
