#!/bin/bash
# Visible train evaluation. Takes no arguments.
# Runs every submission on disk (submissions/*.sh) on its own copy of the
# visible train inputs (data/train/), validates the raw detections, scores
# them against eval/train/train_gt.json and writes scoring/scores_train.json.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m eval.train.evaluate
