#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/run_case.py shortcut_label_jitter
