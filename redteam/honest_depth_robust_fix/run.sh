#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/run_case.py honest_depth_robust_fix
