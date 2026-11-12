#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/run_case.py honest_denser
