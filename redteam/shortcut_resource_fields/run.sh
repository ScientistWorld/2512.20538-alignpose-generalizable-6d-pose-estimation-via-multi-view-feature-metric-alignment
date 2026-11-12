#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/run_case.py shortcut_resource_fields
