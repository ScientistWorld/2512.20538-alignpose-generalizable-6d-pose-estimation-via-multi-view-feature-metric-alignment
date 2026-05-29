#!/bin/bash
# Job script executed by the scheduler.

set -euo pipefail

cd /home/user
./scripts/reproduce.sh
