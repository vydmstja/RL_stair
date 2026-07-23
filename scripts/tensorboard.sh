#!/usr/bin/env bash
set -euo pipefail

cyclo_dir="${CYCLO_LAB_DIR:-/workspace/cyclo_lab}"
isaac_python="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
port="${TENSORBOARD_PORT:-6006}"

cd "${cyclo_dir}"
exec "${isaac_python}" -m tensorboard.main \
  --logdir logs/rsl_rl \
  --host 0.0.0.0 \
  --port "${port}"
