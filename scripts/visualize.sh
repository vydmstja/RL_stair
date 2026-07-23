#!/usr/bin/env bash
set -euo pipefail

cyclo_dir="${CYCLO_LAB_DIR:-/workspace/cyclo_lab}"
isaac_python="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
num_envs="${NUM_ENVS:-12}"

cd "${cyclo_dir}"
exec "${isaac_python}" \
  scripts/tools/visualize_stairs.py \
  --task Cyclo-Velocity-Stairs-K1-Rev1-Play-v0 \
  --num_envs "${num_envs}" \
  "$@"
