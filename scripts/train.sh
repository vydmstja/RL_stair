#!/usr/bin/env bash
set -euo pipefail

cyclo_dir="${CYCLO_LAB_DIR:-/workspace/cyclo_lab}"
isaac_python="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
num_envs="${NUM_ENVS:-4096}"
max_iterations="${MAX_ITERATIONS:-10000}"

cd "${cyclo_dir}"
exec "${isaac_python}" \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Cyclo-Velocity-Stairs-K1-Rev1-v0 \
  --num_envs "${num_envs}" \
  --max_iterations "${max_iterations}" \
  --headless \
  "$@"
