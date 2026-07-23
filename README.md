# RL_stair

This repository provides a stair-locomotion overlay for training the ROBOTIS
K1 Rev1 in Isaac Lab/Cyclo Lab **without exteroceptive terrain sensors**.

It is not a complete copy of Cyclo Lab. The files under `overlay/` are applied
to an existing `cyclo_lab` checkout on the target server. The training actor
does not receive camera images, depth images, ray-caster height scans, or
explicit stair geometry.

## Included

- Procedural terrain with a flat approach followed by alternating three-step
  ascending and descending stair courses
- Curriculum over stair heights from `0.03` to `0.14 m` and tread depths of
  `0.25`, `0.30`, and `0.38 m`
- Custom K1 URDF and required mesh assets
- Rewards for velocity tracking, posture stability, foot clearance, foot
  sliding, contact behavior, and action smoothness
- A 15-frame history of IMU, joint state, previous action, velocity command,
  and relative goal position observations
- RSL-RL PPO configuration with `10000` maximum iterations and checkpoints
  every 100 iterations
- Scripts for environment visualization, training, and TensorBoard

## Requirements

- NVIDIA GPU and a compatible NVIDIA driver
- An existing container with Isaac Sim, Isaac Lab, and Cyclo Lab installed
- Cyclo Lab path inside the container: `/workspace/cyclo_lab` by default
- Isaac Sim Python launcher: `/isaac-sim/python.sh` by default

## 1. Install the overlay

Run on the host:

```bash
git clone https://github.com/vydmstja/RL_stair.git
cd RL_stair
./install.sh /path/to/cyclo_lab
```

Example for the development server:

```bash
./install.sh /home/robotis-ai/eunseom/k1_rl_tasks/cyclo_lab
```

Before overwriting files, `install.sh` copies existing versions to
`.rl_stair_backup/<timestamp>/`. The Cyclo Lab checkout must be writable.

## 2. Visualize the environment

Run inside the container:

```bash
cd /workspace/cyclo_lab
/path/to/RL_stair/scripts/visualize.sh
```

Or run the underlying command directly:

```bash
/isaac-sim/python.sh scripts/tools/visualize_stairs.py \
  --task Cyclo-Velocity-Stairs-K1-Rev1-Play-v0 \
  --num_envs 12
```

## 3. Train

Run inside the container:

```bash
cd /workspace/cyclo_lab
/path/to/RL_stair/scripts/train.sh
```

Or run the underlying command directly:

```bash
/isaac-sim/python.sh \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Cyclo-Velocity-Stairs-K1-Rev1-v0 \
  --num_envs 4096 \
  --max_iterations 10000 \
  --headless
```

If GPU memory is insufficient, reduce `--num_envs` to 2048 or 1024.

## 4. Open TensorBoard

Run inside the container:

```bash
cd /workspace/cyclo_lab
/path/to/RL_stair/scripts/tensorboard.sh
```

Create an SSH tunnel from the local machine:

```bash
ssh -L 6006:localhost:6006 robotis-ai@SERVER_IP
```

Then open `http://localhost:6006` in a browser.

## Registered tasks

- Training: `Cyclo-Velocity-Stairs-K1-Rev1-v0`
- Visualization: `Cyclo-Velocity-Stairs-K1-Rev1-Play-v0`

## Blind-policy scope

The actor does not use a height scanner. The relative goal-position
observation should only be retained for real deployment if equivalent
odometry or localization is available on the robot.

Before deploying the policy to a physical K1, match the action scale, joint
ordering, PD gains, control frequency, joint limits, IMU coordinate frame, and
observation normalization between simulation and the real controller.

Training logs, checkpoints, Isaac Sim caches, and exported policies are
intentionally excluded from this repository.
