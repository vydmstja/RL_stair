# RL_stair

ROBOTIS K1 Rev1을 Isaac Lab/Cyclo Lab에서 **외부 지형 센서 없이** 학습하기 위한
stair locomotion overlay입니다.

이 저장소는 Cyclo Lab 전체 복사본이 아닙니다. 대상 서버에 이미 존재하는
`cyclo_lab` checkout 위에 `overlay/`를 적용합니다. 학습 actor에는 카메라,
depth image, ray-caster height scan 또는 계단 형상 정보가 들어가지 않습니다.

## 포함 내용

- 평지 접근 구간과 3단 오르막/내리막이 교대로 배치되는 procedural terrain
- 계단 높이 `0.03~0.14 m`, 디딤판 깊이 `0.25/0.30/0.38 m` curriculum
- K1 custom URDF와 해당 mesh
- 목표 속도 추종, 자세 안정성, 발 clearance/slide, 충격 및 action smoothness 보상
- IMU/관절/이전 action/목표 속도/목표 상대 위치의 15-frame history
- RSL-RL PPO 설정: 최대 `10000` iterations, 100 iterations마다 checkpoint
- 환경 시각화, 학습, TensorBoard 실행 스크립트

## 요구 사항

- NVIDIA GPU와 호환되는 NVIDIA driver
- Isaac Sim/Isaac Lab/Cyclo Lab이 설치된 기존 컨테이너
- 컨테이너 내부 Cyclo Lab 경로: 기본값 `/workspace/cyclo_lab`
- Isaac Sim Python: 기본값 `/isaac-sim/python.sh`

## 1. Cyclo Lab에 적용

호스트에서:

```bash
git clone https://github.com/vydmstja/RL_stair.git
cd RL_stair
./install.sh /path/to/cyclo_lab
```

현재 서버 예시:

```bash
./install.sh /home/robotis-ai/eunseom/k1_rl_tasks/cyclo_lab
```

`install.sh`는 덮어쓸 기존 파일을 `.rl_stair_backup/<timestamp>/`에 먼저
백업합니다. Cyclo Lab checkout은 쓰기 가능한 상태여야 합니다.

## 2. 환경 시각화

컨테이너 안에서:

```bash
cd /workspace/cyclo_lab
/path/to/RL_stair/scripts/visualize.sh
```

또는 직접:

```bash
/isaac-sim/python.sh scripts/tools/visualize_stairs.py \
  --task Cyclo-Velocity-Stairs-K1-Rev1-Play-v0 \
  --num_envs 12
```

## 3. 학습

컨테이너 안에서:

```bash
cd /workspace/cyclo_lab
/path/to/RL_stair/scripts/train.sh
```

직접 실행하려면:

```bash
/isaac-sim/python.sh \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Cyclo-Velocity-Stairs-K1-Rev1-v0 \
  --num_envs 4096 \
  --max_iterations 10000 \
  --headless
```

GPU 메모리가 부족하면 `--num_envs`를 2048 또는 1024로 낮춥니다.

## 4. TensorBoard

컨테이너에서:

```bash
cd /workspace/cyclo_lab
/path/to/RL_stair/scripts/tensorboard.sh
```

로컬 PC에서 SSH tunnel:

```bash
ssh -L 6006:localhost:6006 robotis-ai@SERVER_IP
```

브라우저에서 `http://localhost:6006`을 엽니다.

## 등록된 task

- 학습: `Cyclo-Velocity-Stairs-K1-Rev1-v0`
- 시각화: `Cyclo-Velocity-Stairs-K1-Rev1-Play-v0`

## Blind 정책 범위

Actor는 height scanner를 사용하지 않습니다. 상대 목표 위치도 실제 배포 시
odometry/localization으로 제공할 수 있을 때만 동일하게 사용해야 합니다.
실제 K1 배포 전에는 action scale, joint 순서, PD gain, 제어 주기, 관절 한계,
IMU 좌표계와 observation normalization을 반드시 동일하게 맞춰야 합니다.

학습 로그, checkpoint, Isaac Sim cache 및 export된 policy는 저장소에 포함하지
않습니다.
