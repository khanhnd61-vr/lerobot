# VIGIL rollout on SO-101

This guide runs the VIGIL policy checkpoint supplied in `/mnt/khanh/work/r3m`.
The policy is the frozen VIGIL visual encoder plus the GR00T N1.7 action head. It
predicts absolute joint targets: five arm joints are in degrees and the gripper
is in LeRobot's calibrated `0..100` range.

The rollout command is synchronous. It reads both cameras and the measured joint
state, predicts a short action chunk, limits each command against the measured
pose, and sends the command through LeRobot's SO-101 driver.

## 1. Check the files

The commands below assume this checkout is `/home/khanh/work/r3m/lerobot` and the
model assets are `/mnt/khanh/work/r3m`:

```bash
cd /home/khanh/work/r3m/lerobot
export R3M_ASSETS=/mnt/khanh/work/r3m
export R3M_PYTHON=$R3M_ASSETS/.venv/bin/python

test -f "$R3M_ASSETS/ckpt/checkpoints/p256_semvggtglobaldelta_corpusnoreg-tcn01_bs128x2_s0.pt"
test -d "$R3M_ASSETS/Isaac-GR00T"
test -d "$R3M_ASSETS/r3m-src"
```

If the environment has not been installed, run this once:

```bash
uv pip install --python "$R3M_PYTHON" \
  -r examples/r3m_groot/requirements.txt
```

The GR00T action-head code needs Transformers 4.57.3, so keep this deployment
environment separate from a LeRobot training environment.

## 2. Choose the VIGIL checkpoint

```bash
export VIGIL_CKPT="$R3M_ASSETS/ckpt/checkpoints/p256_semvggtglobaldelta_corpusnoreg-tcn01_bs128x2_s0.pt"
```

Inspect its task names and normalization statistics without loading the network:

```bash
bash examples/r3m_groot/rollout.sh --ckpt "$VIGIL_CKPT" --inspect
```

The valid tasks are exactly:

```text
Put the cup into the box
Put the tape into the box
Put the tape into the cup
```

## 3. Replay offline before touching the arm

Replay never opens a serial port or camera. Start with one CPU step to verify the
paths and checkpoint:

```bash
bash examples/r3m_groot/rollout.sh \
  --ckpt "$VIGIL_CKPT" \
  --device cpu \
  --robot replay \
  --episode "$R3M_ASSETS/ckpt/sample_frames/ep002.npz" \
  --replan 1 \
  --max-steps 1 \
  --out /tmp/vigil-smoke.json
```

On a machine with a working NVIDIA driver, use the GPU for the full cached
episode:

```bash
bash examples/r3m_groot/rollout.sh \
  --ckpt "$VIGIL_CKPT" \
  --device cuda:0 \
  --robot replay \
  --episode "$R3M_ASSETS/ckpt/sample_frames/ep002.npz" \
  --replan 4 \
  --max-steps 10000 \
  --out /tmp/vigil-replay.json
```

Read `raw_mae` for the policy prediction error and `guarded_mae` for the error
after safety limiting. Replay is an offline check; it does not establish task
success on the physical arm.

## 4. Calibrate the robot separately

Use LeRobot's calibration command with your own robot ID and serial port. Do not
guess the ID: it selects the calibration file.

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=YOUR_CALIBRATED_ID
```

Use the same camera views and physical mounting used for the training data. Find
camera devices with:

```bash
lerobot-find-cameras opencv
```

## 5. Run a live dry run

Set all four hardware values explicitly. The command connects to the arm and
cameras, so keep the arm in a safe position; `--dry-run` prevents position targets
from being sent but does not make hardware connection risk-free.

```bash
bash examples/r3m_groot/rollout.sh \
  --ckpt "$VIGIL_CKPT" \
  --robot so101 \
  --device cuda:0 \
  --robot-id YOUR_CALIBRATED_ID \
  --port /dev/ttyACM0 \
  --front-cam 0 \
  --wrist-cam 2 \
  --task 'Put the cup into the box' \
  --replan 4 \
  --max-step 3 \
  --max-steps 30 \
  --dry-run
```

Replace `YOUR_CALIBRATED_ID`, `/dev/ttyACM0`, `0`, and `2` with your setup. The
front and wrist cameras must be different devices. Stable paths such as
`/dev/v4l/by-id/...` are accepted.

## 6. Execute a short rollout

After the dry run shows valid RGB frames, measured joint positions, and inference
latency below the configured limit, run a short motion test:

```bash
bash examples/r3m_groot/rollout.sh \
  --ckpt "$VIGIL_CKPT" \
  --robot so101 \
  --device cuda:0 \
  --robot-id YOUR_CALIBRATED_ID \
  --port /dev/ttyACM0 \
  --front-cam 0 \
  --wrist-cam 2 \
  --task 'Put the cup into the box' \
  --replan 4 \
  --max-step 3 \
  --max-steps 60 \
  --out /tmp/vigil-live.json
```

Start with `--max-step 3`. Increase it only after checking the motion and the
`clamped_box`/`clamped_rate` counters. The default disconnect behavior keeps
motor torque enabled. Add `--disable-torque-on-disconnect` only when you want the
arm to release torque at exit.

Press `Ctrl-C` to stop. The driver closes the cameras and motor bus in cleanup.

## Safety and troubleshooting

- `--task` must match one of the three strings exactly.
- Keep `--replan` at `4` for the first test. Larger values leave the arm open-loop
  for longer; smaller values do not make slow inference faster.
- The default inference timeout is two seconds. If inference is slower, fix the GPU
  setup or adjust `--max-inference-seconds` only after observing the latency.
- RGB camera frames are required. A BGR camera feed silently damages the policy's
  visual features.
- `clamped_box` means the prediction left the training envelope. `clamped_rate`
  means the prediction moved too far from the measured pose in one command.
- The supplied data has an almost constant wrist-roll observation near `87.5` while
  its action targets are near `153`. Check the real wrist-roll readback before
  attempting a task; this adapter does not invent an offset.
- If you see `CUDA is unavailable`, use CPU only for replay or install a working
  NVIDIA driver before a live rollout. CPU inference is too slow for normal control.

For the full implementation notes and validation results, see
[`examples/r3m_groot/README.md`](examples/r3m_groot/README.md) and
[`examples/r3m_groot/VALIDATION.md`](examples/r3m_groot/VALIDATION.md).
