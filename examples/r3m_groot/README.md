# Roll out the four frozen-encoder GR00T heads on SO-101

This integration loads the **protocol-B `.pt` checkpoints**, not a full GR00T VLA.
It uses the supplied `so101_encoders.py` and the training version of R3M, preserves
PCA, task embeddings, mean/std normalization, and strict-loads the 537-tensor
GR00T N1.7 action head. No checkpoint conversion or retraining is required.

It provides its own rollout CLI with the fork's SO-101 driver. These checkpoints
are not `PreTrainedPolicy` exports and are not accepted by `lerobot-train`,
`lerobot-record --policy.path`, or the generic `lerobot-rollout` command.

## Environment

The default asset root is `/mnt/khanh/work/r3m`, containing:

```text
.venv/                Python 3.12 deployment runtime
Isaac-GR00T/          existing GR00T N1.7 source
r3m-src/              training R3M source, including models_dinov2.py
groot_base/           GR00T config.json (the VLM weights are not loaded)
ckpt/
  so101_encoders.py
  checkpoints/p256_<arm>_s0.pt
  encoders/
  sample_frames/ep002.npz, ep005.npz
```

From this LeRobot checkout:

```bash
uv pip install --python /mnt/khanh/work/r3m/.venv/bin/python \
  -r examples/r3m_groot/requirements.txt
```

Use the launcher below; it imports this checkout directly. Do not `uv sync` this
runtime or install the general LeRobot extras into it: the supplied GR00T code
requires Transformers 4.57.3/Hub 0.x, while the fork's main training dependencies
use Transformers 5/Hub 1.x. The dedicated requirements cover this deployment
path only. `R3M_ASSETS` relocates the assets, and `R3M_PYTHON` overrides the runtime.
The tested runtime uses PyTorch 2.11/CUDA 12.8, rather than training's PyTorch 2.9.
CPU smoke inference works with this build even without a working GPU driver.
The supplied Isaac-GR00T checkout is revision
`d2b7e75b937e3ec9aa5dbc798f08b89692c49734`.

The old R3M repository in the parent workspace lacks the VIGIL/SigLIP classes;
the launcher selects `r3m-src` from the asset bundle instead. It does not change
`HOME`, download GR00T weights, or modify the supplied checkpoints.

For SigLIP-2, the supplied encoder constructor first loads Google's base model,
then replaces its vision weights with the trained snapshot. Cache that base once
(it has already been cached on this machine):

```bash
/mnt/khanh/work/r3m/.venv/bin/python -c \
  'from huggingface_hub import snapshot_download; snapshot_download("google/siglip2-base-patch16-224", allow_patterns=["config.json", "model.safetensors"])'
```

Once cached, `HF_HUB_OFFLINE=1` can be used for entirely offline inference.

## Inspect and replay

VIGIL is used in the examples. Substitute any checkpoint from this table:

| Encoder | Checkpoint suffix `<arm>` |
|---|---|
| VIGIL | `semvggtglobaldelta_corpusnoreg-tcn01_bs128x2` |
| R3M | `official` |
| SigLIP-2 | `siglipfrozen` |
| VC-1 | `vc1` |

```bash
ckpt=/mnt/khanh/work/r3m/ckpt/checkpoints/p256_semvggtglobaldelta_corpusnoreg-tcn01_bs128x2_s0.pt
bash examples/r3m_groot/rollout.sh --ckpt "$ckpt" --inspect

# CPU smoke check; use cuda:0 on a machine with a working NVIDIA driver.
bash examples/r3m_groot/rollout.sh --ckpt "$ckpt" --device cpu \
  --robot replay --replan 1 --max-steps 1 --out /tmp/r3m-smoke.json

# Dense held-out replay, including full future-chunk error.
bash examples/r3m_groot/rollout.sh --ckpt "$ckpt" --device cuda:0 \
  --robot replay --replan 1 --max-steps 10000 \
  --episode /mnt/khanh/work/r3m/ckpt/sample_frames/ep002.npz \
  --out /tmp/r3m-replay.json
```

`raw_mae` measures the executed portion of the unguarded predictions;
`raw_chunk_mae` measures all available future targets in each prediction.
`guarded_mae` separately measures targets after motion limiting against recorded
state. Arm errors are degrees, gripper errors are percentage points, so the
aggregate is mixed units. A single bundled episode is not the full validation
split and need not reproduce the reported 4–5 aggregate MAE. Replay never
connects to hardware and does not pretend to simulate the robot's dynamics.

## Live SO-101

Use the **same calibration and camera geometry as data collection**. The driver
uses degrees for the five arm joints and the calibrated 0–100 range for gripper.
No second normalization or conversion to relative actions is applied.

Specify your existing calibration ID and devices; the integration never starts
interactive calibration inside a policy rollout. Calibrate separately with
LeRobot's normal `lerobot-calibrate --robot.type=so101_follower` workflow if needed.

```bash
# First inspect live observations/predictions without sending position targets.
bash examples/r3m_groot/rollout.sh --ckpt "$ckpt" --robot so101 \
  --device cuda:0 --robot-id YOUR_CALIBRATED_ID --port /dev/ttyACM0 \
  --front-cam 0 --wrist-cam 2 --task 'Put the cup into the box' \
  --replan 4 --max-step 3 --max-steps 30 --dry-run

# Execute a short rollout with the same setup.
bash examples/r3m_groot/rollout.sh --ckpt "$ckpt" --robot so101 \
  --device cuda:0 --robot-id YOUR_CALIBRATED_ID --port /dev/ttyACM0 \
  --front-cam 0 --wrist-cam 2 --task 'Put the cup into the box' \
  --replan 4 --max-step 3 --max-steps 300 --out /tmp/r3m-live.json
```

The other supported tasks are exactly `Put the tape into the box` and
`Put the tape into the cup`. Camera indices are integers; stable `/dev/v4l/by-id/…`
paths are also accepted. Camera output is explicitly RGB and is resized with
OpenCV `INTER_AREA`, matching frame extraction before the supplied encoder.
Use `--calibration-dir` if your calibration files are outside LeRobot's default.

The first command and every subsequent command are limited against **measured
position**, including LeRobot's own `max_relative_target` check. Predictions are
clipped to the training target envelope; when the measured pose starts outside
it, the slew limit takes precedence so the robot cannot snap into the envelope.
NaN/Inf values and inference older than `--max-inference-seconds` (default 2s)
abort before sending that chunk. `--dry-run` still connects/configures hardware
and can enable torque; only cached replay is hardware-free.

The default retains motor torque when disconnecting. `Ctrl-C` stops commands
and closes cameras/serial without claiming that disabling torque holds a pose.
Use `--disable-torque-on-disconnect` only if releasing the arm at exit is intended.

This is synchronous execution: inference pauses command dispatch. Reported
effective Hz includes inference, and it may be below `--fps`. At 30 Hz,
`--replan 4` gives only 0.133s per chunk. Measure on your GPU; increasing replan
amortizes inference but increases open-loop execution. Lowering replan does
**not** fix a slow model. This does not implement asynchronous RTC.

## Known data issue and validation limits

The supplied checkpoints have wrist-roll state mean around 87.6 and state std
around 0.188, but action mean around 153.3. In sample episode 2 the wrist-roll
observation is constant around 87.5. This looks like a sensor/data-collection
issue and must be checked against your arm; this adapter does not fabricate a
wrist offset or claim that an offline score establishes physical success.

Automated tests cover preprocessing, normalization, target leakage, measured-state
limits, first-command behavior, dry-run, stale inference, partial connection
cleanup, and the current LeRobot config. Hardware and GPU timing require your
physical setup. Only load trusted `.pt` artifacts: these training checkpoints
contain pickled NumPy metadata and require `weights_only=False`.
