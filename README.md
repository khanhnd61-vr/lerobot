# lerobot - SO-101 fork

A fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) carrying an
instruction-conditioned policy small enough to run at the robot, and two clients that drive
an arm against an external inference server.

Everything upstream works as documented upstream. This covers only what is different here.

## What this fork adds

|                                            |                                                                                                                                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **[IMPACT](src/lerobot/policies/impact/)** | ACT with a frozen T5-small language tower - the instruction enters as encoder tokens _and_ as FiLM on the ResNet stages. [Details](docs/source/impact.mdx)                           |
| **IMPACT int8 / QAT**                      | `--policy.int8_groups` runs selected GEMMs through a simulated W8A8 kernel, in training and at inference. A QAT checkpoint is an ordinary fp32 checkpoint.                           |
| **`lerobot-vla-cpp`**                      | Client for a [`vla.cpp`](https://github.com/VinRobotics/vla.cpp) server (ZeroMQ + protobuf). Does the per-arch preprocessing, so it covers SmolVLA, π0, π0.5 and GR00T N1.5/1.6/1.7. |
| **`lerobot-vla-simd`**                     | Client for a [`vla.simd`](https://github.com/cair-vinuni/vla.simd) server, over lerobot's own async-inference protocol.                                                              |
| **`loads_action_chunk`**                   | Lets the async client read chunks from a server that keeps lerobot and torch off its own machine.                                                                                    |

```bash
pip install -e ".[impact,async,vla-cpp]"
```

## Wiring - paste once per shell

**A command below copied without this will not run.** Substitute your own ports and ids;
`lerobot-find-port` reports the ports.

```bash
FOLLOWER_PORT=/dev/ttyACM0
LEADER_PORT=/dev/ttyACM1
ARM_ID=my_awesome_follower_arm
LEADER_ID=my_awesome_leader_arm

CAMS='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}'

ROBOT=(--robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$ARM_ID
       --robot.cameras="$CAMS")

# SmolVLA only - it was trained through this rename. IMPACT and ACT use the dataset's own keys.
RENAME=(--rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}')
```

**Camera keys must stay `front` and `wrist`, in that order.** The vision passes are
position-dependent, so a swapped pair gives plausible, wrong actions rather than an error.

## Record and train

Motor setup, calibration and teleoperation are stock lerobot - see the
[hardware guide](https://huggingface.co/docs/lerobot/il_robots). Recording:

```bash
lerobot-record "${ROBOT[@]}" \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID \
  --display_data=true --dataset.repo_id=$HF_USER/so101-tape \
  --dataset.num_episodes=20 --dataset.single_task="Put the tape into the box"
```

For a multi-task dataset, record each instruction into the same `repo_id` with
`--dataset.no_stamp=true` and `--resume=true` after the first run, changing only
`--dataset.single_task`.

Then curate. **A mislabelled episode costs more than a missing one** - a
language-conditioned policy learns the wrong instruction from it.

```bash
lerobot-dataset-viz --repo-id $HF_USER/so101-long --root $DATA_ROOT/so101-long --episode-index 7

lerobot-edit-dataset --repo_id $HF_USER/so101-long --root $DATA_ROOT/so101-long \
  --new_repo_id $HF_USER/so101-long-clean --new_root $DATA_ROOT/so101-long-clean \
  --operation.type delete_episodes --operation.episode_indices "[3, 17, 42]"
```

```bash
# IMPACT - no --rename_map, it trains on the dataset's own front/wrist keys.
# Add --policy.int8_groups=63 for quantization-aware training.
lerobot-train --policy.type=impact \
  --dataset.repo_id=$HF_USER/so101-multi-task-clean \
  --batch_size=8 --steps=4000 --save_freq=1000 \
  --output_dir=outputs/train/impact --job_name=impact --policy.device=cuda

# SmolVLA - needs the rename
lerobot-train --policy.path=lerobot/smolvla_base "${RENAME[@]}" \
  --dataset.repo_id=$HF_USER/so101-multi-task-clean \
  --batch_size=8 --steps=4000 --policy.scheduler_decay_steps=4000 \
  --output_dir=outputs/train/smolvla --job_name=smolvla --policy.device=cuda
```

## Checkpoints and tasks

| Model             | `so101-long-clean`                   | `so101-multi-task-clean`                       |
| ----------------- | ------------------------------------ | ---------------------------------------------- |
| IMPACT            | `khanhnd61/impact_so101-long-clean`  | `khanhnd61/impact_so101-multi-task-clean`      |
| IMPACT-int8 (QAT) | -                                    | `khanhnd61/impact-int8_so101-multi-task-clean` |
| SmolVLA           | `khanhnd61/smolvla_so101-long-clean` | `khanhnd61/smolvla_so101-multi-task-clean`     |

**`--task` must match a trained string exactly.** A reworded instruction is not an error,
just a worse policy.

- `so101-long-clean` - `Put the tape in the drawer`, the only one.
- `so101-multi-task-clean` - `Put the tape into the box` · `Put the tape into the cup` ·
  `Put the cup into the box`

Note `in the drawer` against `into the ...`; do not normalise them. Only the multi-task
checkpoint can test the language tower - three instructions over a shared scene. On
`so101-long-clean` every frame carries the same sentence, so a good rollout there says
nothing about instruction following.

## Rolling out

### PyTorch

```bash
lerobot-rollout --strategy.type=base \
  --policy.path=khanhnd61/impact_so101-multi-task-clean \
  --policy.device=cuda "${ROBOT[@]}" \
  --task="Put the tape into the box" --duration=60 --display_data=true
```

The int8 checkpoint declares `int8_groups: 63`, so a plain rollout reproduces the W8A8
rounding a deployment kernel will run. This is fake quantization, not fast quantization -
the arithmetic underneath is still fp32, so it is ~1.3× _slower_. What it buys is watching
the deployment numerics move a real arm.

```bash
--policy.int8_groups=0     # fp32 numerics - the control
--policy.int8_groups=47    # everything except the decoder  <- prefer this on hardware
--policy.int8_groups=63    # every group (the checkpoint's own default)
```

**Prefer 47 on hardware.** Quantizing the decoder raises within-chunk jerk ~46% and QAT
cannot fix it: the objective is L1 on the chunk and is indifferent to roughness between two
correct waypoints. The chunk executes open loop, so that roughness reaches the servos as
shaking.

### Through a `vla.cpp` server

[`vla.cpp`](https://github.com/VinRobotics/vla.cpp) serves one self-contained GGUF per
checkpoint with no Python at inference, and has a backend per target: CUDA from consumer
cards down to Jetson, Metal, Intel GPU/NPU via SYCL and OpenVINO, Adreno and Hexagon on
Snapdragon, and CPU.

**Run it on a GPU.** The backend decides whether the policy keeps up, and it is not close.
SmolVLA latency per query, from the engine's own
[benchmarks](https://github.com/VinRobotics/vla.cpp/tree/main/docs/benchmark), against the
1.67 s of motion a 50-step chunk buys at 30 fps:

| Backend              | SmolVLA | Headroom                     |
| -------------------- | ------: | ---------------------------- |
| RTX 5090 (CUDA)      | 38.2 ms | 44x                          |
| RTX 3060 (CUDA)      |  125 ms | 13x                          |
| Jetson AGX Orin      |  218 ms | 7.6x                         |
| Apple M4 (Metal)     |  358 ms | 4.7x                         |
| Core i7-14700F (CPU) | 1689 ms | 1.0x - at the stall boundary |

The CPU row has no margin: the next chunk lands as the last runs out, and any jitter starves
the queue. Use CPU when the board has no accelerator, not by default.

```bash
vla-server "$VLA_GGUF"          # binds tcp://*:5555; --bind moves it, -hf takes a Hub id
```

```bash
# no arm attached - prints round-trip latency next to the recorded actions
lerobot-vla-cpp --server_address=tcp://127.0.0.1:5555 --arch=smolvla \
  --replay.repo_id=khanhnd61/so101-multi-task-clean --replay.steps=10

lerobot-vla-cpp --server_address=tcp://127.0.0.1:5555 --arch=smolvla "${ROBOT[@]}" \
  --task="Put the tape into the box" --n_action_steps=25 --fps=30 --duration=60
```

`--arch` selects the preprocessing, not just a label: **the client tokenizes, resizes and
normalizes**, because the server is sent token ids and normalized floats. The wrong arch
loads, runs and returns plausible actions. Choices: `smolvla`, `pi0`, `pi05`,
`gr00t_n1_5/6/7`, and `passthrough` for a server that preprocesses itself.

`--stats_json` is required for `pi05` and the GR00T archs; GR00T also takes `--embodiment`,
and `--rel_stats_json` for an N1.7 checkpoint trained with relative actions.

<!-- prettier-ignore -->
> **Async inference is not supported yet.** `vla-server` answers one request at a time, so
> the loop is synchronous and `--n_action_steps` is exactly the feedback rate - 25 at 30 fps
> leaves ~0.83 s between observations.

### Through a `vla.simd` server

Needs the separate [`vla.simd`](https://github.com/cair-vinuni/vla.simd) checkout; this repo
owns only the client, which speaks the stock async-inference protocol.

```bash
cd $VLA_SIMD
cmake -S . -B build && cmake --build build -j --target vla_simd_impact

# Pass a LOCAL checkpoint directory, never a Hub id, and always pass --instruction.
uv run --project $LEROBOT python tools/impact/dump_impact_golden.py \
  --checkpoint $TRAIN_ROOT/impact_so101-multi-task-clean/checkpoints/last/pretrained_model \
  --out build/impact-multi --instruction "Put the tape into the box" --frames 8

OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
uv run --project $LEROBOT python serve/impact_policy_server.py \
  --model-dir build/impact-multi --host 0.0.0.0 --port 8080
```

**Both of those flags fail silently.** Given a Hub id the converter finds no normalizer and
falls back to _identity_ normalization - the engine then emits normalized actions (|a| ≈ 1)
as if they were degrees, and the arm slews somewhere nothing like the training distribution.
Read the dumper's `stats` line before serving. `--instruction` has a hardcoded default and
becomes the instruction the server uses when the client sends no `--task`.

```bash
lerobot-vla-simd --server_address=127.0.0.1:8080 --policy_type=impact \
  --pretrained_name_or_path=khanhnd61/impact_so101-multi-task-clean \
  "${ROBOT[@]}" --task="Put the tape into the box" \
  --actions_per_chunk=25 --aggregate_fn_name=latest_only --fps=30
```

`--actions_per_chunk=25`, not 50, on a desktop: IMPACT re-plans only when its queue drains,
so that number _is_ the feedback rate. `--aggregate_fn_name=latest_only` is worth trying -
the default blends two chunks predicted from _different_ observations, which smears two
plans together rather than following either.

On a remote board the arm stays on the workstation and the client points at it. The payload
is ~1.8 MB per query, so use wired gigabit, and note **the protocol is pickle over an
unauthenticated port**: lab LAN or an SSH tunnel, never an open network. A Pi 5 clears the
30 Hz loop for IMPACT at 41.2 Hz cool / 33.5 Hz soaked, so it needs
`--actions_per_chunk=50 --chunk_size_threshold=0.85` where a desktop does not.

## Sizing the queue

A refill fires once the queue drops below `actions_per_chunk × chunk_size_threshold`, and
must arrive before it empties:

```
chunk_size_threshold  >  inference_latency × fps / actions_per_chunk
```

At a full chunk the right-hand side is the **stall number** - query latency over the
duration of the chunk it produces.

| Latency | `--actions_per_chunk` | Minimum threshold at 30 fps               |
| ------: | --------------------: | ----------------------------------------- |
|  200 ms |                    50 | 0.12 - default 0.5 is fine                |
|  800 ms |                    50 | 0.48 - default 0.5 is marginal            |
| 1300 ms |                    50 | 0.78 - run at 0.85                        |
| 1700 ms |                    50 | **1.02 - impossible, no threshold works** |

Clearing the threshold only means the queue never starves; how closed-loop the rollout is
depends on how often a fresh observation arrives, which is
`max(A(1 - t) / fps, latency)`. A desktop is queue-bound, so the flags decide everything; a
Pi 5 is latency-bound, and refilling earlier cannot buy feedback the engine has no time to
produce.

**If the queue drains, raise `--chunk_size_threshold` or `--actions_per_chunk` - never lower
them.** Both sit in the denominator. When the minimum exceeds 1.0 no flag rescues it: use a
faster host, a smaller model, or int8.

## Gotchas

- **`--rename_map` is a top-level `lerobot-rollout` flag**, not `--policy.rename_map`. It
  applies to SmolVLA only, and to neither engine client.
- **The engine servers match cameras by name** and refuse a session they cannot match. The
  PyTorch path does not check, so a swapped `front`/`wrist` shows up only as odd behaviour.
- **Flow matching is stochastic.** SmolVLA draws fresh noise per query, so differing
  trajectories between runs are correct. Pin it server-side to compare two runs.

## Status

Covered by tests: IMPACT and its int8 path in
[`tests/policies/impact/`](tests/policies/impact/), the `vla.cpp` client in
[`tests/vla_cpp/`](tests/vla_cpp/), including its protocol handling against a real ZeroMQ
server. What tests cannot cover is whether an engine agrees.

- **The end-to-end arm rollouts here have not all been run against hardware.** Latency
  figures are the engines' published measurements on other checkpoints.
- **The GR00T paths of `lerobot-vla-cpp` are untested against real checkpoints**, and they
  assume a robot's flat joint vector splits into the modality order the statistics declare -
  checked for total width and nothing else. Verify against a recorded episode first.

## Acknowledgements

**[vla.cpp](https://github.com/VinRobotics/vla.cpp)** - a unified C++/ggml runtime that serves
many VLA architectures from one self-contained GGUF, on CUDA, Metal, Intel GPU/NPU, Snapdragon
and CPU. Driven here by `lerobot-vla-cpp`.

```bibtex
@misc{nguyen2026vlacpp,
    title  = {vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models},
    author = {Nguyen, Khanh D. and Ho, Hung T. and Nguyen, Chinh T. and Duong, Thanh Q. and Le, Linh D. and Nguyen, Duy M. H. and Ngo, Vien A. and Le, An T.},
    year   = {2026},
    eprint = {2606.08094},
    archivePrefix = {arXiv},
    url    = {https://arxiv.org/abs/2606.08094}
}
```

**[vla.simd](https://github.com/cair-vinuni/vla.simd)** - a pure-SIMD CPU engine for
language-conditioned manipulation, with no GPU, CUDA or ggml involved. Driven here by
`lerobot-vla-simd`, and the engine the IMPACT int8 path is written against.

```bibtex
@misc{nguyen2026vlasimd,
    title  = {vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation},
    author = {Nguyen, Khanh D. and Truong, Hoang M. and Le, An T.},
    year   = {2026},
    eprint = {2609.24274},
    archivePrefix = {arXiv},
    url    = {https://arxiv.org/abs/2609.24274}
}
```

**[lerobot](https://github.com/huggingface/lerobot)** - installation, the dataset format, the
other policies, hardware guides and tutorials are in the
[upstream documentation](https://huggingface.co/docs/lerobot).

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
