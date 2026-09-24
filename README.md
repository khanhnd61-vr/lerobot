# lerobot — SO-101 fork

A fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) carrying an
instruction-conditioned policy small enough to run on a CPU at the robot, and the client
that drives an arm against a CPU inference server.

Everything upstream still works as documented upstream. This README covers only what is
different here, and the SO-101 workflow the additions were built for.

## What this fork adds

|                                            |                                                                                                                                                                                                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[IMPACT](src/lerobot/policies/impact/)** | ACT with a frozen T5-small language tower. The instruction enters as encoder tokens _and_ as FiLM on the ResNet stages. 50-step chunk, ~78M trainable parameters. Full write-up: [docs/source/impact.mdx](docs/source/impact.mdx) |
| **IMPACT int8 / QAT**                      | `--policy.int8_groups` runs selected GEMMs through a simulated W8A8 kernel, in training and at inference, matching a specific CPU deployment kernel. A QAT checkpoint is an ordinary fp32 checkpoint — no quantization state.     |
| **`lerobot-vla-simd`**                     | Robot-side client for a CPU inference server speaking lerobot's async-inference protocol, plus a hardware-free replay mode for checking a server before wiring an arm to it.                                                      |
| **`loads_action_chunk`**                   | Lets the async client read chunks from a server that keeps lerobot and torch off its own machine. Payloads from lerobot's own policy server are unpickled exactly as before.                                                      |

```bash
pip install -e ".[impact,async]"
```

## Wiring — paste once per shell

Every command below uses these. **A command copied without them will not run.** Substitute
your own ports and arm id; `lerobot-find-port` reports the ports.

```bash
FOLLOWER_PORT=/dev/ttyACM0
LEADER_PORT=/dev/ttyACM1
ARM_ID=my_awesome_follower_arm
LEADER_ID=my_awesome_leader_arm

CAMS='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}'

ROBOT=(--robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$ARM_ID
       --robot.cameras="$CAMS")

# SmolVLA only — it was trained through this rename. IMPACT and ACT use the dataset's own keys.
RENAME=(--rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}')
```

**Camera keys must stay `front` and `wrist`, in that order.** The vision passes are
position-dependent, so a swapped pair gives plausible, wrong actions rather than an error.

## The SO-101 loop

### Setup and calibrate

```bash
sudo chmod 666 $FOLLOWER_PORT $LEADER_PORT

# before daisy-chain wiring, one arm at a time
lerobot-setup-motors --robot.type=so101_follower --robot.port=$FOLLOWER_PORT
lerobot-setup-motors --robot.type=so101_leader   --robot.port=$LEADER_PORT

lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$ARM_ID
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID
```

```bash
lerobot-teleoperate "${ROBOT[@]}" \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID \
  --display_data=true
```

### Record

`n` next episode · `r` restart current · `q` quit.

```bash
lerobot-record "${ROBOT[@]}" \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID \
  --display_data=true --dataset.repo_id=$HF_USER/so101-tape \
  --dataset.push_to_hub=False --dataset.num_episodes=20 \
  --dataset.single_task="Put the tape into the box" --dataset.encoder_threads=2
```

For a multi-task dataset, record each instruction into the same `repo_id` with
`--dataset.no_stamp=true`, and `--resume=true` on every run after the first, changing only
`--dataset.single_task`. Without `--dataset.no_stamp=true` the local directory is stamped
`<name>_<YYYYMMDD>_<HHMMSS>`, which is then the name `lerobot-replay` and
`--dataset.repo_id` need.

### Curate

A bad episode costs more than a missing one — a language-conditioned policy learns the
wrong instruction from a mislabelled one.

```bash
lerobot-edit-dataset --repo_id $HF_USER/so101-long --root $DATA_ROOT/so101-long \
  --operation.type info --operation.show_features true

lerobot-dataset-viz --repo-id $HF_USER/so101-long --root $DATA_ROOT/so101-long --episode-index 7

lerobot-edit-dataset --repo_id $HF_USER/so101-long --root $DATA_ROOT/so101-long \
  --new_repo_id $HF_USER/so101-long-clean --new_root $DATA_ROOT/so101-long-clean \
  --operation.type delete_episodes --operation.episode_indices "[3, 17, 42]"
```

### Train

Steps ↔ epochs: `steps_per_epoch = total_frames / batch_size`.

```bash
# IMPACT — no --rename_map, it trains on the dataset's own front/wrist keys
lerobot-train --policy.type=impact \
  --dataset.repo_id=$HF_USER/so101-multi-task-clean \
  --batch_size=8 --steps=4000 --save_freq=1000 \
  --output_dir=outputs/train/impact --job_name=impact \
  --policy.device=cuda --policy.push_to_hub=false

# IMPACT with quantization-aware training
lerobot-train --policy.type=impact --policy.int8_groups=63 ...

# SmolVLA — needs the rename
lerobot-train --policy.path=lerobot/smolvla_base "${RENAME[@]}" \
  --dataset.repo_id=$HF_USER/so101-multi-task-clean \
  --batch_size=8 --steps=4000 --save_freq=1000 \
  --policy.scheduler_decay_steps=4000 --policy.scheduler_warmup_steps=200 \
  --output_dir=outputs/train/smolvla --job_name=smolvla --policy.device=cuda
```

On an RTX 3060 12GB, IMPACT runs ~2.7 step/s at batch 8. SmolVLA fits at batch 8 with its
vision encoder frozen (the default); unfreezing needs batch 2–4.

## Checkpoints and tasks

Trained ~10 epochs at batch 8 on the curated datasets.

| Model             | `so101-long-clean`                   | `so101-multi-task-clean`                       |
| ----------------- | ------------------------------------ | ---------------------------------------------- |
| IMPACT            | `khanhnd61/impact_so101-long-clean`  | `khanhnd61/impact_so101-multi-task-clean`      |
| IMPACT-int8 (QAT) | —                                    | `khanhnd61/impact-int8_so101-multi-task-clean` |
| SmolVLA           | `khanhnd61/smolvla_so101-long-clean` | `khanhnd61/smolvla_so101-multi-task-clean`     |

**`--task` must match a trained string exactly.** A reworded instruction degrades the
policy silently.

| Dataset                  | Task strings                                                                           |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `so101-long-clean`       | `Put the tape in the drawer` — the only one                                            |
| `so101-multi-task-clean` | `Put the tape into the box` · `Put the tape into the cup` · `Put the cup into the box` |

Note `in the drawer` against `into the ...` — different strings, do not normalise them. A
typo is not an error, just a worse policy; the server re-tokenizes only when the string
changes, so switching instructions mid-rollout is cheap.

Only the multi-task checkpoint can test IMPACT's language tower — three instructions over a
shared scene. On `so101-long-clean` every frame carries the same sentence, so a good rollout
there says nothing about instruction following.

## Rolling out

### PyTorch, GPU or CPU

```bash
lerobot-rollout --strategy.type=base \
  --policy.path=khanhnd61/impact_so101-multi-task-clean \
  --policy.device=cuda "${ROBOT[@]}" \
  --task="Put the tape into the box" --duration=60 --display_data=true
```

`--policy.device=cpu` is the only change for the CPU path, and it is the slow baseline —
useful to sanity-check a checkpoint without a GPU, not to drive the arm for real work.

**The int8 checkpoint runs the deployment arithmetic on your GPU.** Its `config.json`
declares `int8_groups: 63`, so a plain rollout reproduces the W8A8 rounding a CPU engine
will run. This is fake quantization, not fast quantization: the values are quantized but
the arithmetic underneath is still fp32, so it is ~1.3× _slower_ than the plain checkpoint.
What it buys is watching the deployment numerics move a real arm without deploying
anything.

```bash
--policy.int8_groups=0     # fp32 numerics — the control
--policy.int8_groups=47    # everything except the decoder  <- prefer this on hardware
--policy.int8_groups=63    # every group (the checkpoint's own default)
```

**Prefer 47 on hardware.** Quantizing the decoder raises within-chunk jerk ~46% and QAT
cannot fix it: the training objective is L1 on the chunk and is indifferent to roughness
between two correct waypoints. The chunk executes open loop, so that roughness reaches the
servos as shaking. The policy logs the mask it loaded; no line at all means fp32.

### Through a `vla.simd` CPU server

> This section needs the separate `vla.simd` engine repository, which is not part of this
> fork. The client side — `lerobot-vla-simd` — is all this repo owns, and it speaks the
> stock async-inference protocol, so any server implementing that protocol works.

**One-time — build, convert, verify.** One engine serves every checkpoint of a family;
shapes are read from the `.meta` files at load time.

```bash
cd $VLA_SIMD                 # the engine checkout
cmake -S . -B build && cmake --build build -j --target vla_simd_impact

# Pass a LOCAL checkpoint directory, never a Hub id, and always pass --instruction.
uv run --project $LEROBOT python tools/impact/dump_impact_golden.py \
  --checkpoint $TRAIN_ROOT/impact_so101-multi-task-clean/checkpoints/last/pretrained_model \
  --out build/impact-multi --instruction "Put the tape into the box" --frames 8

uv run --project $LEROBOT python tools/impact/check_parity.py --model build/impact-multi
```

Both of those flags fail **silently** when wrong:

- `load_dataset_stats()` globs the checkpoint path for `*normalizer*.safetensors`, so it
  only works on a local directory. Given a Hub id the glob matches nothing and the
  converter falls back to _identity_ normalization — the engine then emits normalized
  actions (|a| ≈ 1) as if they were degrees, and the arm slews to a pose nothing like the
  training distribution. Read the dumper's `stats` line: `dataset statistics from the
checkpoint's normalizer` is correct, `IDENTITY - no normalizer found` means stop.
- `--instruction` has a hardcoded default. It becomes `config.txt`'s instruction, which the
  server uses whenever the client sends no `--task`.

**`PARITY OK` does not mean the normalization is right.** The parity run feeds its torch
reference the same stats the engine got, so identity stats make both sides agree and it
passes. Check `stats.bin` separately, and confirm `actions_norm` and `actions (env)` differ
in the parity output: `env` should sit in the arm's degree range, `norm` around 1. Two
identical lines mean denormalization is a no-op.

**Terminal 1 — the server.** It holds the weights across client restarts and prints one
warm timing before it accepts a client. **Read that number; it sizes the client flags.**

```bash
cd $VLA_SIMD
OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
uv run --project $LEROBOT python serve/impact_policy_server.py \
  --model-dir build/impact-multi \
  --checkpoint khanhnd61/impact_so101-multi-task-clean \
  --host 0.0.0.0 --port 8080
```

One server owns the port, so run one checkpoint at a time or give the second `--port 8081`.
`--checkpoint` only warns on a client mismatch; `--model-dir` decides what is served.
`OMP_NUM_THREADS` is read by libgomp when the `.so` loads, so it has to be in the process
environment, not set later from Python.

**Terminal 2 — the rollout.** Check the server with no arm attached first:

```bash
lerobot-vla-simd --server_address=127.0.0.1:8080 --policy_type=impact \
  --replay.repo_id=khanhnd61/so101-multi-task-clean --replay.steps=10
```

```bash
lerobot-vla-simd --server_address=127.0.0.1:8080 \
  --policy_type=impact \
  --pretrained_name_or_path=khanhnd61/impact_so101-multi-task-clean \
  "${ROBOT[@]}" --task="Put the tape into the box" \
  --actions_per_chunk=25 --aggregate_fn_name=latest_only --fps=30
```

`--policy_type` is required and the client defaults to `act`; the server refuses a mismatch
rather than misreading the chunk. No `--rename_map` on this path — the converter recorded
the key map in `config.txt` and the server matches through it.

**`--actions_per_chunk=25`, not 50, on a desktop.** IMPACT re-plans only when its queue
drains, so that number _is_ the feedback rate: 25 leaves ~0.4 s of open-loop motion between
observations where 50 leaves ~0.83 s. Raise it only on a host slow enough that the queue
would otherwise starve.

`--aggregate_fn_name=latest_only` is worth trying alongside it. The default
`weighted_average` blends `0.3 × old + 0.7 × new` over the overlap, mixing two chunks
predicted from _different_ observations — for a chunked trajectory policy that smears two
plans together rather than following either.

### Remote board

The arm stays on the workstation, the inference server runs on the board, and the client
here points at it. The payload is two raw 640×480 RGB frames, ~1.8 MB per query, so use
wired gigabit. **The wire protocol is pickle over an unauthenticated port: lab LAN or an
SSH tunnel, never an open network.**

IMPACT's dumper needs torch, so convert on the workstation and copy the directory across —
then re-run the stats check on the remote copy, because an interrupted `rsync` leaves a
`stats.bin` that still loads.

```bash
rsync -a $VLA_SIMD/build/impact-multi $REMOTE:$VLA_SIMD/build/

# on the board (Raspberry Pi 5: 4 homogeneous A76 cores, so pinning is worth 3-8%)
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  .venv-serve/bin/python serve/impact_policy_server.py \
    --model-dir build/impact-multi --host 0.0.0.0 --port 8080
```

A Pi 5 clears the 30 Hz loop for IMPACT at 41.2 Hz cool / 33.5 Hz soaked — 12% margin hot,
which is why its client flags differ from the desktop ones:

```bash
lerobot-vla-simd --server_address=$REMOTE_IP:8080 \
  --policy_type=impact \
  --pretrained_name_or_path=khanhnd61/impact_so101-multi-task-clean \
  "${ROBOT[@]}" --task="Put the tape into the box" \
  --actions_per_chunk=50 --chunk_size_threshold=0.85 \
  --aggregate_fn_name=latest_only --fps=30
```

**The two hosts need different flags for the same policy.** The board must request the
whole chunk and refill early just to stay fed, which costs feedback rate; a desktop has
margin to spend on a shorter chunk and therefore fresher observations.

On Apple silicon, do not pin — the cores are heterogeneous — and unify OpenMP before
starting: the engine dylib links Homebrew's `libomp.dylib` while torch bundles its own, and
a process that loads both dies with `OMP: Error #15` right after the warm inference line.

## Sizing the queue

A refill is requested once the queue drops below `actions_per_chunk × chunk_size_threshold`
actions, and it must arrive before the queue empties:

```
chunk_size_threshold  >  inference_latency × fps / actions_per_chunk
```

At a full chunk the right-hand side is exactly the **stall number** — query latency divided
by the duration of the chunk it produces.

| Latency (measure it) | `--actions_per_chunk` | Minimum threshold at 30 fps               |
| -------------------: | --------------------: | ----------------------------------------- |
|               200 ms |                    50 | 0.12 — default 0.5 is fine                |
|               800 ms |                    50 | 0.48 — default 0.5 is marginal            |
|              1300 ms |                    50 | 0.78 — run at 0.85                        |
|              1700 ms |                    50 | **1.02 — impossible, no threshold works** |

Clearing the threshold only means the queue never starves. How _closed-loop_ the rollout is
depends on how often a fresh observation reaches the policy, which is a separate quantity:

```
feedback interval = max( A(1 - t) / fps ,  L )     A = actions_per_chunk, t = chunk_size_threshold
```

The queue policy sets the first term; latency floors it with the second. A desktop is
queue-bound, so the flags decide everything. A Pi 5 is latency-bound — refilling earlier
cannot buy feedback the engine has no time to produce.

**If the queue drains, raise `--chunk_size_threshold` or `--actions_per_chunk` — never
lower them.** Both sit in the denominator, so lowering either makes the starvation worse.
When the minimum exceeds 1.0 the model is too slow for that board at that chunk and no flag
rescues it: use a faster host, a smaller model, or int8.

## Gotchas

Three that are not stated above, and that cost time:

- **`--rename_map` is a top-level `lerobot-rollout` flag**, not `--policy.rename_map`. It
  applies to SmolVLA only, and never to the `lerobot-vla-simd` client.
- **The `vla.simd` servers match cameras by name** and refuse a session they cannot match.
  The PyTorch path does not check, so a swapped `front`/`wrist` is only caught there by the
  arm behaving oddly.
- **Flow matching is stochastic.** SmolVLA draws a fresh noise sample per query; the
  `vla.simd` server derives it from `seed + query_index`, and `--seed` is a _server_ flag —
  there is none on the client. Differing trajectories between runs are correct behaviour.

## Status

The IMPACT implementation and its int8 path are covered by tests in
[`tests/policies/impact/`](tests/policies/impact/). The latency figures above are the
engine's published measurements on other checkpoints; **the end-to-end arm rollouts in this
README have not all been run against hardware**, and the numbers for these specific
checkpoints are still to be taken. Treat the commands as derived from the checkpoints as
trained and from the engine's current interfaces.

## Upstream

This fork tracks [huggingface/lerobot](https://github.com/huggingface/lerobot). For
installation, the dataset format, the other policies, hardware guides and tutorials, see the
[upstream documentation](https://huggingface.co/docs/lerobot). Contribution guidelines are
in [CONTRIBUTING.md](CONTRIBUTING.md).

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
