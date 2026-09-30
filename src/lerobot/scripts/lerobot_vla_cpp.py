# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drive a robot against a policy served by a `vla.cpp` inference server.

`vla-server` runs one GGUF checkpoint on whichever backend it was built for - CUDA,
Metal, SYCL, OpenVINO, OpenCL, Hexagon or CPU - and answers ZeroMQ REQ/REP requests
carrying protobuf. That is a different wire format from lerobot's own gRPC
async-inference protocol, so this is a separate command from `lerobot-vla-simd`
rather than a flag on it.

The backend is the engine's business, not this client's, but it decides whether the
policy keeps up: SmolVLA is ~125 ms per query on an RTX 3060 against ~1.7 s on a
desktop CPU, and a 50-step chunk is only 1.67 s of motion at 30 fps. Prefer a GPU.

The loop here is **synchronous**, because the server is: one request, one reply, no
background receiver. A chunk is executed `n_action_steps` deep before the next
request goes out, so that number is the feedback rate.

Drive a robot (the server is already running elsewhere):

```shell
lerobot-vla-cpp \
    --server_address=tcp://127.0.0.1:5555 \
    --arch=smolvla \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                       wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --task="pick up the tape" \
    --n_action_steps=25
```

Check a server with no robot attached, replaying recorded frames instead:

```shell
lerobot-vla-cpp --server_address=tcp://127.0.0.1:5555 --arch=smolvla \
    --replay.repo_id=<user>/<dataset>
```

Replay mode sends real observations and prints the round-trip latency next to the
actions the server returned and the ones actually recorded. It answers "is the
server up, serving the model I think, and producing sane actions" -- it is not an
evaluation: the actions are never executed, so the observations never reflect them.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import draccus
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401

# `lerobot.cameras.realsense` imports pyrealsense2 eagerly whenever the package is merely
# installed, and on some platforms it is installed but not loadable (e.g. a Jetson wheel built
# against a newer glibc). Probe the dependency itself rather than the camera module, so a missing
# realsense only costs us that camera type, while any other import error still surfaces.
try:
    import pyrealsense2  # noqa: F401
except ImportError as e:
    logging.warning("realsense camera type unavailable: pyrealsense2 failed to import (%s)", e)
else:
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.vla_cpp import ARCH_PRESETS, DEFAULT_ADDRESS, VlaCppClient
from lerobot.vla_cpp.archs import MODALITY_KEYED_ARCHS


@dataclass
class ReplayConfig:
    """Hardware-free check: feed recorded frames to the server instead of a robot."""

    repo_id: str | None = field(default=None, metadata={"help": "Dataset to replay observations from"})
    revision: str | None = field(default=None, metadata={"help": "Dataset revision (default: main)"})
    episode: int = field(default=0, metadata={"help": "Episode to replay"})
    steps: int = field(default=10, metadata={"help": "Number of observations to send"})
    stride: int = field(default=10, metadata={"help": "Dataset frames to skip between observations"})


@dataclass
class HomeConfig:
    """Where Ctrl+C parks the arm before it is disconnected.

    The follower disconnects with `disable_torque_on_disconnect=True`, so without
    this the arm goes limp and drops from wherever the policy stopped it.
    """

    enabled: bool = field(default=True, metadata={"help": "Move to the home pose on Ctrl+C"})
    position: dict[str, float] | None = field(
        default=None,
        metadata={
            "help": "Home pose as {joint.pos: value}. Joints left out keep their final "
            "pose. Left unset, Ctrl+C returns the arm to the pose it was in at startup, "
            "which is only a rest pose if it was parked when the command launched."
        },
    )
    duration_s: float = field(default=3.0, metadata={"help": "Seconds to take getting there"})
    fps: int = field(default=50, metadata={"help": "Interpolation rate"})


@dataclass
class VlaCppClientConfig:
    """Robot-side configuration for a `vla.cpp` policy server."""

    server_address: str = field(
        default=DEFAULT_ADDRESS, metadata={"help": "ZeroMQ endpoint of the vla-server"}
    )
    arch: str = field(
        default="smolvla",
        metadata={"help": f"Which preprocessing to apply. Options: {sorted(ARCH_PRESETS)}"},
    )
    task: str = field(default="", metadata={"help": "Instruction; must match a trained string"})
    fps: int = field(default=30, metadata={"help": "Control rate"})
    duration: float = field(default=60.0, metadata={"help": "Seconds to run before stopping"})
    n_action_steps: int = field(
        default=25,
        metadata={
            "help": "Steps of a chunk to execute before re-querying. This is the feedback "
            "rate: fewer means fresher observations, more means fewer queries."
        },
    )
    action_dim: int = field(
        default=0,
        metadata={"help": "Action columns to execute. 0 means the robot's joint count."},
    )

    # Preprocessing overrides; each defaults to the arch preset.
    tokenizer: str | None = field(default=None, metadata={"help": "Hub id or local processor dir"})
    image_size: int | None = field(default=None, metadata={"help": "Square resize target"})
    max_state_dim: int | None = field(default=None, metadata={"help": "State padding width"})
    max_length: int | None = field(default=None, metadata={"help": "Prompt token budget"})
    recv_timeout_ms: int = field(default=30_000, metadata={"help": "Reply timeout"})

    # Normalization.
    stats_json: str | None = field(
        default=None, metadata={"help": "Statistics JSON; required for pi05 and the GR00T archs"}
    )
    rel_stats_json: str | None = field(
        default=None, metadata={"help": "Relative-action statistics, for GR00T N1.7"}
    )
    embodiment: str | None = field(
        default=None, metadata={"help": "Which top-level key of stats_json to normalize with"}
    )

    robot: RobotConfig | None = field(default=None, metadata={"help": "Robot to drive"})
    home: HomeConfig = field(default_factory=HomeConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def __post_init__(self):
        # Note: "neither robot nor replay" is checked in vla_cpp_client() instead.
        # draccus wraps anything raised here into a generic ParsingError, which would
        # hide the one message a first-time user most needs to read.
        if self.arch not in ARCH_PRESETS:
            raise ValueError(f"unknown arch {self.arch!r}; expected one of {sorted(ARCH_PRESETS)}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}")


def _camera_keys(robot: Robot) -> list[str]:
    """Camera names in the order the robot declares them.

    That order is the order frames reach the checkpoint, and the checkpoint's vision
    passes are position-dependent, so a robot whose cameras are declared the other
    way round produces plausible, wrong actions rather than an error.
    """
    return [
        name
        for name, spec in robot.observation_features.items()
        if isinstance(spec, tuple) and not name.endswith("_depth")
    ]


def _joint_keys(robot: Robot) -> list[str]:
    return [name for name in robot.observation_features if name.endswith(".pos")]


def build_client(cfg: VlaCppClientConfig, image_keys: tuple[str, ...], action_dim: int) -> VlaCppClient:
    return VlaCppClient(
        cfg.server_address,
        arch=cfg.arch,
        tokenizer=cfg.tokenizer,
        image_size=cfg.image_size,
        max_state_dim=cfg.max_state_dim,
        max_length=cfg.max_length,
        image_keys=image_keys,
        action_dim=action_dim,
        n_action_steps=cfg.n_action_steps,
        recv_timeout_ms=cfg.recv_timeout_ms,
        stats_json=cfg.stats_json,
        rel_stats_json=cfg.rel_stats_json,
        embodiment=cfg.embodiment,
    )


def split_state_by_layout(
    flat: np.ndarray, keys: tuple[str, ...], dims: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Cut a flat state vector into the `state.<modality>` keys GR00T expects.

    A lerobot robot reports one flat vector of joint positions; GR00T's observations
    are modality-keyed. The split follows the order the checkpoint's statistics
    declare, because that is the order GR00T's own processor concatenates them in.

    **This mapping is an assumption, not a measurement.** It is right only if the
    robot's joints happen to be ordered the way the checkpoint's modalities are. It
    is checked for total width and nothing else, so verify it against a recorded
    episode before trusting it on hardware.
    """
    total = sum(dims)
    if flat.size != total:
        raise ValueError(
            f"robot reports a {flat.size}-D state but this checkpoint's statistics "
            f"declare {list(zip(keys, dims, strict=True))} ({total}-D in total). Pass a "
            f"checkpoint whose embodiment matches the arm, or drive it with --arch=passthrough."
        )
    out: dict[str, np.ndarray] = {}
    offset = 0
    for key, dim in zip(keys, dims, strict=True):
        out[f"state.{key}"] = flat[offset : offset + dim].astype(np.float32)
        offset += dim
    return out


def robot_observation(
    cfg: VlaCppClientConfig,
    client: VlaCppClient,
    raw: dict,
    joint_keys: list[str],
    camera_keys: list[str],
) -> dict:
    """A robot's `get_observation()` dict, in the shape the client's arch reads.

    Two shapes, because the architectures disagree: the generic path takes CHW
    floats in [0, 1] under `observation.images.*` plus a flat `observation.state`,
    while the GR00T paths take HWC uint8 under `video.*` plus modality-keyed state.
    """
    flat_state = np.array([float(raw[k]) for k in joint_keys], dtype=np.float32)

    if cfg.arch in MODALITY_KEYED_ARCHS:
        keys, dims = (
            (client.gr00t.state_keys, client.gr00t.state_dims) if client.gr00t is not None else ((), ())
        )
        if not keys:
            raise ValueError(
                f"--arch={cfg.arch} needs --stats_json: without it there is no modality "
                f"layout to split the robot's state into, and no action un-normalizer."
            )
        observation: dict = {"task": cfg.task}
        observation.update(split_state_by_layout(flat_state, keys, dims))
        # GR00T reads two named views; anything beyond the first two is dropped.
        for target, name in zip(("video.image", "video.wrist_image"), camera_keys, strict=False):
            observation[target] = np.asarray(raw[name], dtype=np.uint8)
        return observation

    observation = {"task": cfg.task, OBS_STATE: flat_state}
    for name in camera_keys:
        frame = np.asarray(raw[name], dtype=np.float32) / 255.0
        observation[f"observation.images.{name}"] = np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))
    return observation


def resolve_home_position(cfg: HomeConfig, robot: Robot) -> dict[str, float] | None:
    """Work out where Ctrl+C should park the arm, and fail now rather than at shutdown.

    lerobot defines no home pose for any robot - `so_follower.connect()` only assumes
    the arm is already resting - and a rest pose is a property of the individual arm
    and its mounting, not of the model. So there is no built-in default: without
    `--home.position` the startup pose is the best reference available, and that is
    only 'home' if the arm was parked when the command launched.
    """
    if not cfg.enabled:
        return None

    present = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    if cfg.position is None:
        logging.warning(
            "No --home.position given: Ctrl+C will return the arm to the pose it is in "
            "right now, which is only 'home' if it is parked. Pass --home.position to set "
            "a real rest pose."
        )
        return present

    unknown = set(cfg.position) - set(present)
    if unknown:
        raise SystemExit(
            f"--home.position names joints this robot does not have: {sorted(unknown)}\n"
            f"Known joints: {sorted(present)}"
        )
    return dict(cfg.position)


def move_to(robot: Robot, target: dict[str, float], duration_s: float = 3.0, fps: int = 50) -> None:
    """Interpolate the arm to `target` before it is disconnected.

    The follower disconnects with `disable_torque_on_disconnect=True`, so whatever
    pose the policy left it in is the pose it *drops* from. Walking it home first
    means it lets go from somewhere it can be let go of.

    This is a straight line in joint space and knows nothing about obstacles - keep
    the home pose somewhere the arm can reach from anywhere it might end up.

    Failures here are logged, never raised: this runs on the shutdown path, and an
    arm that would not move is not a reason to skip disconnecting it.
    """
    try:
        current = {k: v for k, v in robot.get_observation().items() if k in target}
        if not current:
            logging.warning("Robot reported no joint positions; leaving it in its final pose")
            return
        steps = max(int(duration_s * fps), 1)
        for step in range(1, steps + 1):
            t = step / steps
            robot.send_action({k: current[k] * (1 - t) + target[k] * t for k in current})
            precise_sleep(1 / fps)
    except KeyboardInterrupt:
        logging.warning("Interrupted while returning to the home position")
    except Exception as e:  # noqa: BLE001 - shutdown path
        logging.warning(f"Could not return to the home position: {e}")


def run_robot(cfg: VlaCppClientConfig) -> None:
    """Synchronous control loop: observe, query when the queue is dry, act."""
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    joint_keys = _joint_keys(robot)
    camera_keys = _camera_keys(robot)
    if not camera_keys:
        raise SystemExit("this robot has no cameras configured; a VLA policy needs at least one")
    logging.info("joints %s | cameras %s (order matters)", joint_keys, camera_keys)

    action_dim = cfg.action_dim or len(joint_keys)
    image_keys = tuple(f"observation.images.{name}" for name in camera_keys)
    client = build_client(cfg, image_keys, action_dim)

    # Resolved up front so a bad joint name fails now, not after a rollout when the
    # operator is holding Ctrl+C and expecting the arm to park itself.
    home_position = resolve_home_position(cfg.home, robot)
    if home_position:
        logging.info("Home pose on Ctrl+C: %s", {k: round(v, 1) for k, v in home_position.items()})

    period = 1.0 / cfg.fps
    deadline = time.perf_counter() + cfg.duration
    queries = 0
    try:
        while time.perf_counter() < deadline:
            tick = time.perf_counter()
            raw = robot.get_observation()
            observation = robot_observation(cfg, client, raw, joint_keys, camera_keys)
            will_query = client.pending == 0
            action = client.get_action(observation)
            if will_query:
                queries += 1
                response = client.last_response
                if response is not None:
                    logging.info(
                        "query %d | %.1f ms total, %.1f ms inference",
                        queries,
                        response.latency_ms_total,
                        response.latency_ms_inference,
                    )
            robot.send_action({k: float(v) for k, v in zip(joint_keys, action, strict=False)})
            precise_sleep(max(0.0, period - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        logging.info("Interrupted, shutting down")
    finally:
        if home_position:
            logging.info("Returning robot to the home position before shutdown...")
            move_to(robot, home_position, cfg.home.duration_s, cfg.home.fps)
        client.close()
        robot.disconnect()
        logging.info("Client stopped after %d queries", queries)


def run_replay(cfg: VlaCppClientConfig) -> None:
    """Send recorded observations to the server and report what comes back."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logging.info("Loading %s to replay observations", cfg.replay.repo_id)
    dataset = LeRobotDataset(cfg.replay.repo_id, revision=cfg.replay.revision or "main")
    camera_keys = [k for k, ft in dataset.meta.features.items() if ft["dtype"] in ("image", "video")]
    logging.info("%d frames, cameras %s", dataset.num_frames, camera_keys)

    if cfg.replay.episode >= dataset.num_episodes:
        raise SystemExit(f"episode {cfg.replay.episode} not in a dataset with {dataset.num_episodes}")
    episode = dataset.meta.episodes[cfg.replay.episode]
    ep_from, ep_to = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])

    action_dim = cfg.action_dim or int(dataset.meta.features["action"]["shape"][0])
    client = build_client(cfg, tuple(camera_keys), action_dim)

    latencies, errors = [], []
    try:
        for step in range(cfg.replay.steps):
            index = ep_from + step * cfg.replay.stride
            if index >= ep_to:
                logging.info("reached the end of the episode")
                break
            item = dataset[index]

            observation: dict = {"task": cfg.task or item.get("task", "")}
            if cfg.arch in MODALITY_KEYED_ARCHS:
                keys, dims = (
                    (client.gr00t.state_keys, client.gr00t.state_dims)
                    if client.gr00t is not None
                    else ((), ())
                )
                if not keys:
                    raise SystemExit(f"--arch={cfg.arch} needs --stats_json")
                flat = item[OBS_STATE].numpy().astype(np.float32)
                observation.update(split_state_by_layout(flat, keys, dims))
                for target, key in zip(("video.image", "video.wrist_image"), camera_keys, strict=False):
                    chw = item[key].numpy()
                    observation[target] = np.round(chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            else:
                observation[OBS_STATE] = item[OBS_STATE].numpy().astype(np.float32)
                for key in camera_keys:
                    observation[key] = item[key].numpy().astype(np.float32)

            started = time.perf_counter()
            chunk = client.predict_chunk(observation)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)

            predicted = chunk[0, :action_dim]
            recorded = item["action"].numpy()[:action_dim]
            errors.append(float(np.abs(predicted - recorded).max()))
            logging.info(
                "frame %5d | %3d x %d chunk | %6.1f ms | a[0]=%s | recorded=%s",
                index,
                chunk.shape[0],
                chunk.shape[1],
                elapsed,
                np.round(predicted, 2).tolist(),
                np.round(recorded, 2).tolist(),
            )
    finally:
        client.close()

    if not latencies:
        raise SystemExit("no actions received -- is the server serving the arch you asked for?")
    lat = np.array(latencies)
    logging.info(
        "%d round trips | median %.1f ms, min %.1f, max %.1f | median |predicted - recorded| %.2f",
        len(lat),
        np.median(lat),
        lat.min(),
        lat.max(),
        np.median(errors),
    )
    logging.info(
        "Note: the recorded action is the teleoperator's, not ground truth for this "
        "observation, so a non-zero difference is expected."
    )


@draccus.wrap()
def vla_cpp_client(cfg: VlaCppClientConfig) -> None:
    """Entry point: drive a robot, or check a server against recorded frames."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.replay.repo_id is not None and cfg.robot is not None:
        raise SystemExit("pass either --robot.type=... or --replay.repo_id=..., not both")

    if cfg.replay.repo_id is not None:
        run_replay(cfg)
    elif cfg.robot is not None:
        run_robot(cfg)
    else:
        raise SystemExit(
            "nothing to drive. Either attach a robot:\n"
            "  lerobot-vla-cpp --server_address=tcp://HOST:PORT --arch=smolvla "
            "--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_arm\n"
            "or check the server against recorded frames, with no hardware:\n"
            "  lerobot-vla-cpp --server_address=tcp://HOST:PORT --arch=smolvla "
            "--replay.repo_id=<dataset>"
        )


def main() -> None:
    """Console-script entry point."""
    register_third_party_plugins()
    vla_cpp_client()


if __name__ == "__main__":
    main()
