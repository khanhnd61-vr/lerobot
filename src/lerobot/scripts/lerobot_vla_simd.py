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

"""Run a robot against a policy served by a `vla.simd` CPU inference server.

`vla.simd` runs chunked policies -- ACT and IMPACT -- on the CPU with SIMD
kernels, and serves them over the same gRPC async-inference protocol as
`lerobot.async_inference.policy_server`, so the robot side is the ordinary
lerobot async client. This command is the entry point for it, plus a
hardware-free mode for checking a server before wiring a robot to it.

Nothing here is specific to that engine beyond the name: any server speaking the
async-inference protocol works, including one that keeps lerobot and torch off
its own machine (see `loads_action_chunk` in `lerobot.async_inference.helpers`).

Drive a robot (the server is already running elsewhere):

```shell
lerobot-vla-simd \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                       wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --task="pick up the tape" \
    --actions_per_chunk=50
```

Check a server with no robot attached, replaying recorded frames instead:

```shell
lerobot-vla-simd --server_address=127.0.0.1:8080 --replay.repo_id=<user>/<dataset>
```

Replay mode sends real observations from a dataset and prints the round-trip
latency and the returned actions next to the ones actually recorded. It answers
"is the server up, serving the model I think, and producing sane actions" -- it
is not a policy evaluation: the actions are never executed, so the observations
never reflect them.
"""

import logging
import pickle  # nosec
import threading
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import draccus
import grpc
import numpy as np
import torch

from lerobot.async_inference.configs import AGGREGATE_FUNCTIONS, RobotClientConfig
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    loads_action_chunk,
    visualize_action_queue_size,
)
from lerobot.async_inference.robot_client import RobotClient
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
    RobotConfig,
    bi_so_follower,
    koch_follower,
    omx_follower,
    so_follower,
    unitree_g1,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


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
            "help": "Home pose as {joint.pos: value}, e.g. "
            '\'{"shoulder_lift.pos": -63.2, "elbow_flex.pos": 97.2}\'. '
            "Joints left out keep their final pose. Left unset, Ctrl+C returns the arm "
            "to the pose it was in at startup, which is only a rest pose if it was "
            "parked when the command launched."
        },
    )
    duration_s: float = field(default=3.0, metadata={"help": "Seconds to take getting there"})
    fps: int = field(default=50, metadata={"help": "Interpolation rate"})


@dataclass
class VlaSimdClientConfig:
    """Robot-side configuration for a vla.simd policy server."""

    server_address: str = field(
        default="127.0.0.1:8080", metadata={"help": "host:port of the vla.simd policy server"}
    )
    task: str = field(default="", metadata={"help": "Task instruction (ACT ignores it; VLAs do not)"})
    actions_per_chunk: int = field(
        default=50, metadata={"help": "Actions to request per chunk (server truncates its chunk)"}
    )
    fps: int = field(default=30, metadata={"help": "Control rate"})
    chunk_size_threshold: float = field(
        default=0.5, metadata={"help": "Request a new chunk once the queue drops below this fraction"}
    )
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"How overlapping chunks combine. Options: {list(AGGREGATE_FUNCTIONS)}"},
    )
    client_device: str = field(default="cpu", metadata={"help": "Device to move received actions to"})

    # Informational: a vla.simd server serves the checkpoint it was converted from
    # and does not fetch this, but it is sent so the server can warn on a mismatch.
    pretrained_name_or_path: str = field(
        default="vla.simd", metadata={"help": "Checkpoint id the server is expected to be serving"}
    )
    policy_type: str = field(default="act", metadata={"help": "Policy family the server serves"})

    robot: RobotConfig | None = field(default=None, metadata={"help": "Robot to drive"})
    home: HomeConfig = field(default_factory=HomeConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    debug_visualize_queue_size: bool = field(default=False, metadata={"help": "Plot the action queue"})

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps

    def __post_init__(self):
        # Note: "neither robot nor replay" is checked in vla_simd_client() instead.
        # draccus wraps anything raised here into a generic ParsingError, which
        # would hide the one message a first-time user most needs to read.
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")


def _to_robot_client_config(cfg: VlaSimdClientConfig) -> RobotClientConfig:
    """The async client already implements the control loop; this only adapts names."""
    return RobotClientConfig(
        policy_type=cfg.policy_type,
        pretrained_name_or_path=cfg.pretrained_name_or_path,
        robot=cfg.robot,
        actions_per_chunk=cfg.actions_per_chunk,
        task=cfg.task,
        server_address=cfg.server_address,
        policy_device="cpu",  # a vla.simd server is CPU-only by construction
        client_device=cfg.client_device,
        chunk_size_threshold=cfg.chunk_size_threshold,
        fps=cfg.fps,
        aggregate_fn_name=cfg.aggregate_fn_name,
        debug_visualize_queue_size=cfg.debug_visualize_queue_size,
    )


def _joint_positions(robot) -> dict[str, float]:
    """The `<joint>.pos` entries of an observation, dropping the camera frames."""
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


def resolve_home_position(cfg: "HomeConfig", robot) -> dict[str, float] | None:
    """Work out where Ctrl+C should park the arm, and fail now rather than at shutdown.

    lerobot defines no home pose for any robot - `so_follower.connect()` only assumes the
    arm is already resting - and a rest pose is a property of the individual arm and its
    mounting, not of the model. So there is no built-in default to fall back to: without
    `--home.position` the startup pose is the best reference available, and that is only
    'home' if the arm was parked when the command launched. Hence the warning.
    """
    if not cfg.enabled:
        return None

    present = _joint_positions(robot)
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
    # Joints left out of the dict stay where the policy left them, which is a
    # legitimate choice (e.g. homing the arm but not the gripper).
    return {**{k: cfg.position[k] for k in cfg.position}}


def _move_to(robot, target: dict[str, float], duration_s: float = 3.0, fps: int = 50) -> None:
    """Interpolate the arm to `target` before it is disconnected.

    Same teardown as `lerobot-rollout` (`RolloutStrategy._return_to_initial_position`):
    the follower disconnects with `disable_torque_on_disconnect=True`, so whatever
    pose the policy left it in is the pose it *drops* from. Walking it home first
    means it lets go from somewhere it can be let go of.

    This is a straight line in joint space and knows nothing about obstacles - keep
    the home pose somewhere the arm can reach from anywhere it might end up.

    Failures here are logged, never raised: this runs on the shutdown path, and an
    arm that would not move is not a reason to skip disconnecting it.
    """
    try:
        current = {k: v for k, v in _joint_positions(robot).items() if k in target}
        if not current:
            logging.warning("Robot reported no joint positions; leaving it in its final pose")
            return

        steps = max(int(duration_s * fps), 1)
        for step in range(1, steps + 1):
            t = step / steps
            robot.send_action({k: current[k] * (1 - t) + target[k] * t for k in current})
            precise_sleep(1 / fps)
    except KeyboardInterrupt:
        # a second Ctrl+C during the walk home: stop where it is and disconnect
        logging.warning("Interrupted while returning to the home position")
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Could not return to the home position: {e}")


def run_robot(cfg: VlaSimdClientConfig) -> None:
    """Drive the robot through lerobot's async client against the vla.simd server."""
    client = RobotClient(_to_robot_client_config(cfg))
    if not client.start():
        raise SystemExit(
            f"could not reach a policy server at {cfg.server_address}.\n"
            "Start one with your own engine's server, or lerobot's:\n"
            "  python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080"
        )

    # Resolved up front so a bad joint name fails now, not after a rollout when
    # the operator is holding Ctrl+C and expecting the arm to park itself.
    home_position = resolve_home_position(cfg.home, client.robot)
    if home_position:
        logging.info(f"Home pose on Ctrl+C: { {k: round(v, 1) for k, v in home_position.items()} }")

    client.logger.info("Starting action receiver thread...")
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    receiver.start()
    try:
        client.control_loop(task=cfg.task)
    except KeyboardInterrupt:
        client.logger.info("Interrupted, shutting down")
    finally:
        # Stop the receiver before moving: the control loop has already exited, so
        # nothing else is driving the robot, and homing must finish before
        # client.stop() disconnects (and de-energizes) the arm.
        client.shutdown_event.set()
        if home_position:
            client.logger.info("Returning robot to the home position before shutdown...")
            _move_to(client.robot, home_position, cfg.home.duration_s, cfg.home.fps)

        client.stop()
        receiver.join(timeout=5.0)  # it may be parked in a GetActions call
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        client.logger.info("Client stopped")


# ---------------------------------------------------------------------------
# replay (no robot)
# ---------------------------------------------------------------------------
def _observation_features(meta) -> dict[str, dict]:
    """Dataset metadata -> the feature dict the server expects.

    This is the same structure `map_robot_keys_to_lerobot_features` builds from a
    live robot, which is what makes the replay path exercise the server's real
    key handling rather than a shortcut around it.
    """
    return {k: v for k, v in meta.features.items() if k.startswith(f"{OBS_STR}.")}


def _raw_observation(item, features: dict[str, dict]) -> dict:
    """Dataset frame -> the raw dict a robot's get_observation() would return.

    Motors come back as individual `<name>.pos` scalars and cameras as HxWx3
    uint8 under their short name, because that is what `build_dataset_frame`
    reassembles on the server.
    """
    raw = {}
    state = item[f"{OBS_STR}.state"].numpy().astype(np.float32)
    for i, name in enumerate(features[f"{OBS_STR}.state"]["names"]):
        raw[name] = float(state[i])

    for key, ft in features.items():
        if ft["dtype"] not in ("image", "video"):
            continue
        chw = item[key].numpy()  # dataset frames are CHW float in [0, 1]
        raw[key.removeprefix(f"{OBS_STR}.images.")] = np.round(chw.transpose(1, 2, 0) * 255.0).astype(
            np.uint8
        )
    return raw


def run_replay(cfg: VlaSimdClientConfig) -> None:
    """Send recorded observations to the server and report what comes back."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logging.info(f"Loading {cfg.replay.repo_id} to replay observations")
    ds = LeRobotDataset(cfg.replay.repo_id, revision=cfg.replay.revision or "main")
    features = _observation_features(ds.meta)
    cameras = [k for k in features if k.startswith(f"{OBS_STR}.images.")]
    logging.info(f"{ds.num_frames} frames, cameras {cameras}")

    if cfg.replay.episode >= ds.num_episodes:
        raise SystemExit(f"episode {cfg.replay.episode} not in a dataset with {ds.num_episodes}")
    episode = ds.meta.episodes[cfg.replay.episode]
    ep_from = int(episode["dataset_from_index"])
    ep_to = int(episode["dataset_to_index"])

    channel = grpc.insecure_channel(
        cfg.server_address, grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s")
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    try:
        stub.Ready(services_pb2.Empty())
    except grpc.RpcError as e:
        raise SystemExit(
            f"could not reach a policy server at {cfg.server_address}: {e.details()}\n"
            "Start one with your own engine's server, or lerobot's:\n"
            "  python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080"
        ) from e

    stub.SendPolicyInstructions(
        services_pb2.PolicySetup(
            data=pickle.dumps(  # nosec
                RemotePolicyConfig(
                    policy_type=cfg.policy_type,
                    pretrained_name_or_path=cfg.pretrained_name_or_path,
                    lerobot_features=features,
                    actions_per_chunk=cfg.actions_per_chunk,
                    device="cpu",
                )
            )
        )
    )
    logging.info(f"Connected to {cfg.server_address}; replaying episode {cfg.replay.episode}")

    shutdown = threading.Event()
    latencies, errors = [], []
    for step in range(cfg.replay.steps):
        idx = ep_from + step * cfg.replay.stride
        if idx >= ep_to:
            logging.info("reached the end of the episode")
            break

        item = ds[idx]
        obs = TimedObservation(
            timestamp=time.time(),
            timestep=step,
            observation={**_raw_observation(item, features), "task": cfg.task},
            must_go=True,  # never let the server's similarity filter drop a probe
        )

        t0 = time.perf_counter()
        stub.SendObservations(
            send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation, shutdown, "")  # nosec
        )
        received = stub.GetActions(services_pb2.Empty())
        dt = (time.perf_counter() - t0) * 1000

        if not received.data:
            logging.warning(f"step {step}: server returned no actions (queue timeout?)")
            continue
        chunk = loads_action_chunk(received.data)
        latencies.append(dt)

        predicted = chunk[0].get_action()
        recorded = item["action"]
        err = torch.abs(predicted - recorded).max().item()
        errors.append(err)
        logging.info(
            f"frame {idx:5d} | {len(chunk):3d} actions | {dt:6.1f} ms | "
            f"a[0]={np.round(predicted.numpy(), 2).tolist()} | "
            f"recorded={np.round(recorded.numpy(), 2).tolist()}"
        )

    channel.close()
    if not latencies:
        raise SystemExit("no actions received -- is the server serving an ACT checkpoint?")

    lat = np.array(latencies)
    logging.info(
        f"{len(lat)} round trips | median {np.median(lat):.1f} ms, min {lat.min():.1f}, "
        f"max {lat.max():.1f} | median |predicted - recorded| {np.median(errors):.2f}"
    )
    logging.info(
        "Note: the recorded action is the teleoperator's, not ground truth for this "
        "observation, so a non-zero difference is expected."
    )


@draccus.wrap()
def vla_simd_client(cfg: VlaSimdClientConfig) -> None:
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
            "  lerobot-vla-simd --server_address=HOST:PORT --robot.type=so101_follower "
            "--robot.port=/dev/ttyACM0 --robot.id=my_arm\n"
            "or check the server against recorded frames, with no hardware:\n"
            "  lerobot-vla-simd --server_address=HOST:PORT --replay.repo_id=<dataset>"
        )


def main() -> None:
    register_third_party_plugins()
    vla_simd_client()


if __name__ == "__main__":
    main()
