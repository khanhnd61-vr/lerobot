"""Offline replay and SO-101 execution for the supplied action-head checkpoints."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .policy import JOINTS, R3MGrootPolicy, load_checkpoint, vector

LOG = logging.getLogger(__name__)
# Target envelope observed in training. Gripper is 0..100, NOT degrees.
LIMITS = np.array(
    [[-72, -11.43], [-104.92, 53.93], [-7.56, 98.46], [18.2, 96.44], [111.52, 163.03], [0, 37.68]]
)


def guard(target, present, max_step):
    """Limit each command against measured position, including the FIRST command.

    When starting outside the training envelope, approach it gradually. The slew
    limit takes precedence over the box; snapping into the box would cause a jump.
    """
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive")
    target, present = vector(target, "target"), vector(present, "present")
    bounded = np.clip(target, LIMITS[:, 0], LIMITS[:, 1])
    command = present + np.clip(bounded - present, -max_step, max_step)
    return command, bool(np.any(bounded != target)), bool(np.any(np.abs(bounded - present) > max_step))


def camera_device(value):
    """Preserve device paths while parsing numeric OpenCV indices as integers."""
    return int(value) if value.isdecimal() else Path(value)


def robot_config(args):
    """Build the current LeRobot SO-101 config with explicit training units."""
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=camera_device(device),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            color_mode=ColorMode.RGB,
        )
        for name, device in (("front", args.front_cam), ("wrist", args.wrist_cam))
    }
    return SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=args.max_step,
        disable_torque_on_disconnect=args.disable_torque_on_disconnect,
    )


def disconnect_robot(robot):
    """Also handle a partially failed connect, where robot.is_connected is false."""
    for camera in robot.cameras.values():
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception:
                LOG.exception("Camera disconnect failed")
    if robot.bus.is_connected:
        try:
            robot.bus.disconnect(robot.config.disable_torque_on_disconnect)
        except Exception:
            LOG.exception("Motor bus disconnect failed")


def replay(policy, episode, replan, max_steps, max_step):
    """Score cached observations, keeping raw and guarded errors separate."""
    with np.load(episode, allow_pickle=False) as data:
        front, wrist, states, actions = (data[key] for key in ("front", "wrist", "state", "action"))
        task = str(data["task"].item())
    if not len(states) or states.shape != actions.shape or states.shape[1:] != (6,):
        raise ValueError("Replay state/action must be nonempty aligned Nx6 arrays")
    if len(front) != len(states) or len(wrist) != len(states):
        raise ValueError("Replay cameras and state lengths disagree")
    if task not in policy.tasks:
        raise ValueError(f"Unknown replay task: {task!r}")
    predictions, commands, references, latency = [], [], [], []
    chunk_error, chunk_values = 0.0, 0
    n_box = n_rate = 0
    count = min(max_steps, len(states))
    for step in range(0, count, replan):
        start = time.perf_counter()
        chunk = policy.predict_action_chunk(front[step], wrist[step], states[step], task)
        latency.append(time.perf_counter() - start)
        valid = min(len(chunk), len(states) - step)
        chunk_error += float(np.abs(chunk[:valid] - actions[step : step + valid]).sum())
        chunk_values += valid * 6
        for offset in range(min(replan, count - step)):
            index = step + offset
            command, box, rate = guard(chunk[offset], states[index], max_step)
            n_box += box
            n_rate += rate
            predictions.append(chunk[offset])
            commands.append(command)
            references.append(actions[index])
        LOG.info("Replay %d/%d, chunk %.3fs", min(step + replan, count), count, latency[-1])
    raw_error = np.abs(np.asarray(predictions) - references)
    return {
        "steps": count,
        "task": task,
        "clamped_box": n_box,
        "clamped_rate": n_rate,
        "raw_mae": float(raw_error.mean()),
        "raw_mae_per_joint": dict(zip(JOINTS, raw_error.mean(0).tolist(), strict=True)),
        "guarded_mae": float(np.abs(np.asarray(commands) - references).mean()),
        "raw_chunk_mae": chunk_error / chunk_values,
        "inference_seconds": latency,
        "state_std": states.std(0).tolist(),
        "action_minus_state_mean": (actions - states).mean(0).tolist(),
    }


def live(policy, robot, args):
    """Synchronous replanning; inference time counts toward the observed control rate."""
    count, n_box, n_rate = 0, 0, 0
    latency = []
    start = time.perf_counter()
    try:
        if not robot.calibration:
            raise ValueError("No saved calibration for this robot ID. Run lerobot-calibrate first.")
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise ValueError("Motor calibration does not match the saved robot ID; calibrate before rollout")
        while count < args.max_steps:
            tick = time.perf_counter()
            observation = robot.get_observation()
            state = vector([observation[f"{name}.pos"] for name in JOINTS], "state")
            inference_start = time.perf_counter()
            chunk = policy.predict_action_chunk(observation["front"], observation["wrist"], state, args.task)
            elapsed = time.perf_counter() - inference_start
            latency.append(elapsed)
            LOG.info(
                "Measured state %s; first prediction %s", state.round(2).tolist(), chunk[0].round(2).tolist()
            )
            if elapsed > args.max_inference_seconds:
                raise TimeoutError(f"Inference took {elapsed:.2f}s; refusing to send a stale action chunk")
            for offset in range(min(args.replan, args.max_steps - count)):
                if offset:
                    tick = time.perf_counter()
                # Re-read after inference and before EVERY command. A stalled joint
                # must not accumulate a target far beyond its measured position.
                positions = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
                present = vector([positions[name] for name in JOINTS], "present")
                command, box, rate = guard(chunk[offset], present, args.max_step)
                n_box += box
                n_rate += rate
                if not args.dry_run:
                    robot.send_action(
                        dict(zip((f"{name}.pos" for name in JOINTS), command.tolist(), strict=True))
                    )
                count += 1
                time.sleep(max(0, 1 / args.fps - (time.perf_counter() - tick)))
            LOG.info(
                "Steps %d, inference %.3fs, effective %.2f Hz",
                count,
                elapsed,
                count / (time.perf_counter() - start),
            )
    except KeyboardInterrupt:
        LOG.info("Interrupted; stopping action commands")
    finally:
        disconnect_robot(robot)
    return {
        "steps": count,
        "clamped_box": n_box,
        "clamped_rate": n_rate,
        "dry_run": args.dry_run,
        "inference_seconds": latency,
        "effective_hz": count / (time.perf_counter() - start),
        "disable_torque_on_disconnect": args.disable_torque_on_disconnect,
    }


def parser():
    """Build the standalone deployment CLI without importing hardware drivers."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, default=Path("/mnt/khanh/work/r3m"))
    ap.add_argument("--ckpt", type=Path, required=True, help="Trusted local protocol-B .pt checkpoint")
    ap.add_argument("--robot", choices=("replay", "so101"), default="replay")
    ap.add_argument(
        "--inspect", action="store_true", help="Print checkpoint metadata without loading the network"
    )
    ap.add_argument("--episode", type=Path, help="Cached RGB/state/action .npz, defaults to sample ep002")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task")
    ap.add_argument("--replan", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument(
        "--max-step", type=float, default=3, help="Degrees per command; gripper percentage points"
    )
    ap.add_argument("--max-inference-seconds", type=float, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port")
    ap.add_argument("--robot-id")
    ap.add_argument("--calibration-dir", type=Path)
    ap.add_argument("--front-cam")
    ap.add_argument("--wrist-cam")
    ap.add_argument("--camera-width", type=int, default=640)
    ap.add_argument("--camera-height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument(
        "--dry-run", action="store_true", help="Read live hardware and infer without sending position targets"
    )
    ap.add_argument(
        "--disable-torque-on-disconnect",
        action="store_true",
        help="Release the arm at exit; default retains torque",
    )
    ap.add_argument("--out", type=Path)
    return ap


def main(argv=None):
    """Run inspection, cached replay, or live SO-101 inference."""
    ap = parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in (
        "replan",
        "max_steps",
        "fps",
        "max_step",
        "max_inference_seconds",
        "camera_width",
        "camera_height",
        "camera_fps",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be finite and positive")
    checkpoint = load_checkpoint(args.ckpt)
    if args.inspect:
        print(
            json.dumps(
                {
                    "arm": checkpoint["arm"],
                    "horizon": checkpoint["horizon"],
                    "tasks": list(checkpoint["text_features"]),
                    "state_mean": checkpoint["state_mean"].tolist(),
                    "state_std": checkpoint["state_std"].tolist(),
                    "action_mean": checkpoint["action_mean"].tolist(),
                },
                indent=2,
            )
        )
        return
    if args.replan > checkpoint["horizon"]:
        ap.error("--replan must not exceed checkpoint horizon")
    if args.robot == "so101":
        for name in ("port", "robot_id", "front_cam", "wrist_cam"):
            if getattr(args, name) is None:
                ap.error(f"--{name.replace('_', '-')} is required for SO-101")
        if args.front_cam == args.wrist_cam:
            ap.error("Front and wrist must use different camera devices")
        if args.task not in checkpoint["text_features"]:
            ap.error(f"--task must be one of {list(checkpoint['text_features'])}")
    del checkpoint
    # Use the training version of R3M (supports VIGIL/SigLIP), not the older
    # representation-only repository. Paths are local and relocatable via --assets.
    for source in (args.assets / "Isaac-GR00T", args.assets / "r3m-src"):
        if not source.is_dir():
            ap.error(f"Missing training source: {source}")
        sys.path.insert(0, str(source.resolve()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = R3MGrootPolicy(args.ckpt, args.assets / "ckpt", args.assets / "groot_base", args.device)
    if args.robot == "replay":
        episode = args.episode or args.assets / "ckpt/sample_frames/ep002.npz"
        result = replay(policy, episode, args.replan, args.max_steps, args.max_step)
    else:
        from lerobot.robots.so_follower import SO101Follower

        result = live(policy, SO101Follower(robot_config(args)), args)
    result.update(
        arm=policy.arm, checkpoint=str(args.ckpt), backend=args.robot, replan=args.replan, seed=args.seed
    )
    if result["inference_seconds"]:
        result["mean_inference_seconds"] = float(np.mean(result["inference_seconds"]))
        result["chunk_budget_seconds"] = args.replan / args.fps
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
