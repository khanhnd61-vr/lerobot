"""Hardware-free regression tests for the protocol-B deployment boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.r3m_groot.policy import JOINTS, R3MGrootPolicy, prepare_image
from lerobot.r3m_groot.rollout import camera_device, guard, live, parser, replay, robot_config


@pytest.mark.parametrize("bad", [np.full(6, np.nan), np.full(6, np.inf), np.zeros(5)])
def test_guard_rejects_invalid_predictions(bad):
    with pytest.raises(ValueError):
        guard(bad, np.zeros(6), 3)


def test_first_command_is_bounded_even_outside_training_envelope():
    present = np.zeros(6)
    command, box, rate = guard(np.full(6, 1000), present, 3)
    assert np.max(np.abs(command - present)) <= 3
    assert box and rate


def test_guard_uses_measurement_instead_of_accumulating_on_stalled_joint():
    state = np.array([-40, 0, 50, 50, 87.5, 0])
    for _ in range(20):
        command, _, _ = guard(np.array([-40, 0, 50, 50, 153, 10]), state, 3)
        assert command[4] == 90.5


def test_preprocessing_preserves_rgb_and_rejects_normalized_pixels():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    resized = prepare_image(rgb)
    assert resized.shape == (224, 224, 3)
    assert np.all(resized[..., 0] == 255)
    assert np.all(resized[..., 2] == 0)
    with pytest.raises(ValueError):
        prepare_image(rgb.astype(np.float32) / 255)


def test_actual_lerobot_config_uses_explicit_units_and_rgb(tmp_path):
    from lerobot.motors import MotorNormMode
    from lerobot.robots.so_follower import SO101Follower

    args = parser().parse_args(
        [
            "--ckpt",
            "unused.pt",
            "--port",
            "/dev/ttyACM0",
            "--robot-id",
            "test",
            "--front-cam",
            "0",
            "--wrist-cam",
            "/dev/video2",
        ]
    )
    config = robot_config(args)
    assert config.use_degrees
    assert config.max_relative_target == 3
    assert config.cameras["front"].index_or_path == 0
    assert config.cameras["front"].color_mode.value == "rgb"
    assert config.cameras["wrist"].index_or_path == camera_device("/dev/video2")
    assert not config.disable_torque_on_disconnect
    config.calibration_dir = tmp_path
    robot = SO101Follower(config)
    assert not robot.bus.is_connected
    assert all(robot.bus.motors[name].norm_mode == MotorNormMode.DEGREES for name in JOINTS[:5])
    assert robot.bus.motors["gripper"].norm_mode == MotorNormMode.RANGE_0_100


def test_policy_normalizes_state_and_unscales_absolute_actions_without_target_leakage():
    policy = R3MGrootPolicy.__new__(R3MGrootPolicy)
    policy.device = torch.device("cpu")
    policy.tasks = ("task",)
    policy.horizon, policy.state_dim = 2, 8
    policy.state_mean, policy.state_std = np.ones(6), np.full(6, 2)
    policy.action_mean, policy.action_std = np.full(6, 10), np.full(6, 3)
    policy.text = {"task": np.ones(2, dtype=np.float32)}
    policy.projectors = [torch.nn.Identity()] * 3
    policy._feature = lambda *args: torch.ones(1, 2)

    def predict(backbone, inputs):
        assert "action" not in inputs
        assert inputs["state"].shape == (1, 1, 8)
        assert torch.equal(inputs["state"][0, 0, :6], torch.ones(6))
        assert torch.equal(inputs["state"][0, 0, 6:], torch.zeros(2))
        assert inputs["embodiment_id"].item() == 0
        assert backbone["image_mask"].tolist() == [[True, True, False]]
        return {"action_pred": torch.ones(1, 40, 8)}

    policy.head = SimpleNamespace(get_action=predict)
    chunk = policy.predict_action_chunk(None, None, np.full(6, 3), "task")
    np.testing.assert_array_equal(chunk, np.full((2, 6), 13))
    with pytest.raises(ValueError, match="Unknown task"):
        policy.predict_action_chunk(None, None, np.full(6, 3), "unknown")


def test_replay_reports_raw_and_guarded_error_separately_and_truncates_tail(tmp_path):
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        front=np.zeros((3, 2, 2, 3), dtype=np.uint8),
        wrist=np.zeros((3, 2, 2, 3), dtype=np.uint8),
        state=np.zeros((3, 6)),
        action=np.full((3, 6), 1000),
        task="task",
    )
    policy = SimpleNamespace(tasks=("task",), predict_action_chunk=Mock(return_value=np.full((2, 6), 1000)))
    result = replay(policy, path, replan=2, max_steps=10, max_step=3)
    assert result["steps"] == 3
    assert result["raw_mae"] == result["raw_chunk_mae"] == 0
    assert result["guarded_mae"] > 900
    assert policy.predict_action_chunk.call_count == 2


def fake_robot():
    state = dict(zip(JOINTS, [-40, 0, 50, 50, 87.5, 0], strict=True))
    robot = Mock()
    robot.calibration = {"saved": True}
    robot.is_calibrated = True
    robot.config.disable_torque_on_disconnect = False
    robot.bus.is_connected = True
    robot.bus.sync_read.return_value = state
    robot.cameras = {"front": Mock(is_connected=True), "wrist": Mock(is_connected=True)}
    robot.get_observation.return_value = {
        **{f"{name}.pos": value for name, value in state.items()},
        "front": None,
        "wrist": None,
    }
    return robot


@pytest.mark.parametrize("dry_run", [False, True])
def test_live_reads_measured_state_every_command_and_cleans_up(dry_run):
    robot = fake_robot()
    args = SimpleNamespace(
        max_steps=3,
        replan=2,
        fps=1e6,
        task="task",
        max_step=3,
        dry_run=dry_run,
        max_inference_seconds=10,
        disable_torque_on_disconnect=False,
    )
    policy = SimpleNamespace(predict_action_chunk=Mock(return_value=np.full((2, 6), 150)))
    result = live(policy, robot, args)
    assert result["steps"] == 3
    assert robot.bus.sync_read.call_count == 3
    assert robot.send_action.call_count == (0 if dry_run else 3)
    for call in robot.send_action.call_args_list:
        assert call.args[0]["wrist_roll.pos"] == 90.5
    robot.bus.disconnect.assert_called_once_with(False)
    for camera in robot.cameras.values():
        camera.disconnect.assert_called_once()


def test_live_cleans_up_partial_connection_failure():
    robot = fake_robot()
    robot.connect.side_effect = RuntimeError("camera failed")
    args = SimpleNamespace(max_steps=1)
    with pytest.raises(RuntimeError, match="camera failed"):
        live(None, robot, args)
    robot.send_action.assert_not_called()
    robot.bus.disconnect.assert_called_once()


def test_live_rejects_stale_predictions_without_moving():
    robot = fake_robot()
    args = SimpleNamespace(max_steps=1, task="task", max_inference_seconds=-1)
    policy = SimpleNamespace(predict_action_chunk=Mock(return_value=np.zeros((2, 6))))
    with pytest.raises(TimeoutError):
        live(policy, robot, args)
    robot.send_action.assert_not_called()
    robot.bus.disconnect.assert_called_once()
