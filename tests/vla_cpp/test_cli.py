#!/usr/bin/env python

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
"""The lerobot-vla-cpp CLI's config validation and its observation adapters."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zmq")

from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.vla_cpp import vla_pb2  # noqa: E402
from lerobot.vla_cpp.archs import ARCH_PRESETS  # noqa: E402


def _cli():
    from lerobot.scripts import lerobot_vla_cpp

    return lerobot_vla_cpp


# ---------------------------------------------------------------------------
# The vendored protobuf stub
# ---------------------------------------------------------------------------


def test_vendored_stub_round_trips_every_field_the_client_uses():
    """If the engine's schema moves, this is what notices."""
    request = vla_pb2.PredictRequest(request_id=3)
    image = request.images.add()
    image.encoding = vla_pb2.Image.RGB_U8
    image.height, image.width, image.data = 2, 2, b"\x01" * 12
    request.lang_tokens.extend([5, 6, 7])
    request.state.extend([0.25, 0.5])
    request.noise.extend([1.0])

    parsed = vla_pb2.PredictRequest()
    parsed.ParseFromString(request.SerializeToString())
    assert parsed.request_id == 3
    assert parsed.images[0].encoding == vla_pb2.Image.RGB_U8
    assert list(parsed.lang_tokens) == [5, 6, 7]
    assert list(parsed.state) == [0.25, 0.5]
    assert list(parsed.noise) == [1.0]


def test_stub_exposes_the_three_image_encodings():
    assert vla_pb2.Image.JPEG == 0
    assert vla_pb2.Image.RGB_U8 == 1
    assert vla_pb2.Image.F32_RGB_01 == 2


def test_response_error_field_round_trips():
    response = vla_pb2.PredictResponse(error="boom")
    parsed = vla_pb2.PredictResponse()
    parsed.ParseFromString(response.SerializeToString())
    assert parsed.error == "boom"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_an_unknown_arch():
    cli = _cli()
    with pytest.raises(ValueError, match="unknown arch"):
        cli.VlaCppClientConfig(arch="nope")


@pytest.mark.parametrize("arch", sorted(ARCH_PRESETS))
def test_config_accepts_every_known_arch(arch):
    assert _cli().VlaCppClientConfig(arch=arch).arch == arch


def test_config_rejects_a_non_positive_fps():
    with pytest.raises(ValueError, match="fps"):
        _cli().VlaCppClientConfig(fps=0)


def test_config_rejects_a_zero_chunk_depth():
    with pytest.raises(ValueError, match="n_action_steps"):
        _cli().VlaCppClientConfig(n_action_steps=0)


def test_config_accepts_both_inference_modes():
    cli = _cli()
    assert cli.VlaCppClientConfig(mode="sync").mode == "sync"
    assert cli.VlaCppClientConfig(mode="async").mode == "async"


def test_config_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        _cli().VlaCppClientConfig(mode="rtc")


def test_config_validates_the_async_queue_parameters():
    cli = _cli()
    with pytest.raises(ValueError, match="actions_per_chunk"):
        cli.VlaCppClientConfig(actions_per_chunk=0)
    with pytest.raises(ValueError, match="chunk_size_threshold"):
        cli.VlaCppClientConfig(chunk_size_threshold=1.5)
    with pytest.raises(ValueError, match="aggregate_fn_name"):
        cli.VlaCppClientConfig(aggregate_fn_name="median")


def test_build_async_client_reads_the_config():
    from lerobot.vla_cpp.async_client import AGGREGATE_FUNCTIONS

    cli = _cli()
    cfg = cli.VlaCppClientConfig(
        mode="async", actions_per_chunk=20, chunk_size_threshold=0.25, aggregate_fn_name="latest_only", fps=20
    )

    class StubClient:
        action_dim = 6

    engine = cli.build_async_client(cfg, StubClient())
    assert engine.actions_per_chunk == 20
    assert engine.chunk_size_threshold == 0.25
    assert engine.aggregate_fn is AGGREGATE_FUNCTIONS["latest_only"]
    assert engine.dt == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# The GR00T state split - an assumption, so it is pinned
# ---------------------------------------------------------------------------


def test_split_state_by_layout_follows_the_declared_order():
    out = _cli().split_state_by_layout(np.arange(8, dtype=np.float32), ("single_arm", "gripper"), (6, 2))
    np.testing.assert_allclose(out["state.single_arm"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(out["state.gripper"], [6, 7])


def test_split_state_by_layout_refuses_a_width_mismatch():
    """The only thing this mapping can check is the total width, so it must check it.

    A 6-dof arm against a checkpoint declaring 8 dimensions is a different embodiment,
    and silently zero-padding it would send the policy a state it never saw.
    """
    with pytest.raises(ValueError, match="6-D state"):
        _cli().split_state_by_layout(np.zeros(6, dtype=np.float32), ("a", "b"), (6, 2))


def test_split_state_by_layout_handles_a_single_modality():
    out = _cli().split_state_by_layout(np.arange(3, dtype=np.float32), ("all",), (3,))
    assert list(out) == ["state.all"]


# ---------------------------------------------------------------------------
# Observation adapters
# ---------------------------------------------------------------------------


class _FakeClient:
    gr00t = None


def test_generic_observation_is_chw_float_and_flat_state():
    cli = _cli()
    cfg = cli.VlaCppClientConfig(arch="smolvla", task="lift it")
    raw = {
        "shoulder.pos": 1.0,
        "gripper.pos": 2.0,
        "front": np.full((4, 4, 3), 255, dtype=np.uint8),
    }
    out = cli.robot_observation(cfg, _FakeClient(), raw, ["shoulder.pos", "gripper.pos"], ["front"])
    assert out["task"] == "lift it"
    np.testing.assert_allclose(out[OBS_STATE], [1.0, 2.0])
    frame = out["observation.images.front"]
    assert frame.shape == (3, 4, 4), "the client's generic path reads CHW"
    assert frame.dtype == np.float32
    np.testing.assert_allclose(frame, 1.0), "uint8 255 must scale to 1.0"


def test_modality_keyed_observation_needs_statistics():
    cli = _cli()
    cfg = cli.VlaCppClientConfig(arch="gr00t_n1_7")
    with pytest.raises(ValueError, match="stats_json"):
        cli.robot_observation(cfg, _FakeClient(), {"a.pos": 0.0}, ["a.pos"], ["front"])


def test_modality_keyed_observation_sends_uint8_video_keys():
    cli = _cli()
    cfg = cli.VlaCppClientConfig(arch="gr00t_n1_7")

    class WithStats:
        class gr00t:  # noqa: N801 - mirrors the attribute name on the real client
            state_keys = ("single_arm", "gripper")
            state_dims = (6, 2)

    raw = {f"j{i}.pos": float(i) for i in range(8)}
    raw["front"] = np.zeros((4, 4, 3), dtype=np.uint8)
    raw["wrist"] = np.ones((4, 4, 3), dtype=np.uint8)
    out = cli.robot_observation(cfg, WithStats(), raw, [f"j{i}.pos" for i in range(8)], ["front", "wrist"])
    assert out["video.image"].dtype == np.uint8
    assert out["video.wrist_image"].dtype == np.uint8
    np.testing.assert_allclose(out["state.single_arm"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(out["state.gripper"], [6, 7])
