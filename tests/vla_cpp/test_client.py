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
"""The vla.cpp client, against a real ZeroMQ server.

The fake server here is a genuine REP socket speaking the real protobuf schema, so
these tests cover the whole wire path - serialization, the REQ/REP exchange and the
response reshape - rather than a mock of it. What it does not cover is whether the
engine agrees, which needs the engine.
"""

import threading
import uuid

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zmq = pytest.importorskip("zmq")

from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.vla_cpp import vla_pb2  # noqa: E402
from lerobot.vla_cpp.client import VlaCppClient, VlaCppError  # noqa: E402

CHUNK_SIZE, ACTION_DIM = 4, 7


class FakeServer:
    """A REP socket that answers with a canned chunk and records what it was sent."""

    def __init__(self, chunk: np.ndarray | None = None, error: str = ""):
        self.address = f"inproc://vla-cpp-test-{uuid.uuid4().hex}"
        self.chunk = (
            chunk
            if chunk is not None
            else np.arange(CHUNK_SIZE * ACTION_DIM, dtype=np.float32).reshape(CHUNK_SIZE, ACTION_DIM)
        )
        self.error = error
        self.requests: list[vla_pb2.PredictRequest] = []
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(self.address)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=50)):
                continue
            request = vla_pb2.PredictRequest()
            request.ParseFromString(self._sock.recv())
            self.requests.append(request)
            response = vla_pb2.PredictResponse(request_id=request.request_id)
            if self.error:
                response.error = self.error
            else:
                response.action_chunk.extend(self.chunk.reshape(-1).tolist())
                response.chunk_size = self.chunk.shape[0]
                response.action_dim = self.chunk.shape[1]
                response.latency_ms_total = 12.5
                response.latency_ms_inference = 9.0
            self._sock.send(response.SerializeToString())

    @property
    def last(self) -> vla_pb2.PredictRequest:
        assert self.requests, "the server was never called"
        return self.requests[-1]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sock.close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


class StubTokenizer:
    """Records the text it was handed and returns one id per word."""

    def __init__(self):
        self.seen: list[str] = []
        self.chat_template = None

    def __call__(self, text, **kwargs):
        self.seen.append(text)
        ids = np.array([[abs(hash(w)) % 1000 for w in text.split()] or [0]], dtype=np.int64)
        if kwargs.get("return_tensors") == "np":
            return {"input_ids": ids}
        return {"input_ids": ids[0].tolist()}

    def apply_chat_template(self, conversation, **kwargs):
        return " ".join(
            part.get("text", "IMG")
            for turn in conversation
            for part in (
                turn["content"] if isinstance(turn["content"], list) else [{"text": turn["content"]}]
            )
        )


@pytest.fixture
def stub_tokenizer(monkeypatch):
    tok = StubTokenizer()
    monkeypatch.setattr(VlaCppClient, "_load_tokenizer", staticmethod(lambda *a, **k: tok))
    return tok


def _frames(n: int = 2, size: int = 32) -> dict:
    rng = np.random.default_rng(0)
    return {f"observation.images.cam{i}": rng.random((3, size, size), dtype=np.float32) for i in range(n)}


def passthrough_client(server: FakeServer, **kwargs) -> VlaCppClient:
    return VlaCppClient(
        server.address,
        arch="passthrough",
        image_keys=("observation.images.cam0", "observation.images.cam1"),
        action_dim=ACTION_DIM,
        recv_timeout_ms=4000,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------


def test_roundtrip_returns_the_servers_chunk(server):
    with passthrough_client(server) as client:
        observation = {**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""}
        chunk = client.predict_chunk(observation)
    assert chunk.shape == (CHUNK_SIZE, ACTION_DIM)
    np.testing.assert_allclose(chunk, server.chunk)


def test_request_carries_one_image_per_key_in_order(server):
    with passthrough_client(server) as client:
        client.predict_chunk({**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""})
    images = server.last.images
    assert len(images) == 2
    for image in images:
        assert image.encoding == vla_pb2.Image.F32_RGB_01
        assert (image.height, image.width) == (32, 32)
        # HWC float32: 32 * 32 * 3 * 4 bytes.
        assert len(image.data) == 32 * 32 * 3 * 4


def test_state_is_zero_padded_to_the_arch_width(server):
    with passthrough_client(server) as client:
        client.predict_chunk(
            {**_frames(), OBS_STATE: np.array([1, 2, 3, 4, 5, 6], dtype=np.float32), "task": ""}
        )
    state = list(server.last.state)
    assert len(state) == 32
    assert state[:6] == [1, 2, 3, 4, 5, 6]
    assert set(state[6:]) == {0.0}


def test_request_id_increments(server):
    with passthrough_client(server) as client:
        for _ in range(3):
            client.predict_chunk({**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""})
    assert [r.request_id for r in server.requests] == [0, 1, 2]


def test_server_error_becomes_an_exception():
    s = FakeServer(error="no such embodiment")
    try:
        with passthrough_client(s) as client, pytest.raises(VlaCppError, match="no such embodiment"):
            client.predict_chunk({**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""})
    finally:
        s.close()


def test_latencies_are_exposed(server):
    with passthrough_client(server) as client:
        client.predict_chunk({**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""})
        assert client.last_response.latency_ms_total == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# The action queue
# ---------------------------------------------------------------------------


def test_queue_serves_n_action_steps_from_one_query(server):
    with passthrough_client(server, n_action_steps=3) as client:
        observation = {**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""}
        actions = [client.get_action(observation) for _ in range(3)]
    assert len(server.requests) == 1, "the queue should have covered all three steps"
    for i, action in enumerate(actions):
        np.testing.assert_allclose(action, server.chunk[i, :ACTION_DIM])


def test_queue_requeries_once_drained(server):
    with passthrough_client(server, n_action_steps=2) as client:
        observation = {**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""}
        for _ in range(4):
            client.get_action(observation)
    assert len(server.requests) == 2


def test_pending_reports_the_queue_depth(server):
    with passthrough_client(server, n_action_steps=3) as client:
        observation = {**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""}
        assert client.pending == 0
        client.get_action(observation)
        assert client.pending == 2


def test_reset_drops_the_queue(server):
    with passthrough_client(server, n_action_steps=3) as client:
        observation = {**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""}
        client.get_action(observation)
        client.reset()
        assert client.pending == 0
        client.get_action(observation)
    assert len(server.requests) == 2


def test_action_dim_slices_the_chunk(server):
    with passthrough_client(server, n_action_steps=1) as client:
        client.action_dim = 3
        action = client.get_action({**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""})
    assert action.shape == (3,)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_image_key_names_what_was_given(server):
    with passthrough_client(server) as client, pytest.raises(KeyError, match="observation.images.cam0"):
        client.predict_chunk({OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""})


def test_missing_state_is_reported(server):
    with passthrough_client(server) as client, pytest.raises(KeyError, match="observation.state"):
        client.predict_chunk({**_frames(), "task": ""})


def test_state_wider_than_the_arch_is_refused(server):
    with passthrough_client(server) as client, pytest.raises(ValueError, match="max_state_dim"):
        client.predict_chunk({**_frames(), OBS_STATE: np.zeros(64, dtype=np.float32), "task": ""})


def test_wrong_image_layout_is_refused(server):
    with passthrough_client(server) as client, pytest.raises(ValueError, match="CHW"):
        client.predict_chunk(
            {
                "observation.images.cam0": np.zeros((32, 32, 3), dtype=np.float32),
                "observation.images.cam1": np.zeros((3, 32, 32), dtype=np.float32),
                OBS_STATE: np.zeros(6, dtype=np.float32),
                "task": "",
            }
        )


def test_unknown_arch_is_refused(server):
    with pytest.raises(ValueError, match="unknown arch"):
        VlaCppClient(server.address, arch="not-an-arch")


def test_n_action_steps_must_be_positive(server):
    with pytest.raises(ValueError, match="n_action_steps"):
        passthrough_client(server, n_action_steps=0)


def test_an_arch_without_a_default_tokenizer_is_refused(server):
    with pytest.raises(ValueError, match="no default tokenizer"):
        VlaCppClient(server.address, arch="gr00t_n1_6")


# ---------------------------------------------------------------------------
# Tokenized paths
# ---------------------------------------------------------------------------


def test_generic_path_appends_a_newline_to_the_task(server, stub_tokenizer):
    """The checkpoints were trained on prompts carrying it; dropping it shifts every id."""
    client = VlaCppClient(
        server.address,
        arch="smolvla",
        image_keys=("observation.images.cam0",),
        action_dim=ACTION_DIM,
        recv_timeout_ms=4000,
    )
    with client:
        client.predict_chunk(
            {
                "observation.images.cam0": np.zeros((3, 64, 64), dtype=np.float32),
                OBS_STATE: np.zeros(6, dtype=np.float32),
                "task": "pick up the tape",
            }
        )
    assert stub_tokenizer.seen[-1] == "pick up the tape\n"
    assert len(server.last.lang_tokens) == 4


def test_generic_path_does_not_double_the_newline(server, stub_tokenizer):
    client = VlaCppClient(
        server.address,
        arch="smolvla",
        image_keys=("observation.images.cam0",),
        recv_timeout_ms=4000,
    )
    with client:
        client.predict_chunk(
            {
                "observation.images.cam0": np.zeros((3, 64, 64), dtype=np.float32),
                OBS_STATE: np.zeros(6, dtype=np.float32),
                "task": "already ends\n",
            }
        )
    assert stub_tokenizer.seen[-1] == "already ends\n"


def test_smolvla_resizes_to_the_preset(server, stub_tokenizer):
    client = VlaCppClient(
        server.address, arch="smolvla", image_keys=("observation.images.cam0",), recv_timeout_ms=4000
    )
    with client:
        client.predict_chunk(
            {
                "observation.images.cam0": np.zeros((3, 480, 640), dtype=np.float32),
                OBS_STATE: np.zeros(6, dtype=np.float32),
                "task": "x",
            }
        )
    assert (server.last.images[0].height, server.last.images[0].width) == (512, 512)


def test_pi05_sends_a_zero_state_and_digitizes_it_into_the_prompt(server, stub_tokenizer, tmp_path):
    """pi0.5 reads the state from the text, so the float field is deliberately zeros.

    Sending the state in both places would not error, and the policy would read the
    tensor it was never trained to use.
    """
    import json

    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"observation.state": {"q01": [0.0] * 6, "q99": [10.0] * 6}}))

    client = VlaCppClient(
        server.address,
        arch="pi05",
        image_keys=("observation.images.cam0",),
        recv_timeout_ms=4000,
        stats_json=stats,
    )
    with client:
        client.predict_chunk(
            {
                "observation.images.cam0": np.zeros((3, 64, 64), dtype=np.float32),
                OBS_STATE: np.full(6, 5.0, dtype=np.float32),
                "task": "put_the tape\nin the box",
            }
        )

    assert set(server.last.state) == {0.0}, "pi05's float state field must be zeros"
    prompt = stub_tokenizer.seen[-1]
    assert prompt.startswith("Task: put the tape in the box, State: ")
    assert prompt.endswith(";\nAction: ")
    # 5.0 is the midpoint of [0, 10] -> normalized 0.0, and bins[128] is exactly 0.0,
    # so digitize lands it in bin 128. The endpoints are 0 and 255.
    assert prompt.count(" 128") == 6


def test_pi05_without_statistics_is_refused(server, stub_tokenizer):
    with pytest.raises(ValueError, match="quantiles"):
        VlaCppClient(server.address, arch="pi05", recv_timeout_ms=4000)
