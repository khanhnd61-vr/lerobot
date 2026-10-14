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
"""Asynchronous inference for the vla.cpp client.

The queue tests are pure. The worker tests run against the same real-socket fake
server as the synchronous client's tests, with a reply delay so the control-loop
side can be observed while a round trip is in flight.
"""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zmq = pytest.importorskip("zmq")

from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.vla_cpp.async_client import (  # noqa: E402
    AGGREGATE_FUNCTIONS,
    MAX_CONSECUTIVE_FAILURES,
    AsyncVlaCppClient,
    TimedActionQueue,
    get_aggregate_function,
)
from lerobot.vla_cpp.client import VlaCppClient  # noqa: E402
from tests.vla_cpp.test_client import ACTION_DIM, CHUNK_SIZE, FakeServer, _frames  # noqa: E402


def _chunk(first: int, n: int = CHUNK_SIZE, dim: int = ACTION_DIM) -> np.ndarray:
    """Row i is filled with first + i, so a row's value says which timestep it is for."""
    return np.repeat(np.arange(first, first + n, dtype=np.float32)[:, None], dim, axis=1)


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------------------
# TimedActionQueue
# ---------------------------------------------------------------------------


def test_first_chunk_is_appended_whole():
    q = TimedActionQueue(AGGREGATE_FUNCTIONS["latest_only"])
    assert q.merge(0, _chunk(0)) == (0, 0, CHUNK_SIZE)
    assert q.peek_timesteps() == [0, 1, 2, 3]
    assert q.latest_action == -1


def test_pop_advances_latest_action_and_starves_when_empty():
    q = TimedActionQueue(AGGREGATE_FUNCTIONS["latest_only"])
    q.merge(0, _chunk(0))
    np.testing.assert_array_equal(q.pop(), np.full(ACTION_DIM, 0.0))
    np.testing.assert_array_equal(q.pop(), np.full(ACTION_DIM, 1.0))
    assert q.latest_action == 1
    q.pop()
    q.pop()
    assert q.pop() is None
    assert q.latest_action == 3


def test_elapsed_steps_are_dropped():
    """A chunk predicted from the observation at t=1 arrives after t=1..2 were executed."""
    q = TimedActionQueue(AGGREGATE_FUNCTIONS["latest_only"])
    q.merge(0, _chunk(0))
    q.pop()  # t=0
    q.pop()  # t=1 -> the observation stamp
    q.pop()  # t=2 elapsed during inference
    dropped, blended, appended = q.merge(1, _chunk(1))
    assert (dropped, blended, appended) == (2, 1, 1)  # t=1,2 dropped; t=3 blended; t=4 appended
    assert q.peek_timesteps() == [3, 4]


def test_overlap_is_blended_with_the_aggregate_fn():
    q = TimedActionQueue(AGGREGATE_FUNCTIONS["average"])
    q.merge(0, _chunk(0))  # rows 0..3 valued 0..3
    q.pop()  # latest_action = 0
    q.merge(1, _chunk(11, n=2))  # t=1 -> 11, t=2 -> 12
    t1, t2, t3 = q.pop(), q.pop(), q.pop()
    np.testing.assert_allclose(t1, 0.5 * 1 + 0.5 * 11)
    np.testing.assert_allclose(t2, 0.5 * 2 + 0.5 * 12)
    np.testing.assert_allclose(t3, 3.0)  # untouched tail of the old chunk


def test_latest_only_replaces_the_overlap():
    q = TimedActionQueue(AGGREGATE_FUNCTIONS["latest_only"])
    q.merge(0, _chunk(0))
    q.pop()
    q.merge(1, _chunk(11, n=2))
    np.testing.assert_allclose(q.pop(), 11.0)
    np.testing.assert_allclose(q.pop(), 12.0)


def test_clear_forgets_everything():
    q = TimedActionQueue(AGGREGATE_FUNCTIONS["latest_only"])
    q.merge(0, _chunk(0))
    q.pop()
    q.clear()
    assert len(q) == 0 and q.latest_action == -1


def test_unknown_aggregate_fn_is_refused():
    with pytest.raises(ValueError, match="unknown aggregate function"):
        get_aggregate_function("mean-ish")


# ---------------------------------------------------------------------------
# AsyncVlaCppClient against the fake server
# ---------------------------------------------------------------------------


def _observation() -> dict:
    return {**_frames(), OBS_STATE: np.zeros(6, dtype=np.float32), "task": ""}


def _client(server: FakeServer) -> VlaCppClient:
    return VlaCppClient(
        server.address,
        arch="passthrough",
        image_keys=("observation.images.cam0", "observation.images.cam1"),
        action_dim=ACTION_DIM,
        recv_timeout_ms=4000,
    )


def test_actions_arrive_without_blocking_the_caller():
    server = FakeServer(chunk=_chunk(0), delay_s=0.2)
    try:
        with AsyncVlaCppClient(_client(server), actions_per_chunk=CHUNK_SIZE, fps=30) as engine:
            assert engine.pop_action() is None, "nothing should be queued before the first reply"
            t0 = time.perf_counter()
            assert engine.offer(_observation())
            assert time.perf_counter() - t0 < 0.1, "offer must not wait for the round trip"
            assert engine.in_flight and not engine.ready_to_send()
            assert not engine.offer(_observation()), "one request in flight at a time"
            assert _wait(lambda: engine.qsize == CHUNK_SIZE)
            assert engine.queries == 1
            np.testing.assert_allclose(engine.pop_action(), 0.0)
    finally:
        server.close()
    assert len(server.requests) == 1


def test_next_observation_goes_out_at_the_threshold_and_is_stamped():
    server = FakeServer(chunk=_chunk(0))
    try:
        with AsyncVlaCppClient(
            _client(server), actions_per_chunk=CHUNK_SIZE, chunk_size_threshold=0.5, fps=30
        ) as engine:
            engine.offer(_observation())
            assert _wait(lambda: engine.qsize == CHUNK_SIZE)
            # Four queued, threshold is two: not yet.
            assert not engine.ready_to_send()
            engine.pop_action()  # t=0
            assert not engine.ready_to_send()
            engine.pop_action()  # t=1 -> two left
            assert engine.ready_to_send()
            assert engine.offer(_observation())
            assert _wait(lambda: engine.queries == 2)
            # Stamped with the last executed timestep: rows 0 and 1 of the new chunk
            # are for t=1 (dropped) and t=2 (blended); t=3 blended; t=4 appended.
            assert engine.last_delay_steps == 0
            assert engine.queue.peek_timesteps() == [2, 3, 4]
    finally:
        server.close()


def test_elapsed_steps_during_a_slow_round_trip_are_skipped():
    server = FakeServer(chunk=_chunk(0), delay_s=0.3)
    try:
        with AsyncVlaCppClient(
            _client(server), actions_per_chunk=CHUNK_SIZE, chunk_size_threshold=1.0, fps=30
        ) as engine:
            engine.offer(_observation())
            assert _wait(lambda: engine.qsize == CHUNK_SIZE)
            engine.pop_action()  # t=0 executed; the observation is stamped t=0
            assert engine.offer(_observation())
            # Two more ticks pass while the server is busy.
            engine.pop_action()  # t=1
            engine.pop_action()  # t=2
            assert _wait(lambda: engine.queries == 2)
            assert engine.last_delay_steps == 2
            # New rows t=0,1,2 dropped; t=3 blended with the old tail.
            assert engine.queue.peek_timesteps() == [3]
            assert "3 dropped" in engine.last_report
    finally:
        server.close()


def test_reset_drops_the_queue_and_restarts_the_timeline():
    server = FakeServer(chunk=_chunk(0))
    try:
        with AsyncVlaCppClient(_client(server), actions_per_chunk=CHUNK_SIZE) as engine:
            engine.offer(_observation())
            assert _wait(lambda: engine.qsize == CHUNK_SIZE)
            engine.pop_action()
            engine.reset()
            assert engine.qsize == 0 and engine.queue.latest_action == -1
            assert engine.ready_to_send()
    finally:
        server.close()


def test_server_errors_eventually_mark_the_worker_failed():
    server = FakeServer(error="no such embodiment")
    try:
        engine = AsyncVlaCppClient(_client(server), actions_per_chunk=CHUNK_SIZE, fps=1000)
        engine.start()
        try:
            for _ in range(MAX_CONSECUTIVE_FAILURES):
                assert _wait(lambda: not engine.in_flight)
                if engine.failed:
                    break
                assert engine.offer(_observation())
            assert _wait(lambda: engine.failed)
            assert "no such embodiment" in str(engine.failure)
            assert not engine.offer(_observation()), "a failed worker takes nothing more"
        finally:
            engine.stop(timeout=2.0)
    finally:
        server.close()


def test_stop_returns_while_a_round_trip_is_pending():
    server = FakeServer(chunk=_chunk(0), delay_s=0.3)
    try:
        engine = AsyncVlaCppClient(_client(server), actions_per_chunk=CHUNK_SIZE).start()
        engine.offer(_observation())
        t0 = time.perf_counter()
        engine.stop(timeout=2.0)
        assert time.perf_counter() - t0 < 1.5
        assert not engine.in_flight
    finally:
        server.close()


def test_bad_parameters_are_refused():
    server = FakeServer()
    try:
        client = _client(server)
        with pytest.raises(ValueError, match="actions_per_chunk"):
            AsyncVlaCppClient(client, actions_per_chunk=0)
        with pytest.raises(ValueError, match="chunk_size_threshold"):
            AsyncVlaCppClient(client, actions_per_chunk=4, chunk_size_threshold=1.5)
        with pytest.raises(ValueError, match="unknown aggregate function"):
            AsyncVlaCppClient(client, actions_per_chunk=4, aggregate_fn="nope")
    finally:
        server.close()
