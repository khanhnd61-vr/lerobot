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
"""Asynchronous inference on top of :class:`~lerobot.vla_cpp.client.VlaCppClient`.

The synchronous client stops the control loop for a whole round trip every time its
queue runs dry. This module moves the round trip to a worker thread and keeps a
**timestep-aligned action queue**, the scheme of lerobot's own
:mod:`~lerobot.async_inference`: an observation is stamped with the timestep of the
last action executed, the chunk it produces covers that timestep onwards, and when
the chunk arrives the actions whose timesteps have already passed are dropped, the
ones that overlap the queue are blended with an aggregate function, and the rest are
appended. The robot keeps executing the old chunk while the new one is computed.

Only one request is in flight at a time, because the client's socket is REQ and the
server serves one model. Freshness comes from *when* the request is made: the control
loop offers an observation every tick and the worker takes one only when it is idle
and the queue has drained to ``chunk_size_threshold``, so the observation the model
sees is at most one tick old when it leaves.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from .client import VlaCppClient

logger = logging.getLogger(__name__)

AggregateFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: How two chunks' actions for the same timestep are combined. Same names and
#: weights as :data:`lerobot.async_inference.configs.AGGREGATE_FUNCTIONS`; kept here
#: so the vla.cpp path needs neither torch tensors nor grpc.
AGGREGATE_FUNCTIONS: dict[str, AggregateFn] = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}

#: Consecutive failed round trips before the worker gives up and sets ``failed``.
MAX_CONSECUTIVE_FAILURES = 10


def get_aggregate_function(name: str) -> AggregateFn:
    """Look an aggregate function up by name."""
    if name not in AGGREGATE_FUNCTIONS:
        raise ValueError(
            f"unknown aggregate function {name!r}; expected one of {sorted(AGGREGATE_FUNCTIONS)}"
        )
    return AGGREGATE_FUNCTIONS[name]


class TimedActionQueue:
    """Actions keyed by the timestep they are meant for, oldest first.

    Thread-safe. ``latest_action`` is the timestep of the last action popped, -1
    before any was; it is what the next observation is stamped with.
    """

    def __init__(self, aggregate_fn: AggregateFn):
        """Create an empty queue that blends overlapping actions with `aggregate_fn`."""
        self._aggregate_fn = aggregate_fn
        self._lock = threading.Lock()
        self._queue: deque[tuple[int, np.ndarray]] = deque()
        self.latest_action = -1

    def __len__(self) -> int:
        """How many actions are queued."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Drop everything and forget the last executed timestep."""
        with self._lock:
            self._queue.clear()
            self.latest_action = -1

    def pop(self) -> np.ndarray | None:
        """The next action, or ``None`` when starved."""
        with self._lock:
            if not self._queue:
                return None
            timestep, action = self._queue.popleft()
            self.latest_action = timestep
            return action

    def peek_timesteps(self) -> list[int]:
        """The queued timesteps, oldest first, without popping anything."""
        with self._lock:
            return [t for t, _ in self._queue]

    def merge(self, first_timestep: int, chunk: np.ndarray) -> tuple[int, int, int]:
        """Fold a chunk whose row ``i`` is the action for ``first_timestep + i``.

        Returns ``(dropped, blended, appended)``: rows already executed, rows that
        overlapped the queue and were aggregated, and rows added at the tail.
        """
        with self._lock:
            incoming = {first_timestep + i: np.asarray(row, dtype=np.float32) for i, row in enumerate(chunk)}
            dropped = sum(1 for t in incoming if t <= self.latest_action)
            fresh = {t: a for t, a in incoming.items() if t > self.latest_action}

            merged: dict[int, np.ndarray] = {}
            blended = 0
            for t, old in self._queue:
                if t <= self.latest_action:
                    continue
                if t in fresh:
                    merged[t] = np.asarray(self._aggregate_fn(old, fresh.pop(t)), dtype=np.float32)
                    blended += 1
                else:
                    merged[t] = old
            appended = len(fresh)
            merged.update(fresh)

            self._queue = deque(sorted(merged.items()))
            return dropped, blended, appended


class AsyncVlaCppClient:
    """A worker thread that keeps a :class:`TimedActionQueue` filled from a server.

    The control loop calls :meth:`pop_action` every tick and :meth:`offer` with a
    fresh observation whenever it has one; the worker takes an observation only when
    it is idle and the queue is at or below ``actions_per_chunk * chunk_size_threshold``
    entries, and merges each chunk back as it arrives. Preprocessing and the round trip
    both happen on the worker, so a tick costs the control thread nothing but the
    observation.

    The wrapped ``client`` is used from the worker thread only, which is what its
    one-socket design requires; do not call it from elsewhere while this is running.
    """

    def __init__(
        self,
        client: VlaCppClient,
        *,
        actions_per_chunk: int,
        chunk_size_threshold: float = 0.5,
        aggregate_fn: AggregateFn | str = "weighted_average",
        fps: float = 30.0,
    ):
        """Wrap `client`; nothing runs until :meth:`start`.

        Args:
            client: The synchronous client whose `predict_chunk` does the round trip.
            actions_per_chunk: Rows of each chunk to keep; the rest are discarded.
            chunk_size_threshold: Fraction of `actions_per_chunk` at or below which the
                queue is refilled. Must exceed `latency * fps / actions_per_chunk`.
            aggregate_fn: A key of :data:`AGGREGATE_FUNCTIONS` or a callable blending
                the queued and the incoming action for one timestep.
            fps: Control rate, used only to pace retries after a failed round trip.
        """
        if actions_per_chunk < 1:
            raise ValueError(f"actions_per_chunk must be >= 1, got {actions_per_chunk}")
        if not 0.0 <= chunk_size_threshold <= 1.0:
            raise ValueError(f"chunk_size_threshold must be in [0, 1], got {chunk_size_threshold}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.client = client
        self.actions_per_chunk = actions_per_chunk
        self.chunk_size_threshold = chunk_size_threshold
        self.aggregate_fn = (
            get_aggregate_function(aggregate_fn) if isinstance(aggregate_fn, str) else aggregate_fn
        )
        self.dt = 1.0 / fps
        self.queue = TimedActionQueue(self.aggregate_fn)

        self._cv = threading.Condition()
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._in_flight = False
        self._stop = False
        self._thread: threading.Thread | None = None

        self.failed = False
        self.failure: BaseException | None = None
        self._consecutive_failures = 0
        self.queries = 0
        self.last_latency_ms = 0.0
        self.last_delay_steps = 0
        self.last_report = ""

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> AsyncVlaCppClient:
        """Start the worker thread. Returns self so it can be chained."""
        if self._thread is not None:
            raise RuntimeError("already started")
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="vla-cpp-inference", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> None:
        """Stop the worker. Waits for a round trip in progress, up to ``timeout``."""
        with self._cv:
            self._stop = True
            self._pending = None
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def reset(self) -> None:
        """Drop queued actions and any offered observation. Call between episodes."""
        with self._cv:
            self._pending = None
        self.queue.clear()

    def __enter__(self) -> AsyncVlaCppClient:
        """Start the worker for the duration of a `with` block."""
        return self.start()

    def __exit__(self, *_) -> None:
        """Stop the worker."""
        self.stop()

    # -- the control-loop side ----------------------------------------------

    @property
    def in_flight(self) -> bool:
        """True while a round trip is in progress on the worker."""
        with self._cv:
            return self._in_flight

    @property
    def qsize(self) -> int:
        """How many actions are queued."""
        return len(self.queue)

    def ready_to_send(self) -> bool:
        """True when an observation offered now would go out immediately."""
        with self._cv:
            if self._in_flight or self._pending is not None or self._stop or self.failed:
                return False
        return len(self.queue) <= self.actions_per_chunk * self.chunk_size_threshold

    def offer(self, observation: dict[str, Any]) -> bool:
        """Hand the worker an observation, stamped with the last executed timestep.

        Returns False, and does nothing, when a request is already in flight or the
        queue is still above the threshold. Call it every tick; it is cheap.
        """
        if not self.ready_to_send():
            return False
        with self._cv:
            if self._in_flight or self._pending is not None or self._stop or self.failed:
                return False
            self._pending = (max(self.queue.latest_action, 0), observation)
            self._in_flight = True
            self._cv.notify()
        return True

    def pop_action(self) -> np.ndarray | None:
        """The action for this tick, or ``None`` when the queue is empty."""
        return self.queue.pop()

    # -- the worker ---------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    self._in_flight = False
                    return
                timestep, observation = self._pending
                self._pending = None
            try:
                self._query(timestep, observation)
                self._consecutive_failures = 0
            except Exception as e:  # noqa: BLE001 - the thread must report, not die
                self._consecutive_failures += 1
                logger.warning(
                    "vla.cpp query failed (%d/%d): %s",
                    self._consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES,
                    e,
                )
                if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.failure = e
                    self.failed = True
                    with self._cv:
                        self._in_flight = False
                    return
                time.sleep(min(0.5, self.dt))
            finally:
                with self._cv:
                    self._in_flight = False

    def _query(self, timestep: int, observation: dict[str, Any]) -> None:
        started = time.perf_counter()
        chunk = self.client.predict_chunk(observation)[: self.actions_per_chunk, : self.client.action_dim]
        latency = time.perf_counter() - started
        dropped, blended, appended = self.queue.merge(timestep, chunk)

        self.queries += 1
        self.last_latency_ms = latency * 1000.0
        self.last_delay_steps = max(0, self.queue.latest_action - timestep)
        response = self.client.last_response
        server_ms = response.latency_ms_total if response is not None else 0.0
        self.last_report = (
            f"query {self.queries} | obs t={timestep} | {self.last_latency_ms:.1f} ms round trip "
            f"({server_ms:.1f} ms server) | {self.last_delay_steps} steps elapsed | "
            f"{dropped} dropped, {blended} blended, {appended} appended | queue {len(self.queue)}"
        )
        logger.info(self.last_report)
