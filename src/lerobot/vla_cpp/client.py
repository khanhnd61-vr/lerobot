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
"""A client for a `vla.cpp` inference server.

`vla-server` loads one GGUF checkpoint and answers **ZeroMQ REQ/REP** requests
carrying **protobuf** - not lerobot's own gRPC async-inference protocol, which is
what `lerobot-vla-simd` speaks. The two are unrelated wire formats and this module
implements the former.

The division of labour is the thing to understand before reading further: the
engine runs the network, and **the client owns the preprocessing**. Tokenization,
image resizing, state normalization and action un-normalization all happen here,
because the server receives token ids and normalized floats. So an arch is not
"supported" by pointing this client at a server - it is supported by getting its
preprocessing right, which is why :mod:`~lerobot.vla_cpp.archs` and
:mod:`~lerobot.vla_cpp.stats` exist and why ``passthrough`` is offered separately
for servers that do their own.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from . import vla_pb2
from .archs import (
    FIXED_NOISE_ARCHS,
    GR00T_CROP_FRACTION,
    GR00T_N15_IMG_CTX_ID,
    GR00T_N15_N_TOK_PER_VIEW,
    GR00T_N16_IMG_PAD_ID,
    GR00T_N16_N_TOK_PER_VIEW,
    GR00T_N17_IMG_PAD_ID,
    GR00T_N17_N_TOK_PER_VIEW,
    GR00T_N17_SHORTEST_EDGE,
    GR00T_N17_TARGET_SIZE,
    GR00T_STATE_DIMS,
    GR00T_STATE_KEYS,
    GR00T_VIDEO_KEYS,
    MODALITY_KEYED_ARCHS,
    resolve_preset,
)
from .preprocessing import (
    gr00t_image_transform,
    gr00t_n16_image_transform,
    resize_with_pad,
)
from .stats import Gr00tNormalizers, build_gr00t_normalizers, pi05_state_quantiles

DEFAULT_ADDRESS = "tcp://localhost:5555"
DEFAULT_RECV_TIMEOUT_MS = 30_000

#: pi0.5 spells the state into its prompt as this many uniform bins over [-1, 1].
PI05_STATE_BINS = 256


class VlaCppError(RuntimeError):
    """The server answered with an error string rather than an action chunk."""


class VlaCppClient:
    """Talks to one `vla-server` over ZeroMQ, doing the architecture's preprocessing.

    One instance owns one socket and one action queue, so it is not thread-safe and
    not shareable between control loops.
    """

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        *,
        arch: str = "smolvla",
        tokenizer: str | None = None,
        image_size: int | None = None,
        max_state_dim: int | None = None,
        max_length: int | None = None,
        image_keys: tuple[str, ...] = (
            "observation.images.front",
            "observation.images.wrist",
        ),
        action_dim: int = 7,
        n_action_steps: int = 1,
        recv_timeout_ms: int = DEFAULT_RECV_TIMEOUT_MS,
        stats_json: str | Path | None = None,
        rel_stats_json: str | Path | None = None,
        embodiment: str | None = None,
    ):
        """Connect to a server and build the architecture's preprocessing.

        Args:
            address (`str`, *optional*, defaults to `"tcp://localhost:5555"`):
                ZeroMQ endpoint `vla-server` is bound to.
            arch (`str`, *optional*, defaults to `"smolvla"`):
                Which preprocessing to apply, a key of
                [`~lerobot.vla_cpp.archs.ARCH_PRESETS`]. `"passthrough"` applies none.
            tokenizer (`str`, *optional*):
                Hub id or local directory, overriding the architecture's default.
                Required for any architecture whose preset has no default.
            image_size (`int`, *optional*):
                Square size frames are resized to, overriding the preset. `None`
                sends them at their native resolution.
            max_state_dim (`int`, *optional*):
                Width the state vector is zero-padded to, overriding the preset.
                A preset of `None` (ACT) sends the state at its own width.
            max_length (`int`, *optional*):
                Token budget for the prompt, overriding the preset.
            image_keys (`tuple[str, ...]`, *optional*, defaults to front and wrist):
                Observation keys to read frames from, **in the order the checkpoint
                expects them**. A swapped pair produces plausible, wrong actions
                rather than an error.
            action_dim (`int`, *optional*, defaults to `7`):
                How many leading columns of the returned chunk to execute.
            n_action_steps (`int`, *optional*, defaults to `1`):
                How many steps of one chunk to hand out before re-querying.
            recv_timeout_ms (`int`, *optional*, defaults to `30000`):
                How long to wait for a reply before raising.
            stats_json (`str` or `Path`, *optional*):
                Statistics JSON for normalization. Required for `"pi05"`; strongly
                recommended for the GR00T architectures.
            rel_stats_json (`str` or `Path`, *optional*):
                Relative-action statistics, for a GR00T N1.7 checkpoint trained with
                them.
            embodiment (`str`, *optional*):
                Which top-level key of `stats_json` to normalize with. Inferred when
                there is exactly one candidate.

        Raises:
            ValueError: If `arch` is unknown, if a required tokenizer or statistics
                file was not given, or if `n_action_steps` is below 1.
            ImportError: If `pyzmq` or `transformers` is not installed.
        """
        require_package("pyzmq", extra="vla-cpp", import_name="zmq")
        import zmq

        preset = resolve_preset(arch)
        self.arch = arch
        self.image_size = image_size if image_size is not None else preset["image_size"]
        self.max_state_dim = max_state_dim if max_state_dim is not None else preset.get("max_state_dim")
        self.max_length = max_length if max_length is not None else preset.get("max_length", 48)
        self.image_keys = tuple(image_keys)
        self.action_dim = action_dim
        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        self.n_action_steps = n_action_steps

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        # REQ_RELAXED + REQ_CORRELATE let a timed-out request be retried without
        # the socket refusing to send again, which a strict REQ socket would.
        self._sock.setsockopt(zmq.REQ_RELAXED, 1)
        self._sock.setsockopt(zmq.REQ_CORRELATE, 1)
        self._sock.connect(address)
        self.address = address
        logging.info("vla.cpp[%s]: connected to %s", arch, address)

        self._tok = self._load_tokenizer(arch, tokenizer, preset)

        self._queue: deque[np.ndarray] = deque(maxlen=n_action_steps)
        self._step = 0
        self._episode = 0
        self._noise_len: int | None = None
        self.last_response: Any | None = None

        self._pi05_q01: np.ndarray | None = None
        self._pi05_q99: np.ndarray | None = None
        if arch == "pi05":
            if stats_json is None:
                raise ValueError(
                    "arch=pi05 digitizes the state into its prompt, so it needs state "
                    "quantiles: pass stats_json (a LeRobot meta/stats.json)."
                )
            self._pi05_q01, self._pi05_q99 = pi05_state_quantiles(stats_json)
            logging.info(
                "vla.cpp[pi05]: state quantiles (%d-D) from %s",
                self._pi05_q01.size,
                stats_json,
            )

        self.gr00t: Gr00tNormalizers | None = None
        if arch in MODALITY_KEYED_ARCHS and stats_json is not None:
            self.gr00t = build_gr00t_normalizers(
                arch, stats_json, embodiment=embodiment, rel_stats_json=rel_stats_json
            )
            logging.info("vla.cpp[%s]: normalizers - %s", arch, self.gr00t.description)
        elif arch in MODALITY_KEYED_ARCHS:
            logging.warning(
                "vla.cpp[%s]: no stats_json, so state is sent un-normalized and actions "
                "are returned un-normalized. This is almost never what you want on real "
                "hardware.",
                arch,
            )
        self._last_state_raw: np.ndarray | None = None

    # -- setup helpers ------------------------------------------------------

    @staticmethod
    def _load_tokenizer(arch: str, tokenizer: str | None, preset: dict[str, Any]):
        name = tokenizer if tokenizer is not None else preset["tokenizer"]
        if name is None:
            if arch in ("passthrough", "act"):
                return None
            raise ValueError(
                f"arch={arch} has no default tokenizer; pass one (an HF id or a local checkpoint directory)."
            )
        require_package("transformers", extra="vla-cpp")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=bool(preset.get("trust_remote_code", False)),
            use_fast=bool(preset.get("use_fast_tokenizer", True)),
        )
        if arch == "gr00t_n1_6":
            # N1.6's chat template is not in tokenizer_config.json; it sits beside
            # the processor. Without it apply_chat_template emits a default that
            # does not carry the image blocks.
            import json

            path = Path(name) / "chat_template.json"
            if path.exists():
                tok.chat_template = json.loads(path.read_text())["chat_template"]
        return tok

    # -- lifecycle ----------------------------------------------------------

    @property
    def pending(self) -> int:
        """How many queued actions are left before the next request goes out."""
        return len(self._queue)

    def reset(self) -> None:
        """Drop the queued chunk. Call between episodes."""
        self._queue.clear()
        self._episode += 1
        self._step = 0
        self._last_state_raw = None

    def close(self) -> None:
        """Close the socket. Safe to call twice; never raises."""
        # Teardown runs on the shutdown path, where a socket that will not close is
        # not a reason to fail the caller.
        with contextlib.suppress(Exception):
            self._sock.close()

    def __enter__(self) -> VlaCppClient:
        """Enter a context that closes the socket on exit."""
        return self

    def __exit__(self, *_) -> None:
        """Close the socket."""
        self.close()

    # -- the control-loop entry point ---------------------------------------

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        """One action, re-querying the server only when the queue is empty."""
        if not self._queue:
            chunk = self.predict_chunk(observation)
            for row in chunk[: self.n_action_steps, : self.action_dim]:
                self._queue.append(np.ascontiguousarray(row, dtype=np.float32))
            if not self._queue:
                raise VlaCppError(
                    f"server returned a {chunk.shape} chunk, so there was nothing to "
                    f"execute; check action_dim and the checkpoint's chunk size."
                )
        return self._queue.popleft()

    def predict_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        """``(chunk_size, action_dim)`` for one observation, un-normalized if possible."""
        if self.arch == "pi05":
            return self._predict_pi05(observation)
        if self.arch == "gr00t_n1_7":
            return self._predict_gr00t_n17(observation)
        if self.arch == "gr00t_n1_6":
            return self._predict_gr00t_n16(observation)
        if self.arch == "gr00t_n1_5":
            return self._predict_gr00t_n15(observation)
        return self._predict_generic(observation)

    # -- wire ---------------------------------------------------------------

    def _new_request(self) -> Any:
        req = vla_pb2.PredictRequest()
        req.request_id = self._step
        self._step += 1
        return req

    @staticmethod
    def _add_image(req: Any, array: np.ndarray, encoding: int) -> None:
        image = req.images.add()
        image.encoding = encoding
        image.height = array.shape[0]
        image.width = array.shape[1]
        image.data = array.tobytes()

    def _maybe_pin_noise(self, req: Any) -> None:
        """Attach reproducible flow-matching noise when ``VLA_FIXED_NOISE_SEED`` is set.

        Without it the server draws noise from a clock-seeded RNG, so two servers
        cannot be compared action for action. Pinning it here is what makes a kernel
        change verifiable: same inputs plus same noise must give the same actions.

        Finding the right length costs one extra round trip on the first call, since
        it is ``chunk_size * action_dim`` and only the server knows those.
        """
        seed = os.environ.get("VLA_FIXED_NOISE_SEED")
        if seed is None or self.arch not in FIXED_NOISE_ARCHS:
            return
        if self._noise_len is None:
            probe = self._roundtrip(req)
            self._noise_len = probe.chunk_size * probe.action_dim
        rng = np.random.default_rng([int(seed), self._episode, self._step])
        req.noise.extend(rng.standard_normal(self._noise_len, dtype=np.float32).tolist())

    def _roundtrip(self, req: Any) -> Any:
        self._sock.send(req.SerializeToString())
        response = vla_pb2.PredictResponse()
        response.ParseFromString(self._sock.recv())
        if response.error:
            raise VlaCppError(f"vla-server: {response.error}")
        self.last_response = response
        return response

    def _send(self, req: Any) -> np.ndarray:
        response = self._roundtrip(req)
        return np.array(response.action_chunk, dtype=np.float32).reshape(
            response.chunk_size, response.action_dim
        )

    # -- observation readers ------------------------------------------------

    def _float_frames(self, observation: dict[str, Any]) -> list[np.ndarray]:
        """CHW float frames -> HWC float32 at ``image_size``, in ``image_keys`` order."""
        frames = []
        for key in self.image_keys:
            if key not in observation:
                raise KeyError(f"image key {key!r} missing; got {sorted(observation)}")
            img = _as_numpy(observation[key]).astype(np.float32)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"{key}: expected CHW float [3, H, W], got {img.shape}")
            if self.image_size is not None:
                img = resize_with_pad(img, self.image_size, self.image_size)
            frames.append(np.ascontiguousarray(np.transpose(img, (1, 2, 0)), dtype=np.float32))
        return frames

    def _flat_state(self, observation: dict[str, Any]) -> np.ndarray:
        if OBS_STATE not in observation:
            raise KeyError(f"{OBS_STATE!r} missing; got {sorted(observation)}")
        return _as_numpy(observation[OBS_STATE]).astype(np.float32).reshape(-1)

    def _padded_state(self, state: np.ndarray) -> np.ndarray:
        if self.max_state_dim is None:
            return state
        if state.size > self.max_state_dim:
            raise ValueError(
                f"state is {state.size}-D but arch={self.arch} sends "
                f"{self.max_state_dim}-D; pass max_state_dim if this checkpoint differs."
            )
        out = np.zeros(self.max_state_dim, dtype=np.float32)
        out[: state.size] = state
        return out

    @staticmethod
    def _task(observation: dict[str, Any]) -> str:
        task = observation.get("task", "")
        if isinstance(task, tuple):
            task = task[0] if task else ""
        if isinstance(task, bytes):
            task = task.decode()
        return str(task)

    def _tokenize(self, text: str) -> np.ndarray:
        if self._tok is None:
            return np.zeros(0, dtype=np.int32)
        out = self._tok(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return out["input_ids"][0].astype(np.int32)

    # -- per-arch paths -----------------------------------------------------

    def _predict_generic(self, observation: dict[str, Any]) -> np.ndarray:
        """SmolVLA, pi0, ACT and passthrough: frames, flat padded state, tokenized task.

        The trailing newline on the task is not cosmetic - the checkpoints were
        trained on prompts that carry it, and dropping it shifts every token id
        after the instruction.
        """
        frames = self._float_frames(observation)
        state = self._padded_state(self._flat_state(observation))
        task = self._task(observation)
        if task and not task.endswith("\n"):
            task += "\n"

        req = self._new_request()
        for frame in frames:
            self._add_image(req, frame, vla_pb2.Image.F32_RGB_01)
        req.lang_tokens.extend(int(t) for t in self._tokenize(task))
        req.state.extend(float(x) for x in state)
        self._maybe_pin_noise(req)
        return self._send(req)

    def _predict_pi05(self, observation: dict[str, Any]) -> np.ndarray:
        """pi0.5: the state is digitized into the prompt, and ``state`` is sent as zeros.

        Both halves of that matter. The bins are ``linspace(-1, 1, 257)[:-1]`` and
        ``digitize`` is offset by one so the first bin is 0, matching the reference;
        and the float state field is deliberately zero, because the policy reads the
        state from the text, not from the tensor.
        """
        frames = self._float_frames(observation)
        state = self._flat_state(observation)
        width = self._pi05_q01.size
        normed = 2.0 * (state[:width] - self._pi05_q01) / (self._pi05_q99 - self._pi05_q01) - 1.0
        bins = np.linspace(-1.0, 1.0, PI05_STATE_BINS + 1)[:-1]
        discrete = np.digitize(normed, bins=bins) - 1

        cleaned = self._task(observation).strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {cleaned}, State: {' '.join(map(str, discrete.tolist()))};\nAction: "

        req = self._new_request()
        for frame in frames:
            self._add_image(req, frame, vla_pb2.Image.F32_RGB_01)
        req.lang_tokens.extend(int(t) for t in self._tokenize(prompt))
        req.state.extend([0.0] * self.max_state_dim)
        self._maybe_pin_noise(req)
        return self._send(req)

    # -- GR00T --------------------------------------------------------------

    def _gr00t_state(self, observation: dict[str, Any]) -> np.ndarray:
        """Concatenate ``state.<modality>`` in the layout the statistics declare.

        GR00T's observations are modality-keyed, so this is where a caller driving a
        plain robot has to have split its flat state already - see
        ``lerobot_vla_cpp.py``, which does that split from the same layout.
        """
        keys, dims = (
            (self.gr00t.state_keys, self.gr00t.state_dims)
            if self.gr00t is not None
            else (GR00T_STATE_KEYS, GR00T_STATE_DIMS)
        )
        parts = []
        for key, dim in zip(keys, dims, strict=True):
            name = f"state.{key}"
            if name not in observation:
                raise KeyError(
                    f"{self.arch} needs {name!r}; this checkpoint's statistics declare "
                    f"{list(keys)}, got {sorted(observation)}"
                )
            value = _as_numpy(observation[name]).astype(np.float32).reshape(-1)
            if value.size != dim:
                raise ValueError(f"{name}: expected {dim}-D, got {value.size}-D")
            parts.append(value)
        return np.concatenate(parts).astype(np.float32)

    def _gr00t_u8_frames(self, observation: dict[str, Any]) -> list[np.ndarray]:
        frames = []
        for key in GR00T_VIDEO_KEYS:
            if key not in observation:
                raise KeyError(f"{self.arch} needs {key!r}; got {sorted(observation)}")
            img = _as_numpy(observation[key])
            while img.ndim > 3:
                img = img[0]
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"{key}: expected HWC uint8 [H, W, 3], got {img.shape}")
            frames.append(np.asarray(img, dtype=np.uint8))
        return frames

    def _gr00t_prepare(self, observation: dict[str, Any]) -> tuple[np.ndarray, str]:
        """Normalized padded state plus the punctuation-stripped instruction.

        The reference lowercases and strips punctuation before tokenizing; the
        checkpoints were trained that way, so a capitalized instruction is a
        different prompt.
        """
        raw = self._gr00t_state(observation)
        # Kept before normalization: relative-action chunks are offsets from the
        # raw state, so the reference frame has to be the un-normalized one.
        self._last_state_raw = raw.copy()
        if self.gr00t is not None:
            raw = self.gr00t.state_norm(raw)
        language = re.sub(r"[^\w\s]", "", self._task(observation).lower())
        return self._padded_state(raw), language

    def _gr00t_finish(self, req: Any) -> np.ndarray:
        self._maybe_pin_noise(req)
        chunk = self._send(req)
        if self.gr00t is not None:
            chunk = self.gr00t.unnormalize(chunk, self._last_state_raw)
        return chunk

    def _predict_gr00t_n17(self, observation: dict[str, Any]) -> np.ndarray:
        """N1.7: chat template, then every image-pad id expanded to 64 tokens."""
        frames = [
            gr00t_image_transform(img, GR00T_N17_TARGET_SIZE, GR00T_N17_SHORTEST_EDGE, GR00T_CROP_FRACTION)
            for img in self._gr00t_u8_frames(observation)
        ]
        state, language = self._gr00t_prepare(observation)

        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in frames],
                    {"type": "text", "text": language},
                ],
            }
        ]
        text = self._tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        ids = self._tok(text, add_special_tokens=False)["input_ids"]
        expanded: list[int] = []
        for tid in ids:
            if tid == GR00T_N17_IMG_PAD_ID:
                expanded.extend([GR00T_N17_IMG_PAD_ID] * GR00T_N17_N_TOK_PER_VIEW)
            else:
                expanded.append(tid)

        req = self._new_request()
        for frame in frames:
            self._add_image(
                req,
                np.ascontiguousarray(frame.astype(np.float32) / 255.0),
                vla_pb2.Image.F32_RGB_01,
            )
        req.lang_tokens.extend(int(t) for t in expanded)
        req.state.extend(float(x) for x in state)
        return self._gr00t_finish(req)

    def _predict_gr00t_n16(self, observation: dict[str, Any]) -> np.ndarray:
        """N1.6: the prompt spells out the image blocks, so nothing is expanded."""
        frames = [
            gr00t_n16_image_transform(img, self.image_size, GR00T_CROP_FRACTION)
            for img in self._gr00t_u8_frames(observation)
        ]
        state, language = self._gr00t_prepare(observation)

        def block(n: int) -> str:
            return f"<image {n}><img>" + "<IMG_CONTEXT>" * GR00T_N16_N_TOK_PER_VIEW + "</img>"

        content = language + "".join(block(i + 1) for i in range(len(frames)))
        text = self._tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = self._tok(text, add_special_tokens=False)["input_ids"]
        _assert_image_slots(ids, GR00T_N16_IMG_PAD_ID, len(frames) * GR00T_N16_N_TOK_PER_VIEW, "gr00t_n1_6")

        req = self._new_request()
        for frame in frames:
            self._add_image(
                req,
                np.ascontiguousarray(frame.astype(np.float32) / 255.0),
                vla_pb2.Image.F32_RGB_01,
            )
        req.lang_tokens.extend(int(t) for t in ids)
        req.state.extend(float(x) for x in state)
        return self._gr00t_finish(req)

    def _predict_gr00t_n15(self, observation: dict[str, Any]) -> np.ndarray:
        """N1.5: a hand-built prompt, 256 tokens per view, and **uint8** frames.

        The encoding differs from the other two on purpose - N1.5 is served raw
        bytes, not floats - and the instruction is embedded as ``str([task])``,
        brackets and quotes included, which is what the checkpoint saw.
        """
        import cv2

        frames = [
            np.ascontiguousarray(
                cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            )
            for img in self._gr00t_u8_frames(observation)
        ]
        state, _ = self._gr00t_prepare(observation)
        task = self._task(observation)

        images = "".join(
            f"<image {i + 1}><img>" + "<IMG_CONTEXT>" * GR00T_N15_N_TOK_PER_VIEW + "</img>"
            for i in range(len(frames))
        )
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + images + str([task]) + "<|im_end|>\n<|im_start|>assistant\n"
        )
        ids = self._tok(text, add_special_tokens=False)["input_ids"]
        _assert_image_slots(ids, GR00T_N15_IMG_CTX_ID, len(frames) * GR00T_N15_N_TOK_PER_VIEW, "gr00t_n1_5")

        req = self._new_request()
        for frame in frames:
            self._add_image(req, frame, vla_pb2.Image.RGB_U8)
        req.lang_tokens.extend(int(t) for t in ids)
        req.state.extend(float(x) for x in state)
        return self._gr00t_finish(req)


def _assert_image_slots(ids: list[int], token_id: int, expected: int, arch: str) -> None:
    """Fail loudly when the tokenizer swallowed the image placeholders.

    A tokenizer that does not know ``<IMG_CONTEXT>`` as a special token splits it
    into word pieces, and the prompt then reaches the server with no image slots at
    all. The server has no way to notice, so the count is checked here.
    """
    found = sum(1 for t in ids if t == token_id)
    if found != expected:
        raise ValueError(
            f"{arch}: expected {expected} image-placeholder tokens (id {token_id}) in the "
            f"tokenized prompt, got {found}. Check that <IMG_CONTEXT>, <img> and </img> "
            f"are special tokens in this tokenizer."
        )


def _as_numpy(value: Any) -> np.ndarray:
    """Accept a torch tensor or anything array-like without importing torch eagerly."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
