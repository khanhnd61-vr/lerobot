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
"""Per-architecture constants for the vla.cpp wire format.

The engine reads every shape out of the GGUF at load time, so these are not the
model's dimensions - they are the *client's* half of the contract: which tokenizer
produced the ids the checkpoint was trained against, what resolution its images
were, and how wide the state vector it expects is. Getting one wrong loads, runs
and returns plausible actions.
"""

from __future__ import annotations

from typing import Any

#: Architectures this client can preprocess for. ``passthrough`` is the escape
#: hatch: no arch-specific handling at all, for a server that does its own.
ARCH_PRESETS: dict[str, dict[str, Any]] = {
    "smolvla": {
        "image_size": 512,
        "tokenizer": "HuggingFaceTB/SmolVLM2-500M-Instruct",
        "max_state_dim": 32,
    },
    "pi0": {
        "image_size": 224,
        "tokenizer": "google/paligemma-3b-pt-224",
        "max_state_dim": 32,
    },
    "pi05": {
        "image_size": 224,
        "tokenizer": "google/paligemma-3b-pt-224",
        "max_state_dim": 32,
        # pi05 spells the state into the prompt, so its prompt is far longer than
        # the others' and truncating at 48 would cut the state off mid-number.
        "max_length": 200,
    },
    "gr00t_n1_5": {
        "image_size": 224,
        "tokenizer": "lerobot/eagle2hg-processor-groot-n1p5",
        "max_state_dim": 64,
        "trust_remote_code": True,
    },
    "gr00t_n1_6": {
        "image_size": 224,
        # No default: N1.6 needs a local processor directory, because its chat
        # template is read out of chat_template.json next to the tokenizer.
        "tokenizer": None,
        "max_state_dim": 128,
        "trust_remote_code": True,
    },
    "gr00t_n1_7": {
        "image_size": 256,
        "tokenizer": "nvidia/Cosmos-Reason2-2B",
        "max_state_dim": 132,
    },
    "passthrough": {
        # Whatever the frames already are. resize_with_pad is skipped entirely,
        # so the server receives the robot's native resolution.
        "image_size": None,
        "tokenizer": None,
        "max_state_dim": 32,
    },
}

#: Archs whose flow-matching noise the client can pin for reproducibility.
FIXED_NOISE_ARCHS = frozenset({"smolvla", "pi0", "pi05", "gr00t_n1_5", "gr00t_n1_6", "gr00t_n1_7"})

#: Archs that need a statistics JSON before they can run at all.
STATS_REQUIRED_ARCHS = frozenset({"pi05"})

#: Archs whose observations are modality-keyed (``state.x``, ``video.image``)
#: rather than lerobot's flat ``observation.state`` + ``observation.images.*``.
MODALITY_KEYED_ARCHS = frozenset({"gr00t_n1_5", "gr00t_n1_6", "gr00t_n1_7"})

# ---------------------------------------------------------------------------
# GR00T
# ---------------------------------------------------------------------------

#: Fallback state layout: the end-effector embodiments the GR00T paths were first
#: written against. A checkpoint trained on a joint space names its modalities
#: differently ("single_arm", "gripper"), so the real layout is read out of the
#: statistics JSON and this is used only when every one of these is present.
GR00T_STATE_KEYS: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
GR00T_STATE_DIMS: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 2)

GR00T_VIDEO_KEYS: tuple[str, ...] = ("video.image", "video.wrist_image")

#: N1.5 and N1.6 both emit a padded chunk and keep only the first 16 steps.
GR00T_ACTION_HORIZON = 16

# N1.7: Cosmos-Reason2 image-pad expansion.
GR00T_N17_IMG_PAD_ID = 151655
GR00T_N17_N_TOK_PER_VIEW = 64
GR00T_N17_TARGET_SIZE = 256
GR00T_N17_SHORTEST_EDGE = 256
GR00T_CROP_FRACTION = 0.95

# N1.6: the prompt carries explicit <IMG_CONTEXT> runs, so nothing is expanded.
GR00T_N16_IMG_PAD_ID = 151669
GR00T_N16_N_TOK_PER_VIEW = 64

# N1.5: same idea, four times the tokens per view.
GR00T_N15_IMG_CTX_ID = 151669
GR00T_N15_N_TOK_PER_VIEW = 256

#: Embodiment keys each GR00T version looks for in a statistics JSON, in order,
#: when ``--embodiment`` is not given.
GR00T_DEFAULT_EMBODIMENTS: dict[str, tuple[str, ...]] = {
    "gr00t_n1_5": ("new_embodiment", "libero_sim", "libero_panda"),
    "gr00t_n1_6": ("libero_panda", "libero_sim"),
    "gr00t_n1_7": ("libero_sim",),
}


def resolve_preset(arch: str) -> dict[str, Any]:
    """The preset for ``arch``, or a ValueError naming the ones that exist."""
    if arch not in ARCH_PRESETS:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCH_PRESETS)}")
    return ARCH_PRESETS[arch]
