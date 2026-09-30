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
"""Normalizers built from a checkpoint's statistics JSON.

The engine returns actions in the space the policy was trained in, so the client
owns both halves of the normalization: the state it sends and the actions it gets
back. This module builds those two functions and nothing else.

**A wrong normalizer is the failure mode to fear here.** It does not raise, it does
not look wrong in a log, and the parity checks the engine ships cannot see it -
they feed the reference the same statistics the engine got, so identity statistics
make both sides agree. What it does instead is move the arm to a pose nothing like
the training distribution. Every quantile choice below is therefore spelled out.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .archs import (
    GR00T_ACTION_HORIZON,
    GR00T_DEFAULT_EMBODIMENTS,
    GR00T_STATE_DIMS,
    GR00T_STATE_KEYS,
)
from .preprocessing import minmax_norm

Normalizer = Callable[[np.ndarray], np.ndarray]


def load_stats_blob(path: str | Path) -> dict[str, Any]:
    """Read a statistics JSON.

    Args:
        path (`str` or `Path`):
            File to read.

    Returns:
        `dict[str, Any]`: The parsed document.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"statistics JSON not found at {p}")
    return json.loads(p.read_text())


def resolve_embodiment(blob: dict[str, Any], arch: str, explicit: str | None) -> str:
    """Which top-level key of a statistics JSON to normalize with.

    Refuses to guess when there is more than one candidate: picking the wrong
    embodiment is exactly the silent failure this module exists to prevent.
    """
    if explicit is not None:
        if explicit not in blob:
            raise KeyError(f"embodiment {explicit!r} not in statistics; have {list(blob)}")
        return explicit
    for candidate in GR00T_DEFAULT_EMBODIMENTS.get(arch, ()):
        if candidate in blob:
            return candidate
    if len(blob) == 1:
        return next(iter(blob))
    raise ValueError(
        f"statistics JSON has top-level keys {list(blob)} and none is a known default "
        f"for {arch}; pass --embodiment explicitly."
    )


# ---------------------------------------------------------------------------
# pi0.5
# ---------------------------------------------------------------------------


def pi05_state_quantiles(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(q01, q99)`` for ``observation.state``, which pi0.5 digitizes into its prompt.

    Quantiles specifically: pi0.5's prompt carries the state as 256 discrete bins
    over [-1, 1], so a mean/std normalization would put most of the range outside
    the bins and the digitization would saturate.
    """
    blob = load_stats_blob(path)
    stats = blob.get("observation.state") or blob.get("state")
    if stats is None:
        raise KeyError(
            f"{path} has no 'observation.state' entry; pi05 needs a LeRobot "
            f"meta/stats.json with quantile statistics."
        )
    if "q01" not in stats or "q99" not in stats:
        raise ValueError(
            f"{path}::observation.state has no q01/q99. pi05 digitizes the state into "
            f"its prompt and needs quantiles, not mean/std."
        )
    q01 = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    q99 = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    return q01, q99


# ---------------------------------------------------------------------------
# GR00T, shared
# ---------------------------------------------------------------------------


def gr00t_modality_layout(group_stats: dict[str, Any]) -> tuple[list[str], list[int]]:
    """Ordered modality names and widths for one statistics group.

    Keeps the historical end-effector order when the checkpoint declares every one
    of those modalities; otherwise follows the order the checkpoint itself lists,
    which is the order GR00T's own processor concatenates them in. The order is the
    layout of the state vector, so it is not cosmetic.
    """
    if all(k in group_stats for k in GR00T_STATE_KEYS):
        keys = list(GR00T_STATE_KEYS)
    else:
        keys = list(group_stats.keys())
    dims = [
        int(
            np.asarray(group_stats[k]["q01" if "q01" in group_stats[k] else "min"], dtype=np.float32)
            .reshape(-1)
            .size
        )
        for k in keys
    ]
    return keys, dims


def gr00t_concat_field(group_stats: dict[str, Any], keys: list[str], field: str) -> np.ndarray:
    """Concatenate one statistics field across modalities, in the given order."""
    parts = []
    for k in keys:
        if k not in group_stats:
            raise KeyError(f"modality {k!r} not in statistics group; have {list(group_stats)}")
        if field not in group_stats[k]:
            raise KeyError(f"modality {k!r} lacks {field!r}")
        parts.append(np.asarray(group_stats[k][field], dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


class Gr00tNormalizers:
    """State normalizer, action un-normalizer and the state layout, for one checkpoint.

    ``state_keys``/``state_dims`` are what the caller must supply as ``state.<key>``
    observations; they come from the statistics JSON rather than being assumed.
    """

    def __init__(
        self,
        state_norm: Normalizer,
        action_unnorm: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
        state_keys: tuple[str, ...],
        state_dims: tuple[int, ...],
        description: str,
    ):
        """Hold one checkpoint's normalizers and the state layout they assume.

        Args:
            state_norm (`Callable`):
                Maps a raw concatenated state to the range the policy expects.
            action_unnorm (`Callable`):
                Maps a returned chunk back to the robot's units, given the raw
                observation state as a reference for relative actions.
            state_keys (`tuple[str, ...]`):
                Modality names, in the order they concatenate.
            state_dims (`tuple[int, ...]`):
                Width of each modality, aligned with `state_keys`.
            description (`str`):
                One line naming the scheme, for logging.
        """
        self.state_norm = state_norm
        self._action_unnorm = action_unnorm
        self.state_keys = state_keys
        self.state_dims = state_dims
        self.description = description

    def unnormalize(self, chunk: np.ndarray, reference_state: np.ndarray | None = None) -> np.ndarray:
        """Map a returned action chunk back into the robot's units.

        Args:
            chunk (`np.ndarray`):
                `(chunk_size, action_dim)` as the server returned it.
            reference_state (`np.ndarray`, *optional*):
                The raw, un-normalized state of the observation this chunk answers.
                Required only for a checkpoint trained with relative actions, where
                every step of the chunk is an offset from it.

        Returns:
            `np.ndarray`: The chunk in the robot's units.

        Raises:
            RuntimeError: If the checkpoint uses relative actions and no
                `reference_state` was given.
        """
        return self._action_unnorm(chunk, reference_state)


def _gr00t_n17(blob: dict[str, Any], key: str, rel_stats: dict[str, Any]) -> Gr00tNormalizers:
    action_stats = blob[key]["action"]
    modalities, mod_dims = gr00t_modality_layout(action_stats)
    q01 = gr00t_concat_field(action_stats, modalities, "q01")
    q99 = gr00t_concat_field(action_stats, modalities, "q99")
    act_dim = int(q01.size)

    state_stats = blob[key]["state"]
    state_keys, state_dims = gr00t_modality_layout(state_stats)
    state_cols: dict[str, slice] = {}
    off = 0
    for m, dim in zip(state_keys, state_dims, strict=True):
        state_cols[m] = slice(off, off + dim)
        off += dim

    rel_names = [m for m in modalities if m in rel_stats]

    if rel_names:
        # A checkpoint trained with relative actions predicts, for these modalities,
        # the offset from the observed state rather than an absolute target - and
        # those offsets carry their own per-chunk-step statistics.
        horizon = min(len(rel_stats[m]["min"]) for m in rel_names)
        q01_t = np.tile(q01, (horizon, 1)).astype(np.float32)
        q99_t = np.tile(q99, (horizon, 1)).astype(np.float32)
        rel_cols: list[tuple[slice, slice]] = []
        off = 0
        for m, dim in zip(modalities, mod_dims, strict=True):
            if m in rel_stats:
                r = rel_stats[m]
                # min/max, never q01/q99, even when the checkpoint sets
                # use_percentiles. GR00T swaps the whole norm_params entry for a
                # relative key holding the raw relative-stats dict, which bypasses
                # the branch that would substitute percentiles. Using q01/q99 here
                # scales every offset down by roughly 2x and the arm never reaches
                # the pose the policy is steering to.
                a = np.asarray(r["min"], dtype=np.float32)[:horizon]
                b = np.asarray(r["max"], dtype=np.float32)[:horizon]
                if a.shape[1] != dim:
                    raise ValueError(
                        f"relative statistics for {m!r} are {a.shape[1]} wide, statistics say {dim}"
                    )
                sc = state_cols.get(m)
                if sc is None or sc.stop - sc.start != dim:
                    raise ValueError(
                        f"relative modality {m!r} needs a {dim}-wide state.{m}; state "
                        f"statistics have {list(zip(state_keys, state_dims, strict=True))}"
                    )
                q01_t[:, off : off + dim] = a
                q99_t[:, off : off + dim] = b
                rel_cols.append((slice(off, off + dim), sc))
            off += dim
        rng_t = (q99_t - q01_t).astype(np.float32)

        def _unnorm(chunk: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
            n = min(len(chunk), horizon)
            norm = np.clip(chunk[:n, :act_dim].astype(np.float32), -1.0, 1.0)
            raw = (norm + 1.0) * 0.5 * rng_t[:n] + q01_t[:n]
            if reference is None:
                raise RuntimeError(
                    "relative actions need the un-normalized observation state as a "
                    "reference; none was recorded for this request"
                )
            ref = np.asarray(reference, dtype=np.float32)
            # Every step of the chunk is an offset from the one reference state.
            for a_cols, s_cols in rel_cols:
                raw[:, a_cols] += ref[s_cols]
            return raw.astype(np.float32)

    else:
        rng = (q99 - q01).astype(np.float32)

        def _unnorm(chunk: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
            norm = np.clip(chunk[..., :act_dim].astype(np.float32), -1.0, 1.0)
            return ((norm + 1.0) * 0.5 * rng[None, :] + q01[None, :]).astype(np.float32)

    s_q01 = gr00t_concat_field(state_stats, state_keys, "q01")
    s_q99 = gr00t_concat_field(state_stats, state_keys, "q99")
    s_mask = ~np.isclose(s_q99, s_q01)

    def _state_norm(state: np.ndarray) -> np.ndarray:
        return np.clip(minmax_norm(state, s_q01, s_q99, s_mask), -1.0, 1.0)

    return Gr00tNormalizers(
        _state_norm,
        _unnorm,
        tuple(state_keys),
        tuple(state_dims),
        f"q01/q99 + clip via {key}, action {act_dim}-D, relative={rel_names or 'none'}",
    )


def _gr00t_n16(blob: dict[str, Any], key: str) -> Gr00tNormalizers:
    modalities = list(GR00T_STATE_KEYS)
    action_stats = blob[key]["action"]
    a_min = np.array([action_stats[m]["min"][0] for m in modalities], dtype=np.float32)
    a_max = np.array([action_stats[m]["max"][0] for m in modalities], dtype=np.float32)
    a_rng = (a_max - a_min).astype(np.float32)

    def _unnorm(chunk: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
        # 7 columns, not the full width: N1.6's head emits a padded chunk and only
        # the first seven are the end-effector action.
        norm = np.clip(chunk[..., :7].astype(np.float32), -1.0, 1.0)
        raw = (norm + 1.0) * 0.5 * a_rng[None, :] + a_min[None, :]
        return raw[:GR00T_ACTION_HORIZON].astype(np.float32)

    state_stats = blob[key]["state"]
    mins, maxs = [], []
    for m, dim in zip(GR00T_STATE_KEYS, GR00T_STATE_DIMS, strict=True):
        if m not in state_stats:
            raise KeyError(f"state modality {m!r} not in statistics::{key}.state")
        mn = np.asarray(state_stats[m]["min"], dtype=np.float32)
        mx = np.asarray(state_stats[m]["max"], dtype=np.float32)
        if mn.size != dim or mx.size != dim:
            raise ValueError(f"state.{m}: statistics width {mn.size}/{mx.size} != expected {dim}")
        mins.append(mn)
        maxs.append(mx)
    s_min = np.concatenate(mins)
    s_max = np.concatenate(maxs)
    s_mask = ~np.isclose(s_max, s_min)

    def _state_norm(state: np.ndarray) -> np.ndarray:
        return np.clip(minmax_norm(state, s_min, s_max, s_mask), -1.0, 1.0)

    return Gr00tNormalizers(
        _state_norm, _unnorm, GR00T_STATE_KEYS, GR00T_STATE_DIMS, f"min/max + clip via {key}"
    )


def _gr00t_n15(blob: dict[str, Any], key: str) -> Gr00tNormalizers:
    a_min = np.asarray(blob[key]["action"]["min"], dtype=np.float32)
    a_max = np.asarray(blob[key]["action"]["max"], dtype=np.float32)
    a_rng = (a_max - a_min).astype(np.float32)

    def _unnorm(chunk: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
        # No clip, unlike N1.6 and N1.7. The reference does not clamp here and a
        # clamp would quietly bound actions the policy meant to push further.
        norm = chunk[..., : a_min.size].astype(np.float32)
        raw = (norm + 1.0) * 0.5 * a_rng[None, :] + a_min[None, :]
        return raw[:GR00T_ACTION_HORIZON].astype(np.float32)

    s_min = np.asarray(blob[key]["state"]["min"], dtype=np.float32)
    s_max = np.asarray(blob[key]["state"]["max"], dtype=np.float32)
    s_mask = s_min != s_max

    def _state_norm(state: np.ndarray) -> np.ndarray:
        # Also unclipped, matching the reference.
        return minmax_norm(state, s_min, s_max, s_mask)

    return Gr00tNormalizers(
        _state_norm,
        _unnorm,
        GR00T_STATE_KEYS,
        GR00T_STATE_DIMS,
        f"flat min/max, no clip, via {key}",
    )


def build_gr00t_normalizers(
    arch: str,
    stats_json: str | Path,
    *,
    embodiment: str | None = None,
    rel_stats_json: str | Path | None = None,
) -> Gr00tNormalizers:
    """Normalizers for one GR00T checkpoint, from its statistics JSON.

    The three versions normalize differently - N1.7 on q01/q99 with a clip, N1.6 on
    per-modality min/max with a clip, N1.5 on a flat min/max with none - so they
    are built separately rather than parameterized.
    """
    blob = load_stats_blob(stats_json)
    key = resolve_embodiment(blob, arch, embodiment)
    if arch == "gr00t_n1_7":
        rel: dict[str, Any] = {}
        if rel_stats_json is not None:
            rel = load_stats_blob(rel_stats_json)
        return _gr00t_n17(blob, key, rel)
    if arch == "gr00t_n1_6":
        return _gr00t_n16(blob, key)
    if arch == "gr00t_n1_5":
        return _gr00t_n15(blob, key)
    raise ValueError(f"{arch} has no GR00T normalizers")
