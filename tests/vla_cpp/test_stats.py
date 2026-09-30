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
"""Normalizers built from a statistics JSON.

A wrong normalizer does not raise and does not look wrong in a log - it moves the
arm to a pose nothing like the training distribution. These tests are the only place
that failure is visible, so they check the arithmetic rather than that a function
returns an array.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lerobot.vla_cpp.stats import (  # noqa: E402
    build_gr00t_normalizers,
    pi05_state_quantiles,
    resolve_embodiment,
)

EEF = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _quantile_group(lo: float, hi: float) -> dict:
    """An end-effector group with q01/q99, gripper two-wide as GR00T declares it."""
    group = {k: {"q01": [lo], "q99": [hi]} for k in EEF[:-1]}
    group["gripper"] = {"q01": [lo, lo], "q99": [hi, hi]}
    return group


def _minmax_group(lo: float, hi: float) -> dict:
    group = {k: {"min": [lo], "max": [hi]} for k in EEF[:-1]}
    group["gripper"] = {"min": [lo, lo], "max": [hi, hi]}
    return group


def _write(tmp_path, blob, name="stats.json"):
    path = tmp_path / name
    path.write_text(json.dumps(blob))
    return path


# ---------------------------------------------------------------------------
# Embodiment resolution
# ---------------------------------------------------------------------------


def test_resolve_embodiment_takes_the_explicit_key():
    blob = {"a": {}, "b": {}}
    assert resolve_embodiment(blob, "gr00t_n1_7", "b") == "b"


def test_resolve_embodiment_rejects_an_unknown_key():
    with pytest.raises(KeyError, match="not in statistics"):
        resolve_embodiment({"a": {}}, "gr00t_n1_7", "zzz")


def test_resolve_embodiment_accepts_a_lone_key():
    assert resolve_embodiment({"only": {}}, "gr00t_n1_7", None) == "only"


def test_resolve_embodiment_prefers_the_known_default():
    assert resolve_embodiment({"other": {}, "libero_sim": {}}, "gr00t_n1_7", None) == "libero_sim"


def test_resolve_embodiment_refuses_to_guess_between_candidates():
    """Picking the wrong embodiment is the silent failure this module exists to stop."""
    with pytest.raises(ValueError, match="pass --embodiment"):
        resolve_embodiment({"one": {}, "two": {}}, "gr00t_n1_7", None)


# ---------------------------------------------------------------------------
# pi0.5
# ---------------------------------------------------------------------------


def test_pi05_reads_state_quantiles(tmp_path):
    path = _write(tmp_path, {"observation.state": {"q01": [0.0] * 6, "q99": [1.0] * 6}})
    q01, q99 = pi05_state_quantiles(path)
    assert q01.shape == (6,) and q99.shape == (6,)


def test_pi05_refuses_mean_std_statistics(tmp_path):
    """pi0.5 digitizes the state into 256 bins over [-1, 1], so it needs quantiles."""
    path = _write(tmp_path, {"observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6}})
    with pytest.raises(ValueError, match="q01/q99"):
        pi05_state_quantiles(path)


def test_pi05_reports_a_missing_state_entry(tmp_path):
    with pytest.raises(KeyError, match="observation.state"):
        pi05_state_quantiles(_write(tmp_path, {"action": {}}))


# ---------------------------------------------------------------------------
# GR00T N1.7 - q01/q99 with a clip
# ---------------------------------------------------------------------------


def test_n17_unnormalizes_through_q01_q99(tmp_path):
    path = _write(
        tmp_path,
        {"libero_sim": {"action": _quantile_group(-2.0, 2.0), "state": _quantile_group(-1.0, 1.0)}},
    )
    norm = build_gr00t_normalizers("gr00t_n1_7", path)
    # -1 -> q01, 0 -> midpoint, +1 -> q99.
    chunk = np.array([[-1.0] * 8, [0.0] * 8, [1.0] * 8], dtype=np.float32)
    out = norm.unnormalize(chunk)
    np.testing.assert_allclose(out[0], [-2.0] * 8)
    np.testing.assert_allclose(out[1], [0.0] * 8, atol=1e-6)
    np.testing.assert_allclose(out[2], [2.0] * 8)


def test_n17_clips_before_unnormalizing(tmp_path):
    """Out-of-range predictions clamp to the quantiles rather than extrapolating."""
    path = _write(
        tmp_path,
        {"libero_sim": {"action": _quantile_group(-2.0, 2.0), "state": _quantile_group(-1.0, 1.0)}},
    )
    norm = build_gr00t_normalizers("gr00t_n1_7", path)
    out = norm.unnormalize(np.full((1, 8), 5.0, dtype=np.float32))
    np.testing.assert_allclose(out[0], [2.0] * 8)


def test_n17_state_layout_comes_from_the_statistics(tmp_path):
    """The layout is read, not assumed: a joint-space checkpoint names its own."""
    joint = {"single_arm": {"q01": [0.0] * 6, "q99": [1.0] * 6}, "gripper": {"q01": [0.0], "q99": [1.0]}}
    path = _write(tmp_path, {"libero_sim": {"action": joint, "state": joint}})
    norm = build_gr00t_normalizers("gr00t_n1_7", path)
    assert norm.state_keys == ("single_arm", "gripper")
    assert norm.state_dims == (6, 1)


def test_n17_relative_actions_add_the_reference_state(tmp_path):
    """A relative chunk is an offset from the observed state, one reference for all steps."""
    path = _write(
        tmp_path,
        {"libero_sim": {"action": _quantile_group(-2.0, 2.0), "state": _quantile_group(-1.0, 1.0)}},
    )
    # Two chunk steps of relative stats for x only; range [0, 1] so +1 -> +1.0 offset.
    rel = _write(tmp_path, {"x": {"min": [[0.0], [0.0]], "max": [[1.0], [1.0]]}}, "rel.json")
    norm = build_gr00t_normalizers("gr00t_n1_7", path, rel_stats_json=rel)

    reference = np.array([10.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    out = norm.unnormalize(np.ones((2, 8), dtype=np.float32), reference)
    # x: normalized +1 -> min/max range gives 1.0, plus the reference 10.0.
    np.testing.assert_allclose(out[:, 0], [11.0, 11.0])
    # y is absolute, so it stays on the action quantiles and ignores the reference.
    np.testing.assert_allclose(out[:, 1], [2.0, 2.0])


def test_n17_relative_actions_use_min_max_not_the_quantiles(tmp_path):
    """The subtlety that silently halves every offset if got wrong.

    GR00T substitutes the raw relative-stats dict for the whole norm_params entry,
    which bypasses the branch that would otherwise use q01/q99. Scaling a relative
    column by the action quantiles instead makes the arm undershoot every target.
    """
    path = _write(
        tmp_path,
        {"libero_sim": {"action": _quantile_group(-2.0, 2.0), "state": _quantile_group(-1.0, 1.0)}},
    )
    rel = _write(tmp_path, {"x": {"min": [[0.0]], "max": [[1.0]]}}, "rel.json")
    norm = build_gr00t_normalizers("gr00t_n1_7", path, rel_stats_json=rel)

    out = norm.unnormalize(np.ones((1, 8), dtype=np.float32), np.zeros(8, dtype=np.float32))
    # min/max [0, 1] -> 1.0. Had q01/q99 [-2, 2] been used it would be 2.0.
    np.testing.assert_allclose(out[0, 0], 1.0)


def test_n17_relative_actions_need_a_reference(tmp_path):
    path = _write(
        tmp_path,
        {"libero_sim": {"action": _quantile_group(-2.0, 2.0), "state": _quantile_group(-1.0, 1.0)}},
    )
    rel = _write(tmp_path, {"x": {"min": [[0.0]], "max": [[1.0]]}}, "rel.json")
    norm = build_gr00t_normalizers("gr00t_n1_7", path, rel_stats_json=rel)
    with pytest.raises(RuntimeError, match="reference"):
        norm.unnormalize(np.ones((1, 8), dtype=np.float32), None)


# ---------------------------------------------------------------------------
# GR00T N1.6 and N1.5
# ---------------------------------------------------------------------------


def test_n16_unnormalizes_seven_columns_through_min_max(tmp_path):
    """N1.6's head emits a padded chunk; only the first seven columns are the action."""
    path = _write(
        tmp_path,
        {"libero_panda": {"action": _minmax_group(-3.0, 3.0), "state": _minmax_group(-1.0, 1.0)}},
    )
    norm = build_gr00t_normalizers("gr00t_n1_6", path)
    out = norm.unnormalize(np.ones((4, 32), dtype=np.float32))
    assert out.shape == (4, 7)
    np.testing.assert_allclose(out[0], [3.0] * 7)


def test_n16_truncates_to_its_action_horizon(tmp_path):
    path = _write(
        tmp_path,
        {"libero_panda": {"action": _minmax_group(-1.0, 1.0), "state": _minmax_group(-1.0, 1.0)}},
    )
    norm = build_gr00t_normalizers("gr00t_n1_6", path)
    assert norm.unnormalize(np.zeros((50, 32), dtype=np.float32)).shape[0] == 16


def test_n15_does_not_clip(tmp_path):
    """N1.5's reference applies no clamp, and a clamp would bound actions it meant."""
    path = _write(
        tmp_path,
        {
            "new_embodiment": {
                "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
                "state": {"min": [-1.0] * 8, "max": [1.0] * 8},
            }
        },
    )
    norm = build_gr00t_normalizers("gr00t_n1_5", path)
    out = norm.unnormalize(np.full((1, 7), 3.0, dtype=np.float32))
    # (3 + 1) * 0.5 * 2 + (-1) = 3.0 -- past the range, because nothing clips it.
    np.testing.assert_allclose(out[0], [3.0] * 7)


def test_n16_clips_where_n15_does_not(tmp_path):
    """The two versions differ here, so one normalizer cannot serve both."""
    n16 = _write(
        tmp_path,
        {"libero_panda": {"action": _minmax_group(-1.0, 1.0), "state": _minmax_group(-1.0, 1.0)}},
        "n16.json",
    )
    out = build_gr00t_normalizers("gr00t_n1_6", n16).unnormalize(np.full((1, 32), 3.0, dtype=np.float32))
    np.testing.assert_allclose(out[0], [1.0] * 7)


def test_state_normalizer_maps_into_minus_one_one(tmp_path):
    path = _write(
        tmp_path,
        {"libero_sim": {"action": _quantile_group(-1.0, 1.0), "state": _quantile_group(0.0, 10.0)}},
    )
    norm = build_gr00t_normalizers("gr00t_n1_7", path)
    np.testing.assert_allclose(norm.state_norm(np.full(8, 5.0, dtype=np.float32)), [0.0] * 8, atol=1e-6)
    np.testing.assert_allclose(norm.state_norm(np.zeros(8, dtype=np.float32)), [-1.0] * 8)


def test_missing_stats_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        build_gr00t_normalizers("gr00t_n1_7", tmp_path / "nope.json")
