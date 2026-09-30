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
"""vla.cpp preprocessing, against the behaviour the engine's own client has.

Every assertion here is a place where a reimplementation that looks right changes
what the policy sees, so each one is written out rather than compared against the
module's own helpers.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lerobot.vla_cpp.preprocessing import (  # noqa: E402
    gr00t_image_transform,
    gr00t_n16_image_transform,
    minmax_norm,
    resize_with_pad,
)


def test_resize_with_pad_keeps_the_aspect_ratio():
    out = resize_with_pad(np.ones((3, 480, 640), dtype=np.float32), 224, 224)
    assert out.shape == (3, 224, 224)
    # 640x480 is 4:3, so fitting the width leaves 224 * 3/4 = 168 rows of content.
    content_rows = int(np.sum(out[0].max(axis=1) > 0))
    assert content_rows == 168


def test_resize_with_pad_pads_top_and_left_not_centre():
    """The pad is not centred, and that is the whole point of this test.

    Centring it moves every pixel of a non-square frame, which no downstream check
    would catch: the image is still an image and the policy still returns actions.
    """
    out = resize_with_pad(np.ones((3, 480, 640), dtype=np.float32), 224, 224)
    # 56 rows of padding, all of it above the content.
    assert out[:, :56, :].max() == 0.0
    assert out[:, 56:, :].min() > 0.0


def test_resize_with_pad_pads_left_for_a_tall_frame():
    out = resize_with_pad(np.ones((3, 640, 480), dtype=np.float32), 224, 224)
    assert out[:, :, :56].max() == 0.0
    assert out[:, :, 56:].min() > 0.0


def test_resize_with_pad_is_a_noop_at_the_target_size():
    img = np.random.default_rng(0).random((3, 224, 224)).astype(np.float32)
    np.testing.assert_allclose(resize_with_pad(img, 224, 224), img, rtol=0, atol=0)


def test_minmax_norm_maps_the_range_to_minus_one_one():
    lo, hi = np.array([0.0, -1.0]), np.array([10.0, 1.0])
    mask = np.array([True, True])
    np.testing.assert_allclose(minmax_norm(np.array([0.0, -1.0]), lo, hi, mask), [-1.0, -1.0])
    np.testing.assert_allclose(minmax_norm(np.array([10.0, 1.0]), lo, hi, mask), [1.0, 1.0])
    np.testing.assert_allclose(minmax_norm(np.array([5.0, 0.0]), lo, hi, mask), [0.0, 0.0])


def test_minmax_norm_sends_a_degenerate_column_to_zero_not_through():
    """A column with no range normalizes to 0, never to the value itself.

    Passing it through would hand the policy whatever raw units that joint reports,
    which it would read as an extreme of the normalized range.
    """
    out = minmax_norm(
        np.array([7.0, 42.0]),
        np.array([0.0, 42.0]),
        np.array([10.0, 42.0]),
        np.array([True, False]),
    )
    assert out[1] == 0.0
    np.testing.assert_allclose(out[0], 0.4)


@pytest.mark.parametrize("shape", [(480, 640, 3), (640, 480, 3), (256, 256, 3)])
def test_gr00t_transforms_return_the_target_square(shape):
    img = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    assert gr00t_image_transform(img, 256, 256, 0.95).shape == (256, 256, 3)
    assert gr00t_n16_image_transform(img, 224, 0.95).shape == (224, 224, 3)


def test_gr00t_n17_and_n16_transforms_differ():
    """N1.7 shrinks to a shortest edge before cropping; N1.6 does not.

    They therefore keep different fractions of the original frame and are not
    interchangeable, however similar the two call signatures look.
    """
    img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    a = gr00t_image_transform(img, 224, 224, 0.95)
    b = gr00t_n16_image_transform(img, 224, 0.95)
    assert not np.array_equal(a, b)


def test_gr00t_transform_centres_a_non_square_frame():
    """The square pad is centred, unlike resize_with_pad's. Both match the reference."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # fill the content so the pad is the only black region
    out = gr00t_n16_image_transform(img, 64, 1.0)
    column = out[:, 32, 0].astype(int)
    # Content sits in the middle band; the pad is above and below it, symmetrically.
    assert column[0] == 0 and column[-1] == 0
    assert column[32] == 255


def test_gr00t_transforms_are_deterministic():
    img = np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    assert np.array_equal(
        gr00t_image_transform(img, 256, 256, 0.95), gr00t_image_transform(img, 256, 256, 0.95)
    )
