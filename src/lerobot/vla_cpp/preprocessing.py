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
"""Image and state preprocessing for the vla.cpp wire format.

These are transcriptions of what the engine's own eval client does, not general
utilities. A policy is only as good as the pixels it is handed, and every function
here is a place where a reasonable-looking reimplementation silently changes what
the model sees: the pad goes on one side rather than being centred, a resize uses
a different interpolation, a normalizer scales a column it should have left alone.
Each one therefore says which behaviour is load-bearing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def resize_with_pad(img_chw: np.ndarray, target_h: int, target_w: int, pad_value: float = 0.0) -> np.ndarray:
    """Fit ``(3, H, W)`` into ``(3, target_h, target_w)``, preserving aspect ratio.

    **The padding is not centred.** It goes on the top and the left, which is what
    the engine's client does and therefore what the checkpoints were evaluated
    with. Centring it moves every pixel of a non-square frame and the policy has
    no way to tell you it happened.

    Bilinear through ``F.interpolate`` with ``align_corners=False``, again to match
    rather than because it is the best choice available.
    """
    t = torch.from_numpy(np.ascontiguousarray(img_chw)).unsqueeze(0)
    cur_h, cur_w = t.shape[2:]
    ratio = max(cur_w / target_w, cur_h / target_h)
    rh = int(cur_h / ratio)
    rw = int(cur_w / ratio)
    t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
    pad_h = max(0, target_h - rh)
    pad_w = max(0, target_w - rw)
    t = F.pad(t, (pad_w, 0, pad_h, 0), value=pad_value)
    return t.squeeze(0).numpy()


def minmax_norm(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map ``x`` to [-1, 1] over ``mask``, leaving masked-out columns at **zero**.

    The zero is deliberate and is not the same as passing the column through: a
    degenerate column (``lo == hi``) has no range to normalize into, and the
    reference sends 0 rather than a value the model would read as an extreme.
    """
    out = np.zeros_like(x, dtype=np.float32)
    out[..., mask] = 2.0 * (x[..., mask] - lo[mask]) / (hi[mask] - lo[mask]) - 1.0
    return out


def _pad_to_square(img: np.ndarray) -> np.ndarray:
    """Centre a non-square frame on a zero square. GR00T's transforms start here."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=img.dtype)
    out[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = img
    return out


def gr00t_image_transform(
    img_u8_hwc: np.ndarray, target_size: int, shortest_edge: int, crop_fraction: float
) -> np.ndarray:
    """GR00T N1.7's eval-time transform: square-pad, shrink, centre-crop, resize.

    The order is the whole content of this function. Cropping before the shrink,
    or resizing straight to ``target_size``, changes the field of view the policy
    was trained on while producing an image that looks entirely reasonable.

    ``INTER_AREA`` throughout, as in the reference.
    """
    import cv2

    img = _pad_to_square(img_u8_hwc)
    h, w = img.shape[:2]

    short = min(h, w)
    if short > shortest_edge:
        scale = shortest_edge / short
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    nh, nw = int(round(h * crop_fraction)), int(round(w * crop_fraction))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    img = img[y0 : y0 + nh, x0 : x0 + nw]
    h, w = img.shape[:2]

    short = min(h, w)
    if short != shortest_edge:
        scale = shortest_edge / short
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img, dtype=np.uint8)


def gr00t_n16_image_transform(img_u8_hwc: np.ndarray, target_size: int, crop_fraction: float) -> np.ndarray:
    """GR00T N1.6's transform: square-pad, centre-crop, resize.

    Shorter than N1.7's by one step - there is no shrink to a shortest edge before
    the crop - so the two crop different fractions of the original frame. They are
    not interchangeable.
    """
    import cv2

    img = _pad_to_square(img_u8_hwc)
    h, w = img.shape[:2]
    nh, nw = int(round(h * crop_fraction)), int(round(w * crop_fraction))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    img = img[y0 : y0 + nh, x0 : x0 + nw]
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img, dtype=np.uint8)
