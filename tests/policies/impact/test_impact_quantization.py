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
"""IMPACT's W8A8 kernel, pinned against an independent reference.

``quantization.py`` transcribes one specific deployment kernel, so "it quantizes" is not
the property that matters - "it quantizes *the same way*" is. Every test here therefore
compares against arithmetic spelled out again in the test rather than against the module's
own helpers, so that a change to the quantizer has to be argued for rather than absorbed.

These tests need only torch: the quantizer has no text tower and no Hub fetch in it.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402, N812
from torch import nn  # noqa: E402
from torchvision.ops.misc import FrozenBatchNorm2d  # noqa: E402

from lerobot.policies.impact.quant_transformer import quant_mha  # noqa: E402
from lerobot.policies.impact.quantization import (  # noqa: E402
    INT8_ALL,
    INT8_DEC,
    QMAX,
    Int8Runtime,
    fold_frozen_bn,
    quantizable,
    quantize_conv2d,
    quantize_linear,
)

# ---------------------------------------------------------------------------
# A reference quantizer, written out again rather than imported.
# ---------------------------------------------------------------------------


def ref_qdq(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """clamp(rne(x * (1/s)), -127, 127) * s -- the kernel's reciprocal-multiply order."""
    return torch.clamp(torch.round(x * torch.reciprocal(scale)), -QMAX, QMAX) * scale


def ref_row_scale(w: torch.Tensor) -> torch.Tensor:
    amax = w.abs().amax(dim=1, keepdim=True)
    return torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))


def ref_token_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=-1, keepdim=True)
    return torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))


# ---------------------------------------------------------------------------
# The scheme
# ---------------------------------------------------------------------------


def test_linear_matches_the_reference_scheme():
    """Per-row weight scales, per-token activation scales, and nothing else."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 64)
    w = torch.randn(32, 64)
    b = torch.randn(32)

    got = quantize_linear(x, w, b)
    want = F.linear(ref_qdq(x, ref_token_scale(x)), ref_qdq(w, ref_row_scale(w)), b)

    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_weight_scales_are_per_row_not_per_tensor():
    """One outlier row must not cost every other row its resolution.

    A per-tensor weight scale is the single most natural simplification of this scheme and
    it is the one that makes int8 not work, so it gets a test that fails loudly.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 64) * 0.01
    w[0] *= 1000.0  # one row with a vastly larger range
    x = torch.randn(8, 64)

    per_row = quantize_linear(x, w, None)
    per_tensor_w = ref_qdq(w, w.abs().amax() / QMAX * torch.ones(w.shape[0], 1))
    per_tensor = F.linear(ref_qdq(x, ref_token_scale(x)), per_tensor_w, None)

    # The quiet rows are the ones a shared scale destroys, so compare those.
    quiet = slice(1, None)
    exact = F.linear(x, w, None)
    assert (per_row[:, quiet] - exact[:, quiet]).abs().mean() < (
        per_tensor[:, quiet] - exact[:, quiet]
    ).abs().mean()


def test_activation_scales_are_per_token():
    """A loud token must not change a quiet token's quantization.

    Concretely: scaling row 0 of the input up by 1000 must leave row 1's output alone. With
    a per-tensor activation scale it would not.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    x = torch.randn(4, 64)

    base = quantize_linear(x, w, None)
    loud = x.clone()
    loud[0] *= 1000.0
    after = quantize_linear(loud, w, None)

    torch.testing.assert_close(after[1:], base[1:], rtol=0, atol=0)
    assert not torch.allclose(after[0], base[0])


def test_conv_activation_scale_is_per_image_not_per_batch():
    """The engine sees one frame per call, so a batch-pooled scale models nothing.

    Brightening image 0 must leave image 1's feature map untouched.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 3, 3, 3)
    x = torch.rand(2, 3, 16, 16)

    base = quantize_conv2d(x, w, None, padding=1)
    loud = x.clone()
    loud[0] *= 100.0
    after = quantize_conv2d(loud, w, None, padding=1)

    torch.testing.assert_close(after[1], base[1], rtol=0, atol=0)
    assert not torch.allclose(after[0], base[0])


def test_rounding_is_round_to_nearest_even():
    """`trunc(x + copysign(0.5, x))` and RNE disagree at exact .5, and the kernel uses RNE.

    A weight row whose absmax is 127 makes its scale exactly 1.0, so the values below land
    on exact half-integers and the two rules give different integers: RNE sends 0.5 to 0
    and 1.5 to 2, the truncating rule sends them to 1 and 2.
    """
    w = torch.zeros(16, 6)  # 16 rows so the packer gate lets it through
    w[0] = torch.tensor([127.0, 0.5, 1.5, 2.5, -0.5, -1.5])
    x = torch.eye(6) * 127.0  # absmax 127 on every token too, so s_a is 1.0

    # Row j of x is 127 * e_j, so out[j, 0] reads back 127 * q(w[0, j]).
    got = quantize_linear(x, w, None)[:, 0] / 127.0
    torch.testing.assert_close(got, torch.tensor([127.0, 0.0, 2.0, 2.0, -0.0, -2.0]))


def test_reciprocal_multiply_not_divide():
    """``x * (1/s)`` and ``x / s`` differ by an ulp, which flips integers at .5 boundaries.

    Rather than construct that coincidence by hand, assert the module agrees with the
    reciprocal form and, where the two forms differ at all, disagrees with the divide form.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 512)
    x = torch.randn(64, 512)
    s_w, s_a = ref_row_scale(w), ref_token_scale(x)

    recip = F.linear(ref_qdq(x, s_a), ref_qdq(w, s_w), None)
    torch.testing.assert_close(quantize_linear(x, w, None), recip, rtol=0, atol=0)

    divide_w = torch.clamp(torch.round(w / s_w), -QMAX, QMAX) * s_w
    divide_x = torch.clamp(torch.round(x / s_a), -QMAX, QMAX) * s_a
    divide = F.linear(divide_x, divide_w, None)
    if not torch.equal(divide_w, ref_qdq(w, s_w)):
        assert not torch.equal(quantize_linear(x, w, None), divide)


def test_all_zero_row_uses_scale_one_and_stays_finite():
    w = torch.randn(16, 32)
    w[3] = 0.0
    out = quantize_linear(torch.randn(4, 32), w, None)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 3], torch.zeros(4))


def test_clamp_is_symmetric_at_127():
    """-128 is representable in int8, but the kernel clamps to [-127, 127] and never emits it.

    The most-negative entry of a row sets that row's absmax, so it quantizes to exactly
    -127 and dequantizes back to itself. Were the floor -128, it would come back as
    -8 * 128/127 instead.
    """
    w = torch.zeros(16, 4)
    w[0] = torch.tensor([-8.0, 2.0, 0.0, 0.0])  # s_w[0] = 8/127
    x = torch.zeros(1, 4)
    x[0, 0] = 1.0  # s_a = 1/127, so x_q = 127 and the epilogue is a no-op

    out = quantize_linear(x, w, None)
    torch.testing.assert_close(out[0, 0], torch.tensor(-8.0))


# ---------------------------------------------------------------------------
# The packer gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_out,expected", [(16, True), (32, True), (6, False), (17, False)])
def test_quantizable_mirrors_the_packer(n_out: int, expected: bool):
    assert quantizable(n_out) is expected


def test_narrow_layer_stays_fp32():
    """The action head is [6, 512]: the engine leaves it alone, so the simulator must too.

    A simulator that quantizes a layer the engine skips disagrees with it for a reason no
    tolerance will explain.
    """
    torch.manual_seed(0)
    x, w, b = torch.randn(4, 512), torch.randn(6, 512), torch.randn(6)
    torch.testing.assert_close(quantize_linear(x, w, b), F.linear(x, w, b), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------


def test_fold_frozen_bn_is_exact():
    """Folding must reproduce conv-then-BN exactly, not approximately."""
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
    bn = FrozenBatchNorm2d(16)
    bn.weight.data = torch.rand(16) + 0.5
    bn.bias.data = torch.randn(16)
    bn.running_mean.data = torch.randn(16)
    bn.running_var.data = torch.rand(16) + 0.5

    x = torch.randn(2, 3, 8, 8)
    w, b = fold_frozen_bn(conv, bn)

    torch.testing.assert_close(F.conv2d(x, w, b, padding=1), bn(conv(x)), rtol=1e-5, atol=1e-5)


def test_fold_frozen_bn_keeps_the_gradient_on_the_conv_weight():
    """The fold is a function of the live weight, so QAT still trains the convolution."""
    conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
    bn = FrozenBatchNorm2d(16)
    bn.running_var.data = torch.rand(16) + 0.5

    w, _ = fold_frozen_bn(conv, bn)
    w.sum().backward()
    assert conv.weight.grad is not None
    assert conv.weight.grad.abs().sum() > 0


def test_folded_weight_has_a_different_range_than_the_raw_one():
    """Why the fold is load-bearing: the two have different per-channel dynamic ranges.

    Quantizing the raw weight would train the model against rounding error it never meets.
    """
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
    bn = FrozenBatchNorm2d(16)
    bn.weight.data = torch.rand(16) * 4 + 0.1
    bn.running_var.data = torch.rand(16) * 4 + 0.1

    w, _ = fold_frozen_bn(conv, bn)
    raw = ref_row_scale(conv.weight.reshape(16, -1))
    folded = ref_row_scale(w.reshape(16, -1))
    assert not torch.allclose(raw, folded)


# ---------------------------------------------------------------------------
# Training through it
# ---------------------------------------------------------------------------


def test_straight_through_estimator_passes_gradients():
    """QAT is only QAT if the gradient survives the rounding."""
    x = torch.randn(4, 64, requires_grad=True)
    w = torch.randn(32, 64, requires_grad=True)

    quantize_linear(x, w, None).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_exact_mode_agrees_with_simulate_mode():
    """Same arithmetic, different accumulator: they must agree to fp32 noise, not by luck."""
    torch.manual_seed(0)
    x, w, b = torch.randn(8, 3, 256), torch.randn(32, 256), torch.randn(32)
    torch.testing.assert_close(
        quantize_linear(x, w, b, exact=True), quantize_linear(x, w, b), rtol=1e-5, atol=1e-5
    )


def test_exact_mode_conv_agrees_with_simulate_mode():
    torch.manual_seed(0)
    x, w, b = torch.rand(2, 3, 16, 16), torch.randn(16, 3, 3, 3), torch.randn(16)
    torch.testing.assert_close(
        quantize_conv2d(x, w, b, padding=1, exact=True),
        quantize_conv2d(x, w, b, padding=1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_conv_clip_below_one_changes_the_scale():
    torch.manual_seed(0)
    x, w = torch.rand(2, 3, 16, 16), torch.randn(16, 3, 3, 3)
    assert not torch.allclose(
        quantize_conv2d(x, w, None, padding=1, clip=1.0),
        quantize_conv2d(x, w, None, padding=1, clip=0.8),
    )


# ---------------------------------------------------------------------------
# Int8Runtime
# ---------------------------------------------------------------------------


def test_runtime_off_is_exactly_fp32():
    """`int8_groups=0` must cost nothing and change nothing, bit for bit."""
    torch.manual_seed(0)
    i8 = Int8Runtime(0)
    x, w, b = torch.randn(4, 64), torch.randn(32, 64), torch.randn(32)

    assert not i8
    torch.testing.assert_close(i8.linear(INT8_ALL, x, w, b), F.linear(x, w, b), rtol=0, atol=0)

    xc, wc = torch.rand(2, 3, 8, 8), torch.randn(16, 3, 3, 3)
    torch.testing.assert_close(
        i8.conv2d(INT8_ALL, xc, wc, None, padding=1), F.conv2d(xc, wc, None, padding=1), rtol=0, atol=0
    )


def test_runtime_only_touches_selected_groups():
    torch.manual_seed(0)
    i8 = Int8Runtime(INT8_DEC)
    x, w = torch.randn(4, 64), torch.randn(32, 64)

    assert i8.on(INT8_DEC)
    assert not torch.equal(i8.linear(INT8_DEC, x, w, None), F.linear(x, w, None))
    torch.testing.assert_close(i8.linear(1, x, w, None), F.linear(x, w, None), rtol=0, atol=0)


def test_describe_names_the_selected_groups():
    assert Int8Runtime(0).describe() == "int8 off"
    described = Int8Runtime(INT8_ALL, exact=True).describe()
    assert "decoder" in described and "ResNet convolutions" in described and "exact" in described


# ---------------------------------------------------------------------------
# Attention: the four projections do not share one input
# ---------------------------------------------------------------------------


def test_quant_mha_with_groups_off_matches_torch():
    """The unrolled attention must be torch's own, or every int8 number is measured
    against the wrong float baseline."""
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(32, 4)
    mha.eval()
    x = torch.randn(6, 2, 32)

    got = quant_mha(mha, x, x, x, Int8Runtime(0), INT8_DEC)
    want, _ = mha(x, x, x, need_weights=False)

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_quant_mha_honours_the_key_padding_mask():
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(32, 4)
    mha.eval()
    x = torch.randn(6, 2, 32)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 4:] = True

    got = quant_mha(mha, x, x, x, Int8Runtime(0), INT8_DEC, key_padding_mask=mask)
    want, _ = mha(x, x, x, key_padding_mask=mask, need_weights=False)

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_value_is_quantized_against_its_own_input():
    """Self-attention feeds ``x + pos`` to wq/wk and the bare ``x`` to wv.

    Those are two different absmax scales. A simulator that quantized one shared input
    would be modelling a network nothing runs, and the difference is invisible in every
    shape and every loss curve - so it gets pinned here.
    """
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(32, 4)
    mha.eval()
    x = torch.randn(6, 2, 32)
    pos = torch.randn(6, 1, 32) * 10.0  # big enough to move the q/k scale a lot
    i8 = Int8Runtime(INT8_DEC)

    separate = quant_mha(mha, x + pos, x + pos, x, i8, INT8_DEC)
    shared = quant_mha(mha, x + pos, x + pos, x + pos, i8, INT8_DEC)

    assert not torch.allclose(separate, shared)
