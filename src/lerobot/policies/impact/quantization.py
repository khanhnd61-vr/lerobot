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
"""IMPACT's int8 path: the deployment quantizer, in PyTorch.

This is not a generic quantization utility. It is a transcription of one specific
W8A8 CPU kernel, so that quantization-aware training optimizes the arithmetic that
actually runs on the deployment target, and so that a parity harness can assert the two
agree rather than hope they do. Anything below that reads like an arbitrary choice is a
property of that kernel and has to be matched, not improved on.

The scheme, in full::

    weights      w_q[n,k] = clamp(rne(W[n,k] / s_w[n]), -127, 127)
                 s_w[n]   = max_k |W[n,k]| / 127          (1.0 for an all-zero row)

    activations  Linear:  s_a[t] = max_k |x[t,k]| / 127   (per token)
                 Conv2d:  s_a    = max |x| * clip / 127   (per image, whole feature map)
                 x_q      = clamp(rne(x / s_a), -127, 127)

    output       out[t,n] = bias[n] + (float)(SUM_k w_q[n,k] x_q[t,k]) * s_a[t] * s_w[n]

Four properties of that scheme are load-bearing and easy to lose:

- **Symmetric, no zero point.** The inner loop is a plain signed dot, so there are no
  cross-terms to model and nothing to calibrate.
- **The activation scales are dynamic.** They are a function of the tensor at hand, not
  a calibrated constant, so a checkpoint carries *no* quantization state - only weights
  that have learned to survive the rounding. That is why the QAT output is still an
  ordinary fp32 checkpoint and why :func:`quantize_linear` needs no observer.
- **Rounding is round-to-nearest-even.** ``torch.round`` and C's ``lrint`` under the
  default rounding mode agree; ``trunc(x + copysign(0.5, x))`` does not, and the
  disagreement is a single ULP at exactly the inputs a parity harness would call noise.
- **Weight scales are per output row, activation scales per token** (per image for a
  conv). One scale per tensor collapses a ResNet's channel spread and a transformer's
  token outliers onto one exponent, which is the difference between int8 that works and
  int8 that does not.

Two execution modes, because they answer different questions:

``simulate``
    Quantize-dequantize both operands and run the ordinary fp32 matmul, with a
    straight-through estimator so gradients flow. This is what QAT trains through. It
    is mathematically the deployment arithmetic but accumulates in fp32.

``exact``
    Accumulate the integer products in float64, which is exact for every K here
    (|w_q x_q| <= 16129 and K <= 3200, so the sums stay far inside float64's 2**53),
    then apply the epilogue in fp32 in the kernel's own multiply order. Use this to
    compare against the engine: an fp32 accumulation over K = 3200 drifts by ~1e-5
    relative, which would swamp the ~1e-7 disagreement a parity check is looking for.

The groups are selected by a bitmask, and what the mask leaves out matters as much as
what it covers: the T5 tower, the text projection and the FiLM head are cached per
episode by ``set_instruction()``, so quantizing them trades accuracy for a saving that
amortizes to nothing over a rollout. The action head is excluded too - it is [6, 512],
and the engine's weight packer requires N % 16 == 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# The group bits, as the deployment engine's group mask defines them. Values are part of
# the wire contract with the engine: a mask set here and a mask set there must agree.
INT8_ENC_ATTN = 1
INT8_ENC_W1 = 2
INT8_ENC_W2 = 4
INT8_PROJ = 8
INT8_DEC = 16
INT8_CONV = 32
INT8_ALL = INT8_ENC_ATTN | INT8_ENC_W1 | INT8_ENC_W2 | INT8_PROJ | INT8_DEC | INT8_CONV

GROUP_NAMES = {
    INT8_ENC_ATTN: "encoder attention",
    INT8_ENC_W1: "encoder w1",
    INT8_ENC_W2: "encoder w2",
    INT8_PROJ: "token projections",
    INT8_DEC: "decoder",
    INT8_CONV: "ResNet convolutions",
}

QMAX = 127
"""The kernel clamps to [-127, 127], not [-128, 127), so -128 is never produced."""

PACK_ROW_MULTIPLE = 16
"""The engine's weight packer asserts N % 16 == 0, so a narrower layer stays fp32 there.

Mirrored here rather than assumed away: a simulator that quantizes a layer the engine
leaves alone disagrees with it for a reason no tolerance will explain.
"""


def quantizable(n_out: int) -> bool:
    """Whether the engine's packer would take a layer with this many output rows."""
    return n_out % PACK_ROW_MULTIPLE == 0


def _round_ste(x: Tensor) -> Tensor:
    """``torch.round`` (round-to-nearest-even) with a straight-through gradient."""
    return x + (torch.round(x) - x).detach()


def _row_scale(w: Tensor) -> Tensor:
    """``s_w[n] = max_k |W[n, k]| / 127`` over a 2-D ``[N, K]`` weight.

    An all-zero row quantizes to all zeros whatever the scale, so the kernel substitutes
    1.0 rather than divide by zero; the same substitution here keeps the two bit-identical
    on a layer that training has driven to zero.
    """
    amax = w.detach().abs().amax(dim=1, keepdim=True)
    return torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))


def _token_scale(x: Tensor) -> Tensor:
    """``s_a[t] = max_k |x[t, k]| / 127`` over the last dimension of ``x``."""
    amax = x.detach().abs().amax(dim=-1, keepdim=True)
    return torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))


def _tensor_scale(x: Tensor, clip: float) -> Tensor:
    """``s_a = max |x| * clip / 127`` over one image's whole ``[C, H, W]``.

    Per image, not per batch: the engine encodes one camera frame per call, so a scale
    pooled over a training batch is a quantizer no deployment ever runs.
    """
    amax = x.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
    return torch.where(amax > 0, amax * clip / QMAX, torch.ones_like(amax))


def _inv(scale: Tensor) -> Tensor:
    """``1.0f / s``, in the input's own precision.

    The kernel divides by the scale exactly once - to form this reciprocal - and then
    *multiplies* every element by it, in the weight packer and in both activation
    quantizers alike. ``x * (1/s)`` and ``x / s`` differ by up to one ulp, which changes nothing at all
    except when the true quotient lands within an ulp of a .5 boundary - and there it
    changes the rounded integer by one. Over a 512x3200 weight that happens often enough
    to put a simulator 1e-2 away from the engine it claims to model, which is where this
    function came from.
    """
    return torch.reciprocal(scale)


def _qdq(x: Tensor, scale: Tensor) -> Tensor:
    """Quantize-dequantize with a straight-through gradient."""
    q = torch.clamp(_round_ste(x * _inv(scale)), -QMAX, QMAX)
    return q * scale


def _qi(x: Tensor, scale: Tensor) -> Tensor:
    """Quantize to integer-valued floats, no gradient path (the exact mode is inference)."""
    return torch.clamp(torch.round(x.detach() * _inv(scale)), -QMAX, QMAX)


def fold_frozen_bn(conv: torch.nn.Conv2d, bn: torch.nn.Module) -> tuple[Tensor, Tensor]:
    """Fold a ``FrozenBatchNorm2d`` into the convolution ahead of it.

    ``scale = gamma / sqrt(var + eps)``; ``W' = W * scale``, ``b' = beta - mean * scale``.
    The exporter folds at export time, which is why it is the folded weight - never the
    raw one - that has to be quantized: the engine has no BatchNorm left to absorb the
    rounding error.

    Exact rather than an approximation, because ACT and IMPACT build their backbones with
    ``FrozenBatchNorm2d``, whose statistics are buffers and never move. The formula is
    written as ``gamma / sqrt(var + eps)`` rather than torchvision's algebraically equal
    ``gamma * rsqrt(var + eps)`` so that it rounds the way the exporter rounds.
    """
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale.reshape(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        b = b + conv.bias * scale
    return w, b


def quantize_linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    exact: bool = False,
) -> Tensor:
    """``F.linear`` through the W8A8 kernel: per-token activations, per-row weights.

    ``x`` is ``[..., K]``; every leading dimension is a token, which is what the engine
    sees - it runs one sequence of rows at a time and has no batch concept.
    """
    if not quantizable(weight.shape[0]):
        return F.linear(x, weight, bias)

    if not exact:
        return F.linear(_qdq(x, _token_scale(x)), _qdq(weight, _row_scale(weight)), bias)

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    s_a = _token_scale(x2)
    s_w = _row_scale(weight)
    acc = _qi(x2, s_a).double() @ _qi(weight, s_w).double().t()
    # The epilogue in the kernel's own order: (float)acc * ascale[t] * wscale[n], then
    # the bias. Rounding the int32 accumulator to fp32 first is what the kernel does, so
    # it is done here too rather than kept in float64 to look tidier.
    out = acc.float() * s_a.float() * s_w.t().float()
    if bias is not None:
        out = out + bias
    return out.reshape(*shape[:-1], weight.shape[0])


def quantize_conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    stride: int = 1,
    padding: int = 0,
    clip: float = 1.0,
    exact: bool = False,
) -> Tensor:
    """``F.conv2d`` through the W8A8 kernel: per-image activations, per-channel weights.

    The activation scale is taken over the *input* feature map before padding, matching
    the kernel, which quantizes ``x`` and then im2cols the int8 buffer. Zero padding
    survives quantization as zero either way, so the two orders agree exactly.
    """
    if not quantizable(weight.shape[0]):
        return F.conv2d(x, weight, bias, stride=stride, padding=padding)

    w2d = weight.reshape(weight.shape[0], -1)
    if not exact:
        s_a = _tensor_scale(x, clip)
        wq = _qdq(w2d, _row_scale(w2d)).reshape(weight.shape)
        return F.conv2d(_qdq(x, s_a), wq, bias, stride=stride, padding=padding)

    s_a = _tensor_scale(x, clip)
    s_w = _row_scale(w2d)
    xq = _qi(x, s_a)
    wq = _qi(w2d, s_w).reshape(weight.shape)
    # Integer products in float64: exact, since |w_q x_q| <= 16129 and the widest conv
    # here is K = 4608, so no accumulator leaves float64's exact-integer range.
    acc = F.conv2d(xq.double(), wq.double(), None, stride=stride, padding=padding)
    out = acc.float() * s_a.float() * s_w.reshape(1, -1, 1, 1).float()
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


class Int8Runtime:
    """Which groups are quantized, and how the quantized ops execute.

    Held on the model rather than passed down as arguments so that the forward passes
    read as the float ones with a guard, instead of threading a config through every
    call site.
    """

    def __init__(self, groups: int = 0, *, exact: bool = False, conv_clip: float = 1.0):
        self.groups = int(groups)
        self.exact = bool(exact)
        self.conv_clip = float(conv_clip)

    def __bool__(self) -> bool:
        return self.groups != 0

    def on(self, group: int) -> bool:
        return bool(self.groups & group)

    def linear(self, group: int, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if not self.on(group):
            return F.linear(x, weight, bias)
        return quantize_linear(x, weight, bias, exact=self.exact)

    def conv2d(
        self,
        group: int,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> Tensor:
        if not self.on(group):
            return F.conv2d(x, weight, bias, stride=stride, padding=padding)
        return quantize_conv2d(
            x, weight, bias, stride=stride, padding=padding, clip=self.conv_clip, exact=self.exact
        )

    def describe(self) -> str:
        if not self.groups:
            return "int8 off"
        names = [n for bit, n in GROUP_NAMES.items() if self.groups & bit]
        return f"int8 groups {self.groups} ({', '.join(names)}){', exact' if self.exact else ''}"
