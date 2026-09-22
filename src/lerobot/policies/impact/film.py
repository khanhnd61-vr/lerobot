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
"""Instruction-modulated perception: the "IM" in IMPACT.

FiLM (Perez et al., 2018) conditions a vision backbone by scaling and shifting its feature
maps per channel from an external vector - here the pooled instruction embedding. It is the
half of IMPACT that stops the policy from ignoring language.

The alternative - text tokens in the transformer encoder and nothing else - leaves the
model free to solve a multi-task benchmark from vision alone whenever the visible objects
disambiguate the task, which on most manipulation suites they do. Text tokens are then a
side channel the gradient never has to use. FiLM puts the instruction inside the perceptual
path, where routing around it is not an option.

Two details are load-bearing:

- **The modulation is ``(1 + gamma) * x + beta``, not ``gamma * x + beta``.** With the head
  zero-initialized, that makes FiLM exactly the identity at init, so an untrained head
  leaves the pretrained ResNet alone and training starts from ordinary ACT. Written as a
  plain product, a zero-init head annihilates the feature map and nothing recovers.
- **It is applied at the output of each stage**, after the block's residual add and ReLU -
  not inside the residual branch. Four points for a ResNet-18, at 64/128/256/512 channels.
"""

import torch
from torch import Tensor, nn

from .quantization import INT8_CONV, Int8Runtime, fold_frozen_bn


class FiLMHead(nn.Module):
    """Maps a pooled language embedding to per-stage ``(gamma, beta)``.

    Args:
        in_dim: width of the pooled text vector.
        channels: output channels of each stage to modulate, in backbone order.
        hidden_dim: width of an optional hidden layer; 0 for a single linear map.

    The final layer is zero-initialized, so at init every ``gamma`` and ``beta`` is zero and
    :func:`apply_film` is the identity. :meth:`is_identity_at_init` states that as an
    assertion the converter and the tests can check rather than trust.
    """

    def __init__(self, in_dim: int, channels: list[int], hidden_dim: int = 0):
        super().__init__()
        if not channels:
            raise ValueError("FiLMHead needs at least one stage to modulate.")
        self.channels = list(channels)
        self.total = sum(self.channels)

        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * self.total),
            )
            final = self.net[-1]
        else:
            self.net = nn.Linear(in_dim, 2 * self.total)
            final = self.net

        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, pooled: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """``(B, in_dim)`` -> per-stage gammas and betas, each ``(B, C_i)``.

        The flat ``(B, 2 * total)`` output is cut gamma-half first, then beta-half, and each
        half is split in backbone order. A deployment engine reads the same buffer with the
        same convention, so this order is part of the format.
        """
        gb = self.net(pooled)
        gamma_flat, beta_flat = gb[:, : self.total], gb[:, self.total :]
        gammas = list(torch.split(gamma_flat, self.channels, dim=1))
        betas = list(torch.split(beta_flat, self.channels, dim=1))
        return gammas, betas


def apply_film(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    """``(1 + gamma) * x + beta`` broadcast over a ``(B, C, H, W)`` feature map.

    See the module docstring for why the ``1 +`` is not cosmetic.
    """
    g = gamma.unsqueeze(-1).unsqueeze(-1)
    b = beta.unsqueeze(-1).unsqueeze(-1)
    return (1.0 + g) * x + b


class FiLMResNet(nn.Module):
    """A torchvision ResNet truncated at ``layer4``, with FiLM after every stage.

    ACT uses ``IntermediateLayerGetter`` to pull ``layer4`` out of the backbone. That cannot
    express FiLM, which has to reach inside between stages, so the stages are run
    explicitly here instead. The modules are the torchvision ones untouched - same weights,
    same frozen BatchNorm - so a checkpoint's backbone tensors are interchangeable with
    ACT's.

    ``forward`` with ``gammas=None`` runs the plain backbone, which is both the
    ``use_film=False`` configuration and the ablation arm.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.stages = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
        self.stage_channels = [self._block_out_channels(stage[-1]) for stage in self.stages]

    @staticmethod
    def _block_out_channels(block: nn.Module) -> int:
        """Output channels of a residual block, from its last convolution.

        Read off the conv rather than the norm: ACT builds these backbones with
        ``FrozenBatchNorm2d``, which carries its channel count only in the shape of its
        buffers and has no ``num_features``. A basic block (ResNet-18/34) ends at ``conv2``,
        a bottleneck (ResNet-50+) at ``conv3``.

        Computed in ``__init__`` rather than exposed as a property: a property that raises
        AttributeError inside an ``nn.Module`` is swallowed by ``Module.__getattr__`` and
        resurfaces as "no attribute stage_channels", which says nothing about the cause.
        """
        for name in ("conv3", "conv2"):
            conv = getattr(block, name, None)
            if isinstance(conv, nn.Conv2d):
                return conv.out_channels
        raise ValueError(
            f"Cannot infer output channels of {type(block).__name__}: expected a conv2 or conv3."
        )

    def forward(
        self,
        x: Tensor,
        gammas: list[Tensor] | None = None,
        betas: list[Tensor] | None = None,
        i8: Int8Runtime | None = None,
    ) -> Tensor:
        """``(B, 3, H, W)`` -> ``(B, C, h, w)`` feature map from ``layer4``.

        ``i8`` with the convolution group selected runs the BN-folded, W8A8 backbone
        instead; see :meth:`_forward_int8`.
        """
        if i8 is not None and i8.on(INT8_CONV):
            return self._forward_int8(x, gammas, betas, i8)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if gammas is not None and betas is not None:
                x = apply_film(x, gammas[i], betas[i])
        return x

    def _forward_int8(
        self,
        x: Tensor,
        gammas: list[Tensor] | None,
        betas: list[Tensor] | None,
        i8: Int8Runtime,
    ) -> Tensor:
        """The backbone with every BatchNorm folded away and every convolution in W8A8.

        **The folding is not an optimization, it is the point.** The exporter emits folded
        weights, so the tensor the engine quantizes is ``W * gamma / sqrt(var + eps)`` and
        not ``W``. Those two have different per-output-channel
        dynamic ranges, so quantizing the raw weight and then normalizing would train the
        model against rounding error it will never meet - and leave it untrained against
        the rounding error it will. Folding here is free: ``FrozenBatchNorm2d`` holds its
        statistics as buffers, so the fold is a fixed function of the live conv weight and
        the gradient reaches that weight through it.

        ReLU, the residual add and the max-pool stay fp32, as they do in the engine -
        only the convolutions are a group.
        """
        w, b = fold_frozen_bn(self.conv1, self.bn1)
        x = i8.conv2d(INT8_CONV, x, w, b, stride=self.conv1.stride[0], padding=self.conv1.padding[0])
        x = self.maxpool(self.relu(x))
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = _basic_block_int8(block, x, i8)
            if gammas is not None and betas is not None:
                x = apply_film(x, gammas[i], betas[i])
        return x


def _basic_block_int8(block: nn.Module, x: Tensor, i8: Int8Runtime) -> Tensor:
    """One torchvision ``BasicBlock``, BN folded into each convolution, convs in W8A8.

    Written out rather than delegated to the block's own ``forward`` for the same reason
    :class:`FiLMResNet` writes out the stages: there is nowhere to insert a quantizer in
    between a convolution and the norm that has to be folded into it.
    """
    w1, b1 = fold_frozen_bn(block.conv1, block.bn1)
    out = i8.conv2d(INT8_CONV, x, w1, b1, stride=block.conv1.stride[0], padding=block.conv1.padding[0])
    out = block.relu(out)

    w2, b2 = fold_frozen_bn(block.conv2, block.bn2)
    out = i8.conv2d(INT8_CONV, out, w2, b2, stride=block.conv2.stride[0], padding=block.conv2.padding[0])

    identity = x
    if block.downsample is not None:
        conv, norm = block.downsample[0], block.downsample[1]
        wd, bd = fold_frozen_bn(conv, norm)
        identity = i8.conv2d(INT8_CONV, x, wd, bd, stride=conv.stride[0], padding=conv.padding[0])
    return block.relu(out + identity)
