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
"""The transformer stack with the int8 kernel spliced into its GEMMs.

``nn.MultiheadAttention`` computes q, k, v and the output projection behind one opaque
call, so there is nowhere to put a quantizer: the engine quantizes each of ``wq``, ``wk``,
``wv`` and ``wo`` against *its own* input, and three of those four inputs are internal to
that call. These functions open the module up and run the same arithmetic explicitly,
reading the very same parameters - ``in_proj_weight`` sliced in thirds, ``out_proj`` - so
no checkpoint key moves and an IMPACT checkpoint stays loadable by the ordinary path and
by the engine's exporter alike.

The detail worth naming, because it is invisible and silently halves the point of the
exercise if missed: **self-attention feeds a different tensor to wv than to wq and wk.**
``ACTEncoderLayer`` computes ``q = k = x + pos_embed`` but ``value=x``, so wq and wk see
the position-carrying tensor and wv sees the bare one. Those are two different per-token
absmax scales. The engine does exactly this - it passes ``x + tok_pos`` for q and k and
the bare ``x`` for v - and a simulator that quantizes one shared input would be
quantizing a model nothing runs.

Cross-attention has the same shape: queries carry the decoder positions, keys carry the
encoder positions, values are the raw encoder output.

With every group off these functions reduce to the float layers they mirror, to fp32
noise. That is worth checking rather than asserting, and the parity harness does check it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .quantization import INT8_DEC, INT8_ENC_ATTN, INT8_ENC_W1, INT8_ENC_W2, Int8Runtime


def _split_in_proj(mha: nn.MultiheadAttention) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``in_proj_weight`` [3E, E] and its bias, cut into the engine's wq / wk / wv.

    The cut is along output rows, and the quantizer's scales are per output row, so
    quantizing the three slices separately is bit-identical to quantizing them fused -
    which is what lets the engine hold three matrices where torch holds one.
    """
    e = mha.embed_dim
    w = mha.in_proj_weight
    b = mha.in_proj_bias
    bq, bk, bv = (None, None, None) if b is None else (b[:e], b[e : 2 * e], b[2 * e :])
    return w[:e], w[e : 2 * e], w[2 * e :], bq, bk, bv


def quant_mha(
    mha: nn.MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    i8: Int8Runtime,
    group: int,
    key_padding_mask: Tensor | None = None,
    dropout: float = 0.0,
    training: bool = False,
) -> Tensor:
    """``nn.MultiheadAttention`` unrolled, with W8A8 on the four projections.

    Sequence-first ``(S, B, E)`` throughout, as the rest of ACT's transformer is. The
    attention scores and the softmax stay fp32: they are not a selectable group, and the
    engine computes them in fp32 too.
    """
    wq, wk, wv, bq, bk, bv = _split_in_proj(mha)
    heads = mha.num_heads
    e = mha.embed_dim
    head_dim = e // heads

    len_q, batch, _ = query.shape
    len_kv = key.shape[0]

    q = i8.linear(group, query, wq, bq)
    k = i8.linear(group, key, wk, bk)
    v = i8.linear(group, value, wv, bv)

    # (S, B, E) -> (B * heads, S, head_dim), the layout torch's own implementation uses.
    q = q.reshape(len_q, batch * heads, head_dim).transpose(0, 1)
    k = k.reshape(len_kv, batch * heads, head_dim).transpose(0, 1)
    v = v.reshape(len_kv, batch * heads, head_dim).transpose(0, 1)

    scores = torch.bmm(q * (head_dim**-0.5), k.transpose(1, 2))
    if key_padding_mask is not None:
        # (B, S_kv) -> (B * heads, 1, S_kv). True marks a key attention must not read.
        mask = key_padding_mask.reshape(batch, 1, 1, len_kv).expand(-1, heads, -1, -1)
        mask = mask.reshape(batch * heads, 1, len_kv)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=dropout, training=training)

    ctx = torch.bmm(attn, v).transpose(0, 1).reshape(len_q, batch, e)
    return i8.linear(group, ctx, mha.out_proj.weight, mha.out_proj.bias)


def quant_encoder_layer(
    layer: nn.Module,
    x: Tensor,
    i8: Int8Runtime,
    pos_embed: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
) -> Tensor:
    """``ACTEncoderLayer.forward``, with the attention and the two FFN GEMMs quantized."""
    skip = x
    if layer.pre_norm:
        x = layer.norm1(x)
    q = k = x if pos_embed is None else x + pos_embed
    x = quant_mha(
        layer.self_attn,
        q,
        k,
        x,
        i8,
        INT8_ENC_ATTN,
        key_padding_mask=key_padding_mask,
        dropout=layer.self_attn.dropout,
        training=layer.training,
    )
    x = skip + layer.dropout1(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm2(x)
    else:
        x = layer.norm1(x)
        skip = x
    h = layer.activation(i8.linear(INT8_ENC_W1, x, layer.linear1.weight, layer.linear1.bias))
    x = i8.linear(INT8_ENC_W2, layer.dropout(h), layer.linear2.weight, layer.linear2.bias)
    x = skip + layer.dropout2(x)
    if not layer.pre_norm:
        x = layer.norm2(x)
    return x


def quant_encoder(
    encoder: nn.Module,
    x: Tensor,
    i8: Int8Runtime,
    pos_embed: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
) -> Tensor:
    """``ACTEncoder.forward`` over :func:`quant_encoder_layer`."""
    for layer in encoder.layers:
        x = quant_encoder_layer(layer, x, i8, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
    return encoder.norm(x)


def quant_decoder_layer(
    layer: nn.Module,
    x: Tensor,
    encoder_out: Tensor,
    i8: Int8Runtime,
    decoder_pos_embed: Tensor | None = None,
    encoder_pos_embed: Tensor | None = None,
    memory_key_padding_mask: Tensor | None = None,
) -> Tensor:
    """``IMPACTDecoderLayer.forward``, with all ten decoder GEMMs quantized.

    One group covers the whole layer (:data:`INT8_DEC`), matching the engine, which does
    not split the decoder finer than that.
    """
    skip = x
    if layer.pre_norm:
        x = layer.norm1(x)
    q = k = layer.maybe_add_pos_embed(x, decoder_pos_embed)
    x = quant_mha(
        layer.self_attn, q, k, x, i8, INT8_DEC, dropout=layer.self_attn.dropout, training=layer.training
    )
    x = skip + layer.dropout1(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm2(x)
    else:
        x = layer.norm1(x)
        skip = x
    x = quant_mha(
        layer.multihead_attn,
        layer.maybe_add_pos_embed(x, decoder_pos_embed),
        layer.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
        encoder_out,
        i8,
        INT8_DEC,
        key_padding_mask=memory_key_padding_mask,
        dropout=layer.multihead_attn.dropout,
        training=layer.training,
    )
    x = skip + layer.dropout2(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm3(x)
    else:
        x = layer.norm2(x)
        skip = x
    h = layer.activation(i8.linear(INT8_DEC, x, layer.linear1.weight, layer.linear1.bias))
    x = i8.linear(INT8_DEC, layer.dropout(h), layer.linear2.weight, layer.linear2.bias)
    x = skip + layer.dropout3(x)
    if not layer.pre_norm:
        x = layer.norm3(x)
    return x


def quant_decoder(
    decoder: nn.Module,
    x: Tensor,
    encoder_out: Tensor,
    i8: Int8Runtime,
    decoder_pos_embed: Tensor | None = None,
    encoder_pos_embed: Tensor | None = None,
    memory_key_padding_mask: Tensor | None = None,
) -> Tensor:
    """``IMPACTDecoder.forward`` over :func:`quant_decoder_layer`."""
    for layer in decoder.layers:
        x = quant_decoder_layer(
            layer,
            x,
            encoder_out,
            i8,
            decoder_pos_embed=decoder_pos_embed,
            encoder_pos_embed=encoder_pos_embed,
            memory_key_padding_mask=memory_key_padding_mask,
        )
    return decoder.norm(x)
