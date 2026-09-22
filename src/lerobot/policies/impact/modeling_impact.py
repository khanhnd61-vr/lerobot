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
"""IMPACT - Instruction-Modulated Perception + ACTion chunking.

ACT with a language tower. Two camera frames, a joint state and a natural-language
instruction go in; a 50-step action chunk comes out in one forward pass.

The instruction enters twice::

    "pick up the black bowl"
          |
      T5-small encoder (frozen)  ->  (B, L, 512)
          |
          +-> masked mean-pool -> FiLM head -> (gamma, beta) per ResNet stage --+
          |                                                                     |
          +-> text projection -> L tokens appended to the encoder sequence --+  |
                                                                             |  |
    front  ---+                                                              |  |
              +-> ResNet-18 + FiLM <---------------------------------------------+
    wrist  ---+        |                                                     |
                       v                                                     v
        encoder sequence [latent | state | cam0 | cam1 | text]  (+ key-padding mask)
                       |
             n_encoder_layers x encoder layer
                       |
             n_decoder_layers x decoder layer, chunk_size queries
                       |
                 action head -> (B, chunk_size, action_dim)

Why both paths: with ~600 visual tokens beside ~32 text tokens, and scenes whose visible
objects already disambiguate the task, a policy will learn to ignore the instruction. The
text tokens are a channel the gradient may use; FiLM is one it cannot route around.

Everything not concerned with language is ACT unchanged, including the CVAE style encoder
that collapses to a constant token at inference. The encoder and decoder layer blocks are
ACT's own classes, so a backbone or transformer tensor is interchangeable between the two.

The architecture is also the reference for a CPU inference engine, which is why the
int8 path below mirrors a specific deployment kernel rather than a generic quantizer.
"""

import logging
from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..act.modeling_act import (
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
    get_activation_fn,
)
from ..pretrained import PreTrainedPolicy
from .configuration_impact import IMPACTConfig
from .film import FiLMHead, FiLMResNet
from .quant_transformer import quant_decoder, quant_encoder
from .quantization import INT8_PROJ, Int8Runtime
from .text import TextEncoder, masked_mean


class IMPACTPolicy(PreTrainedPolicy):
    """Instruction-Modulated Perception + ACTion chunking."""

    config_class = IMPACTConfig
    name = "impact"

    def __init__(self, config: IMPACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = IMPACT(config)

        if config.int8_groups:
            # Say so. A checkpoint trained for int8 declares `int8_groups` in its own
            # config.json, so loading it reproduces the deployment arithmetic rather
            # than fp32 - which is the honest default but is not what a reader of
            # `lerobot-rollout --policy.path=...` would otherwise assume, and in
            # PyTorch it is simulated rather than fast. Pass `--policy.int8_groups=0`
            # for the fp32 numerics.
            logging.info("IMPACT: %s (simulated; no speedup here)", self.model.i8.describe())

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        """Three groups: the backbone gets its own learning rate, as in ACT, and the frozen
        text tower contributes nothing.

        ``requires_grad`` already excludes the frozen encoder, but naming it here keeps the
        intent visible if someone flips ``freeze_language_encoder``.
        """
        backbone_prefix = "model.backbone"
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith(backbone_prefix) and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n.startswith(backbone_prefix) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """Called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action, refilling the chunk queue when it empties."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        batch = self._stack_images(batch)
        return self.model(batch)[0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the training loss."""
        batch = self._stack_images(batch)

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy so adding a key does not modify the caller's
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch


class IMPACT(nn.Module):
    """The network. See the module docstring for the graph."""

    def __init__(self, config: IMPACTConfig):
        super().__init__()
        self.config = config

        # --- language ------------------------------------------------------------------
        self.text = TextEncoder(config.language_encoder, freeze=config.freeze_language_encoder)
        self.text_input_proj = nn.Linear(self.text.embed_dim, config.dim_model)
        self.text_pos_embed = nn.Embedding(config.tokenizer_max_length, config.dim_model)

        # --- CVAE style encoder (training only; z = 0 at inference) ----------------------
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )
            if config.vae_sees_language:
                # Without this the latent absorbs task identity during training and the
                # language pathway becomes redundant - a failure that is invisible at
                # inference, where the latent is zero, and invisible to parity.
                self.vae_encoder_language_input_proj = nn.Linear(self.text.embed_dim, config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)

            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            if config.vae_sees_language:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # --- vision ----------------------------------------------------------------------
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = FiLMResNet(backbone_model)
        if config.use_film:
            self.film_head = FiLMHead(
                self.text.embed_dim, self.backbone.stage_channels, config.film_hidden_dim
            )

        # --- transformer -----------------------------------------------------------------
        self.encoder = ACTEncoder(config)
        self.decoder = IMPACTDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, config.dim_model, kernel_size=1
        )

        n_1d_tokens = 1  # the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # Which GEMMs run in simulated int8. Plain state, not a buffer or a submodule:
        # the quantizer is dynamic, so there is nothing here for `.to()` or a state dict
        # to carry. `config.int8_groups == 0` makes every call below a plain float op.
        self.i8 = Int8Runtime(config.int8_groups, conv_clip=config.int8_conv_clip)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform on the transformer, as in the original ACT code.

        The FiLM head is deliberately excluded: :class:`~.film.FiLMHead` zero-initializes
        its final layer so that ``(1 + gamma) * x + beta`` is the identity at init, and a
        Xavier pass over it would undo exactly that.
        """
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_language(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Run the text tower and apply language dropout.

        Returns ``(hidden, attention_mask, pooled)``: the raw T5 output ``(B, L, d_text)``,
        the possibly-dropped attention mask ``(B, L)`` and the masked mean-pool
        ``(B, d_text)``.

        Language dropout zeroes a sample's attention mask outright rather than blanking the
        token ids. That drops the instruction from both paths at once - every text column is
        masked in the encoder and the decoder, and the pooled vector goes to zero, which
        makes FiLM the identity. It is the same "no language" the ``use_film=False``
        ablation sees, so a dropout-trained policy has actually practised running blind.
        """
        if OBS_LANGUAGE_TOKENS not in batch:
            raise KeyError(
                f"IMPACT requires tokenized instructions under {OBS_LANGUAGE_TOKENS!r}. Build the batch "
                "with `make_impact_pre_post_processors`, which adds the TokenizerProcessorStep."
            )
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # The tower gets the TRUE padding mask, never a mask of ones. T5 is bidirectional:
        # let it attend to the padded positions and the hidden states of the *real* tokens
        # start depending on whatever the tokenizer happened to pad with, which no
        # downstream mask can undo. `test_padded_text_rows_do_not_reach_the_output` fails
        # on exactly that mistake.
        hidden = self.text(input_ids, attention_mask)

        # Language dropout is applied after, to the mask the policy reads - not to the mask
        # T5 reads. That drops the instruction from both paths at once (every text column is
        # masked in the encoder and the decoder, and the pooled vector goes to zero, making
        # FiLM the identity) without disturbing the tower's own attention.
        if self.training and self.config.language_dropout > 0.0:
            drop = torch.rand(input_ids.shape[0], device=input_ids.device) < self.config.language_dropout
            attention_mask = attention_mask * (~drop).unsqueeze(1).to(attention_mask.dtype)

        pooled = masked_mean(hidden, attention_mask)
        return hidden, attention_mask, pooled

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """See :class:`IMPACTPolicy.forward` for the batch contract.

        Returns ``(B, chunk_size, action_dim)`` actions and the latent PDF parameters.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0]

        text_hidden, text_mask, text_pooled = self.encode_language(batch)

        # --- latent ----------------------------------------------------------------------
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            vae_encoder_input = [cls_embed]
            n_prefix = 1
            if self.config.vae_sees_language:
                vae_encoder_input.append(self.vae_encoder_language_input_proj(text_pooled).unsqueeze(1))
                n_prefix += 1
            if self.config.robot_state_feature:
                vae_encoder_input.append(
                    self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
                )
                n_prefix += 1
            vae_encoder_input.append(self.vae_encoder_action_input_proj(batch[ACTION]))
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            prefix_is_pad = torch.full((batch_size, n_prefix), False, device=vae_encoder_input.device)
            key_padding_mask = torch.cat([prefix_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim],
                dtype=torch.float32,
                device=batch[OBS_IMAGES][0].device,
            )

        # --- FiLM parameters -------------------------------------------------------------
        if self.config.use_film:
            gammas, betas = self.film_head(text_pooled)
        else:
            gammas = betas = None

        # --- encoder tokens: [latent | state | cams | text] --------------------------------
        # The latent projection is never a quantization group: at inference the latent is
        # zero, so the engine folds this whole layer to its bias (`latent_tok`) and there
        # is no GEMM left to quantize.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(
                self.i8.linear(
                    INT8_PROJ,
                    batch[OBS_STATE],
                    self.encoder_robot_state_input_proj.weight,
                    self.encoder_robot_state_input_proj.bias,
                )
            )

        for img in batch[OBS_IMAGES]:
            cam_features = self.backbone(img, gammas, betas, i8=self.i8)
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)

            if self.i8.on(INT8_PROJ):
                # A 1x1 convolution over a feature map is a linear map over its tokens,
                # and that is what the engine holds it as (`tf.img_proj`, an nn::Linear).
                # The difference is not cosmetic: as a convolution it would take one
                # activation scale for the whole map, as a linear it takes one per token,
                # and the engine takes one per token. Rearrange first, then project.
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                proj = self.encoder_img_feat_input_proj
                cam_features = self.i8.linear(
                    INT8_PROJ, cam_features, proj.weight.reshape(proj.out_channels, -1), proj.bias
                )
            else:
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

        n_vision_tokens = len(encoder_in_tokens)

        # The text tokens go last, which keeps the visual block layout byte-identical to
        # ACT's and makes the padding mask a contiguous tail.
        text_tokens = einops.rearrange(self.text_input_proj(text_hidden), "b l c -> l b c")
        encoder_in_tokens.extend(list(text_tokens))
        encoder_in_pos_embed.extend(list(self.text_pos_embed.weight.unsqueeze(1)))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # --- key padding mask ------------------------------------------------------------
        # True marks a position attention must not read. Only the text tail is ever padded,
        # so every row keeps its ~600 visual keys and no row is fully masked (which would
        # make MultiheadAttention emit NaNs).
        vision_is_pad = torch.zeros((batch_size, n_vision_tokens), dtype=torch.bool, device=text_mask.device)
        key_padding_mask = torch.cat([vision_is_pad, ~text_mask.bool()], axis=1)

        # --- transformer ------------------------------------------------------------------
        # With int8 on, the same layers run through quant_transformer, which opens
        # nn.MultiheadAttention up so each of wq/wk/wv/wo can be quantized against its own
        # input. Same modules, same parameters, same result when no group is selected.
        if self.i8:
            encoder_out = quant_encoder(
                self.encoder,
                encoder_in_tokens,
                self.i8,
                pos_embed=encoder_in_pos_embed,
                key_padding_mask=key_padding_mask,
            )
        else:
            encoder_out = self.encoder(
                encoder_in_tokens, pos_embed=encoder_in_pos_embed, key_padding_mask=key_padding_mask
            )
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        if self.i8:
            decoder_out = quant_decoder(
                self.decoder,
                decoder_in,
                encoder_out,
                self.i8,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
                memory_key_padding_mask=key_padding_mask,
            )
        else:
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
                memory_key_padding_mask=key_padding_mask,
            )

        decoder_out = decoder_out.transpose(0, 1)  # back to (B, S, C)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class IMPACTDecoder(nn.Module):
    """ACT's decoder, plus a key-padding mask on the cross-attention.

    ACT never needed one - its memory is all visual and never padded. IMPACT's memory ends
    in ``tokenizer_max_length`` text rows, of which only the real tokens should be read.
    Those rows carry genuine T5 outputs rather than zeros, so leaving them unmasked changes
    the tensor the decoder sees.
    """

    def __init__(self, config: IMPACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([IMPACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)


class IMPACTDecoderLayer(nn.Module):
    """ACT's decoder layer with ``key_padding_mask`` threaded into the cross-attention.

    Kept as a copy rather than a subclass because the mask has to reach the middle of
    ``forward``, and a copy that is obviously ACT's block beats an override that silently
    diverges from it.
    """

    def __init__(self, config: IMPACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: ``(Decoder Sequence, Batch, Channel)`` input tokens.
            encoder_out: ``(Encoder Sequence, B, C)`` memory to cross-attend.
            encoder_pos_embed: ``(ES, 1, C)`` positional embedding for the keys.
            decoder_pos_embed: ``(DS, 1, C)`` positional embedding for the queries.
            memory_key_padding_mask: ``(B, ES)``, True where the memory must not be read.
        Returns:
            ``(DS, B, C)`` decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x
