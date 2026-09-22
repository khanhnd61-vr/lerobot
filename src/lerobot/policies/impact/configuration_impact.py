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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("impact")
@dataclass
class IMPACTConfig(PreTrainedConfig):
    """Configuration for IMPACT - Instruction-Modulated Perception + ACTion chunking.

    ACT with a language tower. The instruction enters the policy twice:

    - as ``n_text`` tokens appended to the transformer encoder sequence, which the decoder
      reads for free because it already cross-attends the whole encoder memory;
    - as FiLM scale/shift on the ResNet stages, so language modulates *perception* and not
      only the fused memory.

    The second path is the one that matters. With ~600 visual tokens beside ~32 text tokens
    in one self-attention stack, and scenes whose visible objects already disambiguate the
    task, a policy will happily learn to ignore the instruction. FiLM puts language where it
    cannot be routed around.

    Everything not listed below behaves exactly as :class:`ACTConfig`, whose defaults this
    inherits in spirit and whose transformer blocks it reuses unchanged.

    Args:
        chunk_size: Action chunk length. 50 rather than ACT's 100: at 30 Hz that halves
            retask latency from 3.33 s to 1.67 s, which is the point of being able to talk
            to the policy at all, and it costs almost nothing in compute because chunk
            length only touches the decoder's query count.
        n_encoder_layers: 6 rather than ACT's 4. This is where vision and language fuse and
            ACT's depth was sized for a task with no language in it.
        n_decoder_layers: 4 rather than ACT's 1. ACT's single layer is very shallow for
            reading a fused multimodal memory.
        language_encoder: key into :data:`~.text.TEXT_BACKBONES`.
        freeze_language_encoder: keep the text tower frozen. Frozen weights are fully
            determined by the Hub id, so they stay out of the checkpoint.
        tokenizer_max_length: instructions are padded/truncated to this many tokens. The
            padded rows are real encoder outputs, not zeros, and are masked as keys in both
            the encoder self-attention and the decoder cross-attention.
        use_film: apply FiLM to the backbone stages. Off makes the text tokens the only
            language path, which is the ablation the "does it actually read the
            instruction" question needs.
        film_hidden_dim: width of the FiLM head's hidden layer; 0 makes it a single linear.
        language_dropout: per-sample probability of blanking the instruction during
            training. Drops it from both paths at once, so the policy practises the same
            "no language" the ``use_film=False`` ablation sees. 1.0 is permitted: it is the
            always-blind arm.
        vae_sees_language: give the CVAE style encoder the pooled instruction during
            training. Without it the latent absorbs task identity and the language pathway
            becomes redundant - a failure invisible at inference, where the latent is zero.
        int8_groups: bitmask selecting which GEMM groups run through the simulated W8A8
            kernel, in training and at inference alike. The bits are the deployment
            engine's own group mask, unchanged, so a value here and a value there mean the
            same thing: 1 encoder attention, 2 encoder w1, 4 encoder w2, 8 token
            projections, 16 decoder, 32 ResNet convolutions, 63 all of it. 0 - the
            default - is the ordinary fp32 policy and costs nothing. See
            :mod:`lerobot.policies.impact.quantization` for the exact arithmetic.

            Set during training this is quantization-aware training: the forward pass
            rounds exactly as the deployment kernel rounds and a straight-through
            estimator carries the gradient back through it, so the weights learn to
            tolerate it. The quantizer is fully dynamic - per-output-row weight scales,
            per-token (per-image for a convolution) activation scales, all computed from
            the tensor at hand - so a QAT checkpoint stores **no** quantization state. It
            is an ordinary fp32 checkpoint whose weights happen to survive rounding, and
            it loads and exports through every existing path unchanged.
        int8_conv_clip: the fraction of the convolution activation range kept before
            clipping, mirroring the engine's own clip setting. 1.0 is plain absmax. Below
            1.0 trades outlier clipping for resolution on the bulk of the distribution.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture - vision backbone (unchanged from ACT).
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    # Architecture - transformer.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 6
    n_decoder_layers: int = 4

    # Architecture - language.
    language_encoder: str = "t5_small_encoder"
    freeze_language_encoder: bool = True
    tokenizer_max_length: int = 32
    use_film: bool = True
    film_hidden_dim: int = 0
    language_dropout: float = 0.0

    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    vae_sees_language: bool = True

    # Quantization.
    int8_groups: int = 0
    int8_conv_clip: float = 1.0

    # Inference.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.tokenizer_max_length < 1:
            raise ValueError(f"`tokenizer_max_length` must be positive. Got {self.tokenizer_max_length}.")
        if not 0.0 <= self.language_dropout <= 1.0:
            raise ValueError(f"`language_dropout` must be in [0, 1]. Got {self.language_dropout}.")
        if not 0 <= self.int8_groups <= 63:
            raise ValueError(
                f"`int8_groups` is a 6-bit group mask, so it must be in [0, 63]. Got {self.int8_groups}."
            )
        if not 0.0 < self.int8_conv_clip <= 1.0:
            raise ValueError(f"`int8_conv_clip` must be in (0, 1]. Got {self.int8_conv_clip}.")
        if self.use_film and self.replace_final_stride_with_dilation:
            # Not a hard incompatibility, but the FiLM point channel counts are read off the
            # backbone's stages and a dilated final stage changes the feature map, not the
            # channels - so this is allowed and only noted here for the reader.
            pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError(
                "IMPACT is an image policy: at least one `observation.images.*` feature is required."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
