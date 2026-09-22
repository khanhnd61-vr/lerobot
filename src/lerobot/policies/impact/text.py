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
"""IMPACT text tower.

A frozen T5-small encoder. Execution-level instructions ("pick up the tape and put it in
the box") need encoding, not generation, so the encoder half is the whole language pathway
- there is no decoder and no LM in the execution path.

T5-small's ``d_model`` is 512, which is also IMPACT's ``dim_model``, so the projection onto
the transformer width is square. That is a coincidence of this pairing and not something to
rely on: the projection is built from the two dimensions, not assumed to be an identity.

The tower is frozen and the instruction is constant within an episode, so its output is an
episode constant. A deployment engine can exploit that (one encode per instruction, not per
query); here it only means the weights stay out of the checkpoint.
"""

import logging

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import require_package

TEXT_BACKBONES = {"t5_small_encoder": "google-t5/t5-small"}


class TextEncoder(nn.Module):
    """Frozen T5 encoder over tokenized instructions.

    Args:
        backbone: key into :data:`TEXT_BACKBONES`.
        freeze: keep the encoder frozen and in eval mode. A frozen encoder is fully
            determined by its Hub id, so its weights are kept out of LeRobot checkpoints and
            reloaded from the Hub on every instantiation.
    """

    def __init__(self, backbone: str = "t5_small_encoder", freeze: bool = True):
        super().__init__()
        if backbone not in TEXT_BACKBONES:
            raise ValueError(
                f"Unknown IMPACT text backbone {backbone!r}; choose from {sorted(TEXT_BACKBONES)}."
            )
        require_package("transformers", extra="impact")
        from transformers import T5EncoderModel

        self.backbone_name = backbone
        self.repo_id = TEXT_BACKBONES[backbone]
        self.frozen = freeze
        self.model = T5EncoderModel.from_pretrained(self.repo_id)
        self.embed_dim = self.model.config.d_model

        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()
            self._register_state_dict_hook(self._drop_frozen_encoder_weights)

        logging.info(
            "IMPACT text tower %s: %.1fM params (%s)",
            backbone,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
            "frozen" if freeze else "trainable",
        )

    @staticmethod
    def _drop_frozen_encoder_weights(module, state_dict, prefix, local_metadata):
        for key in [k for k in state_dict if k.startswith(f"{prefix}model.")]:
            del state_dict[key]
        return state_dict

    @staticmethod
    def is_frozen_encoder_key(key: str) -> bool:
        return key.startswith("text.model.") or ".text.model." in key

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode tokenized instructions.

        Args:
            input_ids: ``(B, L)`` T5 token ids.
            attention_mask: ``(B, L)``, 1 for real tokens.

        Returns:
            ``(B, L, d_text)`` token embeddings. Padded positions carry real encoder
            outputs, not zeros - callers must mask them as keys rather than assume they
            are inert.
        """
        if self.frozen:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval mode whatever the policy does."""
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self


def masked_mean(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool ``(B, L, D)`` over real tokens only.

    The padded rows hold genuine T5 outputs, so an unmasked mean would drag the pooled
    vector toward whatever the padding encodes - and since this vector is what drives FiLM,
    that would make the backbone's modulation depend on instruction *length*.

    Returns ``(B, D)``. An all-pad row (which the tokenizer should never produce) pools to
    zero rather than dividing by zero.
    """
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # (B, L, 1)
    total = (hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return total / count
