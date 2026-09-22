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

from typing import Any

import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_impact import IMPACTConfig
from .text import TEXT_BACKBONES


def make_impact_pre_post_processors(
    config: IMPACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the pre- and post-processing pipelines for IMPACT.

    ACT's pipeline plus one step: the task string is tokenized with T5's tokenizer to a
    fixed length with right padding, which is what puts ``observation.language.tokens`` and
    ``observation.language.attention_mask`` in the batch the policy reads.

    Padding to ``max_length`` rather than to the longest item in the batch is deliberate.
    The padded rows are real encoder outputs that the policy masks explicitly, so a fixed
    length keeps the encoder sequence - and therefore the C++ port's token count and mask
    cache - constant across batches and across episodes.

    Normalization is ACT's: MEAN_STD on state and action, and images through the same path
    the ResNet expects.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        TokenizerProcessorStep(
            tokenizer_name=TEXT_BACKBONES[config.language_encoder],
            padding="max_length",
            padding_side="right",
            max_length=config.tokenizer_max_length,
            truncation=True,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
