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
"""IMPACT's config: registration, defaults and the settings that must be rejected.

Dependency-light by design - no text tower, no Hub fetch - so the contract a training run
depends on is checked even where ``transformers`` is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.impact.configuration_impact import IMPACTConfig  # noqa: E402
from lerobot.policies.impact.quantization import INT8_ALL  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

FEATURES = {
    "input_features": {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    },
    "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
}


def make_config(**overrides) -> IMPACTConfig:
    return IMPACTConfig(**{**FEATURES, **overrides})


def test_registered_under_its_name():
    """`--policy.type=impact` has to resolve, and the factory finds the policy class by
    naming convention from here."""
    from lerobot.configs.policies import PreTrainedConfig

    assert "impact" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("impact") is IMPACTConfig


def test_factory_resolves_the_policy_and_processor():
    pytest.importorskip("transformers")
    from lerobot.policies.factory import get_policy_class

    assert get_policy_class("impact").name == "impact"


def test_configuration_deltas_from_act():
    """The four deltas the README documents. If a default moves, the README is now wrong."""
    config = make_config()
    assert config.chunk_size == 50
    assert config.n_encoder_layers == 6
    assert config.n_decoder_layers == 4
    assert config.vae_sees_language is True


def test_normalization_is_acts():
    config = make_config()
    assert config.normalization_mapping["STATE"] is NormalizationMode.MEAN_STD
    assert config.normalization_mapping["ACTION"] is NormalizationMode.MEAN_STD


def test_int8_is_off_by_default():
    """fp32 is the honest default: the int8 path has to be asked for."""
    config = make_config()
    assert config.int8_groups == 0
    assert config.int8_conv_clip == 1.0


@pytest.mark.parametrize("groups", [-1, 64, 255])
def test_int8_groups_outside_the_mask_is_rejected(groups: int):
    with pytest.raises(ValueError, match="int8_groups"):
        make_config(int8_groups=groups)


@pytest.mark.parametrize("groups", [0, 1, INT8_ALL])
def test_int8_groups_inside_the_mask_is_accepted(groups: int):
    assert make_config(int8_groups=groups).int8_groups == groups


@pytest.mark.parametrize("clip", [0.0, -0.1, 1.5])
def test_int8_conv_clip_outside_the_unit_interval_is_rejected(clip: float):
    with pytest.raises(ValueError, match="int8_conv_clip"):
        make_config(int8_conv_clip=clip)


@pytest.mark.parametrize("dropout", [-0.1, 1.1])
def test_language_dropout_outside_zero_one_is_rejected(dropout: float):
    with pytest.raises(ValueError, match="language_dropout"):
        make_config(language_dropout=dropout)


def test_always_blind_dropout_is_allowed():
    """`language_dropout=1.0` is the always-blind ablation arm, not a mistake."""
    assert make_config(language_dropout=1.0).language_dropout == 1.0


def test_tokenizer_max_length_must_be_positive():
    with pytest.raises(ValueError, match="tokenizer_max_length"):
        make_config(tokenizer_max_length=0)


def test_image_features_are_required():
    config = IMPACTConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        output_features=FEATURES["output_features"],
    )
    with pytest.raises(ValueError, match="image policy"):
        config.validate_features()


def test_non_resnet_backbone_is_rejected():
    with pytest.raises(ValueError, match="vision_backbone"):
        make_config(vision_backbone="efficientnet_b0")
